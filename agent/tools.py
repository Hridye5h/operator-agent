"""Tool registry.

Tools are the only way the agent touches the world. Each one declares a JSON
schema for its arguments and a *kind* describing its role, and the loop reasons
about kinds rather than names — so adding a capability means registering a tool,
not editing the agent loop. That is what keeps the loop task-agnostic.

Kinds:

* ``discover`` - find what exists (list documents, search records)
* ``read``     - extract fields from one source; may report a contested field
* ``write``    - change the system of record
* ``verify``   - read the outcome back and compare it with intent
* ``ask``      - consult a human
* ``other``    - anything else
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable

KINDS = ("discover", "read", "write", "verify", "ask", "other")

# Used only when a tool is registered without an explicit kind. Real tools in
# run_agent.py declare their kind; this keeps quick stubs and tests concise.
_PREFIX_KINDS = (
    (("list_", "find_", "search_"), "discover"),
    (("read_", "get_", "extract_"), "read"),
    (("submit_", "create_", "update_", "delete_", "register_", "record_",
      "post_", "send_"), "write"),
    (("verify_", "check_", "confirm_"), "verify"),
    (("ask_",), "ask"),
)


def infer_kind(name: str) -> str:
    for prefixes, kind in _PREFIX_KINDS:
        if name.startswith(prefixes):
            return kind
    return "other"


@dataclass
class ToolResult:
    """Outcome of one tool call.

    ``data`` carries the structured result the agent reasons over; ``display``
    is the compact rendering shown to the planner. Keeping them separate means
    the agent can hold a full record in working state while the prompt stays
    small.

    ``error_kind`` refines a failure for the planner:

    * ``transient`` - worth retrying unchanged
    * ``conflict``  - the target already exists; the work may already be done
    * ``invalid``   - the input was rejected; retrying unchanged cannot help
    * ``unknown``   - a write failed partway; it may have taken effect, so the
                      record must be read back before anything is repeated
    """

    ok: bool
    data: Any = None
    error: str | None = None
    retriable: bool = False
    artifacts: list[str] | None = None
    display: str | None = None
    error_kind: str | None = None

    def brief(self, limit: int = 600) -> str:
        """Short rendering for the planner's observation buffer."""
        if not self.ok:
            if self.error_kind in ("conflict", "unknown"):
                tag = "conflict" if self.error_kind == "conflict" else "outcome unknown"
            else:
                tag = "retriable" if self.retriable else "permanent"
            return f"ERROR ({tag}): {self.error}"
        if self.display is not None:
            text = self.display
        elif isinstance(self.data, str):
            text = self.data
        else:
            text = repr(self.data)
        return text if len(text) <= limit else text[:limit] + f"... [{len(text)} chars]"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., ToolResult]
    requires_approval: bool = False
    kind: str = "other"

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
        }

    @property
    def params(self) -> list[str]:
        return list(self.parameters.get("properties", {}))


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        if tool.kind not in KINDS:
            raise ValueError(f"unknown tool kind {tool.kind!r} for {tool.name}")
        self._tools[tool.name] = tool

    def tool(self, name: str, description: str, parameters: dict[str, Any],
             requires_approval: bool = False, kind: str | None = None):
        def deco(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
            self.register(Tool(name, description, parameters, fn, requires_approval,
                               kind or infer_kind(name)))
            return fn
        return deco

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def kind_of(self, name: str) -> str | None:
        tool = self._tools.get(name)
        return tool.kind if tool else None

    def by_kind(self, kind: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.kind == kind]

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def describe(self) -> str:
        """Catalogue rendered for the planner prompt."""
        out = []
        for t in self._tools.values():
            props = t.parameters.get("properties", {})
            required = set(t.parameters.get("required", []))
            if props:
                args = ", ".join(
                    f"{k}{'' if k in required else '?'}: {v.get('type','any')}"
                    for k, v in props.items()
                )
            else:
                args = ""
            flag = "  [NEEDS APPROVAL]" if t.requires_approval else ""
            out.append(f"- {t.name}({args}) [{t.kind}] -> {t.description}{flag}")
        return "\n".join(out)

    def invoke(self, name: str, args: dict[str, Any]) -> tuple[ToolResult, int]:
        """Run a tool. Never raises: failures come back as ToolResult."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, error=f"no such tool: {name!r}"), 0

        # Drop arguments the tool does not accept rather than exploding, since
        # the planner is an LLM and will occasionally invent a field. A tool
        # that declares **kwargs accepts everything, so filtering by parameter
        # name would hand it nothing at all.
        sig = inspect.signature(tool.fn)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            accepted, rejected = dict(args), []
        else:
            accepted = {k: v for k, v in args.items() if k in sig.parameters}
            rejected = sorted(set(args) - set(accepted))

        try:
            sig.bind(**accepted)
        except TypeError as exc:
            return ToolResult(False, error=f"bad arguments for {name}: {exc}",
                              error_kind="invalid"), 0

        started = time.perf_counter()
        try:
            result = tool.fn(**accepted)
        except Exception as exc:  # a tool crash is an observation, not a stack trace
            if tool.kind == "write":
                # It may have failed after the submission went through, and a
                # write repeated blind can enter the same record twice.
                result = ToolResult(
                    False, retriable=False, error_kind="unknown",
                    error=f"{type(exc).__name__}: {exc}. The write may or may not have "
                          f"taken effect; read the record back before trying again.")
            else:
                result = ToolResult(False, error=f"{type(exc).__name__}: {exc}",
                                    retriable=True, error_kind="transient")
        elapsed = int((time.perf_counter() - started) * 1000)

        if rejected and result.ok:
            result.error = f"ignored unknown arguments: {rejected}"
        return result, elapsed
