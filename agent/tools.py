"""Tool registry.

Tools are the only way the agent touches the world. Each one declares a JSON
schema for its arguments, and the planner picks from this registry by name — so
adding a capability means registering a tool, not editing the agent loop. That
is what keeps the loop task-agnostic.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    """Outcome of one tool call.

    ``data`` carries the structured result the agent reasons over; ``display``
    is the compact rendering shown to the planner. Keeping them separate means
    the agent can hold a full record in working state while the prompt stays
    small.
    """

    ok: bool
    data: Any = None
    error: str | None = None
    retriable: bool = False
    artifacts: list[str] | None = None
    display: str | None = None

    def brief(self, limit: int = 600) -> str:
        """Short rendering for the planner's observation buffer."""
        if not self.ok:
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

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def tool(self, name: str, description: str, parameters: dict[str, Any],
             requires_approval: bool = False):
        def deco(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
            self.register(Tool(name, description, parameters, fn, requires_approval))
            return fn
        return deco

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

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
            out.append(f"- {t.name}({args}) -> {t.description}{flag}")
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

        started = time.perf_counter()
        try:
            result = tool.fn(**accepted)
        except TypeError as exc:
            result = ToolResult(False, error=f"bad arguments for {name}: {exc}")
        except Exception as exc:  # a tool crash is an observation, not a stack trace
            result = ToolResult(False, error=f"{type(exc).__name__}: {exc}", retriable=True)
        elapsed = int((time.perf_counter() - started) * 1000)

        if rejected and result.ok:
            result.error = f"ignored unknown arguments: {rejected}"
        return result, elapsed
