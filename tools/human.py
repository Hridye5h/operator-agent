"""Human-in-the-loop.

Escalation is a first-class outcome, not an error path. When the agent cannot
safely proceed it asks, and the answer re-enters the loop as an observation.

Two modes:

* ``interactive`` - prompts on the console.
* ``preset``      - answers come from a supplied mapping, so an escalation can
                    be demonstrated and tested without a person at the keyboard.

A ``preset`` answer of ``None`` means "no human available", which lets the
abandon-safely path be exercised too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.tools import ToolResult


@dataclass
class HumanChannel:
    interactive: bool = True
    preset: dict[str, str] = field(default_factory=dict)
    asked: list[dict[str, str]] = field(default_factory=list)

    def ask(self, question: str, options: list[str] | None = None) -> ToolResult:
        rendered = question.strip()
        if options:
            rendered += "\n" + "\n".join(f"  [{i + 1}] {o}" for i, o in enumerate(options))

        answer = self._answer(question, options, rendered)
        self.asked.append({"question": question, "answer": answer or ""})

        if answer is None:
            return ToolResult(
                False,
                error="NO_HUMAN_AVAILABLE: escalation required but unanswered; "
                      "stopping rather than guessing.",
                retriable=False,
            )

        # Allow a numeric reply against the offered options.
        if options and answer.strip().isdigit():
            idx = int(answer.strip()) - 1
            if 0 <= idx < len(options):
                answer = options[idx]

        return ToolResult(True, data={"question": question, "answer": answer},
                          display=f"human answered: {answer}")

    def _answer(self, question: str, options: list[str] | None, rendered: str) -> str | None:
        # A preset key may list alternatives ("amount|total|figure"), because a
        # language model will not always phrase the question the way a rule
        # does. Keys are tried in order, so put the most specific first.
        lowered = question.lower()
        for key, value in self.preset.items():
            if key == "*":
                continue
            if any(alt.strip().lower() in lowered for alt in key.split("|") if alt.strip()):
                return value
        if "*" in self.preset:
            return self.preset["*"]
        if not self.interactive:
            return None

        print("\n" + "=" * 68)
        print("AGENT NEEDS INPUT")
        print("=" * 68)
        print(rendered)
        try:
            reply = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return reply or None
