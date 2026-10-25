#!/usr/bin/env python3
"""Preflight for the live planner: is the key usable, and which model will run?

    python scripts/check_gemini.py

Loads ``.env``, lists the models this key can call, picks one the same way the
agent does, and makes a single tiny planning call. Never prints the key.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.llm import PREFERRED_MODELS, Decision, GeminiPlanner  # noqa: E402
from run_agent import load_env  # noqa: E402


def main() -> int:
    loaded = load_env()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    print(f".env loaded      : {'yes' if loaded else 'no file'}")
    print(f"key present      : {'yes' if key else 'NO'}")
    if not key:
        print("\nAdd a line  GEMINI_API_KEY=...  to .env (get a key at aistudio.google.com/apikey).")
        return 1

    try:
        planner = GeminiPlanner()
    except Exception as exc:
        print(f"client setup     : FAILED — {type(exc).__name__}: {exc}")
        return 1

    try:
        models = sorted(
            m.name.split("/")[-1] for m in planner._client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or [])
        )
        flash = [m for m in models if "flash" in m]
        print(f"models available : {len(models)} ({len(flash)} flash)")
        for name in PREFERRED_MODELS:
            print(f"  {'[x]' if name in models else '[ ]'} {name}")
    except Exception as exc:
        print(f"model listing    : FAILED — {type(exc).__name__}: {str(exc)[:200]}")
        return 1

    print(f"model selected   : {planner.name.split(':', 1)[1]}"
          f"{'  (from GEMINI_MODEL)' if os.environ.get('GEMINI_MODEL') else ''}")

    system = ("Reply with exactly one JSON object choosing the next action: "
              '{"action": "...", "args": {...}, "reasoning": "...", "confidence": 0.0-1.0}')
    context = ("GOAL: confirm you are working.\n\nTOOLS:\n- finish() -> end the run\n\n"
               "HISTORY:\n  (nothing yet)\n\nWhat is the next single action? Reply with JSON only.")
    started = time.perf_counter()
    try:
        decision = planner.decide(system, context)
    except Exception as exc:
        print(f"planning call    : FAILED — {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    elapsed = time.perf_counter() - started

    ok = isinstance(decision, Decision) and decision.action == "finish"
    print(f"planning call    : {'OK' if ok else 'unexpected'} in {elapsed:.1f}s "
          f"-> action={decision.action!r}")
    if planner.waits:
        print(f"rate-limit waits : {len(planner.waits)} ({sum(w for _, w in planner.waits):.0f}s)")
    print("\nReady. Try:  python run_agent.py --scenario happy --planner gemini")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
