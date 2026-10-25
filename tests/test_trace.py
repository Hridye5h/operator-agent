"""The run trace: the evidence behind every claim a run makes."""

import io
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.trace import Trace  # noqa: E402


def test_a_character_the_console_cannot_encode_does_not_end_the_run(tmp_path, monkeypatch):
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    trace = Trace("goal", root=tmp_path)

    trace.add("decision", "amount ₹126,500 → needs approval")

    saved = json.loads((trace.dir / "trace.json").read_text(encoding="utf-8"))
    assert saved["steps"][-1]["summary"] == "amount ₹126,500 → needs approval"


def test_a_multi_line_step_stays_on_one_row_of_the_report(tmp_path):
    trace = Trace("goal", root=tmp_path)
    trace.add("observation", "read_invoice: source: a.pdf\nvendor: Northwind | Traders")

    rows = [r for r in trace.write_markdown().read_text(encoding="utf-8").splitlines()
            if r.startswith("| 1 |")]

    assert rows == ["| 1 | `observation` | read_invoice: source: a.pdf<br>vendor: "
                    "Northwind \\| Traders |"]


def test_the_trace_is_swapped_in_whole(tmp_path):
    trace = Trace("goal", root=tmp_path)
    trace.add("plan", "x")
    trace.finish("completed")

    assert sorted(p.name for p in trace.dir.iterdir()) == ["artifacts", "trace.json"]
    assert json.loads((trace.dir / "trace.json").read_text(encoding="utf-8"))["outcome"] == "completed"
