"""Three-agent pipeline as a LangGraph state machine.

    planner -> implementer -> tester -> (conditional) -> implementer(retry) OR END

Hand-offs are STRUCTURED and flow through the graph State (not raw logs):
  - planner writes a structured PLAN.md            -> State.plan
  - tester writes a machine-readable test_report.json (all_passed + per-failure
    input/expected/actual/reason) -> State.test_results, and the graph renders a
    minimal, specific feedback string -> State.tester_feedback (fed to implementer).

Loop: tester fails & retry_count < 3 -> bump retry_count, back to implementer WITH the
feedback; tester passes -> END; retry_count == 3 & still failing -> END (give up).

Each node runs a mini-swe-agent DefaultAgent from agents_setup.py (factory pattern reused;
agent internals untouched). All agents share workspace/ so PLAN.md, solution.py, and the
tester's own tests coexist.
"""

from dotenv import load_dotenv

load_dotenv(override=True)  # honor project .env before importing minisweagent

import os

os.environ["MSWEA_SILENT_STARTUP"] = "1"
os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "12")  # ride out Groq TPM
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from minisweagent.models import get_model
from minisweagent.models.utils.content_string import get_content_string

from agents_setup import make_implementer3, make_planner3, make_tester3

MODEL_NAME = os.environ["MSWEA_MODEL_NAME"]
WORK = Path("workspace").resolve()
MAX_RETRIES = 3
# Space out LLM calls so the loop stays under Groq's free-tier limit (~12k tokens/minute).
# STEP_DELAY paces calls within a node (~3 calls/min keeps growing-history calls under the
# ceiling); NODE_COOLDOWN nearly clears the rolling 60s window at each hand-off. These are
# deliberately slow -- a full run takes ~15-20 min on free Groq -- and tunable via env vars
# (lower them, or raise them further, to trade speed against 429s).
STEP_DELAY = float(os.environ.get("MULTIAGENT_STEP_DELAY", "0"))
NODE_COOLDOWN = float(os.environ.get("MULTIAGENT_NODE_COOLDOWN", "0"))

# module-level event sink; set per run() call. Nodes call _emit(...) to log progress.
_emit: Callable[[dict], None] = lambda ev: None


class PipelineState(TypedDict):
    task: str
    plan: str
    solution_path: str
    test_results: dict          # parsed test_report.json (has "all_passed", "failures", ...)
    tester_feedback: str        # structured, minimal feedback derived from failures
    retry_count: int
    prev_solution: str          # solution.py content from the previous attempt (no-progress guard)
    prev_feedback: str          # tester_feedback from the previous attempt (no-progress guard)
    no_progress: bool           # set when a retry changed nothing / failed identically
    stop_reason: str            # why the loop stopped early


def _model():
    return get_model(MODEL_NAME, {"model_class": "litellm_textbased"})


def _cooldown(node: str):
    if NODE_COOLDOWN:
        _emit({"type": "log", "message": f"cooldown {NODE_COOLDOWN:g}s before {node} "
               "(letting Groq's per-minute token window recover)"})
        time.sleep(NODE_COOLDOWN)


def _read(name: str) -> str:
    p = WORK / name
    return p.read_text() if p.exists() else ""


def _assistant_text(agent) -> str:
    """Concatenate the agent's own assistant messages -- used as a fallback plan."""
    parts = [get_content_string(m) for m in agent.messages if m.get("role") == "assistant"]
    return "\n\n".join(p for p in parts if p).strip()


def _format_feedback(results: dict) -> str:
    """Turn the tester's structured failures into a minimal, specific feedback block."""
    if results.get("all_passed"):
        return ""
    lines = []
    for f in results.get("failures", []):
        lines.append(
            f"- input: {f.get('input','?')} | expected: {f.get('expected','?')} "
            f"| got: {f.get('actual','?')} | reason: {f.get('reason','?')}"
        )
    return "\n".join(lines) or "- (tester reported failure but listed no specific cases)"


# ---------------------------------------------------------------- nodes
def planner_node(state: PipelineState) -> dict:
    _emit({"type": "node", "node": "planner", "status": "start", "retry": state["retry_count"]})
    agent = make_planner3(_model(), WORK, on_event=_emit, step_delay=STEP_DELAY)
    agent.run(state["task"])
    plan = _read("PLAN.md").strip()
    if not plan:  # fallback: never proceed with an empty plan -- capture the planner's own output
        plan = _assistant_text(agent) or "(planner produced no output)"
        (WORK / "PLAN.md").write_text(plan)  # persist to the shared workspace so it's visible
        _emit({"type": "log", "message": "PLAN.md missing after planner run; captured the "
               "planner's own text output as a fallback plan and wrote it to PLAN.md"})
    _emit({"type": "node", "node": "planner", "status": "done", "plan": plan})
    return {"plan": plan}


def implementer_node(state: PipelineState) -> dict:
    r = state["retry_count"]
    _cooldown("implementer")
    _emit({"type": "node", "node": "implementer", "status": "start", "retry": r})
    make_implementer3(_model(), WORK, on_event=_emit, step_delay=STEP_DELAY).run(
        state["task"], plan=state["plan"], feedback=state.get("tester_feedback", ""), retry=r
    )
    sol = _read("solution.py") or "(no solution.py produced)"
    _emit({"type": "node", "node": "implementer", "status": "done", "retry": r, "solution": sol})
    return {"solution_path": "solution.py"}


def tester_node(state: PipelineState) -> dict:
    r = state["retry_count"]
    _cooldown("tester")
    _emit({"type": "node", "node": "tester", "status": "start", "retry": r})
    make_tester3(_model(), WORK, on_event=_emit, step_delay=STEP_DELAY).run(state["task"])
    results = {"all_passed": False, "failures": [{"reason": "tester produced no parseable test_report.json"}]}
    raw = _read("test_report.json")
    if raw:
        try:
            results = json.loads(raw)
        except Exception as e:
            results = {"all_passed": False, "failures": [{"reason": f"invalid test_report.json: {e}"}]}
    feedback = _format_feedback(results)

    # --- no-progress detection: on a retry, did the implementer actually change anything? ---
    cur_solution = _read("solution.py")
    no_progress, stop_reason = False, ""
    if r > 0 and not results.get("all_passed"):
        if cur_solution == state.get("prev_solution", ""):
            no_progress = True
            stop_reason = "solution.py is unchanged from the previous attempt"
        elif feedback == state.get("prev_feedback", ""):
            no_progress = True
            stop_reason = "the exact same cases failed again with no improvement"
        if no_progress:
            _emit({"type": "no_progress", "retry": r, "reason": stop_reason})

    _emit({"type": "node", "node": "tester", "status": "done", "retry": r,
           "test_results": results, "feedback": feedback, "tests": _read("test_self.py")})
    return {"test_results": results, "tester_feedback": feedback,
            "prev_solution": cur_solution, "prev_feedback": feedback,
            "no_progress": no_progress, "stop_reason": stop_reason}


def bump_retry_node(state: PipelineState) -> dict:
    n = state["retry_count"] + 1
    _emit({"type": "retry", "retry_count": n})
    return {"retry_count": n}


# ---------------------------------------------------------------- routing
def route_after_tester(state: PipelineState) -> str:
    if state["test_results"].get("all_passed"):
        _emit({"type": "route", "decision": "end", "reason": "all self-tests passed"})
        return "end"
    if state.get("no_progress"):
        _emit({"type": "route", "decision": "end",
               "reason": f"stopping early -- no progress: {state.get('stop_reason', '')}"})
        return "end"
    if state["retry_count"] < MAX_RETRIES:
        _emit({"type": "route", "decision": "retry",
               "reason": f"tests failed, retry_count={state['retry_count']} < {MAX_RETRIES}"})
        return "retry"
    _emit({"type": "route", "decision": "end",
           "reason": f"gave up after {MAX_RETRIES} retries (still failing)"})
    return "end"


def build_graph(planner=planner_node, implementer=implementer_node,
                tester=tester_node, bump=bump_retry_node):
    """Compile the graph. Node fns are injectable so the loop can be demoed with stubs."""
    g = StateGraph(PipelineState)
    g.add_node("planner", planner)
    g.add_node("implementer", implementer)
    g.add_node("tester", tester)
    g.add_node("bump_retry", bump)
    g.set_entry_point("planner")
    g.add_edge("planner", "implementer")
    g.add_edge("implementer", "tester")
    g.add_conditional_edges("tester", route_after_tester, {"retry": "bump_retry", "end": END})
    g.add_edge("bump_retry", "implementer")
    return g.compile()


def run(task: str, on_event: Optional[Callable[[dict], None]] = None,
        graph=None, reset: bool = True) -> dict:
    """Run the pipeline on a task; returns the final State. `on_event` receives node logs."""
    global _emit
    _emit = on_event or (lambda ev: None)
    if reset:
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
    app = graph or build_graph()
    init: PipelineState = {"task": task, "plan": "", "solution_path": "",
                           "test_results": {}, "tester_feedback": "", "retry_count": 0,
                           "prev_solution": "", "prev_feedback": "", "no_progress": False,
                           "stop_reason": ""}
    _emit({"type": "start", "task": task, "model": MODEL_NAME})
    final = app.invoke(init, config={"recursion_limit": 60})
    passed = bool(final.get("test_results", {}).get("all_passed"))
    if passed:
        final_status = "passed its own tests"
    elif final.get("no_progress"):
        final_status = f"stopped early -- no progress ({final.get('stop_reason', '')})"
    else:
        final_status = f"gave up after {final.get('retry_count', 0)} retries (self-tests still failing)"
    _emit({"type": "final", "status": final_status, "retry_count": final.get("retry_count", 0),
           "all_passed": passed, "no_progress": bool(final.get("no_progress"))})
    return final


if __name__ == "__main__":  # quick CLI run
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "Write solution.py with a function add(a, b) that returns a + b."
    run(t, on_event=lambda ev: print("EVENT:", ev.get("type"), ev.get("node", ev.get("decision", ev.get("status", "")))))
