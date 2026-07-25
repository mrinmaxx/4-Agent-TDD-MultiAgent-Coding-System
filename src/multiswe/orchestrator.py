"""The TDD orchestrator, built as a LangGraph state machine with deterministic runtime gates.

Graph (nodes = agents + no-LLM gates, edges = conditional routing):

        START
          v
      [planner] --(no spec)--> END(critical_error)
          v
   [test_architect] <----------------+  (test fails to COMPILE -> regenerate, bounded)
          v                           |
   [syntax_validator] --invalid------+
          | valid                     ^
          v                           | (re-validate regenerated harness)
    [implementer]                     |
          v                           |
    [test_runner] --pass--> END(success)                     early exit: Reviewer never runs
          |  \--attempts>=max--> END(max_retries)
          |   \--TEST-harness bug (IndexError/NameError/... in test_solution.py)-->
          |        [test_architect_fix] --> [syntax_validator]   (bounded to 2 fixes)
          v  solution wrong (AssertionError / exception in solution.py) OR test-fixes exhausted
      [reviewer] --> [implementer_fix] --> back to [test_runner]

The correctness signal is the TEST RUNNER (pytest over brute-force-verified fuzz tests),
never an LLM-invented expected value. A **traceback classifier** (`classify_failure`) routes
each failure to the agent that can actually fix it: a broken *solution* -> Reviewer/Implementer;
a broken *test harness* -> Test Architect. This stops the Implementer from burning its retry
budget on a correct solution that only "fails" because the test itself crashed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from multiswe.config import (
    MAX_PROMPT_CHARS,
    MAX_RETRIES,
    MODEL_NAME,
    RESULTS_DIR,
    SOLUTION_FILE,
    SPEC_FILE,
    SUCCESS_DIR,
    TEST_FILE,
    TEST_TIMEOUT,
    WORKSPACE,
)
from multiswe.roles import Implementer, Planner, Reviewer, TestArchitect
from minisweagent.models.utils.content_string import get_content_string

EventSink = Callable[[dict], None]

MAX_SYNTAX_RETRIES = 1   # compile failures: "send back to Test Architect once"
MAX_TEST_FIXES = 2       # runtime harness bugs: cap regenerations to avoid infinite loops

# Exceptions that, when raised from WITHIN test_solution.py, indicate the harness itself is
# broken (as opposed to an AssertionError, which means the candidate's output was wrong).
_HARNESS_EXCEPTIONS = (
    "IndexError", "TypeError", "NameError", "AttributeError", "UnboundLocalError",
    "KeyError", "ValueError", "ZeroDivisionError", "ImportError", "SyntaxError",
)
_FRAME_RE = re.compile(r"(\w+\.py):\d+")


def classify_failure(traceback: str) -> str:
    """Classify a pytest failure as "solution_error" or "test_error" (pure, no LLM).

    Rules:
      * An ``AssertionError`` means the test ran fine and the candidate's output disagreed
        with the brute-force reference -> the SOLUTION is wrong  ->  "solution_error".
      * One of the harness exceptions raised in the DEEPEST frame belonging to
        ``test_solution.py`` (i.e. the error originates in the harness, not in ``solution.py``)
        means the TEST harness crashed  ->  "test_error".
      * Everything else (timeouts, exceptions raised inside ``solution.py``, empty/unrecognised
        output) defaults to  "solution_error"  so the Reviewer looks at it.

    Using the DEEPEST frame (rather than a plain "test_solution.py in traceback" substring)
    correctly keeps a crash that bubbles up from ``solution.py`` classified as a solution bug,
    even though the test file is also present in the stack.
    """
    tb = traceback or ""
    if not tb.strip():
        return "solution_error"
    if "AssertionError" in tb:
        return "solution_error"
    frames = _FRAME_RE.findall(tb)              # ordered outermost -> innermost
    deepest = frames[-1] if frames else ""      # where the exception was actually raised
    if deepest == TEST_FILE and any(exc in tb for exc in _HARNESS_EXCEPTIONS):
        return "test_error"
    return "solution_error"


def _exception_name(traceback: str) -> str:
    """Best-effort: pull the exception type out of a pytest traceback for status messages."""
    m = re.search(r"\b([A-Z][a-zA-Z]*Error|TimeoutExpired|Exception)\b", traceback or "")
    return m.group(1) if m else "failure"


class AgentState(TypedDict):
    problem: str
    spec: str
    test_code: str
    solution: str
    traceback: str
    passed: bool
    attempts: int
    max_retries: int
    last_error: Optional[str]
    node_failed: bool
    final_status: Literal["success", "max_retries", "critical_error"]
    syntax_retries: int      # internal: enforces the compile "send back once" rule
    test_fix_attempts: int   # internal: caps harness regenerations (MAX_TEST_FIXES)


def _trim(text: str, max_chars: int) -> str:
    """Keep a prompt fragment within budget by dropping the middle (head + tail survive)."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head
    return f"{text[:head]}\n...[trimmed {len(text) - max_chars} chars]...\n{text[-tail:]}"


class TDDOrchestrator:
    def __init__(self, emit: Optional[EventSink] = None):
        # emit receives dicts: {"type": "status"|"step"|"final"|"error", ...}
        self.emit = emit or (lambda ev: None)
        self.workspace = WORKSPACE
        self.planner = self.architect = self.implementer = self.reviewer = None

    # ---------------------------------------------------------------- events
    def _status(self, message: str, **extra) -> None:
        self.emit({"type": "status", "message": message, **extra})

    def _emit_step(self, role: str, message: dict) -> None:
        """Forward one raw agent message to the UI as a compact per-agent 'step' event."""
        extra = message.get("extra", {})
        self.emit({
            "type": "step",
            "role": role,
            "msg_role": message.get("role", "assistant"),
            "content": get_content_string(message),
            "commands": [a.get("command", "") for a in extra.get("actions", [])],
            "returncode": extra.get("returncode"),
        })

    # ---------------------------------------------------------------- workspace
    def _reset_workspace(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    # ---------------------------------------------------------------- deterministic gates
    def _validate_syntax(self, code_path: Path) -> tuple[bool, str]:
        """SYNTAX GATE (no LLM): does the file compile? Returns (ok, error_text)."""
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(code_path)],
            capture_output=True, text=True,
        )
        return proc.returncode == 0, proc.stderr.strip()

    def _run_tests(self, test_code: str, solution_code: str) -> tuple[bool, str]:
        """EXECUTION GATE (no LLM): run pytest in a throwaway sandbox.

        The sandbox is a fresh tempdir that is always deleted, so test artifacts never
        pollute the workspace. A timeout (infinite loop / accidentally exponential code) is
        reported as a failure, not a crash.
        """
        sandbox = Path(tempfile.mkdtemp(prefix="tdd_"))
        try:
            (sandbox / SOLUTION_FILE).write_text(solution_code)
            (sandbox / TEST_FILE).write_text(test_code)
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", TEST_FILE, "-q", "--no-header",
                     "-x", "--tb=short"],
                    cwd=str(sandbox), capture_output=True, text=True, timeout=TEST_TIMEOUT,
                )
            except subprocess.TimeoutExpired as e:
                partial = e.stdout if isinstance(e.stdout, str) else ""
                return False, (f"TIMEOUT: tests did not finish within {TEST_TIMEOUT}s "
                               f"(likely an infinite loop or a too-slow algorithm).\n{partial}")
            return proc.returncode == 0, (proc.stdout + proc.stderr).strip()
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    # ============================================================= GRAPH NODES
    def _planner_node(self, state: AgentState) -> dict:
        self._status("📝 Planner writing spec...")
        spec = self.planner.run(state["problem"])
        if not spec.strip():
            self.emit({"type": "error", "message": "Planner produced no spec."})
            return {"spec": "", "node_failed": True, "last_error": "planner produced no spec",
                    "final_status": "critical_error"}
        return {"spec": spec, "node_failed": False, "last_error": None}

    def _test_architect_node(self, state: AgentState) -> dict:
        # A non-empty test_code + node_failed here means this is the compile "send back" retry.
        if state.get("node_failed") and state.get("test_code"):
            self._status("🔧 Test Architect fixing test syntax...", error=state.get("last_error"))
            tests = self.architect.write_tests(state["spec"], prev_tests=state["test_code"],
                                               error=_trim(state.get("last_error") or "", 1500))
        else:
            self._status("🧪 Test Architect building fuzz tests...")
            tests = self.architect.write_tests(state["spec"])
        return {"test_code": tests, "node_failed": False, "last_error": None}

    def _syntax_validator_node(self, state: AgentState) -> dict:
        """No-LLM gate: py_compile the tests before the Implementer ever starts (and again
        after a harness regeneration)."""
        (self.workspace / TEST_FILE).write_text(state["test_code"] or "")
        ok, err = self._validate_syntax(self.workspace / TEST_FILE)
        if ok:
            self._status("✅ Syntax validation passed!")
            return {"node_failed": False, "last_error": None}
        retries = state.get("syntax_retries", 0) + 1
        self._status(f"⚠️ test_solution.py does not compile (attempt {retries}).")
        return {"node_failed": True, "last_error": err, "syntax_retries": retries}

    def _implementer_node(self, state: AgentState) -> dict:
        self._status("💻 Implementer coding...")
        solution = self.implementer.run(spec=state["spec"], tests=state["test_code"])
        return {"solution": solution, "attempts": 0}

    def _test_runner_node(self, state: AgentState) -> dict:
        """No-LLM gate: run the tests, record pass/fail + traceback, and pre-set final_status
        for terminal outcomes. The routing decision itself lives in ``should_retry_execution``."""
        self._status("🧪 Running tests...", attempt=state["attempts"])
        if not (state["solution"] or "").strip():
            passed, output = False, "Implementer produced no solution.py."
        else:
            passed, output = self._run_tests(state["test_code"], state["solution"])

        decision = self._decide({**state, "passed": passed, "traceback": output})
        updates: dict = {"passed": passed, "traceback": output}
        if decision == "success":
            self._status("✅ All tests passed!", attempt=state["attempts"])
            updates["final_status"] = "success"
            self._save_success({**state, "passed": True, "solution": state["solution"]})
        elif decision == "fix_test":
            self._status(f"🧪 Test harness crashed ({_exception_name(output)}) — routing to "
                         "Test Architect, NOT the Implementer.")
        elif decision == "review":
            self._status(f"❌ Solution failed ({_exception_name(output)}) — routing to Reviewer.")
        else:  # max_retries
            self._status("🛑 Giving up — returning best-effort solution.", attempt=state["attempts"])
            updates["final_status"] = "max_retries"
        return updates

    def _test_architect_fix_node(self, state: AgentState) -> dict:
        """NEW: regenerate a crashing test harness (routed here by classify_failure).

        Increments ``test_fix_attempts``, clears the stale ``traceback`` (so an old error is
        never re-sent), and on any failure sets ``node_failed`` + ``last_error`` instead of
        crashing the graph.
        """
        n = state.get("test_fix_attempts", 0) + 1
        self._status(f"🧪 Test Architect regenerating broken test harness (fix {n}/{MAX_TEST_FIXES})...")
        updates: dict = {"test_fix_attempts": n, "traceback": ""}   # clear stale traceback
        try:
            updates["test_code"] = self.architect.write_tests(
                state["spec"],
                prev_tests=_trim(state["test_code"], MAX_PROMPT_CHARS // 3),
                error=_trim(state["traceback"], MAX_PROMPT_CHARS // 3),
            )
            updates["node_failed"] = False
            updates["last_error"] = None
        except Exception as e:  # a role crash must not kill the graph
            updates["test_code"] = state["test_code"]          # keep the old harness
            updates["node_failed"] = True
            updates["last_error"] = f"test_architect_fix failed: {type(e).__name__}: {e}"
            self.emit({"type": "error", "message": updates["last_error"]})
        return updates

    def _reviewer_node(self, state: AgentState) -> dict:
        # Reviewer ONLY translates the traceback to English — it never guesses expected values.
        self._status("👁️ Reviewer analyzing failure...", attempt=state["attempts"])
        explanation = self.reviewer.run(
            solution=_trim(state["solution"], MAX_PROMPT_CHARS // 4),
            tests=_trim(state["test_code"], MAX_PROMPT_CHARS // 4),
            traceback=_trim(state["traceback"], MAX_PROMPT_CHARS // 2),
        )
        return {"last_error": explanation}

    def _implementer_fix_node(self, state: AgentState) -> dict:
        # TOKEN TRIMMING: only the fix + previous code + tests — not spec or earlier history.
        self._status("💻 Implementer applying fix...", attempt=state["attempts"] + 1)
        solution = self.implementer.fix(
            prev_solution=_trim(state["solution"], MAX_PROMPT_CHARS // 2),
            fix_instruction=state.get("last_error") or "",
            tests=_trim(state["test_code"], MAX_PROMPT_CHARS // 4),
        )
        return {"solution": solution, "attempts": state["attempts"] + 1}

    # ============================================================= ROUTING
    def _route_planner(self, state: AgentState) -> str:
        return "critical" if state["node_failed"] else "test_architect"

    def _route_syntax(self, state: AgentState) -> str:
        if not state["node_failed"]:
            return "implementer"
        if state.get("syntax_retries", 0) <= MAX_SYNTAX_RETRIES:
            return "test_architect"     # send back once for compile failures
        return "implementer"            # still broken -> proceed best-effort

    def _decide(self, state: AgentState) -> str:
        """The test-runner decision (shared by the node + the router so they never disagree).

        Returns one of: "success" | "max_retries" | "fix_test" | "review".
        """
        if state["passed"]:
            return "success"
        if state["attempts"] >= state["max_retries"]:
            return "max_retries"
        if (classify_failure(state.get("traceback", "")) == "test_error"
                and state.get("test_fix_attempts", 0) < MAX_TEST_FIXES):
            return "fix_test"
        return "review"                 # solution_error, OR test-fixes exhausted

    def should_retry_execution(self, state: AgentState) -> str:
        return self._decide(state)

    # ============================================================= BUILD + RUN
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("planner", self._planner_node)
        g.add_node("test_architect", self._test_architect_node)
        g.add_node("syntax_validator", self._syntax_validator_node)
        g.add_node("implementer", self._implementer_node)
        g.add_node("test_runner", self._test_runner_node)
        g.add_node("test_architect_fix", self._test_architect_fix_node)   # NEW
        g.add_node("reviewer", self._reviewer_node)
        g.add_node("implementer_fix", self._implementer_fix_node)

        g.add_edge(START, "planner")
        g.add_conditional_edges("planner", self._route_planner,
                                {"test_architect": "test_architect", "critical": END})
        g.add_edge("test_architect", "syntax_validator")
        g.add_conditional_edges("syntax_validator", self._route_syntax,
                                {"test_architect": "test_architect", "implementer": "implementer"})
        g.add_edge("implementer", "test_runner")
        # NEW router: failures go to the agent that can actually fix them.
        g.add_conditional_edges("test_runner", self.should_retry_execution, {
            "success": END,
            "max_retries": END,
            "fix_test": "test_architect_fix",    # broken test harness -> regenerate the test
            "review": "reviewer",                # wrong solution (or test-fixes exhausted)
        })
        # Regenerated harness is re-validated (compile) before we re-run the implementer/tests.
        g.add_edge("test_architect_fix", "syntax_validator")
        g.add_edge("reviewer", "implementer_fix")
        g.add_edge("implementer_fix", "test_runner")
        return g.compile()

    def solve_issue(self, problem: str) -> dict:
        """Run the full 4-agent TDD graph for a natural-language problem statement."""
        problem = (problem or "").strip()
        if not problem:
            self.emit({"type": "error", "message": "Empty problem statement."})
            return self._final(problem="", state=None)

        self._reset_workspace()
        self.planner = Planner(self.workspace, self._emit_step)
        self.architect = TestArchitect(self.workspace, self._emit_step)
        self.implementer = Implementer(self.workspace, self._emit_step)
        self.reviewer = Reviewer(self.workspace, self._emit_step)

        init: AgentState = {
            "problem": problem, "spec": "", "test_code": "", "solution": "", "traceback": "",
            "passed": False, "attempts": 0, "max_retries": MAX_RETRIES, "last_error": None,
            "node_failed": False, "final_status": "max_retries",
            "syntax_retries": 0, "test_fix_attempts": 0,
        }
        final_state = self._build_graph().invoke(init, config={"recursion_limit": 60})
        return self._final(problem=problem, state=final_state)

    def _final(self, *, problem: str, state: Optional[AgentState]) -> dict:
        """Emit + return the final result in the shape the UI expects."""
        state = state or {}
        result = {
            "problem": problem,
            "spec": state.get("spec", ""),
            "tests": state.get("test_code", ""),
            "solution": state.get("solution", ""),
            "passed": bool(state.get("passed")),
            "test_output": state.get("traceback", ""),
            "attempts": state.get("attempts", 0),
            "test_fix_attempts": state.get("test_fix_attempts", 0),
            "final_status": state.get("final_status", "critical_error"),
            "model": MODEL_NAME,
        }
        if problem and (result["solution"] or result["spec"]):
            result["saved_dir"] = self._save_run(result)  # archive every finished run
        self.emit({"type": "final", **result})
        return result

    def _save_success(self, state: AgentState) -> None:
        """On a PASSING run, drop solution_<ts>.py + metadata_<ts>.json into ./results/.

        This is the lightweight "here's the working answer" export; the full per-run archive
        (spec/tests/solution/output) still goes to RESULTS_DIR via _save_run when the run ends.
        """
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        SUCCESS_DIR.mkdir(parents=True, exist_ok=True)   # create ./results if missing
        sol_path = SUCCESS_DIR / f"solution_{ts}.py"
        sol_path.write_text(state.get("solution", "") or "")
        (SUCCESS_DIR / f"metadata_{ts}.json").write_text(json.dumps({
            "timestamp": ts,
            "problem": state.get("problem", ""),
            "model": MODEL_NAME,
            "passed": True,
            "attempts": state.get("attempts", 0),
            "test_fix_attempts": state.get("test_fix_attempts", 0),
            "solution_file": sol_path.name,
        }, indent=2))
        self._status(f"💾 Solution saved to results/{sol_path.name}")

    def _save_run(self, result: dict) -> str:
        """Archive a finished run to RESULTS_DIR/<timestamp>-<slug>/ (never overwritten)."""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", result["problem"].lower()).strip("-")[:30].strip("-") or "run"
        run_dir = RESULTS_DIR / f"{ts}-{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / SPEC_FILE).write_text(result["spec"] or "")
        (run_dir / TEST_FILE).write_text(result["tests"] or "")
        (run_dir / SOLUTION_FILE).write_text(result["solution"] or "")
        (run_dir / "test_output.txt").write_text(result["test_output"] or "")
        (run_dir / "RESULTS.txt").write_text(
            f"4-AGENT TDD RUN  —  {ts}\n"
            f"{'=' * 60}\n"
            f"problem          : {result['problem']}\n"
            f"model            : {result['model']}\n"
            f"final_status     : {result['final_status']}\n"
            f"passed           : {result['passed']}\n"
            f"solution fixes   : {result['attempts']}\n"
            f"test regenerations: {result['test_fix_attempts']}\n"
            f"{'=' * 60}\n"
            f"--- test output ---\n{result['test_output'] or '(none)'}\n"
        )
        try:
            shown = run_dir.relative_to(RESULTS_DIR.parent)
        except ValueError:
            shown = run_dir
        self.emit({"type": "status", "message": f"💾 Saved to {shown}"})
        return str(run_dir)


def solve(problem: str, emit: Optional[EventSink] = None) -> dict:
    """Convenience wrapper for a one-shot run."""
    return TDDOrchestrator(emit).solve_issue(problem)


# Back-compat entry point alias (used by the UI / callers expecting run_agent_graph).
def run_agent_graph(problem: str, emit: Optional[EventSink] = None) -> dict:
    return solve(problem, emit)


if __name__ == "__main__":  # CLI:  python -m multiswe.orchestrator "your problem"
    task = sys.argv[1] if len(sys.argv) > 1 else \
        "Implement two_sum(nums: list[int], target: int) -> list[int] returning indices of two numbers adding to target."

    def _log(ev):
        if ev["type"] == "status":
            print("»", ev["message"], flush=True)
        elif ev["type"] == "final":
            print(f"\n=== final_status={ev['final_status']}  passed={ev['passed']}  "
                  f"attempts={ev['attempts']}  test_fixes={ev['test_fix_attempts']} ===")
            print("\n--- solution.py ---\n" + (ev["solution"] or "(none)"))

    solve(task, _log)
