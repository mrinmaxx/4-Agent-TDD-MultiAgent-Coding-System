"""The TDD orchestrator — a LangGraph state machine with deterministic gates, a memory bank,
a self-correction quality loop, and a critic, designed to be reliable with ANY model.

Graph:

  START → planner → test_architect → syntax_validator → memory_summarizer → implementer
                                          ^  (harness bug)                      |
                                          |                                     v
                        test_architect_fix|                               code_checker
                                          |                    (syntax bad)/    |  \\(clean)
                                          |                        refinement <-+   critic
                                          |                            |             |  \\(issues)
                                          |                            +--> code_checker  (loop, bounded)
                                          |                                          |(approved/exhausted)
                                          |                                          v
                                          +---- test_runner <------------------------+
                                                  |  success -> END
                                                  |  fail(solution) -> reviewer -> (memory) -> memory_summarizer (loop)
                                                  |  fail(harness)  -> test_architect_fix -> syntax_validator
                                                  |  max_retries    -> END

Correctness signal = the TEST RUNNER (pytest over a brute-force-verified fuzz harness), never an
LLM-invented value. Deterministic nodes (syntax_validator, code_checker, memory_summarizer) use no
LLM — they can't hallucinate and cost nothing, which is what makes weak models usable here.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from multiswe.config import (
    MAX_MEMORY_ENTRIES,
    MAX_PROMPT_CHARS,
    MAX_QUALITY_LOOP,
    MAX_RETRIES,
    MAX_TEST_FIXES,
    MAX_TRACEBACK_LINES,
    MODEL_NAME,
    RESULTS_DIR,
    REVIEWER_OUTPUT_JSON,
    SOLUTION_FILE,
    SPEC_FILE,
    SUCCESS_DIR,
    TEST_CODE_MAX_CHARS,
    TEST_FILE,
    TEST_TIMEOUT,
    USE_CODE_CHECKER,
    USE_CRITIC,
    WORKSPACE,
)
from multiswe.roles import Critic, Implementer, Planner, Reviewer, TestArchitect
from minisweagent.models.utils.content_string import get_content_string

EventSink = Callable[[dict], None]

MAX_SYNTAX_RETRIES = 1

_HARNESS_EXCEPTIONS = (
    "IndexError", "TypeError", "NameError", "AttributeError", "UnboundLocalError",
    "KeyError", "ValueError", "ZeroDivisionError", "ImportError", "SyntaxError",
)
_FRAME_RE = re.compile(r"(\w+\.py):\d+")


# ============================================================= pure helpers (no LLM)
def classify_failure(traceback: str) -> str:
    """"solution_error" | "test_error" — routes a failure to the agent that can fix it."""
    tb = traceback or ""
    if not tb.strip():
        return "solution_error"
    if "AssertionError" in tb:
        return "solution_error"
    frames = _FRAME_RE.findall(tb)
    deepest = frames[-1] if frames else ""
    if deepest == TEST_FILE and any(exc in tb for exc in _HARNESS_EXCEPTIONS):
        return "test_error"
    return "solution_error"


def _exception_name(traceback: str) -> str:
    m = re.search(r"\b([A-Z][a-zA-Z]*Error|TimeoutExpired|Exception)\b", traceback or "")
    return m.group(1) if m else "failure"


def truncate_traceback(traceback: str, lines: int = MAX_TRACEBACK_LINES) -> str:
    """Keep only the LAST ``lines`` lines — where the exception message lives."""
    rows = (traceback or "").splitlines()
    return "\n".join(rows[-lines:]).strip() if len(rows) > lines else (traceback or "").strip()


def _error_line(traceback: str, filename: str) -> Optional[int]:
    hits = re.findall(rf"{re.escape(filename)}:(\d+)", traceback or "")
    return int(hits[-1]) if hits else None


def _toplevel_function_span(code: str, error_line: int) -> Optional[tuple[int, int]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([d.lineno for d in node.decorator_list], default=node.lineno)
            end = node.end_lineno or start
            if start <= error_line <= end:
                return start, end
    return None


def extract_failing_function(code: str, error_line: int) -> str:
    """Source of the top-level function containing ``error_line`` ('' if none)."""
    span = _toplevel_function_span(code, error_line)
    if not span:
        return ""
    start, end = span
    return "\n".join(code.splitlines()[start - 1:end])


def _first_function(text: str) -> str:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text.strip()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = min([d.lineno for d in node.decorator_list], default=node.lineno)
            return "\n".join(text.splitlines()[start - 1:node.end_lineno])
    return text.strip()


def _replace_function(code: str, error_line: int, new_func: str) -> Optional[str]:
    span = _toplevel_function_span(code, error_line)
    if not span:
        return None
    start, end = span
    lines = code.splitlines()
    return "\n".join(lines[:start - 1] + new_func.rstrip("\n").splitlines() + lines[end:])


def _compiles(code: str) -> bool:
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False


def check_code(code: str) -> list[str]:
    """CODE CHECKER (no LLM): catch syntax / indentation errors via ast + compile."""
    if not (code or "").strip():
        return ["solution.py is empty"]
    try:
        ast.parse(code)
        compile(code, "<solution>", "exec")
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]
    return []


def cap_test_code(test_code: str, limit: int = TEST_CODE_MAX_CHARS) -> str:
    """Hard cap on generated test length — SAFELY.

    A raw ``test_code[:limit]`` would slice Python mid-statement, producing a file that fails to
    compile (and triggers wasteful regeneration). Instead we drop WHOLE trailing lines to get
    under ``limit`` and accept the result only if it still compiles AND still defines a test;
    otherwise we keep the full (valid) test — a long correct harness beats a short broken one.
    """
    if len(test_code or "") <= limit:
        return test_code
    kept, total = [], 0
    for line in test_code.splitlines():
        if total + len(line) + 1 > limit:
            break
        kept.append(line)
        total += len(line) + 1
    candidate = "\n".join(kept)
    if candidate.strip() and _compiles(candidate) and "def test" in candidate:
        return candidate + "\n# ... truncated (over length cap)"
    return test_code


def summarize_memory(memory_bank: list) -> str:
    """MEMORY SUMMARIZER (no LLM): compress past attempts into a short, actionable brief."""
    fails = [e for e in (memory_bank or []) if e.get("outcome") == "fail"]
    if not fails:
        return ""
    common = ", ".join(f"{k} x{v}" for k, v in Counter(e.get("error_type", "?") for e in fails).most_common(3))
    recent = [f"- attempt {e.get('n', '?')}: {e.get('error_type', '?')} → tried: {e.get('fix') or '(no fix)'}"
              for e in fails[-3:]]
    tried = "; ".join(e.get("fix", "") for e in fails if e.get("fix"))[-300:]
    out = [f"Past failures: {len(fails)} (common: {common}).", *recent]
    if tried:
        out.append(f"Fixes already tried (don't just repeat them): {tried}")
    out.append("Avoid repeating these mistakes.")
    return "\n".join(out)


def _parse_json_obj(raw: str) -> Optional[dict]:
    """Parse a JSON object from possibly-noisy LLM text (whole string, else first {...})."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, dict) else None
            except Exception:
                return None
    return None


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
    syntax_retries: int
    test_fix_attempts: int
    review_instruction: dict
    # --- reliability components ---
    memory_bank: list        # every failed/successful attempt (code + error + fix)
    attempt_counter: int     # total implementation cycles run
    quality_loops: int       # refinement passes in the current cycle (bounded by MAX_QUALITY_LOOP)
    memory_summary: str      # deterministic brief fed to the Implementer
    code_issues: list        # issues from Code Checker / Critic awaiting refinement
    critic: dict             # last Critic verdict {score, issues, approved}


def _trim(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    return f"{text[:head]}\n...[trimmed {len(text) - max_chars} chars]...\n{text[-(max_chars - head):]}"


class TDDOrchestrator:
    def __init__(self, emit: Optional[EventSink] = None):
        self.emit = emit or (lambda ev: None)
        self.workspace = WORKSPACE
        self.planner = self.architect = self.implementer = self.reviewer = self.critic = None

    # ---------------------------------------------------------------- events
    def _status(self, message: str, **extra) -> None:
        self.emit({"type": "status", "message": message, **extra})

    def _emit_step(self, role: str, message: dict) -> None:
        extra = message.get("extra", {})
        self.emit({
            "type": "step", "role": role, "msg_role": message.get("role", "assistant"),
            "content": get_content_string(message),
            "commands": [a.get("command", "") for a in extra.get("actions", [])],
            "returncode": extra.get("returncode"),
        })

    # ---------------------------------------------------------------- workspace + gates
    def _reset_workspace(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    def _validate_syntax(self, code_path: Path) -> tuple[bool, str]:
        proc = subprocess.run([sys.executable, "-m", "py_compile", str(code_path)],
                              capture_output=True, text=True)
        return proc.returncode == 0, proc.stderr.strip()

    def _run_tests(self, test_code: str, solution_code: str) -> tuple[bool, str]:
        sandbox = Path(tempfile.mkdtemp(prefix="tdd_"))
        try:
            (sandbox / SOLUTION_FILE).write_text(solution_code)
            (sandbox / TEST_FILE).write_text(test_code)
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", TEST_FILE, "-q", "--no-header", "-x", "--tb=short"],
                    cwd=str(sandbox), capture_output=True, text=True, timeout=TEST_TIMEOUT,
                )
            except subprocess.TimeoutExpired as e:
                partial = e.stdout if isinstance(e.stdout, str) else ""
                return False, (f"TIMEOUT: tests did not finish within {TEST_TIMEOUT}s "
                               f"(likely an infinite loop or too-slow algorithm).\n{partial}")
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
        if state.get("node_failed") and state.get("test_code"):
            self._status("🔧 Test Architect fixing test syntax...", error=state.get("last_error"))
            tests = self.architect.write_tests(state["spec"], prev_tests=state["test_code"],
                                               error=_trim(state.get("last_error") or "", 1500))
        else:
            self._status("🧪 Test Architect building fuzz tests...")
            tests = self.architect.write_tests(state["spec"])
        capped = cap_test_code(tests)
        if capped != tests:
            self._status(f"✂️ Test harness exceeded {TEST_CODE_MAX_CHARS} chars — safely truncated.")
        return {"test_code": capped, "node_failed": False, "last_error": None}

    def _syntax_validator_node(self, state: AgentState) -> dict:
        (self.workspace / TEST_FILE).write_text(state["test_code"] or "")
        ok, err = self._validate_syntax(self.workspace / TEST_FILE)
        if ok:
            self._status("✅ Test syntax valid.")
            return {"node_failed": False, "last_error": None}
        retries = state.get("syntax_retries", 0) + 1
        self._status(f"⚠️ test_solution.py does not compile (attempt {retries}).")
        return {"node_failed": True, "last_error": err, "syntax_retries": retries}

    def _memory_summarizer_node(self, state: AgentState) -> dict:
        """DETERMINISTIC: compress the memory bank into a brief for the Implementer."""
        summary = summarize_memory(state.get("memory_bank") or [])
        if summary:
            self._status("🧠 Memory Summarizer: recalling past attempts...")
        return {"memory_summary": summary}

    def _implementer_node(self, state: AgentState) -> dict:
        memory = state.get("memory_summary", "")
        attempt = state.get("attempt_counter", 0)
        if attempt == 0:
            self._status("💻 Implementer coding (first attempt)...")
            solution = self.implementer.run(spec=state["spec"], tests=state["test_code"], memory=memory)
            new_attempts = 0
        else:
            self._status("💻 Implementer coding (retry)...", attempt=state.get("attempts", 0) + 1)
            solution = self._retry_implement(state, memory)
            new_attempts = state.get("attempts", 0) + 1
        (self.workspace / SOLUTION_FILE).write_text(solution or "")
        return {"solution": solution, "attempt_counter": attempt + 1, "attempts": new_attempts,
                "quality_loops": 0, "code_issues": []}

    def _retry_implement(self, state: AgentState, memory: str) -> str:
        """Token-efficient retry: patch ONLY the failing function; fall back to a full rewrite."""
        solution = state["solution"]
        short_tb = truncate_traceback(state["traceback"])
        review = state.get("review_instruction") or {}
        err_line = _error_line(state["traceback"], SOLUTION_FILE)
        failing = extract_failing_function(solution, err_line) if err_line else ""
        if failing:
            try:
                patched = self.implementer.fix_function(failing, short_tb, review, memory=memory)
                new_fn = _first_function(patched)
                candidate = _replace_function(solution, err_line, new_fn) if new_fn else None
                if candidate and _compiles(candidate):
                    return candidate
            except Exception as e:
                self.emit({"type": "error", "message": f"targeted fix failed, rewriting whole file: {e}"})
        return self.implementer.fix(prev_solution=_trim(solution, MAX_PROMPT_CHARS // 2),
                                    fix_instruction=self._review_to_text(review) or short_tb, memory=memory)

    def _code_checker_node(self, state: AgentState) -> dict:
        """DETERMINISTIC: ast/compile the solution before the Critic / Test Runner."""
        if not USE_CODE_CHECKER:
            return {"code_issues": []}
        issues = check_code(state["solution"])
        if issues:
            self._status(f"🔎 Code Checker: {len(issues)} syntax issue(s).")
        else:
            self._status("✅ Code Checker: compiles cleanly.")
        return {"code_issues": issues}

    def _critic_node(self, state: AgentState) -> dict:
        """LLM: senior-reviewer quality pass before the tests. Fail-open (never blocks)."""
        if not USE_CRITIC:
            return {"critic": {"approved": True, "issues": []}, "code_issues": []}
        self._status("🧑‍⚖️ Critic reviewing code quality...")
        raw = self.critic.run(_trim(state["solution"], MAX_PROMPT_CHARS // 3))
        data = self._parse_critic(raw)
        if data.get("approved", True):
            self._status(f"✅ Critic approved (score {data.get('score', '?')}).")
            return {"critic": data, "code_issues": []}
        self._status(f"🔧 Critic flagged {len(data.get('issues', []))} issue(s).")
        return {"critic": data, "code_issues": data.get("issues", [])}

    def _refinement_node(self, state: AgentState) -> dict:
        """LLM: fix ONLY the listed issues (from Code Checker or Critic), then re-check."""
        n = state.get("quality_loops", 0) + 1
        issues = state.get("code_issues") or []
        self._status(f"🛠️ Refinement pass {n}/{MAX_QUALITY_LOOP}...")
        issues_text = "\n".join(f"- {i}" for i in issues) if isinstance(issues, list) else str(issues)
        new_solution = self.implementer.refine(_trim(state["solution"], MAX_PROMPT_CHARS // 2), issues_text)
        new_solution = new_solution or state["solution"]
        (self.workspace / SOLUTION_FILE).write_text(new_solution or "")
        return {"solution": new_solution, "quality_loops": n, "code_issues": []}

    def _test_runner_node(self, state: AgentState) -> dict:
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
            updates["memory_bank"] = self._remember(state, {
                "n": state.get("attempt_counter", 0), "outcome": "pass",
                "code": (state["solution"] or "")[:400],
            })
            self._save_success({**state, "passed": True})
        elif decision == "fix_test":
            self._status(f"🧪 Test harness crashed ({_exception_name(output)}) — routing to Test Architect.")
        elif decision == "review":
            self._status(f"❌ Solution failed ({_exception_name(output)}) — routing to Reviewer.")
        else:
            self._status("🛑 Giving up — returning best-effort solution.", attempt=state["attempts"])
            updates["final_status"] = "max_retries"
        return updates

    def _test_architect_fix_node(self, state: AgentState) -> dict:
        n = state.get("test_fix_attempts", 0) + 1
        self._status(f"🧪 Test Architect regenerating broken harness (fix {n}/{MAX_TEST_FIXES})...")
        updates: dict = {"test_fix_attempts": n, "traceback": ""}
        try:
            regenerated = self.architect.write_tests(
                state["spec"], prev_tests=_trim(state["test_code"], MAX_PROMPT_CHARS // 3),
                error=state["traceback"][:500])  # truncate the error to 500 chars before resending
            updates["test_code"] = cap_test_code(regenerated)
            updates["node_failed"] = False
            updates["last_error"] = None
        except Exception as e:
            updates["test_code"] = state["test_code"]
            updates["node_failed"] = True
            updates["last_error"] = f"test_architect_fix failed: {type(e).__name__}: {e}"
            self.emit({"type": "error", "message": updates["last_error"]})
        return updates

    def _reviewer_node(self, state: AgentState) -> dict:
        """Reviewer: short traceback + failing function only -> compact JSON; store a memory entry."""
        self._status("👁️ Reviewer analyzing failure...", attempt=state["attempts"])
        short_tb = truncate_traceback(state["traceback"])
        err_line = _error_line(state["traceback"], SOLUTION_FILE)
        failing = extract_failing_function(state["solution"], err_line) if err_line else ""
        code_ctx = failing or _trim(state["solution"], MAX_PROMPT_CHARS // 4)
        review = self._parse_review(self.reviewer.run(code=code_ctx, traceback=short_tb), short_tb)
        entry = {
            "n": state.get("attempt_counter", 0), "outcome": "fail",
            "error_type": review.get("error_type") or _exception_name(short_tb),
            "error": short_tb, "fix": review.get("suggestion") or self._review_to_text(review),
            "code": failing[:400],
        }
        return {"review_instruction": review, "last_error": self._review_to_text(review),
                "memory_bank": self._remember(state, entry)}

    # ---------------------------------------------------------------- parsing / memory
    def _remember(self, state: AgentState, entry: dict) -> list:
        return ((state.get("memory_bank") or []) + [entry])[-MAX_MEMORY_ENTRIES:]

    def _parse_review(self, raw: str, short_tb: str) -> dict:
        if REVIEWER_OUTPUT_JSON:
            data = _parse_json_obj(raw)
            if data and (data.get("suggestion") or data.get("fix") or data.get("error_type")):
                return data
            return {"error_type": _exception_name(short_tb),
                    "suggestion": short_tb or "Make the failing test pass."}
        return {"suggestion": (raw or short_tb).strip()}

    def _parse_critic(self, raw: str) -> dict:
        data = _parse_json_obj(raw)
        if isinstance(data, dict):
            data.setdefault("issues", [])
            data.setdefault("approved", (data.get("score") or 0) >= 7 and not data["issues"])
            return data
        return {"approved": True, "issues": [], "score": None}  # fail-open on unparseable critic

    @staticmethod
    def _review_to_text(review) -> str:
        if isinstance(review, dict):
            bits = [str(review[k]) for k in ("error_type", "suggestion", "problem", "fix") if review.get(k)]
            return " — ".join(dict.fromkeys(bits))  # de-dup, keep order
        return str(review or "")

    # ============================================================= ROUTING
    def _route_planner(self, state: AgentState) -> str:
        return "critical" if state["node_failed"] else "test_architect"

    def _route_syntax(self, state: AgentState) -> str:
        if not state["node_failed"]:
            return "memory_summarizer"
        if state.get("syntax_retries", 0) <= MAX_SYNTAX_RETRIES:
            return "test_architect"
        return "memory_summarizer"  # still broken -> proceed best-effort

    def _route_code_checker(self, state: AgentState) -> str:
        if state.get("code_issues"):
            if state.get("quality_loops", 0) < MAX_QUALITY_LOOP:
                return "refine"
            return "test"  # exhausted -> let the test runner surface it
        return "critic" if USE_CRITIC else "test"

    def _route_critic(self, state: AgentState) -> str:
        critic = state.get("critic") or {}
        if critic.get("approved", True) or state.get("quality_loops", 0) >= MAX_QUALITY_LOOP:
            return "test"
        return "refine"

    def _decide(self, state: AgentState) -> str:
        if state["passed"]:
            return "success"
        if state["attempts"] >= state["max_retries"]:
            return "max_retries"
        if (classify_failure(state.get("traceback", "")) == "test_error"
                and state.get("test_fix_attempts", 0) < MAX_TEST_FIXES):
            return "fix_test"
        return "review"

    def should_retry_execution(self, state: AgentState) -> str:
        return self._decide(state)

    # ============================================================= BUILD + RUN
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("planner", self._planner_node)
        g.add_node("test_architect", self._test_architect_node)
        g.add_node("syntax_validator", self._syntax_validator_node)
        g.add_node("memory_summarizer", self._memory_summarizer_node)
        g.add_node("implementer", self._implementer_node)
        g.add_node("code_checker", self._code_checker_node)
        g.add_node("refinement", self._refinement_node)
        g.add_node("critic", self._critic_node)
        g.add_node("test_runner", self._test_runner_node)
        g.add_node("test_architect_fix", self._test_architect_fix_node)
        g.add_node("reviewer", self._reviewer_node)

        g.add_edge(START, "planner")
        g.add_conditional_edges("planner", self._route_planner,
                                {"test_architect": "test_architect", "critical": END})
        g.add_edge("test_architect", "syntax_validator")
        g.add_conditional_edges("syntax_validator", self._route_syntax,
                                {"test_architect": "test_architect", "memory_summarizer": "memory_summarizer"})
        g.add_edge("memory_summarizer", "implementer")
        g.add_edge("implementer", "code_checker")
        # Quality loop: Code Checker -> (syntax bad) Refinement -> Code Checker; (clean) -> Critic.
        g.add_conditional_edges("code_checker", self._route_code_checker,
                                {"refine": "refinement", "critic": "critic", "test": "test_runner"})
        g.add_edge("refinement", "code_checker")
        g.add_conditional_edges("critic", self._route_critic,
                                {"refine": "refinement", "test": "test_runner"})
        # Test runner routes each failure to the agent that can fix it.
        g.add_conditional_edges("test_runner", self.should_retry_execution, {
            "success": END, "max_retries": END,
            "fix_test": "test_architect_fix", "review": "reviewer",
        })
        g.add_edge("test_architect_fix", "syntax_validator")
        g.add_edge("reviewer", "memory_summarizer")  # loop back through memory
        return g.compile()

    def solve_issue(self, problem: str) -> dict:
        problem = (problem or "").strip()
        if not problem:
            self.emit({"type": "error", "message": "Empty problem statement."})
            return self._final(problem="", state=None)

        self._reset_workspace()
        self.planner = Planner(self.workspace, self._emit_step)
        self.architect = TestArchitect(self.workspace, self._emit_step)
        self.implementer = Implementer(self.workspace, self._emit_step)
        self.reviewer = Reviewer(self.workspace, self._emit_step)
        self.critic = Critic(self.workspace, self._emit_step)

        init: AgentState = {
            "problem": problem, "spec": "", "test_code": "", "solution": "", "traceback": "",
            "passed": False, "attempts": 0, "max_retries": MAX_RETRIES, "last_error": None,
            "node_failed": False, "final_status": "max_retries",
            "syntax_retries": 0, "test_fix_attempts": 0, "review_instruction": {},
            "memory_bank": [], "attempt_counter": 0, "quality_loops": 0,
            "memory_summary": "", "code_issues": [], "critic": {},
        }
        final_state = self._build_graph().invoke(init, config={"recursion_limit": 100})
        return self._final(problem=problem, state=final_state)

    def _final(self, *, problem: str, state: Optional[AgentState]) -> dict:
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
            result["saved_dir"] = self._save_run(result)
        self.emit({"type": "final", **result})
        return result

    def _save_success(self, state: AgentState) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        SUCCESS_DIR.mkdir(parents=True, exist_ok=True)
        sol_path = SUCCESS_DIR / f"solution_{ts}.py"
        sol_path.write_text(state.get("solution", "") or "")
        (SUCCESS_DIR / f"metadata_{ts}.json").write_text(json.dumps({
            "timestamp": ts, "problem": state.get("problem", ""), "model": MODEL_NAME,
            "passed": True, "attempts": state.get("attempts", 0),
            "test_fix_attempts": state.get("test_fix_attempts", 0), "solution_file": sol_path.name,
        }, indent=2))
        self._status(f"💾 Solution saved to results/{sol_path.name}")

    def _save_run(self, result: dict) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", result["problem"].lower()).strip("-")[:30].strip("-") or "run"
        run_dir = RESULTS_DIR / f"{ts}-{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / SPEC_FILE).write_text(result["spec"] or "")
        (run_dir / TEST_FILE).write_text(result["tests"] or "")
        (run_dir / SOLUTION_FILE).write_text(result["solution"] or "")
        (run_dir / "test_output.txt").write_text(result["test_output"] or "")
        (run_dir / "RESULTS.txt").write_text(
            f"4-AGENT TDD RUN  —  {ts}\n{'=' * 60}\n"
            f"problem          : {result['problem']}\nmodel            : {result['model']}\n"
            f"final_status     : {result['final_status']}\npassed           : {result['passed']}\n"
            f"solution fixes   : {result['attempts']}\ntest regenerations: {result['test_fix_attempts']}\n"
            f"{'=' * 60}\n--- test output ---\n{result['test_output'] or '(none)'}\n")
        try:
            shown = run_dir.relative_to(RESULTS_DIR.parent)
        except ValueError:
            shown = run_dir
        self.emit({"type": "status", "message": f"💾 Saved to {shown}"})
        return str(run_dir)


def solve(problem: str, emit: Optional[EventSink] = None) -> dict:
    return TDDOrchestrator(emit).solve_issue(problem)


def run_agent_graph(problem: str, emit: Optional[EventSink] = None) -> dict:
    return solve(problem, emit)


if __name__ == "__main__":  # CLI:  python -m multiswe.orchestrator "your problem"
    task = sys.argv[1] if len(sys.argv) > 1 else \
        "Implement two_sum(nums: list[int], target: int) -> list[int] returning indices of two numbers adding to target."

    def _log(ev):
        if ev["type"] == "status":
            print("»", ev["message"], flush=True)
        elif ev["type"] == "final":
            print(f"\n=== final_status={ev['final_status']}  passed={ev['passed']}  attempts={ev['attempts']} ===")
            print("\n--- solution.py ---\n" + (ev["solution"] or "(none)"))

    solve(task, _log)
