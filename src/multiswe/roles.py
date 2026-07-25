"""The TDD agent roles, each wrapping mini-swe-agent's ``DefaultAgent``.

Reliability-with-any-model design:
* DOMAIN EXPERTISE — every role has a strict specialist SYSTEM prompt that names its domain and
  FORBIDS out-of-scope behaviour (Planner never codes, Reviewer/Critic emit JSON only, etc.).
* MEMORY — the Implementer receives a compact summary of past attempts so it stops repeating
  the same mistakes.
* TOKEN EFFICIENCY — static context (spec.md, test_solution.py) lives on disk and is referenced
  by name on retries; retries send only the failing function; the Reviewer/Critic speak JSON.

Deterministic gates (Code Checker, Memory Summarizer) live in orchestrator.py — they need no LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

from jinja2 import Template

# config MUST be imported before minisweagent so env vars are in place.
from multiswe.config import (
    ARCHITECT_STEPS,
    COST_LIMIT,
    CRITIC_FILE,
    CRITIC_STEPS,
    IMPLEMENTER_MODEL,
    IMPLEMENTER_STEPS,
    MODEL_NAME,
    PATCH_FILE,
    PLANNER_STEPS,
    REVIEW_FILE,
    REVIEWER_STEPS,
    SOLUTION_FILE,
    SPEC_FILE,
    TEST_FILE,
    model_extra_kwargs,
)

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from minisweagent.models.utils.content_string import get_content_string


class SafeLocalEnvironment(LocalEnvironment):
    """LocalEnvironment that refuses git commands (weak models read 'submit' as 'git push').

    ``execute`` returns the parent's dict shape (``{output, returncode, exception_info}``) — a
    bare string would break observation formatting.
    """

    _BLOCKED = {
        "output": "❌ Git is blocked in this environment. Your solution is already saved "
                  "locally — do NOT run git. Finish by issuing the submit command "
                  "(echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT).",
        "returncode": 0,
        "exception_info": "",
    }

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "") if isinstance(action, dict) else str(action)
        if command.strip().startswith("git"):
            return dict(self._BLOCKED)
        return super().execute(action, cwd, timeout=timeout)


class TolerantTextbasedModel(LitellmTextbasedModel):
    """Executes only the FIRST command block when weak models emit several in one turn."""

    def _parse_actions(self, response) -> list[dict]:
        content = response.choices[0].message.content or ""
        actions = [a.strip() for a in re.findall(self.config.action_regex, content, re.DOTALL)]
        if actions:
            return [{"command": actions[0]}]
        return super()._parse_actions(response)


StepEmitter = Optional[Callable[[str, dict], None]]
_FENCE_RE = re.compile(r"```(?:[\w+-]*)\n(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------- helpers
def _render(template: str, **kw) -> str:
    return Template(template).render(**kw)


def _last_assistant_text(agent: DefaultAgent) -> str:
    parts = [get_content_string(m) for m in agent.messages if m.get("role") == "assistant"]
    return "\n\n".join(p for p in parts if p).strip()


def _extract_block(text: str) -> str:
    blocks = _FENCE_RE.findall(text or "")
    return max(blocks, key=len).strip() if blocks else ""


class _StreamingAgent(DefaultAgent):
    """A DefaultAgent that forwards every message to a UI callback as it happens."""

    def __init__(self, model, env, *, emit_step, role, **kwargs):
        self._emit_step = emit_step
        self._role = role
        super().__init__(model, env, **kwargs)

    def add_messages(self, *messages: dict) -> list[dict]:
        for m in messages:
            try:
                self._emit_step(self._role, m)
            except Exception:
                pass
        return super().add_messages(*messages)


# --------------------------------------------------------------------------- base role
class Role:
    NAME: str = "role"
    SYSTEM: str = ""
    STEP_LIMIT: int = 10
    ARTIFACT: Optional[str] = None
    MODEL: Optional[str] = None

    def __init__(self, workspace: Path, emit_step: StepEmitter = None):
        self.workspace = Path(workspace)
        self.emit_step = emit_step

    def _get_agent(self, system_prompt: str) -> DefaultAgent:
        name = self.MODEL or MODEL_NAME
        model = TolerantTextbasedModel(model_name=name, model_kwargs=model_extra_kwargs(name))
        env = SafeLocalEnvironment(cwd=str(self.workspace))
        kwargs = dict(
            system_template=system_prompt,
            instance_template="{{task}}",
            step_limit=self.STEP_LIMIT,
            cost_limit=COST_LIMIT,
            max_consecutive_format_errors=6,
            output_path=self.workspace / f"{self.NAME}.traj.json",
        )
        if self.emit_step is not None:
            return _StreamingAgent(model, env, emit_step=self.emit_step, role=self.NAME, **kwargs)
        return DefaultAgent(model, env, **kwargs)

    def _run(self, task: str, artifact: Optional[str] = None) -> str:
        agent = self._get_agent(self.SYSTEM)
        agent.run(task)
        target = artifact or self.ARTIFACT
        if target:
            path = self.workspace / target
            if path.exists() and path.read_text().strip():
                return path.read_text()
            recovered = _extract_block(_last_assistant_text(agent))
            if recovered:
                path.write_text(recovered)
                return recovered
            return ""
        return _last_assistant_text(agent)


# =========================================================================== PROMPTS
_FORMAT = """\
Act via EXACTLY ONE fenced command block per step, labelled mswea_bash_command:

THOUGHT: why you run this.
```mswea_bash_command
your_command_here
```
"""

# --- Planner: Requirements Engineer -----------------------------------------
PLANNER_SYSTEM = f"""\
You are the PLANNER — a REQUIREMENTS ENGINEER — in a 4-agent TDD team.

DOMAIN: requirements engineering, edge-case discovery, invariant definition.
FORBIDDEN: writing ANY code, imports, git. English + short pseudocode ONLY. Keep spec.md UNDER
300 words, focused on invariants and edge cases.

{_FORMAT}
Write spec.md with these markdown sections:
- Function: exact signature, e.g. `def two_sum(nums: list[int], target: int) -> list[int]`
- Summary: one line.
- Invariants: properties the output must ALWAYS satisfy.
- Edge cases: empty / single / duplicate / negative / large / boundary.
- Brute-force verifier: a simple, obviously-correct strategy for the tests (English/pseudocode ONLY).

```mswea_bash_command
cat > spec.md <<'EOF'
Function: ...
Summary: ...
Invariants:
- ...
Edge cases:
- ...
Brute-force verifier: ...
EOF
```
After `test -f spec.md && cat spec.md`, finish with EXACTLY:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

PLANNER_TASK = """\
Problem:
{{problem}}

Write spec.md (under 300 words), then submit. Do NOT write implementation code.
"""

# --- Test Architect: Fuzz / Reference Expert --------------------------------
ARCHITECT_SYSTEM = f"""\
You are the TEST ARCHITECT — a PROPERTY-BASED TESTING expert — in a 4-agent TDD team. From
spec.md you write test_solution.py, a pytest file that verifies ANY candidate solution.py.

RULES:
- Write a SIMPLE brute-force reference (< 30 lines). Use O(n^2) or O(n^3) — slow but OBVIOUSLY
  correct. (E.g. Trapping Rain Water: for each position scan left and right for the max, the
  O(n^2) way. Two Sum: try all O(n^2) pairs.)
- NEVER optimise the reference. That is the Implementer's job.
- Generate exactly 10 random fuzz tests (not 20, not 50). `random.seed(0)` first, bounded input
  sizes, and assert `candidate(args) == reference(args)`. NEVER hardcode expected values.
- `from solution import <func>` using the exact name from spec.md.
- Generators must never index out of range. Output ONLY raw Python — no explanations, no markdown.

{_FORMAT}
```mswea_bash_command
cat > test_solution.py <<'EOF'
import random
from solution import your_func

def _reference(*args):
    ...  # simplest correct method, < 30 lines, never optimised

def test_fuzz():
    random.seed(0)
    for _ in range(10):
        # bounded random args
        assert your_func(*args) == _reference(*args)
EOF
```
Then finish with EXACTLY:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

ARCHITECT_TASK = """\
Spec (spec.md):
-----------------------------------
{{spec}}
-----------------------------------
{% if error %}
Your previous test_solution.py is BROKEN — it failed to compile or crashed at runtime (bug in the
TEST HARNESS, not the candidate). Fix the harness and rewrite the file.
Previous test_solution.py:
-----------------------------------
{{prev_tests}}
-----------------------------------
Error:
-----------------------------------
{{error}}
-----------------------------------
{% endif %}
Write test_solution.py (simple brute-force reference < 30 lines + 10-20 fuzz cases), then submit.
"""

# --- Implementer: Algorithm Engineer ----------------------------------------
IMPLEMENTER_SYSTEM = f"""\
You are the IMPLEMENTER — an ALGORITHM ENGINEER — in a 4-agent TDD team. You turn a spec + failing
tests into correct, efficient Python using the STANDARD algorithm for the problem type. CODE ONLY.

DOMAIN: data structures, algorithms, complexity, edge-case-correct code.
FORBIDDEN: writing/editing tests, git, prose, or changing the required function signature.

{_FORMAT}
Write the file NAMED IN YOUR TASK with ONE heredoc, containing exactly the requested code and
nothing else. Static context (spec.md, test_solution.py) is already on disk — `cat` it if needed,
never paste it back. When done, finish with EXACTLY:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

_MEMORY_BLOCK = """\
{% if memory %}Your history on THIS problem (learn from it — DO NOT repeat these mistakes):
-----------------------------------
{{memory}}
-----------------------------------
{% endif %}"""

IMPLEMENTER_TASK = _MEMORY_BLOCK + """\
Spec (spec.md):
-----------------------------------
{{spec}}
-----------------------------------
Tests it must pass (test_solution.py):
-----------------------------------
{{tests}}
-----------------------------------
Write an efficient, correct solution.py now (write to solution.py), then submit.
"""

IMPLEMENTER_FIX_FUNCTION_TASK = _MEMORY_BLOCK + """\
RETRY — TARGETED FUNCTION FIX. Fix ONLY the function below; change nothing else.

Failing function (from solution.py):
-----------------------------------
{{failing_function}}
-----------------------------------
Error (last traceback lines):
-----------------------------------
{{traceback}}
-----------------------------------
Reviewer diagnosis (JSON): {{review}}

Write ONLY the corrected function definition to patch.py — NO imports, NO other functions, NO
comments. It must still satisfy the existing test_solution.py. Then submit:
```mswea_bash_command
cat > patch.py <<'EOF'
def same_function_name(...):
    ...corrected body...
EOF
```
"""

IMPLEMENTER_FIX_FULL_TASK = _MEMORY_BLOCK + """\
RETRY — the targeted patch could not be applied, so rewrite the whole file.
Previous solution.py:
-----------------------------------
{{prev_solution}}
-----------------------------------
What to fix:
-----------------------------------
{{fix_instruction}}
-----------------------------------
Rewrite solution.py to fix exactly this. It must pass the existing test_solution.py and match
spec.md (both on disk). Then submit.
"""

IMPLEMENTER_REFINE_TASK = """\
Your solution.py has these SPECIFIC issues:
-----------------------------------
{{issues}}
-----------------------------------
Current solution.py:
-----------------------------------
{{code}}
-----------------------------------
Fix ONLY these issues; change nothing else. Rewrite solution.py, then submit.
"""

# --- Critic: Senior Code Reviewer (JSON only) -------------------------------
CRITIC_SYSTEM = f"""\
You are the CRITIC — a SENIOR CODE REVIEWER — in a 4-agent TDD team. You review candidate code
BEFORE it reaches the test runner and emit JSON ONLY.

DOMAIN: spotting edge-case gaps, obvious bugs, and gross inefficiency by reading code.
FORBIDDEN: rewriting the code, running it, or ANY text outside the JSON object.

Check for: (1) edge cases (empty input, negatives, off-by-one), (2) performance, (3) obvious bugs.
Be pragmatic — approve code that looks correct even if imperfect.

{_FORMAT}
Write critic.json with EXACTLY these keys:
```mswea_bash_command
cat > critic.json <<'EOF'
{{"score": 0, "issues": ["one short issue", "..."], "approved": true}}
EOF
```
`approved` is true when the code looks correct (score >= 7 and no blocking issue). Then finish:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

CRITIC_TASK = """\
Review this solution.py (do NOT run it) against spec.md (on disk):
-----------------------------------
{{code}}
-----------------------------------
Output critic.json with keys {"score": 0-10, "issues": [..], "approved": bool}. JSON ONLY.
"""

# --- Reviewer: Debugger (JSON only) -----------------------------------------
REVIEWER_SYSTEM = f"""\
You are the REVIEWER — a DEBUGGER — in a 4-agent TDD team. You read a Python traceback and emit the
smallest actionable fix. JSON ONLY.

DOMAIN: interpreting exceptions/assertions and localising the bug.
FORBIDDEN: rewriting the solution, guessing expected values, ANY text outside the JSON object.

{_FORMAT}
Write review.json with EXACTLY these keys:
```mswea_bash_command
cat > review.json <<'EOF'
{{"error_type": "IndexError", "line": 12, "suggestion": "add a bounds check before indexing"}}
EOF
```
`suggestion` MUST be <= 20 words. Then finish with EXACTLY:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

REVIEWER_TASK = """\
Failing function:
-----------------------------------
{{code}}
-----------------------------------
Traceback (last lines only):
-----------------------------------
{{traceback}}
-----------------------------------
Write review.json {"error_type","line","suggestion"} — JSON ONLY, suggestion <= 20 words.
"""


# =========================================================================== ROLES
class Planner(Role):
    NAME, SYSTEM, STEP_LIMIT, ARTIFACT = "planner", PLANNER_SYSTEM, PLANNER_STEPS, SPEC_FILE

    def run(self, problem: str) -> str:
        return self._run(_render(PLANNER_TASK, problem=problem))


class TestArchitect(Role):
    NAME, SYSTEM, STEP_LIMIT, ARTIFACT = "architect", ARCHITECT_SYSTEM, ARCHITECT_STEPS, TEST_FILE

    def run(self, spec: str, prev_tests: str = "", error: str = "") -> str:
        return self._run(_render(ARCHITECT_TASK, spec=spec, prev_tests=prev_tests, error=error))

    def write_tests(self, spec: str, prev_tests: str = "", error: str = "") -> str:
        return self.run(spec, prev_tests=prev_tests, error=error)


class Implementer(Role):
    NAME, SYSTEM, STEP_LIMIT, ARTIFACT = "implementer", IMPLEMENTER_SYSTEM, IMPLEMENTER_STEPS, SOLUTION_FILE
    MODEL = IMPLEMENTER_MODEL

    def run(self, spec: str = "", tests: str = "", memory: str = "") -> str:
        """First attempt: full solution.py from spec + tests (+ memory of past attempts)."""
        return self._run(_render(IMPLEMENTER_TASK, spec=spec, tests=tests, memory=memory))

    def fix_function(self, failing_function: str, traceback: str, review, memory: str = "") -> str:
        """Targeted retry: return ONLY the corrected function (written to patch.py)."""
        review_json = json.dumps(review) if isinstance(review, dict) else str(review or "{}")
        return self._run(
            _render(IMPLEMENTER_FIX_FUNCTION_TASK, failing_function=failing_function,
                    traceback=traceback, review=review_json, memory=memory),
            artifact=PATCH_FILE,
        )

    def fix(self, prev_solution: str, fix_instruction: str, memory: str = "") -> str:
        """Fallback retry: rewrite the whole solution.py."""
        return self._run(
            _render(IMPLEMENTER_FIX_FULL_TASK, prev_solution=prev_solution,
                    fix_instruction=fix_instruction, memory=memory),
            artifact=SOLUTION_FILE,
        )

    def refine(self, code: str, issues: str) -> str:
        """Quality-loop refinement: fix ONLY the listed issues, rewrite solution.py."""
        return self._run(_render(IMPLEMENTER_REFINE_TASK, code=code, issues=issues),
                         artifact=SOLUTION_FILE)


class Critic(Role):
    NAME, SYSTEM, STEP_LIMIT, ARTIFACT = "critic", CRITIC_SYSTEM, CRITIC_STEPS, CRITIC_FILE

    def run(self, code: str) -> str:
        """Return the Critic's raw critic.json text (the orchestrator parses it)."""
        return self._run(_render(CRITIC_TASK, code=code))


class Reviewer(Role):
    NAME, SYSTEM, STEP_LIMIT, ARTIFACT = "reviewer", REVIEWER_SYSTEM, REVIEWER_STEPS, REVIEW_FILE

    def run(self, code: str, traceback: str) -> str:
        return self._run(_render(REVIEWER_TASK, code=code, traceback=traceback))
