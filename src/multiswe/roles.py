"""The four TDD agent roles, each wrapping mini-swe-agent's ``DefaultAgent``.

Design notes
------------
* Every role's SYSTEM prompt is a module-level constant — defined once, reused for every
  call. That is our "global prompt cache": the heavy role definition is never rebuilt, and
  the only per-call text is a small, trimmed ``task`` string. (True provider-side prompt
  caching is not available on the Groq text-based path, so we minimise tokens instead.)
* Each agent runs mini-swe-agent's normal bash loop and writes its artifact to the shared
  workspace with a heredoc. ``_capture`` reads that file back; if the model misbehaved and
  never wrote it, we fall back to the fenced code block in the model's last message so the
  pipeline never proceeds on an empty artifact.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from jinja2 import Template

# config MUST be imported before minisweagent so env vars are in place.
from multiswe.config import (
    ARCHITECT_STEPS,
    COST_LIMIT,
    FIX_FILE,
    IMPLEMENTER_MODEL,
    IMPLEMENTER_STEPS,
    MODEL_NAME,
    model_extra_kwargs,
    PLANNER_STEPS,
    REVIEWER_STEPS,
    SOLUTION_FILE,
    SPEC_FILE,
    TEST_FILE,
)

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
from minisweagent.models.utils.content_string import get_content_string


class SafeLocalEnvironment(LocalEnvironment):
    """LocalEnvironment that refuses git commands.

    Weak models (e.g. llama-3.1-8b) tend to read "submit final output" as "git add/commit/push",
    which would try to publish the workspace (and leak the .env / API keys). We intercept any
    command that starts with ``git`` and return a normal observation telling the agent the work
    is already saved locally, so it stops and just issues the real submit command instead.

    NOTE: ``execute`` must return the SAME dict shape as the parent
    (``{"output", "returncode", "exception_info"}``) — returning a bare string would break the
    agent's observation formatting.
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
    """Weak models (e.g. llama-3.1-8b) frequently emit SEVERAL command blocks in a single
    turn, which the strict parser rejects with a FormatError — aborting the whole role.
    We tolerate that by executing only the FIRST block. Zero blocks still defers to the base
    class (which re-prompts), so genuine truncations are handled normally.
    """

    def _parse_actions(self, response) -> list[dict]:
        content = response.choices[0].message.content or ""
        actions = [a.strip() for a in re.findall(self.config.action_regex, content, re.DOTALL)]
        if actions:
            return [{"command": actions[0]}]
        return super()._parse_actions(response)  # 0 actions -> standard FormatError / re-prompt

# Callback signature used to stream a raw agent message to the UI: (role_name, message).
StepEmitter = Optional[Callable[[str, dict], None]]

_FENCE_RE = re.compile(r"```(?:[\w+-]*)\n(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------- helpers
def _render(template: str, **kw) -> str:
    return Template(template).render(**kw)


def _last_assistant_text(agent: DefaultAgent) -> str:
    parts = [get_content_string(m) for m in agent.messages if m.get("role") == "assistant"]
    return "\n\n".join(p for p in parts if p).strip()


def _extract_block(text: str) -> str:
    """Return the largest fenced code block in ``text`` (fallback when no file was written)."""
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
            except Exception:  # a broken UI stream must never crash the agent
                pass
        return super().add_messages(*messages)


# --------------------------------------------------------------------------- base role
class Role:
    """Base class: builds a fresh agent per call and captures the artifact it produces."""

    NAME: str = "role"
    SYSTEM: str = ""
    STEP_LIMIT: int = 10
    ARTIFACT: Optional[str] = None  # file this role writes, or None
    MODEL: Optional[str] = None     # None -> use the global default MODEL_NAME

    def __init__(self, workspace: Path, emit_step: StepEmitter = None):
        self.workspace = Path(workspace)
        self.emit_step = emit_step

    def _agent(self) -> DefaultAgent:
        name = self.MODEL or MODEL_NAME
        model = TolerantTextbasedModel(model_name=name, model_kwargs=model_extra_kwargs(name))
        env = SafeLocalEnvironment(cwd=str(self.workspace))  # blocks git add/commit/push
        kwargs = dict(
            system_template=self.SYSTEM,
            instance_template="{{task}}",  # per-call content is injected as `task`
            step_limit=self.STEP_LIMIT,
            cost_limit=COST_LIMIT,
            max_consecutive_format_errors=6,  # extra slack for weak models before giving up
            output_path=self.workspace / f"{self.NAME}.traj.json",
        )
        if self.emit_step is not None:
            return _StreamingAgent(model, env, emit_step=self.emit_step, role=self.NAME, **kwargs)
        return DefaultAgent(model, env, **kwargs)

    def _run(self, task: str) -> str:
        """Run the agent on ``task`` and return its artifact (file content, or fallback)."""
        agent = self._agent()
        agent.run(task)
        if self.ARTIFACT:
            path = self.workspace / self.ARTIFACT
            if path.exists() and path.read_text().strip():
                return path.read_text()
            # Fallback: the model forgot to write the file — recover its fenced block.
            recovered = _extract_block(_last_assistant_text(agent))
            if recovered:
                path.write_text(recovered)  # persist so the workspace stays consistent
                return recovered
            return ""
        return _last_assistant_text(agent)


# --------------------------------------------------------------------------- prompts
PLANNER_SYSTEM = """\
You are the PLANNER (role: Oracle) in a 4-agent Test-Driven-Development team. You turn a
natural-language problem into a precise SPEC. You write ZERO code — English and short
pseudocode only.

Act via EXACTLY ONE fenced command block per step, labelled mswea_bash_command:

THOUGHT: why you run this.
```mswea_bash_command
your_command_here
```

Write the spec to spec.md IN THE CURRENT DIRECTORY with these sections:
- Function: the EXACT signature the solution must expose, e.g.
  `def two_sum(nums: list[int], target: int) -> list[int]`
- Summary: one line describing what it computes.
- Invariants: properties the output must ALWAYS satisfy.
- Edge cases: empty input, single element, duplicates, negatives, large input, etc.
- Brute-force verifier: a simple, obviously-correct strategy the tests can use to compute
  the expected answer independently (e.g. "try all O(n^2) pairs", "sort a copy and compare").
  Describe it in English/pseudocode ONLY — never real code.

Write it with a heredoc, then verify it exists:
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
After `test -f spec.md && cat spec.md` shows it, finish with EXACTLY:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

PLANNER_TASK = """\
Problem:
{{problem}}

Write spec.md (Function, Summary, Invariants, Edge cases, Brute-force verifier), then submit.
Do NOT write any implementation code.
"""

ARCHITECT_SYSTEM = """\
You are the TEST ARCHITECT (role: Constrainer) in a 4-agent TDD team. From spec.md you write
test_solution.py — a pytest file that verifies ANY candidate solution.py.

HARD RULES:
- Include a brute-force REFERENCE function (simple, slow, obviously correct) that implements
  the spec's brute-force verifier.
- Write PROPERTY-BASED / FUZZ tests: generate ~50 RANDOM inputs (call random.seed(0) first
  so failures reproduce) and assert the candidate's output EQUALS the reference output.
- Do NOT hardcode expected values for specific inputs. Every expected value must be COMPUTED
  at runtime by the reference function (a few tiny edge cases like empty input are fine, but
  their expected value must also come from the reference function).
- Import the candidate as `from solution import <func>` using the EXACT name from spec.md.
- Keep it deterministic and fast: bound random input sizes so brute force stays quick.

Act via EXACTLY ONE fenced mswea_bash_command block per step. Write the file with a heredoc:
```mswea_bash_command
cat > test_solution.py <<'EOF'
import random
from solution import your_func

def _reference(*args):
    ...  # simple brute force

def test_fuzz():
    random.seed(0)
    for _ in range(50):
        # build random args within small bounds
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
Your previous test_solution.py is BROKEN — it either failed to compile or crashed at runtime
(the bug is in the TEST HARNESS itself, not the candidate solution). Fix the harness and
rewrite the file. Common causes: an out-of-range index in the random-input generator, calling
random.choice('') on an empty string, or a wrong argument count.
Previous test_solution.py:
-----------------------------------
{{prev_tests}}
-----------------------------------
Error / traceback:
-----------------------------------
{{error}}
-----------------------------------
{% endif %}
Write test_solution.py (brute-force reference + ~50-case fuzz test, no hardcoded expected
values, generators that never index out of range), then submit.
"""

IMPLEMENTER_SYSTEM = """\
You are the IMPLEMENTER (role: Solver) in a 4-agent TDD team. You write solution.py: an
efficient, correct implementation of the spec that passes test_solution.py.

RULES:
- Expose the EXACT function signature from the spec.
- Do NOT read, edit, or weaken test_solution.py — make the real code correct instead.
- Handle every edge case the spec lists.

Act via EXACTLY ONE fenced mswea_bash_command block per step. Write with a heredoc:
```mswea_bash_command
cat > solution.py <<'EOF'
def your_func(...):
    ...
EOF
```
You may run it to sanity-check. Then finish with EXACTLY:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

IMPLEMENTER_TASK = """\
{% if fix_instruction %}
RETRY. Your previous solution.py failed the tests. Apply this SPECIFIC fix and change only
what is needed:
-----------------------------------
{{fix_instruction}}
-----------------------------------
Previous solution.py:
-----------------------------------
{{prev_solution}}
-----------------------------------
The tests it must pass (test_solution.py):
-----------------------------------
{{tests}}
-----------------------------------
Rewrite solution.py to fix exactly this, then submit.
{% else %}
Spec (spec.md):
-----------------------------------
{{spec}}
-----------------------------------
The tests your code must pass (test_solution.py):
-----------------------------------
{{tests}}
-----------------------------------
Write an efficient, correct solution.py, then submit.
{% endif %}
"""

REVIEWER_SYSTEM = """\
You are the REVIEWER / FIXER (role: Debugger) in a 4-agent TDD team. You are given the current
solution.py, the tests, and the RAW pytest traceback. Your ONLY job is to translate the
failure into a SPECIFIC, actionable, plain-English fix instruction for the Implementer.

HARD RULES:
- NEVER invent or guess expected output values — the tests already encode the truth.
- Point at the concrete cause: the exception type, the offending line/logic, and what to
  change. E.g. "solution.py line 12 raises TypeError because `x` is None on empty input —
  return [] before indexing", or "off-by-one: the loop should be range(1, n+1)".
- Be concise (a few lines). Diagnose and instruct; do NOT rewrite the whole solution.

Write your instruction to fix.md with a heredoc:
```mswea_bash_command
cat > fix.md <<'EOF'
<specific, plain-English fix instruction>
EOF
```
Then finish with EXACTLY:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

REVIEWER_TASK = """\
Current solution.py:
-----------------------------------
{{solution}}
-----------------------------------
Tests (test_solution.py):
-----------------------------------
{{tests}}
-----------------------------------
RAW pytest traceback:
-----------------------------------
{{traceback}}
-----------------------------------
Write fix.md: a specific, plain-English fix instruction. Do NOT guess expected values.
"""


# --------------------------------------------------------------------------- roles
class Planner(Role):
    NAME, SYSTEM, STEP_LIMIT, ARTIFACT = "planner", PLANNER_SYSTEM, PLANNER_STEPS, SPEC_FILE

    def run(self, problem: str) -> str:
        return self._run(_render(PLANNER_TASK, problem=problem))


class TestArchitect(Role):
    NAME, SYSTEM, STEP_LIMIT, ARTIFACT = "architect", ARCHITECT_SYSTEM, ARCHITECT_STEPS, TEST_FILE

    def run(self, spec: str, prev_tests: str = "", error: str = "") -> str:
        return self._run(_render(ARCHITECT_TASK, spec=spec, prev_tests=prev_tests, error=error))

    def write_tests(self, spec: str, prev_tests: str = "", error: str = "") -> str:
        """Alias for run(); regenerates test_solution.py, optionally repairing a broken harness."""
        return self.run(spec, prev_tests=prev_tests, error=error)


class Implementer(Role):
    NAME, SYSTEM, STEP_LIMIT, ARTIFACT = "implementer", IMPLEMENTER_SYSTEM, IMPLEMENTER_STEPS, SOLUTION_FILE
    MODEL = IMPLEMENTER_MODEL  # stronger model for the actual coding

    def run(self, spec: str = "", tests: str = "", fix_instruction: str = "",
            prev_solution: str = "") -> str:
        return self._run(_render(IMPLEMENTER_TASK, spec=spec, tests=tests,
                                 fix_instruction=fix_instruction, prev_solution=prev_solution))

    def fix(self, prev_solution: str, fix_instruction: str, tests: str) -> str:
        """Targeted retry: rewrite solution.py given the Reviewer's fix + the previous code."""
        return self.run(tests=tests, fix_instruction=fix_instruction, prev_solution=prev_solution)


class Reviewer(Role):
    NAME, SYSTEM, STEP_LIMIT, ARTIFACT = "reviewer", REVIEWER_SYSTEM, REVIEWER_STEPS, FIX_FILE

    def run(self, solution: str, tests: str, traceback: str) -> str:
        return self._run(_render(REVIEWER_TASK, solution=solution, tests=tests, traceback=traceback))
