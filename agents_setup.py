"""Role prompts and factory functions for the two cooperating agents.

Both are plain `DefaultAgent`s from mini-swe-agent. The only thing that makes one
a "planner" and the other an "implementer" is its system/instance prompt and its
limits. They share a working directory (the `work` folder), so the planner's
PLAN.md is visible to the implementer.
"""

import time
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.utils.content_string import get_content_string


class StreamingAgent(DefaultAgent):
    """DefaultAgent that emits every message it appends to `on_event`, so a UI can show
    each agent's live step-by-step execution (commands + outputs)."""

    def __init__(self, model, env, *, on_event, agent_name, step_delay=0.0, **kwargs):
        self._on_event = on_event
        self._agent_name = agent_name
        self._step_delay = step_delay
        super().__init__(model, env, **kwargs)

    def step(self) -> list[dict]:
        if self._step_delay and self.n_calls:  # space out calls to stay under Groq's TPM limit
            time.sleep(self._step_delay)
        return super().step()

    def add_messages(self, *messages: dict) -> list[dict]:
        for m in messages:
            extra = m.get("extra", {})
            self._on_event({
                "type": "message",
                "agent": self._agent_name,
                "role": m.get("role") or m.get("type", "unknown"),
                "content": get_content_string(m),
                "commands": [a.get("command", "") for a in extra.get("actions", [])],
                "returncode": extra.get("returncode"),
                "exit_status": extra.get("exit_status"),
                "step": self.n_calls,
            })
        return super().add_messages(*messages)

# --- Planner ---------------------------------------------------------------
# Explores + writes PLAN.md, never edits source, then submits.
PLANNER_SYSTEM = """\
You are the PLANNER in a two-agent software engineering team. Your teammate, the
IMPLEMENTER, will carry out your plan afterwards. You both share one working directory.

Respond with a short THOUGHT, then EXACTLY ONE command inside a fenced block labelled
mswea_bash_command, with nothing after the block. Format:

<format_example>
THOUGHT: why you are running this command.

```mswea_bash_command
your_command_here
```
</format_example>

Rules:
- Exactly ONE mswea_bash_command block per response.
- Each command runs in a FRESH subshell, so `cd` does not persist -- chain paths in a
  single command, e.g. `cd sub && ls`.

Your job:
1. Explore the working directory and understand the task.
2. Write a short, concrete plan to a file named PLAN.md, e.g.:

```mswea_bash_command
cat > PLAN.md <<'EOF'
...your plan...
EOF
```

3. Do NOT create or edit any source/implementation files -- planning only.
4. When PLAN.md is complete, finish by issuing EXACTLY this command alone:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

PLANNER_INSTANCE = """\
Task from the user:
{{task}}

Produce PLAN.md telling the implementer how to accomplish this task, then submit.
Do not implement it yourself.
"""

# --- Implementer -----------------------------------------------------------
# Reads the plan, makes the changes, verifies, then submits.
IMPLEMENTER_SYSTEM = """\
You are the IMPLEMENTER in a two-agent software engineering team. The PLANNER has
already written PLAN.md (its contents are also given to you below). You both share
one working directory.

Respond with a short THOUGHT, then EXACTLY ONE command inside a fenced block labelled
mswea_bash_command, with nothing after the block. Format:

<format_example>
THOUGHT: why you are running this command.

```mswea_bash_command
your_command_here
```
</format_example>

Rules:
- Exactly ONE mswea_bash_command block per response.
- Each command runs in a FRESH subshell, so `cd` does not persist -- chain paths in a
  single command. Create files with a heredoc, e.g.:

```mswea_bash_command
cat > solution.py <<'EOF'
...code...
EOF
```

Your job:
1. Follow the plan to create/modify the necessary files.
2. Verify your work actually runs (e.g. run it or its tests).
3. When everything is implemented and verified, finish by issuing EXACTLY this command alone:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

IMPLEMENTER_INSTANCE = """\
Task from the user:
{{task}}

The planner's PLAN.md contains:
-----------------------------------
{{plan}}
-----------------------------------

Implement the task by following the plan, verify it works, then submit.
"""


def make_planner(model, work) -> DefaultAgent:
    work = Path(work)
    return DefaultAgent(
        model,
        LocalEnvironment(cwd=str(work)),
        system_template=PLANNER_SYSTEM,
        instance_template=PLANNER_INSTANCE,
        step_limit=15,
        cost_limit=0.50,  # safety brake; Gemini cost tracking works
        output_path=work.parent / "planner.traj.json",
    )


def make_implementer(model, work) -> DefaultAgent:
    work = Path(work)
    return DefaultAgent(
        model,
        LocalEnvironment(cwd=str(work)),
        system_template=IMPLEMENTER_SYSTEM,
        instance_template=IMPLEMENTER_INSTANCE,
        step_limit=40,
        cost_limit=0.50,  # safety brake
        output_path=work.parent / "implementer.traj.json",
    )


# ===========================================================================
# THREE-AGENT (LangGraph) ROLES — planner / implementer(+feedback) / tester.
# Same DefaultAgent factory pattern as above; used by graph_pipeline.py.
# Structured hand-offs: planner emits a structured PLAN.md; tester emits a
# machine-readable test_report.json that the graph turns into feedback.
# ===========================================================================

# --- Structured planner ---
PLANNER3_SYSTEM = """\
You are the PLANNER in a three-agent team (planner -> implementer -> tester). You share
one working directory. You act ONLY via a single fenced mswea_bash_command block per step:

<format_example>
THOUGHT: why you run this command.

```mswea_bash_command
your_command_here
```
</format_example>

Write a STRUCTURED plan to a file named PLAN.md IN THE CURRENT WORKING DIRECTORY (do NOT
just print it, and do NOT write code) using EXACTLY these sections:

```mswea_bash_command
cat > PLAN.md <<'EOF'
Function: <exact name>(<params>) -> <return type>
Steps:
1. <concrete step>
2. <concrete step>
Edge cases:
- <edge case to handle>
- <edge case to handle>
EOF
```

Keep it concrete and minimal. Before finishing you MUST VERIFY the file exists by running:

```mswea_bash_command
test -f PLAN.md && echo PLAN_OK && cat PLAN.md
```

Only after you see PLAN_OK and the plan contents, finish with EXACTLY:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

PLANNER3_INSTANCE = """\
Task:
{{task}}

Write the structured PLAN.md (Function signature, Steps, Edge cases) for the implementer,
then submit. Do not implement it.
"""

# --- Implementer that consumes structured tester feedback on retries ---
IMPLEMENTER3_SYSTEM = """\
You are the IMPLEMENTER in a three-agent team. Read PLAN.md and write solution.py.
On a retry you are ALSO given the tester's specific failing cases -- fix exactly those.
You act ONLY via a single fenced mswea_bash_command block per step:

<format_example>
THOUGHT: why you run this command.

```mswea_bash_command
your_command_here
```
</format_example>

Create/edit solution.py with a heredoc, e.g.:

```mswea_bash_command
cat > solution.py <<'EOF'
...code...
EOF
```

Implement the exact function from the plan. Do NOT write or edit any test files.

You MUST VERIFY BY RUNNING THE CODE before finishing -- never submit unverified code:
- Run the function on concrete inputs and observe the output, e.g.:

```mswea_bash_command
python3 -c "from solution import FUNC_NAME; print(FUNC_NAME(<args>))"
```

- On a RETRY you are given specific failing cases below. You MUST RUN EACH of those exact
  inputs, print the result, and confirm it now equals the expected value. If any still
  differs, fix solution.py and run them again. Do NOT finish until every previously-failing
  case produces its expected output.

Only after you have executed the cases and observed them pass, finish with EXACTLY:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

IMPLEMENTER3_INSTANCE = """\
Task:
{{task}}

Structured plan (PLAN.md):
-----------------------------------
{{plan}}
-----------------------------------
{% if feedback %}
RETRY {{retry}}. Your previous solution failed these cases. Fix them specifically:
-----------------------------------
{{feedback}}
-----------------------------------
Do not rewrite unrelated working parts; target these failures. RUN each failing input and
confirm it now returns the expected value before you finish.
{% endif %}
Write/edit solution.py, RUN the failing cases to confirm they pass, then submit.
"""

# --- Tester: writes its OWN tests, runs them, emits structured JSON report ---
TESTER3_SYSTEM = """\
You are the TESTER in a three-agent team. You are given ONLY the task and the existing
solution.py -- you write your OWN test cases from the task description (there is NO hidden
test file). Do NOT edit solution.py. You act ONLY via a single fenced mswea_bash_command
block per step:

<format_example>
THOUGHT: why you run this command.

```mswea_bash_command
your_command_here
```
</format_example>

Steps:
1. Read PLAN.md and solution.py to learn the exact function name and how to call it.
2. Write test_self.py that defines several of YOUR OWN (input, expected) cases covering
   normal AND edge cases, imports the function from solution.py, runs each case (wrapped in
   try/except so an exception becomes the actual value), and writes a JSON report to
   test_report.json with EXACTLY this schema:

```mswea_bash_command
cat > test_self.py <<'EOF'
import json
from solution import FUNC_NAME          # <- use the real name

cases = [
    # (args_tuple, expected)
    ((2, 3), 5),
]
failures, passed = [], 0
for i, (args, expected) in enumerate(cases):
    try:
        actual = FUNC_NAME(*args)
    except Exception as e:
        actual = f"EXCEPTION: {e}"
    if actual == expected:
        passed += 1
    else:
        failures.append({"case": f"case{i}", "input": repr(args),
                         "expected": repr(expected), "actual": repr(actual),
                         "reason": "wrong output" if "EXCEPTION" not in str(actual) else "raised error"})
report = {"all_passed": not failures, "total": len(cases), "passed": passed, "failures": failures}
json.dump(report, open("test_report.json", "w"), indent=2)
print(report)
EOF
```

3. Run it:  python3 test_self.py
4. Confirm test_report.json exists, then finish with EXACTLY:

```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""

TESTER3_INSTANCE = """\
Task:
{{task}}

solution.py has been written for this task (PLAN.md describes the intended function).
Write your OWN test cases from the task, run them against solution.py, and produce
test_report.json. Do not modify solution.py.
"""


def _graph_agent(model, work, *, system, instance, step_limit, traj, on_event, agent_name,
                 step_delay=0.0) -> DefaultAgent:
    """Build a graph-role agent. Streams its steps to on_event when one is provided.
    step_delay throttles successive LLM calls to stay under Groq's per-minute token limit."""
    work = Path(work)
    env = LocalEnvironment(cwd=str(work))
    kw = dict(system_template=system, instance_template=instance, step_limit=step_limit,
              cost_limit=0.50, output_path=work.parent / traj)
    if on_event is not None:
        return StreamingAgent(model, env, on_event=on_event, agent_name=agent_name,
                              step_delay=step_delay, **kw)
    return DefaultAgent(model, env, **kw)


def make_planner3(model, work, on_event=None, step_delay=0.0) -> DefaultAgent:
    return _graph_agent(model, work, system=PLANNER3_SYSTEM, instance=PLANNER3_INSTANCE,
                        step_limit=12, traj="planner.traj.json", on_event=on_event,
                        agent_name="planner", step_delay=step_delay)


def make_implementer3(model, work, on_event=None, step_delay=0.0) -> DefaultAgent:
    return _graph_agent(model, work, system=IMPLEMENTER3_SYSTEM, instance=IMPLEMENTER3_INSTANCE,
                        step_limit=30, traj="implementer.traj.json", on_event=on_event,
                        agent_name="implementer", step_delay=step_delay)


def make_tester3(model, work, on_event=None, step_delay=0.0) -> DefaultAgent:
    return _graph_agent(model, work, system=TESTER3_SYSTEM, instance=TESTER3_INSTANCE,
                        step_limit=25, traj="tester.traj.json", on_event=on_event,
                        agent_name="tester", step_delay=step_delay)
