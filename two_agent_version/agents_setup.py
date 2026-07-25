"""Role prompts and factory functions for the two cooperating agents.

Both are plain `DefaultAgent`s from mini-swe-agent. The only thing that makes one
a "planner" and the other an "implementer" is its system/instance prompt and its
limits. They share a working directory (the `work` folder), so the planner's
PLAN.md is visible to the implementer.
"""

from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment

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
