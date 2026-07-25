"""Sequential planner -> implementer pipeline.

The two agents never talk directly; they cooperate through files in a shared
`work_dir`. The planner writes PLAN.md there, the orchestrator reads it back, and
hands its contents to the implementer as the `plan` template variable.
"""

from pathlib import Path

from agents_setup import make_implementer, make_planner


def run_pipeline(task: str, model, work_dir: str = "workspace") -> dict:
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: planning -------------------------------------------------
    planner = make_planner(model, work)
    planner_result = planner.run(task)

    # Read the hand-off artifact the planner left in the shared directory.
    plan_path = work / "PLAN.md"
    plan = plan_path.read_text() if plan_path.exists() else "(No PLAN.md was produced.)"

    # >>> HAND-OFF HOOK: a novel mechanism will inspect/critique/transform the
    # >>> plan here before it reaches the implementer. (placeholder for now)

    # --- Phase 2: implementation ------------------------------------------
    implementer = make_implementer(model, work)
    implementer_result = implementer.run(task, plan=plan)  # plan -> {{plan}}

    return {
        "planner": planner_result,
        "implementer": implementer_result,
        "plan": plan,
        "cost": planner.cost + implementer.cost,          # total $ across both agents
        "steps": planner.n_calls + implementer.n_calls,   # total LLM calls across both
    }
