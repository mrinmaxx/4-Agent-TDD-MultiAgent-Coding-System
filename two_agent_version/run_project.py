"""Entry point: load config, build the model, run the demo task through the pipeline.

`load_dotenv` runs FIRST (before any minisweagent import) with override=True, so the
model name / key in this project's .env win over the global mini config that
`import minisweagent` would otherwise load.
"""

from dotenv import load_dotenv

load_dotenv(override=True)  # honor this project's .env; must run before minisweagent import

import os

from minisweagent.models import get_model

from orchestrator import run_pipeline

TASK = (
    "Create a file calc.py containing a function add(a, b) that returns a + b. "
    "Then create test_calc.py with a pytest test asserting add(2, 3) == 5. "
    "Run pytest to confirm the test passes."
)


def main() -> None:
    model_name = os.environ["MSWEA_MODEL_NAME"]  # read from .env via python-dotenv
    print(f"Model: {model_name}")
    model = get_model(model_name, {"model_class": "litellm_textbased"})

    result = run_pipeline(TASK, model)

    print("\n=== PIPELINE RESULT ===")
    print("Planner    exit:", result["planner"].get("exit_status"))
    print("Implementer exit:", result["implementer"].get("exit_status"))
    print("\n--- PLAN.md ---")
    print(result["plan"])


if __name__ == "__main__":
    main()
