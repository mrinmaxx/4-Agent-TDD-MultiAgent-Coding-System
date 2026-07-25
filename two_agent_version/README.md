# Two-Agent Version (preserved snapshot)

A frozen copy of the working **planner + implementer** system as of the 2-agent
benchmark run — kept so you can keep working on it after the 3rd agent is added.

## Contents
- `app.py` — Flask UI backend (SSE live streaming, saved-runs, run console).
- `agents_setup.py` — planner & implementer prompts + factory functions (text-based mode).
- `orchestrator.py` — `run_pipeline()` = planner → PLAN.md hand-off → implementer.
- `run_project.py` — CLI entry point (no UI).
- `templates/index.html` — the UI (prompt → watch agents → saved runs → run files with input).
- `eval_results/` — the 2-agent benchmark:
  - `test_manual.py` — the 3-problem test suite.
  - `solution.py` / `solution2.py` / `solution3.py` — the agent-produced solutions tested.
  - `results_all.txt` — combined PASS/FAIL output.
  - `per_solution/` — per-solution `*.RESULTS.txt`.

## 2-agent result summary
- `solution.py`  (max_profit)      → 6/6 PASS
- `solution2.py` (min_semesters)   → 6/6 PASS
- `solution3.py` (min_cost_drive)  → 4/6 (refuel logic bug)

## Model / config used
- `MSWEA_MODEL_NAME=groq/llama-3.3-70b-versatile` in **text-based mode**
  (`model_class=litellm_textbased`), patient retries for Groq's free-tier TPM.
- Needs a `.env` with `MSWEA_MODEL_NAME` + `GROQ_API_KEY` (not copied here — reuse the
  project `.env`).

## To run this snapshot standalone later
Copy your `.env` into this folder, then `python app.py` and open http://localhost:5000.
