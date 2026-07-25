"""Hidden-test runner for the benchmark.

`run_problem_tests(problem_id, solution_dir)` runs the 6 hidden tests for one problem
(test_easy / test_medium / test_hard, 2 assertions each) against the solution.py in
`solution_dir`, and returns per-tier pass counts, the total, and raw pytest output.

The solution is made importable via PYTHONPATH (so `from solution import ...` in the
test resolves to solution_dir/solution.py) — the solution directory is not modified.
Each tier is one function with 2 assertions, so a tier is all-or-nothing: it counts 2/2
only when both its assertions hold, otherwise 0/2.
"""

import os
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
TIERS = ("easy", "medium", "hard")


def run_problem_tests(problem_id: str, solution_dir) -> dict:
    test_file = BENCH_DIR / "tests" / f"test_{problem_id}.py"
    solution_dir = Path(solution_dir).resolve()
    if not test_file.is_file():
        return {"error": f"no hidden test file: {test_file}"}

    env = dict(os.environ)
    env["PYTHONPATH"] = str(solution_dir) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(solution_dir), env=env, capture_output=True, text=True, timeout=120,
    )
    output = proc.stdout + proc.stderr

    tiers, total_passed = {}, 0
    for tier in TIERS:
        ok = f"::test_{tier} PASSED" in output           # full node id printed by -v
        tiers[tier] = {"passed": 2 if ok else 0, "total": 2, "ok": ok}
        total_passed += tiers[tier]["passed"]

    return {
        "problem_id": problem_id,
        "tiers": tiers,                 # {"easy": {"passed": x, "total": 2, "ok": bool}, ...}
        "total_passed": total_passed,   # x/6
        "total": 6,
        "pytest_output": output,
    }


if __name__ == "__main__":  # quick manual check: python run_tests.py <problem_id> <solution_dir>
    import json

    print(json.dumps(run_problem_tests(sys.argv[1], sys.argv[2]), indent=2))
