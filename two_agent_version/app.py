"""Web UI for the planner/implementer multi-agent pipeline.

Run:  python app.py   then open http://localhost:5000

It streams each agent's steps to the browser over Server-Sent Events (SSE) so you
can watch them work, see the PLAN.md hand-off, browse the files they create, and
get a clear banner when a rate/step/cost limit is hit.

It reuses the role prompts from agents_setup.py and only *subclasses* DefaultAgent
here (to emit events) — the cloned mini-swe-agent package is not modified.
"""

from dotenv import load_dotenv

load_dotenv(override=True)  # honor this project's .env before importing minisweagent

import os

os.environ["MSWEA_SILENT_STARTUP"] = "1"
# Be patient with Groq's free-tier per-minute token cap: retry/back off so a run rides
# through the ~10-60s TPM window and completes, instead of erroring out.
os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "12")
# Some Groq models (e.g. gpt-oss) aren't in litellm's price map; don't crash on cost calc.
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

import csv
import json
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

import litellm
from flask import Flask, Response, jsonify, render_template, request

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.models.utils.content_string import get_content_string

from agents_setup import (
    IMPLEMENTER_INSTANCE,
    IMPLEMENTER_SYSTEM,
    PLANNER_INSTANCE,
    PLANNER_SYSTEM,
)
from orchestrator import run_pipeline as orchestrate  # reuse the existing pipeline

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmark"))
from run_tests import run_problem_tests  # benchmark/ hidden-test runner

app = Flask(__name__)
WORK = Path("workspace").resolve()      # transient scratch: agent works here, reset each run
SAVED = Path("saved").resolve()         # persistent: the latest produced solution lives here
BENCH_DIR = Path("benchmark").resolve()
RESULTS_DIR = Path("results").resolve()
_event_queue: "queue.Queue | None" = None  # events for the currently-open /stream
_bench_queue: "queue.Queue | None" = None  # events for the currently-open /benchmark_stream


class StreamingAgent(DefaultAgent):
    """DefaultAgent that pushes every message it appends to a callback."""

    def __init__(self, model, env, *, on_event, agent_name, **kwargs):
        self._on_event = on_event
        self._agent_name = agent_name
        super().__init__(model, env, **kwargs)

    def add_messages(self, *messages: dict) -> list[dict]:
        for m in messages:
            extra = m.get("extra", {})
            self._on_event(
                {
                    "type": "message",
                    "agent": self._agent_name,
                    "role": m.get("role") or m.get("type", "unknown"),
                    "content": get_content_string(m),
                    "commands": [a.get("command", "") for a in extra.get("actions", [])],
                    "returncode": extra.get("returncode"),
                    "exit_status": extra.get("exit_status"),
                    "step": self.n_calls,
                }
            )
        return super().add_messages(*messages)


def _list_files(base: Path) -> list[dict]:
    """Source files under `base` -- skip hidden dirs (.git) and caches."""
    files = []
    if not base.exists():
        return files
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        try:
            content = p.read_text()
        except Exception:
            content = "(binary file)"
        files.append({"name": str(rel), "content": content})
    return files


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:30] or "task"


def _list_runs() -> list[str]:
    """Saved run folders, newest first (timestamp prefix sorts chronologically)."""
    if not SAVED.exists():
        return []
    return sorted((p.name for p in SAVED.iterdir() if p.is_dir()), reverse=True)


def _save_run(task: str) -> tuple[str, list[dict]]:
    """Save this run's files into its OWN folder saved/<timestamp>-<slug>/. Older runs kept."""
    folder = SAVED / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(task)}"
    folder.mkdir(parents=True, exist_ok=True)
    for f in _list_files(WORK):
        dest = folder / f["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORK / f["name"], dest)
    return folder.name, _list_files(folder)


def _make(model, name, system, instance, on_event, step_limit):
    return StreamingAgent(
        model,
        LocalEnvironment(cwd=str(WORK)),
        on_event=on_event,
        agent_name=name,
        system_template=system,
        instance_template=instance,
        step_limit=step_limit,
        cost_limit=0.50,
        output_path=WORK.parent / f"{name}.traj.json",
    )


def run_pipeline(task: str, on_event) -> None:
    """Run planner -> implementer, emitting events. Catches limit/rate errors."""
    try:
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)

        model_name = os.environ["MSWEA_MODEL_NAME"]
        on_event({"type": "info", "model": model_name, "task": task})
        model = get_model(model_name, {"model_class": "litellm_textbased"})

        # --- Planner ---
        on_event({"type": "phase", "agent": "planner", "status": "start"})
        planner = _make(model, "planner", PLANNER_SYSTEM, PLANNER_INSTANCE, on_event, 15)
        p_exit = planner.run(task).get("exit_status")
        on_event({"type": "phase", "agent": "planner", "status": "done", "exit": p_exit})

        plan = (WORK / "PLAN.md").read_text() if (WORK / "PLAN.md").exists() else "(planner produced no PLAN.md)"
        # >>> HAND-OFF HOOK: inspect/critique/transform `plan` here later <<<
        on_event({"type": "handoff", "plan": plan})
        on_event({"type": "files", "files": _list_files(WORK)})  # live, in-progress

        # --- Implementer ---
        on_event({"type": "phase", "agent": "implementer", "status": "start"})
        implementer = _make(model, "implementer", IMPLEMENTER_SYSTEM, IMPLEMENTER_INSTANCE, on_event, 40)
        i_exit = implementer.run(task, plan=plan).get("exit_status")
        on_event({"type": "phase", "agent": "implementer", "status": "done", "exit": i_exit})

        # save this run's files into its own folder (older runs are kept)
        run_folder, saved = _save_run(task)
        on_event({"type": "files", "files": saved, "run": run_folder})
        on_event({"type": "done", "planner": p_exit, "implementer": i_exit, "saved": len(saved), "run": run_folder})
    except litellm.exceptions.RateLimitError:
        on_event({"type": "error", "kind": "rate_limit",
                  "message": "LLM rate limit exceeded (429). The free-tier quota was hit — wait a bit and try again."})
    except Exception as e:  # surface anything else instead of a silent crash
        on_event({"type": "error", "kind": "other", "message": f"{type(e).__name__}: {e}"})
    finally:
        on_event({"type": "end"})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    global _event_queue
    task = (request.get_json(silent=True) or {}).get("task", "").strip()
    if not task:
        return jsonify({"error": "empty task"}), 400
    q: queue.Queue = queue.Queue()
    _event_queue = q
    threading.Thread(target=run_pipeline, args=(task, q.put), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/stream")
def stream():
    q = _event_queue

    def gen():
        if q is None:
            return
        while True:
            ev = q.get()
            yield f"data: {json.dumps(ev)}\n\n"
            if ev.get("type") == "end":
                break

    return Response(gen(), mimetype="text/event-stream")


@app.route("/files")
def files_route():
    """List saved runs + the files of the selected run (default: newest). Nothing is deleted."""
    runs = _list_runs()
    run = request.args.get("run") or (runs[0] if runs else "")
    return jsonify({"run": run, "runs": runs, "files": _list_files(SAVED / run) if run else []})


@app.route("/runfile", methods=["POST"])
def runfile():
    """Run a file from a chosen saved run, with optional CLI args and stdin, from the UI."""
    data = request.get_json(silent=True) or {}
    run = data.get("run", "") or ""
    name, args, stdin = data.get("name", ""), data.get("args", "") or "", data.get("stdin", "") or ""
    base = (SAVED / run).resolve()
    target = (base / name).resolve()
    if not run or not name or not target.is_file() or not str(target).startswith(str(base)):
        return jsonify({"error": "file not found in that run"}), 404
    if not name.endswith(".py"):
        return jsonify({"returncode": None, "cmd": "", "output": f"No runner for '{name}'. Only .py files can be run."})
    try:
        extra = shlex.split(args)
    except ValueError as e:
        return jsonify({"returncode": -1, "cmd": "", "output": f"Could not parse arguments: {e}"})
    is_test = target.name.startswith("test_") or name.endswith("_test.py")
    cmd = (["python3", "-m", "pytest", "-q", name] if is_test else ["python3", name]) + extra
    try:
        r = subprocess.run(cmd, cwd=str(base), input=stdin, capture_output=True, text=True, timeout=30)
        return jsonify({"returncode": r.returncode, "cmd": " ".join(cmd),
                        "output": (r.stdout + r.stderr) or "(no output)"})
    except subprocess.TimeoutExpired:
        return jsonify({"returncode": -1, "cmd": " ".join(cmd), "output": "Timed out after 30s."})


def _solution_files(work: Path, hidden_test_name: str) -> list[dict]:
    """Source files the implementer produced (exclude the copied hidden test, PLAN.md, caches)."""
    out = []
    for p in sorted(work.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(work))
        if rel in (hidden_test_name, "PLAN.md") or any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(work).parts):
            continue
        try:
            content = p.read_text()
        except Exception:
            content = "(binary file)"
        out.append({"name": rel, "content": content})
    return out


def _load_problems() -> list[dict]:
    return json.loads((BENCH_DIR / "problems.json").read_text())


def _save_csv_row(problem_id: str, grade: dict, cost: float, steps: int, elapsed: float) -> str:
    """Append one row per solve to results/results.csv (create with header if new)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "results.csv"
    fields = ["problem_id", "easy", "medium", "hard", "total_passed", "cost", "steps", "time"]
    t = grade["tiers"]
    row = {"problem_id": problem_id, "easy": t["easy"]["passed"], "medium": t["medium"]["passed"],
           "hard": t["hard"]["passed"], "total_passed": grade["total_passed"],
           "cost": round(cost, 4), "steps": steps, "time": elapsed}
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return str(path)


def solve_and_grade(problem_id: str, on_event) -> None:
    """Give ONE picked problem to the pipeline, then grade the produced solution.py
    against its 6 hidden tests (easy/medium/hard x2). Emits progress then a result."""
    try:
        problems = {p["id"]: p for p in _load_problems()}
        p = problems.get(problem_id)
        if not p:
            on_event({"type": "error", "message": f"Unknown problem id: {problem_id}"})
            return

        model_name = os.environ["MSWEA_MODEL_NAME"]
        model = get_model(model_name, {"model_class": "litellm_textbased"})
        work = (RESULTS_DIR / "solve" / problem_id).resolve()
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)

        on_event({"type": "solve_start", "problem_id": problem_id, "title": p["title"], "model": model_name})

        # --- 1) agent solves the picked problem (reuse orchestrator; agent sees only agent_prompt) ---
        on_event({"type": "status", "message": "Agent is solving (planner → implementer)…"})
        started, err, plan, cost, steps = time.time(), None, "", 0.0, 0
        try:
            res = orchestrate(p["agent_prompt"], model, work_dir=str(work))
            plan, cost, steps = res.get("plan", ""), res.get("cost", 0.0), res.get("steps", 0)
        except litellm.exceptions.RateLimitError:
            err = "LLM rate limit (429) — free-tier quota hit. Try again shortly."
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        elapsed = round(time.time() - started, 1)

        hidden_text = (BENCH_DIR / "tests" / f"test_{problem_id}.py").read_text()
        result = {"type": "result", "problem_id": problem_id, "title": p["title"], "error": err,
                  "cost": round(cost, 4), "steps": steps, "time": elapsed, "plan": plan,
                  "solution": _solution_files(work, f"test_{problem_id}.py"), "hidden_test": hidden_text}

        # --- 2) grade the produced solution against the hidden tiers ---
        if err is None:
            on_event({"type": "status", "message": "Running the 6 hidden tests…"})
            grade = run_problem_tests(problem_id, str(work))
            result.update({"tiers": grade["tiers"], "total_passed": grade["total_passed"],
                           "total": grade["total"], "pytest_output": grade["pytest_output"],
                           "csv": _save_csv_row(problem_id, grade, cost, steps, elapsed)})
        else:
            result.update({"tiers": None, "total_passed": 0, "total": 6, "pytest_output": err})

        on_event(result)
    except Exception as e:
        on_event({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        on_event({"type": "end"})


@app.route("/problems")
def problems_route():
    return jsonify({"problems": [{"id": p["id"], "title": p["title"], "signature": p["signature"],
                                  "agent_prompt": p["agent_prompt"]} for p in _load_problems()]})


@app.route("/solve", methods=["POST"])
def solve_route():
    global _bench_queue
    problem_id = (request.get_json(silent=True) or {}).get("problem_id", "")
    q: queue.Queue = queue.Queue()
    _bench_queue = q
    threading.Thread(target=solve_and_grade, args=(problem_id, q.put), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/solve_stream")
def solve_stream():
    q = _bench_queue

    def gen():
        if q is None:
            return
        while True:
            ev = q.get()
            yield f"data: {json.dumps(ev)}\n\n"
            if ev.get("type") == "end":
                break

    return Response(gen(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("Multi-agent UI running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
