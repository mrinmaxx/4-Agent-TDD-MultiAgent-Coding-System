"""Central configuration for the 4-agent TDD system.

Importing this module has a side effect: it loads ``my-multiagent/.env`` and sets the
mini-swe-agent runtime env vars. It MUST be imported before ``minisweagent`` so the model
picks up the right key/mode — ``roles.py`` guarantees that ordering.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# my-multiagent/  (two levels up from src/multiswe/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load GROQ_API_KEY / MSWEA_MODEL_NAME etc. from the project .env, overriding any globals.
load_dotenv(PROJECT_ROOT / ".env", override=True)

# --- mini-swe-agent runtime knobs (set before minisweagent is imported) ---
os.environ["MSWEA_SILENT_STARTUP"] = "1"
os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "12")  # ride out Groq TPM 429s
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")        # some Groq models unmapped in litellm

# --- Models (per-role) ---
# We drive every model in text mode (the agent parses ```mswea_bash_command blocks from the
# model's prose) — Groq's llama emits malformed native tool calls, and this keeps one code path.
MODEL_KWARGS: dict = {"model_class": "litellm_textbased"}


def model_extra_kwargs(name: str) -> dict:
    """Per-call API args for a given model. Gemini 2.5 spends 'reasoning tokens' that starve
    the visible output in an agent loop, so we turn thinking off; other providers get nothing."""
    return {"reasoning_effort": "disable"} if "gemini" in name.lower() else {}


# Default model used by Planner / Test Architect / Reviewer.
MODEL_NAME: str = os.environ.get("MSWEA_MODEL_NAME", "groq/llama-3.1-8b-instant")
# The Implementer benefits most from a stronger model, so it alone uses Gemini 2.5. Because
# only ONE role calls Gemini, a run makes far fewer Gemini requests — much friendlier to
# Gemini's tiny 20-requests/day free cap than routing every agent through it.
IMPLEMENTER_MODEL: str = os.environ.get("MULTISWE_IMPLEMENTER_MODEL", MODEL_NAME)

MODEL_EXTRA_KWARGS: dict = model_extra_kwargs(MODEL_NAME)  # kept for back-compat

# --- Orchestration limits ---
MAX_RETRIES: int = 3            # Implementer <-> Reviewer fix cycles after the first attempt
MAX_SYNTAX_RETRIES: int = 2     # Test Architect syntax-repair cycles
USE_BRUTE_FORCE_VERIFIER: bool = True  # tests compute expected values from a reference impl

# --- Efficiency knobs ---
TOKEN_BUDGET: int = 8000        # approximate input-token ceiling for retry prompts
_CHARS_PER_TOKEN: int = 4       # rough heuristic used only for trimming
MAX_PROMPT_CHARS: int = TOKEN_BUDGET * _CHARS_PER_TOKEN

# --- Deterministic test execution ---
TEST_TIMEOUT: int = 30          # seconds; kills infinite loops / accidentally exponential code

# --- Per-role agent limits ---
PLANNER_STEPS: int = 8
ARCHITECT_STEPS: int = 12
IMPLEMENTER_STEPS: int = 12
REVIEWER_STEPS: int = 6
COST_LIMIT: float = 1.0         # safety brake per agent

# --- Workspace + artifact names (all four agents share one directory) ---
WORKSPACE: Path = Path(os.environ.get("MULTISWE_WORKSPACE", PROJECT_ROOT / "tdd_workspace"))
# Finished runs are archived here in timestamped subfolders (never overwritten).
RESULTS_DIR: Path = Path(os.environ.get("MULTISWE_RESULTS", PROJECT_ROOT / "four_agent_results"))
# On a PASSING run we also drop a quick solution_<ts>.py + metadata_<ts>.json here.
SUCCESS_DIR: Path = Path(os.environ.get("MULTISWE_SUCCESS_DIR", PROJECT_ROOT / "results"))
SPEC_FILE: str = "spec.md"
TEST_FILE: str = "test_solution.py"
SOLUTION_FILE: str = "solution.py"
FIX_FILE: str = "fix.md"
