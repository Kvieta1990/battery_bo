"""
Runtime configuration: where the working directories live (synthesis_data/, battery_data/,
bo_output/, training_data_clean.xlsx -- all "parallel to the main html file" per the design
notes) and which LLM credentials are available.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _default_workdir() -> str:
    # backend/app/config.py -> backend/app -> backend -> <project root, where index.html lives>
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


WORKDIR = os.path.abspath(os.environ.get("BO_PLATFORM_WORKDIR") or _default_workdir())

SYNTHESIS_DIR = os.path.join(WORKDIR, "synthesis_data")
VAR_FILTERING_DIR = os.path.join(SYNTHESIS_DIR, "var_filtering")
BATTERY_DIR = os.path.join(WORKDIR, "battery_data")
TRAINING_DATA_PATH = os.path.join(WORKDIR, "training_data_clean.xlsx")
CONDITION_PROC_PATH = os.path.join(SYNTHESIS_DIR, "condition_proc.xlsx")
INCOMING_NOTEBOOKS_DIR = os.path.join(SYNTHESIS_DIR, "incoming")
INCOMING_BATTERY_DIR = os.path.join(BATTERY_DIR, "incoming")
BO_OUTPUT_ROOT = os.path.join(WORKDIR, "bo_runs")

for d in (SYNTHESIS_DIR, VAR_FILTERING_DIR, BATTERY_DIR, INCOMING_NOTEBOOKS_DIR,
          INCOMING_BATTERY_DIR, BO_OUTPUT_ROOT):
    os.makedirs(d, exist_ok=True)

LLM_PROVIDER = os.environ.get("BO_PLATFORM_LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("BO_PLATFORM_ANTHROPIC_MODEL", "claude-opus-4-5-20251101")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("BO_PLATFORM_OPENAI_MODEL", "gpt-5")

CORS_ALLOW_ORIGINS = os.environ.get("BO_PLATFORM_CORS_ORIGINS", "*").split(",")

PARAM_DIAGNOSE_THRESHOLD = float(os.environ.get("BO_PLATFORM_PARAM_THRESHOLD", "0.90"))
DUPLICATE_THRESHOLD = float(os.environ.get("BO_PLATFORM_DUPLICATE_THRESHOLD", "0.90"))

# Sanity band for the cycle-1 discharge cutoff voltage; capacities ending far outside this are
# flagged (not rejected) in the Target Data Cross Comparison step for the user to glance at.
DISCHARGE_CUTOFF_V_BAND = (2.0, 3.6)


def llm_configured() -> bool:
    if LLM_PROVIDER == "anthropic":
        return bool(ANTHROPIC_API_KEY)
    if LLM_PROVIDER == "openai":
        return bool(OPENAI_API_KEY)
    return bool(ANTHROPIC_API_KEY or OPENAI_API_KEY)
