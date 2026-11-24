"""
Parameters Filter agent (Agent_design.md's second sub-agent). Deterministic core: run
core.param_diagnose over every synthesis_data/var_filtering/<param>.csv (same 90%
dominant-value-share rule as the reference param_diagnose.py). The LLM's role here is
small and optional: a one-paragraph plain-language summary of the recommendation, so the
user reviewing the checkboxes in the frontend has context -- never the KEEP/IGNORE decision
itself, which stays purely statistical and reproducible.

Agent_design.md: "Manual input needed to decide what parameters to include or exclude, on
top of the agent decision" -- so this agent's output is a *recommendation*; the frontend
shows every parameter with its suggested state pre-checked and lets the user override before
anything downstream treats a parameter as excluded.
"""
from __future__ import annotations

import glob
import os

from .. import config
from ..core import param_diagnose
from ..llm_client import plain_call, LLMNotConfigured

SUMMARY_SYSTEM_PROMPT = """You help a materials scientist decide which synthesis parameters \
to keep for a Bayesian-optimization model. You'll be given a list of parameters with a \
statistical KEEP/IGNORE verdict already computed (IGNORE = effectively constant across the \
samples measured so far). Write a short, concrete paragraph (4-6 sentences) summarizing the \
pattern: which parameters are being dropped as constant, anything that stands out (e.g. a \
parameter that is almost but not quite constant), and a plain reminder that the user should \
still check the list before continuing. Do not invent statistics not given to you."""


def run(progress_cb=None, threshold: float | None = None) -> dict:
    progress_cb = progress_cb or (lambda p: None)
    threshold = threshold if threshold is not None else config.PARAM_DIAGNOSE_THRESHOLD

    csv_files = sorted(glob.glob(os.path.join(config.VAR_FILTERING_DIR, "*.csv")))
    progress_cb({"stage": "diagnosing", "total": len(csv_files), "done": 0})

    results = []
    errors = []
    for i, path in enumerate(csv_files):
        try:
            diag = param_diagnose.diagnose_param_file(path, config.VAR_FILTERING_DIR, threshold)
            results.append({
                "parameter": diag.label, "n": diag.n, "n_missing": diag.n_missing,
                "is_numeric": diag.is_numeric, "distinct": diag.distinct,
                "top_value": diag.top_value, "top_count": diag.top_count, "share": diag.share,
                "verdict": diag.verdict, "reason": diag.reason, "value_counts": diag.value_counts,
                "plot_path": diag.plot_path, "report_path": diag.report_path,
                "recommended_keep": diag.verdict == "KEEP",
            })
        except Exception as e:  # noqa: BLE001
            errors.append({"parameter": os.path.splitext(os.path.basename(path))[0], "error": str(e)})
        progress_cb({"stage": "diagnosing", "total": len(csv_files), "done": i + 1})

    results.sort(key=lambda r: (r["verdict"] != "IGNORE", r["parameter"]))

    summary_path = os.path.join(config.VAR_FILTERING_DIR, "filtering_summary.dat")
    keep = [r["parameter"] for r in results if r["verdict"] == "KEEP"]
    ignore = [r["parameter"] for r in results if r["verdict"] == "IGNORE"]
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(f"Parameters Filter summary (threshold {threshold:.0%})\n")
        fh.write(f"KEEP ({len(keep)}): {', '.join(keep)}\n")
        fh.write(f"IGNORE ({len(ignore)}): {', '.join(ignore)}\n")

    narrative = None
    try:
        progress_cb({"stage": "summarizing"})
        listing = "\n".join(f"- {r['parameter']}: {r['verdict']} ({r['reason']})" for r in results)
        narrative = plain_call(SUMMARY_SYSTEM_PROMPT, listing)
    except LLMNotConfigured:
        narrative = None
    except Exception as e:  # noqa: BLE001
        errors.append({"parameter": "_summary", "error": str(e)})

    return {
        "threshold": threshold, "parameters": results, "keep_count": len(keep), "ignore_count": len(ignore),
        "summary_path": summary_path, "narrative": narrative, "errors": errors,
    }
