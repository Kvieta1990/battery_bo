"""
Parameters Filter agent's deterministic core — ported from the reference `param_diagnose.py`
script, kept numerically identical (same default 0.90 dominant-value-share threshold, same
KEEP/IGNORE rule) but exposed as importable functions instead of a CLI, so the backend can run
it over every var_filtering/<param>.csv and collect the results into one report instead of one
subprocess per parameter.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

MISSING = {"", "na", "nan", "nr", "null"}
BAR = "#6366F1"
BAR_EDGE = "#4338CA"
DEFAULT_THRESHOLD = 0.90


def read_values(path: str):
    vals = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p.strip() for p in s.split(",")]
            vals.append(parts[1] if len(parts) == 2 else s)
    if vals and vals[0].lower() in {"value", "values"}:
        vals = vals[1:]
    if not vals:
        raise ValueError(f"{path}: no values found")
    v = pd.Series(vals, dtype=object)
    v = v[~v.str.lower().isin(MISSING)]
    num = pd.to_numeric(v, errors="coerce")
    is_numeric = bool(len(v) > 0 and num.notna().all())
    return (num if is_numeric else v), len(vals) - len(v), is_numeric


@dataclass
class ParamDiagnosis:
    label: str
    n: int
    n_missing: int
    is_numeric: bool
    distinct: int
    top_value: str
    top_count: int
    share: float
    threshold: float
    verdict: str          # "KEEP" | "IGNORE"
    reason: str
    value_counts: dict = field(default_factory=dict)   # str(value) -> count, top 15
    plot_path: str = ""
    report_path: str = ""


def diagnose(v: pd.Series, threshold: float) -> dict:
    counts = v.value_counts()
    n = int(counts.sum())
    top, top_n = counts.index[0], int(counts.iloc[0])
    share = top_n / n
    verdict = "IGNORE" if share > threshold else "KEEP"
    reason = (f"'{top}' is {top_n}/{n} = {share:.0%} of samples "
              f"{'>' if share > threshold else '<='} threshold {threshold:.0%}")
    reason += " -> treat as constant" if verdict == "IGNORE" else " -> parameter varies"
    return dict(counts=counts, n=n, top=top, share=share, verdict=verdict, reason=reason)


def _plot(v, is_numeric, label, res, out_png):
    plt.rcParams.update({"font.size": 18, "axes.linewidth": 2})
    fig, ax = plt.subplots(figsize=(12, 8.5))
    counts = res["counts"]
    if is_numeric and len(counts) > 10:
        ax.hist(v, bins=min(20, len(counts)), color=BAR, edgecolor=BAR_EDGE, linewidth=1.5)
    else:
        counts = counts.sort_index() if is_numeric else counts
        bars = ax.bar(range(len(counts)), counts.values, width=0.6, color=BAR, edgecolor=BAR_EDGE, linewidth=2)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"{k:g}" if is_numeric else str(k) for k in counts.index], rotation=30, ha="right")
        ax.bar_label(bars, fontsize=16, fontweight="bold", padding=6)
        ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_xlabel(label, fontsize=22, labelpad=10)
    ax.set_ylabel("Number of samples", fontsize=22, labelpad=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="both", which="major", direction="out", length=12, width=2, labelsize=16, top=False, right=False)
    ax.set_title(label, fontweight="bold", fontsize=26, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _report_text(label, path, res, n_missing, threshold) -> str:
    w = 64
    lines = ["=" * w, " PARAMETER DIAGNOSTICS".center(w), "=" * w]
    rows = [("Parameter", label), ("Input file", path), ("Samples (n)", res["n"]),
            ("Missing values", n_missing), ("Distinct values", len(res["counts"])),
            ("Dominant value", f"{res['top']}  ({int(res['counts'].iloc[0])}/{res['n']} = {res['share']:.1%})"),
            ("Threshold", f"{threshold:.0%}")]
    for k, val in rows:
        lines.append(f" {k:<16}: {val}")
    lines.append("-" * w)
    lines.append(" Value counts:")
    shown = res["counts"].head(15)
    width = max(len(str(k)) for k in shown.index)
    for k, c in shown.items():
        lines.append(f"   {str(k):<{width}}  {int(c):>4}  {c / res['n']:6.1%}  " + "#" * round(30 * c / res["n"]))
    if len(res["counts"]) > 15:
        lines.append(f"   ... {len(res['counts']) - 15} more values")
    lines.append("-" * w)
    lines.append(f" RESULT : {res['verdict']}")
    lines.append(f" Reason : {res['reason']}")
    lines.append("=" * w)
    return "\n".join(lines) + "\n"


def diagnose_param_file(csv_path: str, out_dir: str, threshold: float = DEFAULT_THRESHOLD) -> ParamDiagnosis:
    """Run the full param_diagnose.py pipeline for one var_filtering/<param>.csv: diagnosis,
    PNG plot, and a `<param>_report.dat` text report (the file name Agent_design.md asks for)."""
    label = os.path.splitext(os.path.basename(csv_path))[0]
    v, n_missing, is_numeric = read_values(csv_path)
    if len(v) == 0:
        raise ValueError(f"{csv_path}: all values are missing")
    res = diagnose(v, threshold)
    png_path = os.path.join(out_dir, f"{res['verdict']}_{label}_diagnostics.png")
    dat_path = os.path.join(out_dir, f"{label}_report.dat")
    _plot(v, is_numeric, label, res, png_path)
    report_text = _report_text(label, csv_path, res, n_missing, threshold)
    with open(dat_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    top_counts = {str(k): int(c) for k, c in res["counts"].head(15).items()}
    return ParamDiagnosis(
        label=label, n=res["n"], n_missing=n_missing, is_numeric=is_numeric,
        distinct=len(res["counts"]), top_value=str(res["top"]), top_count=int(res["counts"].iloc[0]),
        share=res["share"], threshold=threshold, verdict=res["verdict"], reason=res["reason"],
        value_counts=top_counts, plot_path=png_path, report_path=dat_path,
    )
