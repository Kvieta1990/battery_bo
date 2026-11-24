"""
Diagnostics agent's deterministic core — ported from the reference `parity_calibration.py`,
with the same math (MAE/RMSE/Q2/Pearson/Spearman, 68%/95% interval coverage, the calibration
curve and its binomial band) but returning plain JSON-ready dicts for Plotly instead of a
matplotlib PNG, per the user's request for interactive diagnostics in the frontend.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


def standardized_errors(d: pd.DataFrame):
    if "z_latent" in d.columns:
        return d["z_latent"].to_numpy(float), "latent (transformed) space"
    if "loo_std" in d.columns:
        return ((d["observed"] - d["loo_pred"]) / d["loo_std"]).to_numpy(float), "capacity units (symmetric)"
    return None, None


def coverage_curve(z: np.ndarray, levels: np.ndarray) -> np.ndarray:
    half = norm.ppf(0.5 + levels / 2)
    return np.array([(np.abs(z) <= h).mean() for h in half])


def _calibration_verdict(obs: float, nominal: float, n: int) -> str:
    sd = np.sqrt(nominal * (1 - nominal) / n)
    if obs < nominal - 2 * sd:
        return "overconfident (intervals too narrow)"
    if obs > nominal + 2 * sd:
        return "underconfident (intervals too wide)"
    return "consistent with good calibration"


def compute_diagnostics(loo_df: pd.DataFrame, units: str = "mAh/g", label_worst: int = 4) -> dict:
    """loo_df: the loo_predictions DataFrame written by run_bo() (observed, loo_pred, and
    either lo68/hi68/lo95/hi95/z_latent or loo_std). Returns a JSON-ready dict for the
    frontend to render with Plotly (parity scatter + calibration curve) plus the same
    summary numbers parity_calibration.py prints to the terminal."""
    d = loo_df.reset_index(drop=True)
    for c in ("observed", "loo_pred"):
        if c not in d.columns:
            raise ValueError(f"column '{c}' missing from the LOO predictions")
    scol = d.columns[0]
    y, p = d["observed"].to_numpy(float), d["loo_pred"].to_numpy(float)
    n = len(d)

    if {"lo68", "hi68"} <= set(d.columns):
        err_lo = (p - d["lo68"]).clip(lower=0).to_numpy(float)
        err_hi = (d["hi68"] - p).clip(lower=0).to_numpy(float)
        bar_txt = "error bars: central 68 % interval"
    elif "loo_std" in d.columns:
        err_lo = err_hi = d["loo_std"].to_numpy(float)
        bar_txt = "error bars: ±1σ"
    else:
        err_lo = err_hi = None
        bar_txt = "no uncertainty in the input"

    mae = float(np.abs(p - y).mean())
    rmse = float(np.sqrt(((p - y) ** 2).mean()))
    base = d["baseline_pred"].to_numpy(float) if "baseline_pred" in d.columns else (y.sum() - y) / (n - 1)
    mae0 = float(np.abs(base - y).mean())
    rmse0 = float(np.sqrt(((base - y) ** 2).mean()))
    denom = ((y - base) ** 2).sum()
    q2 = float(1 - ((y - p) ** 2).sum() / denom) if denom > 0 else float("nan")
    r = float(np.corrcoef(y, p)[0, 1])
    rho, rho_p = spearmanr(y, p)

    z, zspace = standardized_errors(d)
    has_cal = z is not None
    calibration = None
    in68 = in95 = None
    if has_cal:
        in68, in95 = float((np.abs(z) <= 1).mean()), float((np.abs(z) <= 2).mean())
        levels = np.linspace(0.0, 0.99, 100)
        cov = coverage_curve(z, levels)
        band = 1.96 * np.sqrt(levels * (1 - levels) / n)
        calibration = {
            "levels": levels.tolist(), "coverage": cov.tolist(),
            "band_lo": np.clip(levels - band, 0, 1).tolist(), "band_hi": np.clip(levels + band, 0, 1).tolist(),
            "mean_z": float(z.mean()), "sd_z": float(z.std(ddof=1)), "zspace": zspace,
            "in68": in68, "in95": in95,
            "verdict68": _calibration_verdict(in68, 0.683, n), "verdict95": _calibration_verdict(in95, 0.954, n),
        }

    worst_idx = np.argsort(-np.abs(p - y))[:label_worst] if label_worst else np.array([], dtype=int)
    lo = min(float(y.min()), float(p.min())) - 15
    hi = max(float(y.max()), float(p.max())) + 15
    if err_lo is not None:
        lo = min(lo, float((p - err_lo).min()) - 5)
        hi = max(hi, float((p + err_hi).max()) + 5)

    diag_txt = (f"n = {n}    MAE = {mae:.1f}  (baseline {mae0:.1f})    RMSE = {rmse:.1f}  (baseline {rmse0:.1f})    "
                f"Q² = {q2:.2f}    r = {r:.2f}    ρ = {rho:.2f}    {bar_txt}")

    largest_errors = [
        {"sample": str(d.iloc[i][scol]), "observed": float(y[i]), "predicted": float(p[i]),
         "z": float(z[i]) if has_cal else None}
        for i in np.argsort(-np.abs(p - y))[:5]
    ]

    return {
        "n": n, "units": units, "axis_range": [lo, hi],
        "parity": {
            "sample": d[scol].astype(str).tolist(), "observed": y.tolist(), "predicted": p.tolist(),
            "err_lo": (err_lo.tolist() if err_lo is not None else None),
            "err_hi": (err_hi.tolist() if err_hi is not None else None),
            "worst_idx": worst_idx.tolist(), "bar_txt": bar_txt,
        },
        "calibration": calibration,
        "stats": {
            "mae": mae, "mae_baseline": mae0, "rmse": rmse, "rmse_baseline": rmse0,
            "q2": q2, "pearson_r": r, "spearman_rho": float(rho), "spearman_p": float(rho_p),
        },
        "largest_errors": largest_errors,
        "diag_text": diag_txt,
        "has_calibration": has_cal,
    }


def write_diagnostics_sidecar(diag: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, "parity_calibration_diagnostics.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(diag["diag_text"] + "\n")
    return path
