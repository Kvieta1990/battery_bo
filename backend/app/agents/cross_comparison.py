"""
Target Data Cross Comparison agent (Agent_design.md's fourth sub-agent), plus the capacity
extraction Agent_design.md asks the same agent to do ("analyze the battery measurement data
to extract the capacity").

Capacity extraction: each sheet of battery_data.xlsx is one sample's raw cycle data (pairs of
columns: specific capacity mAh/g, voltage V, repeated per cycle). Verified against the
template's own known Q1_mAh_g values: the first-cycle discharge capacity is the LAST
non-missing value in the cycle-1 capacity column (capacity delivered down to the low-voltage
cutoff), NOT the column max -- the column also contains the charge leg, whose own capacity
(to the high-voltage cutoff) is usually larger due to first-cycle irreversible loss, so taking
the max silently picks up the wrong number.

Two entry points:
  - check_readiness(): a lightweight sample-ID-only comparison (no parameter encoding
    involved) for the frontend's step-3 "ready to configure BO?" indicator, run right after
    the synthesis and battery uploads.
  - build_training_table(selected_params): the full merge into training_data_clean.xlsx, with
    the actual encoded feature columns -- run once the user has chosen which parameters to
    include (lazily, at BO-configuration time), per Agent_design.md's sub-agent ordering.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from .. import config, io_utils
from . import encoding_determination

TARGET_COLUMN = "Q1_mAh_g"


def _battery_files() -> list[str]:
    return sorted(glob.glob(os.path.join(config.BATTERY_DIR, "*.xlsx")))


def extract_capacities(progress_cb=None) -> dict[str, dict]:
    """Extract Q1_mAh_g for every sample sheet across every battery_data/*.xlsx file."""
    progress_cb = progress_cb or (lambda p: None)
    out: dict[str, dict] = {}
    files = _battery_files()
    for path in files:
        try:
            sheets = pd.ExcelFile(path).sheet_names
        except Exception as e:  # noqa: BLE001
            continue
        for i, sheet in enumerate(sheets):
            progress_cb({"stage": "extracting_capacity", "file": os.path.basename(path), "total": len(sheets), "done": i})
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=None)
                cap_col = pd.to_numeric(df.iloc[1:, 0], errors="coerce").dropna()
                volt_col = pd.to_numeric(df.iloc[1:, 1], errors="coerce").dropna() if df.shape[1] > 1 else pd.Series(dtype=float)
                if cap_col.empty:
                    out[sheet] = {"sample": sheet, "source_file": os.path.basename(path), "error": "no numeric capacity data found"}
                    continue
                q1 = float(cap_col.iloc[-1])
                last_v = float(volt_col.iloc[-1]) if not volt_col.empty else None
                anomaly = None
                if last_v is not None and not (config.DISCHARGE_CUTOFF_V_BAND[0] <= last_v <= config.DISCHARGE_CUTOFF_V_BAND[1]):
                    anomaly = (f"cycle-1 data ends at {last_v:.2f} V, outside the usual "
                               f"{config.DISCHARGE_CUTOFF_V_BAND[0]:.1f}-{config.DISCHARGE_CUTOFF_V_BAND[1]:.1f} V "
                               f"discharge-cutoff band -- check this sheet's cycle-1 data")
                out[sheet] = {"sample": sheet, "source_file": os.path.basename(path), "Q1_mAh_g": q1,
                               "last_voltage_V": last_v, "n_points": int(len(cap_col)), "anomaly": anomaly}
            except Exception as e:  # noqa: BLE001
                out[sheet] = {"sample": sheet, "source_file": os.path.basename(path), "error": str(e)}
    return out


def check_readiness(progress_cb=None) -> dict:
    """Lightweight: which sample IDs have BOTH a used synthesis record AND battery capacity data."""
    progress_cb = progress_cb or (lambda p: None)
    used = io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "used")
    synth_samples = set(used["sample"].astype(str)) if not used.empty else set()

    capacities = extract_capacities(progress_cb)
    battery_samples = {s for s, v in capacities.items() if "Q1_mAh_g" in v}
    battery_errors = {s: v["error"] for s, v in capacities.items() if "error" in v}

    matched = sorted(synth_samples & battery_samples)
    synth_only = sorted(synth_samples - battery_samples)
    battery_only = sorted(battery_samples - synth_samples)
    anomalies = [v for v in capacities.values() if v.get("anomaly")]

    ready = len(matched) >= 3   # bayes_opt_core requires >= 3 training samples
    return {
        "ready": ready, "matched_count": len(matched), "matched_samples": matched,
        "synth_only_count": len(synth_only), "synth_only_samples": synth_only,
        "battery_only_count": len(battery_only), "battery_only_samples": battery_only,
        "battery_read_errors": battery_errors, "voltage_anomalies": anomalies,
        "hint": (None if ready else
                 "Fewer than 3 matched samples. Check that sample IDs match exactly between the "
                 "notebook (e.g. 'NMC811-23') and the battery_data.xlsx sheet names -- a stray "
                 "space or a different dash character is the most common cause."),
    }


def build_training_table(selected_params: list[str], progress_cb=None) -> dict:
    progress_cb = progress_cb or (lambda p: None)
    progress_cb({"stage": "extracting_capacity"})
    capacities = extract_capacities(progress_cb)
    cap_df = pd.DataFrame([v for v in capacities.values() if "Q1_mAh_g" in v])[["sample", "Q1_mAh_g"]]

    progress_cb({"stage": "encoding_features"})
    features = encoding_determination.build_training_features(selected_params)

    merged = features.merge(cap_df, on="sample", how="inner")
    unmatched_synth = sorted(set(features["sample"]) - set(merged["sample"]))
    unmatched_battery = sorted(set(cap_df["sample"]) - set(merged["sample"]))

    # Column order matches training_data_full.xlsx's Data sheet: sample, feature columns in
    # CANDIDATE_PARAMS order (decompose/one-hot sub-columns appended after), Q1_mAh_g last.
    ordered_cols = ["sample"] + [c for c in merged.columns if c not in ("sample", TARGET_COLUMN)] + [TARGET_COLUMN]
    merged = merged[ordered_cols]

    progress_cb({"stage": "writing_training_table"})
    # Full replace, not a key_col-merge: unlike condition_proc.xlsx (which only ever grows as
    # new notebook entries arrive), this table is always fully reproducible from
    # condition_proc.xlsx + battery_data/*.xlsx, so there's no history to lose by rebuilding
    # it whole. A key-based merge would instead union this run's columns with every column
    # that ever existed in an older run (even for 0 surviving rows, pandas.concat still unions
    # the two frames' columns), letting retired parameter names silently resurrect themselves.
    io_utils.upsert_sheets(config.TRAINING_DATA_PATH, {"Data": merged}, key_col=None)
    encoding_determination.write_category_codes_sheet(config.TRAINING_DATA_PATH)
    encoding_determination.write_encoding_sheet(config.TRAINING_DATA_PATH, selected_params)

    full = io_utils.read_sheet_or_empty(config.TRAINING_DATA_PATH, "Data")
    return {
        "matched_count": int(len(merged)), "total_training_rows": int(len(full)),
        "unmatched_synthesis_samples": unmatched_synth, "unmatched_battery_samples": unmatched_battery,
        "columns": list(full.columns), "training_path": config.TRAINING_DATA_PATH,
    }
