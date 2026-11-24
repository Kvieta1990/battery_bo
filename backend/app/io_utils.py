"""
File helpers shared by every agent: idempotent xlsx/CSV create-or-append, so that re-running
a step after new raw files arrive only adds the new samples instead of duplicating or losing
history (Agent_design.md: "If previous steps have ever been executed before, ... update all
the existing output files by appending the new entries to them.").
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

MISSING_TOKENS = {"", "na", "nan", "nr", "null", "none_"}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_sheet_or_empty(path: str, sheet_name: str) -> pd.DataFrame:
    """Read one sheet of an xlsx file; return an empty DataFrame if the file or sheet doesn't exist yet."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except (ValueError, KeyError):
        return pd.DataFrame()


def list_sheets(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    try:
        return pd.ExcelFile(path).sheet_names
    except Exception:
        return []


def upsert_sheets(path: str, sheets: dict[str, pd.DataFrame], key_col: Optional[str] = None) -> None:
    """
    Write `sheets` (name -> DataFrame) into the workbook at `path`, creating it if needed and
    preserving any other sheets already in the file untouched.

    If key_col is given and already present in an existing sheet of the same name, rows in the
    existing sheet whose key matches a row in the new DataFrame are replaced by the new row
    (so re-processing a sample updates it in place); rows with keys not in the new data are
    kept; genuinely new keys are appended. If key_col is None, the sheet is replaced wholesale
    (used for lookup/report sheets like "Encoding" or "Category codes" that are always
    regenerated in full from current state, not incrementally appended).
    """
    merged: dict[str, pd.DataFrame] = {}
    for name, new_df in sheets.items():
        if key_col:
            existing = read_sheet_or_empty(path, name)
            if new_df is None or new_df.empty or key_col not in new_df.columns:
                # Nothing new for this sheet on this run (e.g. every entry was already seen).
                # Carry the existing sheet forward untouched -- NEVER let an empty/columnless
                # new_df fall through to a wholesale replace, which would silently wipe history.
                merged[name] = existing if not existing.empty else new_df
                continue
            if not existing.empty and key_col in existing.columns:
                keep_existing = existing[~existing[key_col].astype(str).isin(new_df[key_col].astype(str))]
                merged[name] = pd.concat([keep_existing, new_df], ignore_index=True)
            else:
                merged[name] = new_df
        else:
            merged[name] = new_df

    file_exists = os.path.exists(path)
    mode = "a" if file_exists else "w"
    kwargs = {"if_sheet_exists": "replace"} if file_exists else {}
    with pd.ExcelWriter(path, engine="openpyxl", mode=mode, **kwargs) as writer:
        for name, df in merged.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def existing_keys(path: str, sheet_name: str, key_col: str) -> set[str]:
    df = read_sheet_or_empty(path, sheet_name)
    if df.empty or key_col not in df.columns:
        return set()
    return set(df[key_col].astype(str))


def write_param_csv(path: str, sample_values: list[tuple[str, object]]) -> None:
    """Write a var_filtering/<param>.csv in the `sample,value` form param_diagnose.py expects.
    Missing values are written as the literal token NR (param_diagnose.py treats NR as missing)."""
    lines = ["sample,value"]
    for sample, value in sample_values:
        v = "NR" if value is None or (isinstance(value, float) and pd.isna(value)) else value
        lines.append(f"{sample},{v}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def is_missing_token(s) -> bool:
    if s is None:
        return True
    if isinstance(s, float) and pd.isna(s):
        return True
    return str(s).strip().lower() in MISSING_TOKENS
