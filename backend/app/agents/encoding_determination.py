"""
Encoding Determination agent (Agent_design.md's third sub-agent):
  - numeric values: stay as-is, no scaling (the GP standardizes internally; see bayes_opt_core).
  - obvious binary/categorical text values: deterministic label-encoding (one integer code per
    distinct value), no LLM or user confirmation needed.
  - values that contain numbers but aren't a clean single float (e.g. pre_hold: 'none',
    '350C 5h', '180C 5h; 350C 5h'): genuinely ambiguous. The LLM proposes an encoding
    (schemas.EncodingProposal) and Agent_design.md explicitly calls for "user confirmation
    about the way to do the encoding" before it's applied.
  - high-cardinality free text that duplicates information already captured by other kept
    numeric columns (e.g. 'steps', which just restates T_peak_C/T_final_C/.../T_weighted_C in
    words): recommended for exclusion rather than guessed at.

Resolved encodings (deterministic ones immediately, ambiguous ones once the user confirms)
are persisted to synthesis_data/encoding_codebook.json so re-running later doesn't re-ask.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from .. import config, io_utils
from ..llm_client import structured_call
from ..schemas import EncodingProposal
from .condition_processor import CANDIDATE_PARAMS

CODEBOOK_PATH = os.path.join(config.SYNTHESIS_DIR, "encoding_codebook.json")
HIGH_CARDINALITY_FRACTION = 0.40
ABSENCE_TOKENS = {"none", "nr", "na", "", "no", "0"}

# A value counts as "numeric-looking" only when a number is directly adjoined by a unit-like
# suffix (350C, 5h, 200度, 2.5min, ...) -- i.e. it encodes a measurement -- not merely "contains
# a digit somewhere". That distinction matters: plain category codes/labels such as 'O2' (vs.
# 'air'), or a target name like 'NMC811' (vs. 'NMC955'), also contain digits but are ordinary
# fixed identifiers, not compound measurements the way 'pre_hold' values ('350C 5h') are.
_UNIT_SUFFIX_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:°?[CF]\b|h\b|hr\b|hrs\b|min\b|mins\b|sec\b|secs\b|s\b|%|mg\b|kg\b|g\b|"
    r"ml\b|l\b|nm\b|um\b|rpm\b|度|分|时|小时)", re.IGNORECASE)

PROPOSAL_SYSTEM_PROMPT = """You help decide how to numerically encode one synthesis parameter \
for a Gaussian-process model. The parameter's values look partly numeric but are not a clean \
single number (for example a hold schedule like '350C 5h' or '180C 5h; 350C 5h'). Propose the \
best of: label_encode_by_frequency (ordinal integer codes -- simple, but implies a false order \
and a GP will treat the gap between codes as meaningful distance, which is misleading for a \
compound schedule); one_hot (a 0/1 column per distinct value -- safe but only practical with \
few distinct values and reasonably many samples per value); or decompose_numeric (break the \
string into genuinely numeric sub-features it encodes, e.g. 'number of pre-holds', 'total \
pre-hold time (h)', 'highest pre-hold temperature (C)' -- almost always the best choice when \
the strings encode structured numeric information, since it keeps the real numeric relationships \
instead of replacing them with arbitrary codes or disconnected indicator columns). Give exact \
numeric values for every distinct string you were shown; do not drop any."""


def _load_codebook() -> dict:
    if os.path.exists(CODEBOOK_PATH):
        with open(CODEBOOK_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_codebook(cb: dict) -> None:
    with open(CODEBOOK_PATH, "w", encoding="utf-8") as fh:
        json.dump(cb, fh, indent=2, default=str)


def _clean_values(series: pd.Series) -> pd.Series:
    # Drop real NaN/None FIRST: pandas' astype(str) does not stringify NaN to "nan" (it leaves
    # the float NaN in place inside the resulting object Series), so filtering missing-ness by
    # text-token match alone silently keeps every NaN entry -- fatal for any numeric column
    # with real gaps (e.g. anneal_T_C, blank whenever a sample wasn't annealed).
    non_null = series[series.notna()]
    s = non_null.astype(str).str.strip()
    return s[~s.str.lower().isin(io_utils.MISSING_TOKENS)]


def classify_column(name: str, series: pd.Series) -> dict:
    v = _clean_values(series)
    n = len(v)
    if n == 0:
        return {"parameter": name, "type": "empty", "n_missing": len(series)}
    numeric = pd.to_numeric(v, errors="coerce")
    if numeric.notna().all():
        distinct = set(numeric.unique().tolist())
        kind = "binary" if distinct <= {0.0, 1.0} else "numeric"
        return {"parameter": name, "type": kind, "distinct": len(distinct), "n_missing": len(series) - n}

    distinct_vals = v.value_counts()
    frac_unique = len(distinct_vals) / n
    has_digit_majority = (v.str.contains(_UNIT_SUFFIX_RE)).mean() > 0.5

    if has_digit_majority:
        return {"parameter": name, "type": "ambiguous_numeric_like", "distinct": len(distinct_vals),
                "value_counts": distinct_vals.to_dict(), "n_missing": len(series) - n}
    if frac_unique > HIGH_CARDINALITY_FRACTION and len(distinct_vals) > 4:
        return {"parameter": name, "type": "high_cardinality_text", "distinct": len(distinct_vals),
                "value_counts": distinct_vals.head(10).to_dict(), "n_missing": len(series) - n}
    return {"parameter": name, "type": "categorical", "distinct": len(distinct_vals),
            "value_counts": distinct_vals.to_dict(), "n_missing": len(series) - n}


def _label_encode_by_frequency(value_counts: dict) -> dict[str, int]:
    absence = [v for v in value_counts if str(v).strip().lower() in ABSENCE_TOKENS]
    rest = sorted((v for v in value_counts if v not in absence), key=lambda v: -value_counts[v])
    ordered = absence[:1] + rest if absence else rest
    return {v: i for i, v in enumerate(ordered)}


def run(kept_params: list[str], progress_cb=None) -> dict:
    """Classify every kept parameter. Deterministic cases (numeric/binary/categorical) are
    resolved immediately and written to the codebook. Ambiguous cases get an LLM proposal and
    are returned as `pending_confirmation` for the frontend; nothing about them is persisted
    until confirm_ambiguous() is called."""
    progress_cb = progress_cb or (lambda p: None)
    full_used = io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "used")
    if full_used.empty:
        raise ValueError("No processed synthesis conditions yet -- run the Condition Processor first.")

    codebook = _load_codebook()
    # Drop any codebook entry left over from an earlier CANDIDATE_PARAMS name set (e.g. the
    # calc_atm/target/wash/anneal_atm names retired when CANDIDATE_PARAMS was aligned to the
    # reference sheet) -- otherwise it keeps feeding stale columns into training_data_clean.xlsx
    # and the BO setup tab long after tab #4's Parameters list stopped showing them.
    stale = [k for k in codebook if k not in CANDIDATE_PARAMS]
    for k in stale:
        del codebook[k]
    if stale:
        _save_codebook(codebook)

    resolved, pending, excluded, errors = [], [], [], []

    for i, name in enumerate(kept_params):
        progress_cb({"stage": "classifying", "total": len(kept_params), "done": i})
        if name not in full_used.columns:
            errors.append({"parameter": name, "error": "not found in the processed synthesis data"})
            continue
        info = classify_column(name, full_used[name])
        kind = info["type"]

        if kind in ("numeric", "binary"):
            codebook[name] = {"method": "identity", "type": kind}
            resolved.append({**info, "method": "identity"})
        elif kind == "categorical":
            codes = _label_encode_by_frequency(info["value_counts"])
            codebook[name] = {"method": "label_encode_by_frequency", "codes": codes}
            resolved.append({**info, "method": "label_encode_by_frequency", "codes": codes})
        elif kind == "high_cardinality_text":
            excluded.append({**info, "reason": "high-cardinality free text; likely redundant with other "
                                                 "kept numeric columns (e.g. a thermal-profile summary "
                                                 "string when the profile's numeric columns are already kept)"})
        elif kind == "ambiguous_numeric_like":
            if name in codebook and codebook[name].get("method"):
                resolved.append({**info, "method": codebook[name]["method"], **codebook[name]})
                continue
            try:
                listing = "\n".join(f"- '{v}': {c} samples" for v, c in info["value_counts"].items())
                proposal = structured_call(PROPOSAL_SYSTEM_PROMPT,
                                            f"Parameter: {name}\nDistinct values:\n{listing}", EncodingProposal)
                pending.append({**info, "proposal": proposal.model_dump()})
            except Exception as e:  # noqa: BLE001
                errors.append({"parameter": name, "error": str(e)})
        else:
            errors.append({"parameter": name, "error": f"all values missing"})

    _save_codebook(codebook)
    return {"resolved": resolved, "pending_confirmation": pending, "excluded_recommended": excluded, "errors": errors}


def confirm_ambiguous(parameter: str, method: str, codes: dict | None = None,
                       decomposed_features: list[dict] | None = None) -> dict:
    """Persist the user's confirmed (or edited) resolution for one ambiguous parameter."""
    codebook = _load_codebook()
    entry = {"method": method}
    if method == "label_encode_by_frequency":
        entry["codes"] = codes or {}
    elif method == "decompose_numeric":
        entry["decomposed_features"] = decomposed_features or []
    elif method == "one_hot":
        entry["codes"] = codes or {}
    codebook[parameter] = entry
    _save_codebook(codebook)
    return entry


def list_resolved_parameters() -> list[str]:
    """Every parameter with a persisted encoding resolution (deterministic ones from run(),
    plus any ambiguous ones the user has confirmed via confirm_ambiguous()), restricted to the
    current CANDIDATE_PARAMS set so a stale codebook entry from a retired parameter name never
    leaks into training_data_clean.xlsx / the BO setup tab on its own."""
    return [k for k in _load_codebook() if k in CANDIDATE_PARAMS]


def build_training_features(selected_params: list[str]) -> pd.DataFrame:
    """Apply every resolved encoding in the codebook to produce the final numeric feature
    table (sample column + one numeric column per selected parameter, one-hot/decomposed
    parameters may expand into more than one output column). Columns are emitted in
    CANDIDATE_PARAMS order (matching training_data_full.xlsx's Data sheet column order)
    rather than whatever order the frontend happened to select them in; any column not in
    CANDIDATE_PARAMS (e.g. a decompose_numeric sub-feature) is appended at the end in the
    order it was generated."""
    full_used = io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "used")
    codebook = _load_codebook()
    out = pd.DataFrame({"sample": full_used["sample"]})

    ordered_params = [p for p in CANDIDATE_PARAMS if p in selected_params] + \
                      [p for p in selected_params if p not in CANDIDATE_PARAMS]

    for name in ordered_params:
        if name not in full_used.columns:
            continue
        entry = codebook.get(name, {"method": "identity"})
        col = full_used[name]
        method = entry.get("method", "identity")

        if method == "identity":
            numeric = pd.to_numeric(col, errors="coerce")
            if name in ("anneal_T_C", "anneal_h"):      # matches the template's documented rule:
                numeric = numeric.fillna(0.0)            # "not annealed (blank) = 0, as in the GP"
            out[name] = numeric
        elif method == "label_encode_by_frequency":
            codes = entry.get("codes", {})
            out[name] = col.astype(str).str.strip().map(codes)
        elif method == "one_hot":
            codes = entry.get("codes", {})
            for value in codes:
                safe = "".join(c if c.isalnum() else "_" for c in str(value))
                out[f"{name}__{safe}"] = (col.astype(str).str.strip() == value).astype(int)
        elif method == "decompose_numeric":
            for feat in entry.get("decomposed_features", []):
                values = feat.get("values", {})
                out[feat["name"]] = col.astype(str).str.strip().map(values)

    return out


def write_category_codes_sheet(training_path: str) -> None:
    """Mirror the template's single 'Category codes' sheet: one stacked block per
    label-encoded parameter (a name row, then a Code/Original value/Cells(n) header and rows),
    for transparency when the training workbook is opened directly."""
    codebook = _load_codebook()
    label_encoded = {k: v for k, v in codebook.items() if v.get("method") == "label_encode_by_frequency"}

    full_used = io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "used")
    block_rows = []
    for name, entry in label_encoded.items():
        counts = full_used[name].astype(str).str.strip().value_counts().to_dict() if name in full_used.columns else {}
        rows = sorted(entry["codes"].items(), key=lambda kv: kv[1])
        block_rows.append({"Code": name, "Original value": "", "Cells (n)": ""})       # block header row
        block_rows.append({"Code": "Code", "Original value": "Original value", "Cells (n)": "Cells (n)"})
        for value, code in rows:
            block_rows.append({"Code": code, "Original value": value, "Cells (n)": counts.get(value, 0)})
        block_rows.append({"Code": "", "Original value": "", "Cells (n)": ""})         # spacer
    if block_rows:
        io_utils.upsert_sheets(training_path, {"Category codes": pd.DataFrame(block_rows)}, key_col=None)

    # Drop any one-sheet-per-parameter "codes_<name>" sheet from an earlier version of this
    # function (superseded by the single consolidated sheet above), and any leftover sheet for
    # a parameter no longer in CANDIDATE_PARAMS -- upsert_sheets only ever adds/replaces sheets,
    # never removes one, so a stale "codes_*" sheet would otherwise linger in the workbook forever.
    if os.path.exists(training_path):
        import openpyxl
        wb = openpyxl.load_workbook(training_path)
        changed = False
        for sheet_name in list(wb.sheetnames):
            if sheet_name.startswith("codes_"):
                del wb[sheet_name]
                changed = True
        if changed:
            wb.save(training_path)


# Unit, by column-name convention -- matches the units already used throughout chem_data.py
# and condition_processor.py for these exact columns (not guessed per-column).
_UNIT_BY_SUFFIX = [
    ("_per_TM", "mol/mol TM"), ("_pct_TM", "mol % of TM"),
    ("_C_min", "deg C/min"), ("_C", "deg C"), ("_h", "h"),
]
_UNIT_OVERRIDES = {"salt_solid_mass_ratio": "g/g", "milled": "-", "dried_powder": "-",
                   "anneal_O2": "-", "is955": "-", "pre_hold": "-"}
# Columns whose encoded value is derived from a differently-named column in condition_proc.xlsx.
_SOURCE_COLUMN_OVERRIDES = {"is955": "target", "anneal_O2": "anneal_atm"}

_METHOD_RULE = {
    "label_encode_by_frequency": "categorical, label-encoded by frequency; see 'Category codes' sheet.",
    "one_hot": "categorical, one-hot encoded into one 0/1 column per distinct value.",
    "decompose_numeric": "decomposed into numeric sub-features; see the generated columns.",
}
_IDENTITY_RULE = {
    "binary": "binary, copied unchanged from condition_proc.xlsx (1 = yes, 0 = no).",
    "numeric": "numeric, copied unchanged from condition_proc.xlsx (no scaling).",
}


def _unit_for(name: str) -> str:
    if name in _UNIT_OVERRIDES:
        return _UNIT_OVERRIDES[name]
    for suffix, unit in _UNIT_BY_SUFFIX:
        if name.endswith(suffix):
            return unit
    return "-"


def write_encoding_sheet(training_path: str, selected_params: list[str]) -> None:
    """Mirror the template's 'Encoding' sheet: one row per output column documenting its
    type, unit, source column in condition_proc.xlsx, and the applied encoding rule."""
    codebook = _load_codebook()
    ordered = [p for p in CANDIDATE_PARAMS if p in selected_params] + \
              [p for p in selected_params if p not in CANDIDATE_PARAMS]
    rows = []
    for name in ordered:
        entry = codebook.get(name, {"method": "identity"})
        method = entry.get("method", "identity")
        rule = _IDENTITY_RULE.get(entry.get("type"), _METHOD_RULE.get(method, method)) \
            if method == "identity" else _METHOD_RULE.get(method, method)
        rows.append({
            "Column": name, "Type": entry.get("type", method), "Unit": _unit_for(name),
            "Source column (condition_proc.xlsx)": _SOURCE_COLUMN_OVERRIDES.get(name, name),
            "Encoding rule": rule,
        })
    rows.append({"Column": "Q1_mAh_g", "Type": "target", "Unit": "mAh/g",
                 "Source column (condition_proc.xlsx)": "battery_data.xlsx (sheet per cell)",
                 "Encoding rule": "First-cycle discharge capacity, not a parameter."})
    io_utils.upsert_sheets(training_path, {"Encoding": pd.DataFrame(rows)}, key_col=None)
