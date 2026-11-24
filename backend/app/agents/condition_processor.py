"""
Condition Processor agent (Agent_design.md's first sub-agent).

Reads raw lab-notebook entries (free text — a .docx with one paragraph per sample, or a
plain .txt/.md file) and turns them into the clean `used`/`unused` synthesis-condition sheet,
plus the per-parameter CSVs in synthesis_data/var_filtering/ that the Parameters Filter agent
reads next.

Design: the LLM (via llm_client.structured_call) only does the free-text -> structured-JSON
extraction (schemas.ConditionExtraction); every derived quantity (molar ratios, thermal-profile
summary) is then computed deterministically here, ported from the reference parse_cond_v2.py's
chemistry but driven off the structured extraction instead of a notebook-specific regex. A
sample is masked (sent to the `unused` sheet) when the LLM could not fully determine it, an
unknown reagent has no known molecular weight, or no thermal steps were found.

Re-running this over a notebook file that includes previously-seen entries costs no extra LLM
calls: every entry (including ones that turned out not to be a sample record, e.g. a section
header) is hashed and checked against synthesis_data/seen_entries.json before extraction.
"""
from __future__ import annotations

import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import docx
import hashlib
import json
import pandas as pd

from .. import chem_data, config, io_utils
from ..llm_client import structured_call

SEEN_ENTRIES_PATH = os.path.join(config.SYNTHESIS_DIR, "seen_entries.json")


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_seen_hashes() -> set[str]:
    if os.path.exists(SEEN_ENTRIES_PATH):
        with open(SEEN_ENTRIES_PATH, encoding="utf-8") as fh:
            return set(json.load(fh))
    return set()


def _save_seen_hashes(hashes: set[str]) -> None:
    with open(SEEN_ENTRIES_PATH, "w", encoding="utf-8") as fh:
        json.dump(sorted(hashes), fh)
from ..schemas import ConditionExtraction

SYSTEM_PROMPT = """You are reading one entry from a battery-materials synthesis lab notebook. \
Each entry describes how one sample was made: the feed reagents and their amounts, the \
calcination thermal profile (temperature holds and ramp rates), the wash procedure, and an \
optional post-wash anneal. Entries may be written in English, Chinese, or a mix, and may use \
different levels of detail. Extract the structured fields exactly as specified. If the given \
text is not a sample synthesis record at all (e.g. a section title or blank heading), return \
sample_id as an empty string and leave other fields at reasonable defaults. Never invent a \
value that is not stated or clearly computable from what is stated -- list anything you could \
not determine in missing_fields instead of guessing."""

CANDIDATE_PARAMS = [
    # Kept in exact 1:1 correspondence with training_data_full.xlsx's feature columns (minus
    # `sample` and the `Q1_mAh_g` target) so the Parameters Filter step's KEEP/IGNORE list
    # matches the reference sheet name-for-name, AND in that sheet's own column order so the
    # Data sheet built from this list reproduces it. Fields computed above but not in that
    # sheet (calc_atm, target, Al_pct_TM, K_per_TM, sulfate_per_TM, steps, wash, anneal_atm,
    # anneal_LiOH_wt) are still written to the used/unused sheet -- just not surfaced here.
    "milled", "T_peak_C", "T_final_C", "t_final_h", "ramp_main_C_min",
    "Li_total_per_TM", "La_pct_TM", "Mn_add_pct_TM", "anneal_T_C", "anneal_O2",
    "Na_per_TM", "is955", "exOH_per_TM", "dried_powder", "anneal_h", "pre_hold",
    "t_peak_h", "t_total_hi_h", "T_weighted_C", "Li_salt_per_TM", "OH_per_TM",
    "nitrate_per_TM", "chloride_per_TM", "salt_solid_mass_ratio",
]

CONDITION_PROC_SHEETS = ("used", "unused")


def extract_entries(path: str) -> list[str]:
    """Split a raw notebook file into candidate per-sample text blocks."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        doc = docx.Document(path)
        return [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip().replace("\n", " ") for b in blocks if b.strip()]


def _compute_derived(ex: ConditionExtraction) -> tuple[dict, bool, list[str]]:
    """Port of parse_cond_v2.py's chemistry, driven off a ConditionExtraction instead of regex.
    Returns (row_dict, masked, masked_fields)."""
    masked_fields = list(ex.missing_fields)
    row: dict = {
        "sample": ex.sample_id, "target": ex.target_material, "raw_text": None,  # filled by caller
        "milled": int(bool(ex.ball_milled)) if ex.ball_milled is not None else None,
        "dried_powder": int(bool(ex.dried_powder)) if ex.dried_powder is not None else None,
        "calc_atm": ex.calcination_atmosphere or "NR",
        # Matches training_data_full.xlsx's convention: 1 if the target product family is the
        # NMC955 variant, else 0 (covers NMC811 and any other non-955 target).
        "is955": int("955" in ex.target_material),
    }
    if ex.ball_milled is None:
        masked_fields.append("ball_milled")
    if ex.dried_powder is None:
        masked_fields.append("dried_powder")

    # ---- feed composition -> moles -----------------------------------------------------
    amt_mg: dict[str, float] = {}
    unknown_reagents = []
    for r in ex.reagents:
        canon = chem_data.canonical_name(r.name)
        if not chem_data.known_reagent(canon):
            unknown_reagents.append(r.name)
            continue
        mg = r.amount * (1000.0 if r.unit == "g" else 1.0)
        amt_mg[canon] = amt_mg.get(canon, 0.0) + mg
    for name in unknown_reagents:
        masked_fields.append(f"unknown_reagent:{name}")
    for canon, mg in amt_mg.items():
        row[f"m_{canon}"] = mg

    n = {k: v / chem_data.molar_mass(k) for k, v in amt_mg.items()}
    n111 = n.get("NMC111", 0.0)
    nNi = sum(n.get(k, 0.0) for k in chem_data.NI_SOURCE_REAGENTS)
    nMn = sum(n.get(k, 0.0) for k in chem_data.MN_SOURCE_REAGENTS)
    nLa = sum(n.get(k, 0.0) for k in chem_data.LA_SOURCE_REAGENTS)
    nAl = sum(n.get(k, 0.0) for k in chem_data.AL_SOURCE_REAGENTS)
    TM = n111 + nNi + nMn

    if TM <= 0:
        masked_fields.append("transition_metal_pool_zero")
        row.update({"Ni_frac_feed": None, "Co_frac_feed": None, "Mn_frac_feed": None,
                    "Mn_add_pct_TM": None, "La_pct_TM": None, "Al_pct_TM": None,
                    "Li_salt_per_TM": None, "Li_total_per_TM": None, "Na_per_TM": None, "K_per_TM": None,
                    "nitrate_per_TM": None, "chloride_per_TM": None, "OH_per_TM": None, "exOH_per_TM": None,
                    "sulfate_per_TM": None, "salt_solid_mass_ratio": None})
    else:
        row["Ni_frac_feed"] = (n111 / 3 + nNi) / TM
        row["Co_frac_feed"] = (n111 / 3) / TM
        row["Mn_frac_feed"] = (n111 / 3 + nMn) / TM
        row["Mn_add_pct_TM"] = 100 * nMn / TM
        row["La_pct_TM"] = 100 * nLa / TM
        row["Al_pct_TM"] = 100 * nAl / TM
        li_salt = sum(n.get(k, 0.0) * v for k, v in chem_data.LI_PER_MOL.items())
        row["Li_salt_per_TM"] = li_salt / TM
        row["Li_total_per_TM"] = (li_salt + n111) / TM
        row["Na_per_TM"] = sum(n.get(k, 0.0) * v for k, v in chem_data.NA_PER_MOL.items()) / TM
        row["K_per_TM"] = sum(n.get(k, 0.0) * v for k, v in chem_data.K_PER_MOL.items()) / TM
        row["nitrate_per_TM"] = sum(n.get(k, 0.0) * v for k, v in chem_data.NITRATE_PER_MOL.items()) / TM
        row["chloride_per_TM"] = sum(n.get(k, 0.0) * v for k, v in chem_data.CHLORIDE_PER_MOL.items()) / TM
        oh = sum(n.get(k, 0.0) * v for k, v in chem_data.HYDROXIDE_PER_MOL.items())
        row["OH_per_TM"] = oh / TM
        row["exOH_per_TM"] = (oh - 2 * nNi) / TM
        row["sulfate_per_TM"] = sum(n.get(k, 0.0) * v for k, v in chem_data.SULFATE_PER_MOL.items()) / TM
        non_nmc_mass = sum(v for k, v in amt_mg.items() if k != "NMC111")
        row["salt_solid_mass_ratio"] = non_nmc_mass / amt_mg.get("NMC111", 1.0)

    # ---- thermal profile ----------------------------------------------------------------
    steps = [(s.temp_C, s.hold_h) for s in ex.thermal_steps]
    if not steps:
        masked_fields.append("thermal_steps")
        row.update({"steps": None, "T_peak_C": None, "t_peak_h": None, "T_final_C": None, "t_final_h": None,
                    "t_total_hi_h": None, "T_weighted_C": None, "pre_hold": None})
    else:
        row["steps"] = " → ".join(f"{T:.0f}°C {h:g}h" for T, h in steps)
        hi = [s for s in steps if s[0] >= chem_data.HIGH_TEMP_THRESHOLD_C]
        if hi:
            t_peak = max(s[0] for s in hi)
            row["T_peak_C"] = t_peak
            row["t_peak_h"] = max(s[1] for s in hi if s[0] == t_peak)
            row["T_final_C"] = hi[-1][0]
            row["t_final_h"] = hi[-1][1]
            row["t_total_hi_h"] = sum(s[1] for s in hi)
            row["T_weighted_C"] = sum(T * h for T, h in hi) / row["t_total_hi_h"]
        else:
            masked_fields.append("no_high_temp_step")
            row.update({"T_peak_C": None, "t_peak_h": None, "T_final_C": None, "t_final_h": None,
                        "t_total_hi_h": None, "T_weighted_C": None})
        pre = [s for s in steps if s[0] < chem_data.HIGH_TEMP_THRESHOLD_C]
        row["pre_hold"] = "; ".join(f"{T:.0f}°C {h:g}h" for T, h in pre) if pre else "none"

    ramps = ex.ramp_rates_C_per_min
    row["ramp_main_C_min"] = ramps[1] if len(ramps) > 1 else (ramps[0] if ramps else None)
    if not ramps:
        masked_fields.append("ramp_rate")

    row["wash"] = ex.wash_procedure or "NR"
    if not ex.wash_procedure:
        masked_fields.append("wash_procedure")

    if ex.anneal is not None:
        row["anneal_T_C"] = ex.anneal.temp_C
        row["anneal_atm"] = ex.anneal.atmosphere
        row["anneal_O2"] = int(ex.anneal.atmosphere == "O2")
        row["anneal_h"] = ex.anneal.hold_h
        row["anneal_LiOH_wt"] = ex.anneal.liOH_wt_pct if ex.anneal.liOH_wt_pct is not None else (
            2.0 if ex.anneal.liOH_added else 0.0)
    else:
        row["anneal_T_C"] = None
        row["anneal_atm"] = "none"
        row["anneal_O2"] = 0
        row["anneal_h"] = None
        row["anneal_LiOH_wt"] = 0.0

    flags = []
    if ex.notes:
        flags.append(ex.notes)
    dup_reagents = [r.name for r in ex.reagents if sum(1 for r2 in ex.reagents if r2.name == r.name) > 1]
    if dup_reagents:
        flags.append(f"{dup_reagents[0]} listed more than once; amounts summed")
    row["flags"] = "; ".join(dict.fromkeys(flags))  # de-dup, keep order

    masked = len(masked_fields) > 0
    row["masked"] = masked
    row["masked_fields"] = "; ".join(masked_fields)
    return row, masked, masked_fields


def _process_one(raw_text: str) -> Optional[dict]:
    extraction = structured_call(SYSTEM_PROMPT, raw_text, ConditionExtraction)
    if not extraction.sample_id.strip():
        return None  # not a sample record (e.g. a section header)
    row, _, _ = _compute_derived(extraction)
    row["raw_text"] = raw_text
    return row


def run(notebook_path: str, progress_cb=None, max_workers: int = 6) -> dict:
    progress_cb = progress_cb or (lambda p: None)
    entries = extract_entries(notebook_path)

    # Dedup against every entry ever submitted -- not just ones that became a used/unused row --
    # so a header line or other non-sample text that recurs on every export isn't re-sent to the
    # LLM on every re-run; only genuinely new raw text costs a fresh extraction call. Also folds
    # in the used/unused sheets' own raw_text (covers condition_proc.xlsx rows that predate this
    # sidecar file, e.g. carried over from an earlier pipeline).
    seen_hashes = _load_seen_hashes()
    for df in (io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "used"),
               io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "unused")):
        if not df.empty and "raw_text" in df.columns:
            seen_hashes.update(_text_hash(t) for t in df["raw_text"].astype(str))
    new_entries = [e for e in entries if _text_hash(e) not in seen_hashes]
    progress_cb({"stage": "extracting", "total": len(new_entries), "done": 0, "skipped_already_seen": len(entries) - len(new_entries)})

    rows, skipped_not_sample, errors, succeeded_texts = [], [], [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, text): text for text in new_entries}
        done = 0
        for fut in as_completed(futures):
            text = futures[fut]
            done += 1
            try:
                row = fut.result()
                succeeded_texts.append(text)
                if row is None:
                    skipped_not_sample.append(text[:80])
                else:
                    rows.append(row)
            except Exception as e:  # noqa: BLE001
                errors.append({"text": text[:120], "error": str(e)})  # NOT marked seen -- retried next run
            progress_cb({"stage": "extracting", "total": len(new_entries), "done": done})

    _save_seen_hashes(seen_hashes | {_text_hash(t) for t in succeeded_texts})

    new_df = pd.DataFrame(rows)
    used_new = new_df[~new_df["masked"]].copy() if not new_df.empty else new_df
    unused_new = new_df[new_df["masked"]].copy() if not new_df.empty else new_df

    progress_cb({"stage": "writing_sheets"})
    io_utils.upsert_sheets(config.CONDITION_PROC_PATH, {"used": used_new, "unused": unused_new}, key_col="sample")

    full_used = io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "used")
    full_unused = io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "unused")

    # Backfill columns that were added to CANDIDATE_PARAMS after earlier entries were already
    # processed (is955/anneal_O2) -- both are pure functions of columns already stored, so no
    # LLM re-call is needed; this makes old sheets self-heal on the next Parameters Filter run
    # instead of silently missing those columns in var_filtering/.
    backfilled = False
    for name, df in (("used", full_used), ("unused", full_unused)):
        if df.empty:
            continue
        if "is955" not in df.columns and "target" in df.columns:
            df["is955"] = df["target"].astype(str).str.contains("955").astype(int)
            backfilled = True
        if "anneal_O2" not in df.columns and "anneal_atm" in df.columns:
            df["anneal_O2"] = (df["anneal_atm"] == "O2").astype(int)
            backfilled = True
    if backfilled:
        io_utils.upsert_sheets(config.CONDITION_PROC_PATH, {"used": full_used, "unused": full_unused})

    # Remove any var_filtering/<param>.csv left over from a parameter that's no longer in
    # CANDIDATE_PARAMS (e.g. a prior name mismatch against the reference sheet) so the
    # Parameters Filter step's output set matches CANDIDATE_PARAMS exactly, not "whatever
    # files happen to be on disk".
    for stale in glob.glob(os.path.join(config.VAR_FILTERING_DIR, "*.csv")):
        if os.path.splitext(os.path.basename(stale))[0] not in CANDIDATE_PARAMS:
            os.remove(stale)

    progress_cb({"stage": "writing_var_filtering"})
    for param in CANDIDATE_PARAMS:
        if param not in full_used.columns:
            continue
        pairs = list(zip(full_used["sample"].astype(str), full_used[param]))
        if not pairs:
            continue
        io_utils.write_param_csv(os.path.join(config.VAR_FILTERING_DIR, f"{param}.csv"), pairs)

    return {
        "new_entries_found": len(new_entries),
        "skipped_already_seen": len(entries) - len(new_entries),
        "skipped_not_sample_record": skipped_not_sample,
        "newly_used": int(len(used_new)),
        "newly_unused": int(len(unused_new)),
        "total_used": int(len(full_used)),
        "total_unused": int(len(full_unused)),
        "errors": errors,
        "flagged_samples": (full_used[full_used["flags"].astype(str).str.len() > 0][["sample", "flags"]]
                             .to_dict("records") if "flags" in full_used.columns and not full_used.empty else []),
        "unused_preview": (full_unused[["sample", "masked_fields"]].to_dict("records")
                            if "masked_fields" in full_unused.columns and not full_unused.empty else []),
    }
