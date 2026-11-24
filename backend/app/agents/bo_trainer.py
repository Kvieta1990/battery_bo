"""
BO Model Trainer agent (Agent_design.md's fifth sub-agent).

"Generate a script on the fly" is implemented as: assemble the bo_config.json-equivalent
dict from the frontend's model/acquisition/search-space/parameter choices, then run it
through the ported, template-validated core.bayes_opt_core (see that module's docstring for
why the numerics themselves are not LLM-generated). Likewise the diagnostics plots are
computed by core.parity_calibration_core and handed to the frontend as Plotly-ready JSON
instead of a rendered image.

get_parameter_catalog() backs the frontend's parameter multi-select: every resolved column
already materialized in training_data_clean.xlsx (Target Data Cross Comparison writes ALL
resolved parameters there, not a pre-narrowed subset -- see cross_comparison.py), annotated
with near-duplicate/collinearity hints from core.duplicate_check. The user's actual per-run
selection is passed straight through as bo_config["data"]["features"]; no re-merge needed to
change it between runs.
"""
from __future__ import annotations

import pandas as pd

from .. import config, io_utils
from ..core import bayes_opt_core as boc
from ..core import duplicate_check
from ..core import parity_calibration_core as pcc


def get_parameter_catalog() -> dict:
    data = io_utils.read_sheet_or_empty(config.TRAINING_DATA_PATH, "Data")
    if data.empty:
        return {"available": False, "parameters": [], "n_samples": 0}

    feature_cols = [c for c in data.columns if c not in ("sample", "Q1_mAh_g")]
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    usable = [c for c in feature_cols if X[c].notna().all()]
    skipped = [c for c in feature_cols if c not in usable]

    report = duplicate_check.find_duplicate_groups(X[usable], threshold=config.DUPLICATE_THRESHOLD) if usable else None

    group_of, keep_of, sim_of = {}, {}, {}
    if report:
        for g in report.groups:
            for m in g.members:
                group_of[m] = g.members
                keep_of[m] = g.suggested_keep
                sim_of[m] = g.similarity_to_kept.get(m)

    params = []
    for c in usable:
        params.append({
            "name": c, "n_unique": int(X[c].nunique()), "min": float(X[c].min()), "max": float(X[c].max()),
            "is_constant": bool(X[c].nunique() <= 1),
            "default_selected": bool(X[c].nunique() > 1 and (c not in group_of or c == keep_of.get(c))),
            "duplicate_group": group_of.get(c), "suggested_representative": keep_of.get(c),
            "similarity_to_representative": sim_of.get(c),
        })
    return {"available": True, "parameters": params, "skipped_non_numeric": skipped, "n_samples": int(len(data))}


DEFAULT_KERNEL = {"type": "matern", "nu": 2.5, "ard": True, "length_scale_bounds": [0.3, 20],
                   "length_scale_init": 2.0, "amplitude_init": 1.0, "amplitude_bounds": [0.01, 100],
                   "noise_init": 0.1, "noise_bounds": [0.001, 1.0]}
DEFAULT_SAAS = {"alpha": 0.1, "num_warmup": 256, "num_samples": 256, "num_chains": 1,
                "jitter": 1e-6, "predict_chunk": 4000}
DEFAULT_ACQUISITION = {"method": "EI", "xi": 0.0, "kappa": 2.0, "best_reference": "max_observed",
                        "best_filter": "", "ei_mc_samples": 2000}
DEFAULT_BATCH = {"n_suggestions": 5, "strategy": "kriging_believer"}


def auto_search_space(selected_features: list[str], training_df: pd.DataFrame,
                       overrides: dict | None = None) -> dict:
    """Default variable spec per selected feature: `from_data: values` (search exactly the
    measured values) for everything, unless the request overrides a feature's spec. Simple,
    always valid, and matches the template's own stated intent ("searched within the ranges
    seen in the training data"); users who want denser exploration pass an explicit override
    for that one feature (e.g. {"from_data": "range", "step": 10})."""
    overrides = overrides or {}
    variables = {}
    for name in selected_features:
        if name in overrides:
            variables[name] = overrides[name]
        else:
            variables[name] = {"from_data": "values"}
    return variables


def assemble_bo_config(req: dict, training_df: pd.DataFrame) -> dict:
    """Turn the frontend's BO-configuration request into the bo_config.json-equivalent dict
    core.bayes_opt_core.run_bo() expects. This -- not LLM-generated code -- is the "script
    generated on the fly": the structure is fixed and proven, only the content changes per run."""
    model_type = req.get("model_type", "map")
    selected = req["selected_features"]

    model = {"inference": model_type, "kernel": {**DEFAULT_KERNEL, **req.get("kernel", {})},
              "target_transform": req.get("target_transform", {"type": "none"}),
              "n_restarts": req.get("n_restarts", 3), "random_state": req.get("random_state", 0)}
    if model_type == "saas":
        model["saas"] = {**DEFAULT_SAAS, **req.get("saas", {})}

    search_space = {
        "mode": req.get("search_space_mode", "random"), "n_random": req.get("n_random", 20000),
        "random_state": 0, "max_grid_size": 2_000_000,
        "variables": auto_search_space(selected, training_df, req.get("search_space_overrides")),
        "linked_groups": req.get("linked_groups", []), "derived": req.get("derived", {}),
        "constraints": req.get("constraints", []),
    }

    import time
    run_id = req.get("run_id") or f"run_{int(time.time())}"
    return {
        "data": {"file": config.TRAINING_DATA_PATH, "sheet": "Data", "sample_column": "sample",
                  "target_column": "Q1_mAh_g", "features": selected, "exclude_samples": req.get("exclude_samples", [])},
        "model": model,
        "acquisition": {**DEFAULT_ACQUISITION, **req.get("acquisition", {})},
        "search_space": search_space,
        "batch": {**DEFAULT_BATCH, **req.get("batch", {})},
        "diagnostics": {"run_loo": True, "group_regex": req.get("group_regex")},
        "output": {"dir": f"bo_runs/{run_id}", "save_top_candidates": 200},
    }


def run(bo_config: dict, progress_cb=None) -> dict:
    progress_cb = progress_cb or (lambda p: None)
    progress_cb({"stage": "fitting_model"})
    result = boc.run_bo(bo_config, base_dir=config.WORKDIR)

    diagnostics = None
    if result.loo_predictions is not None and len(result.loo_predictions) > 0:
        progress_cb({"stage": "computing_diagnostics"})
        diagnostics = pcc.compute_diagnostics(result.loo_predictions)
        pcc.write_diagnostics_sidecar(diagnostics, result.out_dir)

    return {
        "summary": result.summary,
        "suggestions": result.suggestions.to_dict("records"),
        "candidates_top": result.candidates_top.head(50).to_dict("records"),
        "diagnostics": diagnostics,
        "out_dir": result.out_dir,
        "tag": result.tag,
    }
