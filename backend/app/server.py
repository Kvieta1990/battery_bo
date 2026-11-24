"""
FastAPI app wiring the five Agent_design.md sub-agents to the frontend wizard. Run with:
    uvicorn app.server:app --port 8765
from inside backend/ (see the top-level README for full setup). Every endpoint that can take
more than a second or two (anything calling the LLM, or fitting a GP/SAASBO model) starts a
background job and returns a job id immediately; the frontend polls GET /api/jobs/{id}.
"""
from __future__ import annotations

import os
import shutil

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import config, io_utils
from .agents import bo_trainer, condition_processor, cross_comparison, encoding_determination, parameter_filter
from .jobs import jobs

app = FastAPI(title="Battery BO Platform API")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ALLOW_ORIGINS, allow_credentials=False,
                    allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------------- health
@app.get("/api/health")
def health():
    return {"status": "ok", "workdir": config.WORKDIR, "llm_provider": config.LLM_PROVIDER,
            "llm_configured": config.llm_configured()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    d = jobs.to_dict(job_id)
    if d is None:
        raise HTTPException(404, "unknown job id")
    return d


# --------------------------------------------------------------------------------- step 1: synthesis / condition processor
@app.post("/api/synthesis/upload")
async def synthesis_upload(file: UploadFile = File(...)):
    dest = os.path.join(config.INCOMING_NOTEBOOKS_DIR, file.filename)
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"path": dest, "filename": file.filename}


@app.post("/api/synthesis/process")
def synthesis_process(body: dict):
    path = body["path"]
    if not os.path.exists(path):
        raise HTTPException(404, f"file not found: {path}")
    job_id = jobs.start("condition_processor", condition_processor.run, path)
    return {"job_id": job_id}


@app.get("/api/synthesis/status")
def synthesis_status():
    used = io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "used")
    unused = io_utils.read_sheet_or_empty(config.CONDITION_PROC_PATH, "unused")
    return {"total_used": int(len(used)), "total_unused": int(len(unused)),
            "used_samples": sorted(used["sample"].astype(str)) if not used.empty else [],
            "unused_samples": sorted(unused["sample"].astype(str)) if not unused.empty else []}


# --------------------------------------------------------------------------------- step 2: battery data
@app.post("/api/battery/upload")
async def battery_upload(file: UploadFile = File(...)):
    dest = os.path.join(config.BATTERY_DIR, file.filename)
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return {"path": dest, "filename": file.filename}


@app.get("/api/battery/capacities")
def battery_capacities():
    return {"capacities": cross_comparison.extract_capacities()}


# --------------------------------------------------------------------------------- step 3: readiness
@app.get("/api/readiness")
def readiness():
    return cross_comparison.check_readiness()


# --------------------------------------------------------------------------------- parameters filter / encoding
@app.post("/api/parameters/filter/run")
def parameters_filter_run(body: dict | None = None):
    threshold = (body or {}).get("threshold")
    job_id = jobs.start("parameter_filter", parameter_filter.run, threshold=threshold)
    return {"job_id": job_id}


@app.post("/api/parameters/encoding/run")
def parameters_encoding_run(body: dict):
    kept_params = body["kept_params"]
    job_id = jobs.start("encoding_determination", encoding_determination.run, kept_params)
    return {"job_id": job_id}


@app.post("/api/parameters/encoding/confirm")
def parameters_encoding_confirm(body: dict):
    entry = encoding_determination.confirm_ambiguous(
        parameter=body["parameter"], method=body["method"],
        codes=body.get("codes"), decomposed_features=body.get("decomposed_features"))
    return {"parameter": body["parameter"], "resolved": entry}


@app.get("/api/parameters/encoding/resolved")
def parameters_encoding_resolved():
    return {"resolved_parameters": encoding_determination.list_resolved_parameters()}


# --------------------------------------------------------------------------------- training table (cross comparison)
@app.post("/api/training/build")
def training_build(body: dict | None = None):
    params = (body or {}).get("params") or encoding_determination.list_resolved_parameters()
    job_id = jobs.start("cross_comparison", cross_comparison.build_training_table, params)
    return {"job_id": job_id}


@app.get("/api/training/status")
def training_status():
    df = io_utils.read_sheet_or_empty(config.TRAINING_DATA_PATH, "Data")
    return {"exists": not df.empty, "n_rows": int(len(df)), "columns": list(df.columns)}


# --------------------------------------------------------------------------------- BO configuration + run
@app.get("/api/bo/parameters")
def bo_parameters():
    return bo_trainer.get_parameter_catalog()


@app.post("/api/bo/run")
def bo_run(body: dict):
    training_df = io_utils.read_sheet_or_empty(config.TRAINING_DATA_PATH, "Data")
    if training_df.empty:
        raise HTTPException(400, "No training data yet -- build the training table first (/api/training/build).")
    cfg = bo_trainer.assemble_bo_config(body, training_df)
    job_id = jobs.start("bo_trainer", bo_trainer.run, cfg)
    return {"job_id": job_id, "resolved_config": cfg}


# --------------------------------------------------------------------------------- file downloads (plots, CSVs)
@app.get("/api/files")
def get_file(path: str):
    abspath = os.path.abspath(path)
    if not abspath.startswith(os.path.abspath(config.WORKDIR)):
        raise HTTPException(403, "path is outside the working directory")
    if not os.path.isfile(abspath):
        raise HTTPException(404, "file not found")
    return FileResponse(abspath)
