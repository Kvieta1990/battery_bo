# Battery BO Platform

A frontend + agentic backend for the battery-materials Bayesian-optimization workflow described
in `Agent_design.md`: upload raw synthesis notebooks and battery measurements, let five
cooperating agents turn them into a clean training table, then configure and run a
GP-based (ARD or SAASBO) Bayesian optimizer and review the suggested next experiments with
interactive diagnostics.

## What's here

```
index.html          the entire frontend -- one self-contained file, open it in any browser
backend/             the agent backend (FastAPI + LangChain/Anthropic)
  app/
    agents/          the five Agent_design.md sub-agents (one file each)
      condition_processor.py        Step 1: notebook -> structured synthesis conditions
      parameter_filter.py           Step 2: KEEP/IGNORE diagnosis per parameter
      encoding_determination.py     Step 3: numeric / categorical / ambiguous classification
      cross_comparison.py           Step 4: capacity extraction + merge into training table
      bo_trainer.py                 Step 5: assembles the BO config and runs it
    core/            the numerical core, ported from the BO_template scripts and validated
                      against them (bayes_opt_core.py, parity_calibration_core.py,
                      param_diagnose.py, duplicate_check.py) -- deterministic, no LLM involved
    chem_data.py     molar masses / reagent aliases / stoichiometry constants
    server.py        the FastAPI app (all HTTP endpoints)
    config.py, io_utils.py, jobs.py, llm_client.py, schemas.py   shared plumbing
  requirements.txt
  .env.example       copy to .env and fill in an API key
```

The backend creates its working data folders itself, next to wherever it runs (see
"Where data lives" below) -- there is nothing to scaffold by hand.

## Local Setup

1. **Python 3.11+** is required (3.13 was used to build and test this).
2. From `backend/`:
   ```bash
   pixi install
   cp .env.example .env
   ```
   Edit `.env` and set `ANTHROPIC_API_KEY` (or switch `BO_PLATFORM_LLM_PROVIDER=openai` and set
   `OPENAI_API_KEY`). This key is only used for two things: turning a free-text notebook entry
   into structured fields, and proposing an encoding for a parameter that looks numeric but
   isn't a clean single number (e.g. `pre_hold`). Every numeric computation -- stoichiometry,
   thermal-profile summaries, the GP/SAASBO fit, diagnostics -- is deterministic Python with no
   LLM involved, and is validated against the BO_template outputs (see "Validation" below).
3. Start the server:
   ```bash
   pixi run serve
   ```
4. Open `index.html` in a browser (double-click it, or `open index.html` / drag it into a tab --
   no web server needed for the frontend itself). The header has an **API** field defaulting to
   `http://localhost:8765`; change it if the backend runs elsewhere. The status dot next to
   it turns green once it can reach the server.

If the backend runs on a different machine than the one running `index.html`, point the
frontend at the backend by entering the correct address and port in the API field. The
backend port must be reachable from that machine.

## Using it

The page is a 6-step wizard; steps can be revisited via the top nav once reached.

1. **Synthesis data** -- upload a notebook (`.docx`, `.txt`, or `.md`, one entry per sample,
   any language). Re-uploading the same or an updated file later only costs extraction calls for
   genuinely new entries -- every entry ever submitted (including ones that turned out not
   to be a sample record) is hashed and skipped on repeat runs, so this step is safe to re-run
   as often as needed as the notebook grows.
2. **Battery data** -- upload `battery_data.xlsx` (one sheet per sample). The cycle-1 discharge
   capacity is taken as the *last* non-missing value in each sheet's capacity column (the end of
   the discharge leg), not the column max, which would pick up the larger charge-leg value.
   Sheets whose cycle-1 data ends outside roughly 2.0-3.6 V are flagged for review.
3. **Data check** -- a plain sample-ID match between steps 1 and 2; at least 3 matched
   samples are needed to fit anything. Mismatches are listed to help spot a stray space or a
   different dash character between a notebook ID and a sheet name.
4. **Parameters filter** -- runs the same KEEP/IGNORE diagnosis as `param_diagnose.py`
   (one value dominating &ge;90% of samples &rarr; IGNORE), then classifies every retained
   parameter: numeric columns stay as-is, plain categorical/binary columns are label-encoded
   automatically, and anything that looks numeric but isn't a clean single value (a hold
   schedule like `350C 5h`) is proposed by the LLM and shown for confirmation or edit before
   anything is applied. Building the training table merges this with the extracted capacities.
5. **BO setup & run** -- select which of the resolved parameters go into the model, choose ARD-GP
   (fast, point estimate) or SAASBO (fully Bayesian, better for many candidate parameters, much
   slower), set the acquisition function and batch strategy, then run. The parameter list
   supports the same multi-select conventions as a file manager:
   - click selects one; **Ctrl/Cmd+click** toggles one; **Shift+click** selects the contiguous
     range from the last click, replacing the selection; **Ctrl/Cmd+Shift+click** applies
     whichever action (select or deselect) the last Ctrl/Cmd+click did, across the range to
     the clicked item, without touching anything outside that range.
   - near-duplicate/collinear parameters (e.g. `Na_per_TM` vs `chloride_per_TM`) are flagged
     inline with a suggested one to keep.
6. **Results** -- the fitted model's summary, the suggested next experiments (batch, with
   predicted capacity and acquisition score), and interactive Plotly parity + calibration plots
   (loaded from a CDN, so this step needs internet access).

## Where data lives

The backend works out of a single working directory and creates everything it needs there:

```
synthesis_data/
  condition.docx (uploads land in incoming/)
  condition_proc.xlsx       "used" / "unused" sheets -- the running synthesis-condition table
  seen_entries.json         hashes of every notebook entry ever submitted (idempotency)
  encoding_codebook.json    confirmed encodings, so they are not requested again
  var_filtering/*.csv       one file per candidate parameter, regenerated each run
battery_data/
  battery_data.xlsx (and any further files uploaded)
training_data_clean.xlsx    the merged, encoded table the BO step reads
bo_runs/<run_id>/           suggestions, top candidates, LOO predictions, config, per run
```

By default this is the folder containing `backend/` (i.e. next to `index.html`). Override it by
setting `BO_PLATFORM_WORKDIR` in `.env` to an absolute path to keep it elsewhere.

None of `synthesis_data/`, `battery_data/`, their `incoming/` subfolders, `var_filtering/`, or
`bo_runs/` need to exist before starting the service -- `backend/app/config.py` creates all of
them (`os.makedirs(..., exist_ok=True)`) as soon as the app starts. They then fill up as you
upload notebooks/battery files and run BO. A fresh local checkout with none of these folders
present is normal; no manual setup step is required.
Nothing here is specific to one browser or machine -- pointing another copy of `index.html` at
the same running backend picks up where the previous session left off (step 1 and step 2's
"refresh status" / "extract capacities" buttons re-read the current state from disk).

## Validation

Before building the agents, the ported numerical core (`core/bayes_opt_core.py`,
`core/parity_calibration_core.py`) was run against the actual `BO_template` data and checked
against its own outputs: identical kernel, log-marginal-likelihood to 6 decimal places, all 11
ARD length scales, `best_reference`, and `n_candidates`; LOO MAE differs by ~2e-6 relative, which
is ordinary floating-point noise from `scipy`/`sklearn` patch-version differences across 26
leave-one-out refits, not a logic difference. The Condition Processor's chemistry
(`agents/condition_processor.py`) was separately regression-tested against all 63 of the
notebook's real entries and matches `condition_proc.xlsx` exactly across every derived column.

One intentional quirk carried over from `parse_cond_v2.py`: `nitrate_per_TM` does not
count the Al(NO3)3 dopant's own nitrate (its Al content is still fully captured via
`Al_pct_TM`) -- this was true of the existing training data, so it's kept that way rather than
"fixed", to avoid shifting the meaning of that feature between old and new samples. It's noted
in `chem_data.py` for anyone who wants to change it.

## Troubleshooting

- **Status dot stays red / "cannot reach backend"**: check the API field matches where
  `uvicorn` is actually listening, and that it's still running.
- **A step's job finishes with errors for every entry**: usually `ANTHROPIC_API_KEY` (or
  `OPENAI_API_KEY`) isn't set in `.env`, or the server wasn't restarted after editing `.env`.
- **CORS errors in the browser console**: the backend allows every origin by default
  (`BO_PLATFORM_CORS_ORIGINS=*`); only an issue if it has been deliberately restricted.
- **SAASBO is slow / times out**: it's doing full NUTS sampling (`numpyro`/`jax`); start with
  the defaults (256 warmup/samples, 1 chain) and a modest `n_random` candidate count before
  turning those up.
