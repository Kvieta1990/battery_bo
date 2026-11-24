"""
BO Model Trainer agent's deterministic core. Ported from the reference `bayes_opt_saas.py`
(a superset of `bayes_opt.py`: model.inference "map" reproduces bayes_opt.py exactly; "saas"
adds the fully-Bayesian Sparse Axis-Aligned Subspace GP). Kept numerically identical to the
template scripts — same kernels, same acquisition math, same leave-one-out diagnostics — so
that re-running the template's own bo_config.json through `run_bo()` reproduces the template's
own model_summary numbers (this is validated in backend/validate_against_template.py).

The one difference from the scripts: `run_bo()` is a function that takes a config dict and a
base directory and returns the results as Python objects (DataFrames / dicts) in addition to
writing the same CSV/JSON files the scripts wrote, so the API layer can serialize them to JSON
for the frontend without re-reading files off disk.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel


class BOConfigError(ValueError):
    """Raised for problems in the user-supplied BO configuration (bad section, bad column, etc.)."""


_JAX64_READY = False


def _ensure_jax64():
    global _JAX64_READY
    if not _JAX64_READY:
        import jax
        jax.config.update("jax_enable_x64", True)
        _JAX64_READY = True


# ----------------------------------------------------------------------------- config
def active(d: dict) -> dict:
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def normalize_config(cfg: dict) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy, drop any non-JSON-safe values early
    for key in ("data", "model", "acquisition", "search_space"):
        if key not in cfg:
            raise BOConfigError(f"config: missing section '{key}'")
    m = cfg["acquisition"].get("method", "EI").upper()
    if m not in ("EI", "PI", "UCB"):
        raise BOConfigError("config: acquisition.method must be EI, PI or UCB")
    cfg["acquisition"]["method"] = m
    inf = str(cfg["model"].get("inference", "map")).lower()
    if inf not in ("map", "saas", "saas_nuts", "saasbo"):
        raise BOConfigError("config: model.inference must be 'map' or 'saas'")
    cfg["model"]["inference"] = inf
    return cfg


def rel(base_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(base_dir, p)


# ----------------------------------------------------------------------------- data
def load_data(cfg: dict, base_dir: str):
    d = cfg["data"]
    path = rel(base_dir, d["file"])
    if not os.path.exists(path):
        raise BOConfigError(f"training data file not found: {path}")
    df = pd.read_excel(path, sheet_name=d.get("sheet", 0)) if path.lower().endswith((".xlsx", ".xls")) \
        else pd.read_csv(path)
    scol, tcol = d.get("sample_column", "sample"), d["target_column"]
    if tcol not in df.columns:
        raise BOConfigError(f"target column '{tcol}' not in the training data")
    df = df[df[tcol].apply(lambda v: isinstance(v, (int, float)) and np.isfinite(v))].copy()
    df = df[~df[scol].isin(d.get("exclude_samples", []))]
    feats = d.get("features") or [c for c in df.columns if c not in (scol, tcol)]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise BOConfigError(f"features not in the data: {missing}")
    if len(df) < 3:
        raise BOConfigError(f"only {len(df)} training samples with a numeric '{tcol}' value; need at least 3")
    X = df[feats].astype(float)
    if X.isna().any().any():
        raise BOConfigError(f"missing values in features: {X.columns[X.isna().any()].tolist()}")
    return df.reset_index(drop=True), feats, X.values, df[tcol].astype(float).values, scol, tcol


# ----------------------------------------------------------------------------- target transform
class Transform:
    def __init__(self, mcfg):
        t = mcfg.get("target_transform", {"type": "none"})
        self.kind = t.get("type", "none")
        self.qmax = float(t.get("upper_bound", 230.0))

    def fwd(self, q):
        if self.kind == "logit":
            if np.any(q >= self.qmax) or np.any(q <= 0):
                raise BOConfigError("logit transform: every target value must be in (0, upper_bound)")
            return np.log(q / (self.qmax - q))
        return q

    def inv(self, z):
        return self.qmax / (1.0 + np.exp(-z)) if self.kind == "logit" else z


def _mc_predict(tr, mz, sz, n_samples, rng, seed):
    if tr.kind == "none":
        return mz, sz
    rng = rng or np.random.default_rng(seed)
    q = tr.inv(mz[:, None] + sz[:, None] * rng.standard_normal((len(mz), n_samples)))
    return q.mean(1), q.std(1)


# ----------------------------------------------------------------------------- model (MAP)
class Model:
    """GP on standardized features and a standardized (optionally logit) target, hyperparameters
    point-estimated with a single ARD Matern or RBF kernel (maximum marginal likelihood)."""

    def __init__(self, mcfg, n_features):
        self.cfg, self.tr = mcfg, Transform(mcfg)
        self.n_features = n_features
        k = mcfg.get("kernel", {})
        ls_lo, ls_hi = k.get("length_scale_bounds", [1e-2, 1e2])
        ls0 = float(k.get("length_scale_init", max(1.0, ls_lo)))
        ls0 = min(max(ls0, ls_lo), ls_hi)
        ls = np.full(n_features, ls0) if k.get("ard", True) else ls0
        if k.get("type", "matern").lower() == "rbf":
            base = RBF(ls, (ls_lo, ls_hi))
        else:
            base = Matern(ls, (ls_lo, ls_hi), nu=float(k.get("nu", 2.5)))
        amp = ConstantKernel(k.get("amplitude_init", 1.0), tuple(k.get("amplitude_bounds", [1e-2, 1e2])))
        noise = WhiteKernel(k.get("noise_init", 0.1), tuple(k.get("noise_bounds", [1e-3, 1.0])))
        self.kernel = amp * base + noise
        self.restarts = int(mcfg.get("n_restarts", 5))
        self.seed = int(mcfg.get("random_state", 0))

    def fit(self, X, y, optimize=True, kernel=None):
        self.xm, self.xs = X.mean(0), X.std(0)
        self.xs[self.xs == 0] = 1.0
        z = self.tr.fwd(y)
        self.zm, self.zs = z.mean(), z.std() if z.std() > 0 else 1.0
        gp = GaussianProcessRegressor(kernel=kernel or self.kernel, normalize_y=False,
                                       n_restarts_optimizer=self.restarts if optimize else 0,
                                       optimizer="fmin_l_bfgs_b" if optimize else None,
                                       random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit((X - self.xm) / self.xs, (z - self.zm) / self.zs)
        self.gp = gp
        return self

    def latent(self, X):
        mu, sd = self.gp.predict((X - self.xm) / self.xs, return_std=True)
        return mu * self.zs + self.zm, np.maximum(sd * self.zs, 1e-12)

    def predict(self, X, n_samples=400, rng=None):
        mz, sz = self.latent(X)
        return _mc_predict(self.tr, mz, sz, n_samples, rng, self.seed)

    def lengthscales(self):
        ls = np.atleast_1d(self.gp.kernel_.k1.k2.length_scale)
        return ls if len(ls) == self.n_features else np.repeat(ls, self.n_features)

    def diagnostics(self):
        return {"inference": "map", "kernel": str(self.gp.kernel_),
                "log_marginal_likelihood": float(self.gp.log_marginal_likelihood_value_)}


# ----------------------------------------------------------------------------- model (SAAS / NUTS)
def _saas_nuts_model(X, y, alpha):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    n, d = X.shape
    mean = numpyro.sample("mean", dist.Normal(0.0, 1.0))
    outputscale = numpyro.sample("outputscale", dist.Gamma(2.0, 0.15))
    tausq = numpyro.sample("kernel_tausq", dist.HalfCauchy(jnp.array(alpha)))
    with numpyro.plate("dims", d):
        raw = numpyro.sample("_kernel_inv_length_sq", dist.HalfCauchy(jnp.ones(d)))
    inv_length_sq = numpyro.deterministic("kernel_inv_length_sq", tausq * raw)
    noise = numpyro.sample("noise", dist.Gamma(0.9, 10.0))
    diff2 = jnp.sum(inv_length_sq * (X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    r = jnp.sqrt(jnp.maximum(diff2, 1e-30))
    sqrt5_r = jnp.sqrt(5.0) * r
    k = outputscale * (1 + sqrt5_r + 5.0 / 3.0 * r ** 2) * jnp.exp(-sqrt5_r)
    cov = k + (noise + 1e-6) * jnp.eye(n)
    numpyro.sample("y", dist.MultivariateNormal(mean * jnp.ones(n), covariance_matrix=cov), obs=y)


def _run_saas_nuts(Xtr, ztr, alpha, num_warmup, num_samples, num_chains, seed):
    _ensure_jax64()
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    kernel = NUTS(_saas_nuts_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(Xtr), jnp.asarray(ztr), float(alpha))
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


def _gp_posterior_batch(post, Xtr, ztr, Xs, jitter):
    _ensure_jax64()
    import jax
    import jax.numpy as jnp
    mean_j, os_j = jnp.asarray(post["mean"]), jnp.asarray(post["outputscale"])
    noise_j = jnp.asarray(post["noise"])
    ils_j = jnp.asarray(post["kernel_inv_length_sq"])
    Xtr_j, ztr_j, Xs_j = jnp.asarray(Xtr), jnp.asarray(ztr), jnp.asarray(Xs)
    n = Xtr_j.shape[0]
    sqrt5 = jnp.sqrt(5.0)

    def matern52(diff2, os_l):
        r = jnp.sqrt(jnp.maximum(diff2, 1e-30))
        sr = sqrt5 * r
        return os_l * (1 + sr + 5.0 / 3.0 * r ** 2) * jnp.exp(-sr)

    def one(mean_l, os_l, noise_l, ils_l):
        d2_ff = jnp.sum(ils_l * (Xtr_j[:, None, :] - Xtr_j[None, :, :]) ** 2, axis=-1)
        Kff = matern52(d2_ff, os_l) + (noise_l + jitter) * jnp.eye(n)
        d2_fs = jnp.sum(ils_l * (Xtr_j[:, None, :] - Xs_j[None, :, :]) ** 2, axis=-1)
        Kfs = matern52(d2_fs, os_l)
        Lc = jax.scipy.linalg.cholesky(Kff, lower=True)
        a = jax.scipy.linalg.cho_solve((Lc, True), ztr_j - mean_l)
        mu = mean_l + Kfs.T @ a
        v = jax.scipy.linalg.solve_triangular(Lc, Kfs, lower=True)
        var_f = jnp.clip(os_l - jnp.sum(v ** 2, axis=0), 1e-10, None)
        return mu, var_f + noise_l

    mu_all, var_all = jax.vmap(one)(mean_j, os_j, noise_j, ils_j)
    return np.asarray(mu_all), np.asarray(var_all)


class SAASModel:
    """Fully-Bayesian SAAS GP (Eriksson & Jankowiak, UAI 2021) fit via NUTS. Same external
    interface as Model so it drops into acquisition()/loo()/the batch loop unchanged."""

    def __init__(self, mcfg, n_features):
        self.cfg, self.tr = mcfg, Transform(mcfg)
        s = mcfg.get("saas", {})
        self.alpha = float(s.get("alpha", 0.1))
        self.num_warmup = int(s.get("num_warmup", 256))
        self.num_samples = int(s.get("num_samples", 256))
        self.num_chains = int(s.get("num_chains", 1))
        self.jitter = float(s.get("jitter", 1e-6))
        self.chunk = int(s.get("predict_chunk", 4000))
        self.seed = int(mcfg.get("random_state", 0))
        self.n_features = n_features
        self.posterior = None

    def fit(self, X, y, optimize=True, posterior=None):
        self.xm, self.xs = X.mean(0), X.std(0)
        self.xs[self.xs == 0] = 1.0
        z = self.tr.fwd(y)
        self.zm, self.zs = z.mean(), (z.std() if z.std() > 0 else 1.0)
        self.Xtr = (X - self.xm) / self.xs
        self.ztr = (z - self.zm) / self.zs
        if optimize or posterior is None:
            self.posterior = _run_saas_nuts(self.Xtr, self.ztr, self.alpha, self.num_warmup,
                                             self.num_samples, self.num_chains, self.seed)
        else:
            self.posterior = posterior
        return self

    def latent(self, X):
        Xs = (X - self.xm) / self.xs
        m = len(Xs)
        mz, sz = np.empty(m), np.empty(m)
        for i in range(0, m, self.chunk):
            sl = slice(i, i + self.chunk)
            mu_l, var_l = _gp_posterior_batch(self.posterior, self.Xtr, self.ztr, Xs[sl], self.jitter)
            mz[sl] = mu_l.mean(axis=0)
            sz[sl] = np.sqrt(np.maximum(var_l.mean(axis=0) + mu_l.var(axis=0), 1e-12))
        return mz * self.zs + self.zm, np.maximum(sz * self.zs, 1e-12)

    def predict(self, X, n_samples=400, rng=None):
        mz, sz = self.latent(X)
        return _mc_predict(self.tr, mz, sz, n_samples, rng, self.seed)

    def lengthscales(self):
        ils_med = np.median(self.posterior["kernel_inv_length_sq"], axis=0)
        return 1.0 / np.sqrt(np.maximum(ils_med, 1e-12))

    def diagnostics(self):
        p = self.posterior
        med = {k: np.median(v, axis=0) for k, v in p.items() if not k.startswith("_")}
        desc = (f"SAAS-GP (Matern5/2, NUTS, alpha={self.alpha}, L={len(p['mean'])} draws): "
                f"outputscale_med={float(med['outputscale']):.3g}, "
                f"noise_med={float(med['noise']):.3g}, tausq_med={float(med['kernel_tausq']):.3g}")
        return {"inference": "saas_nuts", "kernel": desc, "log_marginal_likelihood": None,
                "posterior_median": {k: (float(v) if np.ndim(v) == 0 else [float(x) for x in v])
                                      for k, v in med.items()}}


def make_model(mcfg, n_features):
    if mcfg.get("inference", "map") in ("saas", "saas_nuts", "saasbo"):
        return SAASModel(mcfg, n_features)
    return Model(mcfg, n_features)


def refit_fixed(mcfg, n_features, Xb, yb, base_model):
    if isinstance(base_model, SAASModel):
        return SAASModel(mcfg, n_features).fit(Xb, yb, optimize=False, posterior=base_model.posterior)
    return Model(mcfg, n_features).fit(Xb, yb, optimize=False, kernel=base_model.gp.kernel_)


# ----------------------------------------------------------------------------- acquisition
def acquisition(model, X, method, acfg, best, rng):
    mz, sz = model.latent(X)
    tr = model.tr
    if method == "UCB":
        kappa = float(acfg.get("kappa", 2.0))
        return tr.inv(mz + kappa * sz)
    xi = float(acfg.get("xi", 0.0))
    thr = best + xi
    if method == "PI":
        if tr.kind == "logit":
            if thr >= tr.qmax:
                return np.zeros(len(mz))
            thr = tr.fwd(np.array([thr]))[0]
        return norm.sf((thr - mz) / sz)
    if tr.kind == "none":
        u = (mz - thr) / sz
        return (mz - thr) * norm.cdf(u) + sz * norm.pdf(u)
    n = int(acfg.get("ei_mc_samples", 2000))
    out = np.empty(len(mz))
    for i in range(0, len(mz), 2000):
        s = slice(i, i + 2000)
        q = tr.inv(mz[s, None] + sz[s, None] * rng.standard_normal((len(mz[s]), n)))
        out[s] = np.maximum(q - thr, 0).mean(1)
    return out


# ----------------------------------------------------------------------------- search space
def axis_values(name, spec, train=None):
    if not isinstance(spec, dict):
        return [spec]
    if "fixed" in spec:
        return [spec["fixed"]]
    if "values" in spec:
        return list(spec["values"])
    if "range" in spec:
        lo, hi, step = spec["range"]
        n = int(round((hi - lo) / step)) + 1
        return list(np.round(lo + step * np.arange(n), 10))
    if "from_data" in spec:
        if train is None:
            raise BOConfigError(f"search_space.variables.{name}: from_data needs the training data")
        t = train.query(spec["filter"], engine="python") if spec.get("filter") else train
        col = spec.get("column", name)
        if col not in t.columns:
            raise BOConfigError(f"search_space.variables.{name}: column '{col}' not in the training data")
        v = t[col].astype(float).dropna()
        if spec["from_data"] == "values":
            return sorted(v.unique().tolist())
        if spec["from_data"] == "range":
            step = float(spec.get("step", 1))
            lo, hi = v.min(), v.max()
            vals = list(np.round(np.arange(lo, hi + step / 2, step), 10))
            return vals if vals[-1] >= hi else vals + [hi]
        raise BOConfigError(f"search_space.variables.{name}: from_data must be 'values' or 'range'")
    raise BOConfigError(f"search_space.variables.{name}: give 'fixed', 'values', 'range' or 'from_data'")


def linked_blocks(sp, train):
    blocks = []
    for g in sp.get("linked_groups", []):
        g = active(g)
        t = train.query(g["filter"], engine="python") if g.get("filter") else train
        miss = [c for c in g["columns"] if c not in t.columns]
        if miss:
            raise BOConfigError(f"search_space.linked_groups: columns not in the training data: {miss}")
        label = g.get("label", "composition_from")
        sc = sp.get("_sample_column", "sample")
        b = t[g["columns"]].astype(float).copy()
        b[label] = t[sc].astype(str) if sc in t.columns else ""
        b = b.groupby(g["columns"], sort=False)[label].agg(" / ".join).reset_index()
        blocks.append(b)
    return blocks


def build_candidates(cfg, feats, train):
    sp = cfg["search_space"]
    var = active(sp.get("variables", {}))
    names = list(var)
    axes = [axis_values(n, var[n], train) for n in names]
    sp["_sample_column"] = cfg["data"].get("sample_column", "sample")
    blocks = linked_blocks(sp, train)
    size = int(np.prod([len(a) for a in axes] + [len(b) for b in blocks])) if (axes or blocks) else 0
    mode = sp.get("mode", "grid")
    rng = np.random.default_rng(sp.get("random_state", 0))
    if mode == "grid":
        limit = int(sp.get("max_grid_size", 2_000_000))
        if size > limit:
            raise BOConfigError(f"grid has {size:,} points > max_grid_size {limit:,}; use mode 'random' or coarsen it")
        C = pd.DataFrame(list(itertools.product(*axes)), columns=names) if names else pd.DataFrame(index=[0])
        for b in blocks:
            C = C.merge(b, how="cross")
    elif mode == "random":
        n = int(sp.get("n_random", 20000))
        C = pd.DataFrame({nm: rng.choice(np.array(a, dtype=float), n) for nm, a in zip(names, axes)}) if names \
            else pd.DataFrame(index=range(n))
        C = C.astype(float) if names else C
        for b in blocks:
            C = pd.concat([C, b.iloc[rng.integers(0, len(b), n)].reset_index(drop=True)], axis=1)
        C = C.drop_duplicates().reset_index(drop=True)
        size = len(C)
    else:
        raise BOConfigError("search_space.mode must be 'grid' or 'random'")
    dup = [c for c in C.columns if list(C.columns).count(c) > 1]
    if dup:
        raise BOConfigError(f"search_space: columns set twice (variables and linked_groups): {sorted(set(dup))}")
    safe = {"np": np, "__builtins__": {}}
    for name, expr in active(sp.get("derived", {})).items():
        try:
            C[name] = eval(expr, safe, {c: C[c].values.astype(float) for c in C.columns
                                         if pd.api.types.is_numeric_dtype(C[c])})
        except Exception as e:  # noqa: BLE001
            raise BOConfigError(f"search_space.derived.{name}: cannot evaluate '{expr}': {e}")
        C[name] = np.broadcast_to(np.asarray(C[name], dtype=float), len(C)).copy()
    for q in sp.get("constraints", []):
        C = C.query(q, engine="python")
    C = C.reset_index(drop=True)
    missing = [f for f in feats if f not in C.columns]
    if missing:
        raise BOConfigError("search_space must set every model feature (as a variable or derived); missing: "
                             + ", ".join(missing))
    if len(C) == 0:
        raise BOConfigError("no candidates left after the constraints")
    return C, size


# ----------------------------------------------------------------------------- diagnostics
def loo(cfg, df, X, y, scol, feats):
    rx = cfg.get("diagnostics", {}).get("group_regex")
    groups = df[scol].astype(str).apply(lambda s: re.match(rx, s).group(1) if rx and re.match(rx, s) else s)
    n = len(y)
    mean, std, mz, sz = (np.empty(n) for _ in range(4))
    tr = None
    for g in groups.unique():
        te = (groups == g).values
        if te.sum() == n:
            raise BOConfigError("diagnostics.group_regex groups every sample together; cannot leave one out")
        m = make_model(cfg["model"], len(feats)).fit(X[~te], y[~te])
        mean[te], std[te] = m.predict(X[te])
        mz[te], sz[te] = m.latent(X[te])
        tr = m.tr
    base = np.array([y[(groups != groups[i]).values].mean() for i in range(n)])
    out = pd.DataFrame({
        scol: df[scol], "group": groups, "observed": y,
        "loo_pred": mean, "loo_std": std, "loo_median": tr.inv(mz),
        "lo68": tr.inv(mz - sz), "hi68": tr.inv(mz + sz),
        "lo95": tr.inv(mz - 2 * sz), "hi95": tr.inv(mz + 2 * sz),
        "z_latent": (tr.fwd(y) - mz) / sz,
        "baseline_pred": base, "abs_err": np.abs(mean - y), "baseline_abs_err": np.abs(base - y)})
    return out, float(out["abs_err"].mean()), float(out["baseline_abs_err"].mean())


# ----------------------------------------------------------------------------- orchestration
@dataclass
class BOResult:
    suggestions: pd.DataFrame
    candidates_top: pd.DataFrame
    loo_predictions: Optional[pd.DataFrame]
    summary: dict
    out_dir: str
    tag: str
    resolved_config: dict = field(default_factory=dict)


def run_bo(config: dict, base_dir: str) -> BOResult:
    cfg = normalize_config(config)
    acfg, bcfg, ocfg = cfg["acquisition"], cfg.get("batch", {}), cfg.get("output", {})
    rng = np.random.default_rng(cfg["model"].get("random_state", 0))

    df, feats, X, y, scol, tcol = load_data(cfg, base_dir)
    model = make_model(cfg["model"], len(feats)).fit(X, y)

    bref = acfg.get("best_reference", "max_observed")
    if isinstance(bref, (int, float)):
        best = float(bref)
    else:
        sub = df.query(acfg["best_filter"], engine="python") if acfg.get("best_filter") else df
        if len(sub) == 0:
            raise BOConfigError("acquisition.best_filter leaves no samples to take the best reference from")
        best = float(sub[tcol].max())

    C, raw_size = build_candidates(cfg, feats, df)
    Xc = C[feats].astype(float).values
    method = acfg["method"]

    n_sug = int(bcfg.get("n_suggestions", 5))
    strategy = bcfg.get("strategy", "kriging_believer")
    chosen, acq_at_pick = [], []
    acq_all = acquisition(model, Xc, method, acfg, best, rng)
    if strategy == "top_k":
        order = np.argsort(-acq_all)
        chosen = list(order[:n_sug])
        acq_at_pick = list(acq_all[chosen])
    elif strategy == "kriging_believer":
        Xb, yb, m, acq = X.copy(), y.copy(), model, acq_all
        for _ in range(min(n_sug, len(C))):
            a = acq.copy()
            a[chosen] = -np.inf
            i = int(np.argmax(a))
            chosen.append(i)
            acq_at_pick.append(acq[i])
            Xb = np.vstack([Xb, Xc[i]])
            yb = np.append(yb, m.predict(Xc[i:i + 1])[0])
            m = refit_fixed(cfg["model"], len(feats), Xb, yb, m)
            acq = acquisition(m, Xc, method, acfg, best, rng)
    else:
        raise BOConfigError("batch.strategy must be 'kriging_believer' or 'top_k'")

    mu, sd = model.predict(Xc)
    C["pred_mean"], C["pred_std"], C["acquisition"] = mu, sd, acq_all
    S = C.iloc[chosen].copy()
    S.insert(0, "rank", range(1, len(S) + 1))
    S["acquisition_at_pick"] = acq_at_pick

    diag = model.diagnostics()
    ls = model.lengthscales()
    loo_out, df_loo = None, None
    if cfg.get("diagnostics", {}).get("run_loo", True):
        df_loo, mae, mae0 = loo(cfg, df, X, y, scol, feats)
        loo_out = {"mae": mae, "mae_mean_baseline": mae0}

    out_dir = rel(base_dir, ocfg.get("dir", "bo_output"))
    os.makedirs(out_dir, exist_ok=True)
    tag = method + (f"_xi{acfg.get('xi', 0)}" if method != "UCB" else f"_kappa{acfg.get('kappa', 2.0)}")
    S.to_csv(os.path.join(out_dir, f"suggestions_{tag}.csv"), index=False)
    candidates_top = C.sort_values("acquisition", ascending=False).head(int(ocfg.get("save_top_candidates", 200)))
    candidates_top.to_csv(os.path.join(out_dir, f"candidates_top_{tag}.csv"), index=False)
    if df_loo is not None:
        df_loo.to_csv(os.path.join(out_dir, "loo_predictions.csv"), index=False)

    summary: dict[str, Any] = {
        "n_train": len(y), "features": feats, "kernel": diag["kernel"], "inference": diag["inference"],
        "log_marginal_likelihood": diag["log_marginal_likelihood"], "best_reference": best,
        "acquisition": acfg, "batch_strategy": strategy, "n_candidates": len(C),
        "length_scales": dict(zip(feats, map(float, ls))), "loo": loo_out,
    }
    if "posterior_median" in diag:
        summary["saas_posterior_median"] = diag["posterior_median"]
    with open(os.path.join(out_dir, f"model_summary_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    with open(os.path.join(out_dir, "resolved_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)

    return BOResult(suggestions=S, candidates_top=candidates_top, loo_predictions=df_loo,
                     summary=summary, out_dir=out_dir, tag=tag, resolved_config=cfg)
