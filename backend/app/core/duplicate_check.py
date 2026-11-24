"""
Near-duplicate / collinear-parameter detection — ported from the reference
`find_duplicates.py`. Not named as its own sub-agent in Agent_design.md, but the template
workbook (`duplicate_parameters.xlsx`) shows it is how the feature set gets from the
Parameters-Filter KEEP list down to the set actually used for the GP (e.g. the near-identical
Li_salt_per_TM / Li_total_per_TM / Na_per_TM / OH_per_TM / nitrate_per_TM / chloride_per_TM /
salt_solid_mass_ratio group in the template, all derived from the same feed ratios).

Rather than auto-removing anything, this module only computes the similarity groups; the
backend surfaces them as hints next to the user's manual parameter multi-select
(Agent_design.md: "Manual input needed ... on top of the agent decision").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

DEFAULT_THRESHOLD = 0.90


@dataclass
class DuplicateGroup:
    members: list[str]
    suggested_keep: str
    criterion: float            # smallest pairwise similarity inside the group
    why: str
    similarity_to_kept: dict = field(default_factory=dict)


@dataclass
class DuplicateReport:
    groups: list[DuplicateGroup]
    unique: list[str]
    constant: list[str]
    similarity: dict            # {col: {col: float}} full matrix, for tooltips


def similarity_matrix(X: pd.DataFrame) -> pd.DataFrame:
    p = X.corr(method="pearson").abs()
    s = X.corr(method="spearman").abs()
    return np.maximum(p, s)


def find_duplicate_groups(X: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD,
                           prefer: list[str] | None = None) -> DuplicateReport:
    """X: numeric feature matrix (sample rows x parameter columns), NO sample/target columns.
    Mirrors find_duplicates.py's complete-linkage clustering on (1 - similarity)."""
    prefer = prefer or []
    const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    Xv = X.drop(columns=const)
    if Xv.shape[1] == 0:
        return DuplicateReport(groups=[], unique=[], constant=const, similarity={})

    S = similarity_matrix(Xv)
    names = list(Xv.columns)
    D = (1 - S.values).clip(min=0)
    np.fill_diagonal(D, 0)
    if len(names) < 2:
        return DuplicateReport(groups=[], unique=names, constant=const, similarity=S.round(3).to_dict())

    labels = fcluster(linkage(squareform(D, checks=False), method="complete"),
                       t=1 - threshold + 1e-12, criterion="distance")

    groups, unique = [], []
    for lab in np.unique(labels):
        members = [n for n, l in zip(names, labels) if l == lab]
        if len(members) == 1:
            unique.append(members[0])
            continue
        sub = S.loc[members, members]
        mean_sim = {m: sub.loc[m, [o for o in members if o != m]].mean() for m in members}
        pref = [p for p in prefer if p in members]
        if pref:
            keep, why = pref[0], "listed in --prefer"
        else:
            keep = sorted(members, key=lambda m: (-round(mean_sim[m], 6), -Xv[m].nunique(), m))[0]
            why = "highest mean similarity to the rest of the group"
        crit = float(sub.values[np.triu_indices(len(members), 1)].min())
        order = [keep] + sorted([m for m in members if m != keep], key=lambda m: -S.loc[keep, m])
        groups.append(DuplicateGroup(
            members=order, suggested_keep=keep, criterion=crit, why=why,
            similarity_to_kept={m: float(S.loc[keep, m]) for m in order if m != keep},
        ))
    groups.sort(key=lambda g: -g.criterion)
    return DuplicateReport(groups=groups, unique=unique, constant=const, similarity=S.round(3).to_dict())
