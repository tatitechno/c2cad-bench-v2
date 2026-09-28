"""Statistics that respect the benchmark's structure: the 3 levels of a family are not
independent, so resampling is done over families (cluster bootstrap)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, kendalltau  # noqa: F401  (re-exported for experiments)


def cluster_bootstrap_mean(values: dict, n_boot: int = 10_000, seed: int = 0, ci: float = 0.95):
    """values: {family: [scores of that family's cases]} -> (mean, lo, hi) of the case-level mean."""
    fams = list(values)
    arr = [np.asarray(values[f], float) for f in fams]
    point = float(np.mean(np.concatenate(arr)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(fams), size=(n_boot, len(fams)))
    sums = np.array([a.sum() for a in arr]); cnts = np.array([len(a) for a in arr])
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    a = (1 - ci) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def cluster_bootstrap_ranks(table: dict, n_boot: int = 2_000, seed: int = 0):
    """table: {model: {family: [scores]}} -> {model: (mean rank, rank lo, rank hi, P(top-3))}."""
    models = list(table)
    fams = sorted({f for m in models for f in table[m]})
    rng = np.random.default_rng(seed)
    ranks = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.choice(fams, size=len(fams), replace=True)
        means = {m: np.mean(np.concatenate([table[m][f] for f in pick])) for m in models}
        order = sorted(models, key=lambda m: -means[m])
        for r, m in enumerate(order, 1):
            ranks[m].append(r)
    out = {}
    for m in models:
        r = np.array(ranks[m])
        out[m] = (float(r.mean()), int(np.quantile(r, 0.025)), int(np.quantile(r, 0.975)), float((r <= 3).mean()))
    return out


def by_family(rows, model_key="model", fam_key="family", val_key="value"):
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[r[model_key]][r[fam_key]].append(r[val_key])
    return d
