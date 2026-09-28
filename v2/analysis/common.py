"""Shared loading and statistics for the v2 analysis (see reports/ANALYSIS_PLAN.md).

Everything is computed from runs/<run>/scores.jsonl (and responses.jsonl for cost, tokens and versions).
Clusters are families; intervals come from family-cluster bootstraps; permutation tests flip signs at the
family level.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))

INCLUDED = {"complete", "truncated", "refusal", "harness_error"}
EXCLUDED = {"infeasible", "api_error"}
KINDS = ("anchor", "mate", "pattern", "orientation", "dimension", "topology")
B = 2000
SEED = 20260928


def run_dirs(names: list[str] | None) -> list[Path]:
    root = V2 / "runs"
    if names:
        return [root / n for n in names]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "scores.jsonl").exists()
                  and not p.name.startswith(("selftest", "smoke")))


def _latest(path: Path, keyf) -> dict:
    out = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            out[keyf(r)] = r
    return out


def _key(r):
    return (r["model"], r["arm"], r.get("split", "main"), r["case_id"], r["sample"], r.get("round", 0))


def load(names: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(scores, responses) as data frames, last record per key, over the given run directories."""
    S, R = {}, {}
    for d in run_dirs(names):
        S.update(_latest(d / "scores.jsonl", _key))
        R.update(_latest(d / "responses.jsonl", _key))
    s = pd.DataFrame(list(S.values()))
    r = pd.DataFrame([{k: v for k, v in x.items() if k != "text"} for x in R.values()])
    for df in (s, r):
        if len(df):
            df["split"] = df.get("split", "main").fillna("main") if "split" in df else "main"
            df["round"] = df["round"].fillna(0).astype(int) if "round" in df else 0
    if len(s):
        s["exact"] = s["exact"].fillna(False).astype(bool) if "exact" in s else False
        s["included"] = s["response_status"].isin(INCLUDED)
    return s, r


def included(s: pd.DataFrame) -> pd.DataFrame:
    return s[s["included"]]


# ---------------------------------------------------------------------------
# family-cluster statistics
# ---------------------------------------------------------------------------
def case_means(df: pd.DataFrame, value: str, by=("model", "arm")) -> pd.DataFrame:
    """Mean of `value` over samples per (by..., family, case)."""
    keys = list(by) + ["family", "case_id"]
    return df.dropna(subset=[value]).groupby(keys, as_index=False)[value].mean()


def _fam_arrays(per_case: pd.DataFrame, value: str, families: list[str]):
    g = per_case.groupby("family")[value]
    sums = g.sum().reindex(families).fillna(0.0).to_numpy()
    cnts = g.count().reindex(families).fillna(0).to_numpy()
    return sums, cnts


def boot_mean(per_case: pd.DataFrame, value: str, b: int = B, seed: int = SEED) -> tuple[float, float, float]:
    """Case-level mean with a family-cluster bootstrap 95% interval."""
    if per_case.empty:
        return float("nan"), float("nan"), float("nan")
    fams = sorted(per_case["family"].unique())
    sums, cnts = _fam_arrays(per_case, value, fams)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(fams), size=(b, len(fams)))
    num, den = sums[idx].sum(1), cnts[idx].sum(1)
    boots = num[den > 0] / den[den > 0]
    if cnts.sum() == 0 or boots.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(sums.sum() / cnts.sum()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def paired(df: pd.DataFrame, arm: str, base: str, value: str, by_model: bool = True) -> pd.DataFrame:
    """Per (model, family, case): mean over samples of arm minus base, for cases present in both."""
    a = case_means(df[df["arm"] == arm], value).rename(columns={value: "a"})
    b = case_means(df[df["arm"] == base], value).rename(columns={value: "b"})
    m = a.drop(columns="arm").merge(b.drop(columns="arm"), on=["model", "family", "case_id"])
    m["d"] = m["a"] - m["b"]
    return m


def boot_paired_pooled(m: pd.DataFrame, b: int = B, seed: int = SEED) -> tuple[float, float, float]:
    """Mean over models of each model's case-mean difference; the same family draw for every model."""
    m = m.dropna(subset=["d"])
    if m.empty:
        return float("nan"), float("nan"), float("nan")
    fams = sorted(m["family"].unique())
    models = sorted(m["model"].unique())
    S = np.zeros((len(models), len(fams)))
    C = np.zeros_like(S)
    for i, mo in enumerate(models):
        s, c = _fam_arrays(m[m["model"] == mo], "d", fams)
        S[i], C[i] = s, c
    point = np.nanmean([S[i].sum() / C[i].sum() for i in range(len(models)) if C[i].sum() > 0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(fams), size=(b, len(fams)))
    num, den = S[:, idx].sum(2), C[:, idx].sum(2)       # (models, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = np.nanmean(num / np.where(den > 0, den, np.nan), axis=0)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0 or not np.isfinite(point):
        return float("nan"), float("nan"), float("nan")
    return float(point), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def signflip_p(m: pd.DataFrame, n: int = 10_000, seed: int = SEED) -> float:
    """Two-sided p-value for mean(d) = 0 with family-level sign flips (d averaged within family first)."""
    m = m.dropna(subset=["d"])
    if m.empty:
        return float("nan")
    fam = m.groupby("family")["d"].mean().to_numpy()
    obs = abs(fam.mean())
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n, len(fam)))
    null = np.abs((flips * fam).mean(1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n + 1))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def fmt_ci(t, pct=False, nd=1) -> str:
    p, lo, hi = t
    f = 100.0 if pct else 1.0
    if not np.isfinite(p):
        return "n/a"
    return f"{p * f:.{nd}f} [{lo * f:.{nd}f}, {hi * f:.{nd}f}]"


def md_table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out


def write(out_dir: Path, name: str, lines: list[str], data: dict | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n")
    if data is not None:
        (out_dir / f"{name}.json").write_text(json.dumps(data, indent=1, default=float))


def macro_name(*parts) -> str:
    """LaTeX macro name from parts: letters kept, digits spelled out, each part capitalised."""
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
             "8": "Eight", "9": "Nine"}
    out = []
    for p in parts:
        t = "".join(ch if ch.isalpha() else words.get(ch, "") for ch in str(p))
        out.append(t[:1].upper() + t[1:])
    return "\\" + "".join(out)
