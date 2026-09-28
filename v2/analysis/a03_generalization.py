"""A03: scale (breaking size on the sweep) and held-out generalization (plan §6)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from .common import SEED, V2, boot_mean, fmt_ci, included, macro_name, md_table, write

LAMBDA = 1.0          # L2 penalty on family offsets
B_SCALE = 200


def _fit(fam_idx, x, y, nf):
    def nll(p):
        a, b, u = p[0], p[1], p[2:]
        z = a + u[fam_idx] + b * x
        return np.sum(np.logaddexp(0, z) - y * z) + LAMBDA * np.sum(u ** 2)
    p0 = np.zeros(2 + nf)
    return minimize(nll, p0, method="L-BFGS-B").x


def n50(params, nf, lo_n: float, hi_n: float) -> float:
    """Part count where the family-averaged P(exact) = 0.5, searched only inside the observed range:
    inf = still >= 50% at the largest observed size; 0 = already < 50% at the smallest (both censored)."""
    a, b, u = params[0], params[1], params[2:2 + nf]

    def f(x):
        return np.mean(1 / (1 + np.exp(-(a + u + b * x)))) - 0.5
    lo, hi = np.log2(lo_n), np.log2(hi_n)
    if f(lo) < 0:
        return 0.0
    if f(hi) > 0:
        return float("inf")
    return float(2 ** brentq(f, lo, hi))


def breaking_size(g: pd.DataFrame, rng) -> dict:
    fams = sorted(g["family"].unique())
    fi = {f: i for i, f in enumerate(fams)}
    x = np.log2(g["n_parts"].to_numpy(float))
    y = g["exact"].astype(float).to_numpy()
    idx = g["family"].map(fi).to_numpy()
    lo_n, hi_n = float(g["n_parts"].min()), float(g["n_parts"].max())
    point = n50(_fit(idx, x, y, len(fams)), len(fams), lo_n, hi_n)
    boots = []
    for _ in range(B_SCALE):
        pick = rng.integers(0, len(fams), len(fams))
        rows = np.concatenate([np.where(idx == p)[0] for p in pick])
        new_idx = np.concatenate([np.full((idx == p).sum(), j) for j, p in enumerate(pick)])
        boots.append(n50(_fit(new_idx, x[rows], y[rows], len(fams)), len(fams), lo_n, hi_n))
    boots = np.array(boots)
    q = lambda p: float(np.quantile(boots, p, method="inverted_cdf"))   # no interpolation across censored values
    return {"n50": point, "lo": q(0.025), "hi": q(0.975), "n": len(g), "min_parts": lo_n, "max_parts": hi_n}


def fmt_n50(v, lo_n, hi_n) -> str:
    if v == 0:
        return f"< {lo_n:.0f}"
    if not np.isfinite(v):
        return f"> {hi_n:.0f}"
    return f"{v:.0f}"


def scale(s: pd.DataFrame) -> tuple[dict, dict, dict]:
    sw = s[(s["split"] == "sweep") & (s["round"] == 0)]
    censored = sw[sw["response_status"].isin(["infeasible", "truncated", "api_error"])]
    use = included(sw)
    use = use[use["response_status"] != "truncated"]
    rng = np.random.default_rng(SEED)
    res, largest, cens = {}, {}, {}
    for (m, arm), g in use.groupby(["model", "arm"]):
        res[(m, arm)] = breaking_size(g, rng)
        largest[(m, arm)] = {f: int(gg[gg["exact"]]["n_parts"].max()) if gg["exact"].any() else 0
                             for f, gg in g.groupby("family")}
    for (m, arm), g in censored.groupby(["model", "arm"]):
        cens[(m, arm)] = g["response_status"].value_counts().to_dict()
    return res, largest, cens


def _cases(split: str) -> dict:
    f = {"main": "cases_v2.jsonl", "heldout": "heldout_v2.jsonl"}[split]
    return {c["case_id"]: c for c in (json.loads(l) for l in open(V2 / "data" / f))}


def memorization(r_dirs: list[Path], s: pd.DataFrame) -> dict:
    """Share of held-out answers closer (Global_v2) to the default main-split reference than to their own."""
    from c2cad.evaluate import evaluate
    from c2cad.runner.run import postprocess
    ho, main = _cases("heldout"), _cases("main")
    by_fl = {(c["family"], c["level"]): c for c in main.values()}
    keep = s[(s["split"] == "heldout") & s["included"]][["model", "arm", "case_id", "sample", "global_v2"]]
    want = {(r.model, r.arm, r.case_id, r.sample): r.global_v2 for r in keep.itertuples()}
    out = {}
    for d in r_dirs:
        p = d / "responses.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            rec = json.loads(line)
            k = (rec["model"], rec["arm"], rec["case_id"], rec["sample"])
            if rec.get("split") != "heldout" or k not in want or not rec.get("ok") or rec.get("round", 0):
                continue
            case = ho[rec["case_id"]]
            post = postprocess(rec["arm"], case, rec["text"])
            v = post["value"] if post["value"] is not None else []
            g_default = evaluate(by_fl[(case["family"], case["level"])], v).global_v2
            out.setdefault((rec["model"], rec["arm"]), []).append(g_default > want[k] + 1e-9)
    return {k: float(np.mean(v)) for k, v in out.items()}


def heldout_gap(s: pd.DataFrame) -> dict:
    ho = included(s[(s["split"] == "heldout") & (s["round"] == 0)]).copy()
    if ho.empty:
        return {}
    fams = set(ho["family"])
    mn = included(s[(s["split"] == "main") & (s["round"] == 0) & (s["family"].isin(fams))]).copy()
    res = {}
    for (m, arm), g in ho.groupby(["model", "arm"]):
        gm = mn[(mn["model"] == m) & (mn["arm"] == arm)]
        if gm.empty:
            continue
        a = g.assign(e=g["exact"].astype(float)).groupby(["family", "level"], as_index=False)["e"].mean()
        b = gm.assign(e=gm["exact"].astype(float)).groupby(["family", "level"], as_index=False)["e"].mean()
        x = a.merge(b, on=["family", "level"], suffixes=("_ho", "_main"))
        x["d"] = x["e_ho"] - x["e_main"]
        x["case_id"] = x["family"] + x["level"].astype(str)
        res[(m, arm)] = {"heldout": boot_mean(x.rename(columns={"e_ho": "v"}), "v"),
                         "main": boot_mean(x.rename(columns={"e_main": "v"}), "v"), "diff": boot_mean(x, "d")}
    return res


def run(s: pd.DataFrame, r: pd.DataFrame, out_dir, r_dirs=None) -> dict:
    L = ["# A03. Scale and generalization", ""]
    data, macros = {}, {}
    if (s["split"] == "sweep").any():
        res, largest, cens = scale(s)
        L += ["## Breaking size (sweep): part count at which the family-averaged P(exact) = 0.5", "",
              "Logistic in log2(parts) with L2-penalised family offsets, searched only within the observed part "
              "counts; 95% family bootstrap CI (200 draws). '> N': still at least 50% at the largest size tried; "
              "'< N': below 50% already at the smallest. Infeasible and truncated responses are censored (counts "
              "at right).", ""]
        L += md_table(["model", "arm", "n50", "95% CI", "n used", "censored"],
                      [[m, a, fmt_n50(v["n50"], v["min_parts"], v["max_parts"]),
                        f"[{fmt_n50(v['lo'], v['min_parts'], v['max_parts'])}, "
                        f"{fmt_n50(v['hi'], v['min_parts'], v['max_parts'])}]", v["n"],
                        json.dumps(cens.get((m, a), {}))] for (m, a), v in sorted(res.items())])
        L += ["", "Largest scale solved exactly, per family:", ""]
        fams = sorted({f for v in largest.values() for f in v})
        L += md_table(["model", "arm"] + fams, [[m, a] + [str(v.get(f, "")) for f in fams]
                                               for (m, a), v in sorted(largest.items())])
        data["breaking_size"] = {f"{m}|{a}": v for (m, a), v in res.items()}
        data["largest_solved"] = {f"{m}|{a}": v for (m, a), v in largest.items()}
        data["censored"] = {f"{m}|{a}": v for (m, a), v in cens.items()}
        for (m, a), v in res.items():
            macros[macro_name("nfifty", a, m)] = fmt_n50(v["n50"], v["min_parts"], v["max_parts"]).replace(">", "$>$").replace("<", "$<$")
    if (s["split"] == "heldout").any():
        gap = heldout_gap(s)
        mem = memorization(r_dirs or [], s) if r_dirs else {}
        L += ["", "## Held-out minus main (same 8 families and levels; exact, pp)", ""]
        L += md_table(["model", "arm", "held-out", "main", "held-out - main", "closer to default answer"],
                      [[m, a, fmt_ci(v["heldout"], True), fmt_ci(v["main"], True), fmt_ci(v["diff"], True),
                        f"{100 * mem[(m, a)]:.1f}%" if (m, a) in mem else "n/a"] for (m, a), v in sorted(gap.items())])
        data["heldout"] = {f"{m}|{a}": v for (m, a), v in gap.items()}
        data["memorization"] = {f"{m}|{a}": v for (m, a), v in mem.items()}
        for (m, a), v in gap.items():
            macros[macro_name("heldoutgap", a, m)] = f"{100 * v['diff'][0]:.1f}"
    write(out_dir, "a03_generalization", L, data)
    return macros
