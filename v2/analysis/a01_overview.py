"""A01: overview per model × arm (plan §2, §3, §5): outcomes, statuses, cost, tiers, variance decomposition."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .common import (boot_mean, case_means, fmt_ci, holm, included, macro_name, md_table, signflip_p, write)


def outcomes(s: pd.DataFrame) -> list[dict]:
    rows = []
    main = s[(s["split"] == "main") & (s["round"] == 0)]
    for (m, arm), g in main.groupby(["model", "arm"]):
        inc = included(g)
        d = {"model": m, "arm": arm, "responses": len(g),
             **{f"n_{st}": int((g["response_status"] == st).sum())
                for st in ("complete", "truncated", "refusal", "infeasible", "api_error", "harness_error")},
             "parse_fail": float((inc["parse_status"] == "fail").mean()) if len(inc) else float("nan"),
             "program_error": float(inc["exec_status"].isin(["error", "invalid_program"]).mean()) if len(inc) else float("nan"),
             "k": int(g["sample"].nunique())}
        if arm == "probe":
            d["probe_part_exact"] = boot_mean(case_means(inc, "probe_exact"), "probe_exact")
            rows.append(d)
            continue
        first = inc[inc["sample"] == 0]
        d["exact_first"] = boot_mean(case_means(first.assign(e=first["exact"].astype(float)), "e"), "e")
        d["exact_mean_k"] = boot_mean(case_means(inc.assign(e=inc["exact"].astype(float)), "e"), "e")
        pk = inc.groupby(["family", "case_id"], as_index=False)["exact"].max()
        d["pass_at_k"] = boot_mean(pk.assign(e=pk["exact"].astype(float)), "e")
        for ax in ("global_v2", "coverage", "geometry", "geometry_equiv", "semantic"):
            d[ax] = boot_mean(case_means(inc, ax), ax)
        best = inc.groupby(["family", "case_id"], as_index=False)["global_v2"].max()
        d["global_best_of_k"] = boot_mean(best, "global_v2")
        rows.append(d)
    return rows


def costs(r: pd.DataFrame) -> list[dict]:
    rows = []
    if r.empty:
        return rows
    for (m, arm), g in r.groupby(["model", "arm"]):
        ok = g[g["ok"]]
        rows.append({"model": m, "arm": arm, "requests": len(g), "usd": float(g["cost_usd"].fillna(0).sum()),
                     "mean_latency_s": float(ok["latency_s"].mean()) if len(ok) else float("nan"),
                     "mean_attempts": float(g["attempts"].mean()) if "attempts" in g else float("nan"),
                     "returned_models": sorted({x for x in ok["returned_model"].dropna().unique() if x}),
                     "first_utc": str(g["time_utc"].min()), "last_utc": str(g["time_utc"].max())})
    return rows


def tiers(s: pd.DataFrame, value: str = "exact_f") -> dict:
    """Plan §5: pairwise family-level sign-flip tests on the json arm, Holm-corrected; greedy tiers."""
    j = included(s[(s["split"] == "main") & (s["arm"] == "json") & (s["round"] == 0)])
    j = j.assign(exact_f=j["exact"].astype(float))
    pc = case_means(j, value)
    models = sorted(pc["model"].unique())
    means = {m: pc[pc["model"] == m][value].mean() for m in models}
    pvals, diffs = {}, {}
    for a, b in itertools.combinations(models, 2):
        x = pc[pc["model"] == a][["family", "case_id", value]].merge(pc[pc["model"] == b][["family", "case_id", value]],
                                                                    on=["family", "case_id"])
        x["d"] = x[f"{value}_x"] - x[f"{value}_y"]
        pvals[(a, b)] = signflip_p(x)
        diffs[(a, b)] = float(x["d"].mean())
    adj = holm(pvals) if pvals else {}
    order = sorted(models, key=lambda m: -means[m])
    tiers_, cur = [], []
    for m in order:
        if cur:
            top = cur[0]
            p = adj.get((top, m), adj.get((m, top), 1.0))
            if p < 0.05:
                tiers_.append(cur)
                cur = []
        cur.append(m)
    if cur:
        tiers_.append(cur)
    sig = sum(p < 0.05 for p in adj.values())
    return {"value": value, "means": means, "tiers": tiers_, "n_pairs": len(adj), "n_separated": int(sig),
            "pairs": {f"{a} vs {b}": {"diff": diffs[(a, b)], "p": pvals[(a, b)], "p_holm": adj[(a, b)]}
                      for (a, b) in pvals}}


def variance_decomposition(s: pd.DataFrame) -> dict:
    j = included(s[(s["split"] == "main") & (s["arm"] == "json") & (s["round"] == 0)])
    pc = case_means(j, "global_v2")
    if pc.empty or pc["model"].nunique() < 2:
        return {}
    y = pc["global_v2"].to_numpy()
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    mm = pc.groupby("model")["global_v2"].transform("mean").to_numpy()
    fm = pc.groupby("family")["global_v2"].transform("mean").to_numpy()
    cell = pc.groupby(["model", "family"])["global_v2"].transform("mean").to_numpy()
    ss_model = ((mm - grand) ** 2).sum()
    ss_fam = ((fm - grand) ** 2).sum()
    ss_int = ((cell - mm - fm + grand) ** 2).sum()
    ss_res = ((y - cell) ** 2).sum()
    return {k: float(v / ss_tot) for k, v in (("model", ss_model), ("family", ss_fam),
                                               ("model_x_family", ss_int), ("residual", ss_res))}


def run(s: pd.DataFrame, r: pd.DataFrame, out_dir) -> dict:
    rows = outcomes(s)
    cost = costs(r)
    t_exact = tiers(s, "exact_f")
    t_global = tiers(s.assign(), "global_v2") if len(s) else {}
    vd = variance_decomposition(s)
    L = ["# A01. Overview (main split, first round)", "", "Values: mean [95% family-cluster bootstrap CI]. "
         "Exact rates in %. Infeasible and api_error responses are excluded (counted).", ""]
    hdr = ["model", "arm", "k", "exact first", "exact mean-k", "pass@k", "Global", "Cov", "Geom", "GeomEq", "Sem",
           "Global best-of-k", "truncated", "infeasible", "api err", "parse fail %", "program err %"]
    body = []
    for d in rows:
        if d["arm"] == "probe":
            continue
        body.append([d["model"], d["arm"], d["k"], fmt_ci(d["exact_first"], True), fmt_ci(d["exact_mean_k"], True),
                     fmt_ci(d["pass_at_k"], True), fmt_ci(d["global_v2"]), fmt_ci(d["coverage"]), fmt_ci(d["geometry"]),
                     fmt_ci(d["geometry_equiv"]), fmt_ci(d["semantic"]), fmt_ci(d["global_best_of_k"]),
                     d["n_truncated"], d["n_infeasible"], d["n_api_error"], f"{100 * d['parse_fail']:.1f}",
                     f"{100 * d['program_error']:.1f}"])
    L += md_table(hdr, body)
    L += ["", "## Tiers (json arm; family-level sign-flip tests, Holm-corrected)", ""]
    for t in (t_exact, t_global):
        if t:
            L.append(f"- by {t['value']}: " + " > ".join("{" + ", ".join(x) + "}" for x in t["tiers"])
                     + f" ({t['n_separated']} of {t['n_pairs']} pairs separated at Holm p < 0.05)")
    if vd:
        L += ["", "## Variance of case-level Global_v2 (json arm)", "",
              ", ".join(f"{k} {100 * v:.1f}%" for k, v in vd.items())]
    L += ["", "## Cost, latency and returned model versions", ""]
    L += md_table(["model", "arm", "requests", "USD", "latency s", "attempts", "returned models", "first", "last"],
                  [[c["model"], c["arm"], c["requests"], f"{c['usd']:.2f}", f"{c['mean_latency_s']:.1f}",
                    f"{c['mean_attempts']:.2f}", ", ".join(c["returned_models"])[:60], c["first_utc"][:10],
                    c["last_utc"][:10]] for c in cost])
    macros = {}
    for d in rows:
        if d["arm"] != "probe":
            macros[macro_name("exact", d["arm"], d["model"])] = f"{100 * d['exact_mean_k'][0]:.1f}"
            macros[macro_name("global", d["arm"], d["model"])] = f"{d['global_v2'][0]:.1f}"
    if vd:
        macros["\\VarShareModel"] = f"{100 * vd['model']:.1f}"
        macros["\\VarShareFamily"] = f"{100 * vd['family']:.1f}"
    write(out_dir, "a01_overview", L, {"outcomes": rows, "cost": cost, "tiers_exact": t_exact,
                                       "tiers_global": t_global, "variance": vd})
    return macros
