"""A02: attribution contrasts on the main split (plan §4: H1-H7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from c2cad import cml_programs

from .common import (KINDS, boot_mean, boot_paired_pooled, case_means, fmt_ci, holm, included, macro_name, md_table,
                     paired, signflip_p, write)

CONTRASTS = [("tool", "json"), ("mates", "json"), ("mates", "tool"), ("neutral", "json"), ("v1", "json"),
             ("schema", "json"), ("cadquery", "json")]
EQ_MARGIN = 0.05
GATED = set(cml_programs.PROGRAMS)


def prepare(s: pd.DataFrame) -> pd.DataFrame:
    d = included(s[(s["split"] == "main") & (s["round"] == 0)]).copy()
    d = d[~((d["arm"] == "schema") & (d["enforcement"] != "strict_schema"))]   # H5 uses enforced responses only
    d["exact_f"] = d["exact"].astype(float)
    if "by_kind" in d:
        for k in KINDS:
            d[f"kind_{k}"] = d["by_kind"].apply(lambda x: x.get(k) if isinstance(x, dict) else np.nan)
    return d


def contrast(d: pd.DataFrame, arm: str, base: str, value: str, families=None) -> dict:
    x = d if families is None else d[d["family"].isin(families)]
    m = paired(x, arm, base, value)
    out = {"n_pairs": len(m), "models": sorted(m["model"].unique()) if len(m) else [],
           "pooled": boot_paired_pooled(m) if len(m) else (np.nan,) * 3, "p": signflip_p(m) if len(m) else np.nan,
           "per_model": {mo: boot_mean(g, "d") for mo, g in m.groupby("model")} if len(m) else {}}
    return out


def directional(ci) -> str:
    """> 0 hypothesis: supported if the CI excludes 0 above, contradicted if it excludes 0 below."""
    p, lo, hi = ci
    if not np.isfinite(p):
        return "n/a"
    return "supported" if lo > 0 else "contradicted" if hi < 0 else "inconclusive"


def equivalence(ci, margin=EQ_MARGIN) -> str:
    """= 0 hypothesis: supported if the CI lies inside +-margin; contradicted if it excludes 0 without lying
    inside the margin (a difference is detected and it is not shown to be negligible)."""
    p, lo, hi = ci
    if not np.isfinite(p):
        return "n/a"
    if -margin <= lo and hi <= margin:
        return "supported"
    return "contradicted" if (lo > 0 or hi < 0) else "inconclusive"


def both(a: str, b: str) -> str:
    if "contradicted" in (a, b):
        return "contradicted"
    return "supported" if a == b == "supported" else "inconclusive"


def verdicts_v3(c: dict, probe: dict) -> dict:
    """Plan §4 with the 2026-09-28 amendment: supported / contradicted / inconclusive."""
    def ci(k):
        return c[k]["pooled"]
    v = {"H1 tool > json and mates > json": (both(directional(ci("tool-json")), directional(ci("mates-json"))),
                                             f"tool-json {fmt_ci(ci('tool-json'), True)}; mates-json {fmt_ci(ci('mates-json'), True)}"),
         "H2 mates > tool": (directional(ci("mates-tool")),
                             f"all {fmt_ci(ci('mates-tool'), True)}; gated {fmt_ci(ci('mates-tool|gated'), True)} "
                             f"({directional(ci('mates-tool|gated'))}); ungated {fmt_ci(ci('mates-tool|ungated'), True)} "
                             f"({directional(ci('mates-tool|ungated'))})"),
         "H3 neutral = json (within +-5 pp)": (equivalence(ci("neutral-json")), fmt_ci(ci("neutral-json"), True)),
         "H4 v1 > json": (directional(ci("v1-json")), fmt_ci(ci("v1-json"), True)),
         "H5 schema = json (within +-5 pp)": (equivalence(ci("schema-json")), fmt_ci(ci("schema-json"), True)),
         "H6 cadquery vs json": ("descriptive", fmt_ci(ci("cadquery-json"), True))}
    if probe.get("id"):
        v["H7 probe > same parts in full json answers (id binding)"] = (
            directional(probe["id"]["diff_pooled"]),
            f"id {fmt_ci(probe['id']['diff_pooled'], True)}; assign (sensitivity) "
            f"{fmt_ci(probe['assign']['diff_pooled'], True)} ({directional(probe['assign']['diff_pooled'])})")
    return {k: {"verdict": s, "evidence": e} for k, (s, e) in v.items()}


def verdicts(c: dict) -> dict:
    def lo(k):
        return c[k]["pooled"][1]

    def hi(k):
        return c[k]["pooled"][2]

    def eq(k):
        return -EQ_MARGIN <= lo(k) and hi(k) <= EQ_MARGIN

    def fmt(k):
        return fmt_ci(c[k]["pooled"], True)
    v = {}
    v["H1 tool > json and mates > json"] = (lo("tool-json") > 0 and lo("mates-json") > 0,
                                            f"tool-json {fmt('tool-json')}; mates-json {fmt('mates-json')}")
    v["H2 mates > tool"] = (lo("mates-tool") > 0, f"all {fmt('mates-tool')}; gated {fmt('mates-tool|gated')}; "
                                                  f"ungated {fmt('mates-tool|ungated')}")
    v["H3 neutral = json (within +-5 pp)"] = (eq("neutral-json"), fmt("neutral-json"))
    v["H4 v1 > json"] = (lo("v1-json") > 0, fmt("v1-json"))
    v["H5 schema = json (within +-5 pp)"] = (eq("schema-json"), fmt("schema-json"))
    v["H6 cadquery vs json"] = (None, fmt("cadquery-json"))
    return {k: {"supported": s, "evidence": e} for k, (s, e) in v.items()}


def cad_conditional(d: pd.DataFrame) -> dict:
    c = d[d["arm"] == "cadquery"]
    out = {}
    for mo, g in c.groupby("model"):
        st = g["cad_stats"].apply(lambda x: x if isinstance(x, dict) else {})
        runs = g["exec_status"] == "ok"
        builds = runs & st.apply(lambda x: x.get("n_invalid", 1) == 0 and x.get("n_unrecognized", 1) == 0
                                 and x.get("n_solids", 0) > 0)
        b = g[builds]
        out[mo] = {"n": len(g), "runs": float(runs.mean()), "builds": float(builds.mean()),
                   "builds_not_exact": float((~b["exact"]).mean()) if len(b) else np.nan,
                   "builds_geomeq_lt70": float((b["geometry_equiv"] < 70).mean()) if len(b) else np.nan,
                   "builds_geomeq_lt90": float((b["geometry_equiv"] < 90).mean()) if len(b) else np.nan}
    return out


def probe_vs_full(d: pd.DataFrame) -> dict:
    p = d[d["arm"] == "probe"][["model", "family", "case_id", "probe_exact", "probe_pair"]]
    j = d[d["arm"] == "json"].copy()
    if p.empty or j.empty:
        return {}
    out = {}
    for binding in ("id", "assign"):
        j["full_exact"] = j["probe"].apply(lambda x: float(np.mean(x[f"exact_{binding}"])) if isinstance(x, dict) else np.nan)
        pc = p.groupby(["model", "family", "case_id"], as_index=False)["probe_exact"].mean()
        jc = j.groupby(["model", "family", "case_id"], as_index=False)["full_exact"].mean()
        m = pc.merge(jc, on=["model", "family", "case_id"])
        m["d"] = m["probe_exact"] - m["full_exact"]
        out[binding] = {"probe": boot_mean(pc, "probe_exact"), "full": boot_mean(jc, "full_exact"),
                        "diff_pooled": boot_paired_pooled(m), "p": signflip_p(m),
                        "per_model": {mo: boot_mean(g, "d") for mo, g in m.groupby("model")}}
    return out


def run(s: pd.DataFrame, r: pd.DataFrame, out_dir) -> dict:
    d = prepare(s)
    arms = set(d["arm"])
    C = {}
    for a, b in CONTRASTS:
        if a in arms and b in arms:
            C[f"{a}-{b}"] = contrast(d, a, b, "exact_f")
            C[f"{a}-{b}|global"] = contrast(d, a, b, "global_v2")
    if {"mates", "tool"} <= arms:
        C["mates-tool|gated"] = contrast(d, "mates", "tool", "exact_f", GATED)
        C["mates-tool|ungated"] = contrast(d, "mates", "tool", "exact_f", set(d["family"]) - GATED)
        C["mates-json|gated"] = contrast(d, "mates", "json", "exact_f", GATED)
        C["mates-json|ungated"] = contrast(d, "mates", "json", "exact_f", set(d["family"]) - GATED)
    pooled_p = {k: v["p"] for k, v in C.items() if "|" not in k and np.isfinite(v["p"])}
    adj = holm(pooled_p)
    for k in adj:
        C[k]["p_holm"] = adj[k]
    need = ["tool-json", "mates-json", "mates-tool", "neutral-json", "v1-json", "schema-json", "cadquery-json",
            "mates-tool|gated", "mates-tool|ungated"]
    probe = probe_vs_full(d)
    V = verdicts_v3(C, probe) if all(k in C for k in need) else {}
    kinds = {}
    for a, b in CONTRASTS:
        if a in arms and b in arms and "by_kind" in d:
            kinds[f"{a}-{b}"] = {k: contrast(d, a, b, f"kind_{k}")["pooled"] for k in KINDS}
    fam = d.groupby(["family", "arm"])["exact_f"].mean().unstack() if len(d) else pd.DataFrame()
    v1fam = {}
    if {"v1", "json"} <= arms:
        m = paired(d, "v1", "json", "exact_f")
        v1fam = {f: boot_mean(g.groupby(["model", "family", "case_id"], as_index=False)["d"].mean(), "d")
                 for f, g in m.groupby("family")}
    mates_invalid = {mo: float((g["exec_status"] == "invalid_program").mean())
                     for mo, g in d[d["arm"] == "mates"].groupby("model")}
    cad = cad_conditional(d)
    L = ["# A02. Attribution contrasts (main split)", "",
         "Effect = arm minus json, per case (mean over samples), averaged over cases; pooled = mean over models with "
         "the same family draw. Exact rates in percentage points; 95% family-cluster bootstrap CI; p = family-level "
         "sign-flip test, Holm across the pooled contrasts.", ""]
    rows = []
    for k, v in C.items():
        if k.endswith("|global"):
            continue
        rows.append([k, len(v["models"]), fmt_ci(v["pooled"], True), f"{v['p']:.4f}" if np.isfinite(v["p"]) else "n/a",
                     f"{v.get('p_holm', float('nan')):.4f}" if "p_holm" in v else "",
                     fmt_ci(C.get(k + "|global", {}).get("pooled", (np.nan,) * 3))])
    L += md_table(["contrast", "models", "exact (pp)", "p", "p Holm", "Global (points)"], rows)
    if V:
        L += ["", "## Pre-registered hypotheses", ""]
        L += ["Verdicts (plan §4, amendment of 2026-09-28): supported / contradicted / inconclusive. Directional: "
              "the CI excludes 0 in the stated / opposite direction. Equivalence (H3, H5): supported if the CI lies "
              "inside +-5 pp; contradicted if it excludes 0 without lying inside the margin.", ""]
        L += md_table(["hypothesis", "verdict", "evidence"], [[k, v["verdict"], v["evidence"]] for k, v in V.items()])
    L += ["", "## Per model (exact, pp)", ""]
    models = sorted({mo for v in C.values() for mo in v["per_model"]})
    keys = [k for k in C if "|" not in k]
    L += md_table(["model"] + keys, [[mo] + [fmt_ci(C[k]["per_model"].get(mo, (np.nan,) * 3), True) for k in keys]
                                     for mo in models])
    if kinds:
        L += ["", "## By constraint kind (pass-rate difference, pp, pooled)", ""]
        L += md_table(["contrast"] + list(KINDS), [[k] + [fmt_ci(v[kk], True) for kk in KINDS] for k, v in kinds.items()])
    if v1fam:
        L += ["", "## v1 - json by family (scaffolding sensitivity, pp)", ""]
        L += md_table(["family", "v1 - json"], [[f, fmt_ci(v, True)] for f, v in sorted(v1fam.items(), key=lambda kv: -kv[1][0])])
    if mates_invalid:
        L += ["", "## Mates: invalid-program rate", "", ", ".join(f"{k} {100 * v:.1f}%" for k, v in mates_invalid.items())]
    if cad:
        L += ["", "## CadQuery: runs, builds (every solid valid and recognised) and builds-but-wrong", ""]
        L += md_table(["model", "n", "runs %", "builds %", "builds & not exact %", "builds & GeomEq<70 %", "builds & GeomEq<90 %"],
                      [[mo, v["n"], f"{100 * v['runs']:.1f}", f"{100 * v['builds']:.1f}", f"{100 * v['builds_not_exact']:.1f}",
                        f"{100 * v['builds_geomeq_lt70']:.1f}", f"{100 * v['builds_geomeq_lt90']:.1f}"] for mo, v in cad.items()])
    if probe:
        L += ["", "## H7. Named parts: asked alone (probe) vs inside the full json answer (part-exact %)", ""]
        L += md_table(["binding", "probe", "full answer", "probe - full (pooled)", "p"],
                      [[b, fmt_ci(v["probe"], True), fmt_ci(v["full"], True), fmt_ci(v["diff_pooled"], True),
                        f"{v['p']:.4f}"] for b, v in probe.items()])
    if not fam.empty:
        L += ["", "## Exact rate by family and arm (pooled over models, %)", ""]
        cols = [c for c in ("json", "neutral", "v1", "schema", "tool", "mates", "cadquery") if c in fam.columns]
        L += md_table(["family", "gated"] + cols, [[f, "yes" if f in GATED else "no"] +
                                                  [f"{100 * fam.loc[f, c]:.0f}" if np.isfinite(fam.loc[f, c]) else "" for c in cols]
                                                  for f in fam.index])
    macros = {}
    for k, v in C.items():
        if "|" not in k:
            p, lo, hi = v["pooled"]
            a_, b_ = k.split("-")
            macros[macro_name("eff", a_, "minus", b_)] = f"{100 * p:.1f}"
            macros[macro_name("eff", a_, "minus", b_, "lo")] = f"{100 * lo:.1f}"
            macros[macro_name("eff", a_, "minus", b_, "hi")] = f"{100 * hi:.1f}"
    write(out_dir, "a02_attribution", L, {"contrasts": C, "verdicts": V, "kinds": kinds, "v1_by_family": v1fam,
                                          "mates_invalid": mates_invalid, "cadquery": cad, "probe": probe,
                                          "family_arm_exact": fam.to_dict() if not fam.empty else {}})
    return macros
