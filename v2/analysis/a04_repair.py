"""A04: multi-turn repair (plan §4, H8) and oracle single-channel upper bounds on the same seed answers."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import V2, boot_mean, boot_paired_pooled, fmt_ci, macro_name, md_table, signflip_p, write

REPAIR = ("repair_generic", "repair_verifier")


def seeds(s: pd.DataFrame) -> pd.DataFrame:
    j = s[(s["split"] == "main") & (s["arm"] == "json") & (s["round"] == 0) & (s["sample"] == 0)]
    return j[(j["response_status"] == "complete") & (~j["exact"])][["model", "family", "case_id", "exact", "global_v2"]]


def trajectories(s: pd.DataFrame, rounds: int) -> pd.DataFrame:
    """One row per (model, arm, case, round 0..R) over the seeds; rounds after a chain stops carry forward."""
    sd = seeds(s)
    rep = s[(s["split"] == "main") & (s["arm"].isin(REPAIR)) & (s["sample"] == 0)]
    rows = []
    for arm in REPAIR:
        if not (rep["arm"] == arm).any():
            continue
        ra = rep[rep["arm"] == arm].set_index(["model", "case_id", "round"])
        for t in sd.itertuples():
            last_e, last_g = False, t.global_v2
            for rnd in range(rounds + 1):
                if rnd > 0 and (t.model, t.case_id, rnd) in ra.index:
                    x = ra.loc[(t.model, t.case_id, rnd)]
                    if x["response_status"] in ("complete", "truncated", "refusal"):
                        last_e, last_g = bool(x["exact"]), float(x["global_v2"])
                rows.append({"model": t.model, "arm": arm, "family": t.family, "case_id": t.case_id, "round": rnd,
                             "exact": float(last_e), "global_v2": last_g})
    return pd.DataFrame(rows)


def oracle_channels(r_dirs: list[Path], s: pd.DataFrame) -> pd.DataFrame:
    """Fix one error channel of each seed answer with the reference, then re-score (upper bounds, as in the rebuttal)."""
    from c2cad.evaluate import evaluate
    from c2cad.geom import normalize
    from c2cad.runner.arms import parse_json
    from c2cad.score import assignment_map
    cases = {c["case_id"]: c for c in (json.loads(l) for l in open(V2 / "data" / "cases_v2.jsonl"))}
    want = {(t.model, t.case_id) for t in seeds(s).itertuples()}
    texts = {}
    for d in r_dirs:
        p = d / "responses.jsonl"
        if p.exists():
            for line in open(p):
                rec = json.loads(line)
                if (rec["arm"] == "json" and rec.get("split", "main") == "main" and rec["sample"] == 0
                        and rec.get("round", 0) == 0 and (rec["model"], rec["case_id"]) in want and rec.get("ok")):
                    texts[(rec["model"], rec["case_id"])] = rec["text"]
    rows = []
    for (m, cid), text in texts.items():
        case = cases[cid]
        val, _ = parse_json(text)
        out, _ = normalize(val if val is not None else [])
        ref, _ = normalize(case["reference"])
        amap = assignment_map(ref, out)
        base = [o.to_json() for o in out]
        fixes = {}
        t = copy.deepcopy(base)                     # types (and their dimensions) from the reference, placed at the answer's center
        for i, j in amap.items():
            if out[j].type != ref[i].type:
                p = copy.deepcopy(case["reference"][i])
                shift = out[j].center - ref[i].center
                for k in ("center", "start", "end"):
                    if k in p:
                        p[k] = (np.array(p[k]) + shift).tolist()
                t[j] = p
        fixes["types"] = t
        pz = copy.deepcopy(base)                    # positions: move each assigned part onto its reference center
        for i, j in amap.items():
            shift = ref[i].center - out[j].center
            for k in ("center", "start", "end"):
                if k in pz[j] and pz[j][k] is not None:
                    pz[j][k] = (np.array(pz[j][k]) + shift).tolist()
        fixes["positions"] = pz
        miss = [case["reference"][i] for i in range(len(ref)) if i not in amap]
        fixes["missing"] = base + copy.deepcopy(miss)
        r0 = evaluate(case, base)
        row = {"model": m, "family": case["family"], "case_id": cid, "exact_base": float(r0.exact),
               "global_base": r0.global_v2}
        for k, v in fixes.items():
            r = evaluate(case, v)
            row[f"exact_{k}"], row[f"global_{k}"] = float(r.exact), r.global_v2
        rows.append(row)
    return pd.DataFrame(rows)


def run(s: pd.DataFrame, r: pd.DataFrame, out_dir, r_dirs=None) -> dict:
    L = ["# A04. Multi-turn repair (seeds: non-exact json answers, sample 0)", ""]
    data, macros = {}, {}
    if not s["arm"].isin(REPAIR).any():
        write(out_dir, "a04_repair", L + ["No repair runs found."], {})
        return macros
    R = int(s[s["arm"].isin(REPAIR)]["round"].max())
    tr = trajectories(s, R)
    L += [f"Seeds: {tr[['model', 'case_id']].drop_duplicates().shape[0]} (model, case) pairs; rounds 0..{R}. "
          "A chain that stops (verifier clean, exact, error) carries its last score forward.", ""]
    rows = []
    for (m, arm), g in tr.groupby(["model", "arm"]):
        cells = [m, arm]
        for rnd in range(R + 1):
            gr = g[g["round"] == rnd]
            cells.append(f"{100 * gr['exact'].mean():.1f} / {gr['global_v2'].mean():.1f}")
        rows.append(cells)
    L += md_table(["model", "arm"] + [f"round {i}: exact % / Global" for i in range(R + 1)], rows)
    if set(REPAIR) <= set(tr["arm"]):
        last = tr[tr["round"] == R]
        a = last[last["arm"] == "repair_verifier"][["model", "family", "case_id", "exact"]]
        b = last[last["arm"] == "repair_generic"][["model", "family", "case_id", "exact"]]
        m = a.merge(b, on=["model", "family", "case_id"], suffixes=("_v", "_g"))
        m["d"] = m["exact_v"] - m["exact_g"]
        pooled = boot_paired_pooled(m)
        from .a02_attribution import directional
        L += ["", f"H8 verifier - generic at round {R} (exact, pp, pooled): {fmt_ci(pooled, True)}; "
                  f"sign-flip p = {signflip_p(m):.4f}; verdict: {directional(pooled)}", ""]
        data["h8"] = {"pooled": pooled, "p": signflip_p(m),
                      "per_model": {mo: boot_mean(g, "d") for mo, g in m.groupby("model")}}
        macros["\\EffVerifierMinusGeneric"] = f"{100 * pooled[0]:.1f}"
    if r_dirs:
        oc = oracle_channels(r_dirs, s)
        if len(oc):
            L += ["## Oracle single-channel repairs of the same seeds (upper bounds)", ""]
            L += md_table(["model", "seeds", "base exact %", "types fixed", "positions fixed", "missing added",
                           "base Global", "types", "positions", "missing"],
                          [[mo, len(g), f"{100 * g['exact_base'].mean():.1f}",
                            *[f"{100 * g[f'exact_{k}'].mean():.1f}" for k in ("types", "positions", "missing")],
                            f"{g['global_base'].mean():.1f}",
                            *[f"{g[f'global_{k}'].mean():.1f}" for k in ("types", "positions", "missing")]]
                           for mo, g in oc.groupby("model")])
            data["oracle"] = oc.groupby("model").mean(numeric_only=True).to_dict()
    data["trajectories"] = tr.groupby(["model", "arm", "round"])[["exact", "global_v2"]].mean().reset_index().to_dict("records")
    write(out_dir, "a04_repair", L, data)
    return macros
