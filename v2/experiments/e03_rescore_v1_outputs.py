"""E03: rescore the 975 released v1 model outputs with the v2 scorer.

The v1 outputs answered the v1 prompts. Their reference geometry equals v2's except for
documented patches; two families had under-determined v1 prompts ("choose the dimensions
freely") and are excluded from reference-based aggregates: Furniture Assembly, Axle Bearing.

Outputs (v2/reports/results/):
  e03_rows.csv                 one row per (model, case) with every v2 axis and the v1 scores
  e03_model_summary.md/json    per-model means with family-clustered 95% CIs, v1 vs v2 ranks
  e03_family_summary.md        per-family means across models
  e03_type_confusion.md        label substitutions among spatially co-located matched pairs
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from c2cad import cases  # noqa: E402
from c2cad.evaluate import evaluate  # noqa: E402
from c2cad.geom import normalize  # noqa: E402
from c2cad.score import pair_matrices, _assign  # noqa: E402
from c2cad.stats import cluster_bootstrap_mean, cluster_bootstrap_ranks, spearmanr, kendalltau  # noqa: E402

EXCLUDED = {"Furniture Assembly", "Axle Bearing"}
AXES = ["coverage", "geometry", "geometry_equiv", "type_fidelity", "semantic", "global_v2"]


def main():
    C = {(c["family"], c["level"]): c for c in cases.load()}
    rows, confusion, conf_all = [], Counter(), Counter()
    for line in open(REPO / "data/model_outputs.jsonl"):
        o = json.loads(line)
        case = C[(o["family"], o["difficulty_id"])]
        raw = json.loads(o["output_shapes_json"])
        r = evaluate(case, raw)
        row = {"model": o["model_id"], "case_id": case["case_id"], "family": o["family"], "phase": case["phase"],
               "level": case["level"], "n_ref": r.n_ref, "n_out": r.n_out, "excluded": o["family"] in EXCLUDED,
               **{a: round(getattr(r, a), 3) for a in AXES},
               "semantic_id_binding": None if r.semantic_id_binding is None else round(r.semantic_id_binding, 3),
               "exact": r.exact,
               **{f"kind_{k}": round(100 * v, 2) for k, v in r.by_kind.items()},
               "v1_cov": o["score_cov"], "v1_geom": o["score_geom"], "v1_sem": o["score_sem"], "v1_global": o["score_global"]}
        rows.append(row)
        # type confusion among co-located pairs (position credit > 0.5 under the optimal assignment)
        ref, _ = normalize(case["reference"]); out, _ = normalize(raw)
        if ref and out and o["family"] not in EXCLUDED:
            S, dist, pos, _ = pair_matrices(ref, out, equivalence=False)
            rr, cc = _assign(S)
            for i, j in zip(rr, cc):
                if ref[i].type != out[j].type:
                    conf_all[(ref[i].type, out[j].type)] += 1
                    if pos[i, j] > 0.5:
                        confusion[(ref[i].type, out[j].type, o["family"])] += 1
    out_dir = ROOT / "reports/results"
    keys = sorted({k for r in rows for k in r})
    with open(out_dir / "e03_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    incl = [r for r in rows if not r["excluded"]]
    models = sorted({r["model"] for r in incl})
    summ = {}
    for m in models:
        mr = [r for r in incl if r["model"] == m]
        d = {}
        for a in AXES + ["v1_global"]:
            fam = defaultdict(list)
            for r in mr:
                fam[r["family"]].append(r[a])
            d[a] = cluster_bootstrap_mean(fam)
        d["exact_rate"] = 100 * np.mean([r["exact"] for r in mr])
        d["n"] = len(mr)
        summ[m] = d
    table = {m: defaultdict(list) for m in models}
    for r in incl:
        table[r["model"]][r["family"]].append(r["global_v2"])
    ranks = cluster_bootstrap_ranks(table)
    v1_order = sorted(models, key=lambda m: -summ[m]["v1_global"][0])
    v2_order = sorted(models, key=lambda m: -summ[m]["global_v2"][0])
    rho = spearmanr([summ[m]["v1_global"][0] for m in models], [summ[m]["global_v2"][0] for m in models])
    tau = kendalltau([summ[m]["v1_global"][0] for m in models], [summ[m]["global_v2"][0] for m in models])
    lines = [f"Rescored outputs: {len(incl)} (excluded families: {sorted(EXCLUDED)})",
             f"Model-level rank agreement v1 Global vs v2 Global: Spearman rho = {rho.statistic:.3f}, Kendall tau = {tau.statistic:.3f}", "",
             "| v2 rank | model | v1 rank | Global v2 [95% CI] | Cov | Geom | GeomEq | TypeFid | Sem | exact % | v1 Global | rank CI (P top3) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, m in enumerate(v2_order, 1):
        s = summ[m]; g = s["global_v2"]; rk = ranks[m]
        lines.append(f"| {i} | {m} | {v1_order.index(m) + 1} | {g[0]:.1f} [{g[1]:.1f}, {g[2]:.1f}] | {s['coverage'][0]:.1f} | "
                     f"{s['geometry'][0]:.1f} | {s['geometry_equiv'][0]:.1f} | {s['type_fidelity'][0]:.1f} | {s['semantic'][0]:.1f} | "
                     f"{s['exact_rate']:.1f} | {s['v1_global'][0]:.1f} | [{rk[1]}, {rk[2]}] ({100 * rk[3]:.0f}%) |")
    open(out_dir / "e03_model_summary.md", "w").write("\n".join(lines) + "\n")
    json.dump({m: {k: v for k, v in s.items()} for m, s in summ.items()}, open(out_dir / "e03_model_summary.json", "w"), indent=1)
    print("\n".join(lines))

    fam_lines = ["| family | phase | Cov | Geom | GeomEq | TypeFid | Sem | Global v2 | v1 Global | exact % | excluded |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    fams = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    for f, fr in sorted(fams.items(), key=lambda kv: np.mean([r["global_v2"] for r in kv[1]])):
        fam_lines.append(f"| {f} | {fr[0]['phase']} | " + " | ".join(f"{np.mean([r[a] for r in fr]):.1f}" for a in AXES)
                         + f" | {np.mean([r['v1_global'] for r in fr]):.1f} | {100 * np.mean([r['exact'] for r in fr]):.1f} | {fr[0]['excluded']} |")
    open(out_dir / "e03_family_summary.md", "w").write("\n".join(fam_lines) + "\n")
    print("\n".join(fam_lines))

    tot = Counter(); fam_break = defaultdict(Counter)
    for (g, o_, f), n in confusion.items():
        tot[(g, o_)] += n; fam_break[(g, o_)][f] += n
    cl = ["| reference type | output type | co-located pairs | all matched pairs | top families |", "|---|---|---|---|---|"]
    for (g, o_), n in tot.most_common(12):
        cl.append(f"| {g} | {o_} | {n} | {conf_all[(g, o_)]} | " + ", ".join(f"{f} ({k})" for f, k in fam_break[(g, o_)].most_common(3)) + " |")
    open(out_dir / "e03_type_confusion.md", "w").write("\n".join(cl) + "\n")
    print("\n".join(cl))


if __name__ == "__main__":
    main()
