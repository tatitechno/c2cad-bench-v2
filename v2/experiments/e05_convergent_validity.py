"""E05: agreement between v2 axes and standard 3D shape metrics (convergent/discriminant validity).

For the 897 comparable released v1 outputs: voxel IoU (96^3), symmetric Chamfer / reference
diagonal, F-score at 2% of the diagonal (1% is at the sampling noise floor, see notebook), and
mean orientation error over matched parts. Reports case-level Spearman correlations, model-level
rank agreement, and the disagreement quadrants (with example cases) that motivate component-level
scoring.
Output: v2/reports/results/e05_convergent_validity.{md,csv}
"""
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from c2cad import cases, metrics3d as M  # noqa: E402
from c2cad.evaluate import evaluate  # noqa: E402
from c2cad.geom import normalize  # noqa: E402
from c2cad.score import assignment_map  # noqa: E402
from c2cad.stats import spearmanr  # noqa: E402

EXCLUDED = {"Furniture Assembly", "Axle Bearing"}
CASES = {(c["family"], c["level"]): c for c in cases.load()}


def work(o):
    case = CASES[(o["family"], o["difficulty_id"])]
    raw = json.loads(o["output_shapes_json"])
    r = evaluate(case, raw)
    ref, _ = normalize(case["reference"]); out, _ = normalize(raw)
    iou = M.iou(ref, out) if out else 0.0
    ch, fs = M.chamfer_fscore(ref, out, n=40000, thresholds=(0.02,)) if out else (float("nan"), {0.02: 0.0})
    oe = M.orientation_error(ref, out, assignment_map(ref, out)) if out else float("nan")
    return dict(model=o["model_id"], case_id=case["case_id"], family=o["family"], coverage=r.coverage, geometry=r.geometry,
                geometry_equiv=r.geometry_equiv, semantic=r.semantic, global_v2=r.global_v2, exact=r.exact,
                iou=iou, chamfer=ch, f2=fs[0.02], orient_err=oe)


def main():
    outs = [json.loads(l) for l in open(REPO / "data/model_outputs.jsonl")]
    outs = [o for o in outs if o["family"] not in EXCLUDED]
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(work, outs, chunksize=8))
    out_dir = ROOT / "reports/results"
    with open(out_dir / "e05_convergent_validity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    axes = ["coverage", "geometry", "geometry_equiv", "semantic", "global_v2"]
    alt = [("iou", 1), ("f2", 1), ("chamfer", -1), ("orient_err", -1)]
    L = ["Case-level Spearman correlation (n = %d outputs). Chamfer and orientation error are sign-flipped so that higher = better." % len(rows), "",
         "| | " + " | ".join(a for a, _ in alt) + " |", "|---|" + "---|" * len(alt)]
    for ax in axes:
        cells = []
        for a, sg in alt:
            pairs = [(r[ax], sg * r[a]) for r in rows if np.isfinite(r[a])]
            cells.append(f"{spearmanr(*zip(*pairs)).statistic:.3f}")
        L.append(f"| {ax} | " + " | ".join(cells) + " |")
    # model level
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in axes + [a for a, _ in alt]:
            if np.isfinite(r[k]):
                per[r["model"]][k].append(r[k])
    models = sorted(per)
    L += ["", "Model-level rank agreement with v2 Global (Spearman over 13 model means):", ""]
    g = [np.mean(per[m]["global_v2"]) for m in models]
    for a, sg in alt:
        L.append(f"- {a}: rho = {spearmanr(g, [sg * np.mean(per[m][a]) for m in models]).statistic:.3f}")
    L += ["", "| model | Global v2 | Geom | Sem | IoU | F@2% | Chamfer | orient err (deg) |", "|---|---|---|---|---|---|---|---|"]
    for m in sorted(models, key=lambda m: -np.mean(per[m]["global_v2"])):
        L.append(f"| {m} | " + " | ".join(f"{np.mean(per[m][k]):.3f}" if k in ("iou", "f2", "chamfer") else f"{np.mean(per[m][k]):.1f}"
                                           for k in ("global_v2", "geometry", "semantic", "iou", "f2", "chamfer", "orient_err")) + " |")
    # disagreement quadrants
    q1 = [r for r in rows if r["iou"] >= 0.8 and r["semantic"] < 60]
    q2 = [r for r in rows if r["iou"] < 0.3 and r["geometry"] >= 80]
    q3 = [r for r in rows if r["f2"] >= 0.9 and not r["exact"]]
    L += ["", f"High IoU (>= 0.8) but Sem < 60: {len(q1)} / {len(rows)} ({100 * len(q1) / len(rows):.1f}%)",
          f"Low IoU (< 0.3) but Geom >= 80: {len(q2)} / {len(rows)} ({100 * len(q2) / len(rows):.1f}%)",
          f"F@2% >= 0.9 but not exact (some stated constraint violated): {len(q3)} / {len(rows)} ({100 * len(q3) / len(rows):.1f}%)", "",
          "Examples (high IoU, low Sem):"]
    for r in sorted(q1, key=lambda r: r["semantic"])[:8]:
        L.append(f"- {r['model']} / {r['case_id']}: IoU {r['iou']:.2f}, F@2% {r['f2']:.2f}, Geom {r['geometry']:.1f}, Sem {r['semantic']:.1f}")
    L.append(""); L.append("Examples (low IoU, high Geom):")
    for r in sorted(q2, key=lambda r: r["iou"])[:8]:
        L.append(f"- {r['model']} / {r['case_id']}: IoU {r['iou']:.2f}, F@2% {r['f2']:.2f}, Geom {r['geometry']:.1f}, Sem {r['semantic']:.1f}")
    open(out_dir / "e05_convergent_validity.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
