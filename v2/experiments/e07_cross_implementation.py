"""E07: agreement between the v2 geometry scorer and an independently written optimal matcher
(the June 2026 `runners/matching.py` in the author's C2CAD working copy, which the v2 code was not
derived from). Both score the 897 comparable released v1 outputs against the v2 references.

The June matcher uses v1's additive pair score, v1 coverage and a 0.7 credit cap for permuted boxes,
so it is compared with v2 run in its additive / v1-coverage configuration (closest like-for-like) and
with the default v2 configuration. Output: v2/reports/results/e07_cross_implementation.md
"""
import io
import contextlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
JUNE = Path("/Users/ebentria/Documents/C2CAD/runners")
sys.path.insert(0, str(ROOT))
from c2cad import cases, score  # noqa: E402
from c2cad.geom import normalize  # noqa: E402
from c2cad.stats import spearmanr, kendalltau  # noqa: E402

EXCLUDED = {"Furniture Assembly", "Axle Bearing"}


def main():
    sys.path.insert(0, str(JUNE))
    with contextlib.redirect_stdout(io.StringIO()):
        import matching as june  # noqa: E402
    C = {(c["family"], c["level"]): c for c in cases.load()}
    outs = [json.loads(l) for l in open(REPO / "data/model_outputs.jsonl")]
    outs = [o for o in outs if o["family"] not in EXCLUDED]
    rows = []
    for o in outs:
        case = C[(o["family"], o["difficulty_id"])]
        raw = json.loads(o["output_shapes_json"])
        ref, _ = normalize(case["reference"]); out, _ = normalize(raw)
        with contextlib.redirect_stdout(io.StringIO()):
            jc, jg = june.eval_cov_geom_v2(raw, case["reference"], matcher="optimal")
        vals = {}
        for name, pf, cf in (("v2_default", "multiplicative", "symmetric"), ("v2_additive", "additive", "v1")):
            score.PAIR_FORM, score.COVERAGE_FORM = pf, cf
            a = score.agreement(ref, out)
            vals[name] = (a.coverage, a.geometry)
        score.PAIR_FORM, score.COVERAGE_FORM = "multiplicative", "symmetric"
        rows.append(dict(model=o["model_id"], case=case["case_id"], june_cov=jc, june_geom=jg,
                         add_cov=vals["v2_additive"][0], add_geom=vals["v2_additive"][1],
                         def_cov=vals["v2_default"][0], def_geom=vals["v2_default"][1]))
    L = [f"n = {len(rows)} released outputs (Furniture, Axle excluded)", ""]
    for lab, key in (("v2 additive/v1-coverage (like-for-like)", "add"), ("v2 default (pose-gated, symmetric coverage)", "def")):
        g_j = np.array([r["june_geom"] for r in rows]); g_v = np.array([r[f"{key}_geom"] for r in rows])
        c_j = np.array([r["june_cov"] for r in rows]); c_v = np.array([r[f"{key}_cov"] for r in rows])
        d = g_v - g_j
        L.append(f"### June matcher vs {lab}")
        L.append(f"- Geometry: case-level Spearman {spearmanr(g_j, g_v).statistic:.3f}; mean |diff| {np.mean(np.abs(d)):.2f}; "
                 f"within 2 points: {100 * np.mean(np.abs(d) <= 2):.1f}%; max |diff| {np.max(np.abs(d)):.1f}")
        L.append(f"- Coverage: exact agreement {100 * np.mean(np.abs(c_j - c_v) <= 0.5):.1f}%")
        per = defaultdict(lambda: ([], []))
        for r in rows:
            per[r["model"]][0].append(r["june_geom"]); per[r["model"]][1].append(r[f"{key}_geom"])
        mj = {m: np.mean(v[0]) for m, v in per.items()}; mv = {m: np.mean(v[1]) for m, v in per.items()}
        ms = sorted(mj)
        L.append(f"- Model-level geometry ranking: Spearman {spearmanr([mj[m] for m in ms], [mv[m] for m in ms]).statistic:.3f}, "
                 f"Kendall {kendalltau([mj[m] for m in ms], [mv[m] for m in ms]).statistic:.3f}")
        worst = sorted(rows, key=lambda r: -abs(r[f"{key}_geom"] - r["june_geom"]))[:5]
        L.append("- Largest disagreements: " + "; ".join(f"{r['model']}/{r['case']} June {r['june_geom']} vs v2 {r[f'{key}_geom']:.1f}" for r in worst))
        L.append("")
    out = ROOT / "reports/results/e07_cross_implementation.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
