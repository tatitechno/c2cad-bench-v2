"""E04: sensitivity of v2 conclusions to every free choice in the scorer.

Rescores the 897 comparable released v1 outputs (Furniture and Axle excluded; see E03) under
one-at-a-time variations of the scorer and reports, per variant: the model ranking's Spearman
rho and Kendall tau against the default, the top-3 set, and the largest rank move.
Output: v2/reports/results/e04_metric_sensitivity.{md,json}
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
from c2cad import cases, score  # noqa: E402
from c2cad.constraints import core  # noqa: E402
from c2cad.evaluate import evaluate  # noqa: E402
from c2cad.stats import spearmanr, kendalltau  # noqa: E402

EXCLUDED = {"Furniture Assembly", "Axle Bearing"}
DEFAULT = dict(PAIR_FORM="multiplicative", COVERAGE_FORM="symmetric", POS_TOL_FRAC=0.05,
               ORIENT_FULL_ERROR_DEG=45.0, TOL_SCALE=1.0, W=(1 / 3, 1 / 3, 1 / 3), SEM="semantic")
VARIANTS = [
    ("default", {}),
    ("pair score additive (v1 form)", dict(PAIR_FORM="additive")),
    ("coverage v1 form", dict(COVERAGE_FORM="v1")),
    ("position tolerance 2.5% diag", dict(POS_TOL_FRAC=0.025)),
    ("position tolerance 10% diag", dict(POS_TOL_FRAC=0.10)),
    ("orientation saturates 15 deg", dict(ORIENT_FULL_ERROR_DEG=15.0)),
    ("orientation saturates 90 deg", dict(ORIENT_FULL_ERROR_DEG=90.0)),
    ("constraint tolerances x0.5", dict(TOL_SCALE=0.5)),
    ("constraint tolerances x2", dict(TOL_SCALE=2.0)),
    ("constraint tolerances x5", dict(TOL_SCALE=5.0)),
    ("weights 0.2/0.3/0.5 (v1)", dict(W=(0.2, 0.3, 0.5))),
    ("weights 0.5/0.25/0.25", dict(W=(0.5, 0.25, 0.25))),
    ("weights 0.25/0.5/0.25", dict(W=(0.25, 0.5, 0.25))),
    ("weights 0.25/0.25/0.5", dict(W=(0.25, 0.25, 0.5))),
    ("Geometry only", dict(W=(0, 1, 0))),
    ("Semantic only", dict(W=(0, 0, 1))),
    ("all v1-like choices", dict(PAIR_FORM="additive", COVERAGE_FORM="v1", W=(0.2, 0.3, 0.5))),
    ("Sem = mean over constraints", dict(SEM="semantic_flat")),
    ("Sem = mean over constraint kinds", dict(SEM="semantic_kind")),
    ("Sem = mean over prompt sentences", dict(SEM="semantic_sentence")),
]


def apply(cfg):
    score.PAIR_FORM = cfg["PAIR_FORM"]; score.COVERAGE_FORM = cfg["COVERAGE_FORM"]
    score.POS_TOL_FRAC = cfg["POS_TOL_FRAC"]; score.ORIENT_FULL_ERROR_DEG = cfg["ORIENT_FULL_ERROR_DEG"]
    core.TOL_SCALE = cfg["TOL_SCALE"]


def main():
    C = {(c["family"], c["level"]): c for c in cases.load()}
    outs = [json.loads(l) for l in open(REPO / "data/model_outputs.jsonl")]
    outs = [o for o in outs if o["family"] not in EXCLUDED]
    results = {}
    for name, delta in VARIANTS:
        cfg = dict(DEFAULT, **delta); apply(cfg)
        per_model = defaultdict(list)
        for o in outs:
            r = evaluate(C[(o["family"], o["difficulty_id"])], json.loads(o["output_shapes_json"]), weights=cfg["W"])
            sem = getattr(r, cfg["SEM"])
            w = cfg["W"]
            per_model[o["model_id"]].append(w[0] * r.coverage + w[1] * r.geometry + w[2] * sem)
        results[name] = {m: float(np.mean(v)) for m, v in per_model.items()}
        print(name, "done", flush=True)
    apply(DEFAULT)
    base = results["default"]; models = sorted(base, key=lambda m: -base[m])
    rank = lambda res: {m: i + 1 for i, m in enumerate(sorted(res, key=lambda m: -res[m]))}
    rb = rank(base)
    lines = ["| variant | Spearman rho | Kendall tau | top-3 | same top-3 set | max rank move |", "|---|---|---|---|---|---|"]
    rows = []
    for name, _ in VARIANTS:
        res = results[name]; rv = rank(res)
        rho = spearmanr([base[m] for m in models], [res[m] for m in models]).statistic
        tau = kendalltau([base[m] for m in models], [res[m] for m in models]).statistic
        top3 = sorted(res, key=lambda m: -res[m])[:3]
        mv = max(abs(rv[m] - rb[m]) for m in models)
        mover = max(models, key=lambda m: abs(rv[m] - rb[m]))
        same = set(top3) == set(models[:3])
        rows.append(dict(variant=name, rho=rho, tau=tau, top3=top3, same_top3=same, max_move=mv, mover=mover, means=res))
        lines.append(f"| {name} | {rho:.3f} | {tau:.3f} | {', '.join(top3)} | {same} | {mv} ({mover}) |")
    out = ROOT / "reports/results"
    open(out / "e04_metric_sensitivity.md", "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(out / "e04_metric_sensitivity.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
