"""A06: annotated failure figures (Reviewer 2's request for Figure 5: failures marked on the geometry).

For each selected answer: the reference beside the answer, answer parts coloured by what went wrong
  grey    placed correctly (pair score >= 0.9)
  orange  right type, misplaced or mis-sized
  red     type substituted (e.g. an oriented beam answered by an axis-aligned box)
  purple  extra part (no reference counterpart)
  pink    reference part the answer lacks (drawn translucent in the answer panel)
with the component scores in the title. Examples are picked from a05_per_case_metrics.csv: per failure category,
the lowest-Global answer, one per model where possible.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .common import write  # noqa: E402

FATE = {"ok": "#9e9e9e", "misplaced": "#f28e2b", "substituted": "#e15759", "extra": "#7b4fa0", "missing": "#ff9da7"}
CATEGORIES = ("type substitution", "wrong part count", "placement or size")


def _draw(ax, shapes, colors, alphas, bounds, title):
    from c2cad.render import faces
    polys, cols = [], []
    n = 12 if len(shapes) < 150 else 8
    for s, c, a in zip(shapes, colors, alphas):
        fs = faces(s, n)
        polys += fs
        rgba = matplotlib.colors.to_rgba(c, a)
        cols += [rgba] * len(fs)
    if polys:
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, linewidths=0))
    lo, hi = np.array(bounds[0]), np.array(bounds[1])
    mid, span = (lo + hi) / 2, max((hi - lo).max(), 1e-6)
    for set_, m in ((ax.set_xlim, mid[0]), (ax.set_ylim, mid[1]), (ax.set_zlim, mid[2])):
        set_(m - span / 2, m + span / 2)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, fontsize=6)


def fates(case, raw):
    from c2cad.geom import normalize
    from c2cad.render import COLORS, bounds_of
    from c2cad.score import pair_matrices
    ref, _ = normalize(case["reference"])
    out, _ = normalize(raw)
    colors, alphas, shapes = [], [], []
    matched_ref = set()
    if out:
        S, *_ = pair_matrices(ref, out, equivalence=False)
        from scipy.optimize import linear_sum_assignment
        r_, c_ = linear_sum_assignment(-S)
        best = {int(j): int(i) for i, j in zip(r_, c_)}
        for j, o in enumerate(out):
            i = best.get(j)
            if i is None:
                fate = "extra"
            else:
                matched_ref.add(i)
                fate = ("ok" if S[i, j] >= 0.9 else "misplaced") if ref[i].type == o.type else "substituted"
            shapes.append(o); colors.append(FATE[fate]); alphas.append(0.9)
    for i, rs in enumerate(ref):
        if i not in matched_ref:
            shapes.append(rs); colors.append(FATE["missing"]); alphas.append(0.25)
    b = bounds_of(ref + out)
    return ref, [COLORS.get(s.type, "#999") for s in ref], shapes, colors, alphas, b


def pick(csv_path, per_category: int = 2) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for cat in CATEGORIES:
        d = df[df["taxonomy"] == cat].sort_values("global_v2")
        seen = set()
        for x in d.itertuples():
            if x.model in seen:
                continue
            rows.append(x._asdict())
            seen.add(x.model)
            if len([r for r in rows if r["taxonomy"] == cat]) >= per_category:
                break
    return pd.DataFrame(rows)


def run(s, r, out_dir, r_dirs=None) -> dict:
    csv_path = out_dir / "a05_per_case_metrics.csv"
    if not csv_path.exists():
        return {}
    from .a05_validity import _cases, json_answers
    from c2cad.runner.arms import parse_json
    sel = pick(csv_path)
    if sel.empty:
        write(out_dir, "a06_failure_figures", ["# A06. Annotated failures", "", "No failures to show."], {})
        return {}
    answers = json_answers(r_dirs or [])
    n = len(sel)
    fig = plt.figure(figsize=(3.2 * 2, 2.6 * n))
    L = ["# A06. Annotated failures (json arm, sample 0)", "", "Colours: " + ", ".join(f"{k} {v}" for k, v in FATE.items()), ""]
    for k, x in enumerate(sel.itertuples()):
        case = _cases()[x.case_id]
        val, _ = parse_json(answers.get((x.model, x.case_id), ""))
        ref, rcol, shapes, cols, alph, b = fates(case, val if val is not None else [])
        ax = fig.add_subplot(n, 2, 2 * k + 1, projection="3d")
        _draw(ax, ref, rcol, [0.9] * len(ref), b, f"reference: {x.case_id} ({x.n_ref} parts)")
        ax = fig.add_subplot(n, 2, 2 * k + 2, projection="3d")
        _draw(ax, shapes, cols, alph, b, f"{x.model}: {x.taxonomy}\nCov {x.coverage:.0f} Geom {x.geometry:.0f} "
                                          f"Sem {x.semantic:.0f} IoU {x.iou:.2f} parts {x.n_out}/{x.n_ref}")
        L.append(f"- {x.model} / {x.case_id}: {x.taxonomy}; Global {x.global_v2:.1f}")
    fig.legend(handles=[Patch(color=v, label=k) for k, v in FATE.items()], loc="lower center", ncol=5, fontsize=6,
               frameon=False)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_dir / "fig_failures.pdf")
    fig.savefig(out_dir / "fig_failures.png", dpi=130)
    plt.close(fig)
    write(out_dir, "a06_failure_figures", L, {"selected": json.loads(sel.to_json(orient="records"))})
    return {}
