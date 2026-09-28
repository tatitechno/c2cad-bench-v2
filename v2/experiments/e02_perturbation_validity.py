"""E02: metric validity under controlled corruptions of the 75 v2 references.

Each perturbation is applied to every reference; the v2 evaluator scores the result.
Expectations (stated before running):
  * identity, shuffle, id renumbering -> 100 on every axis (invariance);
  * rigid translation/rotation        -> Cov 100, Sem loses only anchor constraints, Geom decreases with magnitude;
  * jitter                            -> all axes decrease monotonically with sigma;
  * deletion                          -> Cov, Geom and Sem decrease with the deleted fraction;
  * beam->axis-aligned box            -> type fidelity and Sem drop, geometry_equiv > geometry;
  * mirror (handedness flip)          -> Sem drops on pattern/orientation, Cov 100;
  * random parts / count-only         -> near-zero Geom and Sem.
Output: v2/reports/results/e02_perturbation_validity.{json,md}
"""
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c2cad import cases  # noqa: E402
from c2cad.evaluate import evaluate  # noqa: E402

rng_global = 20260928


def _pts(s):
    return [k for k in ("center", "start", "end") if k in s]


def translate(ref, d):
    out = copy.deepcopy(ref)
    for s in out:
        for k in _pts(s):
            s[k] = [s[k][0] + d, s[k][1], s[k][2]]
    return out


def rotz(ref, deg):
    c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    out = copy.deepcopy(ref)
    for s in out:
        for k in _pts(s) + (["axis"] if "axis" in s else []):
            s[k] = (R @ np.array(s[k])).tolist()
        if s["type"] == "box":        # axis-aligned boxes cannot rotate: keep size (this is a genuine error)
            pass
    return out


def jitter(ref, sigma, rng):
    out = copy.deepcopy(ref)
    for s in out:
        for k in _pts(s):
            s[k] = (np.array(s[k]) + rng.normal(0, sigma, 3)).tolist()
    return out


def delete(ref, frac, rng):
    keep = sorted(rng.choice(len(ref), size=max(1, round(len(ref) * (1 - frac))), replace=False))
    return [copy.deepcopy(ref[i]) for i in keep]


def duplicate(ref, frac, rng):
    extra = [copy.deepcopy(ref[i]) for i in rng.choice(len(ref), size=max(1, round(len(ref) * frac)), replace=False)]
    for j, s in enumerate(extra):
        s["id"] = 10_000 + j
    return copy.deepcopy(ref) + extra


def shuffle(ref, rng):
    out = copy.deepcopy(ref)
    rng.shuffle(out)
    return out


def renumber(ref):
    out = copy.deepcopy(ref)
    for i, s in enumerate(out):
        s["id"] = 1 + i          # 1-based ids
    return out


def beam_to_box(ref):
    out = []
    for s in copy.deepcopy(ref):
        if s["type"] == "beam":
            a, b = np.array(s["start"]), np.array(s["end"])
            L = float(np.linalg.norm(b - a))
            s = {"id": s["id"], "type": "box", "center": ((a + b) / 2).tolist(), "size": [L, s["width"], s.get("height", s["width"])]}
        out.append(s)
    return out


def mirror_y(ref):
    out = copy.deepcopy(ref)
    for s in out:
        for k in _pts(s):
            s[k] = [s[k][0], -s[k][1], s[k][2]]
        if "axis" in s:
            s["axis"] = [s["axis"][0], -s["axis"][1], s["axis"][2]]
    return out


def random_parts(ref, rng, keep_types=True):
    P = []
    for s in ref:
        P += [s[k] for k in _pts(s)]
    P = np.array(P); lo, hi = P.min(0) - 1, P.max(0) + 1
    out = []
    for s in copy.deepcopy(ref):
        d = rng.uniform(lo, hi, 3).tolist()
        if keep_types:
            if "center" in s:
                s["center"] = d
            if "start" in s:
                e = rng.uniform(lo, hi, 3).tolist(); s["start"], s["end"] = d, e
        else:
            s = {"id": s["id"], "type": "box", "center": d, "size": [1, 1, 1]}
        out.append(s)
    return out


PERTURBATIONS = [
    ("identity", lambda r, g: copy.deepcopy(r)),
    ("shuffle order", lambda r, g: shuffle(r, g)),
    ("ids 1-based", lambda r, g: renumber(r)),
    ("translate 0.5 mm", lambda r, g: translate(r, 0.5)),
    ("translate 2 mm", lambda r, g: translate(r, 2.0)),
    ("translate 10 mm", lambda r, g: translate(r, 10.0)),
    ("rotate Z 2 deg", lambda r, g: rotz(r, 2.0)),
    ("rotate Z 10 deg", lambda r, g: rotz(r, 10.0)),
    ("jitter 0.1 mm", lambda r, g: jitter(r, 0.1, g)),
    ("jitter 0.5 mm", lambda r, g: jitter(r, 0.5, g)),
    ("jitter 2 mm", lambda r, g: jitter(r, 2.0, g)),
    ("delete 10%", lambda r, g: delete(r, 0.10, g)),
    ("delete 25%", lambda r, g: delete(r, 0.25, g)),
    ("delete 50%", lambda r, g: delete(r, 0.50, g)),
    ("duplicate 50%", lambda r, g: duplicate(r, 0.50, g)),
    ("beam -> axis-aligned box", lambda r, g: beam_to_box(r)),
    ("mirror Y (handedness)", lambda r, g: mirror_y(r)),
    ("random positions, types kept", lambda r, g: random_parts(r, g, True)),
    ("count only (unit boxes, random)", lambda r, g: random_parts(r, g, False)),
]
AXES = ("coverage", "geometry", "geometry_equiv", "type_fidelity", "semantic", "semantic_id_binding", "global_v2")


def main():
    C = cases.load()
    rows = []
    for name, fn in PERTURBATIONS:
        g = np.random.default_rng(rng_global)
        vals = {a: [] for a in AXES}
        kind = {}
        exact = 0
        for c in C:
            r = evaluate(c, fn(c["reference"], g))
            for a in AXES:
                v = getattr(r, a)
                if v is not None:
                    vals[a].append(v)
            exact += r.exact
            for k, v in r.by_kind.items():
                kind.setdefault(k, []).append(v)
        rows.append({"perturbation": name, **{a: round(float(np.mean(v)), 2) if v else None for a, v in vals.items()},
                     "exact_rate": round(100 * exact / len(C), 1),
                     "by_kind": {k: round(100 * float(np.mean(v)), 1) for k, v in sorted(kind.items())}})
        print(name, {a: rows[-1][a] for a in AXES}, rows[-1]["by_kind"], flush=True)
    out = ROOT / "reports/results"
    json.dump(rows, open(out / "e02_perturbation_validity.json", "w"), indent=1)
    kinds = sorted({k for r in rows for k in r["by_kind"]})
    head = "| perturbation | Cov | Geom | GeomEq | TypeFid | Sem | Sem(ids) | Global | exact % | " + " | ".join(kinds) + " |"
    lines = [head, "|" + "---|" * (9 + len(kinds))]
    for r in rows:
        lines.append(f"| {r['perturbation']} | " + " | ".join(str(r[a]) for a in AXES) + f" | {r['exact_rate']} | "
                     + " | ".join(str(r['by_kind'].get(k, '')) for k in kinds) + " |")
    open(out / "e02_perturbation_validity.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
