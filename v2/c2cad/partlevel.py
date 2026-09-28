"""Part-level agreement for named parts (probe arm, and the same parts inside full answers).

The probe arm asks for three named parts only, so the model derives them without writing the whole assembly.
The same three ids are scored inside every full coordinate answer too, which gives a paired comparison per
(model, case, part): if a part is right when asked alone but wrong inside the full answer, the difference is
the cost of producing everything (serialisation, length); if it is wrong in both, the derivation itself fails.

Per part:
  pair   the scorer's pose-gated pair score (0-100) against the reference part with that id
  exact  type equal, position within the length tolerance of the constraint layer (beams: both ends),
         every dimension within 1 %, axis within 0.5 degrees
Binding: `id` = the output's own id, with full answers numbered from 1 shifted as in evaluate.id_binding
(the primary H7 comparison); `assign` = the part the optimal assignment gives that role (full answers only; the
sensitivity analysis).
"""
from __future__ import annotations

import math

import numpy as np

from .constraints.core import ANG_TOL_DEG, LEN_TOL_FRAC, LEN_TOL_MIN, REL_TOL
from .geom import Shape, assembly_diagonal
from .score import pair_matrices

_DIMS = {"sphere": ["radius"], "cylinder": ["radius", "height"], "pipe": ["inner_radius", "outer_radius", "height"],
         "cone": ["base_radius", "top_radius", "height"], "torus": ["ring_radius", "tube_radius"]}


def _rel_ok(a, b):
    return abs(a - b) <= REL_TOL * max(abs(b), 1e-9) + 1e-9


def _line_deg(u, v, signed=False):
    d = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    if not signed:
        d = abs(d)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def part_exact(r: Shape, o: Shape, tl: float) -> bool:
    if r.type != o.type:
        return False
    if r.type == "beam":
        a = max(np.linalg.norm(r.start - o.start), np.linalg.norm(r.end - o.end))
        b = max(np.linalg.norm(r.start - o.end), np.linalg.norm(r.end - o.start))
        if min(a, b) > tl:
            return False
        rs, os_ = sorted([r.width, r.thickness]), sorted([o.width, o.thickness])
        return all(_rel_ok(x, y) for x, y in zip(os_, rs))
    if np.linalg.norm(r.center - o.center) > tl:
        return False
    if r.type == "box":
        return all(_rel_ok(x, y) for x, y in zip(o.size, r.size))
    if not all(_rel_ok(getattr(o, f), getattr(r, f)) or (f == "top_radius" and abs(getattr(o, f) - getattr(r, f)) <= tl)
               for f in _DIMS[r.type]):
        return False
    if r.type != "sphere" and _line_deg(r.axis, o.axis, signed=(r.type == "cone")) > ANG_TOL_DEG:
        return False
    return True


def by_id(out: list[Shape], one_based_shift: bool = True) -> dict[int, Shape]:
    """Output parts by their own integer id. With one_based_shift, a full answer numbered from 1 (every id an
    integer, the smallest 1, no 0) is shifted to 0-based, the same rule as evaluate.id_binding. The probe arm
    names its ids explicitly, so its answers are bound without the shift."""
    d = {}
    for s in out:
        try:
            v = s.id
            if isinstance(v, bool) or float(v) != int(v):
                continue
            d.setdefault(int(v), s)
        except (TypeError, ValueError):
            continue
    if one_based_shift and d and len(d) == len(out) and min(d) == 1 and 0 not in d:
        d = {k - 1: v for k, v in d.items()}
    return d


def named_parts(ref: list[Shape], out: list[Shape], ids: list[int], amap: dict | None = None,
                full_answer: bool = True) -> dict:
    """Metrics for the reference parts `ids` (reference index == prescribed id). full_answer applies the 1-based
    id shift (full answers); probe answers (ids named in the prompt) are bound without it."""
    D = assembly_diagonal(ref)
    tl = max(LEN_TOL_MIN, LEN_TOL_FRAC * D)
    res = {"ids": list(ids)}
    bindings = {"id": by_id(out, one_based_shift=full_answer)}
    if amap is not None:
        bindings["assign"] = {r: out[j] for r, j in amap.items()}
    for name, bound in bindings.items():
        pair, exact = [], []
        for k in ids:
            o = bound.get(k)
            if o is None or k >= len(ref):
                pair.append(0.0)
                exact.append(False)
                continue
            S, *_ = pair_matrices(ref, [o], equivalence=False)
            pair.append(round(100.0 * float(S[k, 0]), 4))
            exact.append(bool(part_exact(ref[k], o, tl)))
        res[f"pair_{name}"] = pair
        res[f"exact_{name}"] = exact
    return res
