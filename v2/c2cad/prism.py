"""Reading of label-free rectangular solids ("prisms") recovered from CAD code (cadquery arm).

A rectangular solid built in a CAD kernel has no box/beam label and no designated centerline. The same solid
can be written in the v2 schema as an axis-aligned box (only if its faces are aligned with the world axes) or
as a beam along any of its three axes. resolve() replaces each {"type": "prism", "center", "axes", "extents"}
part by the reading whose best pair score against the reference is highest. Every reading occupies the same
volume, so the choice never moves material; it only picks the description. This is the same principle as the
scorer's geometry-equivalent view (an axis-aligned beam and a box of equal extents are the same solid).
"""
from __future__ import annotations

import numpy as np

from .geom import normalize

ALIGN_TOL_DEG = 0.5


def readings(p: dict) -> list[dict]:
    c = np.asarray(p["center"], float)
    A = np.asarray(p["axes"], float)
    e = np.asarray(p["extents"], float)
    out = []
    world = np.abs(A)                         # rows: prism axes; columns: world X, Y, Z
    k = world.argmax(1)
    if sorted(k.tolist()) == [0, 1, 2] and np.all(world.max(1) >= np.cos(np.radians(ALIGN_TOL_DEG))):
        size = np.zeros(3)
        size[k] = e
        out.append({"id": p.get("id"), "type": "box", "center": c.tolist(), "size": size.tolist()})
    for i in range(3):
        u = A[i]
        o = [j for j in range(3) if j != i]
        # section 'height' is the extent along the axis closer to vertical, as in the beam convention
        hj = max(o, key=lambda j: abs(A[j][2]))
        wj = o[0] if o[1] == hj else o[1]
        out.append({"id": p.get("id"), "type": "beam", "start": (c - u * e[i] / 2).tolist(),
                    "end": (c + u * e[i] / 2).tolist(), "width": float(e[wj]), "height": float(e[hj])})
    return out


def resolve(ref_shapes, raw: list) -> tuple[list, int]:
    """Replace prism parts by their best-fitting reading. Returns (parts, number of prisms resolved)."""
    from .score import pair_matrices   # local import: score imports geom only

    prisms = [i for i, p in enumerate(raw) if isinstance(p, dict) and p.get("type") == "prism"]
    if not prisms:
        return raw, 0
    out = list(raw)
    for i in prisms:
        cands = readings(raw[i])
        shapes, _ = normalize(cands)
        if not ref_shapes or len(shapes) != len(cands):
            out[i] = cands[-1] if len(cands) == 3 else cands[0]
            continue
        S, *_ = pair_matrices(ref_shapes, shapes, equivalence=False)
        best = S.max(axis=0)
        out[i] = cands[int(np.argmax(best))]     # ties: the box reading (listed first), then the first axis
    return out, len(prisms)
