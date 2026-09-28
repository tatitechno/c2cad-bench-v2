"""Reference-agreement scoring for C2CAD-Bench v2: Coverage and Geometry.

Differences from v1 (runners/run_unified.py), each motivated by a verified defect:
  * one normalizer for both sides (v1 normalized only the model output);
  * optimal one-to-one assignment (Hungarian) instead of order-dependent greedy passes;
  * orientation is scored for every oriented primitive (v1 ignored cylinder/pipe/cone/torus axes
    and sorted box sides, so a box rotated by 90 degrees got full credit);
  * box sides are compared per world axis, because the schema defines boxes as axis-aligned;
  * position tolerance uses the diagonal of the reference parts' bounding boxes;
  * two views of type agreement: label-strict (primary) and geometry-equivalent
    (an axis-aligned beam and a box of the same extents are the same solid).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geom import Shape, TYPES, assembly_diagonal

W_POS, W_TYPE, W_SHAPE = 0.4, 0.2, 0.4
POS_TOL_FRAC, POS_TOL_MIN, POS_FALLOFF = 0.05, 2.0, 1.5
ORIENT_FULL_ERROR_DEG = 45.0        # angle at which the orientation term reaches its maximum error
PAIR_FORM = "multiplicative"        # pair = pos * (W_POS + W_TYPE*type + W_SHAPE*type*shape); "additive" = v1 form
COVERAGE_FORM = "symmetric"         # min(n,N)/max(n,N); "v1" = min(1,n/N) with excess penalty above 1.5x
EXCESS_START = 1.5                  # (v1 coverage form only)
N_DIMS_PROXY = 3                    # (additive form only) weight of dimensions vs orientation when averaged


def _rel(a: np.ndarray, b: np.ndarray, floor: float | np.ndarray = 1e-6) -> np.ndarray:
    return np.minimum(1.0, np.abs(a - b) / np.maximum(np.abs(b), floor))


def _angle_err(u: np.ndarray, v: np.ndarray, signed: bool) -> np.ndarray:
    """u: (n,3) reference directions, v: (m,3) output directions -> (n,m) error in [0,1]."""
    d = u @ v.T
    if not signed:
        d = np.abs(d)
    ang = np.degrees(np.arccos(np.clip(d, -1.0, 1.0)))
    return np.minimum(1.0, ang / ORIENT_FULL_ERROR_DEG)


def _arr(shapes, attr):
    return np.array([getattr(s, attr) for s in shapes], float)


def _same_type_shape_score(t: str, G: list[Shape], L: list[Shape]):
    """For two lists of the same primitive type return (dims, ori), each (len G, len L) in [0,1]:
    dims = 1 - mean relative dimension error; ori = 1 - orientation error (1 for sphere and box,
    whose orientation is fixed by the schema and checked through per-axis box sides)."""
    col = lambda xs, a: _arr(xs, a)[:, None]
    row = lambda xs, a: _arr(xs, a)[None, :]
    errs = []
    ori = np.ones((len(G), len(L)))
    if t == "sphere":
        errs.append(_rel(row(L, "radius"), col(G, "radius")))
    elif t == "box":
        gs, ls = np.stack([s.size for s in G]), np.stack([s.size for s in L])
        for k in range(3):
            errs.append(_rel(ls[None, :, k], gs[:, None, k]))
    elif t in ("cylinder", "pipe", "cone", "torus"):
        fields = {"cylinder": ["radius", "height"], "pipe": ["inner_radius", "outer_radius", "height"],
                  "cone": ["base_radius", "height"], "torus": ["ring_radius", "tube_radius"]}[t]
        for f in fields:
            errs.append(_rel(row(L, f), col(G, f)))
        if t == "cone":   # a pointed reference tip (0) is compared relative to the base radius
            errs.append(_rel(row(L, "top_radius"), col(G, "top_radius"), floor=0.1 * col(G, "base_radius")))
        ori = 1.0 - _angle_err(np.stack([s.axis for s in G]), np.stack([s.axis for s in L]), signed=(t == "cone"))
    elif t == "beam":
        errs.append(_rel(row(L, "length"), col(G, "length")))
        gsec = np.sort(np.stack([[s.width, s.thickness] for s in G]), axis=1)
        lsec = np.sort(np.stack([[s.width, s.thickness] for s in L]), axis=1)
        for k in range(2):
            errs.append(_rel(lsec[None, :, k], gsec[:, None, k]))
        ori = 1.0 - _angle_err(np.stack([s.direction for s in G]), np.stack([s.direction for s in L]), signed=False)
    return 1.0 - np.mean(np.stack(errs), axis=0), ori


def _prism(s: Shape):
    """Descriptor of a rectangular prism: (main direction, length, sorted cross-section)."""
    if s.type == "beam":
        return s.direction, s.length, sorted([s.width, s.thickness])
    k = int(np.argmax(s.size))
    e = np.zeros(3); e[k] = 1.0
    return e, float(s.size[k]), sorted(float(x) for i, x in enumerate(s.size) if i != k)


def _prism_score(G: list[Shape], L: list[Shape]):
    dims = np.zeros((len(G), len(L))); ori = np.zeros_like(dims)
    PG, PL = [_prism(s) for s in G], [_prism(s) for s in L]
    for i, (gd, gl, gc) in enumerate(PG):
        for j, (ld, ll, lc) in enumerate(PL):
            ang = np.degrees(np.arccos(np.clip(abs(float(gd @ ld)), -1, 1)))
            e = [min(1.0, abs(ll - gl) / max(gl, 1e-6)),
                 min(1.0, abs(lc[0] - gc[0]) / max(gc[0], 1e-6)),
                 min(1.0, abs(lc[1] - gc[1]) / max(gc[1], 1e-6))]
            dims[i, j] = 1.0 - sum(e) / 3.0
            ori[i, j] = 1.0 - min(1.0, ang / ORIENT_FULL_ERROR_DEG)
    return dims, ori


@dataclass
class Agreement:
    coverage: float                  # 0-100
    geometry: float                  # 0-100, label-strict
    geometry_equiv: float            # 0-100, box == axis-aligned beam
    type_fidelity: float             # % of reference parts matched to the requested primitive type
    position: float                  # mean position credit over reference parts (0-100)
    n_ref: int
    n_out: int
    pairs: list = field(default_factory=list)   # (ref_idx, out_idx, pos_dist, score, ref_type, out_type)


def pair_matrices(ref: list[Shape], out: list[Shape], equivalence: bool):
    D = assembly_diagonal(ref)
    tol = max(POS_TOL_MIN, POS_TOL_FRAC * D)
    Gc = np.stack([s.center for s in ref]); Lc = np.stack([s.center for s in out])
    dist = np.linalg.norm(Gc[:, None, :] - Lc[None, :, :], axis=2)
    pos = np.maximum(0.0, 1.0 - dist / (POS_FALLOFF * tol))
    typ = np.zeros_like(pos); shp = np.zeros_like(pos); ori = np.ones_like(pos)
    gt = np.array([s.type for s in ref]); lt = np.array([s.type for s in out])
    for t in TYPES:
        gi, li = np.where(gt == t)[0], np.where(lt == t)[0]
        if len(gi) and len(li):
            typ[np.ix_(gi, li)] = 1.0
            d_, o_ = _same_type_shape_score(t, [ref[i] for i in gi], [out[j] for j in li])
            shp[np.ix_(gi, li)] = d_; ori[np.ix_(gi, li)] = o_
    if PAIR_FORM == "additive":      # v1-like: orientation averaged into the shape term
        shp_o = np.where(typ > 0, (shp * N_DIMS_PROXY + ori) / (N_DIMS_PROXY + 1), 0.0)
        score = W_POS * pos + W_TYPE * typ + W_SHAPE * typ * shp_o
    else:   # pose gate: type and shape credit only count where (and as) the part is placed
        score = pos * ori * (W_POS + W_TYPE * typ + W_SHAPE * typ * shp)
    if equivalence:   # a box and a beam are the same solid when extents and direction agree; never lowers a pair
        for a, b in (("beam", "box"), ("box", "beam")):
            gi, li = np.where(gt == a)[0], np.where(lt == b)[0]
            if len(gi) and len(li):
                d_, o_ = _prism_score([ref[i] for i in gi], [out[j] for j in li])
                p_ = pos[np.ix_(gi, li)]
                if PAIR_FORM == "additive":
                    alt = W_POS * p_ + W_TYPE + W_SHAPE * (d_ * N_DIMS_PROXY + o_) / (N_DIMS_PROXY + 1)
                else:
                    alt = p_ * o_ * (W_POS + W_TYPE + W_SHAPE * d_)
                score[np.ix_(gi, li)] = np.maximum(score[np.ix_(gi, li)], alt)
    return score, dist, pos, tol


def _assign(score):
    r, c = linear_sum_assignment(-score)
    return r, c


def coverage(n_ref: int, n_out: int) -> float:
    if n_ref == 0:
        return 0.0
    if COVERAGE_FORM == "symmetric":
        return 100.0 * min(n_ref, n_out) / max(n_ref, n_out, 1)
    base = min(1.0, n_out / n_ref)
    ratio = n_out / n_ref
    if ratio > EXCESS_START:
        base *= 1.0 / (1.0 + (ratio - EXCESS_START))
    return 100.0 * base


def agreement(ref: list[Shape], out: list[Shape]) -> Agreement:
    n, m = len(ref), len(out)
    if n == 0 or m == 0:
        return Agreement(coverage(n, m), 0.0, 0.0, 0.0, 0.0, n, m)
    S, dist, pos, _ = pair_matrices(ref, out, equivalence=False)
    r, c = _assign(S)
    geom = float(S[r, c].sum()) / n
    same = sum(1 for i, j in zip(r, c) if ref[i].type == out[j].type)
    pos_mean = float(pos[r, c].sum()) / n
    pairs = [(int(i), int(j), float(dist[i, j]), float(S[i, j]), ref[i].type, out[j].type) for i, j in zip(r, c)]
    Se, *_ = pair_matrices(ref, out, equivalence=True)
    re_, ce = _assign(Se)
    geom_eq = float(Se[re_, ce].sum()) / n
    return Agreement(coverage(n, m), 100 * geom, 100 * geom_eq, 100.0 * same / n, 100 * pos_mean, n, m, pairs)


def assignment_map(ref: list[Shape], out: list[Shape]) -> dict[int, int]:
    """Reference index -> output index under the label-strict optimal assignment."""
    if not ref or not out:
        return {}
    S, *_ = pair_matrices(ref, out, equivalence=False)
    r, c = _assign(S)
    return {int(i): int(j) for i, j in zip(r, c)}
