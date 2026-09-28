"""Shape-level 3D metrics used for convergent validity (E05): volumetric IoU, symmetric Chamfer
distance, F-score, and orientation error over matched parts. Pure numpy/scipy, no CAD kernel.

Conventions follow geom.py. A beam's cross-section 'height' axis is world Z projected
perpendicular to the beam (world X if the beam is vertical); 'width' is the remaining axis.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .geom import Shape, assembly_diagonal


def _beam_frame(s: Shape):
    u = s.direction
    up = np.array([0.0, 0.0, 1.0])
    if abs(u @ up) > 0.999:
        up = np.array([1.0, 0.0, 0.0])
    v = up - (up @ u) * u; v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return u, v, w


def _perp(a):
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    b = np.cross(a, t); b /= np.linalg.norm(b)
    return b, np.cross(a, b)


def inside(s: Shape, P: np.ndarray) -> np.ndarray:
    q = P - s.center
    if s.type == "box":
        return np.all(np.abs(q) <= s.size / 2, axis=1)
    if s.type == "sphere":
        return np.einsum("ij,ij->i", q, q) <= s.radius ** 2
    if s.type == "beam":
        u, v, w = _beam_frame(s)
        return (np.abs(q @ u) <= s.length / 2) & (np.abs(q @ v) <= s.thickness / 2) & (np.abs(q @ w) <= s.width / 2)
    a = s.axis
    t = q @ a
    rad = np.linalg.norm(q - np.outer(t, a), axis=1)
    if s.type == "torus":
        return (rad - s.ring_radius) ** 2 + t ** 2 <= s.tube_radius ** 2
    ax_ok = np.abs(t) <= s.height / 2
    if s.type == "cylinder":
        return ax_ok & (rad <= s.radius)
    if s.type == "pipe":
        return ax_ok & (rad <= s.outer_radius) & (rad >= s.inner_radius)
    if s.type == "cone":
        rt = s.base_radius + (s.top_radius - s.base_radius) * (t + s.height / 2) / s.height
        return ax_ok & (rad <= rt)
    return np.zeros(len(P), bool)


def occupancy(shapes: list[Shape], lo, hi, res: int) -> np.ndarray:
    step = (hi - lo) / res
    grid = np.zeros((res, res, res), bool)
    for s in shapes:
        blo, bhi = s.bbox()
        i0 = np.clip(np.floor((blo - lo) / step).astype(int), 0, res - 1)
        i1 = np.clip(np.ceil((bhi - lo) / step).astype(int), 0, res)
        if np.any(i1 <= i0):
            continue
        axes_ = [lo[k] + (np.arange(i0[k], i1[k]) + 0.5) * step[k] for k in range(3)]
        X, Y, Z = np.meshgrid(*axes_, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        m = inside(s, P).reshape(X.shape)
        grid[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= m
    return grid


def iou(ref: list[Shape], out: list[Shape], res: int = 96) -> float:
    if not ref or not out:
        return 0.0
    los, his = zip(*(s.bbox() for s in ref + out))
    lo = np.min(np.stack(los), 0); hi = np.max(np.stack(his), 0)
    span = np.maximum(hi - lo, 1e-6)
    cube = span.max()
    hi = lo + cube                      # cubic voxels
    a = occupancy(ref, lo, hi, res); b = occupancy(out, lo, hi, res)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _area_and_sampler(s: Shape):
    """(area, sampler(n, rng) -> (n,3) points on the surface)."""
    c = s.center
    if s.type == "sphere":
        r = s.radius
        def f(n, g):
            v = g.normal(size=(n, 3)); return c + r * v / np.linalg.norm(v, axis=1, keepdims=True)
        return 4 * math.pi * r * r, f
    if s.type in ("box", "beam"):
        if s.type == "box":
            ex = s.size / 2; R = np.eye(3)
        else:
            u, v, w = _beam_frame(s); R = np.stack([u, v, w], 1); ex = np.array([s.length / 2, s.thickness / 2, s.width / 2])
        faces = []
        for k in range(3):
            i, j = [m for m in range(3) if m != k]
            area = 4 * ex[i] * ex[j]
            faces += [(k, +1, i, j, area), (k, -1, i, j, area)]
        A = np.array([fc[4] for fc in faces])
        def f(n, g):
            idx = g.choice(len(faces), size=n, p=A / A.sum())
            L = g.uniform(-1, 1, size=(n, 3)) * ex
            for m, (k, sg, i, j, _) in enumerate(faces):
                sel = idx == m
                L[sel, k] = sg * ex[k]
            return c + L @ R.T
        return float(A.sum()), f
    if s.type == "torus":
        Rr, r = s.ring_radius, s.tube_radius; a = s.axis; b1, b2 = _perp(a)
        def f(n, g):
            out = []
            while sum(len(o) for o in out) < n:
                th = g.uniform(0, 2 * math.pi, 2 * n); ph = g.uniform(0, 2 * math.pi, 2 * n)
                keep = g.uniform(0, 1, 2 * n) < (Rr + r * np.cos(ph)) / (Rr + r)
                th, ph = th[keep], ph[keep]
                radial = np.outer(np.cos(th), b1) + np.outer(np.sin(th), b2)
                out.append(c + (Rr + r * np.cos(ph))[:, None] * radial + (r * np.sin(ph))[:, None] * a)
            return np.concatenate(out)[:n]
        return 4 * math.pi ** 2 * Rr * r, f
    # axial: cylinder, pipe, cone
    a = s.axis; b1, b2 = _perp(a); h = s.height
    if s.type == "cylinder":
        r0 = r1 = s.radius; ri = 0.0
    elif s.type == "pipe":
        r0 = r1 = s.outer_radius; ri = s.inner_radius
    else:
        r0, r1, ri = s.base_radius, s.top_radius, 0.0
    slant = math.hypot(h, r0 - r1)
    parts = [("side", math.pi * (r0 + r1) * slant), ("cap0", math.pi * (r0 ** 2 - ri ** 2)), ("cap1", math.pi * (r1 ** 2 - ri ** 2))]
    if ri > 0:
        parts.append(("inner", 2 * math.pi * ri * h))
    A = np.array([p[1] for p in parts])
    def f(n, g):
        idx = g.choice(len(parts), size=n, p=A / A.sum())
        th = g.uniform(0, 2 * math.pi, n)
        radial = np.outer(np.cos(th), b1) + np.outer(np.sin(th), b2)
        t = g.uniform(0, 1, n)
        z = np.empty(n); rr = np.empty(n)
        for m, (nm, _) in enumerate(parts):
            sel = idx == m
            if nm == "side":
                z[sel] = -h / 2 + h * t[sel]; rr[sel] = r0 + (r1 - r0) * t[sel]
            elif nm == "inner":
                z[sel] = -h / 2 + h * t[sel]; rr[sel] = ri
            else:
                R_ = r0 if nm == "cap0" else r1
                z[sel] = -h / 2 if nm == "cap0" else h / 2
                rr[sel] = np.sqrt(ri ** 2 + t[sel] * (R_ ** 2 - ri ** 2))
        return c + z[:, None] * a + rr[:, None] * radial
    return float(A.sum()), f


def sample_surface(shapes: list[Shape], n: int = 20000, seed: int = 0) -> np.ndarray:
    g = np.random.default_rng(seed)
    info = [_area_and_sampler(s) for s in shapes]
    A = np.array([i[0] for i in info])
    if not np.isfinite(A.sum()) or A.sum() <= 0:          # e.g. only sectionless beams: no surface to sample
        return np.zeros((0, 3))
    A = A / A.sum()
    counts = g.multinomial(n, A)
    pts = [f(k, g) for (_, f), k in zip(info, counts) if k > 0]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def chamfer_fscore(ref: list[Shape], out: list[Shape], n: int = 20000, thresholds=(0.01, 0.02)):
    """Symmetric mean nearest-neighbour distance / reference diagonal, and F-scores at fractions of the diagonal."""
    if not ref or not out:
        return float("nan"), {t: 0.0 for t in thresholds}
    D = assembly_diagonal(ref)
    A, B = sample_surface(ref, n, 1), sample_surface(out, n, 2)
    if len(A) == 0 or len(B) == 0:
        return float("nan"), {t: 0.0 for t in thresholds}
    da, _ = cKDTree(B).query(A); db, _ = cKDTree(A).query(B)
    ch = (da.mean() + db.mean()) / 2 / D
    fs = {}
    for t in thresholds:
        p = (db <= t * D).mean(); r = (da <= t * D).mean()
        fs[t] = float(2 * p * r / (p + r)) if p + r > 0 else 0.0
    return float(ch), fs


def orientation_error(ref: list[Shape], out: list[Shape], amap: dict) -> float:
    """Mean line angle (deg) between main directions of matched parts that have one (spheres excluded)."""
    errs = []
    for i, j in amap.items():
        a, b = ref[i], out[j]
        if a.type == "sphere" or b.type == "sphere":
            continue
        d = abs(float(a.direction @ b.direction))
        errs.append(math.degrees(math.acos(min(1.0, d))))
    return float(np.mean(errs)) if errs else float("nan")
