"""Render assemblies (reference or model output) to images with matplotlib; one colour per primitive type."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .geom import Shape, normalize  # noqa: E402

# Okabe-Ito-like, distinguishable in colour-vision deficiency
COLORS = {"box": "#0072B2", "beam": "#8C6D31", "cylinder": "#E69F00", "pipe": "#CC79A7", "sphere": "#D55E00",
          "cone": "#009E73", "torus": "#56B4E9"}


def _frame(a):
    a = a / np.linalg.norm(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    b = np.cross(a, t); b /= np.linalg.norm(b)
    return b, np.cross(a, b)


def _box_faces(c, R, ex):
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * ex
    V = c + corners @ R.T
    quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [V[list(q)] for q in quads]


def faces(s: Shape, n: int = 20):
    c = s.center
    if s.type == "box":
        return _box_faces(c, np.eye(3), s.size / 2)
    if s.type == "beam":
        u = s.direction
        up = np.array([0, 0, 1.0]) if abs(u[2]) < 0.999 else np.array([1.0, 0, 0])
        v = up - (up @ u) * u; v /= np.linalg.norm(v); w = np.cross(u, v)
        return _box_faces(c, np.stack([u, v, w], 1), np.array([s.length / 2, s.thickness / 2, s.width / 2]))
    if s.type == "sphere":
        th = np.linspace(0, math.pi, n // 2 + 1); ph = np.linspace(0, 2 * math.pi, n + 1)
        P = c + s.radius * np.stack([np.outer(np.sin(th), np.cos(ph)), np.outer(np.sin(th), np.sin(ph)),
                                    np.outer(np.cos(th), np.ones_like(ph))], -1)
        return [np.array([P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]]) for i in range(len(th) - 1) for j in range(n)]
    if s.type == "torus":
        a = s.axis; b1, b2 = _frame(a)
        th = np.linspace(0, 2 * math.pi, n + 1); ph = np.linspace(0, 2 * math.pi, n // 2 + 1)
        P = np.array([[c + (s.ring_radius + s.tube_radius * math.cos(p)) * (math.cos(t) * b1 + math.sin(t) * b2)
                       + s.tube_radius * math.sin(p) * a for p in ph] for t in th])
        return [np.array([P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]]) for i in range(n) for j in range(len(ph) - 1)]
    a = s.axis; b1, b2 = _frame(a); h = s.height
    r0, r1 = {"cylinder": (s.radius, s.radius), "pipe": (s.outer_radius, s.outer_radius),
              "cone": (s.base_radius, s.top_radius)}[s.type]
    ang = np.linspace(0, 2 * math.pi, n + 1)
    ring = lambda r, z: np.array([c + z * a + r * (math.cos(t) * b1 + math.sin(t) * b2) for t in ang])
    lo, hi = ring(r0, -h / 2), ring(r1, h / 2)
    F = [np.array([lo[j], lo[j + 1], hi[j + 1], hi[j]]) for j in range(n)]
    if s.type == "pipe" and s.inner_radius > 0:
        ilo, ihi = ring(s.inner_radius, -h / 2), ring(s.inner_radius, h / 2)
        F += [np.array([ilo[j], ilo[j + 1], ihi[j + 1], ihi[j]]) for j in range(n)]
        F += [np.array([lo[j], lo[j + 1], ilo[j + 1], ilo[j]]) for j in range(n)]
        F += [np.array([hi[j], hi[j + 1], ihi[j + 1], ihi[j]]) for j in range(n)]
    else:
        F += [lo[:-1], hi[:-1]]
    return F


def draw(ax, shapes: list[Shape], alpha=0.85, edge=False, title=None, elev=22, azim=-58, detail=None, bounds=None):
    polys, cols = [], []
    n = detail or (16 if len(shapes) < 150 else 10)
    for s in shapes:
        fs = faces(s, n)
        polys += fs
        cols += [COLORS.get(s.type, "#999999")] * len(fs)
    if polys:
        pc = Poly3DCollection(polys, facecolors=cols, alpha=alpha, linewidths=0.15 if edge else 0,
                              edgecolors="#333333" if edge else None)
        ax.add_collection3d(pc)
        P = np.concatenate([np.asarray(p) for p in polys]) if bounds is None else np.array(bounds)
        lo, hi = P.min(0), P.max(0)
        mid, span = (lo + hi) / 2, max((hi - lo).max(), 1e-6)
        for set_, m in ((ax.set_xlim, mid[0]), (ax.set_ylim, mid[1]), (ax.set_zlim, mid[2])):
            set_(m - span / 2, m + span / 2)
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    if title:
        ax.set_title(title, fontsize=8)


def bounds_of(shapes: list[Shape]):
    los, his = zip(*(s.bbox() for s in shapes)) if shapes else ((np.zeros(3),), (np.ones(3),))
    return [np.min(np.stack(los), 0), np.max(np.stack(his), 0)]


def render_file(parts_json, path, title=None, size=4.0, **kw):
    shapes, _ = normalize(parts_json)
    fig = plt.figure(figsize=(size, size), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, shapes, title=title, **kw)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def legend_handles():
    from matplotlib.patches import Patch
    return [Patch(color=c, label=t) for t, c in COLORS.items()]
