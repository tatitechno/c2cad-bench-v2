"""CAD-kernel bridge (CadQuery 2.x / OpenCascade) for the `cadquery` arm and for kernel-build checks (E08).

Runs only inside the CAD environment (v2/.venv-cad); nothing else in the package imports it.

  to_solid(part)      mechanical converter: one v2 JSON primitive -> one OpenCascade solid (one constructor per type)
  to_cadquery_code()  the same conversion emitted as a plain CadQuery program (used by tests and the mock provider)
  recover(solid)      the primitive a solid is, read from its B-Rep faces (surface types and their parameters);
                      None if the solid is not one of the seven primitives (e.g. a union of several parts)
  recover_all(obj)    every solid in whatever the program produced ({id: shape}, [(id, shape)], [shape], Workplane,
                      Assembly), with counts of invalid and unrecognised solids

A rectangular solid built in CAD carries neither a box/beam label nor a designated centerline, so recover()
returns it as {"type": "prism", "center", "axes", "extents"}; the evaluator picks the reading (axis-aligned box,
or beam along one of its three axes) that fits best (see c2cad/prism.py). The occupied volume is the same for
every reading.
"""
from __future__ import annotations

import json
import math

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface

from .runner.cad_marker import MARK  # noqa: E402  (plain constant, no CAD imports)
_ANG_TOL = 1e-4          # radians, for parallel / perpendicular tests on recovered axes
_REL_TOL = 1e-6


def _v(x) -> cq.Vector:
    return cq.Vector(float(x[0]), float(x[1]), float(x[2]))


def _unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


from .cadcode import beam_frame, to_cadquery_code  # noqa: E402,F401  (CAD-free; re-exported)


# ---------------------------------------------------------------------------
# JSON -> solid
# ---------------------------------------------------------------------------
def to_solid(p: dict) -> cq.Solid:
    t = p["type"]
    if t == "box":
        c, s = np.asarray(p["center"], float), np.asarray(p["size"], float)
        return cq.Solid.makeBox(*s, pnt=_v(c - s / 2))
    if t == "sphere":
        return cq.Solid.makeSphere(float(p["radius"]), pnt=_v(p["center"]), angleDegrees1=-90, angleDegrees2=90)
    if t == "torus":
        return cq.Solid.makeTorus(float(p["ring_radius"]), float(p["tube_radius"]), pnt=_v(p["center"]),
                                  dir=_v(_unit(p["axis"])))
    if t in ("cylinder", "pipe", "cone"):
        a, h = _unit(p["axis"]), float(p["height"])
        base = _v(np.asarray(p["center"], float) - a * h / 2)
        if t == "cylinder":
            return cq.Solid.makeCylinder(float(p["radius"]), h, base, _v(a))
        if t == "cone":
            return cq.Solid.makeCone(float(p["base_radius"]), float(p["top_radius"]), h, base, _v(a))
        outer = cq.Solid.makeCylinder(float(p["outer_radius"]), h, base, _v(a))
        inner = cq.Solid.makeCylinder(float(p["inner_radius"]), h, base, _v(a))
        return outer.cut(inner).Solids()[0]
    if t == "beam":
        s, e = np.asarray(p["start"], float), np.asarray(p["end"], float)
        u, v, _ = beam_frame(s, e)
        L = float(np.linalg.norm(e - s))
        w, h = float(p.get("width", 0)), float(p.get("height", p.get("width", 0)))
        plane = cq.Plane(origin=_v((s + e) / 2), xDir=_v(u), normal=_v(v))
        return cq.Workplane(plane).box(L, w, h).val()
    raise ValueError(f"unknown type {t!r}")


# ---------------------------------------------------------------------------
# solid -> primitive
# ---------------------------------------------------------------------------
def _pt(p):
    return np.array([p.X(), p.Y(), p.Z()], float)


def _ax(ax):
    return _pt(ax.Location()), np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()], float)


def _canon(d):
    """Sign-normalise a line direction: largest-magnitude component positive."""
    d = _unit(d)
    return d if d[int(np.argmax(np.abs(d)))] > 0 else -d


def _parallel(a, b) -> bool:
    return abs(abs(float(np.dot(_unit(a), _unit(b)))) - 1.0) < _ANG_TOL


def _on_line(p, loc, d, scale) -> bool:
    w = p - loc
    return np.linalg.norm(w - np.dot(w, d) * d) <= 1e-6 * max(1.0, scale)


def _verts(shape):
    return np.array([[v.X, v.Y, v.Z] for v in shape.Vertices()], float)


def recover(solid) -> dict | None:
    faces = solid.Faces()
    kinds = [f.geomType() for f in faces]
    ad = [BRepAdaptor_Surface(f.wrapped) for f in faces]
    V = _verts(solid)
    if not faces:
        return None
    scale = float(np.ptp(V, axis=0).max()) if len(V) else 1.0
    if all(k == "SPHERE" for k in kinds):
        sp = [a.Sphere() for a in ad]
        c, r = _pt(sp[0].Location()), sp[0].Radius()
        if all(np.allclose(_pt(s.Location()), c, atol=1e-6 * max(1, r)) and abs(s.Radius() - r) < 1e-6 * max(1, r)
               for s in sp):
            return {"type": "sphere", "center": c.tolist(), "radius": float(r)}
        return None
    if all(k == "TORUS" for k in kinds):
        to = [a.Torus() for a in ad]
        loc, d = _ax(to[0].Axis())
        return {"type": "torus", "center": loc.tolist(), "axis": _canon(d).tolist(),
                "ring_radius": float(to[0].MajorRadius()), "tube_radius": float(to[0].MinorRadius())}
    curved = [i for i, k in enumerate(kinds) if k != "PLANE"]
    planes = [i for i, k in enumerate(kinds) if k == "PLANE"]
    if curved and all(kinds[i] in ("CYLINDER", "CONE") for i in curved):
        geo = [ad[i].Cylinder() if kinds[i] == "CYLINDER" else ad[i].Cone() for i in curved]
        loc, d = _ax(geo[0].Axis())
        d = _unit(d)
        for g in geo:
            l2, d2 = _ax(g.Axis())
            if not (_parallel(d, d2) and _on_line(l2, loc, d, scale)):
                return None
        for i in planes:     # end faces must be perpendicular to the axis
            if not _parallel(_ax(ad[i].Plane().Axis())[1], d):
                return None
        t = (V - loc) @ d
        t0, t1 = float(t.min()), float(t.max())
        h = t1 - t0
        if h <= 0:
            return None
        center = loc + d * (t0 + t1) / 2
        if any(kinds[i] == "CONE" for i in curved):
            if len(curved) != 1:
                return None
            cone = geo[0]
            k = math.tan(cone.SemiAngle())
            r0, r1 = cone.RefRadius() + k * t0, cone.RefRadius() + k * t1
            r0, r1 = max(0.0, r0), max(0.0, r1)
            axis = d if r0 >= r1 else -d          # the axis points from the base (larger) face to the tip face
            return {"type": "cone", "center": center.tolist(), "axis": axis.tolist(), "base_radius": float(max(r0, r1)),
                    "top_radius": float(min(r0, r1)), "height": float(h)}
        radii = sorted({round(g.Radius(), 9) for g in geo})
        if len(radii) == 1:
            return {"type": "cylinder", "center": center.tolist(), "axis": _canon(d).tolist(), "radius": float(radii[0]),
                    "height": float(h)}
        if len(radii) == 2:
            return {"type": "pipe", "center": center.tolist(), "axis": _canon(d).tolist(), "inner_radius": float(radii[0]),
                    "outer_radius": float(radii[1]), "height": float(h)}
        return None
    if len(faces) == 6 and all(k == "PLANE" for k in kinds):
        normals = [_canon(_ax(a.Plane().Axis())[1]) for a in ad]
        axes = []
        for n in normals:
            if not any(_parallel(n, a) for a in axes):
                axes.append(n)
        if len(axes) != 3 or any(abs(float(axes[i] @ axes[j])) > _ANG_TOL for i in range(3) for j in range(i + 1, 3)):
            return None
        if sum(1 for n in normals for a in axes if _parallel(n, a)) != 6 or len(V) != 8:
            return None
        A = np.stack(axes)
        proj = V @ A.T
        lo, hi = proj.min(0), proj.max(0)
        center = A.T @ ((lo + hi) / 2)
        return {"type": "prism", "center": center.tolist(), "axes": A.tolist(), "extents": (hi - lo).tolist()}
    return None


def _solids(obj):
    if isinstance(obj, cq.Assembly):
        obj = obj.toCompound()
    if isinstance(obj, cq.Workplane):
        out = []
        for v in obj.vals():
            out += _solids(v)
        return out
    if isinstance(obj, cq.Shape):
        return list(obj.Solids())
    return []


def _items(obj):
    if isinstance(obj, dict):
        return list(obj.items())
    if isinstance(obj, (list, tuple)):
        out = []
        for i, x in enumerate(obj):
            if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], (int, float, str)):
                out.append((x[0], x[1]))
            else:
                out.append((i, x))
        return out
    return [(None, obj)]


def recover_all(obj) -> dict:
    parts, stats = [], {"n_items": 0, "n_solids": 0, "n_invalid": 0, "n_unrecognized": 0, "n_prisms": 0,
                        "n_multi_solid_items": 0}
    for pid, shape in _items(obj):
        stats["n_items"] += 1
        sols = _solids(shape)
        if len(sols) > 1:
            stats["n_multi_solid_items"] += 1
        for s in sols:
            stats["n_solids"] += 1
            if not s.isValid():
                stats["n_invalid"] += 1
            try:
                p = recover(s)
            except Exception:       # a kernel query failing on one solid must not lose the others
                p = None
            if p is None:
                stats["n_unrecognized"] += 1
                continue
            if p["type"] == "prism":
                stats["n_prisms"] += 1
            try:
                p["id"] = int(pid) if pid is not None else len(parts)
            except (TypeError, ValueError):
                p["id"] = len(parts)
            parts.append(p)
    return {"parts": parts, "stats": stats}


def emit(obj) -> None:
    print(MARK + json.dumps(recover_all(obj)))
