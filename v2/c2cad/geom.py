"""Shape model for C2CAD-Bench v2.

One normalizer is applied to BOTH the reference and the model output (v1 only
normalized the model side, which is why some references did not score 100
against themselves).

Canonical primitives (all lengths in mm):
  box      center, size [sx, sy, sz]  (axis-aligned, world X/Y/Z)
  cylinder center, axis, radius, height
  pipe     center, axis, inner_radius, outer_radius, height
  cone     center, axis (base -> tip), base_radius, top_radius, height
  sphere   center, radius
  torus    center, axis, ring_radius, tube_radius
  beam     start, end, width, height (square section if height missing)

`center` of axial primitives is the midpoint of the axis segment between the two
end faces (for a cone this is NOT the volume centroid; the v2 schema text says so).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

TYPES = ("box", "cylinder", "pipe", "cone", "sphere", "torus", "beam")

TYPE_ALIASES = {
    "box": "box", "cube": "box", "cuboid": "box", "rectangular_prism": "box", "block": "box",
    "cylinder": "cylinder", "cyl": "cylinder", "rod_cylinder": "cylinder",
    "sphere": "sphere", "ball": "sphere",
    "beam": "beam", "bar": "beam", "rod": "beam", "strut": "beam",
    "cone": "cone", "frustum": "cone",
    "torus": "torus", "ring": "torus",
    "pipe": "pipe", "tube": "pipe", "hollow_cylinder": "pipe",
}

_AXIS_WORDS = {
    "x": (1, 0, 0), "+x": (1, 0, 0), "-x": (-1, 0, 0),
    "y": (0, 1, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
    "z": (0, 0, 1), "+z": (0, 0, 1), "-z": (0, 0, -1),
}


def _f(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _vec(v) -> Optional[np.ndarray]:
    if isinstance(v, str):
        w = _AXIS_WORDS.get(v.strip().lower())
        return np.array(w, float) if w else None
    if isinstance(v, dict):
        if all(k in v for k in ("x", "y", "z")):
            v = [v["x"], v["y"], v["z"]]
        else:
            return None
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        return None
    xs = [_f(x) for x in v[:3]]
    if any(x is None for x in xs):
        return None
    return np.array(xs, float)


def _first(d: dict, keys) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _euler_xyz_deg(e: np.ndarray) -> np.ndarray:
    """Rotation matrix for Euler angles in degrees applied about world X, then Y, then Z
    (three.js default order). Single-axis rotations, the common case, are convention-free."""
    x, y, z = np.radians(e)
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


@dataclass
class Shape:
    type: str
    center: np.ndarray
    id: Any = None
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    radius: float = 0.0          # cylinder, sphere
    height: float = 0.0          # cylinder, pipe, cone (axial length)
    inner_radius: float = 0.0    # pipe
    outer_radius: float = 0.0    # pipe
    base_radius: float = 0.0     # cone
    top_radius: float = 0.0      # cone
    ring_radius: float = 0.0     # torus
    tube_radius: float = 0.0     # torus
    size: np.ndarray = field(default_factory=lambda: np.zeros(3))  # box
    start: Optional[np.ndarray] = None   # beam
    end: Optional[np.ndarray] = None     # beam
    width: float = 0.0           # beam
    thickness: float = 0.0       # beam (section height)
    symbolic: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    # ---- derived geometry -------------------------------------------------
    @property
    def length(self) -> float:
        if self.type == "beam":
            return float(np.linalg.norm(self.end - self.start))
        return self.height

    @property
    def direction(self) -> np.ndarray:
        """Unit main direction (beam start->end, axial primitives' axis, box longest side)."""
        if self.type == "beam":
            return (self.end - self.start) / max(self.length, 1e-12)
        if self.type == "box":
            e = np.zeros(3); e[int(np.argmax(self.size))] = 1.0
            return e
        return self.axis

    def end_centers(self):
        """(p0, p1) centers of the two end faces along the axis (cone: base, tip)."""
        if self.type == "beam":
            return self.start, self.end
        h = self.height / 2.0
        return self.center - self.axis * h, self.center + self.axis * h

    def bbox(self):
        """Axis-aligned bounding box (min, max)."""
        c = self.center
        if self.type == "box":
            h = self.size / 2.0
            return c - h, c + h
        if self.type == "sphere":
            return c - self.radius, c + self.radius
        if self.type == "beam":
            half = max(self.width, self.thickness) / 2.0
            lo = np.minimum(self.start, self.end) - half
            hi = np.maximum(self.start, self.end) + half
            return lo, hi
        if self.type == "torus":
            a = self.axis
            ext = self.ring_radius * np.sqrt(np.clip(1 - a * a, 0, 1)) + self.tube_radius
            return c - ext, c + ext
        # axial: cylinder / pipe / cone -> disc extents at both ends
        r = {"cylinder": self.radius, "pipe": self.outer_radius,
             "cone": max(self.base_radius, self.top_radius)}[self.type]
        p0, p1 = self.end_centers()
        a = self.axis
        ext = r * np.sqrt(np.clip(1 - a * a, 0, 1))
        return np.minimum(p0, p1) - ext, np.maximum(p0, p1) + ext

    def to_json(self) -> dict:
        d: dict = {"id": self.id, "type": self.type}
        if self.type == "beam":
            d.update(start=self.start.tolist(), end=self.end.tolist(), width=self.width, height=self.thickness)
        else:
            d["center"] = self.center.tolist()
        if self.type == "box":
            d["size"] = self.size.tolist()
        elif self.type == "sphere":
            d["radius"] = self.radius
        elif self.type == "cylinder":
            d.update(radius=self.radius, height=self.height, axis=self.axis.tolist())
        elif self.type == "pipe":
            d.update(inner_radius=self.inner_radius, outer_radius=self.outer_radius,
                     height=self.height, axis=self.axis.tolist())
        elif self.type == "cone":
            d.update(base_radius=self.base_radius, top_radius=self.top_radius,
                     height=self.height, axis=self.axis.tolist())
        elif self.type == "torus":
            d.update(ring_radius=self.ring_radius, tube_radius=self.tube_radius, axis=self.axis.tolist())
        if self.symbolic:
            d["symbolic"] = True
        return d


@dataclass
class NormalizeReport:
    n_input: int = 0
    n_kept: int = 0
    dropped_not_object: int = 0
    dropped_unknown_type: int = 0
    dropped_degenerate: int = 0
    unknown_types: dict = field(default_factory=dict)
    ops_records: int = 0
    axis_from_euler_rotation: int = 0     # orientation given as Euler angles instead of an axis
    box_rotation_ignored: int = 0         # boxes are axis-aligned by schema; a rotation field is ignored
    beam_section_missing: int = 0         # beams without width/height: kept, section scored as wrong
    prisms_resolved: int = 0              # label-free CAD prisms read as box/beam (cadquery arm; see prism.py)


def normalize_shape(s: Any) -> tuple[Optional[Shape], str]:
    """Return (Shape or None, reason). reason in {"ok","not_object","op","unknown_type","degenerate"}."""
    if not isinstance(s, dict):
        return None, "not_object"
    if "op" in s and "type" not in s:
        return None, "op"
    raw_t = str(s.get("type", "")).strip().lower()
    t = TYPE_ALIASES.get(raw_t)
    if t is None:
        t = _infer_type(s) if not raw_t else None
    if t is None:
        return None, "unknown_type"

    sid = s.get("id")
    sym = bool(s.get("symbolic", False))
    ax = _vec(_first(s, ("axis", "direction", "dir", "normal")))
    ax = _unit(ax) if ax is not None else None
    rot_used = False
    if ax is None:
        eul = _vec(_first(s, ("rotation", "rotation_deg", "euler", "euler_deg", "rot")))
        ori = s.get("orientation")
        ori_v = _vec(ori) if isinstance(ori, (list, tuple)) else None
        if eul is None and ori_v is not None:
            # "orientation" is ambiguous: a unit vector is read as a direction, anything else as Euler degrees
            if abs(float(np.linalg.norm(ori_v)) - 1.0) < 1e-3:
                ax = ori_v
            else:
                eul = ori_v
        if ax is None and eul is not None:
            ax = _euler_xyz_deg(eul) @ np.array([0.0, 0.0, 1.0]); rot_used = True
    if ax is None:
        ax = np.array([0.0, 0.0, 1.0])

    if t == "beam":
        st = _vec(_first(s, ("start", "from", "from_point", "p1", "point1", "start_point")))
        en = _vec(_first(s, ("end", "to", "to_point", "p2", "point2", "end_point")))
        if st is None or en is None or np.linalg.norm(en - st) <= 1e-9:
            return None, "degenerate"
        w = _f(_first(s, ("width", "w", "thickness", "section", "size_w")))
        h = _f(_first(s, ("height", "h", "depth", "thickness")))
        if w is None and h is None:
            sec = _f(s.get("section_size"))
            w = h = sec
        if w is None:
            w = h
        if h is None:
            h = w
        if w is None:      # section not given: keep the centerline (placement is still scorable);
            return Shape("beam", (st + en) / 2.0, sid, axis=_unit(en - st), start=st, end=en,   # dims score as wrong
                         width=0.0, thickness=0.0, symbolic=sym, raw=s), "ok_no_section"
        if w <= 0 or h <= 0:
            return None, "degenerate"
        return Shape("beam", (st + en) / 2.0, sid, axis=_unit(en - st), start=st, end=en,
                     width=w, thickness=h, symbolic=sym, raw=s), "ok"

    c = _vec(_first(s, ("center", "centre", "position", "pos", "origin", "location")))
    if c is None:
        cx, cy, cz = (_f(s.get(k)) for k in ("x", "y", "z"))
        if None not in (cx, cy, cz):
            c = np.array([cx, cy, cz])
    if c is None:
        return None, "degenerate"

    if t == "box":
        sz = s.get("size", _first(s, ("dimensions", "dims", "dim", "extents")))
        v = _vec(sz) if not isinstance(sz, dict) else None
        if isinstance(sz, dict):
            v = _vec([sz.get("x", sz.get("width")), sz.get("y", sz.get("depth")), sz.get("z", sz.get("height"))])
        if v is None:
            w, d, h = (_f(s.get(k)) for k in ("width", "depth", "height"))
            if None not in (w, d, h):
                v = np.array([w, d, h])
        if v is None or np.any(np.abs(v) <= 1e-9):
            return None, "degenerate"
        rotated = _vec(_first(s, ("rotation", "rotation_deg", "euler", "orientation")) if not isinstance(s.get("orientation"), dict) else None)
        why = "ok_box_rotation_ignored" if (rotated is not None and np.any(np.abs(rotated) > 1e-9)) or isinstance(s.get("orientation"), dict) else "ok"
        return Shape("box", c, sid, size=np.abs(v), symbolic=sym, raw=s), why

    if t == "sphere":
        r = _f(_first(s, ("radius", "r")))
        if r is None and _f(s.get("diameter")):
            r = _f(s["diameter"]) / 2
        if not r or r <= 0:
            return None, "degenerate"
        return Shape("sphere", c, sid, radius=r, symbolic=sym, raw=s), "ok"

    if t == "cylinder":
        r = _f(_first(s, ("radius", "r")))
        h = _f(_first(s, ("height", "length", "h", "len")))
        if not r or not h or r <= 0 or h <= 0:
            return None, "degenerate"
        return Shape("cylinder", c, sid, axis=ax, radius=r, height=h, symbolic=sym, raw=s), "ok_rot" if rot_used else "ok"

    if t == "pipe":
        ri = _f(_first(s, ("inner_radius", "radius_inner", "ri", "bore_radius")))
        ro = _f(_first(s, ("outer_radius", "radius_outer", "ro", "radius")))
        h = _f(_first(s, ("height", "length", "h", "len")))
        if ri is None:
            ri = 0.0
        if not ro or not h or ro <= 0 or h <= 0 or ri < 0 or ri >= ro:
            return None, "degenerate"
        return Shape("pipe", c, sid, axis=ax, inner_radius=ri, outer_radius=ro, height=h,
                     symbolic=sym, raw=s), "ok_rot" if rot_used else "ok"

    if t == "cone":
        rb = _f(_first(s, ("base_radius", "bottom_radius", "radius_bottom", "radius_base",
                           "start_radius", "radius", "r1")))
        rt = _f(_first(s, ("top_radius", "radius_top", "end_radius", "tip_radius", "r2")))
        h = _f(_first(s, ("height", "length", "h", "len")))
        if rt is None:
            rt = 0.0
        if not rb or not h or rb <= 0 or h <= 0 or rt < 0:
            return None, "degenerate"
        return Shape("cone", c, sid, axis=ax, base_radius=rb, top_radius=rt, height=h,
                     symbolic=sym, raw=s), "ok_rot" if rot_used else "ok"

    if t == "torus":
        R = _f(_first(s, ("ring_radius", "major_radius", "R", "large_radius", "radius")))
        r = _f(_first(s, ("tube_radius", "minor_radius", "r", "small_radius", "cross_radius")))
        if not R or not r or R <= 0 or r <= 0:
            return None, "degenerate"
        return Shape("torus", c, sid, axis=ax, ring_radius=R, tube_radius=r, symbolic=sym, raw=s), "ok_rot" if rot_used else "ok"

    return None, "unknown_type"


def _infer_type(s: dict) -> Optional[str]:
    if "start" in s and "end" in s:
        return "beam"
    if "inner_radius" in s and "outer_radius" in s:
        return "pipe"
    if "ring_radius" in s or "tube_radius" in s:
        return "torus"
    if "base_radius" in s or "top_radius" in s:
        return "cone"
    if "radius" in s and ("height" in s or "length" in s):
        return "cylinder"
    if "radius" in s:
        return "sphere"
    if "size" in s or "dimensions" in s:
        return "box"
    return None


def unwrap(parsed: Any) -> list:
    """Accept a bare array or an object with a single list-valued key such as {"shapes": [...]}."""
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for k in ("shapes", "primitives", "parts", "assembly", "components", "objects"):
            if isinstance(parsed.get(k), list):
                return parsed[k]
        lists = [v for v in parsed.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return []


def normalize(shapes: Any) -> tuple[list[Shape], NormalizeReport]:
    rep = NormalizeReport()
    items = unwrap(shapes)
    rep.n_input = len(items)
    out: list[Shape] = []
    for s in items:
        sh, why = normalize_shape(s)
        if sh is not None:
            out.append(sh)
            rep.axis_from_euler_rotation += why == "ok_rot"
            rep.box_rotation_ignored += why == "ok_box_rotation_ignored"
            rep.beam_section_missing += why == "ok_no_section"
        elif why == "not_object":
            rep.dropped_not_object += 1
        elif why == "op":
            rep.ops_records += 1
        elif why == "unknown_type":
            rep.dropped_unknown_type += 1
            t = str(s.get("type", "")).lower() if isinstance(s, dict) else "?"
            rep.unknown_types[t] = rep.unknown_types.get(t, 0) + 1
        else:
            rep.dropped_degenerate += 1
    rep.n_kept = len(out)
    return out, rep


def assembly_diagonal(shapes: list[Shape]) -> float:
    """Diagonal of the bounding box of all part bounding boxes (v1 used centers only)."""
    if not shapes:
        return 1.0
    los, his = zip(*(s.bbox() for s in shapes))
    lo = np.min(np.stack(los), axis=0)
    hi = np.max(np.stack(his), axis=0)
    return max(float(np.linalg.norm(hi - lo)), 1.0)
