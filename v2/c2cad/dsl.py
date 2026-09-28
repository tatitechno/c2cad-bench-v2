"""CML: a small CAD-style mates-and-patterns language, and its deterministic interpreter.

The model declares parts, how they attach to features of other parts, and patterns; the
interpreter computes every coordinate, in program order, with closed-form geometry only.
There are no free-form expressions: numbers are literals or references to declared dimensions.
A program that references an undefined part, uses a feature a primitive does not have, or
leaves a part unplaced raises CMLError (reported as an invalid program, not silently repaired).

Reference (also given to models in the mates arm prompt):

Program   {"steps": [Part | Pattern, ...]}     processed in order
Part      {"id": int, "type": T, <dims>, <placement>}
  dims    box: size [sx,sy,sz]   sphere: radius   cylinder: radius, height   pipe: inner_radius, outer_radius, height
          cone: base_radius, top_radius, height   torus: ring_radius, tube_radius   beam: width, height (+ length if "direction" is used)
          any dim may be a Dist
  placement (non-beam)  "axis": Dir (cylinder/pipe/cone/torus; default "+z"; a cone's axis points base -> tip)
                        and one of  "at": Point (center) | "bottom_at": Point | "top_at": Point
                        | (box only) "align": {"x": [W, Point], "y": [W, Point], "z": [W, Point]} with W in "center" | "+" | "-":
                          on each world axis put the box's center, its + face or its - face at that coordinate of Point
                        optional   "top_through": Point  (with "bottom_at": height = distance to the plane through Point)
                        optional   "rest_on": Plane      (slide along the plane normal until the part touches the plane from above)
  placement (beam)      "start": Point and ("end": Point | "direction": Dir with "length": Dist); optional "rest_on": Plane
Pattern   {"pattern": "circular" | "helical" | "linear" | "mirror", "seed": [ids], "count": n (copies including the seed; mirror: 2),
           "first_id": int (id of the first new part; copies are numbered consecutively, seed order preserved),
           circular/helical: "about": Axis and one of "step_deg" | "per_revolution" | "full_circle": true; helical also "rise": Dist,
           linear: "direction": Dir, "step": Dist, "centered": bool? (shift the whole row so it is centered on the seed's
           original position);   mirror: "plane": Plane}
Point     "origin" | [x,y,z] | {"of": id, "feature": F, "deg": number?, "t": number?} | {"intersect": [Axis, Plane]}
          | {"from": Point, "then": [{"along": Dir, "by": Dist}, ...]}   (every form also accepts "then")
          | {"xyz": [Point, Point, Point]}  (x from the first, y from the second, z from the third)
          | {"between": [Point, Point], "t": number}  (t = 0 at the first point, 1 at the second)
          | {"nest": {"touch": [sphere ids], "rest_on": Plane?, "side": Dir}}  (center of a sphere of this part's radius that
            touches the listed spheres (and rests on the plane); of the two solutions take the one farther along "side")
Feature F  all: center.  cylinder/pipe/cone: bottom, top (end-face centers; cone bottom = base), surface (outer lateral surface at
           azimuth "deg" about the axis, axial offset "t" from the center).  box: +x -x +y -y +z -z, top (=+z), bottom (=-z).
           sphere: top, bottom.  beam: start, end, top_face, bottom_face (centerline midpoint +- half the section height, "up" =
           world Z made perpendicular to the beam).
Dir       "+x" | "-x" | "+y" | "-y" | "+z" | "-z" | {"axis_of": id, "neg": bool?} | {"dir_of": beam id}
          | {"radial": {"about": Axis, "deg": number}} | {"from": Point, "to": Point} | {"rotate": Dir, "about": Dir, "deg": number}
Axis      "x-axis" | "y-axis" | "z-axis" | {"axis_of": id} | {"through": Point, "dir": Dir}
Plane     "xy" | "xz" | "yz" | {"through": Point, "normal": Dir} | {"face": [id, F]}
Dist      number | {"dim": [id | "self", name], "times": number?, "plus": Dist?} | {"touch": id} | {"sum": [Dist, ...]}
          (dimension names are the part's fields; boxes also expose size_x, size_y, size_z)
          {"touch": id}: the center distance at which this part's round surface just touches part id's round surface
          (sphere/cylinder/pipe outer radius or cone base radius + the other part's; measured from the other part's center/axis).
Angles: degrees, counterclockwise about the axis direction (right-hand rule); azimuth 0 is world +X projected onto the plane
normal to the axis (world +Y if the axis is along X).
"""
from __future__ import annotations

import copy
import math

import numpy as np


class CMLError(Exception):
    pass


WORLD_DIRS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
WORLD_AXES = {"x-axis": (1, 0, 0), "y-axis": (0, 1, 0), "z-axis": (0, 0, 1)}
WORLD_PLANES = {"xy": (0, 0, 1), "xz": (0, 1, 0), "yz": (1, 0, 0)}
DIMS = {"box": ["size"], "sphere": ["radius"], "cylinder": ["radius", "height"], "pipe": ["inner_radius", "outer_radius", "height"],
        "cone": ["base_radius", "top_radius", "height"], "torus": ["ring_radius", "tube_radius"], "beam": ["width", "height"]}
AXIAL = ("cylinder", "pipe", "cone")


def _u(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise CMLError("zero-length direction")
    return v / n


def _rot(axis, deg):
    a = _u(axis); t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def _zero_ref(a):
    x = np.array([1.0, 0, 0])
    if abs(a @ x) > 0.999:
        x = np.array([0, 1.0, 0])
    return _u(x - (x @ a) * a)


def _up(d):
    z = np.array([0, 0, 1.0])
    if abs(d @ z) > 0.999:
        z = np.array([1.0, 0, 0])
    return _u(z - (z @ d) * d)


class Interp:
    def __init__(self):
        self.parts: dict[int, dict] = {}
        self.order: list[int] = []

    # ---------------- references ----------------
    def part(self, pid):
        if pid not in self.parts:
            raise CMLError(f"reference to undefined part {pid}")
        return self.parts[pid]

    def dist(self, d, self_part=None):
        if isinstance(d, (int, float)) and not isinstance(d, bool):
            return float(d)
        if not isinstance(d, dict):
            raise CMLError(f"bad Dist {d!r}")
        if "dim" in d:
            pid, name = d["dim"]
            p = self_part if pid == "self" else self.part(pid)
            if p is not None and name in ("size_x", "size_y", "size_z") and p.get("size") is not None:
                val = p["size"]["xyz".index(name[-1])]
            elif p is None or name not in p:
                raise CMLError(f"dimension {name} of {pid} unavailable")
            else:
                val = p[name]
            if val is None:
                raise CMLError(f"dimension {name} of {pid} not yet known")
            v = float(val) * float(d.get("times", 1.0))
            return v + (self.dist(d["plus"], self_part) if "plus" in d else 0.0)
        if "sum" in d:
            return sum(self.dist(x, self_part) for x in d["sum"])
        if "touch" in d:
            other = self.part(d["touch"])
            if self_part is None:
                raise CMLError("touch needs the part being placed")
            return self._round_radius(self_part) + self._round_radius(other)
        raise CMLError(f"bad Dist {d!r}")

    @staticmethod
    def _round_radius(p):
        t = p["type"]
        if t in ("sphere", "cylinder"):
            return float(p["radius"])
        if t == "pipe":
            return float(p["outer_radius"])
        if t == "cone":
            return float(p["base_radius"])
        raise CMLError(f"touch undefined for {t}")

    def dir(self, d, self_part=None):
        if isinstance(d, str):
            if d not in WORLD_DIRS:
                raise CMLError(f"unknown direction {d!r}")
            return np.array(WORLD_DIRS[d], float)
        if isinstance(d, (list, tuple)):
            return _u(d)
        if not isinstance(d, dict):
            raise CMLError(f"bad Dir {d!r}")
        if "axis_of" in d:
            p = self.part(d["axis_of"])
            if "axis" not in p:
                raise CMLError(f"part {d['axis_of']} has no axis")
            a = np.array(p["axis"], float)
            return -a if d.get("neg") else a
        if "dir_of" in d:
            p = self.part(d["dir_of"])
            if p["type"] != "beam":
                raise CMLError("dir_of needs a beam")
            return _u(np.array(p["end"]) - np.array(p["start"]))
        if "radial" in d:
            r = d["radial"]
            c, a = self.axis(r["about"], self_part)
            return _rot(a, float(r["deg"])) @ _zero_ref(a)
        if "from" in d and "to" in d:
            return _u(self.point(d["to"], self_part) - self.point(d["from"], self_part))
        if "rotate" in d:
            return _rot(self.dir(d["about"], self_part), float(d["deg"])) @ self.dir(d["rotate"], self_part)
        raise CMLError(f"bad Dir {d!r}")

    def axis(self, ax, self_part=None):
        """-> (point on axis, unit direction)"""
        if isinstance(ax, str):
            if ax not in WORLD_AXES:
                raise CMLError(f"unknown axis {ax!r}")
            return np.zeros(3), np.array(WORLD_AXES[ax], float)
        if isinstance(ax, dict) and "axis_of" in ax:
            p = self.part(ax["axis_of"])
            if "axis" not in p:
                raise CMLError(f"part {ax['axis_of']} has no axis")
            return np.array(p["center"], float), np.array(p["axis"], float)
        if isinstance(ax, dict) and "through" in ax:
            return self.point(ax["through"], self_part), self.dir(ax["dir"], self_part)
        raise CMLError(f"bad Axis {ax!r}")

    def plane(self, pl, self_part=None):
        """-> (point, unit normal)"""
        if isinstance(pl, str):
            if pl not in WORLD_PLANES:
                raise CMLError(f"unknown plane {pl!r}")
            return np.zeros(3), np.array(WORLD_PLANES[pl], float)
        if isinstance(pl, dict) and "face" in pl:
            pid, f = pl["face"]
            return self.feature(self.part(pid), f), self.face_normal(self.part(pid), f)
        if isinstance(pl, dict) and "through" in pl:
            return self.point(pl["through"], self_part), self.dir(pl["normal"], self_part)
        raise CMLError(f"bad Plane {pl!r}")

    def face_normal(self, p, f):
        t = p["type"]
        if t == "box" and f in WORLD_DIRS:
            return np.array(WORLD_DIRS[f], float)
        if f in ("top", "+z") and t in ("box", "sphere"):
            return np.array([0, 0, 1.0])
        if f in ("bottom", "-z") and t in ("box", "sphere"):
            return np.array([0, 0, -1.0])
        if t in AXIAL and f in ("top", "bottom"):
            a = np.array(p["axis"], float)
            return a if f == "top" else -a
        if t == "beam" and f in ("top_face", "bottom_face"):
            u = _up(_u(np.array(p["end"]) - np.array(p["start"])))
            return u if f == "top_face" else -u
        raise CMLError(f"feature {f!r} of a {t} has no normal")

    def feature(self, p, f, deg=None, t=None):
        typ = p["type"]
        c = np.array(p["center"], float)
        if f == "center":
            return c
        if typ in AXIAL:
            a = np.array(p["axis"], float); h = float(p["height"])
            if f == "top":
                return c + a * h / 2
            if f == "bottom":
                return c - a * h / 2
            if f == "surface":
                r = float(p["outer_radius"] if typ == "pipe" else (p["radius"] if typ == "cylinder" else p["base_radius"]))
                if deg is None:
                    raise CMLError("surface feature needs deg")
                return c + a * float(t or 0.0) + r * (_rot(a, float(deg)) @ _zero_ref(a))
        if typ == "box":
            s = np.array(p["size"], float) / 2
            key = {"top": "+z", "bottom": "-z"}.get(f, f)
            if key in WORLD_DIRS:
                return c + np.array(WORLD_DIRS[key]) * s
        if typ == "sphere" and f in ("top", "bottom"):
            return c + np.array([0, 0, 1.0 if f == "top" else -1.0]) * float(p["radius"])
        if typ == "beam":
            s, e = np.array(p["start"], float), np.array(p["end"], float)
            if f == "start":
                return s
            if f == "end":
                return e
            if f in ("top_face", "bottom_face"):
                u = _up(_u(e - s))
                return (s + e) / 2 + (1 if f == "top_face" else -1) * u * float(p["height"]) / 2
        raise CMLError(f"feature {f!r} not defined for {typ}")

    def point(self, pt, self_part=None):
        if isinstance(pt, str):
            if pt != "origin":
                raise CMLError(f"unknown point {pt!r}")
            base = np.zeros(3)
            moves = []
        elif isinstance(pt, (list, tuple)):
            if len(pt) != 3:
                raise CMLError("points are [x,y,z]")
            return np.array(pt, float)
        elif isinstance(pt, dict):
            if "of" in pt:
                base = self.feature(self.part(pt["of"]), pt.get("feature", "center"), pt.get("deg"), pt.get("t"))
            elif "intersect" in pt:
                (q, d), (p0, n) = self.axis(pt["intersect"][0], self_part), self.plane(pt["intersect"][1], self_part)
                den = float(d @ n)
                if abs(den) < 1e-12:
                    raise CMLError("axis parallel to plane")
                base = q + d * float((p0 - q) @ n) / den
            elif "from" in pt:
                base = self.point(pt["from"], self_part)
            elif "xyz" in pt:
                ps = [self.point(q, self_part) for q in pt["xyz"]]
                base = np.array([ps[0][0], ps[1][1], ps[2][2]])
            elif "between" in pt:
                a_, b_ = (self.point(q, self_part) for q in pt["between"])
                base = a_ + (b_ - a_) * float(pt["t"])
            elif "nest" in pt:
                base = self._nest(pt["nest"], self_part)
            else:
                raise CMLError(f"bad Point {pt!r}")
            moves = pt.get("then", [])
        else:
            raise CMLError(f"bad Point {pt!r}")
        for mv in moves:
            base = base + self.dir(mv["along"], self_part) * self.dist(mv["by"], self_part)
        return base

    def _nest(self, spec, self_part):
        if self_part is None or self_part.get("type") != "sphere":
            raise CMLError("nest places a sphere")
        r = float(self_part["radius"])
        eqs = []   # (center, distance)
        for sid in spec.get("touch", []):
            o = self.part(sid)
            if o["type"] != "sphere":
                raise CMLError("nest touches spheres only")
            eqs.append((np.array(o["center"], float), r + float(o["radius"])))
        planes = []
        if "rest_on" in spec:
            q, n = self.plane(spec["rest_on"], self_part)
            planes.append((q + n * r, n))      # center lies on the plane offset by r
        side = self.dir(spec["side"], self_part)
        if len(eqs) + len(planes) != 3:
            raise CMLError("nest needs exactly three conditions (spheres touched + rest_on plane)")
        # linearise: differences of sphere equations are planes; solve the 2 linear + 1 quadratic system
        lin_A, lin_b = [], []
        for (q, n) in planes:
            lin_A.append(n); lin_b.append(float(n @ q))
        c0, d0 = eqs[0]
        for (c, d) in eqs[1:]:
            lin_A.append(2 * (c - c0)); lin_b.append(float(c @ c - c0 @ c0 - d * d + d0 * d0))
        A = np.array(lin_A); b = np.array(lin_b)
        if A.shape[0] != 2:
            raise CMLError("nest needs at least one touched sphere")
        # solution line of the 2 linear equations: p = p0 + s * dirn
        dirn = np.cross(A[0], A[1])
        if np.linalg.norm(dirn) < 1e-12:
            raise CMLError("nest conditions are degenerate")
        p0 = np.linalg.lstsq(A, b, rcond=None)[0]
        dirn = dirn / np.linalg.norm(dirn)
        w = p0 - c0
        B = 2 * float(w @ dirn); Cc = float(w @ w) - d0 * d0
        disc = B * B - 4 * Cc
        if disc < -1e-9:
            raise CMLError("nest has no solution (spheres too far apart)")
        roots = [(-B + sg * math.sqrt(max(disc, 0))) / 2 for sg in (1, -1)]
        sols = [p0 + t_ * dirn for t_ in roots]
        return max(sols, key=lambda v: float(v @ side))

    # ---------------- parts ----------------
    def place(self, step):
        pid = step.get("id")
        if not isinstance(pid, int) or isinstance(pid, bool):
            raise CMLError("every part needs an integer id")
        if pid in self.parts:
            raise CMLError(f"duplicate id {pid}")
        t = step.get("type")
        if t not in DIMS:
            raise CMLError(f"unknown type {t!r}")
        p = {"id": pid, "type": t}
        for name in DIMS[t] + (["top_radius"] if t == "cone" else []) + (["length"] if t == "beam" else []):
            if name in step:
                v = step[name]
                if name == "size":
                    if not isinstance(v, (list, tuple)) or len(v) != 3:
                        raise CMLError(f"part {pid}: size is [sx, sy, sz]")
                    p[name] = [None if x is None else self.dist(x, p) for x in v]
                else:
                    p[name] = self.dist(v, p)
        if t == "cone" and "top_radius" not in p:
            p["top_radius"] = 0.0
        if t == "beam" and "height" not in p and "width" in p:
            p["height"] = p["width"]
        missing = [n for n in DIMS[t] if n not in p and not (n == "height" and "top_through" in step)]
        if missing:
            raise CMLError(f"part {pid}: missing dims {missing}")
        if t == "beam":
            s = self.point(step["start"], p) if "start" in step else None
            if s is None:
                raise CMLError(f"beam {pid} needs start")
            if "end" in step:
                e = self.point(step["end"], p)
            elif "direction" in step and "length" in p:
                e = s + self.dir(step["direction"], p) * float(p["length"])
            else:
                raise CMLError(f"beam {pid} needs end or direction+length")
            p["start"], p["end"] = s, e
            p["center"] = (s + e) / 2
            p["axis"] = _u(e - s)
        else:
            if t in AXIAL + ("torus",):
                p["axis"] = self.dir(step.get("axis", "+z"), p)
            keys = [k for k in ("at", "bottom_at", "top_at", "align") if k in step]
            if len(keys) != 1:
                raise CMLError(f"part {pid}: give exactly one of at / bottom_at / top_at / align")
            k = keys[0]
            if k == "align":
                if t != "box":
                    raise CMLError("align is defined for boxes only")
                if any(x is None for x in p["size"]):
                    raise CMLError(f"box {pid}: size unknown")
                c = np.zeros(3)
                for i, ax_ in enumerate("xyz"):
                    if ax_ not in step["align"]:
                        raise CMLError(f"box {pid}: align needs x, y and z")
                    how, pt = step["align"][ax_]
                    v = self.point(pt, p)[i]
                    half = p["size"][i] / 2
                    c[i] = {"center": v, "+": v - half, "-": v + half}.get(how, None) if how in ("center", "+", "-") else None
                    if c[i] is None:
                        raise CMLError(f"box {pid}: align mode must be center, + or -")
                p["center"] = c
                self.parts[pid] = p; self.order.append(pid)
                return self._rest(step, p)
            anchor = self.point(step[k], p)
            if k != "at" and t not in AXIAL + ("box", "sphere"):
                raise CMLError(f"{k} not defined for {t}")
            if "top_through" in step:
                if k != "bottom_at" or t not in AXIAL + ("box",):
                    raise CMLError("top_through needs bottom_at on a cylinder/pipe/cone/box")
                a = p["axis"] if t in AXIAL else np.array([0, 0, 1.0])
                h = float((self.point(step["top_through"], p) - anchor) @ a)
                if h <= 0:
                    raise CMLError(f"part {pid}: top_through lies below the bottom face")
                if t == "box":
                    p["size"][2] = h
                else:
                    p["height"] = h
            if t == "box" and any(x is None for x in p["size"]):
                raise CMLError(f"box {pid}: size unknown")
            if k == "at":
                p["center"] = anchor
            else:
                if t in AXIAL:
                    off = p["axis"] * float(p["height"]) / 2
                elif t == "box":
                    off = np.array([0, 0, p["size"][2] / 2])
                else:
                    off = np.array([0, 0, p["radius"]])
                p["center"] = anchor + off if k == "bottom_at" else anchor - off
        self.parts[pid] = p
        self.order.append(pid)
        self._rest(step, p)

    def _rest(self, step, p):
        if "rest_on" in step:
            q, n = self.plane(step["rest_on"], p)
            lo = min(float((v - q) @ n) for v in self._extreme_points(p, n))
            shift = -lo * n
            for key in ("center", "start", "end"):
                if key in p:
                    p[key] = p[key] + shift

    def _extreme_points(self, p, n):
        """Points of the part along -n direction (support in direction -n)."""
        t = p["type"]; c = np.array(p["center"], float)
        if t == "sphere":
            return [c - n * p["radius"]]
        if t == "box":
            s = np.array(p["size"]) / 2
            return [c - np.sign(n) * s]
        if t == "beam":
            s, e = p["start"], p["end"]; d = _u(e - s); u = _up(d); w = np.cross(d, u)
            pts = []
            for base in (s, e):
                for a in (-1, 1):
                    for b in (-1, 1):
                        pts.append(base + a * u * p["height"] / 2 + b * w * p["width"] / 2)
            return pts
        if t == "torus":
            a = p["axis"]; r_dir = n - (n @ a) * a
            ext = p["ring_radius"] * (_u(r_dir) if np.linalg.norm(r_dir) > 1e-9 else 0)
            return [c - ext - n * p["tube_radius"]]
        a = p["axis"]; h = p["height"]
        radii = {"cylinder": (p.get("radius", 0), p.get("radius", 0)), "pipe": (p.get("outer_radius", 0),) * 2,
                 "cone": (p.get("base_radius", 0), p.get("top_radius", 0))}[t]
        pts = []
        for sgn, r in ((-1, radii[0]), (1, radii[1])):
            e = c + sgn * a * h / 2
            rd = n - (n @ a) * a
            pts.append(e - (r * _u(rd) if np.linalg.norm(rd) > 1e-9 else 0))
        return pts

    # ---------------- patterns ----------------
    def pattern(self, step):
        kind = step.get("pattern"); seed = step.get("seed", [])
        count = step.get("count")
        if not seed or not isinstance(count, int) or count < 1:
            raise CMLError("pattern needs seed ids and an integer count")
        for s in seed:
            self.part(s)
        nid = int(step.get("first_id", max(self.parts) + 1))
        created = []
        for k in range(1, count):
            for s in seed:
                p = copy.deepcopy(self.parts[s]); p["id"] = nid
                if kind in ("circular", "helical"):
                    q, a = self.axis(step["about"])
                    if "step_deg" in step:
                        ang = float(step["step_deg"]) * k
                    elif "per_revolution" in step:
                        ang = 360.0 / float(step["per_revolution"]) * k
                    elif step.get("full_circle"):
                        ang = 360.0 / count * k
                    else:
                        raise CMLError("circular/helical pattern needs step_deg, per_revolution or full_circle")
                    R = _rot(a, ang)
                    rise = a * self.dist(step["rise"]) * k if kind == "helical" else np.zeros(3)
                    self._transform(p, lambda v: q + R @ (v - q) + rise, R)
                elif kind == "linear":
                    d = self.dir(step["direction"]) * self.dist(step["step"]) * k
                    self._transform(p, lambda v: v + d, np.eye(3))
                elif kind == "mirror":
                    if count != 2:
                        raise CMLError("mirror count must be 2")
                    q, n = self.plane(step["plane"])
                    M = np.eye(3) - 2 * np.outer(n, n)
                    self._transform(p, lambda v: q + M @ (v - q), M)
                else:
                    raise CMLError(f"unknown pattern {kind!r}")
                if nid in self.parts:
                    raise CMLError(f"pattern would reuse id {nid}")
                self.parts[nid] = p; self.order.append(nid); created.append(nid); nid += 1
        if kind == "linear" and step.get("centered"):
            shift = -self.dir(step["direction"]) * self.dist(step["step"]) * (count - 1) / 2
            for g in list(seed) + created:
                self._transform(self.parts[g], lambda v: v + shift, np.eye(3))

    @staticmethod
    def _transform(p, f, R):
        for key in ("center", "start", "end"):
            if key in p:
                p[key] = f(np.asarray(p[key], float))
        if p["type"] == "box":
            # boxes stay axis-aligned: allowed only for rotations that permute the world axes
            P = np.abs(np.round(R, 9))
            if not np.allclose(P @ np.ones(3), 1) or not np.allclose(np.sort(P.ravel())[-3:], 1):
                raise CMLError("a pattern rotates a box by a non-axis-aligned angle")
            p["size"] = list(P @ np.array(p["size"]))
        elif "axis" in p:
            p["axis"] = _u(R @ np.asarray(p["axis"], float))

    # ---------------- run ----------------
    def run(self, program) -> list[dict]:
        if isinstance(program, dict) and "steps" in program:
            steps = program["steps"]
        elif isinstance(program, list):
            steps = program
        else:
            raise CMLError("program must be {\"steps\": [...]}")
        for st in steps:
            if not isinstance(st, dict):
                raise CMLError("each step is an object")
            if "pattern" in st:
                self.pattern(st)
            else:
                self.place(st)
        return [self._emit(self.parts[i]) for i in sorted(self.parts)]

    @staticmethod
    def _emit(p):
        t = p["type"]
        out = {"id": p["id"], "type": t}
        if t == "beam":
            out.update(start=list(map(float, p["start"])), end=list(map(float, p["end"])), width=p["width"], height=p["height"])
            return out
        out["center"] = list(map(float, p["center"]))
        for k in DIMS[t] + (["top_radius"] if t == "cone" else []):
            out[k] = p[k]
        if "axis" in p:
            out["axis"] = list(map(float, p["axis"]))
        return out


def run(program) -> list[dict]:
    return Interp().run(program)
