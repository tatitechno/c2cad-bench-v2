"""CadQuery source emitted from v2 JSON parts (no CAD imports, so the main environment can use it).

to_cadquery_code(parts) writes the mechanical conversion (one CadQuery constructor per primitive, literal
arguments) as a program in the form the cadquery arm asks for. Tests run it through the cadquery sandbox and
check that every reference survives build + recovery exactly; the mock provider uses it for the cadquery arm.
"""
from __future__ import annotations

import numpy as np


def _unit(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def beam_frame(start, end):
    """(u, v, w): u along the beam, v = section 'height' axis (world Z made perpendicular to u; world X if the
    beam is vertical), w = u x v the 'width' axis. Same convention as metrics3d._beam_frame."""
    u = _unit(np.asarray(end, float) - np.asarray(start, float))
    up = np.array([0.0, 0.0, 1.0])
    if abs(u @ up) > 0.999:
        up = np.array([1.0, 0.0, 0.0])
    v = up - (up @ u) * u
    v /= np.linalg.norm(v)
    return u, v, np.cross(u, v)


def _f(x) -> str:
    return repr(round(float(x), 9))


def _vec(x) -> str:
    return "cq.Vector(" + ", ".join(_f(c) for c in x) + ")"


def to_cadquery_code(parts: list[dict]) -> str:
    """A CadQuery program that builds `parts` (one solid per id) in the form the cadquery arm asks for."""
    L = ["import cadquery as cq", "", "parts = {}"]
    for p in parts:
        t, i = p["type"], int(p["id"])
        if t == "box":
            c, s = np.asarray(p["center"], float), np.asarray(p["size"], float)
            L.append(f"parts[{i}] = cq.Solid.makeBox({_f(s[0])}, {_f(s[1])}, {_f(s[2])}, pnt={_vec(c - s / 2)})")
        elif t == "sphere":
            L.append(f"parts[{i}] = cq.Solid.makeSphere({_f(p['radius'])}, pnt={_vec(p['center'])}, "
                     f"angleDegrees1=-90, angleDegrees2=90)")
        elif t == "torus":
            L.append(f"parts[{i}] = cq.Solid.makeTorus({_f(p['ring_radius'])}, {_f(p['tube_radius'])}, "
                     f"pnt={_vec(p['center'])}, dir={_vec(_unit(p['axis']))})")
        elif t in ("cylinder", "pipe", "cone"):
            a, h = _unit(p["axis"]), float(p["height"])
            base = np.asarray(p["center"], float) - a * h / 2
            if t == "cylinder":
                L.append(f"parts[{i}] = cq.Solid.makeCylinder({_f(p['radius'])}, {_f(h)}, {_vec(base)}, {_vec(a)})")
            elif t == "cone":
                L.append(f"parts[{i}] = cq.Solid.makeCone({_f(p['base_radius'])}, {_f(p['top_radius'])}, {_f(h)}, "
                         f"{_vec(base)}, {_vec(a)})")
            else:
                L.append(f"parts[{i}] = cq.Solid.makeCylinder({_f(p['outer_radius'])}, {_f(h)}, {_vec(base)}, "
                         f"{_vec(a)}).cut(cq.Solid.makeCylinder({_f(p['inner_radius'])}, {_f(h)}, {_vec(base)}, {_vec(a)}))")
        elif t == "beam":
            s, e = np.asarray(p["start"], float), np.asarray(p["end"], float)
            u, v, _ = beam_frame(s, e)
            w, h = float(p["width"]), float(p.get("height", p["width"]))
            L.append(f"parts[{i}] = cq.Workplane(cq.Plane(origin={_vec((s + e) / 2)}, xDir={_vec(u)}, "
                     f"normal={_vec(v)})).box({_f(np.linalg.norm(e - s))}, {_f(w)}, {_f(h)})")
        else:
            raise ValueError(t)
    return "\n".join(L) + "\n"
