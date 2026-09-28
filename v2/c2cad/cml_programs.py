"""Hand-written CML programs (written from the prompt text) used to gate the mates interface:
each must reproduce its case's reference exactly. Only counts vary with the level.
"""
from __future__ import annotations

import re


def at_axis(of, deg=0):
    return {"radial": {"about": {"axis_of": of}, "deg": deg}}


def planetary(case):
    n = case["scale"]
    return {"steps": [
        {"id": 0, "type": "cylinder", "radius": 10, "height": 2, "axis": "+z", "at": "origin"},
        {"id": 1, "type": "cylinder", "radius": 3, "height": 2, "axis": "+z",
         "at": {"of": 0, "feature": "center", "then": [{"along": at_axis(0, 0), "by": {"touch": 0}}]}},
        {"pattern": "circular", "seed": [1], "count": n, "about": {"axis_of": 0}, "full_circle": True, "first_id": 2}]}


def staircase(case):
    n = case["scale"]
    return {"steps": [
        {"id": 1, "type": "beam", "width": 5, "height": 2, "length": 20,
         "start": {"from": "origin", "then": [{"along": "+x", "by": 5}]}, "direction": "+x", "rest_on": "xy"},
        {"pattern": "helical", "seed": [1], "count": n, "about": "z-axis", "per_revolution": 24, "rise": 5, "first_id": 2},
        {"id": 0, "type": "cylinder", "radius": 5, "axis": "+z", "bottom_at": "origin",
         "top_through": {"of": n, "feature": "bottom_face", "then": [{"along": "+z", "by": 5}]}}]}


def ball_bearing(case):
    n = case["scale"]
    return {"steps": [
        {"id": 0, "type": "pipe", "inner_radius": 12.5, "outer_radius": 17, "height": 15, "axis": "+z", "at": "origin"},
        {"id": 2, "type": "sphere", "radius": 3,
         "at": {"of": 0, "feature": "center", "then": [{"along": at_axis(0, 0), "by": {"touch": 0}}]}},
        {"pattern": "circular", "seed": [2], "count": n, "about": {"axis_of": 0}, "full_circle": True, "first_id": 3},
        {"id": 1, "type": "pipe", "inner_radius": {"dim": [0, "outer_radius"], "plus": {"dim": [2, "radius"], "times": 2}},
         "outer_radius": 26, "height": 15, "axis": "+z", "at": "origin"}]}


def flanged(case):
    n = case["scale"]
    return {"steps": [
        {"id": 2, "type": "pipe", "inner_radius": 25, "outer_radius": 50, "height": 8, "axis": "+x",
         "top_at": {"from": "origin", "then": [{"along": "-x", "by": 1}]}},
        {"id": 3, "type": "pipe", "inner_radius": 25, "outer_radius": 50, "height": 8, "axis": "+x",
         "bottom_at": {"from": "origin", "then": [{"along": "+x", "by": 1}]}},
        {"id": 0, "type": "pipe", "inner_radius": 25, "outer_radius": 30, "height": 60, "axis": "+x", "top_at": {"of": 2, "feature": "bottom"}},
        {"id": 1, "type": "pipe", "inner_radius": 25, "outer_radius": 30, "height": 60, "axis": "+x", "bottom_at": {"of": 3, "feature": "top"}},
        {"id": 4, "type": "cylinder", "radius": 3, "height": 20, "axis": "+x", "at": {"from": "origin", "then": [{"along": "+y", "by": 40}]}},
        {"id": 5, "type": "torus", "ring_radius": {"dim": [4, "radius"]}, "tube_radius": 2, "axis": "+x",
         "at": {"intersect": [{"axis_of": 4}, {"face": [3, "top"]}]}},
        {"pattern": "circular", "seed": [4, 5], "count": n, "about": "x-axis", "full_circle": True, "first_id": 6}]}


def manifold(case):
    b = case["scale"]
    first, last = 4, 4 + 4 * (b - 1)
    axis_pt = lambda q, sgn: {"xyz": [{"of": q, "feature": "bottom", "then": [{"along": "+x" if sgn > 0 else "-x", "by": 40}]},
                                     "origin", {"from": "origin", "then": [{"along": "+z", "by": 50}]}]}
    return {"steps": [
        {"id": 4, "type": "pipe", "inner_radius": 6, "outer_radius": 8, "height": 35, "axis": "+y",
         "bottom_at": {"from": "origin", "then": [{"along": "+z", "by": 50}, {"along": "+y", "by": 15}]}},
        {"id": 5, "type": "pipe", "inner_radius": {"dim": [4, "outer_radius"], "plus": 0.3}, "outer_radius": {"dim": [4, "outer_radius"], "plus": 5},
         "height": 4, "axis": "+y", "bottom_at": {"of": 4, "feature": "bottom"}},
        {"id": 6, "type": "cylinder", "radius": 10, "height": 10, "axis": "+y", "at": {"of": 4, "feature": "center"}},
        {"id": 7, "type": "box", "size": [20, 10, None], "bottom_at": {"xyz": [{"of": 4, "feature": "bottom"}, "origin", "origin"]},
         "top_through": {"from": "origin", "then": [{"along": "+z", "by": 50}, {"along": "-z", "by": 15}]}},
        {"pattern": "linear", "seed": [4, 5, 6, 7], "count": b, "direction": "+x", "step": 40, "centered": True, "first_id": 8},
        {"id": 0, "type": "pipe", "inner_radius": 12, "outer_radius": 15, "axis": "+x", "bottom_at": axis_pt(first, -1),
         "top_through": axis_pt(last, +1)},
        {"id": 1, "type": "box", "size": [{"sum": [{"dim": [0, "height"]}, 10, 10]}, 5, {"sum": [50, {"dim": [0, "outer_radius"]}, 20]}],
         "align": {"x": ["center", "origin"], "y": ["+", {"from": "origin", "then": [{"along": "-y", "by": {"dim": [0, "outer_radius"]}}]}],
                   "z": ["-", "origin"]}},
        {"id": 2, "type": "cylinder", "radius": {"dim": [0, "outer_radius"]}, "height": 3, "axis": "+x", "top_at": {"of": 0, "feature": "bottom"}},
        {"id": 3, "type": "cylinder", "radius": {"dim": [0, "outer_radius"]}, "height": 3, "axis": "+x", "bottom_at": {"of": 0, "feature": "top"}}]}


def gantry(case):
    b = case["scale"]; ncol = b + 1
    R0, R1, BR, BRIDGE = 2 * ncol, 2 * ncol + 1, 2 * ncol + 2, 2 * ncol + 2 + 2 * b
    TROL, DRUM, CAB, HOOK = BRIDGE + 1, BRIDGE + 2, BRIDGE + 3, BRIDGE + 4
    return {"steps": [
        {"id": 0, "type": "cylinder", "radius": 5, "height": 100, "axis": "+z", "bottom_at": "origin"},
        {"pattern": "linear", "seed": [0], "count": 2, "direction": "+y", "step": 80, "centered": True, "first_id": 1},
        {"pattern": "linear", "seed": [0, 1], "count": ncol, "direction": "+x", "step": 40, "centered": True, "first_id": 2},
        {"id": R0, "type": "beam", "width": 4, "height": 4, "start": {"of": 0, "feature": "top"},
         "end": {"of": 2 * b, "feature": "top"}, "rest_on": {"face": [0, "top"]}},
        {"id": R1, "type": "beam", "width": 4, "height": 4, "start": {"of": 1, "feature": "top"},
         "end": {"of": 2 * b + 1, "feature": "top"}, "rest_on": {"face": [1, "top"]}},
        {"id": BR, "type": "beam", "width": 2.5, "height": 2.5, "start": {"of": 0, "feature": "bottom"}, "end": {"of": 2, "feature": "top"}},
        {"id": BR + 1, "type": "beam", "width": 2.5, "height": 2.5, "start": {"of": 0, "feature": "top"}, "end": {"of": 2, "feature": "bottom"}},
        {"pattern": "linear", "seed": [BR, BR + 1], "count": b, "direction": "+x", "step": 40, "first_id": BR + 2},
        {"id": BRIDGE, "type": "beam", "width": 5, "height": 5, "start": {"of": R0, "feature": "center"}, "end": {"of": R1, "feature": "center"}},
        {"id": TROL, "type": "box", "size": [15, 12, 8], "at": {"of": BRIDGE, "feature": "center"}, "rest_on": {"face": [R0, "top_face"]}},
        {"id": DRUM, "type": "cylinder", "radius": 6, "height": 10, "axis": "+y", "at": {"of": TROL, "feature": "top"},
         "rest_on": {"face": [TROL, "top"]}},
        {"id": CAB, "type": "pipe", "inner_radius": 1, "outer_radius": 2, "height": 60, "axis": "+z", "top_at": {"of": TROL, "feature": "bottom"}},
        {"id": HOOK, "type": "cone", "base_radius": 4, "top_radius": 0.5, "height": 10, "axis": "+z", "top_at": {"of": CAB, "feature": "bottom"}}]}


def pyramid(case):
    L = case["scale"]; steps = []; nid = 0; layers = []
    for l in range(L):
        rows = []
        for j in range(L - l):
            n_row = L - l - j
            first = nid
            if l == 0 and j == 0:
                at = {"xyz": ["origin", "origin", "origin"]}
                steps.append({"id": first, "type": "sphere", "radius": 10, "bottom_at": "origin"})
            elif j == 0:
                below = layers[l - 1]
                steps.append({"id": first, "type": "sphere", "radius": 10,
                              "at": {"nest": {"touch": [below[0][0], below[0][1], below[1][0]], "side": "+z"}}})
            else:
                prev = rows[j - 1]
                steps.append({"id": first, "type": "sphere", "radius": 10,
                              "at": {"nest": {"touch": [prev[0], prev[1]], "rest_on": {"through": {"of": rows[0][0], "feature": "bottom"}, "normal": "+z"},
                                              "side": "+y"}}})
            if n_row > 1:
                steps.append({"pattern": "linear", "seed": [first], "count": n_row, "direction": "+x",
                              "step": {"dim": [first, "radius"], "times": 2}, "first_id": first + 1})
            rows.append(list(range(first, first + n_row))); nid = first + n_row
        layers.append(rows)
    return {"steps": steps}


def voxel(case):
    n = case["scale"]; step = {"sum": [10, 2]}
    return {"steps": [
        {"id": 0, "type": "box", "size": [10, 10, 10], "align": {"x": ["-", "origin"], "y": ["-", "origin"], "z": ["-", "origin"]}},
        {"pattern": "linear", "seed": [0], "count": n, "direction": "+x", "step": step, "first_id": 1},
        {"pattern": "linear", "seed": list(range(n)), "count": n, "direction": "+y", "step": step, "first_id": n},
        {"pattern": "linear", "seed": list(range(n * n)), "count": n, "direction": "+z", "step": step, "first_id": n * n}]}


def domino(case):
    n = case["scale"]
    return {"steps": [
        {"id": 0, "type": "cylinder", "radius": 4, "height": 30, "axis": "+z",
         "bottom_at": {"from": "origin", "then": [{"along": {"radial": {"about": "z-axis", "deg": -3}}, "by": 80}]}},
        {"pattern": "mirror", "seed": [0], "count": 2, "plane": "xz", "first_id": 1},
        {"id": 2, "type": "beam", "width": {"dim": [0, "radius"], "times": 2}, "height": {"dim": [0, "radius"], "times": 1.5},
         "start": {"of": 0, "feature": "top"}, "end": {"of": 1, "feature": "top"}},
        {"pattern": "circular", "seed": [0, 1, 2], "count": n, "about": "z-axis", "full_circle": True, "first_id": 3}]}


def dna(case):
    n = case["scale"]
    return {"steps": [
        {"id": 0, "type": "sphere", "radius": 3, "at": {"from": "origin", "then": [{"along": "+x", "by": 20}]}},
        {"id": 1, "type": "sphere", "radius": 3, "at": {"from": "origin", "then": [{"along": "-x", "by": 20}]}},
        {"id": 2, "type": "beam", "width": 2, "height": 2, "start": {"of": 0, "feature": "center"}, "end": {"of": 1, "feature": "center"}},
        {"pattern": "helical", "seed": [0, 1, 2], "count": n, "about": "z-axis", "per_revolution": 10, "rise": 4, "first_id": 3}]}


def bridge(case):
    n = case["scale"]
    steps = [
        {"id": 0, "type": "beam", "width": 1, "height": 1, "length": 100,
         "start": {"from": "origin", "then": [{"along": "+z", "by": 10}, {"along": "-x", "by": {"dim": ["self", "length"], "times": 0.5}}]},
         "direction": "+x"},
        {"id": 1, "type": "cylinder", "radius": 2, "height": 100, "axis": "+z",
         "bottom_at": {"intersect": [{"through": {"of": 0, "feature": "start"}, "dir": "+z"}, "xy"]}},
        {"id": 2, "type": "cylinder", "radius": 2, "height": 100, "axis": "+z",
         "bottom_at": {"intersect": [{"through": {"of": 0, "feature": "end"}, "dir": "+z"}, "xy"]}}]
    outer_l = {"of": 0, "feature": "start", "then": [{"along": "+x", "by": 2}]}
    inner_l = {"of": 0, "feature": "center", "then": [{"along": "-x", "by": 2}]}
    outer_r = {"of": 0, "feature": "end", "then": [{"along": "-x", "by": 2}]}
    inner_r = {"of": 0, "feature": "center", "then": [{"along": "+x", "by": 2}]}
    for i in range(n):
        t = i / (n - 1)
        steps.append({"id": 3 + i, "type": "beam", "width": 1, "height": 1, "start": {"of": 1, "feature": "top"},
                      "end": {"between": [outer_l, inner_l], "t": t}})
    for i in range(n):
        t = i / (n - 1)
        steps.append({"id": 3 + n + i, "type": "beam", "width": 1, "height": 1, "start": {"of": 2, "feature": "top"},
                      "end": {"between": [inner_r, outer_r], "t": t}})
    return {"steps": steps}


def fractal(case):
    g = case["scale"]; steps = [{"id": 0, "type": "beam", "width": 1, "height": 1, "length": 100, "start": "origin", "direction": "+z"}]
    counter = [1]
    def grow(parent, depth):
        if depth == g:
            return
        for sgn in (1, -1):
            cid = counter[0]; counter[0] += 1
            steps.append({"id": cid, "type": "beam", "width": 1, "height": 1, "length": {"dim": [parent, "length"], "times": 0.5},
                          "start": {"of": parent, "feature": "end"},
                          "direction": {"rotate": {"dir_of": parent}, "about": "+y", "deg": 45 * sgn}})
            grow(cid, depth + 1)
    grow(0, 0)
    return {"steps": steps}


def axle(case):
    cl = 0.5 * case["scale"]
    return {"steps": [
        {"id": 0, "type": "box", "size": [80, 50, 50], "align": {"x": ["center", "origin"], "y": ["center", "origin"], "z": ["-", "origin"]}},
        {"id": 2, "type": "cylinder", "radius": 8, "height": {"sum": [{"dim": [0, "size_x"]}, 20, 20]}, "axis": "+x", "at": {"of": 0, "feature": "center"}},
        {"id": 1, "type": "cylinder", "radius": {"dim": [2, "radius"], "plus": cl}, "height": {"sum": [{"dim": [0, "size_x"]}, 1, 1]},
         "axis": "+x", "at": {"of": 0, "feature": "center"}},
        {"id": 3, "type": "pipe", "inner_radius": {"dim": [2, "radius"]}, "outer_radius": {"dim": [2, "radius"], "plus": 6}, "height": 12,
         "axis": "+x", "top_at": {"of": 0, "feature": "-x"}},
        {"id": 4, "type": "pipe", "inner_radius": {"dim": [2, "radius"]}, "outer_radius": {"dim": [2, "radius"], "plus": 6}, "height": 12,
         "axis": "+x", "bottom_at": {"of": 0, "feature": "+x"}}]}


def clock(case):
    m = case["scale"]; MIN, HR = 4 + m, 5 + m
    return {"steps": [
        {"id": 0, "type": "cylinder", "radius": 85, "height": 3, "axis": "+z", "top_at": "origin"},
        {"id": 1, "type": "cylinder", "radius": 4, "height": 18, "axis": "+z", "bottom_at": "origin"},
        {"id": 2, "type": "pipe", "inner_radius": {"dim": [1, "radius"], "plus": 0.3}, "outer_radius": 7, "height": 14, "axis": "+z", "bottom_at": "origin"},
        {"id": 3, "type": "torus", "ring_radius": 78, "tube_radius": 3, "axis": "+z", "at": "origin"},
        {"id": 4, "type": "beam", "width": 2, "height": 2, "length": 12,
         "start": {"from": "origin", "then": [{"along": {"radial": {"about": "z-axis", "deg": 0}}, "by": 60}]},
         "direction": {"radial": {"about": "z-axis", "deg": 0}}},
        {"pattern": "circular", "seed": [4], "count": m, "about": "z-axis", "full_circle": True, "first_id": 5},
        {"id": MIN, "type": "beam", "width": 3, "height": 1.5, "length": 62, "start": {"from": "origin", "then": [{"along": "+z", "by": 1}]}, "direction": "+y"},
        {"id": HR, "type": "beam", "width": 3, "height": 2, "length": 42, "start": {"from": "origin", "then": [{"along": "+z", "by": 2}]},
         "direction": {"radial": {"about": "z-axis", "deg": 120}}},
        {"id": HR + 1, "type": "cone", "base_radius": 4, "top_radius": 0.5, "height": 10, "axis": {"axis_of": MIN, "neg": True},
         "bottom_at": {"of": MIN, "feature": "start", "then": [{"along": {"axis_of": MIN, "neg": True}, "by": 1}]}},
        {"id": HR + 2, "type": "cone", "base_radius": 4, "top_radius": 0.5, "height": 10, "axis": {"axis_of": HR, "neg": True},
         "bottom_at": {"of": HR, "feature": "start", "then": [{"along": {"axis_of": HR, "neg": True}, "by": 1}]}}]}


def furniture(case):
    m = case["scale"] * 2; P_ = (m - 4) // 2
    def corner(sx, sy):
        fx, fy = ("+x" if sx > 0 else "-x"), ("+y" if sy > 0 else "-y")
        return {"xyz": [{"of": 0, "feature": fx, "then": [{"along": "-x" if sx > 0 else "+x", "by": {"dim": ["self", "radius"], "times": 2}}]},
                        {"of": 0, "feature": fy, "then": [{"along": "-y" if sy > 0 else "+y", "by": {"dim": ["self", "radius"], "times": 2}}]},
                        "origin"]}
    steps = [{"id": 0, "type": "box", "size": [100, 60, 5],
              "align": {"x": ["center", "origin"], "y": ["center", "origin"], "z": ["-", {"from": "origin", "then": [{"along": "+z", "by": 70}]}]}}]
    for lid, (sx, sy) in zip(range(1, 5), ((1, 1), (-1, 1), (-1, -1), (1, -1))):
        steps.append({"id": lid, "type": "cylinder", "radius": 3, "height": 70, "axis": "+z", "bottom_at": corner(sx, sy)})
    for k in range(1, P_ + 1):
        a = 3 + 2 * k
        xb_neg = {"of": 0, "feature": "-x", "then": [{"along": "+x", "by": {"dim": [1, "radius"], "times": 6}}]}
        steps.append({"id": a, "type": "cylinder", "radius": 3, "height": 70, "axis": "+z",
                      "bottom_at": {"xyz": [{"between": [{"of": 1, "feature": "center"}, xb_neg], "t": k / (P_ + 1)}, {"of": 1, "feature": "center"}, "origin"]}})
        steps.append({"pattern": "mirror", "seed": [a], "count": 2, "plane": "xz", "first_id": a + 1})
    return {"steps": steps}


def bcc(case):
    n = case["scale"]; steps = []
    steps.append({"id": 0, "type": "sphere", "radius": 1, "at": "origin"})
    steps.append({"pattern": "linear", "seed": [0], "count": n + 1, "direction": "+x", "step": 10, "first_id": 1})
    steps.append({"pattern": "linear", "seed": list(range(n + 1)), "count": n + 1, "direction": "+y", "step": 10, "first_id": n + 1})
    nc = (n + 1) ** 2
    steps.append({"pattern": "linear", "seed": list(range(nc)), "count": n + 1, "direction": "+z", "step": 10, "first_id": nc})
    ncorner = (n + 1) ** 3
    cid = lambda i, j, k: i + (n + 1) * j + (n + 1) ** 2 * k
    b0 = ncorner
    steps.append({"id": b0, "type": "sphere", "radius": 1, "at": {"between": [{"of": cid(0, 0, 0), "feature": "center"}, {"of": cid(1, 1, 1), "feature": "center"}], "t": 0.5}})
    steps.append({"pattern": "linear", "seed": [b0], "count": n, "direction": "+x", "step": 10, "first_id": b0 + 1})
    steps.append({"pattern": "linear", "seed": list(range(b0, b0 + n)), "count": n, "direction": "+y", "step": 10, "first_id": b0 + n})
    steps.append({"pattern": "linear", "seed": list(range(b0, b0 + n * n)), "count": n, "direction": "+z", "step": 10, "first_id": b0 + n * n})
    s0 = b0 + n ** 3
    corners = [cid(i, j, k) for k in (0, 1) for j in (0, 1) for i in (0, 1)]
    for q, c in enumerate(corners):
        steps.append({"id": s0 + q, "type": "beam", "width": 1, "height": 1, "start": {"of": b0, "feature": "center"}, "end": {"of": c, "feature": "center"}})
    steps.append({"pattern": "linear", "seed": list(range(s0, s0 + 8)), "count": n, "direction": "+x", "step": 10, "first_id": s0 + 8})
    steps.append({"pattern": "linear", "seed": list(range(s0, s0 + 8 * n)), "count": n, "direction": "+y", "step": 10, "first_id": s0 + 8 * n})
    steps.append({"pattern": "linear", "seed": list(range(s0, s0 + 8 * n * n)), "count": n, "direction": "+z", "step": 10, "first_id": s0 + 8 * n * n})
    return {"steps": steps}


PROGRAMS = {"Planetary Array": planetary, "Spiral Staircase": staircase, "Ball Bearing Assembly": ball_bearing,
            "Flanged Pipe Joint": flanged, "Pipe Manifold": manifold, "Gantry Crane Assembly": gantry,
            "Cannonball Pyramid": pyramid, "Voxel Grid": voxel, "Domino Ring": domino, "DNA Helix": dna,
            "Suspension Bridge": bridge, "Fractal Y-Tree": fractal, "Axle Bearing": axle, "Clock Tower Mechanism": clock,
            "Furniture Assembly": furniture, "BCC Lattice": bcc}

CONSTANTS = {0.0, 1.0, 2.0, 0.5}      # geometric constants allowed without a prompt source (halving, diameter, identity)


def literals(obj, out=None):
    out = [] if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("id", "first_id", "of", "seed", "axis_of", "dir_of", "touch", "dim"):
                continue
            if k == "face":
                literals(v[1:], out)
            elif k == "nest":
                literals({kk: vv for kk, vv in v.items() if kk != "touch"}, out)
            else:
                literals(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            literals(v, out)
    return out


def provenance(program, prompt_body):
    """Classify every numeric literal: 'prompt' (appears in the prompt), 'constant' (0, 0.5, 1, 2) or 'derived'."""
    nums = {float(x) for x in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", prompt_body)}
    res = {"prompt": 0, "constant": 0, "derived": []}
    for v in literals(program):
        if abs(v) in nums or v in nums:
            res["prompt"] += 1
        elif abs(v) in CONSTANTS:
            res["constant"] += 1
        else:
            res["derived"].append(v)
    return res
