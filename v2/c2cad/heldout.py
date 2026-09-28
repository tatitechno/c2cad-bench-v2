"""Held-out split: the same families with values, handedness and anchors never seen in the main set.

For each family: DEFAULT parameters; `prompt(case_main, params)` rewrites the main v2 prompt by
anchored substitutions (every substitution must match exactly once); `reference(params, n)`
computes the reference. Tests require that at DEFAULT parameters both reproduce the main case
exactly, and that every held-out reference satisfies all constraints (builders read case["params"]).
"""
from __future__ import annotations

import copy
import math

from . import cases as CASES

HAND_WORD = {1: "counterclockwise", -1: "clockwise"}
AXIS_WORD = {0: "positive X", 90: "positive Y", 180: "negative X", 270: "negative Y"}


def _sub(text, pairs):
    for old, new in pairs:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"substitution anchor {old!r} found {n} times")
        text = text.replace(old, new)
    return text


def _fmt(v):
    return f"{v:g}"


def _pol(r, deg, z=0.0):
    t = math.radians(deg)
    return [r * math.cos(t), r * math.sin(t), z]


# ---------------------------------------------------------------------------
class Staircase:
    family = "Spiral Staircase"
    DEFAULT = dict(R=5, L=20, W=5, T=2, RISE=5, PER_REV=24, HAND=1, FIRST=0)
    VARIANTS = [dict(R=6, L=18, W=4, T=1.5, RISE=4, PER_REV=16, HAND=-1, FIRST=90),
                dict(R=4, L=25, W=6, T=2.5, RISE=6, PER_REV=30, HAND=1, FIRST=180)]

    @staticmethod
    def prompt(body, p):
        return _sub(body, [
            ("The cylindrical pillar has radius 5.", f"The cylindrical pillar has radius {_fmt(p['R'])}."),
            ("Each tread has radial length 20, width 5,\nand thickness 2.", f"Each tread has radial length {_fmt(p['L'])}, width {_fmt(p['W'])},\nand thickness {_fmt(p['T'])}."),
            ("differ in elevation by 5.", f"differ in elevation by {_fmt(p['RISE'])}."),
            ("There are 24 treads", f"There are {_fmt(p['PER_REV'])} treads"),
            ("the staircase winds counterclockwise", f"the staircase winds {HAND_WORD[p['HAND']]}"),
            ("The first tread points along positive X", f"The first tread points along {AXIS_WORD[p['FIRST']]}")])

    @staticmethod
    def reference(p, n):
        out = [{"type": "cylinder", "center": [0, 0, n * p["RISE"] / 2], "radius": p["R"], "height": n * p["RISE"], "axis": [0, 0, 1]}]
        for k in range(n):
            a = p["FIRST"] + p["HAND"] * 360.0 / p["PER_REV"] * k
            z = p["T"] / 2 + k * p["RISE"]
            out.append({"type": "beam", "start": _pol(p["R"], a, z), "end": _pol(p["R"] + p["L"], a, z), "width": p["W"], "height": p["T"]})
        return out


class Planetary:
    family = "Planetary Array"
    DEFAULT = dict(RS=10, RP=3, H=2, HAND=1, FIRST=0)
    VARIANTS = [dict(RS=12, RP=4, H=3, HAND=-1, FIRST=90), dict(RS=8, RP=2.5, H=1.5, HAND=1, FIRST=270)]

    @staticmethod
    def prompt(body, p):
        return _sub(body, [
            ("sun cylinder of radius 10 and height 2", f"sun cylinder of radius {_fmt(p['RS'])} and height {_fmt(p['H'])}"),
            ("planet cylinders of radius 3 and height 2", f"planet cylinders of radius {_fmt(p['RP'])} and height {_fmt(p['H'])}"),
            ("the first lies on positive X, followed\ncounterclockwise", f"the first lies on {AXIS_WORD[p['FIRST']]}, followed\n{HAND_WORD[p['HAND']]}")])

    @staticmethod
    def reference(p, n):
        out = [{"type": "cylinder", "center": [0, 0, 0], "radius": p["RS"], "height": p["H"], "axis": [0, 0, 1]}]
        for k in range(n):
            out.append({"type": "cylinder", "center": _pol(p["RS"] + p["RP"], p["FIRST"] + p["HAND"] * 360.0 * k / n),
                        "radius": p["RP"], "height": p["H"], "axis": [0, 0, 1]})
        return out


class DNA:
    family = "DNA Helix"
    DEFAULT = dict(RB=20, RS=3, SEC=2, RISE=4, PER_REV=10, HAND=1)
    VARIANTS = [dict(RB=15, RS=2.5, SEC=1.5, RISE=3, PER_REV=12, HAND=-1), dict(RB=25, RS=4, SEC=2.5, RISE=5, PER_REV=8, HAND=1)]

    @staticmethod
    def prompt(body, p):
        return _sub(body, [
            ("Model a right-handed double helix", f"Model a {'right' if p['HAND'] > 0 else 'left'}-handed double helix"),
            ("backbone locus has radius 20", f"backbone locus has radius {_fmt(p['RB'])}"),
            ("sphere nodes of radius 3 joined by a beam of\nsquare section 2",
             f"sphere nodes of radius {_fmt(p['RS'])} joined by a beam of\nsquare section {_fmt(p['SEC'])}"),
            ("Successive levels rise by 4, winding\ncounterclockwise", f"Successive levels rise by {_fmt(p['RISE'])}, winding\n{HAND_WORD[p['HAND']]}"),
            ("with 10 levels per revolution", f"with {_fmt(p['PER_REV'])} levels per revolution")])

    @staticmethod
    def reference(p, n):
        out = []
        for l in range(n):
            a = p["HAND"] * 360.0 / p["PER_REV"] * l; z = l * p["RISE"]
            s1, s2 = _pol(p["RB"], a, z), _pol(p["RB"], a + 180, z)
            out += [{"type": "sphere", "radius": p["RS"], "center": s1}, {"type": "sphere", "radius": p["RS"], "center": s2},
                    {"type": "beam", "width": p["SEC"], "height": p["SEC"], "start": s1, "end": s2}]
        return out


class Domino:
    family = "Domino Ring"
    DEFAULT = dict(RP=4, HP=30, RC=80, PAIR=6, HAND=1, FIRST=0)
    VARIANTS = [dict(RP=3, HP=24, RC=60, PAIR=8, HAND=-1, FIRST=90), dict(RP=5, HP=36, RC=100, PAIR=5, HAND=1, FIRST=180)]

    @staticmethod
    def prompt(body, p):
        return _sub(body, [
            ("circle of radius 80", f"circle of radius {_fmt(p['RC'])}"),
            ("radius 4 and height 30", f"radius {_fmt(p['RP'])} and height {_fmt(p['HP'])}"),
            ("is 6 degrees", f"is {_fmt(p['PAIR'])} degrees"),
            ("bisector points along positive X; arches follow counterclockwise",
             f"bisector points along {AXIS_WORD[p['FIRST']]}; arches follow {HAND_WORD[p['HAND']]}")])

    @staticmethod
    def reference(p, n):
        out = []
        for a in range(n):
            phi = p["FIRST"] + p["HAND"] * 360.0 * a / n
            cw, ccw = _pol(p["RC"], phi - p["PAIR"] / 2, p["HP"] / 2), _pol(p["RC"], phi + p["PAIR"] / 2, p["HP"] / 2)
            out += [{"type": "cylinder", "radius": p["RP"], "height": p["HP"], "center": cw, "axis": [0, 0, 1]},
                    {"type": "cylinder", "radius": p["RP"], "height": p["HP"], "center": ccw, "axis": [0, 0, 1]},
                    {"type": "beam", "width": 2 * p["RP"], "height": 1.5 * p["RP"], "start": cw[:2] + [p["HP"]], "end": ccw[:2] + [p["HP"]]}]
        return out


class Voxel:
    family = "Voxel Grid"
    DEFAULT = dict(S=10, GAP=2)
    VARIANTS = [dict(S=8, GAP=3), dict(S=12, GAP=1.5)]

    @staticmethod
    def prompt(body, p):
        return _sub(body, [("cube of side 10", f"cube of side {_fmt(p['S'])}"), ("gap\nof 2", f"gap\nof {_fmt(p['GAP'])}")])

    @staticmethod
    def reference(p, n):
        step = p["S"] + p["GAP"]
        return [{"type": "box", "center": [p["S"] / 2 + i * step, p["S"] / 2 + j * step, p["S"] / 2 + k * step], "size": [p["S"]] * 3}
                for k in range(n) for j in range(n) for i in range(n)]


class Bearing:
    family = "Ball Bearing Assembly"
    DEFAULT = dict(RB=3, RI=12.5, RO=17, W=15, ROUT=26, HAND=1, FIRST=0)
    VARIANTS = [dict(RB=4, RI=15, RO=20, W=18, ROUT=32, HAND=-1, FIRST=90), dict(RB=2.5, RI=10, RO=14, W=12, ROUT=22, HAND=1, FIRST=180)]

    @staticmethod
    def prompt(body, p):
        return _sub(body, [
            ("rolling\nballs of radius 3", f"rolling\nballs of radius {_fmt(p['RB'])}"),
            ("bore radius 12.5 and outer radius 17", f"bore radius {_fmt(p['RI'])} and outer radius {_fmt(p['RO'])}"),
            ("axial\nwidth is 15", f"axial\nwidth is {_fmt(p['W'])}"),
            ("outer race has outer radius 26", f"outer race has outer radius {_fmt(p['ROUT'])}"),
            ("first along positive X then\ncounterclockwise", f"first along {AXIS_WORD[p['FIRST']]} then\n{HAND_WORD[p['HAND']]}")])

    @staticmethod
    def reference(p, n):
        out = [{"type": "pipe", "center": [0, 0, 0], "inner_radius": p["RI"], "outer_radius": p["RO"], "height": p["W"], "axis": [0, 0, 1]},
               {"type": "pipe", "center": [0, 0, 0], "inner_radius": p["RO"] + 2 * p["RB"], "outer_radius": p["ROUT"], "height": p["W"], "axis": [0, 0, 1]}]
        for k in range(n):
            out.append({"type": "sphere", "center": _pol(p["RO"] + p["RB"], p["FIRST"] + p["HAND"] * 360.0 * k / n), "radius": p["RB"]})
        return out


class Flanged:
    family = "Flanged Pipe Joint"
    DEFAULT = dict(BI=25, BO=30, BL=60, FO=50, FT=8, GAP=2, BR=3, BLEN=20, BC=40, NT=2, FIRST="Y")
    VARIANTS = [dict(BI=20, BO=24, BL=50, FO=42, FT=6, GAP=3, BR=2.5, BLEN=18, BC=33, NT=1.5, FIRST="Z"),
                dict(BI=30, BO=36, BL=70, FO=60, FT=10, GAP=1.5, BR=4, BLEN=24, BC=48, NT=2.5, FIRST="Y")]

    @staticmethod
    def prompt(body, p):
        first = ("positive Y; index toward positive Z around X" if p["FIRST"] == "Y" else "positive Z; index toward negative Y around X")
        return _sub(body, [
            ("bore radius 25, outer radius 30 and length 60", f"bore radius {_fmt(p['BI'])}, outer radius {_fmt(p['BO'])} and length {_fmt(p['BL'])}"),
            ("outer radius 50,\nthickness 8", f"outer radius {_fmt(p['FO'])},\nthickness {_fmt(p['FT'])}"),
            ("axial\ngap of 2", f"axial\ngap of {_fmt(p['GAP'])}"),
            ("radius 3 and length 20", f"radius {_fmt(p['BR'])} and length {_fmt(p['BLEN'])}"),
            ("bolt circle of radius 40", f"bolt circle of radius {_fmt(p['BC'])}"),
            ("tube radius 2;", f"tube radius {_fmt(p['NT'])};"),
            ("positive Y; index toward positive Z around X", first)])

    @staticmethod
    def reference(p, n):
        g, t, L = p["GAP"] / 2, p["FT"], p["BL"]
        out = [{"type": "pipe", "center": [-(g + t + L / 2), 0, 0], "inner_radius": p["BI"], "outer_radius": p["BO"], "height": L, "axis": [1, 0, 0]},
               {"type": "pipe", "center": [g + t + L / 2, 0, 0], "inner_radius": p["BI"], "outer_radius": p["BO"], "height": L, "axis": [1, 0, 0]},
               {"type": "pipe", "center": [-(g + t / 2), 0, 0], "inner_radius": p["BI"], "outer_radius": p["FO"], "height": t, "axis": [1, 0, 0]},
               {"type": "pipe", "center": [g + t / 2, 0, 0], "inner_radius": p["BI"], "outer_radius": p["FO"], "height": t, "axis": [1, 0, 0]}]
        a0 = 0.0 if p["FIRST"] == "Y" else 90.0
        for k in range(n):
            a = math.radians(a0 + 360.0 * k / n)          # measured from +Y toward +Z about X
            y, z = p["BC"] * math.cos(a), p["BC"] * math.sin(a)
            out += [{"type": "cylinder", "center": [0, y, z], "radius": p["BR"], "height": p["BLEN"], "axis": [1, 0, 0]},
                    {"type": "torus", "center": [g + t, y, z], "ring_radius": p["BR"], "tube_radius": p["NT"], "axis": [1, 0, 0]}]
        return out


class Bridge:
    family = "Suspension Bridge"
    DEFAULT = dict(SPAN=100, SEC=1, DZ=10, TR=2, TH=100, INSET=2)
    VARIANTS = [dict(SPAN=120, SEC=1.5, DZ=12, TR=3, TH=90, INSET=4), dict(SPAN=80, SEC=0.8, DZ=8, TR=1.5, TH=70, INSET=3)]

    @staticmethod
    def prompt(body, p):
        return _sub(body, [
            ("span 100 and square section 1", f"span {_fmt(p['SPAN'])} and square section {_fmt(p['SEC'])}"),
            ("centerline 10\nabove", f"centerline {_fmt(p['DZ'])}\nabove"),
            ("radius 2 and height 100", f"radius {_fmt(p['TR'])} and height {_fmt(p['TH'])}"),
            ("cable beams of square section 1", f"cable beams of square section {_fmt(p['SEC'])}"),
            ("outermost attachment is 2 inward", f"outermost attachment is {_fmt(p['INSET'])} inward"),
            ("innermost is\n2 from", f"innermost is\n{_fmt(p['INSET'])} from")])

    @staticmethod
    def reference(p, n):
        h = p["SPAN"] / 2
        out = [{"type": "beam", "start": [-h, 0, p["DZ"]], "end": [h, 0, p["DZ"]], "width": p["SEC"], "height": p["SEC"]},
               {"type": "cylinder", "center": [-h, 0, p["TH"] / 2], "radius": p["TR"], "height": p["TH"], "axis": [0, 0, 1]},
               {"type": "cylinder", "center": [h, 0, p["TH"] / 2], "radius": p["TR"], "height": p["TH"], "axis": [0, 0, 1]}]
        a0, a1 = -h + p["INSET"], -p["INSET"]
        xs = [a0 + (a1 - a0) * i / (n - 1) for i in range(n)]
        for x in xs:
            out.append({"type": "beam", "start": [-h, 0, p["TH"]], "end": [x, 0, p["DZ"]], "width": p["SEC"], "height": p["SEC"]})
        for x in xs[::-1]:
            out.append({"type": "beam", "start": [h, 0, p["TH"]], "end": [-x, 0, p["DZ"]], "width": p["SEC"], "height": p["SEC"]})
        return out


FAMILIES = [Staircase, Planetary, DNA, Domino, Voxel, Bearing, Flanged, Bridge]
BY_NAME = {F.family: F for F in FAMILIES}


def build_case(F, level: int, params: dict, tag: str) -> dict:
    main = next(c for c in CASES.load() if c["family"] == F.family and c["level"] == level)
    body = F.prompt(main["prompt_body"], params)
    ref = CASES._canonical_reference(F.reference(params, main["scale"]))
    c = copy.deepcopy(main)
    c.update(prompt_body=body, prompt=body + "\n" + CASES.SCHEMA_CONTRACT, reference=ref, n_parts=len(ref), params=params,
             case_id=f"{main['case_id']}_heldout_{tag}", split="heldout", patches=main["patches"] + [f"heldout:{tag}"])
    c["prompt_sha256"] = __import__("hashlib").sha256(c["prompt"].encode()).hexdigest()[:16]
    return c


def build_all() -> list[dict]:
    return [build_case(F, lvl, v, f"v{i + 1}") for F in FAMILIES for i, v in enumerate(F.VARIANTS) for lvl in (1, 2, 3)]


def export(path=None):
    import json
    path = path or CASES.ROOT / "v2" / "data" / "heldout_v2.jsonl"
    with open(path, "w") as fh:
        for c in build_all():
            fh.write(json.dumps(c) + "\n")
    return path
