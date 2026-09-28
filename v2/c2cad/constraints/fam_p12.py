"""Constraint builders, Phases 1-2. Each constraint quotes the prompt clause it enforces.

Design rule ("anchor once, relate the rest"): the first element of a pattern is checked
against the prompt's anchor; every other element is checked relative to the model's own
previous element or mating part. A model with a correct pattern but a wrong anchor
therefore loses only the anchor constraints.
"""
from __future__ import annotations

import math

import numpy as np

from .core import (Ctx, Missing, V, ang, axis, dist_point_line, endpoints_match, ends, need, polar_deg,
                   radius, unit, wrap180, zmax, zmin, all_pairs)

Z = V(0, 0, 1); X = V(1, 0, 0); Y = V(0, 1, 0)


def params(case, defaults):
    """Family constants: the held-out split overrides them through case["params"]."""
    return {**defaults, **case.get("params", {})}


def _dir(deg):
    return V(math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0)


def _beam_bottom(s):
    need(s, "beam")
    return float(s.center[2]) - s.thickness / 2.0


# ---------------------------------------------------------------------------
def spiral_staircase(case, ref, X_: Ctx):
    q = params(case, dict(R=5, L=20, W=5, T=2, RISE=5, PER_REV=24, HAND=1, FIRST=0))
    n = case["scale"]; R, L, W, T, RISE = q["R"], q["L"], q["W"], q["T"], q["RISE"]
    STEP = q["HAND"] * 360.0 / q["PER_REV"]
    C = {k: f"staircase:{k}" for k in ("types", "pillar", "tread_dims", "inner", "radial", "rise", "turn", "first", "pillar_top")}
    X_.type_is(0, "cylinder", C["types"])
    treads = list(range(1, n + 1))
    for t in treads:
        X_.type_is(t, "beam", C["types"])
    X_.eq_rel("pillar radius", "dimension", (0,), lambda P: radius(need(P[0], "cylinder")), R, C["pillar"])
    X_.zero_deg("pillar axis Z", "orientation", (0,), lambda P: ang(axis(P[0]), Z, False), C["pillar"])
    X_.zero_len("pillar on origin axis", "anchor", (0,), lambda P: math.hypot(*P[0].center[:2]), C["pillar"])
    X_.zero_len("pillar on ground", "anchor", (0,), lambda P: zmin(P[0]), C["pillar"])

    def inner_outer(P, t):
        a, b = ends(need(P[t], "beam"))
        pc = P[0].center
        da, db = math.hypot(*(a - pc)[:2]), math.hypot(*(b - pc)[:2])
        return (a, b) if da <= db else (b, a)

    for t in treads:
        X_.eq_rel(f"tread{t} length", "dimension", (t,), lambda P, t=t: P[t].length, L, C["tread_dims"])
        X_.eq_rel(f"tread{t} width", "dimension", (t,), lambda P, t=t: P[t].width, W, C["tread_dims"])
        X_.eq_rel(f"tread{t} thickness", "dimension", (t,), lambda P, t=t: P[t].thickness, T, C["tread_dims"])
        # "Each tread's inner centerline endpoint meets the pillar's cylindrical surface"
        X_.zero_len(f"tread{t} inner on pillar surface", "mate", (0, t),
                    lambda P, t=t: math.hypot(*(inner_outer(P, t)[0] - P[0].center)[:2]) - radius(P[0]), C["inner"])
        # "... its outer endpoint lies radially outward" (horizontal, radial direction)
        def radial_err(P, t=t):
            a, b = inner_outer(P, t)
            d = b - a
            rad = a - P[0].center; rad[2] = 0
            return max(ang(d, rad), ang(d, d * V(1, 1, 0)))
        X_.zero_deg(f"tread{t} radial+horizontal", "orientation", (0, t), radial_err, C["radial"])
    X_.zero_len("first tread on ground", "anchor", (1,), lambda P: _beam_bottom(P[1]), C["first"])
    X_.zero_deg("first tread along the anchor direction", "anchor", (0, 1),
                lambda P: ang(inner_outer(P, 1)[1] - inner_outer(P, 1)[0], _dir(q["FIRST"])), C["first"])
    for a, b in zip(treads, treads[1:]):
        X_.eq_len(f"rise {a}->{b}", "pattern", (a, b), lambda P, a=a, b=b: _beam_bottom(P[b]) - _beam_bottom(P[a]),
                  RISE, C["rise"])
        def turn(P, a=a, b=b):
            da = inner_outer(P, a); db = inner_outer(P, b)
            return wrap180(polar_deg(db[1] - db[0]) - polar_deg(da[1] - da[0]) - STEP)
        X_.zero_deg(f"turn {a}->{b} +15 ccw", "pattern", (0, a, b), turn, C["turn"])
    last = treads[-1]
    X_.zero_len("pillar top one rise above last tread bottom", "mate", (0, last),
                lambda P: zmax(P[0]) - (_beam_bottom(P[last]) + RISE), C["pillar_top"])


# ---------------------------------------------------------------------------
def cannonball_pyramid(case, ref, X_: Ctx):
    R = 10.0
    C = {k: f"pyramid:{k}" for k in ("types", "radius", "ground", "corner", "edge", "tangent", "no_overlap")}
    ids = list(range(len(ref)))
    for i in ids:
        X_.type_is(i, "sphere", C["types"])
        X_.eq_rel(f"s{i} radius", "dimension", (i,), lambda P, i=i: radius(need(P[i], "sphere")), R, C["radius"])
    base = [i for i in ids if abs(ref[i].center[2] - R) < 1e-6]
    for i in base:
        X_.zero_len(f"s{i} rests on ground", "anchor", (i,), lambda P, i=i: zmin(P[i]), C["ground"])
    X_.zero_len("corner sphere over origin", "anchor", (0,), lambda P: math.hypot(*P[0].center[:2]), C["corner"])
    X_.zero_deg("base edge along +X", "anchor", (0, 1), lambda P: ang((P[1].center - P[0].center) * V(1, 1, 0), X), C["edge"])
    # tangent pairs = role pairs the prompt's close packing makes touch (topology from the reference)
    for i, j in all_pairs(ids):
        d = float(np.linalg.norm(ref[i].center - ref[j].center))
        if abs(d - 2 * R) < 1e-6:
            X_.zero_len(f"tangent {i}-{j}", "mate", (i, j),
                        lambda P, i=i, j=j: np.linalg.norm(P[i].center - P[j].center) - P[i].radius - P[j].radius,
                        C["tangent"])
        else:
            X_.add(f"no overlap {i}-{j}", "mate", (i, j),
                   lambda P, i=i, j=j: max(0.0, P[i].radius + P[j].radius - np.linalg.norm(P[i].center - P[j].center)),
                   C["no_overlap"])


# ---------------------------------------------------------------------------
def voxel_grid(case, ref, X_: Ctx):
    q = params(case, dict(S=10, GAP=2))
    n = case["scale"]; S, GAP = q["S"], q["GAP"]
    C = {k: f"voxel:{k}" for k in ("types", "cube", "corner", "gap", "aligned")}
    idx = lambda i, j, k: i + n * j + n * n * k      # "Index X fastest, then Y, then Z"
    for r in range(len(ref)):
        X_.type_is(r, "box", C["types"])
        for a in range(3):
            X_.eq_rel(f"b{r} side{a}", "dimension", (r,), lambda P, r=r, a=a: need(P[r], "box").size[a], S, C["cube"])
    X_.zero_len("grid min corner at origin", "anchor", (0,), lambda P: np.linalg.norm(P[0].bbox()[0]), C["corner"])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = idx(i, j, k)
                for ax_, nb in ((0, (i + 1, j, k)), (1, (i, j + 1, k)), (2, (i, j, k + 1))):
                    if max(nb) >= n:
                        continue
                    b = idx(*nb)
                    X_.eq_len(f"gap {a}-{b}", "mate", (a, b),
                              lambda P, a=a, b=b, ax_=ax_: P[b].bbox()[0][ax_] - P[a].bbox()[1][ax_], GAP, C["gap"])
                    X_.zero_len(f"aligned {a}-{b}", "pattern", (a, b),
                                lambda P, a=a, b=b, ax_=ax_: np.linalg.norm(np.delete(P[b].center - P[a].center, ax_)),
                                C["aligned"])


# ---------------------------------------------------------------------------
def domino_ring(case, ref, X_: Ctx):
    q = params(case, dict(RP=4, HP=30, RC=80, PAIR=6, HAND=1, FIRST=0))
    n = case["scale"]; RP, HP, RC, PAIR = q["RP"], q["HP"], q["RC"], q["PAIR"]
    C = {k: f"domino:{k}" for k in ("types", "pillar", "ground", "circle", "pair", "arches", "first", "lintel_ends", "lintel_dims")}
    for a in range(n):
        cw, ccw, lt = 3 * a, 3 * a + 1, 3 * a + 2
        for p in (cw, ccw):
            X_.type_is(p, "cylinder", C["types"])
            X_.eq_rel(f"p{p} radius", "dimension", (p,), lambda P, p=p: radius(need(P[p], "cylinder")), RP, C["pillar"])
            X_.eq_rel(f"p{p} height", "dimension", (p,), lambda P, p=p: P[p].height, HP, C["pillar"])
            X_.zero_deg(f"p{p} vertical", "orientation", (p,), lambda P, p=p: ang(axis(P[p]), Z, False), C["pillar"])
            X_.zero_len(f"p{p} on ground", "anchor", (p,), lambda P, p=p: zmin(P[p]), C["ground"])
            X_.eq_len(f"p{p} axis on r=80", "pattern", (p,), lambda P, p=p: math.hypot(*P[p].center[:2]), RC, C["circle"])
        X_.zero_deg(f"arch{a} pair subtends +6 (cw first)", "pattern", (cw, ccw),
                    lambda P, cw=cw, ccw=ccw: wrap180(polar_deg(P[ccw].center) - polar_deg(P[cw].center) - PAIR), C["pair"])
        bis = lambda P, cw=cw, ccw=ccw: polar_deg(unit(unit(P[cw].center * V(1, 1, 0)) + unit(P[ccw].center * V(1, 1, 0))))
        if a == 0:
            X_.zero_deg("first arch bisector on the anchor direction", "anchor", (cw, ccw), lambda P, bis=bis: wrap180(bis(P) - q["FIRST"]), C["first"])
        else:
            pcw, pccw = 3 * (a - 1), 3 * (a - 1) + 1
            pb = lambda P, pcw=pcw, pccw=pccw: polar_deg(unit(unit(P[pcw].center * V(1, 1, 0)) + unit(P[pccw].center * V(1, 1, 0))))
            X_.zero_deg(f"arch{a} +{360 / n:g} ccw", "pattern", (pcw, pccw, cw, ccw),
                        lambda P, bis=bis, pb=pb: wrap180(bis(P) - pb(P) - q["HAND"] * 360.0 / n), C["arches"])
        X_.type_is(lt, "beam", C["types"])
        X_.zero_len(f"lintel{a} joins pillar top centers", "mate", (cw, ccw, lt),
                    lambda P, cw=cw, ccw=ccw, lt=lt: endpoints_match(P[lt], ends(P[cw])[1] if ends(P[cw])[1][2] > ends(P[cw])[0][2] else ends(P[cw])[0],
                                                                     ends(P[ccw])[1] if ends(P[ccw])[1][2] > ends(P[ccw])[0][2] else ends(P[ccw])[0]),
                    C["lintel_ends"])
        X_.eq_rel(f"lintel{a} width = pillar diameter", "dimension", (lt,), lambda P, lt=lt: need(P[lt], "beam").width, 2 * RP, C["lintel_dims"])
        X_.eq_rel(f"lintel{a} thickness = 1.5 r", "dimension", (lt,), lambda P, lt=lt: need(P[lt], "beam").thickness, 1.5 * RP, C["lintel_dims"])


# ---------------------------------------------------------------------------
def dna_helix(case, ref, X_: Ctx):
    q = params(case, dict(RB=20, RS=3, SEC=2, RISE=4, PER_REV=10, HAND=1))
    n = case["scale"]; RB, RS, SEC, RISE = q["RB"], q["RS"], q["SEC"], q["RISE"]
    STEP = q["HAND"] * 360.0 / q["PER_REV"]
    C = {k: f"dna:{k}" for k in ("types", "dims", "locus", "opposite", "bond", "first", "rise", "turn")}
    for l in range(n):
        a, b, bd = 3 * l, 3 * l + 1, 3 * l + 2
        for s in (a, b):
            X_.type_is(s, "sphere", C["types"])
            X_.eq_rel(f"n{s} radius", "dimension", (s,), lambda P, s=s: radius(need(P[s], "sphere")), RS, C["dims"])
            X_.eq_len(f"n{s} on backbone r=20", "pattern", (s,), lambda P, s=s: math.hypot(*P[s].center[:2]), RB, C["locus"])
        X_.zero_len(f"level{l} diametrically opposite", "pattern", (a, b),
                    lambda P, a=a, b=b: np.linalg.norm((P[a].center + P[b].center) * V(1, 1, 0)) + abs(P[a].center[2] - P[b].center[2]),
                    C["opposite"])
        X_.type_is(bd, "beam", C["types"])
        X_.eq_rel(f"bond{l} section", "dimension", (bd,), lambda P, bd=bd: need(P[bd], "beam").width, SEC, C["dims"])
        X_.zero_len(f"bond{l} joins node centers", "mate", (a, b, bd),
                    lambda P, a=a, b=b, bd=bd: endpoints_match(P[bd], P[a].center, P[b].center), C["bond"])
        if l == 0:
            X_.zero_len("first pair in XY plane", "anchor", (a,), lambda P: abs(P[0].center[2]), C["first"])
            X_.zero_deg("first node on +X", "anchor", (a,), lambda P: wrap180(polar_deg(P[0].center)), C["first"])
        else:
            pa = 3 * (l - 1)
            X_.eq_len(f"rise {l}", "pattern", (pa, a), lambda P, pa=pa, a=a: P[a].center[2] - P[pa].center[2], RISE, C["rise"])
            X_.zero_deg(f"turn {l} +36 ccw", "pattern", (pa, a),
                        lambda P, pa=pa, a=a: wrap180(polar_deg(P[a].center) - polar_deg(P[pa].center) - STEP), C["turn"])


# ---------------------------------------------------------------------------
def flanged_pipe_joint(case, ref, X_: Ctx):
    q = params(case, dict(BI=25, BO=30, BL=60, FO=50, FT=8, GAP=2, BR=3, BLEN=20, BC=40, NT=2, FIRST="Y"))
    n = case["scale"]
    C = {k: f"flange:{k}" for k in ("types", "dims", "axis", "gap", "abut", "bolt_center", "bolt_circle", "first", "spacing",
                                    "nut_coax", "nut_plane", "nut_dims")}
    B0, B1, F0, F1 = 0, 1, 2, 3
    for i in (B0, B1, F0, F1):
        X_.type_is(i, "pipe", C["types"])
        X_.zero_deg(f"part{i} axis X", "orientation", (i,), lambda P, i=i: ang(axis(P[i]), X, False), C["axis"])
        X_.zero_len(f"part{i} on X axis", "anchor", (i,), lambda P, i=i: math.hypot(P[i].center[1], P[i].center[2]), C["axis"])
    for i in (B0, B1):
        X_.eq_rel(f"body{i} bore", "dimension", (i,), lambda P, i=i: need(P[i], "pipe").inner_radius, q["BI"], C["dims"])
        X_.eq_rel(f"body{i} outer", "dimension", (i,), lambda P, i=i: P[i].outer_radius, q["BO"], C["dims"])
        X_.eq_rel(f"body{i} length", "dimension", (i,), lambda P, i=i: P[i].height, q["BL"], C["dims"])
    for i in (F0, F1):
        X_.eq_rel(f"flange{i} bore", "dimension", (i,), lambda P, i=i: need(P[i], "pipe").inner_radius, q["BI"], C["dims"])
        X_.eq_rel(f"flange{i} outer", "dimension", (i,), lambda P, i=i: P[i].outer_radius, q["FO"], C["dims"])
        X_.eq_rel(f"flange{i} thickness", "dimension", (i,), lambda P, i=i: P[i].height, q["FT"], C["dims"])
    xlo = lambda s: float(s.bbox()[0][0]); xhi = lambda s: float(s.bbox()[1][0])
    X_.eq_len("flange axial gap 2", "mate", (F0, F1), lambda P: xlo(P[F1]) - xhi(P[F0]), q["GAP"], C["gap"])
    X_.zero_len("origin mid-gap", "anchor", (F0, F1), lambda P: (xlo(P[F1]) + xhi(P[F0])) / 2, C["gap"])
    X_.zero_len("-X body abuts flange outer face", "mate", (B0, F0), lambda P: xhi(P[B0]) - xlo(P[F0]), C["abut"])
    X_.zero_len("+X body abuts flange outer face", "mate", (B1, F1), lambda P: xlo(P[B1]) - xhi(P[F1]), C["abut"])
    yz = lambda p: math.degrees(math.atan2(p[2], p[1]))      # angle from +Y toward +Z about X
    for k in range(n):
        b, nt = 4 + 2 * k, 5 + 2 * k
        X_.type_is(b, "cylinder", C["types"])
        X_.eq_rel(f"bolt{k} radius", "dimension", (b,), lambda P, b=b: radius(need(P[b], "cylinder")), q["BR"], C["dims"])
        X_.eq_rel(f"bolt{k} length", "dimension", (b,), lambda P, b=b: P[b].height, q["BLEN"], C["dims"])
        X_.zero_deg(f"bolt{k} parallel X", "orientation", (b,), lambda P, b=b: ang(axis(P[b]), X, False), C["axis"])
        X_.zero_len(f"bolt{k} centered across gap", "mate", (F0, F1, b),
                    lambda P, b=b: P[b].center[0] - (xlo(P[F1]) + xhi(P[F0])) / 2, C["bolt_center"])
        X_.eq_len(f"bolt{k} on circle r=40", "pattern", (b,), lambda P, b=b: math.hypot(P[b].center[1], P[b].center[2]), q["BC"], C["bolt_circle"])
        if k == 0:
            X_.zero_deg("first bolt on the anchor direction", "anchor", (b,), lambda P, b=b: wrap180(yz(P[b].center) - (0.0 if q["FIRST"] == "Y" else 90.0)), C["first"])
        else:
            pb = b - 2
            X_.zero_deg(f"bolt{k} +{360 / n:g} toward +Z", "pattern", (pb, b),
                        lambda P, b=b, pb=pb: wrap180(yz(P[b].center) - yz(P[pb].center) - 360.0 / n), C["spacing"])
        X_.type_is(nt, "torus", C["types"])
        X_.eq_rel(f"nut{k} ring = bolt radius", "dimension", (b, nt), lambda P, b=b, nt=nt: need(P[nt], "torus").ring_radius / radius(P[b]), 1.0, C["nut_dims"])
        X_.eq_rel(f"nut{k} tube", "dimension", (nt,), lambda P, nt=nt: P[nt].tube_radius, q["NT"], C["nut_dims"])
        X_.zero_deg(f"nut{k} axis X", "orientation", (nt,), lambda P, nt=nt: ang(axis(P[nt]), X, False), C["nut_coax"])
        X_.zero_len(f"nut{k} on bolt axis", "mate", (b, nt), lambda P, b=b, nt=nt: dist_point_line(P[nt].center, P[b].center, axis(P[b])), C["nut_coax"])
        X_.zero_len(f"nut{k} plane = +X flange outer face", "mate", (F1, nt), lambda P, nt=nt: P[nt].center[0] - xhi(P[F1]), C["nut_plane"])


# ---------------------------------------------------------------------------
def suspension_bridge(case, ref, X_: Ctx):
    q = params(case, dict(SPAN=100, SEC=1, DZ=10, TR=2, TH=100, INSET=2))
    n = case["scale"]; DECK, T0, T1 = 0, 1, 2
    C = {k: f"bridge:{k}" for k in ("types", "deck", "tower", "tower_at_end", "cable_top", "cable_on_deck",
                                    "outermost", "innermost", "uniform", "mirror", "section")}
    X_.type_is(DECK, "beam", C["types"])
    X_.eq_rel("deck span", "dimension", (DECK,), lambda P: P[DECK].length, q["SPAN"], C["deck"])
    X_.eq_rel("deck section", "dimension", (DECK,), lambda P: need(P[DECK], "beam").width, q["SEC"], C["section"])
    X_.zero_deg("deck along X", "orientation", (DECK,), lambda P: ang(P[DECK].direction, X, False), C["deck"])
    X_.zero_len("deck midpoint over origin at z=10", "anchor", (DECK,), lambda P: np.linalg.norm(P[DECK].center - V(0, 0, q["DZ"])), C["deck"])
    deck_end = lambda P, sgn: max(ends(P[DECK]), key=lambda p: sgn * p[0])
    for t, sgn in ((T0, -1), (T1, 1)):
        X_.type_is(t, "cylinder", C["types"])
        X_.eq_rel(f"tower{t} radius", "dimension", (t,), lambda P, t=t: radius(need(P[t], "cylinder")), q["TR"], C["tower"])
        X_.eq_rel(f"tower{t} height", "dimension", (t,), lambda P, t=t: P[t].height, q["TH"], C["tower"])
        X_.zero_deg(f"tower{t} vertical", "orientation", (t,), lambda P, t=t: ang(axis(P[t]), Z, False), C["tower"])
        X_.zero_len(f"tower{t} on ground", "anchor", (t,), lambda P, t=t: zmin(P[t]), C["tower"])
        X_.zero_len(f"tower{t} at deck end", "mate", (DECK, t),
                    lambda P, t=t, sgn=sgn: dist_point_line(deck_end(P, sgn), P[t].center, axis(P[t])), C["tower_at_end"])
    halves = [(T0, list(range(3, 3 + n)), -1), (T1, list(range(3 + n, 3 + 2 * n)), 1)]
    top = lambda s: max(ends(s), key=lambda p: p[2])
    for t, cab, sgn in halves:
        att = lambda P, c, t=t: min(ends(need(P[c], "beam")), key=lambda p: np.linalg.norm(p - top(P[t])) * -1)
        for c in cab:
            X_.type_is(c, "beam", C["types"])
            X_.eq_rel(f"cable{c} section", "dimension", (c,), lambda P, c=c: need(P[c], "beam").width, q["SEC"], C["section"])
            X_.zero_len(f"cable{c} starts at tower top center", "mate", (t, c),
                        lambda P, c=c, t=t: min(np.linalg.norm(p - top(P[t])) for p in ends(need(P[c], "beam"))), C["cable_top"])
            X_.zero_len(f"cable{c} ends on deck centerline", "mate", (DECK, t, c),
                        lambda P, c=c, att=att: dist_point_line(att(P, c), P[DECK].center, P[DECK].direction), C["cable_on_deck"])
        # order: "cables from negative X to positive X" within each half
        outer_c, inner_c = (cab[0], cab[-1]) if sgn < 0 else (cab[-1], cab[0])
        X_.eq_len(f"half{sgn} outermost 2 inboard of tower axis", "pattern", (DECK, t, outer_c),
                  lambda P, t=t, oc=outer_c, att=att: abs(att(P, oc)[0] - P[t].center[0]), q["INSET"], C["outermost"])
        X_.eq_len(f"half{sgn} innermost 2 from midpoint", "pattern", (DECK, t, inner_c),
                  lambda P, t=t, ic=inner_c, att=att: abs(att(P, ic)[0] - P[DECK].center[0]), q["INSET"], C["innermost"])
        for a, b, c in zip(cab, cab[1:], cab[2:]):
            X_.zero_len(f"uniform {a},{b},{c}", "pattern", (t, a, b, c),
                        lambda P, a=a, b=b, c=c, att=att: (att(P, c)[0] - att(P, b)[0]) - (att(P, b)[0] - att(P, a)[0]), C["uniform"])
    for i in range(n):   # mirror: i-th negative cable <-> (n-1-i)-th positive cable
        a, b = 3 + i, 3 + 2 * n - 1 - i
        mirror = lambda P, a=a, b=b: max(endpoints_match(P[b], ends(P[a])[0] * V(-1, 1, 1), ends(P[a])[1] * V(-1, 1, 1)), 0.0)
        X_.zero_len(f"mirror {a}<->{b}", "pattern", (a, b), mirror, C["mirror"])


# ---------------------------------------------------------------------------
def planetary_array(case, ref, X_: Ctx):
    q = params(case, dict(RS=10, RP=3, H=2, HAND=1, FIRST=0))
    n = case["scale"]
    C = {k: f"planetary:{k}" for k in ("types", "dims", "axes", "plane", "sun", "tangent", "first", "spacing")}
    for i in range(n + 1):
        X_.type_is(i, "cylinder", C["types"])
        X_.eq_rel(f"c{i} radius", "dimension", (i,), lambda P, i=i: radius(need(P[i], "cylinder")), q["RS"] if i == 0 else q["RP"], C["dims"])
        X_.eq_rel(f"c{i} height", "dimension", (i,), lambda P, i=i: P[i].height, q["H"], C["dims"])
        X_.zero_deg(f"c{i} axis Z", "orientation", (i,), lambda P, i=i: ang(axis(P[i]), Z, False), C["axes"])
        X_.zero_len(f"c{i} center in XY plane", "anchor", (i,), lambda P, i=i: abs(P[i].center[2]), C["plane"])
    X_.zero_len("sun at origin", "anchor", (0,), lambda P: np.linalg.norm(P[0].center), C["sun"])
    for p in range(1, n + 1):
        X_.zero_len(f"planet{p} tangent to sun", "mate", (0, p),
                    lambda P, p=p: np.linalg.norm((P[p].center - P[0].center) * V(1, 1, 0)) - radius(P[0]) - radius(P[p]), C["tangent"])
        if p == 1:
            X_.zero_deg("first planet on the anchor direction", "anchor", (1,), lambda P: wrap180(polar_deg(P[1].center) - q["FIRST"]), C["first"])
        else:
            X_.zero_deg(f"planet{p} +{360 / n:g}", "pattern", (p - 1, p),
                        lambda P, p=p: wrap180(polar_deg(P[p].center) - polar_deg(P[p - 1].center) - q["HAND"] * 360.0 / n), C["spacing"])


# ---------------------------------------------------------------------------
def cross_braced_truss(case, ref, X_: Ctx):
    s_n = case["scale"]; H, HALF = 10.0, 5.0
    C = {k: f"truss:{k}" for k in ("types", "section", "column", "corner", "story", "joint")}
    corners = [(-HALF, -HALF), (HALF, -HALF), (HALF, HALF), (-HALF, HALF)]
    for r in range(len(ref)):
        X_.type_is(r, "beam", C["types"])
        X_.eq_rel(f"b{r} section", "dimension", (r,), lambda P, r=r: need(P[r], "beam").width, 1.0, C["section"])
    per = 12
    for s in range(s_n):
        cols = [s * per + k for k in range(4)]
        for k, c in enumerate(cols):
            X_.zero_deg(f"col{c} vertical", "orientation", (c,), lambda P, c=c: ang(P[c].direction, Z, False), C["column"])
            X_.eq_rel(f"col{c} story height", "dimension", (c,), lambda P, c=c: P[c].length, H, C["column"])
            X_.zero_len(f"col{c} at corner {corners[k]}", "anchor", (c,),
                        lambda P, c=c, k=k: math.hypot(P[c].center[0] - corners[k][0], P[c].center[1] - corners[k][1]), C["corner"])
            if s == 0:
                X_.zero_len(f"col{c} on ground", "anchor", (c,), lambda P, c=c: min(p[2] for p in ends(P[c])), C["story"])
            else:
                below = c - per
                X_.zero_len(f"col{c} stacks on col{below}", "mate", (below, c),
                            lambda P, c=c, below=below: np.linalg.norm(min(ends(P[c]), key=lambda p: p[2]) - max(ends(P[below]), key=lambda p: p[2])),
                            C["story"])
    # braces: each end coincides with a column joint (topology taken from the reference roles)
    col_ids = [i for i in range(len(ref)) if abs(ref[i].direction[2]) > 0.999]
    for b in range(len(ref)):
        if b in col_ids:
            continue
        for e in ref[b].start, ref[b].end:
            # the reference column end at this joint
            c, which = next((c, w) for c in col_ids for w, p in enumerate(ends(ref[c])) if np.linalg.norm(p - e) < 1e-6)
            X_.zero_len(f"brace{b} end at col{c}.{which}", "mate", (b, c),
                        lambda P, b=b, c=c, e_top=(ends(ref[c])[which][2] > ref[c].center[2]):
                        min(np.linalg.norm(p - (max(ends(P[c]), key=lambda q: q[2]) if e_top else min(ends(P[c]), key=lambda q: q[2])))
                            for p in ends(need(P[b], "beam"))), C["joint"])


# ---------------------------------------------------------------------------
def fractal_y_tree(case, ref, X_: Ctx):
    g = case["scale"]
    C = {k: f"fractal:{k}" for k in ("types", "section", "trunk", "attach", "half", "turn", "plane")}
    # rebuild DFS tree from the prompt's id rule
    parent, order = {}, []
    def dfs(node_id, depth):
        order.append(node_id)
        if depth == g:
            return
        for _ in range(2):
            child = len(order)
            parent[child] = node_id
            dfs(child, depth + 1)
    dfs(0, 0)
    first_child = {}
    for c, p in parent.items():
        first_child.setdefault(p, c)
    for r in range(len(ref)):
        X_.type_is(r, "beam", C["types"])
        X_.eq_rel(f"b{r} section", "dimension", (r,), lambda P, r=r: need(P[r], "beam").width, 1.0, C["section"])
        X_.zero_len(f"b{r} in XZ plane", "orientation", (r,), lambda P, r=r: max(abs(P[r].start[1]), abs(P[r].end[1])), C["plane"])
    # orientation-aware ends: root start = origin; child start = parent end
    def se(P, r):
        """(start, end) with start = the end nearer the parent tip (root: nearer origin)."""
        a, b = ends(need(P[r], "beam"))
        tip = V(0, 0, 0) if r == 0 else se(P, parent[r])[1]
        return (a, b) if np.linalg.norm(a - tip) <= np.linalg.norm(b - tip) else (b, a)
    X_.zero_len("trunk starts at origin", "anchor", (0,), lambda P: np.linalg.norm(se(P, 0)[0]), C["trunk"])
    X_.zero_deg("trunk points +Z", "anchor", (0,), lambda P: ang(se(P, 0)[1] - se(P, 0)[0], Z), C["trunk"])
    X_.eq_rel("trunk length", "dimension", (0,), lambda P: P[0].length, 100.0, C["trunk"])
    for c, p in parent.items():
        chain = []
        q = c
        while q in parent:
            chain.append(parent[q]); q = parent[q]
        roles = tuple([c] + chain)
        X_.zero_len(f"b{c} meets parent tip", "mate", roles, lambda P, c=c: np.linalg.norm(se(P, c)[0] - se(P, parent[c])[1]), C["attach"])
        X_.eq_rel(f"b{c} half parent", "pattern", roles, lambda P, c=c: P[c].length / P[parent[c]].length, 0.5, C["half"])
        sgn = 1.0 if first_child[p] == c else -1.0
        def turn(P, c=c, sgn=sgn):
            ps, pe = se(P, parent[c]); cs, ce = se(P, c)
            d = unit(pe - ps); th = math.radians(45.0 * sgn)
            want = V(d[0] * math.cos(th) + d[2] * math.sin(th), 0.0, -d[0] * math.sin(th) + d[2] * math.cos(th))
            return ang(ce - cs, want)
        X_.zero_deg(f"b{c} turned {'+' if sgn > 0 else '-'}45", "pattern", roles, turn, C["turn"])


# ---------------------------------------------------------------------------
def bcc_lattice(case, ref, X_: Ctx):
    n = case["scale"]; A = 10.0
    C = {k: f"bcc:{k}" for k in ("types", "dims", "lattice", "corner", "extent", "strut_nodes", "strut_diag", "unique")}
    spheres = [i for i, s in enumerate(ref) if s.type == "sphere"]
    struts = [i for i, s in enumerate(ref) if s.type == "beam"]
    for i in spheres:
        X_.type_is(i, "sphere", C["types"])
        X_.eq_rel(f"n{i} radius", "dimension", (i,), lambda P, i=i: radius(need(P[i], "sphere")), 1.0, C["dims"])
        def on_lattice(P, i=i):
            q = P[i].center / (A / 2.0)
            k = np.round(q)
            par = {int(v) % 2 for v in k}
            inside = np.all(k >= -1e-9) and np.all(k <= 2 * n + 1e-9)
            return float(np.linalg.norm(q - k) * A / 2.0) if (len(par) == 1 and inside) else float("inf")
        X_.zero_len(f"n{i} on BCC site", "pattern", (i,), on_lattice, C["lattice"])
    X_.zero_len("min corner at origin", "anchor", tuple(spheres),
                lambda P: np.linalg.norm(np.min(np.stack([P[i].center for i in spheres]), axis=0)), C["corner"])
    X_.zero_len("max corner", "anchor", tuple(spheres),
                lambda P: np.linalg.norm(np.max(np.stack([P[i].center for i in spheres]), axis=0) - n * A), C["extent"])
    X_.add("corner nodes represented once", "topology", tuple(spheres),
           lambda P: float(sum(1 for a, b in all_pairs(spheres) if np.linalg.norm(P[a].center - P[b].center) < 1e-3)),
           C["unique"], tol=0.0, unit="count")
    for b in struts:
        X_.type_is(b, "beam", C["types"])
        X_.eq_rel(f"s{b} section", "dimension", (b,), lambda P, b=b: need(P[b], "beam").width, 1.0, C["dims"])
        X_.zero_len(f"s{b} ends on nodes", "mate", tuple([b] + spheres),
                    lambda P, b=b: max(min(np.linalg.norm(p - P[i].center) for i in spheres) for p in ends(need(P[b], "beam"))),
                    C["strut_nodes"])
        X_.zero_deg(f"s{b} along a cube diagonal", "pattern", (b,),
                    lambda P, b=b: min(ang(need(P[b], "beam").direction, V(sx, sy, 1), False) for sx in (-1, 1) for sy in (-1, 1)),
                    C["strut_diag"])
        X_.eq_len(f"s{b} body-to-corner length", "pattern", (b,), lambda P, b=b: need(P[b], "beam").length,
                  A * math.sqrt(3) / 2, C["strut_diag"])


# ---------------------------------------------------------------------------
def ball_bearing(case, ref, X_: Ctx):
    q = params(case, dict(RB=3, RI=12.5, RO=17, W=15, ROUT=26, HAND=1, FIRST=0))
    n = case["scale"]
    C = {k: f"bearing:{k}" for k in ("types", "dims", "axis", "center", "tangent", "plane", "first", "spacing")}
    for i in (0, 1):
        X_.type_is(i, "pipe", C["types"])
        X_.zero_deg(f"race{i} axis Z", "orientation", (i,), lambda P, i=i: ang(axis(P[i]), Z, False), C["axis"])
        X_.zero_len(f"race{i} centered at origin", "anchor", (i,), lambda P, i=i: np.linalg.norm(P[i].center), C["center"])
        X_.eq_rel(f"race{i} width", "dimension", (i,), lambda P, i=i: P[i].height, q["W"], C["dims"])
    X_.eq_rel("inner race bore", "dimension", (0,), lambda P: need(P[0], "pipe").inner_radius, q["RI"], C["dims"])
    X_.eq_rel("inner race outer", "dimension", (0,), lambda P: P[0].outer_radius, q["RO"], C["dims"])
    X_.eq_rel("outer race outer", "dimension", (1,), lambda P: need(P[1], "pipe").outer_radius, q["ROUT"], C["dims"])
    for k in range(n):
        b = 2 + k
        X_.type_is(b, "sphere", C["types"])
        X_.eq_rel(f"ball{k} radius", "dimension", (b,), lambda P, b=b: radius(need(P[b], "sphere")), q["RB"], C["dims"])
        X_.zero_len(f"ball{k} in midplane", "anchor", (b,), lambda P, b=b: abs(P[b].center[2]), C["plane"])
        rr = lambda P, b=b: math.hypot(*P[b].center[:2])
        X_.zero_len(f"ball{k} tangent to inner raceway", "mate", (0, b), lambda P, b=b, rr=rr: rr(P) - P[b].radius - P[0].outer_radius, C["tangent"])
        X_.zero_len(f"ball{k} tangent to outer raceway", "mate", (1, b), lambda P, b=b, rr=rr: rr(P) + P[b].radius - need(P[1], "pipe").inner_radius, C["tangent"])
        if k == 0:
            X_.zero_deg("first ball on the anchor direction", "anchor", (b,), lambda P, b=b: wrap180(polar_deg(P[b].center) - q["FIRST"]), C["first"])
        else:
            X_.zero_deg(f"ball{k} +{360 / n:g}", "pattern", (b - 1, b),
                        lambda P, b=b: wrap180(polar_deg(P[b].center) - polar_deg(P[b - 1].center) - q["HAND"] * 360.0 / n), C["spacing"])
