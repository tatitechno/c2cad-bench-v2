"""Constraint builders, Phases 3-4 (engineering and bio-inspired families)."""
from __future__ import annotations

import math

import numpy as np

from .core import (Ctx, Missing, V, ang, axis, dist_point_line, endpoints_match, ends, need, polar_deg,
                   radius, unit, wrap180, zmax, zmin, all_pairs)

Z = V(0, 0, 1); X = V(1, 0, 0); Y = V(0, 1, 0)
PHI = (1 + 5 ** 0.5) / 2


def lo(s, k):
    return float(s.bbox()[0][k])


def hi(s, k):
    return float(s.bbox()[1][k])


def top_center(s):
    return max(ends(s), key=lambda p: p[2])


def bottom_center(s):
    return min(ends(s), key=lambda p: p[2])


def icosa_dirs():
    """Vertices of the prompt's icosahedron: golden rectangles in YZ (long || Z), XY (long || Y), XZ (long || X)."""
    vs = []
    for a in (-1, 1):
        for b in (-1, 1):
            vs += [V(0, a, b * PHI), V(a, b * PHI, 0), V(b * PHI, 0, a)]
    return [unit(v) for v in vs]


def geodesic(levels: int, R: float):
    """Spherical midpoint refinement of the prompt's icosahedron: (vertices (n,3), edge set of index pairs)."""
    verts = [v.copy() for v in icosa_dirs()]
    # faces: triples of mutually adjacent vertices (edge length = min pairwise distance)
    P = np.stack(verts)
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    e = d[d > 1e-9].min()
    adj = (np.abs(d - e) < 1e-6)
    faces = [(i, j, k) for i in range(12) for j in range(i + 1, 12) for k in range(j + 1, 12)
             if adj[i, j] and adj[j, k] and adj[i, k]]
    for _ in range(levels):
        cache, nf = {}, []
        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                verts.append(unit(verts[a] + verts[b])); cache[key] = len(verts) - 1
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    edges = set()
    for a, b, c in faces:
        edges |= {(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))}
    return np.stack(verts) * R, edges


# ---------------------------------------------------------------------------
def furniture(case, ref, X_: Ctx):
    m = len(ref) - 1; P_ = (m - 4) // 2; TOP = 0
    C = {k: f"furniture:{k}" for k in ("types", "top", "legs", "ground", "touch", "inset", "mirror", "edge_line", "stations")}
    X_.type_is(TOP, "box", C["types"])
    for a, v in enumerate((100.0, 60.0, 5.0)):
        X_.eq_rel(f"top size{a}", "dimension", (TOP,), lambda P, a=a: need(P[TOP], "box").size[a], v, C["top"])
    X_.zero_len("top midpoint over origin", "anchor", (TOP,), lambda P: math.hypot(*P[TOP].center[:2]), C["top"])
    legs = list(range(1, m + 1))
    for l in legs:
        X_.type_is(l, "cylinder", C["types"])
        X_.eq_rel(f"leg{l} radius", "dimension", (l,), lambda P, l=l: radius(need(P[l], "cylinder")), 3.0, C["legs"])
        X_.eq_rel(f"leg{l} height", "dimension", (l,), lambda P, l=l: P[l].height, 70.0, C["legs"])
        X_.zero_deg(f"leg{l} vertical", "orientation", (l,), lambda P, l=l: ang(axis(P[l]), Z, False), C["legs"])
        X_.zero_len(f"leg{l} on ground", "anchor", (l,), lambda P, l=l: zmin(P[l]), C["ground"])
        X_.zero_len(f"leg{l} touches underside", "mate", (TOP, l), lambda P, l=l: zmax(P[l]) - lo(P[TOP], 2), C["touch"])
    signs = {1: (1, 1), 2: (-1, 1), 3: (-1, -1), 4: (1, -1)}
    for l, (sx, sy) in signs.items():
        for k, s in ((0, sx), (1, sy)):
            X_.zero_len(f"leg{l} inset 2r from edge{k}", "mate", (TOP, l),
                        lambda P, l=l, k=k, s=s: s * ((hi(P[TOP], k) if s > 0 else -lo(P[TOP], k)) * s - P[l].center[k] * 1.0) * 0
                        + ((hi(P[TOP], k) - P[l].center[k]) if s > 0 else (P[l].center[k] - lo(P[TOP], k))) - 2 * radius(P[l]),
                        C["inset"])
    for p in range(P_):
        a, b = 5 + 2 * p, 6 + 2 * p         # +Y leg, -Y leg
        X_.zero_len(f"pair{p} mirrored across XZ", "pattern", (a, b),
                    lambda P, a=a, b=b: abs(P[a].center[0] - P[b].center[0]) + abs(P[a].center[1] + P[b].center[1]), C["mirror"])
        X_.zero_len(f"pair{p} +Y leg on long-edge leg line", "pattern", (1, a), lambda P, a=a: P[a].center[1] - P[1].center[1], C["edge_line"])
        k = p + 1
        def station(P, a=a, k=k):
            xb_pos = P[1].center[0]                               # +X corner-leg axis
            xb_neg = lo(P[TOP], 0) + 6 * radius(P[1])             # six leg radii from -X edge
            return P[a].center[0] - (xb_pos - k * (xb_pos - xb_neg) / (P_ + 1))
        X_.zero_len(f"pair{p} station", "pattern", (TOP, 1, a), station, C["stations"])


# ---------------------------------------------------------------------------
def pipe_manifold(case, ref, X_: Ctx):
    nb = case["scale"]; H, W, C0, C1 = 0, 1, 2, 3
    C = {k: f"manifold:{k}" for k in ("types", "header", "junctions", "wall", "caps", "branch", "flange", "valve", "bracket")}
    X_.type_is(H, "pipe", C["types"])
    X_.eq_rel("header bore", "dimension", (H,), lambda P: need(P[H], "pipe").inner_radius, 12.0, C["header"])
    X_.eq_rel("header outer", "dimension", (H,), lambda P: P[H].outer_radius, 15.0, C["header"])
    X_.zero_deg("header parallel X", "orientation", (H,), lambda P: ang(axis(P[H]), X, False), C["header"])
    X_.zero_len("header centered above origin at z=50", "anchor", (H,), lambda P: np.linalg.norm(P[H].center - V(0, 0, 50)), C["header"])
    br = [4 + 4 * k for k in range(nb)]
    xs = lambda P: [P[b].center[0] for b in br]
    X_.eq_len("first junction 40 from -X end", "pattern", (H, br[0]), lambda P: P[br[0]].center[0] - lo(P[H], 0), 40.0, C["junctions"])
    X_.eq_len("last junction 40 from +X end", "pattern", (H, br[-1]), lambda P: hi(P[H], 0) - P[br[-1]].center[0], 40.0, C["junctions"])
    for a, b in zip(br, br[1:]):
        X_.eq_len(f"junction spacing {a}-{b}", "pattern", (a, b), lambda P, a=a, b=b: P[b].center[0] - P[a].center[0], 40.0, C["junctions"])
    X_.type_is(W, "box", C["types"])
    X_.eq_rel("wall thickness", "dimension", (W,), lambda P: need(P[W], "box").size[1], 5.0, C["wall"])
    X_.zero_len("wall on ground", "anchor", (W,), lambda P: lo(P[W], 2), C["wall"])
    X_.zero_len("wall front tangent to header rear", "mate", (H, W), lambda P: hi(P[W], 1) - (P[H].center[1] - P[H].outer_radius), C["wall"])
    X_.eq_len("wall -X margin 10", "mate", (H, W), lambda P: lo(P[H], 0) - lo(P[W], 0), 10.0, C["wall"])
    X_.eq_len("wall +X margin 10", "mate", (H, W), lambda P: hi(P[W], 0) - hi(P[H], 0), 10.0, C["wall"])
    X_.eq_len("wall top 20 above header top", "mate", (H, W), lambda P: hi(P[W], 2) - (P[H].center[2] + P[H].outer_radius), 20.0, C["wall"])
    for cp, sgn in ((C0, -1), (C1, 1)):
        X_.type_is(cp, "cylinder", C["types"])
        X_.eq_rel(f"cap{cp} radius = header outer", "dimension", (H, cp), lambda P, cp=cp: radius(need(P[cp], "cylinder")) / P[H].outer_radius, 1.0, C["caps"])
        X_.eq_rel(f"cap{cp} thickness", "dimension", (cp,), lambda P, cp=cp: P[cp].height, 3.0, C["caps"])
        X_.zero_deg(f"cap{cp} coaxial angle", "mate", (H, cp), lambda P, cp=cp: ang(axis(P[cp]), axis(P[H]), False), C["caps"])
        X_.zero_len(f"cap{cp} coaxial offset", "mate", (H, cp), lambda P, cp=cp: dist_point_line(P[cp].center, P[H].center, axis(P[H])), C["caps"])
        X_.zero_len(f"cap{cp} abuts end face", "mate", (H, cp),
                    lambda P, cp=cp, sgn=sgn: (lo(P[cp], 0) - hi(P[H], 0)) if sgn > 0 else (lo(P[H], 0) - hi(P[cp], 0)), C["caps"])
    for k in range(nb):
        b, f, v, k_ = 4 + 4 * k, 5 + 4 * k, 6 + 4 * k, 7 + 4 * k
        X_.type_is(b, "pipe", C["types"])
        for fld, val in (("inner_radius", 6.0), ("outer_radius", 8.0), ("height", 35.0)):
            X_.eq_rel(f"branch{k} {fld}", "dimension", (b,), lambda P, b=b, fld=fld: getattr(need(P[b], "pipe"), fld), val, C["branch"])
        X_.zero_deg(f"branch{k} along +Y", "orientation", (b,), lambda P, b=b: ang(axis(P[b]), Y, False), C["branch"])
        X_.zero_len(f"branch{k} starts at header front plane", "mate", (H, b), lambda P, b=b: lo(P[b], 1) - (P[H].center[1] + P[H].outer_radius), C["branch"])
        X_.zero_len(f"branch{k} at axis elevation", "mate", (H, b), lambda P, b=b: P[b].center[2] - P[H].center[2], C["branch"])
        X_.type_is(f, "pipe", C["types"])
        X_.zero_len(f"flange{k} coaxial", "mate", (b, f), lambda P, b=b, f=f: dist_point_line(P[f].center, P[b].center, axis(P[b])) + ang(axis(P[f]), axis(P[b]), False) * 0.0, C["flange"])
        X_.zero_len(f"flange{k} starts at junction plane", "mate", (b, f), lambda P, b=b, f=f: lo(P[f], 1) - lo(P[b], 1), C["flange"])
        X_.eq_rel(f"flange{k} thickness", "dimension", (f,), lambda P, f=f: P[f].height, 4.0, C["flange"])
        X_.eq_len(f"flange{k} bore clearance 0.3", "mate", (b, f), lambda P, b=b, f=f: need(P[f], "pipe").inner_radius - P[b].outer_radius, 0.3, C["flange"])
        X_.eq_len(f"flange{k} extends 5 beyond branch", "mate", (b, f), lambda P, b=b, f=f: P[f].outer_radius - P[b].outer_radius, 5.0, C["flange"])
        X_.type_is(v, "cylinder", C["types"])
        X_.eq_rel(f"valve{k} radius", "dimension", (v,), lambda P, v=v: radius(need(P[v], "cylinder")), 10.0, C["valve"])
        X_.eq_rel(f"valve{k} length", "dimension", (v,), lambda P, v=v: P[v].height, 10.0, C["valve"])
        X_.zero_len(f"valve{k} coaxial, centered on branch", "mate", (b, v), lambda P, b=b, v=v: np.linalg.norm(P[v].center - P[b].center), C["valve"])
        X_.zero_deg(f"valve{k} axis", "mate", (b, v), lambda P, b=b, v=v: ang(axis(P[v]), axis(P[b]), False), C["valve"])
        X_.type_is(k_, "box", C["types"])
        X_.eq_rel(f"bracket{k} X width", "dimension", (k_,), lambda P, k_=k_: need(P[k_], "box").size[0], 20.0, C["bracket"])
        X_.eq_rel(f"bracket{k} Y depth", "dimension", (k_,), lambda P, k_=k_: need(P[k_], "box").size[1], 10.0, C["bracket"])
        X_.zero_len(f"bracket{k} on ground", "anchor", (k_,), lambda P, k_=k_: lo(P[k_], 2), C["bracket"])
        X_.zero_len(f"bracket{k} under junction, below header axis", "mate", (H, b, k_),
                    lambda P, b=b, k_=k_: math.hypot(P[k_].center[0] - P[b].center[0], P[k_].center[1] - P[H].center[1]), C["bracket"])
        X_.zero_len(f"bracket{k} reaches header underside", "mate", (H, k_), lambda P, k_=k_: hi(P[k_], 2) - (P[H].center[2] - P[H].outer_radius), C["bracket"])


# ---------------------------------------------------------------------------
def axle_bearing(case, ref, X_: Ctx):
    cl = 0.5 * case["scale"]; B, BORE, SH, L, R = 0, 1, 2, 3, 4
    C = {k: f"axle:{k}" for k in ("types", "block", "bore", "shaft", "bearing", "abut")}
    X_.type_is(B, "box", C["types"])
    for a, v in enumerate((80.0, 50.0, 50.0)):
        X_.eq_rel(f"block size{a}", "dimension", (B,), lambda P, a=a: need(P[B], "box").size[a], v, C["block"])
    X_.zero_len("block on ground at origin", "anchor", (B,), lambda P: math.hypot(*P[B].center[:2]) + abs(lo(P[B], 2)), C["block"])
    line = lambda P: (P[B].center, X)
    X_.type_is(BORE, "cylinder", C["types"])
    X_.zero_len("bore through block centroid || X", "mate", (B, BORE), lambda P: dist_point_line(P[BORE].center, P[B].center, X) + 0 * ang(axis(P[BORE]), X, False), C["bore"])
    X_.zero_deg("bore axis X", "orientation", (BORE,), lambda P: ang(axis(P[BORE]), X, False), C["bore"])
    X_.eq_len("bore extends 1 beyond each face", "mate", (B, BORE), lambda P: P[BORE].height - (P[B].size[0] + 2), 0.0, C["bore"])
    X_.zero_len("bore centered", "mate", (B, BORE), lambda P: P[BORE].center[0] - P[B].center[0], C["bore"])
    X_.eq_len("bore radius = shaft + clearance", "mate", (BORE, SH), lambda P: radius(P[BORE]) - radius(P[SH]), cl, C["bore"])
    X_.type_is(SH, "cylinder", C["types"])
    X_.eq_rel("shaft radius", "dimension", (SH,), lambda P: radius(need(P[SH], "cylinder")), 8.0, C["shaft"])
    X_.zero_deg("shaft axis X", "orientation", (SH,), lambda P: ang(axis(P[SH]), X, False), C["shaft"])
    X_.zero_len("shaft concentric with bore", "mate", (B, SH), lambda P: dist_point_line(P[SH].center, P[B].center, X), C["shaft"])
    X_.eq_len("shaft protrudes 20 at -X", "mate", (B, SH), lambda P: lo(P[B], 0) - lo(P[SH], 0), 20.0, C["shaft"])
    X_.eq_len("shaft protrudes 20 at +X", "mate", (B, SH), lambda P: hi(P[SH], 0) - hi(P[B], 0), 20.0, C["shaft"])
    for br, sgn in ((L, -1), (R, 1)):
        X_.type_is(br, "pipe", C["types"])
        X_.eq_rel(f"bearing{br} width", "dimension", (br,), lambda P, br=br: need(P[br], "pipe").height, 12.0, C["bearing"])
        X_.eq_len(f"bearing{br} wall 6", "dimension", (br,), lambda P, br=br: P[br].outer_radius - P[br].inner_radius, 6.0, C["bearing"])
        X_.zero_len(f"bearing{br} zero clearance on shaft", "mate", (SH, br), lambda P, br=br: P[br].inner_radius - radius(P[SH]), C["bearing"])
        X_.zero_len(f"bearing{br} concentric", "mate", (SH, br), lambda P, br=br: dist_point_line(P[br].center, P[SH].center, axis(P[SH])), C["bearing"])
        X_.zero_deg(f"bearing{br} axis", "mate", (SH, br), lambda P, br=br: ang(axis(P[br]), axis(P[SH]), False), C["bearing"])
        X_.zero_len(f"bearing{br} abuts block face outside", "mate", (B, br),
                    lambda P, br=br, sgn=sgn: (lo(P[br], 0) - hi(P[B], 0)) if sgn > 0 else (lo(P[B], 0) - hi(P[br], 0)), C["abut"])


# ---------------------------------------------------------------------------
def armillary(case, ref, X_: Ctx):
    S = case["scale"]; radii = [20.0 * 2 ** k for k in range(S)]
    dirs = icosa_dirs()
    C = {k: f"armillary:{k}" for k in ("types", "rings", "girdle", "rod_dims", "rod_dir", "rod_ends", "rod_unique")}
    plane_axes = (Z, Y, X)             # XY, XZ, YZ rings
    for k in range(S):
        for j, pa in enumerate(plane_axes):
            r = 3 * k + j
            X_.type_is(r, "torus", C["types"])
            X_.eq_rel(f"ring{r} R", "dimension", (r,), lambda P, r=r: need(P[r], "torus").ring_radius, radii[k], C["rings"])
            X_.eq_rel(f"ring{r} tube", "dimension", (r,), lambda P, r=r: P[r].tube_radius, 1.0, C["rings"])
            X_.zero_deg(f"ring{r} plane", "orientation", (r,), lambda P, r=r, pa=pa: ang(axis(P[r]), pa, False), C["rings"])
            X_.zero_len(f"ring{r} centered", "anchor", (r,), lambda P, r=r: np.linalg.norm(P[r].center), C["rings"])
        g = 3 * S + k
        X_.type_is(g, "torus", C["types"])
        X_.eq_rel(f"girdle{k} R", "dimension", (g,), lambda P, g=g: need(P[g], "torus").ring_radius, radii[k], C["girdle"])
        X_.eq_rel(f"girdle{k} tube", "dimension", (g,), lambda P, g=g: P[g].tube_radius, 1.5, C["girdle"])
        X_.zero_deg(f"girdle{k} equatorial", "orientation", (g,), lambda P, g=g: ang(axis(P[g]), Z, False), C["girdle"])
        X_.zero_len(f"girdle{k} centered", "anchor", (g,), lambda P, g=g: np.linalg.norm(P[g].center), C["girdle"])
    base = 4 * S
    for k in range(S - 1):
        rods = list(range(base + 12 * k, base + 12 * (k + 1)))
        for r in rods:
            X_.type_is(r, "cylinder", C["types"])
            X_.eq_rel(f"rod{r} radius", "dimension", (r,), lambda P, r=r: radius(need(P[r], "cylinder")), 1.2, C["rod_dims"])
            X_.zero_deg(f"rod{r} icosahedral direction", "orientation", (r,),
                        lambda P, r=r: min(ang(P[r].center, d) for d in dirs) + ang(axis(P[r]), P[r].center, False), C["rod_dir"])
            X_.zero_len(f"rod{r} ends on shells", "mate", (r,),
                        lambda P, r=r, k=k: max(abs(sorted(np.linalg.norm(p) for p in ends(P[r]))[0] - radii[k]),
                                                abs(sorted(np.linalg.norm(p) for p in ends(P[r]))[1] - radii[k + 1])), C["rod_ends"])
        X_.add(f"pair{k} uses 12 distinct directions", "topology", tuple(rods),
               lambda P, rods=rods: 12 - len({int(np.argmin([ang(P[r].center, d) for d in dirs])) for r in rods}),
               C["rod_unique"], tol=0.0, unit="count")


# ---------------------------------------------------------------------------
def clock(case, ref, X_: Ctx):
    M = case["scale"]; PL, SH, BU, DI = 0, 1, 2, 3
    MK = list(range(4, 4 + M)); MIN, HR, CWM, CWH = 4 + M, 5 + M, 6 + M, 7 + M
    C = {k: f"clock:{k}" for k in ("types", "plate", "shaft", "bushing", "dial", "markers", "first", "spacing",
                                   "minute", "hour", "counterweight")}
    X_.type_is(PL, "cylinder", C["types"])
    X_.eq_rel("plate radius", "dimension", (PL,), lambda P: radius(need(P[PL], "cylinder")), 85.0, C["plate"])
    X_.eq_rel("plate thickness", "dimension", (PL,), lambda P: P[PL].height, 3.0, C["plate"])
    X_.zero_deg("plate axis Z", "orientation", (PL,), lambda P: ang(axis(P[PL]), Z, False), C["plate"])
    X_.zero_len("plate front face at face plane", "anchor", (PL,), lambda P: hi(P[PL], 2) + math.hypot(*P[PL].center[:2]), C["plate"])
    X_.type_is(SH, "cylinder", C["types"])
    X_.eq_rel("shaft radius", "dimension", (SH,), lambda P: radius(need(P[SH], "cylinder")), 4.0, C["shaft"])
    X_.zero_deg("shaft axis Z", "orientation", (SH,), lambda P: ang(axis(P[SH]), Z, False), C["shaft"])
    X_.zero_len("shaft from face", "anchor", (SH,), lambda P: lo(P[SH], 2) + math.hypot(*P[SH].center[:2]), C["shaft"])
    X_.eq_len("shaft projects 18", "dimension", (SH,), lambda P: hi(P[SH], 2), 18.0, C["shaft"])
    X_.type_is(BU, "pipe", C["types"])
    X_.zero_len("bushing concentric from face", "mate", (SH, BU), lambda P: lo(P[BU], 2) + dist_point_line(P[BU].center, P[SH].center, axis(P[SH])), C["bushing"])
    X_.eq_len("bushing projects 14", "dimension", (BU,), lambda P: need(P[BU], "pipe").height, 14.0, C["bushing"])
    X_.eq_rel("bushing outer", "dimension", (BU,), lambda P: P[BU].outer_radius, 7.0, C["bushing"])
    X_.eq_len("bushing clearance 0.3", "mate", (SH, BU), lambda P: P[BU].inner_radius - radius(P[SH]), 0.3, C["bushing"])
    X_.type_is(DI, "torus", C["types"])
    X_.eq_rel("dial ring", "dimension", (DI,), lambda P: need(P[DI], "torus").ring_radius, 78.0, C["dial"])
    X_.eq_rel("dial tube", "dimension", (DI,), lambda P: P[DI].tube_radius, 3.0, C["dial"])
    X_.zero_len("dial central plane = face plane", "anchor", (DI,), lambda P: np.linalg.norm(P[DI].center) + ang(axis(P[DI]), Z, False) * 0, C["dial"])
    X_.zero_deg("dial axis", "orientation", (DI,), lambda P: ang(axis(P[DI]), Z, False), C["dial"])
    inner = lambda s: min(ends(need(s, "beam")), key=lambda p: math.hypot(*p[:2]))
    outer = lambda s: max(ends(need(s, "beam")), key=lambda p: math.hypot(*p[:2]))
    for i, mk in enumerate(MK):
        X_.type_is(mk, "beam", C["types"])
        X_.eq_rel(f"marker{i} length", "dimension", (mk,), lambda P, mk=mk: P[mk].length, 12.0, C["markers"])
        X_.eq_rel(f"marker{i} section", "dimension", (mk,), lambda P, mk=mk: need(P[mk], "beam").width, 2.0, C["markers"])
        X_.eq_len(f"marker{i} inner tip r=60", "pattern", (mk,), lambda P, mk=mk: math.hypot(*inner(P[mk])[:2]), 60.0, C["markers"])
        X_.zero_len(f"marker{i} in face plane", "anchor", (mk,), lambda P, mk=mk: max(abs(P[mk].start[2]), abs(P[mk].end[2])), C["markers"])
        X_.zero_deg(f"marker{i} radial", "orientation", (mk,), lambda P, mk=mk: ang(outer(P[mk]) - inner(P[mk]), inner(P[mk]) * V(1, 1, 0)), C["markers"])
        if i == 0:
            X_.zero_deg("first marker +X", "anchor", (mk,), lambda P, mk=mk: wrap180(polar_deg(P[mk].center)), C["first"])
        else:
            X_.zero_deg(f"marker{i} +{360 / M:g} ccw", "pattern", (mk - 1, mk),
                        lambda P, mk=mk: wrap180(polar_deg(P[mk].center) - polar_deg(P[mk - 1].center) - 360.0 / M), C["spacing"])
    hands = ((MIN, 62.0, (1.5, 3.0), 1.0, 90.0, CWM, "minute"), (HR, 42.0, (2.0, 3.0), 2.0, 120.0, CWH, "hour"))
    for h, Lh, sec, zh, deg, cw, nm in hands:
        X_.type_is(h, "beam", C["types"])
        X_.eq_rel(f"{nm} length", "dimension", (h,), lambda P, h=h: P[h].length, Lh, C[nm])
        X_.eq_rel(f"{nm} section a", "dimension", (h,), lambda P, h=h: min(P[h].width, P[h].thickness), sec[0], C[nm])
        X_.eq_rel(f"{nm} section b", "dimension", (h,), lambda P, h=h: max(P[h].width, P[h].thickness), sec[1], C[nm])
        X_.eq_len(f"{nm} centerline elevation", "anchor", (h,), lambda P, h=h: (P[h].start[2] + P[h].end[2]) / 2, zh, C[nm])
        X_.zero_len(f"{nm} starts at shaft axis", "mate", (SH, h), lambda P, h=h: dist_point_line(inner(P[h]), P[SH].center, axis(P[SH])), C[nm])
        hd = lambda P, h=h: unit((outer(P[h]) - inner(P[h])) * V(1, 1, 0))
        X_.zero_deg(f"{nm} direction", "orientation", (h,), lambda P, hd=hd, deg=deg: wrap180(polar_deg(hd(P)) - deg), C[nm])
        X_.type_is(cw, "cone", C["types"])
        X_.eq_rel(f"{nm} cw base", "dimension", (cw,), lambda P, cw=cw: need(P[cw], "cone").base_radius, 4.0, C["counterweight"])
        X_.eq_rel(f"{nm} cw tip", "dimension", (cw,), lambda P, cw=cw: P[cw].top_radius, 0.5, C["counterweight"])
        X_.eq_rel(f"{nm} cw length", "dimension", (cw,), lambda P, cw=cw: P[cw].height, 10.0, C["counterweight"])
        X_.zero_deg(f"{nm} cw points away (opposite hand)", "orientation", (h, cw), lambda P, cw=cw, hd=hd: ang(axis(P[cw]), -hd(P)), C["counterweight"])
        X_.zero_len(f"{nm} cw base 1 behind axis at hand elevation", "mate", (SH, h, cw),
                    lambda P, h=h, cw=cw, hd=hd: np.linalg.norm(ends(P[cw])[0] - (V(P[SH].center[0], P[SH].center[1], (P[h].start[2] + P[h].end[2]) / 2) - hd(P))),
                    C["counterweight"])


# ---------------------------------------------------------------------------
def gantry(case, ref, X_: Ctx):
    nb = case["scale"]; ncol = nb + 1
    col = lambda i, row: 2 * i + row           # row 0 = -Y (front), 1 = +Y
    RAIL = [2 * ncol, 2 * ncol + 1]
    BR0 = 2 * ncol + 2
    BRIDGE, TROL, DRUM, CAB, HOOK = BR0 + 2 * nb, BR0 + 2 * nb + 1, BR0 + 2 * nb + 2, BR0 + 2 * nb + 3, BR0 + 2 * nb + 4
    C = {k: f"gantry:{k}" for k in ("types", "columns", "ground", "grid", "center", "rails", "braces", "bridge",
                                    "trolley", "drum", "cable", "hook")}
    for i in range(ncol):
        for row in (0, 1):
            c = col(i, row)
            X_.type_is(c, "cylinder", C["types"])
            X_.eq_rel(f"col{c} radius", "dimension", (c,), lambda P, c=c: radius(need(P[c], "cylinder")), 5.0, C["columns"])
            X_.eq_rel(f"col{c} height", "dimension", (c,), lambda P, c=c: P[c].height, 100.0, C["columns"])
            X_.zero_deg(f"col{c} vertical", "orientation", (c,), lambda P, c=c: ang(axis(P[c]), Z, False), C["columns"])
            X_.zero_len(f"col{c} on ground", "anchor", (c,), lambda P, c=c: zmin(P[c]), C["ground"])
            if i > 0:
                p = col(i - 1, row)
                X_.eq_len(f"col{c} bay 40", "pattern", (p, c), lambda P, p=p, c=c: P[c].center[0] - P[p].center[0], 40.0, C["grid"])
                X_.zero_len(f"col{c} row aligned", "pattern", (p, c), lambda P, p=p, c=c: P[c].center[1] - P[p].center[1], C["grid"])
        a, b = col(i, 0), col(i, 1)
        X_.eq_len(f"row spacing {i}", "pattern", (a, b), lambda P, a=a, b=b: P[b].center[1] - P[a].center[1], 80.0, C["grid"])
    allc = [col(i, r) for i in range(ncol) for r in (0, 1)]
    X_.zero_len("gantry centered on origin", "anchor", tuple(allc),
                lambda P: np.linalg.norm(np.mean(np.stack([P[c].center[:2] for c in allc]), axis=0)), C["center"])
    for row, rl in enumerate(RAIL):
        first, last = col(0, row), col(ncol - 1, row)
        X_.type_is(rl, "beam", C["types"])
        X_.eq_rel(f"rail{row} section", "dimension", (rl,), lambda P, rl=rl: need(P[rl], "beam").width, 4.0, C["rails"])
        X_.zero_len(f"rail{row} rests on column tops", "mate", (first, rl), lambda P, rl=rl, first=first: (P[rl].center[2] - P[rl].thickness / 2) - zmax(P[first]), C["rails"])
        X_.zero_len(f"rail{row} ends over outer column axes", "mate", (first, last, rl),
                    lambda P, rl=rl, first=first, last=last: max(min(math.hypot(*(p - P[q].center)[:2]) for p in ends(need(P[rl], "beam"))) for q in (first, last)),
                    C["rails"])
    for bay in range(nb):
        a, b = col(bay, 0), col(bay + 1, 0)
        for j, rising in ((0, True), (1, False)):
            br = BR0 + 2 * bay + j
            X_.type_is(br, "beam", C["types"])
            X_.eq_rel(f"brace{br} section", "dimension", (br,), lambda P, br=br: need(P[br], "beam").width, 2.5, C["braces"])
            def target(P, a=a, b=b, rising=rising):
                pa = bottom_center(P[a]) if rising else top_center(P[a])
                pb = top_center(P[b]) if rising else bottom_center(P[b])
                return pa, pb
            X_.zero_len(f"brace{br} joins column-axis points", "mate", (a, b, br),
                        lambda P, br=br, target=target: endpoints_match(P[br], *target(P)), C["braces"])
    X_.type_is(BRIDGE, "beam", C["types"])
    X_.eq_rel("bridge section", "dimension", (BRIDGE,), lambda P: need(P[BRIDGE], "beam").width, 5.0, C["bridge"])
    X_.zero_len("bridge at gantry midpoint x", "anchor", (BRIDGE,), lambda P: P[BRIDGE].center[0], C["bridge"])
    X_.zero_deg("bridge along Y", "orientation", (BRIDGE,), lambda P: ang(P[BRIDGE].direction, Y, False), C["bridge"])
    X_.zero_len("bridge ends on rail centerlines", "mate", (RAIL[0], RAIL[1], BRIDGE),
                lambda P: max(min(dist_point_line(p, P[r].center, P[r].direction) for r in RAIL) for p in ends(need(P[BRIDGE], "beam"))), C["bridge"])
    X_.type_is(TROL, "box", C["types"])
    for a, v in enumerate((15.0, 12.0, 8.0)):
        X_.eq_rel(f"trolley size{a}", "dimension", (TROL,), lambda P, a=a: need(P[TROL], "box").size[a], v, C["trolley"])
    X_.zero_len("trolley central over bridge", "mate", (BRIDGE, TROL), lambda P: math.hypot(*(P[TROL].center - P[BRIDGE].center)[:2]), C["trolley"])
    X_.zero_len("trolley bottom in rail-top plane", "mate", (RAIL[0], TROL), lambda P: lo(P[TROL], 2) - (P[RAIL[0]].center[2] + P[RAIL[0]].thickness / 2), C["trolley"])
    X_.type_is(DRUM, "cylinder", C["types"])
    X_.eq_rel("drum radius", "dimension", (DRUM,), lambda P: radius(need(P[DRUM], "cylinder")), 6.0, C["drum"])
    X_.eq_rel("drum length", "dimension", (DRUM,), lambda P: P[DRUM].height, 10.0, C["drum"])
    X_.zero_deg("drum axis Y", "orientation", (DRUM,), lambda P: ang(axis(P[DRUM]), Y, False), C["drum"])
    X_.zero_len("drum centered on trolley top", "mate", (TROL, DRUM),
                lambda P: math.hypot(*(P[DRUM].center - P[TROL].center)[:2]) + abs(lo(P[DRUM], 2) - hi(P[TROL], 2)), C["drum"])
    X_.type_is(CAB, "pipe", C["types"])
    for fld, val in (("inner_radius", 1.0), ("outer_radius", 2.0), ("height", 60.0)):
        X_.eq_rel(f"cable {fld}", "dimension", (CAB,), lambda P, fld=fld: getattr(need(P[CAB], "pipe"), fld), val, C["cable"])
    X_.zero_deg("cable vertical", "orientation", (CAB,), lambda P: ang(axis(P[CAB]), Z, False), C["cable"])
    X_.zero_len("cable hangs from trolley bottom midpoint", "mate", (TROL, CAB),
                lambda P: np.linalg.norm(top_center(P[CAB]) - V(P[TROL].center[0], P[TROL].center[1], lo(P[TROL], 2))), C["cable"])
    X_.type_is(HOOK, "cone", C["types"])
    for fld, val in (("base_radius", 4.0), ("top_radius", 0.5), ("height", 10.0)):
        X_.eq_rel(f"hook {fld}", "dimension", (HOOK,), lambda P, fld=fld: getattr(need(P[HOOK], "cone"), fld), val, C["hook"])
    X_.zero_deg("hook axis upward", "orientation", (HOOK,), lambda P: ang(axis(P[HOOK]), Z), C["hook"])
    X_.zero_len("hook tip abuts cable bottom", "mate", (CAB, HOOK), lambda P: np.linalg.norm(ends(P[HOOK])[1] - bottom_center(P[CAB])), C["hook"])


# ---------------------------------------------------------------------------
GOLDEN_ANGLE = 360.0 * (1 - 1 / PHI)


def phyllotaxis(case, ref, X_: Ctx):
    n = case["scale"]; RC = 0
    C = {k: f"phyllotaxis:{k}" for k in ("types", "seed", "radius_law", "first", "angle", "embed", "receptacle")}
    X_.type_is(RC, "cylinder", C["types"])
    X_.eq_rel("receptacle thickness", "dimension", (RC,), lambda P: need(P[RC], "cylinder").height, 1.0, C["receptacle"])
    X_.zero_deg("receptacle axis Z", "orientation", (RC,), lambda P: ang(axis(P[RC]), Z, False), C["receptacle"])
    X_.zero_len("receptacle centered", "anchor", (RC,), lambda P: np.linalg.norm(P[RC].center), C["receptacle"])
    seeds = list(range(1, n + 1))
    X_.zero_len("receptacle radius = farthest seed + 2 seed radii", "mate", tuple([RC] + seeds),
                lambda P: radius(P[RC]) - max(math.hypot(*P[s].center[:2]) + 2 * P[s].radius for s in seeds), C["receptacle"])
    for s in seeds:
        X_.type_is(s, "sphere", C["types"])
        X_.eq_rel(f"seed{s} radius", "dimension", (s,), lambda P, s=s: radius(need(P[s], "sphere")), 3.0, C["seed"])
        X_.eq_len(f"seed{s} Fermat radius", "pattern", (s,), lambda P, s=s: math.hypot(*P[s].center[:2]), 6.6 * math.sqrt(s), C["radius_law"])
        X_.zero_len(f"seed{s} lowest point at receptacle midplane", "mate", (RC, s), lambda P, s=s: zmin(P[s]) - P[RC].center[2], C["embed"])
        if s == 1:
            X_.zero_deg("first seed one golden angle", "anchor", (s,), lambda P: wrap180(polar_deg(P[1].center) - GOLDEN_ANGLE), C["first"])
        else:
            X_.zero_deg(f"seed{s} +golden angle", "pattern", (s - 1, s),
                        lambda P, s=s: wrap180(polar_deg(P[s].center) - polar_deg(P[s - 1].center) - GOLDEN_ANGLE), C["angle"])


# ---------------------------------------------------------------------------
def compound_eye(case, ref, X_: Ctx):
    R = case["scale"]; SUP, NERVE, RING = 0, 1, 2
    C = {k: f"eye:{k}" for k in ("types", "support", "nerve", "ring", "lens", "locus", "polar", "first", "azimuth", "cone", "guide")}
    X_.type_is(SUP, "sphere", C["types"])
    X_.eq_rel("support radius", "dimension", (SUP,), lambda P: radius(need(P[SUP], "sphere")), 50.0, C["support"])
    X_.zero_len("support at origin", "anchor", (SUP,), lambda P: np.linalg.norm(P[SUP].center), C["support"])
    X_.type_is(NERVE, "pipe", C["types"])
    for fld, val in (("inner_radius", 5.0), ("outer_radius", 7.0), ("height", 20.0)):
        X_.eq_rel(f"nerve {fld}", "dimension", (NERVE,), lambda P, fld=fld: getattr(need(P[NERVE], "pipe"), fld), val, C["nerve"])
    X_.zero_len("nerve down Z from equator", "anchor", (NERVE,), lambda P: np.linalg.norm(top_center(P[NERVE])) + ang(axis(P[NERVE]), Z, False) * 0, C["nerve"])
    X_.zero_deg("nerve axis Z", "orientation", (NERVE,), lambda P: ang(axis(P[NERVE]), Z, False), C["nerve"])
    X_.type_is(RING, "torus", C["types"])
    X_.zero_len("ring follows equator", "mate", (SUP, RING),
                lambda P: np.linalg.norm(P[RING].center - P[SUP].center) + abs(P[RING].ring_radius - P[SUP].radius), C["ring"])
    X_.zero_deg("ring axis Z", "orientation", (RING,), lambda P: ang(axis(P[RING]), Z, False), C["ring"])
    X_.eq_rel("ring tube", "dimension", (RING,), lambda P: need(P[RING], "torus").tube_radius, 3.0, C["ring"])
    groups = [(0, 0, 1)] + [(k, j, 6 * k) for k in range(1, R + 1) for j in range(6 * k)]
    g_ids = {}
    for gi, (k, j, cnt) in enumerate(groups):
        L_, Cn, G_ = 3 + 3 * gi, 4 + 3 * gi, 5 + 3 * gi
        g_ids[(k, j)] = (L_, Cn, G_)
        X_.type_is(L_, "sphere", C["types"])
        X_.eq_rel(f"lens{gi} radius", "dimension", (L_,), lambda P, L_=L_: radius(need(P[L_], "sphere")), 4.0, C["lens"])
        X_.zero_len(f"lens{gi} on support locus", "mate", (SUP, L_), lambda P, L_=L_: np.linalg.norm(P[L_].center - P[SUP].center) - P[SUP].radius, C["locus"])
        polar = lambda P, L_=L_: math.degrees(math.acos(max(-1, min(1, unit(P[L_].center)[2]))))
        X_.eq_len(f"lens{gi} polar angle", "pattern", (L_,), lambda P, polar=polar: polar(P), 70.0 * k / R, C["polar"])
        if k > 0:
            if j == 0:
                X_.zero_deg(f"ring{k} first unit +X meridian", "anchor", (L_,), lambda P, L_=L_: wrap180(polar_deg(P[L_].center)), C["first"])
            else:
                pl = g_ids[(k, j - 1)][0]
                X_.zero_deg(f"lens{gi} azimuth step", "pattern", (pl, L_),
                            lambda P, L_=L_, pl=pl, cnt=cnt: wrap180(polar_deg(P[L_].center) - polar_deg(P[pl].center) - 360.0 / cnt), C["azimuth"])
        u = lambda P, L_=L_: unit(P[L_].center - P[SUP].center)
        X_.type_is(Cn, "cone", C["types"])
        for fld, val in (("base_radius", 3.5), ("top_radius", 0.8), ("height", 12.0)):
            X_.eq_rel(f"cone{gi} {fld}", "dimension", (Cn,), lambda P, Cn=Cn, fld=fld: getattr(need(P[Cn], "cone"), fld), val, C["cone"])
        X_.zero_deg(f"cone{gi} radially inward", "orientation", (SUP, L_, Cn), lambda P, Cn=Cn, u=u: ang(axis(P[Cn]), -u(P)), C["cone"])
        X_.zero_len(f"cone{gi} starts at lens inward pole", "mate", (SUP, L_, Cn),
                    lambda P, Cn=Cn, L_=L_, u=u: np.linalg.norm(ends(P[Cn])[0] - (P[L_].center - P[L_].radius * u(P))), C["cone"])
        X_.type_is(G_, "cylinder", C["types"])
        X_.eq_rel(f"guide{gi} radius", "dimension", (G_,), lambda P, G_=G_: radius(need(P[G_], "cylinder")), 0.8, C["guide"])
        X_.eq_rel(f"guide{gi} length", "dimension", (G_,), lambda P, G_=G_: P[G_].height, 10.0, C["guide"])
        X_.zero_deg(f"guide{gi} coaxial", "orientation", (Cn, G_), lambda P, Cn=Cn, G_=G_: ang(axis(P[G_]), axis(P[Cn]), False), C["guide"])
        X_.zero_len(f"guide{gi} continues from cone tip", "mate", (Cn, G_),
                    lambda P, Cn=Cn, G_=G_: min(np.linalg.norm(p - ends(P[Cn])[1]) for p in ends(P[G_])) +
                    max(0.0, np.linalg.norm(P[G_].center) - np.linalg.norm(ends(P[Cn])[1])) * 0, C["guide"])


# ---------------------------------------------------------------------------
def diatom(case, ref, X_: Ctx):
    S = case["scale"]; ND, TV, BV, TR, BRp, MA, TB, BB = range(8)
    C = {k: f"diatom:{k}" for k in ("types", "valves", "gap", "nodule", "raphe", "mantle", "bands", "costae", "stations", "pores")}
    for v in (TV, BV):
        X_.type_is(v, "box", C["types"])
        for a, val in enumerate((80.0, 30.0, 3.0)):
            X_.eq_rel(f"valve{v} size{a}", "dimension", (v,), lambda P, v=v, a=a: need(P[v], "box").size[a], val, C["valves"])
        X_.zero_len(f"valve{v} centered in XY", "anchor", (v,), lambda P, v=v: math.hypot(*P[v].center[:2]), C["valves"])
    X_.eq_len("valve facing gap 8", "mate", (TV, BV), lambda P: lo(P[TV], 2) - hi(P[BV], 2), 8.0, C["gap"])
    X_.zero_len("origin midway between valves", "anchor", (TV, BV), lambda P: (lo(P[TV], 2) + hi(P[BV], 2)) / 2, C["gap"])
    X_.type_is(ND, "sphere", C["types"])
    X_.zero_len("nodule at origin", "anchor", (ND,), lambda P: np.linalg.norm(P[ND].center), C["nodule"])
    X_.eq_rel("nodule radius", "dimension", (ND,), lambda P: radius(need(P[ND], "sphere")), 2.5, C["nodule"])
    for r, v in ((TR, TV), (BRp, BV)):
        X_.type_is(r, "beam", C["types"])
        X_.eq_rel(f"raphe{r} section", "dimension", (r,), lambda P, r=r: need(P[r], "beam").width, 1.5, C["raphe"])
        X_.zero_deg(f"raphe{r} longitudinal", "orientation", (r,), lambda P, r=r: ang(P[r].direction, X, False), C["raphe"])
        X_.zero_len(f"raphe{r} in valve midplane, on axis", "mate", (v, r), lambda P, r=r, v=v: abs(P[r].center[2] - P[v].center[2]) + abs(P[r].center[1] - P[v].center[1]), C["raphe"])
        X_.zero_len(f"raphe{r} ends 2 short of edges", "mate", (v, r),
                    lambda P, r=r, v=v: max(abs(min(p[0] for p in ends(P[r])) - (lo(P[v], 0) + 2)), abs(max(p[0] for p in ends(P[r])) - (hi(P[v], 0) - 2))), C["raphe"])
    X_.type_is(MA, "pipe", C["types"])
    X_.zero_len("mantle spans valve gap", "mate", (TV, BV, MA), lambda P: abs(lo(P[MA], 2) - hi(P[BV], 2)) + abs(hi(P[MA], 2) - lo(P[TV], 2)), C["mantle"])
    X_.zero_deg("mantle axis Z", "orientation", (MA,), lambda P: ang(axis(P[MA]), Z, False), C["mantle"])
    X_.zero_len("mantle centered", "anchor", (MA,), lambda P: math.hypot(*P[MA].center[:2]), C["mantle"])
    X_.zero_len("mantle outer = half valve width", "mate", (TV, MA), lambda P: need(P[MA], "pipe").outer_radius - P[TV].size[1] / 2, C["mantle"])
    X_.eq_len("mantle wall 2", "dimension", (MA,), lambda P: P[MA].outer_radius - P[MA].inner_radius, 2.0, C["mantle"])
    for b, sgn in ((TB, 1), (BB, -1)):
        X_.type_is(b, "torus", C["types"])
        X_.eq_rel(f"band{b} tube", "dimension", (b,), lambda P, b=b: need(P[b], "torus").tube_radius, 1.2, C["bands"])
        X_.zero_len(f"band{b} on mantle end rim", "mate", (MA, b),
                    lambda P, b=b, sgn=sgn: np.linalg.norm(P[b].center - (top_center(P[MA]) if sgn > 0 else bottom_center(P[MA]))) + abs(P[b].ring_radius - P[MA].outer_radius),
                    C["bands"])
        X_.zero_deg(f"band{b} axis", "orientation", (MA, b), lambda P, b=b: ang(axis(P[b]), axis(P[MA]), False), C["bands"])
    def rid(base, v, side, k, xs):
        return base + v * 4 * S + side * 2 * S + 2 * k + xs
    for v, (valve, raphe) in enumerate(((TV, TR), (BV, BRp))):
        for side, sy in enumerate((1, -1)):
            for k in range(S):
                for xs, sx in enumerate((1, -1)):
                    c = rid(8, v, side, k, xs)
                    p = rid(8 + 8 * S, v, side, k, xs)
                    X_.type_is(c, "cylinder", C["types"])
                    X_.eq_rel(f"costa{c} radius", "dimension", (c,), lambda P, c=c: radius(need(P[c], "cylinder")), 1.0, C["costae"])
                    X_.zero_deg(f"costa{c} transverse", "orientation", (c,), lambda P, c=c: ang(axis(P[c]), Y, False), C["costae"])
                    X_.zero_len(f"costa{c} in valve midplane", "mate", (valve, c), lambda P, c=c, valve=valve: P[c].center[2] - P[valve].center[2], C["costae"])
                    X_.zero_len(f"costa{c} from raphe to 2 short of edge", "mate", (valve, raphe, c),
                                lambda P, c=c, valve=valve, raphe=raphe, sy=sy: max(
                                    abs(min(abs(p[1] - P[raphe].center[1]) for p in ends(P[c]))),
                                    abs(max(abs(p[1] - P[raphe].center[1]) for p in ends(P[c])) - (P[valve].size[1] / 2 - 2)))
                                + (0.0 if sy * (P[c].center[1] - P[raphe].center[1]) > 0 else float("inf")), C["costae"])
                    X_.zero_len(f"costa{c} at station {k + 1}", "pattern", (raphe, c),
                                lambda P, c=c, raphe=raphe, k=k, sx=sx: P[c].center[0] - sx * (k + 1) / S * max(abs(p[0]) for p in ends(P[raphe])), C["stations"])
                    X_.type_is(p, "sphere", C["types"])
                    X_.eq_rel(f"pore{p} radius", "dimension", (p,), lambda P, p=p: radius(need(P[p], "sphere")), 0.8, C["pores"])
                    X_.zero_len(f"pore{p} at interval midpoint on costa centerline", "pattern", (raphe, c, p),
                                lambda P, p=p, c=c, raphe=raphe, k=k, sx=sx: np.linalg.norm(
                                    P[p].center - V(sx * (k + 0.5) / S * max(abs(q[0]) for q in ends(P[raphe])), P[c].center[1], P[c].center[2])), C["pores"])


# ---------------------------------------------------------------------------
def hex_cells(R: int, d: float = 10.0):
    """Cell planar centers in prompt order: center, then rings starting on the 240-degree ray, CCW."""
    cells = [V(0, 0, 0)]
    for r in range(1, R + 1):
        p = r * d * V(math.cos(math.radians(240)), math.sin(math.radians(240)), 0)
        for side in range(6):
            step = d * V(math.cos(math.radians(60 * side)), math.sin(math.radians(60 * side)), 0)
            for _ in range(r):
                cells.append(p.copy()); p = p + step
    return cells


def honeycomb(case, ref, X_: Ctx):
    R = case["scale"]; cells = hex_cells(R); nc = len(cells)
    BASE, FRAME = 0, 1
    pipes = list(range(2, 2 + nc)); caps = [p + nc for p in pipes]; cones = [p + 2 * nc for p in pipes]
    links_pairs = sorted((i, j) for i in range(nc) for j in range(i + 1, nc) if abs(np.linalg.norm(cells[i] - cells[j]) - 10) < 1e-6)
    links = list(range(2 + 3 * nc, 2 + 3 * nc + len(links_pairs)))
    ribs = list(range(links[-1] + 1, links[-1] + 7))
    C = {k: f"honeycomb:{k}" for k in ("types", "base", "cells", "lattice", "plane", "tilt", "caps", "cones", "links", "frame", "ribs")}
    X_.type_is(BASE, "cylinder", C["types"])
    X_.eq_rel("base thickness", "dimension", (BASE,), lambda P: need(P[BASE], "cylinder").height, 1.5, C["base"])
    X_.zero_deg("base horizontal", "orientation", (BASE,), lambda P: ang(axis(P[BASE]), Z, False), C["base"])
    X_.zero_len("origin at base top-face center", "anchor", (BASE,), lambda P: np.linalg.norm(top_center(P[BASE])), C["base"])
    maxr = max(np.linalg.norm(c) for c in cells)
    X_.zero_len("base radius = largest cell radius + cell diameter", "mate", tuple([BASE] + pipes),
                lambda P: radius(P[BASE]) - (max(math.hypot(*P[p].center[:2]) for p in pipes) + 2 * P[pipes[0]].outer_radius), C["base"])
    cz = lambda P: P[pipes[0]].center[2]
    for i, p in enumerate(pipes):
        X_.type_is(p, "pipe", C["types"])
        for fld, val in (("inner_radius", 4.2), ("outer_radius", 5.0), ("height", 12.0)):
            X_.eq_rel(f"cell{i} {fld}", "dimension", (p,), lambda P, p=p, fld=fld: getattr(need(P[p], "pipe"), fld), val, C["cells"])
        X_.zero_len(f"cell{i} lattice site", "pattern", (p,), lambda P, p=p, i=i: math.hypot(*(P[p].center - cells[i])[:2]), C["lattice"])
        if i == 0:
            X_.zero_deg("center cell vertical", "orientation", (p,), lambda P, p=p: ang(axis(P[p]), Z, False), C["tilt"])
            X_.zero_len("center cell stands on base plate", "mate", (BASE, p), lambda P, p=p: bottom_center(P[p])[2] - top_center(P[BASE])[2], C["plane"])
        else:
            X_.zero_len(f"cell{i} centroid in central mid-plane", "pattern", (pipes[0], p), lambda P, p=p: P[p].center[2] - cz(P), C["plane"])
            def tilt(P, p=p):
                a = axis(P[p]); a = a if a[2] > 0 else -a
                inward = unit(-P[p].center * V(1, 1, 0))
                want = math.cos(math.radians(13)) * Z + math.sin(math.radians(13)) * inward
                return ang(a, want)
            X_.zero_deg(f"cell{i} tilts 13 toward center", "orientation", (p,), tilt, C["tilt"])
    for i, (p, c, k) in enumerate(zip(pipes, caps, cones)):
        X_.type_is(c, "cylinder", C["types"])
        X_.zero_len(f"cap{i} radius = bore", "mate", (p, c), lambda P, p=p, c=c: radius(need(P[c], "cylinder")) - P[p].inner_radius, C["caps"])
        X_.eq_rel(f"cap{i} thickness", "dimension", (c,), lambda P, c=c: P[c].height, 0.8, C["caps"])
        X_.zero_deg(f"cap{i} axis || pipe", "orientation", (p, c), lambda P, p=p, c=c: ang(axis(P[c]), axis(P[p]), False), C["caps"])
        X_.zero_len(f"cap{i} projects onto pipe centroid", "mate", (p, c), lambda P, p=p, c=c: math.hypot(*(P[c].center - P[p].center)[:2]), C["caps"])
        if i == 0:
            X_.zero_len("center cap abuts center pipe top", "mate", (p, c), lambda P, p=p, c=c: bottom_center(P[c])[2] - top_center(P[p])[2], C["caps"])
        else:
            X_.zero_len(f"cap{i} in central cap plane", "pattern", (caps[0], c), lambda P, c=c: P[c].center[2] - P[caps[0]].center[2], C["caps"])
        X_.type_is(k, "cone", C["types"])
        X_.zero_len(f"cone{i} base = bore", "mate", (p, k), lambda P, p=p, k=k: need(P[k], "cone").base_radius - P[p].inner_radius, C["cones"])
        X_.eq_rel(f"cone{i} length", "dimension", (k,), lambda P, k=k: P[k].height, 3.0, C["cones"])
        X_.zero_len(f"cone{i} pointed", "dimension", (k,), lambda P, k=k: need(P[k], "cone").top_radius, C["cones"])
        X_.zero_deg(f"cone{i} axis || pipe, pointing up", "orientation", (p, k),
                    lambda P, p=p, k=k: ang(axis(P[k]), axis(P[p]) if axis(P[p])[2] > 0 else -axis(P[p])), C["cones"])
        X_.zero_len(f"cone{i} projects onto pipe centroid", "mate", (p, k), lambda P, p=p, k=k: math.hypot(*(P[k].center - P[p].center)[:2]), C["cones"])
        if i == 0:
            X_.zero_len("center cone tip abuts base underside", "mate", (BASE, k), lambda P, k=k: ends(P[k])[1][2] - bottom_center(P[BASE])[2], C["cones"])
        else:
            X_.zero_len(f"cone{i} in central cone plane", "pattern", (cones[0], k), lambda P, k=k: P[k].center[2] - P[cones[0]].center[2], C["cones"])
    for li, (a, b) in zip(links, links_pairs):
        pa, pb = pipes[a], pipes[b]
        X_.type_is(li, "beam", C["types"])
        X_.eq_rel(f"link{li} section", "dimension", (li,), lambda P, li=li: need(P[li], "beam").width, 1.0, C["links"])
        X_.zero_len(f"link{li} joins centroids", "mate", (pa, pb, li), lambda P, li=li, pa=pa, pb=pb: endpoints_match(P[li], P[pa].center, P[pb].center), C["links"])
    X_.type_is(FRAME, "torus", C["types"])
    X_.zero_len("frame at pipe-centroid plane, centered", "mate", (pipes[0], FRAME),
                lambda P: math.hypot(*P[FRAME].center[:2]) + abs(P[FRAME].center[2] - cz(P)), C["frame"])
    X_.zero_len("frame ring = base radius", "mate", (BASE, FRAME), lambda P: need(P[FRAME], "torus").ring_radius - radius(P[BASE]), C["frame"])
    X_.eq_rel("frame tube", "dimension", (FRAME,), lambda P: P[FRAME].tube_radius, 2.0, C["frame"])
    X_.zero_deg("frame axis Z", "orientation", (FRAME,), lambda P: ang(axis(P[FRAME]), Z, False), C["frame"])
    for j, rb in enumerate(ribs):
        X_.type_is(rb, "beam", C["types"])
        X_.eq_rel(f"rib{j} section", "dimension", (rb,), lambda P, rb=rb: need(P[rb], "beam").width, 1.5, C["ribs"])
        inn = lambda P, rb=rb: min(ends(need(P[rb], "beam")), key=lambda q: math.hypot(*q[:2]))
        out = lambda P, rb=rb: max(ends(need(P[rb], "beam")), key=lambda q: math.hypot(*q[:2]))
        X_.zero_len(f"rib{j} from center axis at 3/4 height", "mate", (pipes[0], rb),
                    lambda P, inn=inn: math.hypot(*(inn(P) - P[pipes[0]].center)[:2]) + abs(inn(P)[2] - (bottom_center(P[pipes[0]])[2] + 0.75 * P[pipes[0]].height)), C["ribs"])
        X_.zero_len(f"rib{j} to frame circle", "mate", (FRAME, rb),
                    lambda P, out=out: abs(math.hypot(*(out(P) - P[FRAME].center)[:2]) - P[FRAME].ring_radius) + abs(out(P)[2] - P[FRAME].center[2]) * 0
                    + abs(out(P)[2] - inn(P)[2]) * 0, C["ribs"])
        X_.zero_deg(f"rib{j} direction", "pattern", (rb,), lambda P, out=out, inn=inn, j=j: wrap180(polar_deg(out(P) - inn(P)) - 60.0 * j), C["ribs"])


# ---------------------------------------------------------------------------
def vertebral(case, ref, X_: Ctx):
    n = case["scale"]
    incs = {7: [40 / 7] * 7, 12: [-40 / 12] * 12, 19: [40 / 7] * 7 + [-40 / 12] * 12}[n]
    th = [0.0]
    for k in range(1, n):
        th.append(th[-1] + incs[k - 1])
    PITCH = 26.0
    C = {k: f"vertebral:{k}" for k in ("types", "body", "first", "curve", "cone", "process", "disc", "canal")}
    sup = lambda t: V(0, math.sin(math.radians(t)), math.cos(math.radians(t)))
    post = lambda t: V(0, -math.cos(math.radians(t)), math.sin(math.radians(t)))
    body = lambda k: 4 * k
    for k in range(n):
        b, cn, px, nx = 4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3
        X_.type_is(b, "box", C["types"])
        for a, v in enumerate((25.0, 18.0, 20.0)):
            X_.eq_rel(f"body{k} size{a}", "dimension", (b,), lambda P, b=b, a=a: need(P[b], "box").size[a], v, C["body"])
        if k == 0:
            X_.zero_len("first body at origin", "anchor", (b,), lambda P: np.linalg.norm(P[0].center), C["first"])
        else:
            pb = body(k - 1)
            X_.zero_len(f"body{k} one pitch along tilted superior", "pattern", (pb, b),
                        lambda P, b=b, pb=pb, t=th[k]: np.linalg.norm(P[b].center - (P[pb].center + PITCH * sup(t))), C["curve"])
        X_.type_is(cn, "cone", C["types"])
        X_.eq_rel(f"cone{k} base", "dimension", (cn,), lambda P, cn=cn: need(P[cn], "cone").base_radius, 4.0, C["cone"])
        X_.eq_rel(f"cone{k} length", "dimension", (cn,), lambda P, cn=cn: P[cn].height, 22.0, C["cone"])
        X_.zero_deg(f"cone{k} posterior in local frame", "orientation", (cn,), lambda P, cn=cn, t=th[k]: ang(axis(P[cn]), post(t)), C["cone"])
        X_.zero_len(f"cone{k} base on posterior face", "mate", (b, cn),
                    lambda P, b=b, cn=cn, t=th[k]: np.linalg.norm(ends(P[cn])[0] - (P[b].center + 9.0 * post(t))), C["cone"])
        for pr, sx in ((px, 1), (nx, -1)):
            X_.type_is(pr, "beam", C["types"])
            X_.eq_rel(f"process{pr} length", "dimension", (pr,), lambda P, pr=pr: P[pr].length, 20.0, C["process"])
            X_.eq_rel(f"process{pr} section", "dimension", (pr,), lambda P, pr=pr: need(P[pr], "beam").width, 3.0, C["process"])
            X_.zero_len(f"process{pr} from body center along {'+' if sx > 0 else '-'}X", "mate", (b, pr),
                        lambda P, b=b, pr=pr, sx=sx: endpoints_match(P[pr], P[b].center, P[b].center + V(sx * P[pr].length, 0, 0)), C["process"])
    for k in range(n - 1):
        d, cnl = 4 * n + k, 4 * n + (n - 1) + k
        a, b = body(k), body(k + 1)
        X_.type_is(d, "cylinder", C["types"])
        X_.eq_rel(f"disc{k} radius", "dimension", (d,), lambda P, d=d: radius(need(P[d], "cylinder")), 12.0, C["disc"])
        X_.eq_rel(f"disc{k} thickness", "dimension", (d,), lambda P, d=d: P[d].height, 6.0, C["disc"])
        X_.zero_len(f"disc{k} midway", "mate", (a, b, d), lambda P, a=a, b=b, d=d: np.linalg.norm(P[d].center - (P[a].center + P[b].center) / 2), C["disc"])
        X_.zero_deg(f"disc{k} axis along joining line", "orientation", (a, b, d), lambda P, a=a, b=b, d=d: ang(axis(P[d]), P[b].center - P[a].center, False), C["disc"])
        X_.type_is(cnl, "pipe", C["types"])
        X_.eq_rel(f"canal{k} bore", "dimension", (cnl,), lambda P, cnl=cnl: need(P[cnl], "pipe").inner_radius, 6.0, C["canal"])
        X_.eq_rel(f"canal{k} outer", "dimension", (cnl,), lambda P, cnl=cnl: P[cnl].outer_radius, 7.5, C["canal"])
        X_.zero_len(f"canal{k} connects body centers", "mate", (a, b, cnl),
                    lambda P, a=a, b=b, cnl=cnl: max(min(np.linalg.norm(e - P[q].center) for e in ends(P[cnl])) for q in (a, b)), C["canal"])


# ---------------------------------------------------------------------------
def cochlea(case, ref, X_: Ctx):
    T = case["scale"] // 12; nseg = 12 * T; R0 = 30.0; DR = R0 * 0.55 / nseg
    MOD, RING, ENTRY = 0, 1, 2
    beams = [3 + 2 * i for i in range(nseg)]; mems = [4 + 2 * i for i in range(nseg)]; APEX = 3 + 2 * nseg
    C = {k: f"cochlea:{k}" for k in ("types", "segment", "start", "continuity", "turn", "rise", "taper", "membrane",
                                     "modiolus", "ring", "entry", "apex")}
    def se(P, bm):
        a, b = ends(need(P[bm], "beam"))
        # helix order: start = the endpoint with smaller elevation (rise is positive)
        return (a, b) if a[2] <= b[2] else (b, a)
    for i, (bm, mb) in enumerate(zip(beams, mems)):
        X_.type_is(bm, "beam", C["types"])
        X_.eq_rel(f"seg{i} section", "dimension", (bm,), lambda P, bm=bm: need(P[bm], "beam").width, 4.0, C["segment"])
        X_.zero_deg(f"seg{i} +30 ccw", "pattern", (bm,), lambda P, bm=bm: wrap180(polar_deg(se(P, bm)[1]) - polar_deg(se(P, bm)[0]) - 30.0), C["turn"])
        X_.eq_len(f"seg{i} rise 0.5", "pattern", (bm,), lambda P, bm=bm: se(P, bm)[1][2] - se(P, bm)[0][2], 0.5, C["rise"])
        X_.eq_len(f"seg{i} linear taper", "pattern", (bm,), lambda P, bm=bm: math.hypot(*se(P, bm)[0][:2]) - math.hypot(*se(P, bm)[1][:2]), DR, C["taper"])
        if i == 0:
            X_.zero_len("helix starts on +X at r=30, z=0", "anchor", (bm,), lambda P, bm=bm: np.linalg.norm(se(P, bm)[0] - V(R0, 0, 0)), C["start"])
        else:
            pb = beams[i - 1]
            X_.zero_len(f"seg{i} continues seg{i - 1}", "mate", (pb, bm), lambda P, bm=bm, pb=pb: np.linalg.norm(se(P, bm)[0] - se(P, pb)[1]), C["continuity"])
        X_.type_is(mb, "box", C["types"])
        for a, v in enumerate((3.0, 5.0, 0.8)):
            X_.eq_rel(f"mem{i} size{a}", "dimension", (mb,), lambda P, mb=mb, a=a: need(P[mb], "box").size[a], v, C["membrane"])
        X_.zero_len(f"mem{i} on beam midpoint", "mate", (bm, mb), lambda P, bm=bm, mb=mb: np.linalg.norm(P[mb].center - P[bm].center), C["membrane"])
    X_.type_is(MOD, "cylinder", C["types"])
    X_.eq_rel("modiolus radius", "dimension", (MOD,), lambda P: radius(need(P[MOD], "cylinder")), 3.5, C["modiolus"])
    X_.zero_deg("modiolus axis Z", "orientation", (MOD,), lambda P: ang(axis(P[MOD]), Z, False), C["modiolus"])
    X_.zero_len("modiolus from basal plane on Z axis", "anchor", (MOD,), lambda P: abs(zmin(P[MOD])) + math.hypot(*P[MOD].center[:2]), C["modiolus"])
    X_.zero_len("modiolus to terminal elevation", "mate", (beams[-1], MOD), lambda P: zmax(P[MOD]) - se(P, beams[-1])[1][2], C["modiolus"])
    X_.type_is(RING, "torus", C["types"])
    X_.zero_len("basal ring centered, through helix start", "mate", (beams[0], RING),
                lambda P: np.linalg.norm(P[RING].center) + abs(P[RING].ring_radius - math.hypot(*se(P, beams[0])[0][:2])), C["ring"])
    X_.zero_deg("basal ring axis Z", "orientation", (RING,), lambda P: ang(axis(P[RING]), Z, False), C["ring"])
    X_.eq_rel("basal ring tube", "dimension", (RING,), lambda P: need(P[RING], "torus").tube_radius, 2.0, C["ring"])
    X_.type_is(ENTRY, "pipe", C["types"])
    for fld, val in (("inner_radius", 2.0), ("outer_radius", 5.0), ("height", 6.0)):
        X_.eq_rel(f"entry {fld}", "dimension", (ENTRY,), lambda P, fld=fld: getattr(need(P[ENTRY], "pipe"), fld), val, C["entry"])
    X_.zero_deg("entry vertical", "orientation", (ENTRY,), lambda P: ang(axis(P[ENTRY]), Z, False), C["entry"])
    X_.zero_len("entry upper face at helix start", "mate", (beams[0], ENTRY), lambda P: np.linalg.norm(top_center(P[ENTRY]) - se(P, beams[0])[0]), C["entry"])
    X_.type_is(APEX, "cone", C["types"])
    for fld, val in (("base_radius", 4.0), ("top_radius", 0.5), ("height", 8.0)):
        X_.eq_rel(f"apex {fld}", "dimension", (APEX,), lambda P, fld=fld: getattr(need(P[APEX], "cone"), fld), val, C["apex"])
    X_.zero_deg("apex points up", "orientation", (APEX,), lambda P: ang(axis(P[APEX]), Z), C["apex"])
    X_.zero_len("apex base on modiolus top", "mate", (MOD, APEX), lambda P: np.linalg.norm(ends(P[APEX])[0] - top_center(P[MOD])), C["apex"])


# ---------------------------------------------------------------------------
def radiolarian(case, ref, X_: Ctx):
    lv = case["scale"] - 1; RC = 50.0
    verts, edges = geodesic(lv, RC); nv, ne = len(verts), len(edges)
    RING = 0; nodes = list(range(1, 1 + nv)); struts = list(range(1 + nv, 1 + nv + ne)); spines = list(range(1 + nv + ne, 1 + 2 * nv + ne))
    C = {k: f"radiolarian:{k}" for k in ("types", "ring", "nodes", "vertex", "unique", "struts", "edge", "spines")}
    X_.type_is(RING, "torus", C["types"])
    X_.zero_len("ring on equator", "anchor", (RING,), lambda P: np.linalg.norm(P[RING].center) + abs(need(P[RING], "torus").ring_radius - RC), C["ring"])
    X_.zero_deg("ring axis Z", "orientation", (RING,), lambda P: ang(axis(P[RING]), Z, False), C["ring"])
    X_.eq_rel("ring tube", "dimension", (RING,), lambda P: P[RING].tube_radius, 1.5, C["ring"])
    near = lambda p: int(np.argmin(np.linalg.norm(verts - p, axis=1)))
    for nd in nodes:
        X_.type_is(nd, "sphere", C["types"])
        X_.eq_rel(f"node{nd} radius", "dimension", (nd,), lambda P, nd=nd: radius(need(P[nd], "sphere")), 2.0, C["nodes"])
        X_.zero_len(f"node{nd} at a geodesic vertex", "pattern", (nd,), lambda P, nd=nd: float(np.min(np.linalg.norm(verts - P[nd].center, axis=1))), C["vertex"])
    X_.add("every vertex has exactly one node", "topology", tuple(nodes),
           lambda P: float(nv - len({near(P[nd].center) for nd in nodes})), C["unique"], tol=0.0, unit="count")
    for s in struts:
        X_.type_is(s, "beam", C["types"])
        X_.eq_rel(f"strut{s} section", "dimension", (s,), lambda P, s=s: need(P[s], "beam").width, 1.5, C["struts"])
        X_.zero_len(f"strut{s} ends on node centers", "mate", tuple([s] + nodes),
                    lambda P, s=s: max(min(np.linalg.norm(p - P[nd].center) for nd in nodes) for p in ends(need(P[s], "beam"))), C["struts"])
        X_.add(f"strut{s} is a mesh edge", "topology", (s,),
               lambda P, s=s: 0.0 if tuple(sorted(near(p) for p in ends(need(P[s], "beam")))) in edges else 1.0, C["edge"], tol=0.0, unit="bool")
    for sp in spines:
        X_.type_is(sp, "cone", C["types"])
        for fld, val in (("base_radius", 2.5), ("top_radius", 0.3), ("height", 12.0)):
            X_.eq_rel(f"spine{sp} {fld}", "dimension", (sp,), lambda P, sp=sp, fld=fld: getattr(need(P[sp], "cone"), fld), val, C["spines"])
        X_.zero_len(f"spine{sp} base on a vertex", "mate", (sp,), lambda P, sp=sp: float(np.min(np.linalg.norm(verts - ends(P[sp])[0], axis=1))), C["spines"])
        X_.zero_deg(f"spine{sp} outward radial", "orientation", (sp,), lambda P, sp=sp: ang(axis(P[sp]), ends(P[sp])[0]), C["spines"])


BUILDERS = {
    "Furniture Assembly": furniture,
    "Pipe Manifold": pipe_manifold,
    "Axle Bearing": axle_bearing,
    "Armillary Sphere": armillary,
    "Clock Tower Mechanism": clock,
    "Gantry Crane Assembly": gantry,
    "Phyllotaxis Disc": phyllotaxis,
    "Compound Eye": compound_eye,
    "Diatom Frustule": diatom,
    "Honeycomb Lattice": honeycomb,
    "Vertebral Column": vertebral,
    "Cochlear Spiral": cochlea,
    "Radiolarian Skeleton": radiolarian,
}
