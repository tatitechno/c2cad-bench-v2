"""Semantic Scoring Module — Physics + Per-Family Structural Evaluators

This module contains the semantic scoring system for 3D assembly evaluation.
It includes:
  - Physics validators: gravity support, connectivity, interference detection
  - Geometry helpers: spacing, regularity, alignment, planarity checks
  - Per-family evaluators: specific structural rules for each assembly family
  - Scoring dispatch and baseline normalization
"""

import math
from scoring_utils import _shape_center, _dist3, _normalize_shapes
from probe.validators import validate_gravity, validate_connectivity, validate_interference
from probe.validators import validate_wall_thickness, validate_clearance_fit, validate_mates


def _apply_physics_gate(raw, shapes, tol_mm=3.0):
    """Universal physics gate — crush semantic when basic physics are violated.

    Computes avg(gravity, connectivity). If < 0.4, multiplies raw score by
    (physics_avg / 0.4), smoothly driving it toward 0 for physically invalid
    outputs. Evaluators can opt in by calling this on their final raw score.
    """
    if not shapes or len(shapes) < 2:
        return 0.0
    g = _phys_gravity(shapes, tol_mm=tol_mm)
    c = _phys_connectivity(shapes, tol_mm=tol_mm)
    physics_avg = (g + c) / 2.0
    if physics_avg < 0.4:
        return raw * (physics_avg / 0.4)
    return raw


def _phys_gravity(shapes, tol_mm=2.0):
    """Proportion of shapes that are gravity-supported. 0.0–1.0."""
    if not shapes:
        return 0.0
    res = validate_gravity(shapes, tol_mm=tol_mm)
    n = max(1, len(shapes))
    unsup = len(res.get("floating_ids", []))
    return max(0.0, 1.0 - unsup / n)

def _phys_connectivity(shapes, tol_mm=2.0):
    """1/num_components — fully connected=1.0. 0.0–1.0."""
    if not shapes:
        return 0.0
    res = validate_connectivity(shapes, tol_mm=tol_mm)
    n_comp = res.get("islands", 1)
    return 1.0 / max(1, n_comp)

def _phys_interference(shapes, tol_mm=1.0):
    """Proportion of non-interfering pairs. 0.0–1.0."""
    if not shapes or len(shapes) < 2:
        return 1.0
    res = validate_interference(shapes, tol_mm=tol_mm)
    n_interf = len(res.get("overlapping_pairs", []))
    n_pairs = max(1, len(shapes) * (len(shapes) - 1) // 2)
    return max(0.0, 1.0 - n_interf / n_pairs)

def _check_uniform_z_spacing(shapes_subset):
    """How uniformly spaced are shapes in Z? 0.0–1.0."""
    centers = [_shape_center(s) for s in shapes_subset]
    z_vals = sorted([c[2] for c in centers if c])
    if len(z_vals) < 3:
        return 1.0
    gaps = [z_vals[i+1] - z_vals[i] for i in range(len(z_vals)-1)]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap < 0.1:
        return 0.0
    dev = sum(abs(g - mean_gap) / mean_gap for g in gaps) / len(gaps)
    return max(0.0, 1.0 - dev)

def _check_angular_regularity(shapes_subset, cx=0.0, cy=0.0):
    """How evenly are shapes distributed angularly around (cx,cy)? 0.0–1.0."""
    angles = []
    for s in shapes_subset:
        c = _shape_center(s)
        if c:
            dx, dy = c[0] - cx, c[1] - cy
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                angles.append(math.atan2(dy, dx))
    if len(angles) < 2:
        return 1.0
    angles.sort()
    n = len(angles)
    gaps = []
    for i in range(n):
        g = angles[(i+1) % n] - angles[i]
        if g <= 0:
            g += 2 * math.pi
        gaps.append(g)
    mean_gap = 2 * math.pi / n
    dev = sum(abs(g - mean_gap) / mean_gap for g in gaps) / n
    return max(0.0, 1.0 - dev)

def _check_coplanar_z(shapes_subset):
    """Are all shapes at the same Z? 0.0–1.0."""
    z_vals = []
    for s in shapes_subset:
        c = _shape_center(s)
        if c:
            z_vals.append(c[2])
    if len(z_vals) < 2:
        return 1.0
    spread = max(z_vals) - min(z_vals)
    return max(0.0, 1.0 - spread / max(1.0, abs(sum(z_vals)/len(z_vals))))

def _check_constant_radius(shapes_subset, cx=0.0, cy=0.0):
    """Are all shapes at the same XY distance from (cx,cy)? 0.0–1.0."""
    radii = []
    for s in shapes_subset:
        c = _shape_center(s)
        if c:
            r = math.sqrt((c[0]-cx)**2 + (c[1]-cy)**2)
            radii.append(r)
    if len(radii) < 2:
        return 1.0
    mean_r = sum(radii) / len(radii)
    if mean_r < 0.1:
        return 1.0
    dev = sum(abs(r - mean_r) / mean_r for r in radii) / len(radii)
    return max(0.0, 1.0 - dev)

def _check_uniform_sizes(shapes_subset, field="radius"):
    """Are all shapes the same size for a given field? 0.0–1.0."""
    vals = [s.get(field, 0) for s in shapes_subset if s.get(field)]
    if len(vals) < 2:
        return 1.0
    mean_v = sum(vals) / len(vals)
    if mean_v < 1e-9:
        return 1.0
    dev = sum(abs(v - mean_v) / mean_v for v in vals) / len(vals)
    return max(0.0, 1.0 - dev)

def _check_beam_connects_pair(beam, shape_a, shape_b, tol_mm=3.0):
    """Does this beam connect shape_a center to shape_b center? 0.0 or 1.0."""
    bs = beam.get("start", [0,0,0])
    be = beam.get("end", [0,0,0])
    ca = _shape_center(shape_a)
    cb = _shape_center(shape_b)
    if not ca or not cb:
        return 0.0
    # Check both orientations
    d1 = _dist3(bs, ca) + _dist3(be, cb)
    d2 = _dist3(bs, cb) + _dist3(be, ca)
    return 1.0 if min(d1, d2) <= tol_mm * 2 else 0.0

def _check_planarity_y0(shapes_subset):
    """Are all shapes in the Y=0 plane? 0.0–1.0."""
    y_devs = []
    for s in shapes_subset:
        c = _shape_center(s)
        if c:
            y_devs.append(abs(c[1]))
        if s.get("type") == "beam":
            for pt in [s.get("start"), s.get("end")]:
                if pt:
                    y_devs.append(abs(pt[1]))
    if not y_devs:
        return 1.0
    max_dev = max(y_devs)
    return max(0.0, 1.0 - max_dev / 5.0)  # penalize if Y > 5mm

def _check_tangent_spheres(shapes_subset, expected_distance=None):
    """Are adjacent spheres tangent? Average tangency quality. 0.0–1.0."""
    spheres = [s for s in shapes_subset if s.get("type") == "sphere"]
    if len(spheres) < 2:
        return 1.0
    R = spheres[0].get("radius", 1.0)
    exp_d = expected_distance or (2.0 * R)
    # Find nearest-neighbor distances
    scores = []
    for i, s1 in enumerate(spheres):
        c1 = _shape_center(s1)
        if not c1:
            continue
        min_d = float('inf')
        for j, s2 in enumerate(spheres):
            if i == j:
                continue
            c2 = _shape_center(s2)
            if c2:
                d = _dist3(c1, c2)
                min_d = min(min_d, d)
        if min_d < float('inf'):
            err = abs(min_d - exp_d) / exp_d
            scores.append(max(0.0, 1.0 - err))
    return sum(scores) / len(scores) if scores else 1.0


# ── Per-family semantic evaluators ────────────────────────────

def _sem_staircase(shapes, golden, scale):
    """Spiral Staircase: pillar+steps spiral with uniform rise and rotation.

    Physics-first scoring (v2): gravity, connectivity, and radial attachment
    are prerequisites — not optional bonuses. A staircase with floating,
    disconnected beams that happen to be evenly spaced is NOT a valid staircase.

    Physics gate: if the average of gravity + connectivity + radial < 0.4,
    the entire semantic score is multiplied by (physics_avg / 0.4), crushing
    outputs that fail basic structural requirements regardless of pattern quality.
    """
    beams = [s for s in shapes if s.get("type") == "beam"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    tol = max(3.0, scale * 0.3)

    # ── Physics checks (dominant: 60% total weight) ──────────
    gravity_s      = _phys_gravity(shapes)
    connectivity_s = _phys_connectivity(shapes, tol_mm=tol)
    interference_s = _phys_interference(shapes)

    # Radial attachment: beam starts near pillar surface
    if cylinders and beams:
        pillar = max(cylinders, key=lambda s: s.get("height", 0))
        pr = pillar.get("radius", 5.0)
        pc = _shape_center(pillar) or [0,0,0]
        errs = []
        for b in beams:
            st = b.get("start", [0,0,0])
            d_xy = math.sqrt((st[0]-pc[0])**2 + (st[1]-pc[1])**2)
            errs.append(min(1.0, abs(d_xy - pr) / max(pr, 1)))
        radial_s = max(0, 1.0 - sum(errs)/len(errs)) if errs else 0.0
    else:
        radial_s = 0.0

    # ── Pattern checks (secondary: 40% total weight) ─────────
    z_spacing_s = _check_uniform_z_spacing(beams)

    beam_data = []
    for b in beams:
        c = _shape_center(b)
        if c and (abs(c[0]) > 0.1 or abs(c[1]) > 0.1):
            beam_data.append((c[2], math.atan2(c[1], c[0])))
    beam_data.sort()
    if len(beam_data) >= 3:
        diffs = []
        for i in range(len(beam_data)-1):
            d = beam_data[i+1][1] - beam_data[i][1]
            while d > math.pi: d -= 2*math.pi
            while d < -math.pi: d += 2*math.pi
            diffs.append(abs(d))
        mean_d = sum(diffs)/len(diffs) if diffs else 0
        dev = sum(abs(x - mean_d)/max(mean_d, 0.01) for x in diffs)/len(diffs) if diffs and mean_d > 0.01 else 1.0
        angle_reg_s = max(0, 1.0 - dev)
    else:
        angle_reg_s = 0.0

    # ── Weighted sum ─────────────────────────────────────────
    raw = (gravity_s      * 0.20 +
           connectivity_s * 0.15 +
           radial_s       * 0.25 +
           interference_s * 0.10 +
           z_spacing_s    * 0.15 +
           angle_reg_s    * 0.15)

    # ── Physics gate ─────────────────────────────────────────
    # If gravity + connectivity + radial are poor, the structure is physically
    # invalid — crush the score regardless of pattern quality.
    physics_avg = (gravity_s + connectivity_s + radial_s) / 3.0
    if physics_avg < 0.4:
        raw *= physics_avg / 0.4

    return raw

def _sem_pyramid(shapes, golden, scale):
    """Cannonball Pyramid: dense-packed sphere stacking."""
    spheres = [s for s in shapes if s.get("type") == "sphere"]
    w = {}
    w['gravity']      = (_phys_gravity(shapes, tol_mm=12.0), 0.20)
    w['interference']  = (_phys_interference(shapes, tol_mm=2.0), 0.10)
    w['tangency']     = (_check_tangent_spheres(shapes), 0.25)
    z_vals = sorted(set(round(_shape_center(s)[2], 1) for s in spheres if _shape_center(s)))
    expected_layers = scale
    layer_score = min(1.0, len(z_vals) / max(1, expected_layers)) if z_vals else 0.0
    w['layers']       = (layer_score, 0.20)
    w['size_uniform'] = (_check_uniform_sizes(spheres, "radius"), 0.25)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=12.0)

def _sem_voxel(shapes, golden, scale):
    """Voxel Grid: regular 3D box lattice with uniform gaps."""
    boxes = [s for s in shapes if s.get("type") == "box"]
    w = {}
    w['gravity']      = (_phys_gravity(shapes), 0.15)
    w['interference']  = (_phys_interference(shapes, tol_mm=0.5), 0.10)
    for axis_name, axis_idx in [('x', 0), ('y', 1), ('z', 2)]:
        vals = sorted(set(round(_shape_center(b)[axis_idx], 1) for b in boxes if _shape_center(b)))
        if len(vals) >= 2:
            gaps = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
            mean_g = sum(gaps)/len(gaps)
            dev = sum(abs(g-mean_g)/max(mean_g, 0.01) for g in gaps)/len(gaps) if mean_g > 0.01 else 1.0
            w[f'grid_{axis_name}'] = (max(0, 1.0 - dev), 0.20)
        else:
            w[f'grid_{axis_name}'] = (0.5, 0.20)
    if boxes:
        sizes = [tuple(sorted(s.get("size", [0,0,0]))) for s in boxes if s.get("size")]
        if sizes:
            ref = sizes[0]
            matches = sum(1 for s in sizes if all(abs(s[i]-ref[i]) < 0.5 for i in range(3)))
            w['size_uniform'] = (matches / len(sizes), 0.15)
        else:
            w['size_uniform'] = (0.0, 0.15)
    else:
        w['size_uniform'] = (0.0, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes)

def _sem_stonehenge(shapes, golden, scale):
    """Domino Ring / Stonehenge: circular archway array."""
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    w['gravity']      = (_phys_gravity(shapes), 0.18)
    w['connectivity']  = (_phys_connectivity(shapes, tol_mm=5.0), 0.17)
    w['interference']  = (_phys_interference(shapes), 0.08)
    # Circular distribution of pillars
    w['angular_reg']  = (_check_angular_regularity(cylinders), 0.20)
    # Constant radius from center
    w['const_radius'] = (_check_constant_radius(cylinders), 0.15)
    # Header beams connect pillar tops — check beams at correct Z
    if beams and cylinders:
        pillar_top_z = max(c.get("height", 0) for c in cylinders)
        beam_z_scores = []
        for b in beams:
            bstart = b.get("start", [0,0,0])
            bend = b.get("end", [0,0,0])
            avg_z = (bstart[2] + bend[2]) / 2.0
            beam_z_scores.append(max(0, 1.0 - abs(avg_z - pillar_top_z) / max(pillar_top_z, 1)))
        w['header_z'] = (sum(beam_z_scores)/len(beam_z_scores) if beam_z_scores else 0.0, 0.15)
    else:
        w['header_z'] = (0.0, 0.15)
    w['pillar_size']  = (_check_uniform_sizes(cylinders, "radius"), 0.12)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=5.0)

def _sem_dna(shapes, golden, scale):
    """DNA Helix: double-helix with uniform pitch and anti-parallel pairing."""
    spheres = [s for s in shapes if s.get("type") == "sphere"]
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    # DNA has intentional close contacts at rungs — relaxed tolerance
    w['interference']  = (_phys_interference(shapes, tol_mm=2.0), 0.10)
    # Uniform Z pitch for spheres
    w['z_pitch']      = (_check_uniform_z_spacing(spheres), 0.20)
    # Backbone radius consistency
    w['const_radius'] = (_check_constant_radius(spheres), 0.20)
    # Anti-parallel pairing: spheres at same Z should be ~180° apart
    z_groups = {}
    for s in spheres:
        c = _shape_center(s)
        if c:
            zk = round(c[2], 1)
            z_groups.setdefault(zk, []).append(c)
    pair_scores = []
    for zk, pts in z_groups.items():
        if len(pts) == 2:
            a1 = math.atan2(pts[0][1], pts[0][0])
            a2 = math.atan2(pts[1][1], pts[1][0])
            diff = abs(a1 - a2)
            diff = min(diff, 2*math.pi - diff)
            pair_scores.append(max(0, 1.0 - abs(diff - math.pi) / math.pi))
    w['antiparallel'] = (sum(pair_scores)/len(pair_scores) if pair_scores else 0.0, 0.20)
    # Sphere size uniformity
    w['size_uniform'] = (_check_uniform_sizes(spheres, "radius"), 0.10)
    # Rung beams connect sphere pairs at same Z
    if beams and spheres:
        rung_ok = 0
        for b in beams:
            bs, be = b.get("start", [0,0,0]), b.get("end", [0,0,0])
            if abs(bs[2] - be[2]) < 2.0:  # horizontal rung
                rung_ok += 1
        w['rung_horiz'] = (rung_ok / max(1, len(beams)), 0.20)
    else:
        w['rung_horiz'] = (0.0, 0.20)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=3.0)

def _sem_bridge(shapes, golden, scale):
    """Suspension Bridge: deck + towers + symmetric cables."""
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    w['gravity']      = (_phys_gravity(shapes), 0.18)
    w['connectivity']  = (_phys_connectivity(shapes, tol_mm=5.0), 0.17)
    w['interference']  = (_phys_interference(shapes), 0.10)
    # Deck colinearity: longest beam should be ~horizontal
    if beams:
        deck = max(beams, key=lambda b: _dist3(b.get("start",[0,0,0]), b.get("end",[0,0,0])))
        ds, de = deck.get("start", [0,0,0]), deck.get("end", [0,0,0])
        z_diff = abs(ds[2] - de[2])
        horiz_len = math.sqrt((ds[0]-de[0])**2 + (ds[1]-de[1])**2)
        w['deck_horiz'] = (max(0, 1.0 - z_diff / max(horiz_len, 1)), 0.15)
    else:
        w['deck_horiz'] = (0.0, 0.15)
    # Tower grounding: cylinders base at Z=0
    if cylinders:
        grounded = sum(1 for c in cylinders
                       if (_shape_center(c) or [0,0,0])[2] - c.get("height",0)/2 < 2.0)
        w['tower_ground'] = (grounded / len(cylinders), 0.15)
    else:
        w['tower_ground'] = (0.0, 0.15)
    # Cable symmetry: check L/R cable count balance
    cable_beams = [b for b in beams if b != max(beams, key=lambda b: _dist3(b.get("start",[0,0,0]), b.get("end",[0,0,0])))] if beams else []
    left_cables = [b for b in cable_beams if (_shape_center(b) or [0,0,0])[0] < 0]
    right_cables = [b for b in cable_beams if (_shape_center(b) or [0,0,0])[0] > 0]
    nl, nr = len(left_cables), len(right_cables)
    sym = 1.0 - abs(nl - nr) / max(nl + nr, 1)
    w['cable_sym'] = (sym, 0.15)
    # Cable endpoints evenly spaced on deck
    if cable_beams:
        deck_xs = sorted([(_shape_center(b) or [0,0,0])[0] for b in cable_beams])
        if len(deck_xs) >= 3:
            # check spacing in left and right halves independently
            w['cable_dist'] = (0.5 * _spacing_regularity([x for x in deck_xs if x < 0])
                              + 0.5 * _spacing_regularity([x for x in deck_xs if x > 0]), 0.15)
        else:
            w['cable_dist'] = (0.5, 0.15)
    else:
        w['cable_dist'] = (0.0, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=5.0)

def _spacing_regularity(vals):
    """Helper: how regular are sorted values spaced? 0.0–1.0."""
    if len(vals) < 2:
        return 1.0
    vals = sorted(vals)
    gaps = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
    if not gaps:
        return 1.0
    mean_g = sum(gaps)/len(gaps)
    if mean_g < 0.01:
        return 0.0
    dev = sum(abs(g - mean_g)/mean_g for g in gaps)/len(gaps)
    return max(0.0, 1.0 - dev)

def _sem_planetary(shapes, golden, scale):
    """Planetary Array: tangential sun+planet gear arrangement."""
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    w = {}
    w['interference']  = (_phys_interference(shapes, tol_mm=0.3), 0.15)
    # Identify sun (largest radius at center) and planets
    if cylinders:
        sun = max(cylinders, key=lambda c: c.get("radius", 0))
        planets = [c for c in cylinders if c is not sun]
    else:
        sun, planets = None, []
    # Tangential contact: each planet center distance = sun_r + planet_r
    if sun and planets:
        sr = sun.get("radius", 10)
        sc = _shape_center(sun) or [0,0,0]
        tang_scores = []
        for p in planets:
            pr = p.get("radius", 3)
            pc = _shape_center(p) or [0,0,0]
            dist = math.sqrt((pc[0]-sc[0])**2 + (pc[1]-sc[1])**2)
            expected = sr + pr
            err = abs(dist - expected) / max(expected, 1)
            tang_scores.append(max(0, 1.0 - err))
        w['tangency'] = (sum(tang_scores)/len(tang_scores), 0.25)
    else:
        w['tangency'] = (0.0, 0.25)
    # Even angular distribution of planets
    w['angular_reg'] = (_check_angular_regularity(planets, *((_shape_center(sun) or [0,0])[:2])) if sun else 0.0, 0.25)
    # Coplanarity (all in same Z plane)
    w['coplanar']    = (_check_coplanar_z(cylinders), 0.20)
    # Planet size uniformity
    w['planet_size'] = (_check_uniform_sizes(planets, "radius") if planets else 0.0, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=3.0)

def _sem_truss(shapes, golden, scale):
    """Cross-Braced Truss: stories of pillars + X-diagonal braces."""
    beams = [s for s in shapes if s.get("type") == "beam"]
    tol = max(3.0, scale * 1.5)
    w = {}
    w['gravity']      = (_phys_gravity(shapes, tol_mm=2.0), 0.15)
    w['connectivity']  = (_phys_connectivity(shapes, tol_mm=tol), 0.20)
    w['interference']  = (_phys_interference(shapes), 0.10)
    # Story stacking: Z endpoints cluster at uniform heights
    z_endpoints = set()
    for b in beams:
        for pt in [b.get("start"), b.get("end")]:
            if pt:
                z_endpoints.add(round(pt[2], 0))
    z_levels = sorted(z_endpoints)
    if len(z_levels) >= 3:
        gaps = [z_levels[i+1]-z_levels[i] for i in range(len(z_levels)-1)]
        mean_g = sum(gaps)/len(gaps)
        dev = sum(abs(g-mean_g)/max(mean_g, 0.01) for g in gaps)/len(gaps) if mean_g > 0.01 else 1.0
        w['story_height'] = (max(0, 1.0 - dev), 0.20)
    else:
        w['story_height'] = (0.5, 0.20)
    # Square footprint: XY endpoints should cluster at 4 corner positions
    xy_pts = set()
    for b in beams:
        for pt in [b.get("start"), b.get("end")]:
            if pt:
                xy_pts.add((round(pt[0], 0), round(pt[1], 0)))
    n_xy = len(xy_pts)
    # For a perfect truss, exactly 4 unique XY positions
    w['square_foot'] = (min(1.0, 4.0 / max(n_xy, 1)), 0.20)
    # Beam count per story should be 12 (4 pillars + 8 diagonals)
    expected_per_story = 12
    expected_total = scale * expected_per_story
    count_score = min(1.0, len(beams) / max(expected_total, 1))
    w['beam_count'] = (count_score, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=tol)

def _sem_fractal(shapes, golden, scale):
    """Fractal Y-Tree: recursive binary branching in XZ plane."""
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    w['interference']  = (_phys_interference(shapes, tol_mm=0.5), 0.10)
    # Planarity: all beams in XZ plane (Y ≈ 0)
    w['planarity']    = (_check_planarity_y0(beams), 0.20)
    # Parent-child continuity: each beam end should match another beam start
    if len(beams) >= 2:
        endpoints = []
        startpoints = []
        for b in beams:
            endpoints.append(tuple(b.get("end", [0,0,0])))
            startpoints.append(tuple(b.get("start", [0,0,0])))
        matches = 0
        for ep in endpoints:
            for sp in startpoints:
                if _dist3(list(ep), list(sp)) < 1.0:
                    matches += 1
                    break
        # Root has no parent, so max matches = len-1
        w['continuity'] = (matches / max(1, len(beams) - 1), 0.20)
    else:
        w['continuity'] = (0.0, 0.20)
    # Length halving: child beams should be ~half their parent's length
    beam_lengths = sorted([_dist3(b.get("start",[0,0,0]), b.get("end",[0,0,0])) for b in beams], reverse=True)
    if len(beam_lengths) >= 3:
        # Group by approximate length level
        ratios = []
        for i in range(len(beam_lengths)-1):
            if beam_lengths[i] > 1e-3:
                r = beam_lengths[i+1] / beam_lengths[i]
                if 0.3 < r < 0.8:  # should be ~0.5
                    ratios.append(abs(r - 0.5) / 0.5)
        w['halving'] = (max(0, 1.0 - sum(ratios)/len(ratios)) if ratios else 0.5, 0.25)
    else:
        w['halving'] = (0.5, 0.25)
    # Expected beam count: 2^(depth+1) - 1
    expected = 2**(scale+1) - 1
    count_score = min(1.0, len(beams) / max(expected, 1))
    w['beam_count'] = (count_score, 0.25)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=2.0)

def _sem_bcc(shapes, golden, scale):
    """BCC Lattice: body-centered cubic crystal structure."""
    spheres = [s for s in shapes if s.get("type") == "sphere"]
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    w['interference']  = (_phys_interference(shapes, tol_mm=0.3), 0.10)
    w['connectivity']  = (_phys_connectivity(shapes, tol_mm=8.0), 0.15)
    # Lattice regularity: sphere positions on regular grid
    if spheres:
        # Check if positions cluster to grid-aligned values
        for axis_idx, axis_name in enumerate(('x', 'y', 'z')):
            vals = sorted(set(round((_shape_center(s) or [0,0,0])[axis_idx], 0) for s in spheres))
            if len(vals) >= 2:
                gaps = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
                # BCC should have 5-unit spacing (corners at 0,10,20 and centers at 5,15)
                w[f'grid_{axis_name}'] = (_spacing_regularity(vals), 0.10)
            else:
                w[f'grid_{axis_name}'] = (0.5, 0.10)
    else:
        for axis_name in ('x', 'y', 'z'):
            w[f'grid_{axis_name}'] = (0.0, 0.10)
    # Center-to-corner beams: expected 8*N^3 beams
    expected_beams = 8 * (scale ** 3)
    beam_score = min(1.0, len(beams) / max(expected_beams, 1))
    w['beam_count'] = (beam_score, 0.20)
    # Sphere count: (N+1)^3 corners + N^3 centers (with dedup)
    expected_spheres = (scale+1)**3 + scale**3
    sphere_score = min(1.0, len(spheres) / max(expected_spheres, 1))
    w['sphere_count'] = (sphere_score, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    # Physics gate: only apply when beams are present.
    # Missing beams = incomplete model, not physically broken —
    # don't crush the node-placement score just because connectivity = 0.
    if beams:
        return _apply_physics_gate(raw, shapes, tol_mm=8.0)
    # No beams: partial credit for node structure only, capped at 45%
    # (a nodes-only output can never be a correct BCC lattice)
    return min(raw, 0.45)

def _sem_furniture(shapes, ops, specs):
    """Furniture Assembly (Phase 3): table with support legs."""
    w = {}
    w['gravity'] = (_phys_gravity(shapes), 0.15)
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=3.0), 0.15)
    w['interference'] = (_phys_interference(shapes), 0.10)
    # Mates: legs coincident to tabletop
    if "mates" in specs:
        res = validate_mates(shapes, specs["mates"])
        chk = max(1, res.get("checked", 1))
        w['mates'] = (res.get("correct", 0) / chk, 0.30)
    else:
        w['mates'] = (0.5, 0.30)
    # Leg angular symmetry
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    w['leg_symmetry'] = (_check_angular_regularity(cylinders), 0.15)
    # Leg size uniformity
    w['leg_uniform'] = (_check_uniform_sizes(cylinders, "radius"), 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=3.0)

def _sem_manifold(shapes, ops, specs):
    """Pipe Manifold (Phase 3): header pipe with branches, flanges, valves, brackets.

    Strengthened evaluator — additional structural checks:
      - Branch count matches expected N
      - End-cap presence (2 cylinders closing header ends)
      - Wall/mounting surface presence
      - Flange count and flange-branch pairing
      - Valve count per branch
      - Header elevation above ground
      - Branch-header connectivity (branches must touch header)
    """
    w = {}
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    boxes     = [s for s in shapes if s.get("type") == "box"]

    # --- Helper: get normalised axis vector for a shape (default Z-up) ---
    def _get_axis(s):
        ax = s.get("axis", [0, 0, 1])
        if isinstance(ax, list) and len(ax) == 3:
            mag = math.sqrt(sum(v*v for v in ax))
            return [v / mag for v in ax] if mag > 1e-9 else [0, 0, 1]
        return [0, 0, 1]

    def _dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    # Identify header (longest pipe)
    header = None
    header_axis = None
    if pipes:
        header = max(pipes, key=lambda p: p.get("height", 0))
        header_axis = _get_axis(header)

    # Identify branch pipes (shorter pipes, perpendicular to header)
    branch_pipes = []
    flange_pipes = []
    if header and header_axis:
        non_header = [p for p in pipes if p is not header]
        for p in non_header:
            pax = _get_axis(p)
            dot_val = abs(_dot(pax, header_axis))
            h = p.get("height", 0)
            # Branches: perpendicular to header, reasonably long
            if dot_val < 0.3 and h > 15:
                branch_pipes.append(p)
            # Flanges: perpendicular, short (< 10mm)
            elif dot_val < 0.3 and h <= 15:
                flange_pipes.append(p)

    # --- Physics (12%) ---
    w['gravity']      = (_phys_gravity(shapes), 0.04)
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=5.0), 0.04)
    w['interference'] = (_phys_interference(shapes, tol_mm=0.5), 0.04)

    # --- Header orientation (8%) ---
    header_orient_score = 0.0
    if header_axis:
        z_component = abs(header_axis[2])
        header_orient_score = max(0.0, 1.0 - z_component)
    w['header_orient'] = (header_orient_score, 0.08)

    # --- Header elevation (5%) ---
    # Header should be elevated above ground (Z > 20mm typically)
    header_elev_score = 0.0
    if header:
        hc = _shape_center(header)
        if hc and hc[2] > 15.0:
            header_elev_score = min(1.0, hc[2] / 50.0)
    w['header_elevation'] = (header_elev_score, 0.05)

    # --- Branch count (8%) ---
    # Infer expected N from reference shapes
    ref = specs.get("reference", [])
    ref_pipes = [s for s in ref if s.get("type") == "pipe"]
    # In reference: 1 header + N branches + N flanges = 1 + 2N pipes
    expected_branches = max(1, (len(ref_pipes) - 1) // 2) if ref_pipes else 2
    branch_count_score = 0.0
    if expected_branches > 0:
        ratio = len(branch_pipes) / expected_branches
        if 0.8 <= ratio <= 1.2:
            branch_count_score = 1.0
        elif ratio > 0:
            branch_count_score = max(0.0, 1.0 - abs(ratio - 1.0))
    w['branch_count'] = (branch_count_score, 0.08)

    # --- Branch perpendicularity and consistency (10%) ---
    branch_perp_score = 0.0
    branch_consistency_score = 0.0
    if branch_pipes and header_axis:
        perp_scores = []
        branch_axes = []
        for bp in branch_pipes:
            bax = _get_axis(bp)
            branch_axes.append(bax)
            dot_val = abs(_dot(bax, header_axis))
            perp_scores.append(max(0.0, 1.0 - dot_val * 2.0))
        if perp_scores:
            branch_perp_score = sum(perp_scores) / len(perp_scores)
        if len(branch_axes) >= 2:
            ref_ax = branch_axes[0]
            consist_scores = [abs(_dot(bax, ref_ax)) for bax in branch_axes[1:]]
            branch_consistency_score = sum(consist_scores) / len(consist_scores)
        else:
            branch_consistency_score = 1.0
    w['branch_perp'] = (branch_perp_score, 0.06)
    w['branch_consistency'] = (branch_consistency_score, 0.04)

    # --- Flange count (5%) ---
    # Should have one flange per branch
    flange_score = 0.0
    if expected_branches > 0:
        flange_ratio = len(flange_pipes) / expected_branches
        flange_score = min(1.0, flange_ratio)
    w['flange_count'] = (flange_score, 0.05)

    # --- End-cap presence (5%) ---
    # 2 short cylinders at header ends, aligned with header axis
    endcap_score = 0.0
    if header_axis and cylinders:
        endcaps = []
        for c in cylinders:
            cax = _get_axis(c)
            # Must be aligned with header (|dot| > 0.8) and short
            if abs(_dot(cax, header_axis)) > 0.7 and c.get("height", 999) < 15.0:
                endcaps.append(c)
        endcap_score = min(1.0, len(endcaps) / 2.0)
    w['endcap_count'] = (endcap_score, 0.05)

    # --- Valve count (4%) ---
    # Should have N valve cylinders (one per branch), perpendicular to header
    valve_cyls = []
    if header_axis and cylinders:
        for c in cylinders:
            cax = _get_axis(c)
            if abs(_dot(cax, header_axis)) < 0.3:  # perpendicular to header
                valve_cyls.append(c)
    valve_score = min(1.0, len(valve_cyls) / max(1, expected_branches))
    w['valve_count'] = (valve_score, 0.04)

    # --- Mates (10%) ---
    if "mates" in specs:
        res = validate_mates(shapes, specs["mates"])
        chk = max(1, res.get("checked", 1))
        w['mates'] = (res.get("correct", 0) / chk, 0.10)
    else:
        w['mates'] = (0.5, 0.10)

    # --- Clearance fit (8%) ---
    if "clearance_fit" in specs:
        res = validate_clearance_fit(shapes, specs["clearance_fit"])
        chk = max(1, res.get("checked", 1))
        w['clearance'] = (res.get("correct", 0) / chk, 0.08)
    else:
        w['clearance'] = (0.5, 0.08)

    # --- Branch spacing uniformity (8%) ---
    if len(branch_pipes) >= 2:
        # Project branch centres onto header axis for spacing check
        branch_centres = [_shape_center(p) for p in branch_pipes if _shape_center(p)]
        if header_axis and len(branch_centres) >= 2:
            # Project onto header axis direction
            projections = sorted(_dot(c, header_axis) for c in branch_centres)
            gaps = [projections[i+1] - projections[i] for i in range(len(projections)-1)]
            mean_gap = sum(gaps) / len(gaps) if gaps else 1.0
            if mean_gap > 0.1:
                dev = sum(abs(g - mean_gap) / mean_gap for g in gaps) / len(gaps)
                w['spacing'] = (max(0.0, 1.0 - dev), 0.08)
            else:
                w['spacing'] = (0.0, 0.08)
        else:
            w['spacing'] = (0.0, 0.08)
    else:
        w['spacing'] = (0.5, 0.08)

    # --- Type variety (3%) ---
    types = set(s.get("type") for s in shapes)
    expected_types = {"box", "cylinder", "pipe"}
    w['type_variety'] = (len(types & expected_types) / len(expected_types), 0.03)

    # --- Bracket ground contact (5%) ---
    # Brackets should sit on Z=0 ground and reach up to header
    bracket_score = 0.0
    if boxes:
        grounded = 0
        for b in boxes:
            bc = _shape_center(b)
            sz = b.get("size", [0, 0, 0])
            if bc and isinstance(sz, list) and len(sz) == 3:
                base_z = bc[2] - sz[2] / 2.0
                if abs(base_z) < 3.0:
                    grounded += 1
        bracket_score = grounded / len(boxes)
    w['bracket_ground'] = (bracket_score, 0.05)

    # --- Wall presence (4%) ---
    # Should have a back wall (large box behind header)
    wall_score = 0.0
    if boxes:
        for b in boxes:
            sz = b.get("size", [0, 0, 0])
            if isinstance(sz, list) and len(sz) == 3:
                # Wall is thin in one dimension, large in the other two
                dims = sorted(sz)
                if dims[0] < 10.0 and dims[1] > 50.0 and dims[2] > 30.0:
                    wall_score = 1.0
                    break
    w['wall_presence'] = (wall_score, 0.04)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=5.0)

def _sem_axle(shapes, ops, specs):
    """Axle Bearing (Phase 3): rotational shaft with bearings in support block.

    Strengthened evaluator — structural checks beyond clearance/mates:
      - Block presence and orientation
      - Shaft horizontal alignment and length vs block width
      - Bore full penetration through block
      - Bearing count and symmetrical placement
      - Shaft protrusion beyond block (functional requirement)
    """
    boxes     = [s for s in shapes if s.get("type") == "box"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    w = {}

    # ── Physics (15%) ────────────────────────────────────────
    w['gravity']      = (_phys_gravity(shapes), 0.05)
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=3.0), 0.05)
    if "interference_check" in specs:
        ir = validate_interference(shapes, ops, tol_mm=0.2)
        w['interference'] = (1.0 if ir.get("interference_free", False) else 0.0, 0.05)
    else:
        w['interference'] = (_phys_interference(shapes), 0.05)

    # ── Clearance fit (15%) ──────────────────────────────────
    if "clearance_fit" in specs:
        res = validate_clearance_fit(shapes, specs["clearance_fit"])
        chk = max(1, res.get("checked", 1))
        w['clearance'] = (res.get("correct", 0) / chk, 0.15)
    else:
        w['clearance'] = (0.5, 0.15)

    # ── Mates / concentricity (15%) ──────────────────────────
    if "mates" in specs:
        res = validate_mates(shapes, specs["mates"])
        chk = max(1, res.get("checked", 1))
        w['mates'] = (res.get("correct", 0) / chk, 0.15)
    else:
        w['mates'] = (0.5, 0.15)

    # ── Shape count and variety (10%) ────────────────────────
    # Expect: 1 box (block), 2 cylinders (bore + shaft), 2 pipes (bearings)
    types = set(s.get("type") for s in shapes)
    expected_types = {"box", "cylinder", "pipe"}
    w['type_variety'] = (len(types & expected_types) / len(expected_types), 0.05)
    # Exact shape count: 5 shapes
    count_score = min(1.0, len(shapes) / 5.0) if len(shapes) <= 7 else max(0.0, 1.0 - (len(shapes) - 5) / 10.0)
    w['shape_count'] = (count_score, 0.05)

    # ── Support block check (8%) ─────────────────────────────
    # Must have exactly 1 box sitting on ground (Z≈0 base)
    block_score = 0.0
    block = None
    if boxes:
        # Find the box with the largest volume (the support block)
        def _box_vol(b):
            sz = b.get("size", [0, 0, 0])
            return sz[0] * sz[1] * sz[2] if isinstance(sz, list) and len(sz) == 3 else 0
        block = max(boxes, key=_box_vol)
        c = _shape_center(block)
        sz = block.get("size", [0, 0, 0])
        if c and isinstance(sz, list) and len(sz) == 3:
            # Block bottom should be near Z=0
            base_z = c[2] - sz[2] / 2.0
            if abs(base_z) < 5.0:
                block_score += 0.5
            # Block should be roughly cubic or rectangular (all dims > 10mm)
            if all(d > 10.0 for d in sz):
                block_score += 0.5
    w['block_check'] = (block_score, 0.08)

    # ── Shaft alignment (10%) ────────────────────────────────
    # Shaft should be horizontal (axis ≈ X), longer than block width
    shaft_score = 0.0
    shaft = None
    if cylinders:
        # Shaft = longest cylinder
        shaft = max(cylinders, key=lambda c: c.get("height", 0))
        ax = shaft.get("axis", [0, 0, 1])
        if isinstance(ax, list) and len(ax) == 3:
            mag = math.sqrt(sum(v*v for v in ax))
            if mag > 1e-9:
                ax_norm = [v / mag for v in ax]
                # Horizontal: |x-component| should be near 1
                shaft_score += abs(ax_norm[0]) * 0.5
                # Shaft should protrude beyond block (shaft length > block width)
                if block:
                    block_w = block.get("size", [0, 0, 0])[0]
                    if shaft.get("height", 0) > block_w * 1.0:
                        shaft_score += 0.5
    w['shaft_alignment'] = (shaft_score, 0.10)

    # ── Bearing count and symmetry (12%) ─────────────────────
    # Should have exactly 2 bearings (pipes), placed symmetrically about X=0
    bearing_score = 0.0
    if len(pipes) == 2:
        bearing_score += 0.4
        c1 = _shape_center(pipes[0])
        c2 = _shape_center(pipes[1])
        if c1 and c2:
            # Symmetry: X coords should be equal and opposite (or mirrored)
            if abs(c1[0] + c2[0]) < 5.0:  # mirror about X=0
                bearing_score += 0.3
            # Both bearings horizontal (axis ≈ X)
            for p in pipes:
                ax = p.get("axis", [0, 0, 1])
                if isinstance(ax, list) and len(ax) == 3:
                    mag = math.sqrt(sum(v*v for v in ax))
                    if mag > 1e-9 and abs(ax[0] / mag) > 0.8:
                        bearing_score += 0.15
    elif len(pipes) >= 1:
        bearing_score += 0.2  # at least some bearings
    w['bearing_symmetry'] = (bearing_score, 0.12)

    # ── Bore penetration (10%) ───────────────────────────────
    # The bore cylinder should fully penetrate the block (bore length ≥ block width)
    bore_score = 0.0
    if block and len(cylinders) >= 2:
        # Bore = the cylinder that's NOT the shaft (shorter one, or one with bigger radius)
        bore_candidates = [c for c in cylinders if c is not shaft]
        if bore_candidates:
            bore = bore_candidates[0]
            bore_h = bore.get("height", 0)
            block_w = block.get("size", [0, 0, 0])[0]
            if bore_h >= block_w * 0.95:
                bore_score += 0.5
            # Bore must be coaxial with shaft
            if shaft:
                bc = _shape_center(bore)
                sc = _shape_center(shaft)
                if bc and sc:
                    # Y and Z coords should match
                    yz_dist = math.sqrt((bc[1] - sc[1])**2 + (bc[2] - sc[2])**2)
                    bore_score += max(0.0, 0.5 - yz_dist / 20.0)
    w['bore_penetration'] = (bore_score, 0.10)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=3.0)


# ── Phase 4 Bio-Inspired semantic evaluators ─────────────────

def _sem_phyllotaxis(shapes, golden, scale):
    """Phyllotaxis Disc: sunflower golden-angle Fibonacci spiral of spheres."""
    spheres = [s for s in shapes if s.get("type") == "sphere"]
    w = {}
    # Base physics
    w['gravity']      = (_phys_gravity(shapes, tol_mm=5.0), 0.05)
    w['interference']  = (_phys_interference(shapes, tol_mm=1.0), 0.10)

    # Golden angle regularity: sort spheres by angle, check angular increments ≈ 137.508°
    GOLDEN_ANGLE = math.radians(137.50776405003785)
    sphere_data = []
    for s in spheres:
        c = _shape_center(s)
        if c and (abs(c[0]) > 0.01 or abs(c[1]) > 0.01):
            angle = math.atan2(c[1], c[0])
            r = math.sqrt(c[0]**2 + c[1]**2)
            sphere_data.append((r, angle))
    # Sort by radius (inner→outer = seed index order in a Fermat spiral)
    sphere_data.sort(key=lambda x: x[0])
    if len(sphere_data) >= 3:
        # Check consecutive angular increment ≈ golden angle (mod 2π)
        angle_errs = []
        for i in range(len(sphere_data) - 1):
            diff = sphere_data[i+1][1] - sphere_data[i][1]
            # Normalize to [0, 2π)
            diff = diff % (2*math.pi)
            # Closest to golden angle
            err1 = abs(diff - GOLDEN_ANGLE) / GOLDEN_ANGLE
            err2 = abs(diff - (2*math.pi - GOLDEN_ANGLE)) / GOLDEN_ANGLE
            angle_errs.append(min(err1, err2))
        mean_err = sum(angle_errs) / len(angle_errs)
        w['golden_angle'] = (max(0.0, 1.0 - mean_err), 0.30)
    else:
        w['golden_angle'] = (0.0, 0.30)

    # Fermat spiral radius check: r(n) ∝ √n → r²(n) should be linearly spaced
    if len(sphere_data) >= 3:
        r_sq = [d[0]**2 for d in sphere_data]
        gaps = [r_sq[i+1] - r_sq[i] for i in range(len(r_sq)-1)]
        mean_gap = sum(gaps) / len(gaps) if gaps else 1.0
        if mean_gap > 0.01:
            dev = sum(abs(g - mean_gap) / mean_gap for g in gaps) / len(gaps)
            w['spiral_spacing'] = (max(0.0, 1.0 - dev * 0.5), 0.25)
        else:
            w['spiral_spacing'] = (0.0, 0.25)
    else:
        w['spiral_spacing'] = (0.0, 0.25)

    # Planarity (all seeds at same Z)
    w['planarity'] = (_check_coplanar_z(spheres), 0.10)

    # Seed count
    expected = scale
    actual = len(spheres)
    w['count'] = (min(1.0, actual / max(1, expected)), 0.10)

    # Size uniformity (all seeds same radius)
    w['size_uniform'] = (_check_uniform_sizes(spheres, "radius"), 0.10)

    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=5.0)


def _sem_compound_eye(shapes, golden, scale):
    """Compound Eye Array: hemispherical dome with ommatidia in concentric rings."""
    spheres   = [s for s in shapes if s.get("type") == "sphere"]
    cones     = [s for s in shapes if s.get("type") == "cone"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    tori      = [s for s in shapes if s.get("type") == "torus"]
    w = {}

    N_rings = scale
    n_omm = 1 + sum(6 * k for k in range(1, N_rings + 1))
    dome_R = 50.0

    # Type variety — should have at least 4 of 6 required types
    type_set = set(s.get("type") for s in shapes)
    required = {"sphere", "cone", "cylinder", "pipe", "torus"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.10)

    # Component counts — each ommatidium = 1 sphere + 1 cone + 1 cylinder
    # Plus 1 dome sphere + 1 pipe + 1 torus as supports
    expected_lens = n_omm + 1  # dome sphere + lens spheres
    expected_cone = n_omm
    expected_cyl  = n_omm
    w['lens_count'] = (min(1.0, len(spheres) / max(1, expected_lens)), 0.08)
    w['cone_count'] = (min(1.0, len(cones) / max(1, expected_cone)), 0.08)
    w['cyl_count']  = (min(1.0, len(cylinders) / max(1, expected_cyl)), 0.08)

    # Dome presence — should be a large sphere near origin
    dome_found = any(s.get("radius", 0) > 30 for s in spheres)
    w['dome_present'] = (1.0 if dome_found else 0.0, 0.06)

    # Hemispherical distribution — lens spheres should be on a dome surface
    # Check they lie at roughly dome_R from origin
    lens_radii = []
    for s in spheres:
        c = _shape_center(s)
        if c and s.get("radius", 0) < 10:  # not the dome itself
            r = math.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
            lens_radii.append(r)
    if lens_radii:
        mean_r = sum(lens_radii) / len(lens_radii)
        r_err = abs(mean_r - dome_R) / dome_R
        w['dome_radius'] = (max(0.0, 1.0 - r_err * 2), 0.12)
    else:
        w['dome_radius'] = (0.0, 0.12)

    # Ring structure — check concentric ring counts (ring k has 6k units)
    # Cluster lens spheres by polar angle from +Z axis
    lens_thetas = []
    for s in spheres:
        c = _shape_center(s)
        if c and s.get("radius", 0) < 10:
            r = math.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
            if r > 1:
                theta = math.acos(max(-1, min(1, c[2] / r)))
                lens_thetas.append(theta)
    if len(lens_thetas) >= 5:
        # Sort and cluster by theta
        lens_thetas.sort()
        clusters = [[lens_thetas[0]]]
        for t in lens_thetas[1:]:
            if t - clusters[-1][-1] < math.radians(8):
                clusters[-1].append(t)
            else:
                clusters.append([t])
        expected_rings = N_rings + 1  # ring 0..N
        w['ring_count'] = (min(1.0, len(clusters) / max(1, expected_rings)), 0.15)
    else:
        w['ring_count'] = (0.0, 0.15)

    # Upper hemisphere — most ommatidia should have Z > 0
    above = sum(1 for t in lens_thetas if t < math.pi / 2)
    w['upper_hemi'] = (above / max(1, len(lens_thetas)) if lens_thetas else 0, 0.08)

    # Radial alignment — cones and cylinders should point inward (toward origin)
    inward_count = 0
    total_axes = 0
    for s in cones + cylinders:
        c = _shape_center(s)
        ax = s.get("axis")
        if c and ax and any(abs(v) > 0.01 for v in c):
            # Inward direction = from surface toward origin = -normalize(c)
            r = math.sqrt(sum(v**2 for v in c))
            if r > 1:
                inward = [-v/r for v in c]
                # Dot product with axis (either direction is fine for alignment)
                dot = abs(sum(inward[i] * ax[i] for i in range(3)))
                if dot > 0.5:
                    inward_count += 1
                total_axes += 1
    w['axis_alignment'] = (inward_count / max(1, total_axes), 0.15)

    # Connectivity — shapes should form connected groups
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=15.0), 0.10)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=15.0)


def _sem_diatom(shapes, golden, scale):
    """Diatom Frustule: bilateral symmetric silica shell with costae, areolae, girdle bands."""
    boxes     = [s for s in shapes if s.get("type") == "box"]
    beams     = [s for s in shapes if s.get("type") == "beam"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    spheres   = [s for s in shapes if s.get("type") == "sphere"]
    tori      = [s for s in shapes if s.get("type") == "torus"]
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    w = {}

    N = scale  # costae per half-valve

    # Type variety — should have at least 5 of 6 required types
    type_set = set(s.get("type") for s in shapes)
    required = {"box", "beam", "cylinder", "sphere", "torus", "pipe"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.10)

    # Valve count — expect 2 boxes (top and bottom valves)
    w['valve_count'] = (min(1.0, len(boxes) / 2.0), 0.08)

    # Valve symmetry — two valves should be at ±Z, symmetric about Z=0
    if len(boxes) >= 2:
        zs = sorted([_shape_center(b)[2] for b in boxes if _shape_center(b)])
        if len(zs) >= 2:
            # Check that the two extreme Z values are symmetric
            sym_err = abs(zs[0] + zs[-1]) / max(1, abs(zs[-1] - zs[0]))
            w['valve_symmetry'] = (max(0.0, 1.0 - sym_err * 2), 0.08)
        else:
            w['valve_symmetry'] = (0.0, 0.08)
    else:
        w['valve_symmetry'] = (0.0, 0.08)

    # Raphe count — expect 2 beams
    w['raphe_count'] = (min(1.0, len(beams) / 2.0), 0.06)

    # Costa count — expect 4 × N × 2 (mirror) = 8N costae
    expected_costae = 8 * N
    w['costa_count'] = (min(1.0, len(cylinders) / max(1, expected_costae)), 0.10)

    # Costa regularity — costae should be evenly spaced along X
    costa_x = sorted([abs(_shape_center(c)[0]) for c in cylinders
                      if _shape_center(c) and abs(_shape_center(c)[0]) > 0.5])
    if len(costa_x) >= 4:
        # Check spacing uniformity
        diffs = [costa_x[i+1] - costa_x[i] for i in range(len(costa_x) - 1)]
        if diffs:
            mean_d = sum(diffs) / len(diffs)
            if mean_d > 0.01:
                var = sum((d - mean_d)**2 for d in diffs) / len(diffs)
                cv = math.sqrt(var) / mean_d  # coefficient of variation
                w['costa_regularity'] = (max(0.0, 1.0 - cv * 2), 0.12)
            else:
                w['costa_regularity'] = (0.0, 0.12)
        else:
            w['costa_regularity'] = (0.0, 0.12)
    else:
        w['costa_regularity'] = (0.0, 0.12)

    # Bilateral symmetry of costae — X positions should be mirrored
    if len(costa_x) >= 2:
        pos_x = sorted(set(round(abs(_shape_center(c)[0]), 1) for c in cylinders
                           if _shape_center(c)))
        # For each |x|, count costae with +x and -x
        mirror_score = 0
        for xv in pos_x:
            n_pos = sum(1 for c in cylinders if _shape_center(c) and abs(_shape_center(c)[0] - xv) < 1)
            n_neg = sum(1 for c in cylinders if _shape_center(c) and abs(_shape_center(c)[0] + xv) < 1)
            if n_pos > 0 and n_neg > 0:
                mirror_score += min(n_pos, n_neg) / max(n_pos, n_neg)
        w['bilateral_sym'] = (mirror_score / max(1, len(pos_x)), 0.12)
    else:
        w['bilateral_sym'] = (0.0, 0.12)

    # Areolae count — expect 8 × N small spheres (minus the nodule)
    small_spheres = [s for s in spheres if s.get("radius", 0) < 2.0]
    expected_areolae = 8 * N
    w['areola_count'] = (min(1.0, len(small_spheres) / max(1, expected_areolae)), 0.08)

    # Girdle bands — expect 2 torus rings
    w['girdle_count'] = (min(1.0, len(tori) / 2.0), 0.06)

    # Mantle wall — expect 1 pipe
    w['mantle_count'] = (min(1.0, len(pipes) / 1.0) if pipes else 0.0, 0.05)

    # Central nodule — expect 1 large sphere near origin
    nodule_found = any(s.get("radius", 0) > 1.5 and
                       all(abs(v) < 5 for v in (_shape_center(s) or [99,99,99]))
                       for s in spheres)
    w['nodule'] = (1.0 if nodule_found else 0.0, 0.05)

    # Overall connectivity
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=12.0), 0.10)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=12.0)


# ── NEW BIO-INSPIRED EVALUATORS ──────────────────────────────

def _sem_honeycomb(shapes, golden, scale):
    """Honeycomb Lattice: tilted hex pipe cells + caps + bottom cones + shared
    wall beams + radial reinforcement ribs + base plate + frame torus.

    15 weighted sub-checks testing physics, topology, geometry, and variety.
    """
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    tori      = [s for s in shapes if s.get("type") == "torus"]
    cones     = [s for s in shapes if s.get("type") == "cone"]
    beams     = [s for s in shapes if s.get("type") == "beam"]
    w = {}

    N_rings = scale
    n_cells = 1 + 3 * N_rings * (N_rings + 1)
    cell_outer_r = 5.0
    hex_dist = 2.0 * cell_outer_r  # 10mm centre-to-centre
    tilt_deg = 13.0
    n_ribs = 6

    # Expected wall count: each interior edge of the hex grid
    # For a hex grid with R rings, walls = 3*R*(R+1) + 3*R*(R-1) + ...
    # Simpler: each cell has up to 6 neighbours, each pair counted once
    # Exact formula isn't needed — we count from golden
    golden_beams = [s for s in golden if s.get("type") == "beam"]
    n_walls_expected = len(golden_beams) - n_ribs  # wall beams = total beams − 6 ribs

    # ── Physics (12%) ────────────────────────────────────────────
    w['gravity']      = (_phys_gravity(shapes, tol_mm=5.0), 0.06)
    w['interference']  = (_phys_interference(shapes, tol_mm=1.0), 0.06)

    # ── Cell pipes (15%) ─────────────────────────────────────────
    w['cell_count'] = (min(1.0, len(pipes) / max(1, n_cells)), 0.10)
    # Check pipe dimensions (inner/outer radius)
    if pipes:
        dim_ok = 0
        for p in pipes:
            ir = p.get("inner_radius", 0)
            orr = p.get("outer_radius", 0)
            if abs(ir - 4.2) < 1.0 and abs(orr - 5.0) < 1.0:
                dim_ok += 1
        w['pipe_dims'] = (dim_ok / len(pipes), 0.05)
    else:
        w['pipe_dims'] = (0.0, 0.05)

    # ── Cell tilt (10%) — non-centre pipes should have non-vertical axis ──
    if pipes:
        tilt_scores = []
        for p in pipes:
            ax = p.get("axis", [0, 0, 1])
            if isinstance(ax, (list, tuple)) and len(ax) >= 3:
                try:
                    az = float(ax[2])
                except (ValueError, TypeError):
                    az = 1.0
                c = _shape_center(p)
                if c and (abs(c[0]) > 1.0 or abs(c[1]) > 1.0):
                    # Non-centre cell — axis should be tilted (az < 1.0)
                    # cos(13°) ≈ 0.9744, so az should be near 0.974
                    expected_az = math.cos(math.radians(tilt_deg))
                    err = abs(az - expected_az)
                    tilt_scores.append(max(0.0, 1.0 - err * 5.0))
                else:
                    # Centre cell — should be vertical (az ≈ 1.0)
                    tilt_scores.append(max(0.0, 1.0 - abs(az - 1.0) * 5.0))
            else:
                tilt_scores.append(0.0)
        w['cell_tilt'] = (sum(tilt_scores) / len(tilt_scores), 0.10)
    else:
        w['cell_tilt'] = (0.0, 0.10)

    # ── Hex regularity (12%) — nearest-neighbour distance ≈ 10mm ─
    pipe_centres = [_shape_center(p) for p in pipes if _shape_center(p)]
    if len(pipe_centres) >= 3:
        nn_dists = []
        for i, c1 in enumerate(pipe_centres):
            min_d = float('inf')
            for j, c2 in enumerate(pipe_centres):
                if i == j:
                    continue
                d = math.sqrt(sum((a - b)**2 for a, b in zip(c1[:2], c2[:2])))
                if d < min_d:
                    min_d = d
            nn_dists.append(min_d)
        mean_nn = sum(nn_dists) / len(nn_dists)
        if mean_nn > 0.01:
            # Check against expected 10mm
            target_err = abs(mean_nn - hex_dist) / hex_dist
            dev = sum(abs(d - mean_nn) / mean_nn for d in nn_dists) / len(nn_dists)
            w['hex_regularity'] = (max(0.0, 1.0 - dev * 2.0 - target_err), 0.12)
        else:
            w['hex_regularity'] = (0.0, 0.12)
    else:
        w['hex_regularity'] = (0.0, 0.12)

    # ── Top caps (8%) — expect n_cells small cylinders ────────────
    small_cyls = [s for s in cylinders if s.get("height", 999) < 3.0]
    w['cap_count'] = (min(1.0, len(small_cyls) / max(1, n_cells)), 0.08)

    # ── Bottom cones (8%) — expect n_cells cones ─────────────────
    w['cone_count'] = (min(1.0, len(cones) / max(1, n_cells)), 0.08)

    # ── Shared wall beams (10%) — expect n_walls_expected beams ───
    # Wall beams are short (~10mm), ribs are long (> 20mm)
    short_beams = []
    long_beams = []
    for b in beams:
        st = b.get("start", [0,0,0])
        en = b.get("end", [0,0,0])
        length = math.sqrt(sum((st[i]-en[i])**2 for i in range(3)))
        if length < 15.0:
            short_beams.append(b)
        else:
            long_beams.append(b)
    w['wall_beams'] = (min(1.0, len(short_beams) / max(1, n_walls_expected)), 0.10)

    # ── Reinforcement ribs (7%) — expect 6 long radial beams ─────
    # Each rib should start near (0,0) and end far from centre
    rib_ok = 0
    for b in long_beams:
        st = b.get("start", [0,0,0])
        en = b.get("end", [0,0,0])
        d_start = math.sqrt(st[0]**2 + st[1]**2)
        d_end = math.sqrt(en[0]**2 + en[1]**2)
        # One endpoint near centre, other far
        near = min(d_start, d_end)
        far = max(d_start, d_end)
        if near < 5.0 and far > 15.0:
            rib_ok += 1
    w['rib_beams'] = (min(1.0, rib_ok / n_ribs), 0.07)

    # ── Base plate (3%) — expect 1 large cylinder at Z<0 ─────────
    base_cyls = [s for s in cylinders if s.get("radius", 0) > 15.0
                 and (_shape_center(s) or [0,0,99])[2] < 1.0]
    w['base_plate'] = (min(1.0, len(base_cyls)), 0.03)

    # ── Frame torus (3%) ─────────────────────────────────────────
    w['frame_ring'] = (min(1.0, len(tori)), 0.03)

    # ── Type variety (5%) — must have all 5 types ─────────────────
    type_set = set(s.get("type") for s in shapes)
    required = {"pipe", "cylinder", "torus", "cone", "beam"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.05)

    # ── Connectivity (3%) ────────────────────────────────────────
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=12.0), 0.03)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=8.0)


def _sem_radiolarian(shapes, golden, scale):
    """Armillary Sphere: concentric great-circle ring shells (torus wireframes) + icosahedral axis rods + girdle rings."""
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    tori      = [s for s in shapes if s.get("type") == "torus"]
    w = {}

    N_shells = scale
    n_spines = 12 * (N_shells - 1)
    # Ground truth: 3 shell tori per shell + 1 girdle torus per shell
    n_expected_tori = N_shells * 3 + N_shells  # shell lattice + girdles

    # Shell lattice tori — origin-centred tori (3 per shell × N_shells + N_shells girdles)
    origin_tori = [s for s in tori
                   if _shape_center(s) and
                   all(abs(v) < 2.0 for v in _shape_center(s))]
    # Count distinct ring_radius values to determine how many shells are represented
    shell_ring_radii = sorted(set(round(s.get("ring_radius", 0), 1) for s in origin_tori))
    w['shell_count'] = (min(1.0, len(shell_ring_radii) / max(1, N_shells)), 0.15)

    # Shell radius progression: should be r_base × 2^k (geometric ratio ≈ 2.0)
    if len(shell_ring_radii) >= 2:
        ratios = [shell_ring_radii[i+1] / shell_ring_radii[i]
                  for i in range(len(shell_ring_radii)-1) if shell_ring_radii[i] > 0.01]
        if ratios:
            ratio_err = sum(abs(r - 2.0) / 2.0 for r in ratios) / len(ratios)
            w['shell_progression'] = (max(0.0, 1.0 - ratio_err), 0.15)
        else:
            w['shell_progression'] = (0.0, 0.15)
    else:
        w['shell_progression'] = (0.0, 0.15)

    # Spine count
    w['spine_count'] = (min(1.0, len(cylinders) / max(1, n_spines)), 0.15)

    # Icosahedral symmetry: spine axes should point radially (axis ≈ normalized centre)
    radial_score = 0.0
    n_checked = 0
    for cyl in cylinders:
        c = _shape_center(cyl)
        axis = cyl.get("axis", [0, 0, 1])
        if c and any(abs(v) > 1.0 for v in c):
            c_len = math.sqrt(sum(v**2 for v in c))
            if c_len > 0.01:
                c_norm = [v / c_len for v in c]
                dot = abs(sum(a * b for a, b in zip(axis, c_norm)))
                radial_score += dot
                n_checked += 1
    if n_checked > 0:
        w['radial_alignment'] = (radial_score / n_checked, 0.15)
    else:
        w['radial_alignment'] = (0.0, 0.15)

    # Torus count (shell lattice tori + girdle tori)
    w['torus_count'] = (min(1.0, len(origin_tori) / max(1, n_expected_tori)), 0.10)

    # Concentricity — all tori at origin
    if tori:
        origin_count = sum(1 for s in tori
                           if _shape_center(s) and
                           all(abs(v) < 3.0 for v in _shape_center(s)))
        w['concentricity'] = (origin_count / len(tori), 0.10)
    else:
        w['concentricity'] = (0.0, 0.10)

    # Type variety — expect torus and cylinder (no solid spheres)
    type_set = set(s.get("type") for s in shapes)
    required = {"cylinder", "torus"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.10)

    # Connectivity
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=25.0), 0.10)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=25.0)


def _sem_vertebral(shapes, golden, scale):
    """Vertebral Column: linear articulated chain with physiological curvature."""
    boxes     = [s for s in shapes if s.get("type") == "box"]
    cones     = [s for s in shapes if s.get("type") == "cone"]
    beams     = [s for s in shapes if s.get("type") == "beam"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    w = {}

    N = scale
    n_discs = N - 1

    # Vertebral body count (boxes)
    w['body_count'] = (min(1.0, len(boxes) / max(1, N)), 0.10)

    # Spinous process count (cones)
    w['process_count'] = (min(1.0, len(cones) / max(1, N)), 0.08)

    # Transverse process count (beams) — expect 2 per vertebra
    w['trans_count'] = (min(1.0, len(beams) / max(1, 2 * N)), 0.08)

    # Disc count (cylinders, small height)
    disc_cyls = [s for s in cylinders if s.get("height", 999) < 10.0]
    w['disc_count'] = (min(1.0, len(disc_cyls) / max(1, n_discs)), 0.08)

    # Spinal canal (segmented pipe — N-1 segments following curvature)
    n_canal_expected = N - 1
    w['canal'] = (min(1.0, len(pipes) / max(1, n_canal_expected)), 0.06)

    # Curvature check: vertebral bodies should NOT be in a straight line
    box_centres = sorted([_shape_center(b) for b in boxes if _shape_center(b)],
                         key=lambda c: c[2])  # sort by Z (superior)
    if len(box_centres) >= 3:
        first = box_centres[0]
        last = box_centres[-1]
        chord_dy = last[1] - first[1]
        chord_dz = last[2] - first[2]
        chord_len = math.sqrt(chord_dy**2 + chord_dz**2)
        if chord_len > 0.01:
            max_dev = 0.0
            for c in box_centres:
                t = ((c[1] - first[1]) * chord_dy + (c[2] - first[2]) * chord_dz) / (chord_len**2)
                proj_y = first[1] + t * chord_dy
                proj_z = first[2] + t * chord_dz
                dev = math.sqrt((c[1] - proj_y)**2 + (c[2] - proj_z)**2)
                if dev > max_dev:
                    max_dev = dev
            w['curvature'] = (min(1.0, max_dev / 10.0), 0.15)
        else:
            w['curvature'] = (0.0, 0.15)
    else:
        w['curvature'] = (0.0, 0.15)

    # Regularity: spacing between consecutive bodies should be roughly uniform
    if len(box_centres) >= 2:
        spacings = []
        for i in range(len(box_centres) - 1):
            d = math.sqrt(sum((a - b)**2 for a, b in zip(box_centres[i], box_centres[i+1])))
            spacings.append(d)
        mean_sp = sum(spacings) / len(spacings)
        if mean_sp > 0.01:
            dev = sum(abs(s - mean_sp) / mean_sp for s in spacings) / len(spacings)
            w['spacing_regularity'] = (max(0.0, 1.0 - dev * 2.0), 0.10)
        else:
            w['spacing_regularity'] = (0.0, 0.10)
    else:
        w['spacing_regularity'] = (0.0, 0.10)

    # Bilateral symmetry: beams should be mirrored in X
    beam_centres = [_shape_center(b) for b in beams if _shape_center(b)]
    if beam_centres:
        x_vals = [c[0] for c in beam_centres]
        pos_x = [x for x in x_vals if x > 1.0]
        neg_x = [x for x in x_vals if x < -1.0]
        if pos_x and neg_x:
            symmetry = min(len(pos_x), len(neg_x)) / max(len(pos_x), len(neg_x))
            w['bilateral_symmetry'] = (symmetry, 0.10)
        else:
            w['bilateral_symmetry'] = (0.0, 0.10)
    else:
        w['bilateral_symmetry'] = (0.0, 0.10)

    # Type variety
    type_set = set(s.get("type") for s in shapes)
    required = {"box", "cone", "beam", "cylinder", "pipe"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.10)

    # Physics
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=30.0), 0.10)
    w['interference']  = (_phys_interference(shapes, tol_mm=2.0), 0.05)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=15.0)


# ── Semantic dispatcher ───────────────────────────────────────

_SEM_DISPATCH = {
    "Spiral Staircase":   lambda s, g, sc, sp, o: _sem_staircase(s, g, sc),
    "Cannonball Pyramid":  lambda s, g, sc, sp, o: _sem_pyramid(s, g, sc),
    "Voxel Grid":          lambda s, g, sc, sp, o: _sem_voxel(s, g, sc),
    "Domino Ring":         lambda s, g, sc, sp, o: _sem_stonehenge(s, g, sc),
    "DNA Helix":           lambda s, g, sc, sp, o: _sem_dna(s, g, sc),
    "Suspension Bridge":   lambda s, g, sc, sp, o: _sem_bridge(s, g, sc),
    "Planetary Array":     lambda s, g, sc, sp, o: _sem_planetary(s, g, sc),
    "Cross-Braced Truss":  lambda s, g, sc, sp, o: _sem_truss(s, g, sc),
    "Fractal Y-Tree":      lambda s, g, sc, sp, o: _sem_fractal(s, g, sc),
    "BCC Lattice":         lambda s, g, sc, sp, o: _sem_bcc(s, g, sc),
    "Furniture Assembly":  lambda s, g, sc, sp, o: _sem_furniture(s, o, sp),
    "Pipe Manifold":       lambda s, g, sc, sp, o: _sem_manifold(s, o, sp),
    "Axle Bearing":         lambda s, g, sc, sp, o: _sem_axle(s, o, sp),
    "Phyllotaxis Disc":    lambda s, g, sc, sp, o: _sem_phyllotaxis(s, g, sc),
    "Compound Eye":        lambda s, g, sc, sp, o: _sem_compound_eye(s, g, sc),
    "Diatom Frustule":     lambda s, g, sc, sp, o: _sem_diatom(s, g, sc),
    "Honeycomb Lattice":   lambda s, g, sc, sp, o: _sem_honeycomb(s, g, sc),
    "Armillary Sphere":    lambda s, g, sc, sp, o: _sem_radiolarian(s, g, sc if sc else 2),
    "Vertebral Column":    lambda s, g, sc, sp, o: _sem_vertebral(s, g, sc),
}

# ── Golden baseline cache for Sem normalization ──────────────
# The raw Sem score depends on validator limitations (e.g. bbox-based
# interference flags tangent contacts, gravity checker misses nested
# pockets).  To make scores interpretable across families we normalize:
#   Sem_normalized = raw_llm / raw_golden * 100
# so the golden reference always benchmarks at 100%.
_SEM_BASELINE = {}   # key: (family, scale) → raw float [0..1]

def _sem_raw(family, shapes, golden, scale, specs=None, ops=None):
    """Raw semantic score (0.0–1.0) before golden-baseline normalization."""
    fn = _SEM_DISPATCH.get(family)
    if not fn:
        print(f"  [WARN] No semantic evaluator for family '{family}' — skipping Sem score")
        return 0.0
    try:
        raw = fn(shapes, golden, scale, specs or {}, ops or [])
        return max(0.0, min(1.0, raw))
    except Exception as e:
        print(f"  [WARN] Semantic evaluator for '{family}' raised {type(e).__name__}: {e}")
        return 0.0

def _get_sem_baseline(family, golden, scale, specs=None):
    """Compute and cache the golden reference's raw Sem score."""
    key = (family, scale)
    if key not in _SEM_BASELINE:
        normed_golden = _normalize_shapes(golden)
        baseline = _sem_raw(family, normed_golden, golden, scale, specs)
        if baseline < 0.01:
            print(f"  [WARN] Golden baseline for '{family}' scale={scale} is {baseline:.4f} — "
                  f"setting to 1.0 (check evaluator). Golden has {len(normed_golden)} shapes.")
            baseline = 1.0
        _SEM_BASELINE[key] = baseline
    return _SEM_BASELINE[key]

def eval_sem(family, shapes, golden, scale, specs=None, ops=None, geom_score=0):
    """Unified Semantic score, normalized against golden baseline.
    Golden reference always → 100%.  LLM output scored proportionally.

    STRICT gating (v2):
      - Coverage gate: exponent raised to 1.5 (was 0.6) — missing shapes
        now devastate the semantic score instead of getting a free pass.
        5% coverage → ~1% sem, 30% → ~16% sem, 70% → ~59% sem.
      - Geometry quality gate: if geometry accuracy is below 20%, the output
        is spatially incoherent — semantic structural checks are meaningless.
        geom < 10%  → sem multiplied by 0.0 (complete garbage)
        geom 10-20% → linear ramp from 0.0 to 1.0
        geom >= 20% → no penalty from this gate
    Returns 0–100."""
    raw = _sem_raw(family, shapes, golden, scale, specs, ops)
    baseline = _get_sem_baseline(family, golden, scale, specs)
    normalized = raw / baseline

    # ── Coverage gate (STRICT) ────────────────────────────────
    n_actual = len(shapes)
    n_golden = len(_normalize_shapes(golden)) if golden else 1
    coverage_ratio = min(1.0, n_actual / max(1, n_golden))
    # Power > 1 means partial coverage gets crushed hard.
    coverage_gate = coverage_ratio ** 1.5

    # ── Geometry quality gate ─────────────────────────────────
    # If the geometry score is terrible, the model didn't produce a real
    # 3D assembly — semantic checks on random shapes are meaningless noise.
    geom_frac = geom_score / 100.0  # convert 0-100 → 0-1
    if geom_frac < 0.10:
        geom_gate = 0.0   # below 10% geom → semantic is zero
    elif geom_frac < 0.20:
        geom_gate = (geom_frac - 0.10) / 0.10  # linear ramp 10%-20%
    else:
        geom_gate = 1.0

    gated = normalized * coverage_gate * geom_gate
    return round(max(0.0, min(1.0, gated)) * 100)


def eval_global(cov, geom, sem):
    """Global composite score with structural validity gate.

    Weights: Cov 20%, Geom 30%, Sem 50%.

    STRICT validity gate (v2):
      If the geometry is below 15%, the model did NOT produce a recognizable 3D
      assembly.  In that case, the raw weighted score is multiplied by a harsh
      gate that drives the overall score close to 0.

      geom < 5%   → global × 0.0  (no real output)
      geom 5–15%  → linear ramp 0.0 → 1.0
      geom >= 15% → no penalty from validity gate

    This prevents models from getting 50–70% global scores by gaming coverage
    or receiving inflated semantic scores on spatially incoherent output.
    Returns 0–100."""
    raw_global = cov * 0.20 + geom * 0.30 + sem * 0.50

    # Structural validity gate — geometry is the hard proof of real output
    geom_frac = geom / 100.0
    if geom_frac < 0.05:
        validity = 0.0
    elif geom_frac < 0.15:
        validity = (geom_frac - 0.05) / 0.10
    else:
        validity = 1.0

    return round(raw_global * validity)
