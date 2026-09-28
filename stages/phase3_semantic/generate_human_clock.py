"""
Clock Tower Mechanism — Phase 3 Semantic
==========================================
A mechanical clock face: dial ring, equiangular hour markers, two hands
at distinct angles, central shaft with a mounting bushing, and a
counterweight cone at the tail of each hand to balance it on the shaft.

Physical layout:
  - The back plate is the clock face surface (Z=0 plane, front face)
  - The dial ring (torus) sits flush ON the back plate surface at Z=0
  - Hour markers also lie flat on the face at Z=0
  - The shaft protrudes FORWARD from the face (+Z direction)
  - Hands float just above the face (Z=+1 and Z=+2) on the shaft
  - No floating elements — everything is grounded to the face plane

WHY THIS IS HARD:
  1. CONTINUOUS ANGULAR COMPUTATION — Each of N markers sits at a unique
     angle, requiring cos(k·360°/N) and sin(k·360°/N) for k=0..N-1.
     Our data shows this is the primary bottleneck: Spiral Staircase
     (similar continuous-angle task) scores only 60.5% on average.
  2. MULTI-TYPE (5 types) — cylinder + pipe + torus + beam + cone must
     all appear simultaneously. The 2→3 type transition drops scores 34%.
  3. BEAM ORIENTATION DIVERSITY — Every hour marker is a beam at a unique
     radial angle. The minute and hour hands are at two independent angles.
  4. CONE ORIENTATION — Counterweight cones at the hand tails must point
     opposite to their hand. Cones are recognised only 54.4% of the time.

Components:
  • Back plate             — cylinder (flat disc, front face at Z=0)
  • Central shaft          — cylinder (protrudes forward from Z=0)
  • Mounting bushing       — pipe (concentric with shaft)
  • Dial ring              — torus (sits flush on face at Z=0)
  • N hour markers         — beams (radiating outward on the face, Z=0)
  • Minute hand            — beam (above face at Z=+1, pointing to 12)
  • Hour hand              — beam (above face at Z=+2, pointing to 10)
  • Minute counterweight   — cone (tail of minute hand, Z=+1)
  • Hour counterweight     — cone (tail of hour hand, Z=+2)

Difficulty (N = number of hour markers):
  L1 =  4 (12 / 3 / 6 / 9 positions)   →  4+2+1+1+1+2 = 11 shapes
  L2 = 12 (full clock face)              → 12+2+1+1+1+2 = 19 shapes
  L3 = 24 (half-hour + hour, 15° step)  → 24+2+1+1+1+2 = 31 shapes
"""

import math


def _reference_solution(scale):
    N = scale         # number of hour markers
    shapes = []
    sid = 0

    # ── Dimensions ────────────────────────────────────────────────────────
    face_z         = 0.0         # clock face plane — all face elements sit here
    shaft_r        = 4.0         # shaft radius
    shaft_h        = 18.0        # shaft height (protrudes forward in +Z)
    back_r         = 85.0        # back plate radius
    back_h         = 3.0         # back plate thickness (behind the face)
    bushing_inner  = 4.3         # 0.3 mm clearance around shaft
    bushing_outer  = 7.0
    bushing_h      = 14.0
    dial_R         = 78.0        # torus major radius
    dial_r         = 3.0         # torus minor radius
    mark_in        = 60.0        # inner tip of each hour marker (radius)
    mark_out       = 72.0        # outer tip (radius)
    mark_w         = 2.0         # marker beam width
    min_hand       = 62.0        # minute hand length from centre
    hour_hand      = 42.0        # hour hand length
    hand_w         = 3.0         # hand beam width
    cw_h           = 10.0        # counterweight cone height
    cw_base        = 4.0         # counterweight base radius
    cw_tip         = 0.5         # counterweight tip radius
    cw_offset      = 12.0        # counterweight tail offset from centre

    # ── Back plate — behind the face, front face flush at Z=0 ─────────────
    shapes.append({
        "id": sid, "type": "cylinder",
        "center": [0.0, 0.0, -(back_h / 2.0)],
        "radius": back_r, "height": back_h,
    }); sid += 1

    # ── Shaft — protrudes forward from the face plane ──────────────────────
    shapes.append({
        "id": sid, "type": "cylinder",
        "center": [0.0, 0.0, shaft_h / 2.0],
        "radius": shaft_r, "height": shaft_h,
    }); sid += 1

    # ── Mounting bushing (pipe, concentric with shaft) ─────────────────────
    shapes.append({
        "id": sid, "type": "pipe",
        "center": [0.0, 0.0, bushing_h / 2.0],
        "inner_radius": bushing_inner,
        "outer_radius": bushing_outer,
        "height": bushing_h,
    }); sid += 1

    # ── Dial ring (torus) — sits flush ON the back plate face at Z=0 ───────
    shapes.append({
        "id": sid, "type": "torus",
        "center": [0.0, 0.0, face_z],
        "major_radius": dial_R, "minor_radius": dial_r,
    }); sid += 1

    # ── Hour markers — beams lying on the face plane (Z=0) ────────────────
    step = 2.0 * math.pi / N
    for k in range(N):
        a = k * step
        ca, sa = math.cos(a), math.sin(a)
        shapes.append({
            "id": sid, "type": "beam",
            "start": [round(mark_in * ca, 2), round(mark_in * sa, 2), face_z],
            "end":   [round(mark_out * ca, 2), round(mark_out * sa, 2), face_z],
            "width": mark_w, "height": mark_w,
        }); sid += 1

    # ── Minute hand — floats just above face (Z=+1), points toward 12 o'clock
    hand1_z = face_z + 1.0
    shapes.append({
        "id": sid, "type": "beam",
        "start": [0.0, 0.0, hand1_z],
        "end":   [0.0, round(min_hand, 2), hand1_z],
        "width": hand_w, "height": 1.5,
    }); sid += 1

    # ── Hour hand — floats above minute hand (Z=+2), points toward 10 o'clock
    # 10 o'clock in standard polar (CCW from +X): 120°
    hand2_z = face_z + 2.0
    hour_rad = math.radians(120.0)
    hx = round(hour_hand * math.cos(hour_rad), 2)
    hy = round(hour_hand * math.sin(hour_rad), 2)
    shapes.append({
        "id": sid, "type": "beam",
        "start": [0.0, 0.0, hand2_z],
        "end":   [hx, hy, hand2_z],
        "width": hand_w, "height": 2.0,
    }); sid += 1

    # ── Minute counterweight — cone at tail of minute hand (−Y side) ───────
    shapes.append({
        "id": sid, "type": "cone",
        "center": [0.0, -(cw_offset / 2.0), hand1_z],
        "base_radius": cw_base, "top_radius": cw_tip, "height": cw_h,
        "axis": [0.0, -1.0, 0.0],
    }); sid += 1

    # ── Hour counterweight — cone at tail of hour hand (opposite direction) ─
    cw_cx = round(-(cw_offset / 2.0) * math.cos(hour_rad), 2)
    cw_cy = round(-(cw_offset / 2.0) * math.sin(hour_rad), 2)
    shapes.append({
        "id": sid, "type": "cone",
        "center": [cw_cx, cw_cy, hand2_z],
        "base_radius": cw_base, "top_radius": cw_tip, "height": cw_h,
        "axis": [round(-math.cos(hour_rad), 4), round(-math.sin(hour_rad), 4), 0.0],
    }); sid += 1

    return shapes


def generate_clock(scale):
    N = scale
    ref = _reference_solution(scale)
    total = len(ref)

    shaft_r       = 4.0
    shaft_h       = 18.0
    back_r        = 85.0
    back_h        = 3.0
    bushing_inner = 4.3
    bushing_outer = 7.0
    bushing_h     = 14.0
    dial_R        = 78.0
    dial_r        = 3.0
    mark_in       = 60.0
    mark_out      = 72.0
    mark_w        = 2.0
    min_hand      = 62.0
    hour_hand     = 42.0
    hand_w        = 3.0
    cw_offset     = 12.0
    cw_base       = 4.0
    cw_tip        = 0.5
    cw_h          = 10.0

    prompt = f"""Model a clock face with {scale} equally spaced
radial markers. The origin is the center of the front face, the face plane is
XY, positive Z is forward and positive Y is twelve o'clock. The cylinder back
plate has radius 85 and thickness 3, with its front face at the face plane.
The cylinder shaft has radius 4 and projects 18 forward from the face. A
concentric pipe bushing starts at the face, projects 14 forward, has outer
radius 7 and radial shaft clearance 0.3. A dial torus of ring radius 78 and
tube radius 3 has its central plane at the face plane.
Each marker is a radial beam of length 12 and square section 2, with its inner
tip at radius 60. Its centerline lies in the face plane. The first marker points
along positive X, followed counterclockwise as viewed from positive Z.
The minute hand has length 62, width 3, thickness 1.5; its centerline lies 1
forward of the face, starts at the shaft axis and points toward twelve o'clock.
The hour hand has length 42, width 3, thickness 2; its centerline lies 2 forward
and points toward eleven o'clock. Each hand has a coaxial counterweight cone
on its opposite radial side, base radius 4, tip radius 0.5 and length 10.
Its base-face center is 1 from the shaft axis in that opposite direction,
at the hand centerline elevation, and its tip points away from the shaft.
IDs: plate 0, shaft 1, bushing 2, dial 3, then markers, minute hand, hour hand,
minute counterweight, hour counterweight. Dial and markers are symbolic
embedded face components, not separate surface-mounted solids.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    specs = {
        "gravity_check": False,
        "interference_check": True,
        "mates": [{"type": "concentric", "ids": [0, 1, 2, 3]}],
        "clearance_fit": [
            {"shaft_id": 1, "hole_id": 2,
             "expected_clearance": bushing_inner - shaft_r, "tol": 0.1}
        ],
        "reference": ref
    }
    return prompt, specs


if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=12)
    args = parser.parse_args()
    prompt, specs = generate_clock(args.scale)
    ref = specs["reference"]
    from collections import Counter
    print(f"Clock — {args.scale} markers, {len(ref)} shapes")
    print(f"Types: {dict(Counter(s['type'] for s in ref))}")
    print(prompt)
