"""
Flanged Pipe Joint — Phase 1 Basic (Engineering CAD)
=====================================================
A real engineering assembly: two flanged pipes bolted together.
This is one of the most common assemblies in mechanical/process
engineering CAD — every piping system has hundreds of these joints.

The assembly has:
  • 2 pipe bodies (pipes, concentric along the X-axis)
  • 2 flange rings (pipes — hollow so the bore is visible through)
  • N bolts on a bolt-circle (cylinders, equally spaced around the flange)
  • N nuts (tori, one per bolt, concentric with each bolt)

WHY THIS IS EXTREMELY HARD FOR LLMs:
  1. BOLT-CIRCLE ANGULAR COMPUTATION — N bolts equally spaced on a circle
     of radius R: each at (R·cos(k·2π/N), R·sin(k·2π/N)). This is the
     same continuous-angle computation that breaks Spiral Staircase (60.5%)
     but now in a horizontal plane. At L3 with 16 bolts, each at a unique
     angle at 22.5° spacing — LLMs must compute 16 cos/sin pairs.

  2. CONCENTRIC CONSTRAINT CHAINS — The bolts pass through BOTH flanges
     (their axis must align perfectly). The flanges and pipes must all
     share the same central axis. Multiple concentric chains.

  3. BOLT ORIENTATION — Each bolt cylinder must be oriented along the
     pipe axis (X-axis), NOT vertically. LLMs default to Z-axis cylinders.
     Getting the bolt axis wrong while keeping position right still fails.

  4. TORUS AS NUT — Each nut is a torus concentric with its bolt. The
     torus must sit at the correct X-position (outer face of flange).
     Tori are recognised 89.5% of the time, but placing N tori at N
     unique bolt-circle positions in 3D requires the same angle computation.

  5. PIPE vs CYLINDER CONFUSION — Pipes (hollow) and cylinders (solid)
     coexist. Flanges and pipe bodies are all pipes (hollow), bolts are
     cylinders (solid). Models that collapse pipe→cylinder (24.1%
     substitution rate) will fail semantically.

Components:
  • 2 pipe bodies (pipe, concentric along X)
  • 2 flange rings (pipe, concentric along X — hollow so the bore shows)
  • N bolts (cylinder, on bolt circle, oriented along X)
  • N nuts (torus, concentric with each bolt)

Difficulty (N = number of bolts):
  L1 =  4 bolts → 2+2+4+4 = 12 shapes
  L2 =  8 bolts → 2+2+8+8 = 20 shapes
  L3 = 16 bolts → 2+2+16+16 = 36 shapes
"""

import math


def _reference_solution(scale):
    N = scale   # number of bolts
    shapes = []
    sid = 0

    # ── Pipe dimensions ────────────────────────────────────────────────────
    pipe_inner   = 25.0     # mm — pipe bore radius
    pipe_outer   = 30.0     # mm — pipe wall outer radius
    pipe_length  = 60.0     # mm — each pipe segment length
    flange_inner = pipe_inner  # mm — flange bore = pipe bore (hollow through)
    flange_outer = 50.0     # mm — flange disc outer radius
    flange_h     = 8.0      # mm — flange disc thickness
    bolt_circle  = 40.0     # mm — bolt circle radius (centre-to-centre)
    bolt_r       = 3.0      # mm — bolt shaft radius
    bolt_length  = 20.0     # mm — bolt shaft length (spans both flanges)
    nut_major    = 3.0      # mm — nut torus major radius (= bolt_r)
    nut_minor    = 2.0      # mm — nut torus tube radius
    gap          = 2.0      # mm — gap between flanges

    # The joint sits at X=0. Left pipe extends in -X, right pipe in +X.
    flange_x_left  = -(gap / 2.0 + flange_h / 2.0)
    flange_x_right = +(gap / 2.0 + flange_h / 2.0)
    pipe_x_left    = flange_x_left - flange_h / 2.0 - pipe_length / 2.0
    pipe_x_right   = flange_x_right + flange_h / 2.0 + pipe_length / 2.0

    # ── Left pipe body ─────────────────────────────────────────────────────
    shapes.append({
        "id": sid, "type": "pipe",
        "center": [round(pipe_x_left, 2), 0.0, 0.0],
        "inner_radius": pipe_inner, "outer_radius": pipe_outer,
        "height": pipe_length, "axis": [1, 0, 0],
    }); sid += 1

    # ── Right pipe body ────────────────────────────────────────────────────
    shapes.append({
        "id": sid, "type": "pipe",
        "center": [round(pipe_x_right, 2), 0.0, 0.0],
        "inner_radius": pipe_inner, "outer_radius": pipe_outer,
        "height": pipe_length, "axis": [1, 0, 0],
    }); sid += 1

    # ── Left flange ring (pipe — hollow so bore is visible) ───────────────
    shapes.append({
        "id": sid, "type": "pipe",
        "center": [round(flange_x_left, 2), 0.0, 0.0],
        "inner_radius": flange_inner, "outer_radius": flange_outer,
        "height": flange_h, "axis": [1, 0, 0],
    }); sid += 1

    # ── Right flange ring (pipe — hollow so bore is visible) ──────────────
    shapes.append({
        "id": sid, "type": "pipe",
        "center": [round(flange_x_right, 2), 0.0, 0.0],
        "inner_radius": flange_inner, "outer_radius": flange_outer,
        "height": flange_h, "axis": [1, 0, 0],
    }); sid += 1

    # ── Bolts + Nuts on bolt circle ────────────────────────────────────────
    angle_step = 2.0 * math.pi / N
    bolt_cx = 0.0
    nut_x = flange_x_right + flange_h / 2.0

    for k in range(N):
        angle = k * angle_step
        by = round(bolt_circle * math.cos(angle), 3)
        bz = round(bolt_circle * math.sin(angle), 3)

        # Bolt (cylinder oriented along X-axis)
        shapes.append({
            "id": sid, "type": "cylinder",
            "center": [bolt_cx, by, bz],
            "radius": bolt_r, "height": bolt_length, "axis": [1, 0, 0],
        }); sid += 1

        # Nut (torus concentric with bolt, at right flange face)
        shapes.append({
            "id": sid, "type": "torus",
            "center": [round(nut_x, 2), by, bz],
            "major_radius": nut_major, "minor_radius": nut_minor,
        }); sid += 1

    return shapes


def generate_flange(scale):
    N = scale
    ref = _reference_solution(scale)
    total = len(ref)

    pipe_inner   = 25.0
    pipe_outer   = 30.0
    pipe_length  = 60.0
    flange_inner = pipe_inner
    flange_outer = 50.0
    flange_h     = 8.0
    bolt_circle  = 40.0
    bolt_r       = 3.0
    bolt_length  = 20.0
    nut_major    = 3.0
    nut_minor    = 2.0
    gap          = 2.0

    prompt = f"""Model a joint with {scale} bolts. The joint axis is
X and the origin is midway across the gap between the mating flanges. On each
side is a pipe body of bore radius 25, outer radius 30 and length 60, extending
away from its flange's outer face. Each flange is a pipe of outer radius 50,
thickness 8, with the same bore as the bodies. The facing flanges have an axial
gap of 2 and the bodies abut their respective outer flange faces.
Cylinder bolts have radius 3 and length 20, are centered across the joint gap,
parallel to X, and are equally spaced on a bolt circle of radius 40. The first
bolt lies on positive Y; index toward positive Z around X. Each bolt has a
torus nut with ring radius equal to the bolt radius and tube radius 2; its
central plane coincides with the positive-X flange's outer face and its axis
is parallel to X. Nuts and bolts are symbolic uncut parts.
IDs 0 and 1 are negative-X and positive-X bodies; IDs 2 and 3 are their
flanges. Then index alternating bolt and nut pairs in bolt order.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    mates = [{"type": "concentric", "ids": [0, 1, 2, 3]}]
    for k in range(N):
        bolt_id = 4 + 2 * k
        nut_id  = 4 + 2 * k + 1
        mates.append({"type": "concentric", "ids": [bolt_id, nut_id]})

    specs = {
        "gravity_check": False,
        "interference_check": True,
        "mates": mates,
        "clearance_fit": [],
        "reference": ref,
    }
    return prompt, specs


if __name__ == "__main__":
    import json, argparse
    from collections import Counter
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=8)
    args = parser.parse_args()
    prompt, specs = generate_flange(args.scale)
    ref = specs["reference"]
    print(f"Flanged Pipe Joint — {args.scale} bolts, {len(ref)} shapes")
    print(f"Types: {dict(Counter(s['type'] for s in ref))}")
    print(prompt)
