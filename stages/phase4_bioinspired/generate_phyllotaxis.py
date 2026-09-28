"""
Phyllotaxis Disc — Phase 4 Bio-Inspired (Family 1)
====================================================
Models a sunflower seed-head: N spheres arranged in a Fibonacci spiral
pattern using the golden angle (≈ 137.508°).  Each seed is placed at
polar coordinates  r(n) = c·√n ,  θ(n) = n · φ  where φ = 137.508°.

The LLM must derive every coordinate from the golden-angle rule — the
prompt gives only the mathematical principle, seed count, seed radius,
and the spacing constant *c*.  Errors are *immediately* visible: wrong
golden angle produces ugly clumps / spoke lines instead of the smooth
counter-rotating Fibonacci spirals the human eye expects.

Difficulty (scale = number of seeds):
  L1 = 21   (Fibonacci number — 1 parastichy visible)
  L2 = 55   (two clear parastichy families)
  L3 = 89   (three parastichy families, 89 seeds)
"""

import math

GOLDEN_ANGLE_DEG = 137.50776405003785   # 360 / φ²  where φ = (1+√5)/2

def generate_phyllotaxis(scale):
    """
    Phyllotaxis Disc (Difficulty Level 4 — Phase 4 Bio-Inspired).
    Scale = number of seeds (should be a Fibonacci number for clean parastichies).
    """
    N = scale
    seed_radius = 3.0               # mm — each seed sphere
    c = seed_radius * 2.2           # spacing constant  (ensures tangent-near contact)
    disc_base_z = seed_radius       # seeds rest on Z=0 ground (center at Z=radius)

    # ── Build golden array ───────────────────────────────────
    shapes = []

    # ID 0 = central receptacle disc (flat cylinder)
    max_r = c * math.sqrt(N)
    shapes.append({
        "id": 0,
        "type": "cylinder",
        "center": [0.0, 0.0, 0.0],
        "radius": round(max_r + seed_radius * 2, 2),
        "height": 1.0,
        "axis": [0, 0, 1]
    })

    for n in range(1, N + 1):
        theta = math.radians(n * GOLDEN_ANGLE_DEG)
        r     = c * math.sqrt(n)
        x     = round(r * math.cos(theta), 4)
        y     = round(r * math.sin(theta), 4)
        shapes.append({
            "id": n,
            "type": "sphere",
            "center": [x, y, disc_base_z],
            "radius": seed_radius
        })

    # ── Prompt (zero-scaffolding — Laws 1-5) ─────────────────
    prompt = f"""Model a golden-angle phyllotaxis disc with {scale}
spherical seeds of radius 3. Their planar pattern follows a Fermat spiral with
radial scale 6.6, using seed indices starting at 1. The zero-angle ray is
positive X; the first seed has advanced one golden angle from that ray,
and subsequent seeds progress counterclockwise when viewed from positive Z.
The receptacle is a cylinder of thickness 1 centered at the origin along Z.
Its radius extends two seed radii beyond the most distant seed's axis projection.
Seeds are embedded with their lowest points in the receptacle's midplane. ID 0 is the receptacle,
followed by seeds in spiral-index order.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    return prompt, shapes


if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=21)
    args = parser.parse_args()
    prompt, shapes = generate_phyllotaxis(args.scale)
    print(f"Phyllotaxis Disc — {args.scale} seeds, {len(shapes)} shapes")
    print(prompt)
    with open("phyllotaxis_golden.json", "w") as f:
        json.dump(shapes, f, indent=2)
