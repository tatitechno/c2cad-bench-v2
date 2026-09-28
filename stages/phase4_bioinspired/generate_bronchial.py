"""
Bronchial Tree — Phase 4 Bio-Inspired (Family 2)
==================================================
Models a bronchial airway tree governed by Murray's Law:
    r_parent³ = r_child1³ + r_child2³
For symmetric bifurcation (equal children):
    r_child = r_parent / 2^(1/3)  ≈  r_parent × 0.7937

Each generation:
  • splits into 2 child branches
  • child radius obeys Murray's Law
  • child length = parent length × 0.7 (lung morphometry)
  • bifurcation half-angle = 25° from parent axis (in the branching plane)
  • alternating branching planes (rotate 90° each generation)

The prompt tells the LLM the starting trunk parameters, Murray's Law,
the length ratio, and the bifurcation angle — but NOT the derived
coordinates.  The LLM must recursively compute every branch endpoint.

Errors are extremely visible:
  • Wrong Murray's ratio → unnatural thick/thin transitions
  • Wrong bifurcation angle → overlapping or splayed branches
  • Missing branches → asymmetric tree
  • Wrong length ratio → branches that are too long/short

Difficulty (scale = recursion depth):
  L1 = 3   →  1 + 2 + 4 + 8  = 15 branches
  L2 = 4   →  + 16            = 31 branches
  L3 = 5   →  + 32            = 63 branches
"""

import math

MURRAY_FACTOR = 1.0 / (2.0 ** (1.0 / 3.0))   # ≈ 0.7937
LENGTH_RATIO  = 0.7
HALF_ANGLE    = 25.0   # degrees

def generate_bronchial(scale):
    """
    Bronchial Tree (Difficulty Level 4 — Phase 4 Bio-Inspired).
    Scale = recursion depth (3, 4, or 5).
    """
    depth = scale
    trunk_radius = 8.0   # mm
    trunk_length = 80.0  # mm

    shapes = []
    sid = [0]  # mutable counter

    def _recurse(start, direction, length, radius, gen, branch_plane_normal):
        """
        start:     [x, y, z] — branch base point
        direction: [dx, dy, dz] unit vector — growth direction
        length:    mm
        radius:    mm
        gen:       current generation (0 = trunk)
        branch_plane_normal: unit vector perpendicular to the branching plane
        """
        # This branch: cylinder along direction
        end = [
            round(start[0] + direction[0] * length, 4),
            round(start[1] + direction[1] * length, 4),
            round(start[2] + direction[2] * length, 4),
        ]
        center = [
            round((start[0] + end[0]) / 2, 4),
            round((start[1] + end[1]) / 2, 4),
            round((start[2] + end[2]) / 2, 4),
        ]
        shapes.append({
            "id": sid[0],
            "type": "cylinder",
            "center": center,
            "radius": round(radius, 4),
            "height": round(length, 4),
            "axis": [round(direction[0], 6), round(direction[1], 6), round(direction[2], 6)]
        })
        sid[0] += 1

        if gen >= depth:
            return

        # ── Bifurcate ────────────────────────────────────────
        child_radius = round(radius * MURRAY_FACTOR, 4)
        child_length = round(length * LENGTH_RATIO, 4)

        angle_rad = math.radians(HALF_ANGLE)

        # Rodrigues rotation: rotate direction by ±HALF_ANGLE around branch_plane_normal
        d = direction
        n = branch_plane_normal

        def _rotate(vec, axis, theta):
            """Rodrigues rotation formula."""
            ct = math.cos(theta)
            st = math.sin(theta)
            dot = sum(vec[i] * axis[i] for i in range(3))
            cross = [
                axis[1]*vec[2] - axis[2]*vec[1],
                axis[2]*vec[0] - axis[0]*vec[2],
                axis[0]*vec[1] - axis[1]*vec[0],
            ]
            return [
                round(vec[i]*ct + cross[i]*st + axis[i]*dot*(1-ct), 8)
                for i in range(3)
            ]

        def _normalize(v):
            mag = math.sqrt(sum(x*x for x in v))
            return [x/mag for x in v] if mag > 1e-12 else v

        dir_left  = _normalize(_rotate(d, n, +angle_rad))
        dir_right = _normalize(_rotate(d, n, -angle_rad))

        # Next generation's branching plane is rotated 90° around the parent direction
        # This gives the realistic 3D spread of bronchial trees
        next_plane_normal = _normalize([
            d[1]*n[2] - d[2]*n[1],
            d[2]*n[0] - d[0]*n[2],
            d[0]*n[1] - d[1]*n[0],
        ])
        # Fallback if cross product is degenerate
        if all(abs(x) < 1e-10 for x in next_plane_normal):
            next_plane_normal = n

        _recurse(end, dir_left,  child_length, child_radius, gen + 1, next_plane_normal)
        _recurse(end, dir_right, child_length, child_radius, gen + 1, next_plane_normal)

    # Trunk grows upward along Z
    _recurse(
        start=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        length=trunk_length,
        radius=trunk_radius,
        gen=0,
        branch_plane_normal=[1.0, 0.0, 0.0]  # first bifurcation in XZ plane
    )

    total_branches = 2**(depth + 1) - 1

    # ── Prompt (zero-scaffolding — Laws 1-5) ─────────────────
    prompt = f"""Model a bronchial airway tree obeying Murray's Law for optimal fluid transport.

The tree is a hierarchy of cylinders that bifurcate recursively up to generation {depth} (trunk = generation 0).

Trunk (generation 0):
  • Cylinder, radius {trunk_radius}mm, length {trunk_length}mm.
  • Base at the origin [0, 0, 0], growing upward along the +Z axis.

Bifurcation rules (applied at every branch tip, up to generation {depth}):
  1. Each parent branch spawns exactly 2 child branches at its tip.
  2. Murray's Law — each child's radius = parent radius × {MURRAY_FACTOR:.4f}  (i.e. r_parent³ = 2 × r_child³).
  3. Each child's length = parent length × {LENGTH_RATIO}.
  4. The two children diverge symmetrically at ±{HALF_ANGLE}° from the parent's directional axis, within a branching plane.
  5. The branching plane alternates by 90° around the parent axis at each new generation (generation 0 branches in the XZ plane, generation 1 in the YZ plane, etc.).

Total expected cylinders: {total_branches}.
Every child branch must start at the exact endpoint of its parent — no gaps.

Output only the raw JSON array of geometric elements — no markdown, no explanation."""

    return prompt, shapes


if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()
    prompt, shapes = generate_bronchial(args.scale)
    print(f"Bronchial Tree — depth {args.scale}, {len(shapes)} branches")
    print(prompt)
    with open("bronchial_golden.json", "w") as f:
        json.dump(shapes, f, indent=2)
