"""
Armillary Sphere — Phase 4 Bio-Inspired
=========================================
Models a nested armillary sphere: concentric celestial frameworks
made of great-circle rings at different orientations, connected by
radial axis rods at icosahedral symmetry points, with equatorial
girdle rings at each shell level.

An armillary sphere is an ancient astronomical instrument representing
celestial coordinate systems with nested rings (horizon, meridian,
equatorial, ecliptic circles).  Our model extends it to multiple
concentric shells, each a wireframe cage of 3 orthogonal great-circle
torus rings (XY, XZ, YZ planes), connected by 12 icosahedral spines.

The LLM must derive:
  • Icosahedral vertex unit vectors (12 vertices):
      (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)   normalised
      where φ = (1+√5)/2 ≈ 1.618
  • Shell radii in geometric progression: r_k = r_base × 2^k
  • Each shell = 3 torus rings (XY, XZ, YZ great circles)
  • Spine segment endpoints on each shell surface
  • Girdle torus placement at each shell equator

Errors are extremely visible:
  • Solid spheres instead of rings → opaque blob hiding inner structure
  • Non-icosahedral spine directions → asymmetric instrument
  • Wrong shell radii → shells too close or too far apart
  • Missing spines → incomplete axis rods
  • Wrong girdle placement → rings floating between shells

Difficulty (scale = number of concentric shells):
  L1 = 2 shells  →  2×3 shell tori + 12×1 spines + 2 girdle tori = ~20 shapes
  L2 = 3 shells  →  3×3 shell tori + 12×2 spines + 3 girdle tori = ~36 shapes
  L3 = 4 shells  →  4×3 shell tori + 12×3 spines + 4 girdle tori = ~52 shapes
"""

import math

PHI = (1 + math.sqrt(5)) / 2  # golden ratio ≈ 1.618

# 12 vertices of a regular icosahedron (normalised to unit sphere)
_RAW_VERTS = [
    ( 0,  1,  PHI), ( 0,  1, -PHI), ( 0, -1,  PHI), ( 0, -1, -PHI),
    ( 1,  PHI,  0), ( 1, -PHI,  0), (-1,  PHI,  0), (-1, -PHI,  0),
    ( PHI,  0,  1), ( PHI,  0, -1), (-PHI,  0,  1), (-PHI,  0, -1),
]
_NORM = math.sqrt(1 + PHI**2)
ICO_VERTICES = [(round(x/_NORM, 6), round(y/_NORM, 6), round(z/_NORM, 6))
                for x, y, z in _RAW_VERTS]


def generate_radiolarian(scale):
    """
    Armillary Sphere (Phase 4 Bio-Inspired).
    Scale = number of concentric shells (2, 3, or 4).
    """
    N_shells = scale
    r_base    = 20.0       # mm — innermost shell radius
    spine_r   = 1.2        # mm — spine cylinder radius
    girdle_r  = 1.5        # mm — girdle torus tube radius
    lattice_r = 1.0        # mm — shell lattice torus tube radius

    shapes = []
    sid = 0

    # Shell radii: r_k = r_base × 2^k  (k = 0, 1, ..., N-1)
    shell_radii = [round(r_base * (2 ** k), 4) for k in range(N_shells)]

    # ── Concentric shells (3 orthogonal torus rings per shell) ──
    # Each shell is represented by 3 great-circle torus rings:
    #   - Equatorial (XY plane): axis = [0, 0, 1]
    #   - Meridional 1 (XZ plane): axis = [0, 1, 0]
    #   - Meridional 2 (YZ plane): axis = [1, 0, 0]
    # This creates a wireframe cage that shows the spherical shape
    # while remaining transparent to reveal inner shells.
    shell_axes = [
        [0, 0, 1],  # XY plane (equatorial)
        [0, 1, 0],  # XZ plane (meridional)
        [1, 0, 0],  # YZ plane (meridional)
    ]
    for k, r in enumerate(shell_radii):
        for ax in shell_axes:
            shapes.append({
                "id": sid, "type": "torus",
                "center": [0.0, 0.0, 0.0],
                "ring_radius": r,
                "tube_radius": lattice_r,
                "axis": ax
            })
            sid += 1

    # ── Girdle rings (thicker torus at each shell equator) ────
    for k, r in enumerate(shell_radii):
        shapes.append({
            "id": sid, "type": "torus",
            "center": [0.0, 0.0, 0.0],
            "ring_radius": r,
            "tube_radius": girdle_r,
            "axis": [0, 0, 1]
        })
        sid += 1

    # ── Radial spines (cylinders between consecutive shells) ──
    # 12 icosahedral directions × (N_shells - 1) inter-shell segments
    for shell_idx in range(N_shells - 1):
        r_inner = shell_radii[shell_idx]
        r_outer = shell_radii[shell_idx + 1]
        seg_len = r_outer - r_inner

        for vi, (vx, vy, vz) in enumerate(ICO_VERTICES):
            # Spine segment from inner shell surface to outer shell surface
            cx = round(vx * (r_inner + r_outer) / 2, 4)
            cy = round(vy * (r_inner + r_outer) / 2, 4)
            cz = round(vz * (r_inner + r_outer) / 2, 4)

            shapes.append({
                "id": sid, "type": "cylinder",
                "center": [cx, cy, cz],
                "radius": spine_r,
                "height": round(seg_len, 4),
                "axis": [round(vx, 6), round(vy, 6), round(vz, 6)]
            })
            sid += 1

    n_shell_tori = N_shells * 3
    n_girdle_tori = N_shells
    n_spines_total = 12 * (N_shells - 1)
    total_shapes = n_shell_tori + n_girdle_tori + n_spines_total

    # ── Prompt ────────────────────────────────────────────────
    ico_str = "  (0, ±1, ±φ),  (±1, ±φ, 0),  (±φ, 0, ±1)   where φ=(1+√5)/2≈1.618"
    prompt = f"""Model {scale} concentric armillary shells centered at
the origin. The innermost radius is 20 and each successive shell doubles that
radius. Each shell has torus great-circle rings in the XY, XZ and YZ planes
with tube radius 1, and an additional equatorial torus girdle with tube radius
1.5 sharing the equatorial center circle. Between neighboring shell spheres,
cylinder rods of radius 1.2 follow the vertex directions of a regular
icosahedron and end at the shells' ideal spherical loci.
Center the regular icosahedron at the origin. Its vertices
are the corners of three mutually perpendicular golden rectangles centered at
the origin, lying in the YZ, XY, and XZ planes. Their long edges are parallel
to Z, Y, and X respectively. This fixes the orientation without a vertex list.
IDs: great-circle rings from inner shell outward, XY then XZ then YZ;
girdles from inner shell outward; then rods by consecutive shell pair.
Within a pair any consistent ordering of the icosahedral directions is allowed.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    specs = {"reference": shapes}
    return prompt, specs


if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    prompt, specs = generate_radiolarian(args.scale)
    shapes = specs["reference"]
    print(f"Armillary Sphere — {args.scale} shells, {len(shapes)} shapes")
    print(prompt)
    with open("radiolarian_golden.json", "w") as f:
        json.dump(shapes, f, indent=2)
