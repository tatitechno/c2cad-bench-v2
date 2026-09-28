"""
Honeycomb Lattice — Phase 4 Bio-Inspired
==========================================
Models a bee honeycomb: a hexagonal tiling of prismatic cells
arranged in concentric rings around a central cell, standing on
a base plate and enclosed by a frame ring.

Structural complexity (what makes this HARD):
  1. Hex ring algorithm — axial coordinates with pointy-top orientation
  2. Tilted cells — each cell axis is inclined 13° from vertical toward
     the comb centre (like real bee cells), requiring per-cell axis vectors
  3. Shared hex walls — beam segments connect each pair of adjacent cell
     centres, testing topological adjacency detection
  4. Reinforcement ribs — radial beams from centre to frame perimeter,
     one per ring-1 corner, testing angular placement
  5. Bottom caps — cone-shaped wax closures at the cell base,
     axis matching cell tilt, testing multi-type variety

The LLM must:
  • Compute hex axial coordinates and convert to Cartesian
  • Derive per-cell tilt axis (pointing inward toward centre)
  • Enumerate all unique adjacent pairs for wall beams
  • Place 6 evenly-spaced radial ribs
  • Keep base plate, frame torus, and all pieces connected

Difficulty (scale = number of concentric rings):
  L1 = 2 rings →  19 cells + walls + ribs + caps  = ~100 shapes
  L2 = 3 rings →  37 cells + walls + ribs + caps  = ~190 shapes
  L3 = 4 rings →  61 cells + walls + ribs + caps  = ~310 shapes
"""

import math


def _hex_ring_coords(ring):
    """Return list of (q, r) axial coordinates for a hex ring of given radius.

    Uses the standard ring-walk algorithm (Red Blob Games):
      - Start at cube coord (0, -ring, +ring) → axial (0, -ring)
      - Walk 6 edges, each edge has `ring` steps
      - Directions (axial): (+1,0), (0,+1), (-1,+1), (-1,0), (0,-1), (+1,-1)
    """
    if ring == 0:
        return [(0, 0)]
    coords = []
    directions = [
        (1, 0), (0, 1), (-1, 1),
        (-1, 0), (0, -1), (1, -1)
    ]
    q, r = 0, -ring
    for dq, dr in directions:
        for _ in range(ring):
            coords.append((q, r))
            q += dq
            r += dr
    return coords


def _hex_neighbours(q, r):
    """Return the 6 axial neighbours of hex cell (q, r)."""
    return [
        (q+1, r), (q-1, r), (q, r+1),
        (q, r-1), (q+1, r-1), (q-1, r+1)
    ]


def generate_honeycomb(scale):
    """
    Honeycomb Lattice (Phase 4 Bio-Inspired).
    Scale = number of concentric rings (2, 3, or 4).
    """
    N_rings = scale
    cell_outer_r = 5.0       # mm — cell outer radius (pipe outer)
    cell_inner_r = 4.2       # mm — cell inner radius (pipe inner)
    cell_height  = 12.0      # mm — cell depth (Z)
    base_thick   = 1.5       # mm — base plate thickness
    frame_thick  = 2.0       # mm — frame wall thickness
    cap_thick    = 0.8       # mm — cap cylinder lid thickness
    cone_height  = 3.0       # mm — bottom wax cone height
    tilt_deg     = 13.0      # degrees — cell tilt toward centre (real bee comb ≈ 9-13°)
    wall_width   = 1.0       # mm — wall beam width
    rib_width    = 1.5       # mm — reinforcement rib beam width
    n_ribs       = 6         # radial ribs (one per hex corner direction)

    tilt_rad = math.radians(tilt_deg)

    # Hex size: adjacent centres separated by 2 × cell_outer_r = 10mm.
    # For pointy-top: √3 × size = centre_distance → size = 10/√3
    hex_size = (2.0 * cell_outer_r) / math.sqrt(3)  # ≈ 5.7735 mm

    shapes = []
    sid = 0

    # Collect all cell centres and build coord→index map
    all_qr = []
    for ring in range(N_rings + 1):
        for q, r in _hex_ring_coords(ring):
            all_qr.append((q, r))
    qr_set = set(all_qr)

    def qr_to_xy(q, r):
        x = hex_size * math.sqrt(3) * (q + r / 2.0)
        y = hex_size * (3.0 / 2.0) * r
        return (round(x, 4), round(y, 4))

    cell_centres = [qr_to_xy(q, r) for q, r in all_qr]
    n_cells = len(cell_centres)

    # ── Compute per-cell tilt axis ────────────────────────────────
    # Each cell tilts 13° inward toward (0,0).  The tilt axis is
    # perpendicular to the radial direction in the XY plane.
    # For centre cell (0,0): no tilt (vertical).
    def cell_axis(cx, cy):
        dist = math.sqrt(cx*cx + cy*cy)
        if dist < 0.01:
            return [0.0, 0.0, 1.0]  # centre cell: vertical
        # Radial unit vector from centre to cell
        rx, ry = cx / dist, cy / dist
        # Tilt the Z-axis toward centre by tilt_deg
        az = math.cos(tilt_rad)
        a_radial = -math.sin(tilt_rad)  # negative = toward centre
        ax = a_radial * rx
        ay = a_radial * ry
        mag = math.sqrt(ax*ax + ay*ay + az*az)
        return [round(ax/mag, 6), round(ay/mag, 6), round(az/mag, 6)]

    cell_axes = [cell_axis(cx, cy) for cx, cy in cell_centres]

    # ── Max extent for base plate and frame ───────────────────────
    max_extent = (2.0 * cell_outer_r) * (N_rings + 0.5) + cell_outer_r

    # ── Base plate (flat cylinder underneath all cells) ───────────
    shapes.append({
        "id": sid, "type": "cylinder",
        "center": [0.0, 0.0, -base_thick / 2.0],
        "radius": round(max_extent, 2),
        "height": base_thick,
        "axis": [0, 0, 1]
    })
    sid += 1

    # ── Frame ring (torus around the perimeter) ──────────────────
    shapes.append({
        "id": sid, "type": "torus",
        "center": [0.0, 0.0, cell_height / 2.0],
        "ring_radius": round(max_extent, 2),
        "tube_radius": frame_thick,
        "axis": [0, 0, 1]
    })
    sid += 1

    # ── Honeycomb cells (pipes, tilted) ──────────────────────────
    for i, (cx, cy) in enumerate(cell_centres):
        ax = cell_axes[i]
        shapes.append({
            "id": sid, "type": "pipe",
            "center": [cx, cy, cell_height / 2.0],
            "inner_radius": cell_inner_r,
            "outer_radius": cell_outer_r,
            "height": cell_height,
            "axis": ax
        })
        sid += 1

    # ── Top caps (cylinder lids on top of each cell) ─────────────
    for i, (cx, cy) in enumerate(cell_centres):
        ax = cell_axes[i]
        shapes.append({
            "id": sid, "type": "cylinder",
            "center": [cx, cy, cell_height + cap_thick / 2.0],
            "radius": cell_inner_r,
            "height": cap_thick,
            "axis": ax
        })
        sid += 1

    # ── Bottom cones (wax closures at cell base) ─────────────────
    for i, (cx, cy) in enumerate(cell_centres):
        ax = cell_axes[i]
        shapes.append({
            "id": sid, "type": "cone",
            "center": [cx, cy, -base_thick - cone_height / 2.0],
            "base_radius": cell_inner_r,
            "top_radius": 0.0,
            "height": cone_height,
            "axis": ax
        })
        sid += 1

    # ── Shared hex walls (beams between adjacent cell centres) ────
    # Enumerate unique adjacent pairs: for each cell, check its 6
    # neighbours; only add beam if neighbour has higher index to
    # avoid duplicates.
    qr_to_idx = {qr: i for i, qr in enumerate(all_qr)}
    wall_pairs = []
    for i, (q, r) in enumerate(all_qr):
        for nq, nr in _hex_neighbours(q, r):
            if (nq, nr) in qr_set:
                j = qr_to_idx[(nq, nr)]
                if j > i:
                    wall_pairs.append((i, j))

    wall_z = cell_height / 2.0  # walls at mid-height
    for i, j in wall_pairs:
        cx1, cy1 = cell_centres[i]
        cx2, cy2 = cell_centres[j]
        shapes.append({
            "id": sid, "type": "beam",
            "start": [cx1, cy1, wall_z],
            "end": [cx2, cy2, wall_z],
            "width": wall_width
        })
        sid += 1

    n_walls = len(wall_pairs)

    # ── Reinforcement ribs (radial beams from centre to frame) ────
    # 6 radial ribs at 60° intervals
    rib_z = cell_height * 0.75  # placed at 3/4 height
    rib_endpoints = []
    for k in range(n_ribs):
        angle = k * (2.0 * math.pi / n_ribs)
        ex = round(max_extent * math.cos(angle), 4)
        ey = round(max_extent * math.sin(angle), 4)
        rib_endpoints.append((ex, ey))
        shapes.append({
            "id": sid, "type": "beam",
            "start": [0.0, 0.0, rib_z],
            "end": [ex, ey, rib_z],
            "width": rib_width
        })
        sid += 1

    total_shapes = len(shapes)
    # Breakdown: 1 base + 1 frame + n_cells pipes + n_cells caps
    #          + n_cells cones + n_walls wall beams + 6 rib beams

    # ── Prompt ────────────────────────────────────────────────────
    prompt = f"""Model a reinforced honeycomb with {scale} complete
hexagonal neighbor rings around a central cell on a triangular lattice.
Pipe cells have inner radius 4.2, outer radius 5 and axial height 12. The planar
separation of nearest cell axes equals a cell diameter. A lattice neighbor of
the central cell lies on positive X. The origin is the center of a horizontal
cylinder base plate's top face. The base thickness is 1.5; its radius exceeds
the largest planar cell-axis radius by one cell diameter. All pipe centroids
lie in the central vertical cell's mid-height plane. Noncentral axes tilt
13 degrees from upward Z toward the center; the center cell remains vertical.
Each cell has a cylinder cap of radius equal to its bore and thickness 0.8,
and a pointed cone of the same base radius and length 3 below the base plate.
Cap and cone centroids project vertically onto the pipe centroid's projection;
their axes are parallel to the pipe axis, with cones pointing upward.
Caps share the horizontal centroid plane of the cap that abuts the central
vertical pipe's upper face. Cones share the horizontal centroid plane of the
cone whose tip face abuts the base-plate underside beneath the central pipe.
Each nearest-neighbor pair of cell centers has a beam link of square section
1, with endpoints at those centroids; include each link once. A torus frame
at the pipe-centroid plane has ring radius equal to the base radius and tube
radius 2. There are 6 uniformly spaced radial beam ribs of square section 1.5,
from the central vertical cell axis to the frame's radial center circle at
three quarters of that cell's height. The first rib points along positive X.
The frame and links are symbolic reinforcing centerlines. IDs: base 0, frame 1,
then all pipes, all caps, all cones, links, ribs. Within each cell group start
at the center, then complete rings outward, starting each ring on the ray
240 degrees counterclockwise from positive X and progressing counterclockwise
along the hexagon perimeter. Link IDs follow increasing endpoint-ID pairs;
rib IDs follow counterclockwise order viewed from positive Z.

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
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    prompt, shapes = generate_honeycomb(args.scale)
    print(f"Honeycomb Lattice — {args.scale} ring(s), {len(shapes)} shapes")
    print(prompt)
    with open("honeycomb_golden.json", "w") as f:
        json.dump(shapes, f, indent=2)
