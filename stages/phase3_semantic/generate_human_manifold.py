"""
Pipe Manifold Assembly — Phase 3 Semantic (replaces Machined Enclosure)
========================================================================
A plumbing/hydraulic manifold: one main header pipe runs horizontally,
with N branch pipes tee-ing off at regular intervals.  Each branch has:
  • a flange (pipe) where it meets the header
  • a valve body (cylinder) at the midpoint
  • a mounting bracket (box) anchoring the header to a back wall

This tests:
  - Concentricity (branch pipes concentric with flanges)
  - Clearance fit (branch OD vs flange ID)
  - Gravity (brackets sit on Z=0, header at bracket top)
  - Connectivity (branches must touch header)
  - Interference (nothing overlaps)
  - Uniform spacing (branches evenly distributed along header)
  - Type variety (must use box, cylinder, pipe)

Difficulty (scale = number of branch pipes):
  L1 = 2  branches  →  11 shapes (header + wall + 2×(branch + flange + valve + bracket) + end-cap×2)
  L2 = 3  branches  →  15 shapes
  L3 = 5  branches  →  23 shapes

Errors are very visible:
  - Wrong spacing → uneven branch distribution along header
  - Wrong concentricity → flanges misaligned from branches
  - Missing brackets → header floats in space
  - Wrong clearance → flanges too tight or too loose on branches
"""

import math


def _reference_solution(scale):
    """Build a canonical valid manifold assembly."""
    N = scale               # number of branch pipes
    shapes = []
    sid = 0

    # ── Header pipe (horizontal along X axis) ──────────────────
    header_r_inner  = 12.0        # mm
    header_r_outer  = 15.0
    header_length   = 40.0 * (N + 1)   # room for N branches evenly spaced
    header_z        = 50.0        # header centre height
    spacing         = header_length / (N + 1)

    # ID 0: header pipe
    shapes.append({
        "id": sid, "type": "pipe",
        "center": [0.0, 0.0, header_z],
        "inner_radius": header_r_inner,
        "outer_radius": header_r_outer,
        "height": header_length,
        "axis": [1, 0, 0]
    })
    sid += 1

    # ID 1: back wall (the mounting surface)
    wall_thick = 5.0
    wall_w = header_length + 20.0
    wall_h = header_z + header_r_outer + 20.0
    shapes.append({
        "id": sid, "type": "box",
        "center": [0.0, -(header_r_outer + wall_thick / 2.0), wall_h / 2.0],
        "size": [wall_w, wall_thick, wall_h]
    })
    wall_id = sid
    sid += 1

    # ── End caps (two pipe caps closing the header) ────────────
    cap_thick = 3.0
    for sign in [-1, +1]:
        x_pos = sign * (header_length / 2.0 + cap_thick / 2.0)
        shapes.append({
            "id": sid, "type": "cylinder",
            "center": [x_pos, 0.0, header_z],
            "radius": header_r_outer,
            "height": cap_thick,
            "axis": [1, 0, 0]
        })
        sid += 1

    # ── Per-branch components ──────────────────────────────────
    branch_r_inner  = 6.0
    branch_r_outer  = 8.0
    branch_length   = 35.0
    flange_r_inner  = branch_r_outer + 0.3   # clearance gap
    flange_r_outer  = branch_r_outer + 5.0
    flange_thick    = 4.0
    valve_r         = branch_r_outer + 2.0
    valve_h         = 10.0
    bracket_w       = 20.0
    bracket_d       = 10.0
    bracket_h       = header_z - header_r_outer   # from ground to header bottom

    for i in range(N):
        # X position of this branch along header
        bx = -header_length / 2.0 + spacing * (i + 1)

        # Branch pipe — extends forward (+Y) from header
        branch_cy = header_r_outer + branch_length / 2.0
        shapes.append({
            "id": sid, "type": "pipe",
            "center": [bx, branch_cy, header_z],
            "inner_radius": branch_r_inner,
            "outer_radius": branch_r_outer,
            "height": branch_length,
            "axis": [0, 1, 0]
        })
        branch_id = sid
        sid += 1

        # Flange — sits at the junction (header surface → branch start)
        flange_cy = header_r_outer + flange_thick / 2.0
        shapes.append({
            "id": sid, "type": "pipe",
            "center": [bx, flange_cy, header_z],
            "inner_radius": flange_r_inner,
            "outer_radius": flange_r_outer,
            "height": flange_thick,
            "axis": [0, 1, 0]
        })
        flange_id = sid
        sid += 1

        # Valve body — cylinder midway along the branch
        valve_cy = header_r_outer + branch_length / 2.0
        shapes.append({
            "id": sid, "type": "cylinder",
            "center": [bx, valve_cy, header_z],
            "radius": valve_r,
            "height": valve_h,
            "axis": [0, 1, 0]
        })
        valve_id = sid
        sid += 1

        # Mounting bracket — box from ground to header underside
        shapes.append({
            "id": sid, "type": "box",
            "center": [bx, 0.0, bracket_h / 2.0],
            "size": [bracket_w, bracket_d, bracket_h]
        })
        bracket_id = sid
        sid += 1

    return shapes


def generate_manifold(scale):
    """
    Pipe Manifold Assembly (Phase 3 Semantic).
    Scale = number of branch pipes (2, 3, or 5).
    """
    N = scale
    ref = _reference_solution(scale)
    total = len(ref)

    # Extract key dimensions for the prompt
    header_r_inner = 12.0
    header_r_outer = 15.0
    header_length  = 40.0 * (N + 1)
    header_z       = 50.0
    spacing        = header_length / (N + 1)
    branch_r_outer = 8.0
    flange_clearance = 0.3

    # Build the ID mapping description
    # IDs: 0=header, 1=wall, 2-3=end-caps, then per branch: (branch, flange, valve, bracket)
    branch_start_id = 4

    id_desc = f"""You must use exactly these IDs:
- Header pipe: Pipe, ID=0 (horizontal along X-axis at Z={header_z}mm, length={header_length}mm)
- Back wall: Box, ID=1 (flat mounting surface behind the header)
- Left end-cap: Cylinder, ID=2
- Right end-cap: Cylinder, ID=3"""

    for i in range(N):
        base = branch_start_id + i * 4
        id_desc += f"""
- Branch {i+1} pipe: Pipe, ID={base} (extends forward from header in +Y direction)
- Branch {i+1} flange: Pipe, ID={base+1} (ring at junction between header and branch)
- Branch {i+1} valve: Cylinder, ID={base+2} (valve body concentric with branch, at branch midpoint)
- Branch {i+1} bracket: Box, ID={base+3} (support from Z=0 ground up to header underside)"""

    prompt = f"""Model a header manifold with {scale} forward branches.
The header is a pipe parallel to X, centered laterally above the origin with
its axis 50 above the XY ground plane. Its inner and outer radii are 12 and 15.
Branch junctions divide the header axis into equal intervals of length 40,
including the intervals between the end faces and nearest junctions.
The back wall is an axis-aligned box of thickness 5 standing on the ground,
its front face tangent to the rear of the header. Its side margins extend
10 beyond each header end and its top margin is 20 above the header's outer top.
End caps are cylinders of radius equal to the header outer radius and thickness
3, abutting and concentric with the header end faces.
Branches point along positive Y from the header's front tangent plane, at its
axis elevation. Each pipe has inner radius 6, outer radius 8, and length 35.
A coaxial flange starts at that junction plane, has thickness 4, radial bore
clearance 0.3 around the branch; its outer surface extends radially 5 beyond
the branch outer surface. A coaxial
cylinder valve body is centered axially on each branch, radius 10 and length 10.
A box bracket under each junction has X width 20 and Y depth 10; it stands
on the ground, centered below the header axis, and reaches the header underside.
The valve bodies are symbolic embedded solids rather than bored valves.
IDs: header 0, wall 1, negative-X cap 2, positive-X cap 3; then groups of branch,
flange, valve, bracket ordered by increasing X.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    # Build specs for semantic grading
    mates = []
    clearance_fits = []
    for i in range(N):
        base = branch_start_id + i * 4
        branch_id = base
        flange_id = base + 1
        valve_id  = base + 2
        # Branch concentric with flange
        mates.append({"type": "concentric", "ids": [branch_id, flange_id]})
        # Branch concentric with valve
        mates.append({"type": "concentric", "ids": [branch_id, valve_id]})
        # Flange clearance around branch
        clearance_fits.append({
            "shaft_id": branch_id, "hole_id": flange_id,
            "expected_clearance": flange_clearance, "tol": 0.1
        })
    # End-caps concentric with header
    mates.append({"type": "concentric", "ids": [0, 2]})
    mates.append({"type": "concentric", "ids": [0, 3]})

    specs = {
        "gravity_check": True,
        "interference_check": True,
        "mates": mates,
        "clearance_fit": clearance_fits,
        "reference": ref
    }
    return prompt, specs


if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()
    prompt, specs = generate_manifold(args.scale)
    print(f"Pipe Manifold — {args.scale} branches, {len(specs['reference'])} shapes")
    print(prompt)
    with open("manifold_golden.json", "w") as f:
        json.dump(specs["reference"], f, indent=2)
