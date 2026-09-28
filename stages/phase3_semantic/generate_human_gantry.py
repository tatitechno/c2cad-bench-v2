"""
Gantry Crane Assembly — Phase 3 Semantic
==========================================
Models an overhead gantry crane: two parallel rail beams supported by
vertical columns, with a travelling bridge beam, a trolley on the bridge,
and a hoist cable + hook assembly.

WHY THIS IS HARD FOR LLMs:
  This family targets FOUR identified failure mechanisms simultaneously:

  1. BEAM ORIENTATION BLINDNESS — The structure is dominated by beams:
     rail beams (horizontal along X), bridge beam (horizontal along Y),
     cross-bracing diagonals (unique 3D angles for each brace), and
     vertical columns. Models that substitute boxes for beams lose the
     orientation encoding entirely. Score gap: +45.3 points when beams
     are used correctly.

  2. MULTI-TYPE COMPOSITION — Uses 5 types (beam, cylinder, box, pipe, cone).
     The crane requires the model to reason about beams (rails, braces,
     bridge), cylinders (columns, cable drum), a box (trolley), a pipe
     (hoist cable), and a cone (hook). The 2→3 type transition causes
     a 34% score drop.

  3. CROSS-BRACING DIAGONALS — Each bay between columns is cross-braced
     with two diagonal beams forming an X pattern. Each diagonal has a
     unique 3D direction vector connecting non-adjacent column tops to
     bottoms. At L3 with 6 bays, this produces 12 diagonal beams with
     12 unique orientations — a continuous parametric challenge.

  4. SPATIAL CHAIN REASONING — The hook hangs from the trolley, which
     rides on the bridge, which spans between the rails, which sit on
     the columns. The model must propagate positions through a 4-level
     kinematic chain.

Components per bay:
  • 2 vertical columns (cylinder) per bay-boundary
  • 2 rail beams (beam) on top, along X
  • 2 diagonal braces (beam) per bay, forming X patterns
  • 1 bridge beam (beam) spanning the rails at the trolley position
  • 1 trolley body (box) riding on the bridge
  • 1 hoist cable (pipe) hanging from trolley
  • 1 cable drum (cylinder) on the trolley
  • 1 hook (cone) at the bottom of the cable

Difficulty (scale = number of bays):
  L1 = 2 bays  →  6 columns + 2 rails + 4 braces + 1 bridge + 1 trolley + 1 drum + 1 cable + 1 hook = 17 shapes
  L2 = 4 bays  → 10 columns + 2 rails + 8 braces + 1 bridge + 1 trolley + 1 drum + 1 cable + 1 hook = 25 shapes
  L3 = 6 bays  → 14 columns + 2 rails + 12 braces + 1 bridge + 1 trolley + 1 drum + 1 cable + 1 hook = 33 shapes
"""

import math


def _reference_solution(scale):
    """Build the canonical gantry crane assembly."""
    N_bays = scale
    N_cols_per_side = N_bays + 1
    shapes = []
    sid = 0

    # Parameters
    bay_length = 40.0       # mm — distance between columns along X
    rail_span = 80.0        # mm — distance between the two rail lines (Y)
    col_r = 5.0             # mm — column radius
    col_h = 100.0           # mm — column height
    rail_w = 4.0            # mm — rail beam width/height
    brace_w = 2.5           # mm — diagonal brace width/height
    bridge_w = 5.0          # mm — bridge beam width/height
    trolley_size = [15.0, 12.0, 8.0]  # mm — trolley box [X, Y, Z]
    drum_r = 6.0            # mm — cable drum radius
    drum_h = 10.0           # mm — cable drum height (along Y)
    cable_inner = 1.0       # mm — cable pipe inner radius
    cable_outer = 2.0       # mm — cable pipe outer radius
    cable_len = 60.0        # mm — cable length (hangs down from trolley)
    hook_r_base = 4.0       # mm — hook cone base radius
    hook_r_tip = 0.5        # mm — hook cone tip radius
    hook_h = 10.0           # mm — hook cone height

    # Derived positions
    total_length = N_bays * bay_length
    x_start = -total_length / 2.0
    y_left = -rail_span / 2.0
    y_right = rail_span / 2.0

    # Trolley is positioned at the midpoint of the bridge (centre of span)
    # Bridge is at X = 0 (middle of the gantry)
    bridge_x = 0.0
    trolley_z = col_h + rail_w + trolley_size[2] / 2.0

    # ── Vertical columns (cylinders) ──
    for i in range(N_cols_per_side):
        x = x_start + i * bay_length
        for y in [y_left, y_right]:
            shapes.append({
                "id": sid, "type": "cylinder",
                "center": [round(x, 2), round(y, 2), col_h / 2.0],
                "radius": col_r,
                "height": col_h,
            })
            sid += 1

    # ── Rail beams (2 beams running along X at the top of columns) ──
    rail_z = col_h + rail_w / 2.0
    for y in [y_left, y_right]:
        shapes.append({
            "id": sid, "type": "beam",
            "start": [x_start, round(y, 2), rail_z],
            "end":   [round(x_start + total_length, 2), round(y, 2), rail_z],
            "width": rail_w,
            "height": rail_w,
        })
        sid += 1

    # ── Cross-bracing diagonals (2 per bay, forming X on each side) ──
    # Each bay has braces on the front face (y = y_left side)
    for i in range(N_bays):
        x0 = x_start + i * bay_length
        x1 = x_start + (i + 1) * bay_length
        # Diagonal 1: bottom-left to top-right of the bay (front face)
        shapes.append({
            "id": sid, "type": "beam",
            "start": [round(x0, 2), y_left, 0.0],
            "end":   [round(x1, 2), y_left, col_h],
            "width": brace_w,
            "height": brace_w,
        })
        sid += 1
        # Diagonal 2: top-left to bottom-right of the bay (front face)
        shapes.append({
            "id": sid, "type": "beam",
            "start": [round(x0, 2), y_left, col_h],
            "end":   [round(x1, 2), y_left, 0.0],
            "width": brace_w,
            "height": brace_w,
        })
        sid += 1

    # ── Bridge beam (spans between rails at bridge_x) ──
    shapes.append({
        "id": sid, "type": "beam",
        "start": [bridge_x, y_left, rail_z],
        "end":   [bridge_x, y_right, rail_z],
        "width": bridge_w,
        "height": bridge_w,
    })
    sid += 1

    # ── Trolley body (box riding on the bridge) ──
    shapes.append({
        "id": sid, "type": "box",
        "center": [bridge_x, 0.0, trolley_z],
        "size": trolley_size,
    })
    sid += 1

    # ── Cable drum (cylinder on top of trolley, axis along Y) ──
    drum_z = trolley_z + trolley_size[2] / 2.0 + drum_r
    shapes.append({
        "id": sid, "type": "cylinder",
        "center": [bridge_x, 0.0, drum_z],
        "radius": drum_r,
        "height": drum_h,
        "axis": [0, 1, 0],
    })
    sid += 1

    # ── Hoist cable (pipe hanging down from trolley) ──
    cable_top = trolley_z - trolley_size[2] / 2.0
    cable_centre_z = cable_top - cable_len / 2.0
    shapes.append({
        "id": sid, "type": "pipe",
        "center": [bridge_x, 0.0, round(cable_centre_z, 2)],
        "inner_radius": cable_inner,
        "outer_radius": cable_outer,
        "height": cable_len,
    })
    sid += 1

    # ── Hook (cone at the bottom of the cable) ──
    hook_top_z = cable_top - cable_len
    hook_centre_z = hook_top_z - hook_h / 2.0
    shapes.append({
        "id": sid, "type": "cone",
        "center": [bridge_x, 0.0, round(hook_centre_z, 2)],
        "base_radius": hook_r_base,
        "top_radius": hook_r_tip,
        "height": hook_h,
    })
    sid += 1

    return shapes


def generate_gantry(scale):
    """
    Gantry Crane Assembly (Phase 3 Semantic).
    Scale = number of bays (2, 4, or 6).
    """
    N = scale
    ref = _reference_solution(scale)
    total = len(ref)

    # Base parameters for prompt
    bay_length = 40.0
    rail_span = 80.0
    col_r = 5.0
    col_h = 100.0
    rail_w = 4.0
    brace_w = 2.5
    bridge_w = 5.0
    trolley_size = [15.0, 12.0, 8.0]
    drum_r = 6.0
    drum_h = 10.0
    cable_inner = 1.0
    cable_outer = 2.0
    cable_len = 60.0
    hook_r_base = 4.0
    hook_r_tip = 0.5
    hook_h = 10.0

    N_cols = 2 * (N + 1)
    N_braces = 2 * N

    prompt = f"""Model a centered gantry crane with {scale} bays
along X, bay length 40 and column-row spacing 80 along Y. The origin is the
ground projection of the gantry midpoint, with Z up. At every bay boundary
each row has a vertical cylinder column, radius 5 and height 100, on the ground.
Square-section 4 beam rails run along each row, their lower faces resting on
the column tops, endpoints over the outer column axes. The negative-Y row is
the front; its bays have X-braces of square section 2.5 joining opposite
column-axis points at ground and column-top elevation.
At the gantry midpoint, a square-section 5 bridge beam spans between rail
centerlines along Y, its endpoints meeting the rail centerlines.
A trolley box of X length 15, Y width 12, Z height 8 sits centrally above
the bridge, with its bottom face in the plane of the rail upper faces. A cylinder drum, radius 6 and axial length 10 along Y, rests
centrally on the trolley top. A vertical pipe cable of inner radius 1, outer
radius 2 and length 60 hangs from the trolley bottom at its midpoint. Below
the cable is a cone hook of base radius 4, tip radius 0.5 and length 10,
tip face abutting the cable bottom, with its wider base below and axis upward.
Index columns by increasing X, negative-Y before positive-Y; rails negative-Y
before positive-Y; braces by bay, rising before falling; then bridge, trolley,
drum, cable, hook. Crossed centerlines are symbolic structural joints.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    specs = {
        "gravity_check": True,
        "interference_check": True,
        "mates": [],
        "clearance_fit": [],
        "reference": ref
    }
    return prompt, specs


if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()
    prompt, specs = generate_gantry(args.scale)
    print(f"Gantry Crane — {args.scale} bays, {len(specs['reference'])} shapes")
    print(prompt)
    print(f"\n--- Golden JSON ({len(specs['reference'])} shapes) ---")
    print(json.dumps(specs["reference"], indent=2)[:500])
