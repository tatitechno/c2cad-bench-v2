"""
Vertebral Column — Phase 4 Bio-Inspired
==========================================
Models a section of the human vertebral column with its characteristic
physiological double-S curvature and repeating vertebral units.

Each vertebra consists of:
  • Vertebral body: box
  • Intervertebral disc (between vertebrae): flat cylinder
  • Spinous process: cone projecting posteriorly (−Y)
  • 2 transverse processes: beams projecting laterally (±X)

A continuous spinal canal pipe threads through all vertebral bodies.

The LLM must derive:
  • Cumulative angular offsets from physiological curvature:
      Cervical (7 vert): lordosis ≈ 40° total → ~5.7° per level
      Thoracic (12 vert): kyphosis ≈ 40° total → ~3.3° per level
  • Vertebral body positions along the curved path
  • Disc placement between consecutive vertebrae
  • Process orientations matching each vertebra's local frame

Errors are extremely visible:
  • No curvature → straight stack instead of S-curve
  • Wrong curvature direction → lordosis where kyphosis should be
  • Uniform spacing ignoring disc height ratios
  • Missing processes → bare column
  • Processes not rotated with vertebral local frame

Difficulty (scale = number of vertebrae):
  L1 = 7  (cervical spine)     → 7×4 + 6 discs + 6 canal pipes = ~40 shapes
  L2 = 12 (thoracic spine)     → 12×4 + 11 discs + 11 canal pipes = ~70 shapes
  L3 = 19 (cervical + thoracic)→ 19×4 + 18 discs + 18 canal pipes = ~112 shapes
"""

import math


def generate_vertebral(scale):
    """
    Vertebral Column (Phase 4 Bio-Inspired).
    Scale = number of vertebrae (7, 12, or 19).
    """
    N = scale

    body_w = 25.0       # mm — vertebral body width (X)
    body_d = 18.0       # mm — vertebral body depth (Y, anterior-posterior)
    body_h = 20.0       # mm — vertebral body height (Z)
    disc_h = 6.0        # mm — intervertebral disc height (30% of body)
    disc_r = 12.0       # mm — disc radius

    spine_proc_h = 22.0 # mm — spinous process cone height
    spine_proc_r = 4.0  # mm — spinous process cone base radius
    trans_len    = 20.0 # mm — transverse process beam length
    trans_w      = 3.0  # mm — transverse process beam width/height

    canal_r_in   = 6.0  # mm — spinal canal inner radius
    canal_r_out  = 7.5  # mm — spinal canal outer radius

    # ── Curvature model ──────────────────────────────────────
    # Cervical: lordosis (anterior convex → curve bends forward in +Y)
    # Thoracic: kyphosis (posterior convex → curve bends backward in −Y)
    # The angle per vertebra is the tilt increment about the X-axis.
    # Positive angle = lordotic (forward tilt), negative = kyphotic (backward tilt)

    if N == 7:
        # Pure cervical: 40° lordosis over 7 vertebrae
        angles_per_vert = [40.0 / 7.0] * N   # ~5.71° each, lordotic
    elif N == 12:
        # Pure thoracic: 40° kyphosis over 12 vertebrae
        angles_per_vert = [-40.0 / 12.0] * N  # ~−3.33° each, kyphotic
    elif N == 19:
        # Cervical (7) + Thoracic (12)
        angles_per_vert = [40.0 / 7.0] * 7 + [-40.0 / 12.0] * 12
    else:
        # Generic: distribute evenly as kyphosis
        angles_per_vert = [-30.0 / N] * N

    shapes = []
    sid = 0
    unit_height = body_h + disc_h  # total height per vertebral unit

    # Compute cumulative positions and orientations along curved path
    # We walk along the spine: each vertebra tilts by its angle about X-axis
    # Path is in the Y-Z plane (Y = anterior, Z = superior)
    cum_angle = 0.0       # cumulative tilt angle in degrees
    positions = []        # centre of each vertebral body
    orientations = []     # cumulative angle at each level

    y_pos = 0.0
    z_pos = 0.0

    for i in range(N):
        cum_angle_rad = math.radians(cum_angle)

        # Direction vector along the spine at current angle
        # Spine runs primarily along Z, tilted by cum_angle in Y-Z plane
        dy = math.sin(cum_angle_rad)
        dz = math.cos(cum_angle_rad)

        if i == 0:
            # First vertebra at origin
            positions.append((0.0, 0.0, 0.0))
        else:
            # Step along the tilted direction by one unit height
            y_pos += dy * unit_height
            z_pos += dz * unit_height
            positions.append((0.0, round(y_pos, 4), round(z_pos, 4)))

        orientations.append(cum_angle)
        cum_angle += angles_per_vert[i]

    # ── Build shapes ─────────────────────────────────────────

    for i in range(N):
        px, py, pz = positions[i]
        angle_deg = orientations[i]
        angle_rad = math.radians(angle_deg)

        # Local coordinate frame: Z' along spine (tilted), Y' anterior
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 1. Vertebral body (box)
        shapes.append({
            "id": sid, "type": "box",
            "center": [px, round(py, 4), round(pz, 4)],
            "size": [body_w, body_d, body_h]
        })
        sid += 1

        # 2. Spinous process (cone) — projects posteriorly (−Y in local frame)
        # In global coords: posterior = −Y×cos(angle) component
        proc_cy = py - cos_a * (body_d / 2 + spine_proc_h / 2)
        proc_cz = pz + sin_a * (body_d / 2 + spine_proc_h / 2)
        shapes.append({
            "id": sid, "type": "cone",
            "center": [px, round(proc_cy, 4), round(proc_cz, 4)],
            "radius": spine_proc_r,
            "height": spine_proc_h,
            "axis": [0, round(-cos_a, 6), round(sin_a, 6)]
        })
        sid += 1

        # 3. Left transverse process (beam) — projects in +X
        shapes.append({
            "id": sid, "type": "beam",
            "start": [px, round(py, 4), round(pz, 4)],
            "end":   [round(px + trans_len, 4), round(py, 4), round(pz, 4)],
            "width": trans_w,
            "height": trans_w
        })
        sid += 1

        # 4. Right transverse process (beam) — projects in −X
        shapes.append({
            "id": sid, "type": "beam",
            "start": [px, round(py, 4), round(pz, 4)],
            "end":   [round(px - trans_len, 4), round(py, 4), round(pz, 4)],
            "width": trans_w,
            "height": trans_w
        })
        sid += 1

    # ── Intervertebral discs (flat cylinders between vertebrae) ──
    for i in range(N - 1):
        p1 = positions[i]
        p2 = positions[i + 1]
        mid_y = (p1[1] + p2[1]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        # Disc axis = direction from p1 to p2 (normalised)
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        dist = math.sqrt(dy**2 + dz**2)
        if dist < 0.001:
            ax_y, ax_z = 0.0, 1.0
        else:
            ax_y, ax_z = dy / dist, dz / dist

        shapes.append({
            "id": sid, "type": "cylinder",
            "center": [0.0, round(mid_y, 4), round(mid_z, 4)],
            "radius": disc_r,
            "height": disc_h,
            "axis": [0, round(ax_y, 6), round(ax_z, 6)]
        })
        sid += 1

    # ── Spinal canal (segmented pipe following the curvature) ──
    # The vertebral canal is formed by stacking vertebral foramina —
    # it follows the exact S-curvature, NOT a straight chord.
    # We place one pipe segment per consecutive vertebra pair,
    # each segment centred at the midpoint with axis along the
    # local spine direction (same as the disc axis).
    for i in range(N - 1):
        p1 = positions[i]
        p2 = positions[i + 1]
        mid_y = (p1[1] + p2[1]) / 2
        mid_z = (p1[2] + p2[2]) / 2
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        seg_len = math.sqrt(dy**2 + dz**2)
        if seg_len < 0.001:
            seg_len = unit_height
            ax_y, ax_z = 0.0, 1.0
        else:
            ax_y, ax_z = dy / seg_len, dz / seg_len

        shapes.append({
            "id": sid, "type": "pipe",
            "center": [0.0, round(mid_y, 4), round(mid_z, 4)],
            "inner_radius": canal_r_in,
            "outer_radius": canal_r_out,
            "height": round(seg_len, 4),
            "axis": [0, round(ax_y, 6), round(ax_z, 6)]
        })
        sid += 1

    n_bodies = N
    n_discs = N - 1
    n_canal_segs = N - 1
    total_shapes = N * 4 + n_discs + n_canal_segs  # 4 per vert + discs + canal segments

    # ── Prompt ────────────────────────────────────────────────
    if N == 7:
        segment_desc = "cervical spine (7 vertebrae, lordotic curvature of 40° total, ~5.71° per vertebra)"
    elif N == 12:
        segment_desc = "thoracic spine (12 vertebrae, kyphotic curvature of 40° total, ~3.33° per vertebra)"
    else:
        segment_desc = (f"cervical + thoracic spine (19 vertebrae: 7 cervical with 40° lordosis, "
                       f"then 12 thoracic with 40° kyphosis)")

    prompt = f"""Model a symbolic {'cervical' if N == 7 else 'thoracic' if N == 12 else 'cervical and thoracic'} vertebral column in the YZ
plane. The first vertebral-body center is the origin; Z is superior and Y
anterior. Bodies are axis-aligned boxes of X width 25, Y depth 18 and Z height
20. Discs have axial thickness 6 and radius 12. {'Cervical lordosis is 40 degrees distributed over 7 bodies, bending toward positive Y.' if N == 7 else 'Thoracic kyphosis is 40 degrees distributed over 12 bodies, bending toward negative Y.' if N == 12 else 'There are 7 cervical bodies with 40 degrees of lordosis toward positive Y, followed by 12 thoracic bodies with 40 degrees of kyphosis toward negative Y.'}
The first local superior direction is positive Z. Equal tilt increments
accumulate per body within each region. Regional totals are measured between
region boundaries, including the pitch after the region's last body; a body's
local frame uses its lower boundary orientation. At the region transition,
the orientation is continuous. The center-to-center pitch is
the body height plus disc thickness. Each superior body's center lies along
its own accumulated local superior direction from the preceding body's center.
Each body has a posterior pointed cone of base radius 4 and length 22; its
base center is on the posterior face of the ideal locally tilted body frame
and its axis follows that frame's posterior direction. Body boxes themselves
remain axis-aligned as envelope primitives. Each body also has transverse
beam processes of length 20 and square section 3 from its center along positive
and negative X. Disc centers are midway between consecutive body centers;
their axes follow the joining line. A pipe canal segment connects each
consecutive pair of body centers, bore radius 6 and outer radius 7.5.
IDs: body, posterior cone, positive-X process, negative-X process groups in
ascending body order; then discs, then canal segments in ascending order.
Discs and canal are symbolic embedded components, not a Boolean spinal solid.

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
    parser.add_argument("--scale", type=int, default=7)
    args = parser.parse_args()
    prompt, shapes = generate_vertebral(args.scale)
    print(f"Vertebral Column — {args.scale} vertebrae, {len(shapes)} shapes")
    print(prompt)
    with open("vertebral_golden.json", "w") as f:
        json.dump(shapes, f, indent=2)
