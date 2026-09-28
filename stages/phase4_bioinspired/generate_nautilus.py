"""
Nautilus Shell — Phase 4 Bio-Inspired (Family 3)
==================================================
Models a cross-section of a Nautilus shell: a logarithmic spiral of
growing chambers separated by septa (walls).

Each chamber is a box whose dimensions grow by the golden ratio φ ≈ 1.618
per quarter-turn (90°) of the spiral.  Septa are thin beams separating
consecutive chambers.  A central siphuncle (pipe) threads through all
chambers along the spiral axis.

The LLM must derive:
  • Chamber positions along the logarithmic spiral
  • Chamber sizes growing by φ each step
  • Septum placements between consecutive chambers
  • Siphuncle path threading through chamber centres

Errors are extremely visible:
  • Wrong growth ratio → spiral that collapses inward or explodes outward
  • Wrong placement angle → chambers overlap or leave gaps
  • Missing septa → open chambers (no internal walls)
  • Wrong siphuncle → disconnected tube or wrong path

Difficulty (scale = number of chambers):
  L1 = 6    chambers (1.5 full turns)
  L2 = 9    chambers (2.25 turns)
  L3 = 12   chambers (3 full turns)
"""

import math

PHI = (1 + math.sqrt(5)) / 2   # ≈ 1.618 — golden ratio
ANGLE_PER_CHAMBER = 90.0       # degrees — quarter-turn per chamber
# Growth factor per chamber: φ^(1/1) but we want per-quarter-turn scaling
# Classic nautilus: size grows by φ per full turn → per quarter: φ^(1/4)
GROWTH_PER_STEP = PHI ** 0.25  # ≈ 1.1277 per 90°

def generate_nautilus(scale):
    """
    Nautilus Shell (Difficulty Level 4 — Phase 4 Bio-Inspired).
    Scale = number of chambers (6, 9, or 12).
    """
    N = scale
    base_size   = 8.0     # mm — first chamber box side length
    wall_thick  = 1.0     # mm — septum (beam) thickness
    siphuncle_r = 1.5     # mm — siphuncle tube inner/outer radius
    spiral_a    = 12.0    # mm — initial spiral radius (distance from origin to first chamber centre)

    shapes = []
    sid = 0

    chamber_centres = []
    chamber_sizes   = []

    for i in range(N):
        angle_deg = i * ANGLE_PER_CHAMBER
        angle_rad = math.radians(angle_deg)

        # Logarithmic spiral: r(θ) = a · growth^i
        r_i     = spiral_a * (GROWTH_PER_STEP ** i)
        size_i  = base_size * (GROWTH_PER_STEP ** i)

        cx = round(r_i * math.cos(angle_rad), 4)
        cy = round(r_i * math.sin(angle_rad), 4)
        cz = 0.0   # planar shell cross-section

        chamber_centres.append([cx, cy, cz])
        chamber_sizes.append(round(size_i, 4))

        # Chamber box
        shapes.append({
            "id": sid,
            "type": "box",
            "center": [cx, cy, cz],
            "size": [round(size_i, 4), round(size_i, 4), round(size_i * 0.6, 4)]
        })
        sid += 1

    # Septa (thin beams between consecutive chambers)
    for i in range(N - 1):
        c1 = chamber_centres[i]
        c2 = chamber_centres[i + 1]
        shapes.append({
            "id": sid,
            "type": "beam",
            "start": [round(c1[0], 4), round(c1[1], 4), round(c1[2], 4)],
            "end":   [round(c2[0], 4), round(c2[1], 4), round(c2[2], 4)],
            "width": wall_thick,
            "height": wall_thick
        })
        sid += 1

    # Siphuncle — thin tube (pipe) from first to last chamber centre
    siphuncle_center = [
        round((chamber_centres[0][0] + chamber_centres[-1][0]) / 2, 4),
        round((chamber_centres[0][1] + chamber_centres[-1][1]) / 2, 4),
        0.0
    ]
    # Approximate siphuncle as a series of small cylinder segments
    for i in range(N - 1):
        c1 = chamber_centres[i]
        c2 = chamber_centres[i + 1]
        seg_center = [round((c1[0]+c2[0])/2, 4), round((c1[1]+c2[1])/2, 4), 0.0]
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        seg_len = math.sqrt(dx*dx + dy*dy)
        axis = [round(dx/max(seg_len, 1e-6), 6), round(dy/max(seg_len, 1e-6), 6), 0.0]
        shapes.append({
            "id": sid,
            "type": "pipe",
            "center": seg_center,
            "inner_radius": siphuncle_r,
            "outer_radius": round(siphuncle_r + 0.5, 2),
            "height": round(seg_len, 4),
            "axis": axis
        })
        sid += 1

    total_shapes = N + (N - 1) + (N - 1)  # chambers + septa + siphuncle segments

    # ── Prompt (zero-scaffolding — Laws 1-5) ─────────────────
    prompt = f"""Model a Nautilus shell cross-section as a logarithmic spiral of {N} growing chambers.

The shell lies in the XY plane, centred at the origin.

Chamber placement — logarithmic spiral:
  For chamber i (i = 0, 1, …, {N - 1}):
    angle(i)  = i × {ANGLE_PER_CHAMBER}°     (quarter-turn per chamber)
    radius(i) = {spiral_a} × {GROWTH_PER_STEP:.4f}^i   (logarithmic spiral)
    x(i) = radius(i) × cos(angle(i))
    y(i) = radius(i) × sin(angle(i))
    z(i) = 0
  Each chamber is a box at [x(i), y(i), 0].
  Chamber size grows with the spiral: side(i) = {base_size} × {GROWTH_PER_STEP:.4f}^i  (box is [side, side, side×0.6]).

Septa — thin walls between consecutive chambers:
  For each pair of consecutive chambers (i, i+1), place a beam (width={wall_thick}mm, height={wall_thick}mm) whose start is at chamber i's centre and whose end is at chamber (i+1)'s centre.
  Total septa: {N - 1}.

Siphuncle — continuous tube threading through all chambers:
  For each consecutive pair (i, i+1), place a pipe segment centred at the midpoint of the two chamber centres, axis pointing from chamber i to chamber i+1.
  Inner radius = {siphuncle_r}mm, outer radius = {siphuncle_r + 0.5}mm, height = distance between the two chamber centres.
  Total siphuncle segments: {N - 1}.

Total shapes: {total_shapes} ({N} boxes + {N - 1} beams + {N - 1} pipes).

Output only the raw JSON array of geometric elements — no markdown, no explanation."""

    return prompt, shapes


if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=6)
    args = parser.parse_args()
    prompt, shapes = generate_nautilus(args.scale)
    print(f"Nautilus Shell — {args.scale} chambers, {len(shapes)} shapes")
    print(prompt)
    with open("nautilus_golden.json", "w") as f:
        json.dump(shapes, f, indent=2)
