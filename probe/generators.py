"""
Stress test generators G1-G11 — "Encrypted" procedural tests.
The probe generates random specs -> computes exact ground truth
-> asks the LLM to express it -> validates the JSON output.
"""

import math, random
from typing import Tuple, List
from .config import _cyl, _box, _sph


# ═══════════════════════════════════════════════════════════════
# GENERATOR REGISTRY
# ═══════════════════════════════════════════════════════════════

GENERATOR_NAMES = [
    "Uniform Grid",
    "Radial Array",
    "Stacked Tower",
    "Mirror Assembly",
    "Alternating Lattice",
    "Supported Pyramid",
    "Connected Bridge",
    "Proportioned Tower",
    "Cantilever Assembly",
    "Aerodynamic Hull",
    "Oriented Pipes",
]


def _wave_params(wave: int) -> Tuple[int, float]:
    """
    Escalation schedule:
      Wave  1 ->   4 parts, +/-20mm
      Wave  5 ->  20 parts, +/-60mm
      Wave 10 ->  70 parts, +/-110mm
      Wave 15 -> 170 parts, +/-160mm
      Wave 20 -> 370 parts, +/-210mm
    """
    n = 2 + wave + max(0, (wave - 5)) * 3 + max(0, (wave - 10)) * 8
    coord_range = 20.0 + wave * 10.0
    return n, coord_range


# ── G1: UNIFORM GRID ──
def gen_uniform_grid(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)
    n, cr = _wave_params(wave)
    side = max(2, round(n ** (1/3)))
    nx = side; ny = side; nz = max(1, n // (side * side))
    total = nx * ny * nz
    spacing = round(rng.uniform(15, 35), 1)
    shape_type = rng.choice(["sphere", "cylinder", "box"])
    r = round(rng.uniform(3, spacing * 0.35), 2)
    h = round(rng.uniform(r * 1.5, r * 3), 2) if shape_type == "cylinder" else None

    gt = []
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cx = round((ix - (nx-1)/2) * spacing, 3)
                cy = round((iy - (ny-1)/2) * spacing, 3)
                cz = round((iz - (nz-1)/2) * spacing, 3)
                if shape_type == "sphere":
                    gt.append(_sph(idx, cx, cy, cz, r))
                elif shape_type == "cylinder":
                    gt.append(_cyl(idx, cx, cy, cz, r, h))
                else:
                    gt.append(_box(idx, cx, cy, cz, r*2, r*2, r*2))
                idx += 1

    prompt = (
        f"Generate a {nx}x{ny}x{nz} uniform 3D grid of {shape_type}s.\n"
        f"  - Grid spacing: {spacing}mm in all three directions\n"
        f"  - Grid centered at origin (0, 0, 0)\n"
        f"  - {'Radius: ' + str(r) + 'mm' if shape_type != 'box' else 'Side: ' + str(r*2) + 'mm'}"
        + (f', Height: {h}mm' if shape_type == 'cylinder' else '') + '\n'
        f"  - Total parts: {total}\n"
        f"Output all {total} shapes in the JSON array."
    )
    meta = {"n": total, "coord_range": cr, "grid": (nx, ny, nz),
            "spacing": spacing, "shape_type": shape_type}
    return prompt, gt, meta


# ── G2: RADIAL ARRAY ──
def gen_radial_array(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)
    n, cr = _wave_params(wave)
    n_radial = min(n, 4 + wave * 3)

    radius = round(rng.uniform(20, cr * 0.5), 1)
    cyl_r = round(rng.uniform(3, radius * 0.2), 2)
    cyl_h = round(rng.uniform(cyl_r * 2, cyl_r * 6), 2)
    z_center = round(rng.uniform(-20, 20), 1)
    has_center = rng.random() > 0.4

    gt = []
    if has_center:
        central_r = round(radius * 0.25, 2)
        gt.append(_cyl(0, 0, 0, z_center, central_r, cyl_h * 1.2))
        start_id = 1
    else:
        start_id = 0

    angle_step = 2 * math.pi / n_radial
    for i in range(n_radial):
        a = i * angle_step
        cx = round(radius * math.cos(a), 3)
        cy = round(radius * math.sin(a), 3)
        gt.append(_cyl(start_id + i, cx, cy, z_center, cyl_r, cyl_h))

    total = len(gt)
    center_desc = (f"  - 1 central cylinder at origin: radius={round(radius*0.25,2)}mm, "
                   f"height={round(cyl_h*1.2,2)}mm\n") if has_center else ""
    prompt = (
        f"Generate a radial array of {n_radial} cylinders around the Z-axis.\n"
        f"{center_desc}"
        f"  - Array radius: {radius}mm from Z-axis\n"
        f"  - Each cylinder: radius={cyl_r}mm, height={cyl_h}mm\n"
        f"  - All cylinder centers at z={z_center}mm\n"
        f"  - Equally spaced: {round(360/n_radial, 2)} degrees apart\n"
        f"  - First cylinder at angle=0 degrees -> position ({radius}, 0, {z_center})\n"
        f"Output all {total} shapes in the JSON array."
    )
    meta = {"n": total, "coord_range": cr, "n_radial": n_radial,
            "radius": radius, "has_center": has_center}
    return prompt, gt, meta


# ── G3: STACKED TOWER ──
def gen_stacked_tower(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)
    n, cr = _wave_params(wave)
    n_layers = min(n, 3 + wave * 2)

    radii = [round(rng.uniform(5, 30) * (0.85 ** i), 2) for i in range(n_layers)]
    heights = [round(rng.uniform(10, 40), 2) for _ in range(n_layers)]

    gt = []
    z_bottom = 0.0
    for i in range(n_layers):
        z_center = round(z_bottom + heights[i] / 2, 3)
        gt.append(_cyl(i, 0, 0, z_center, radii[i], heights[i]))
        z_bottom += heights[i]

    layer_specs = "\n".join(
        f"  Layer {i}: radius={radii[i]}mm, height={heights[i]}mm"
        for i in range(n_layers))
    prompt = (
        f"Build a stacked tower of {n_layers} cylinders, all centered on the Z-axis.\n"
        f"Each cylinder sits directly on top of the previous one (no gap, no overlap).\n"
        f"The tower starts at z=0 (bottom of layer 0).\n"
        f"Layer specifications:\n{layer_specs}\n"
        f"IMPORTANT: derive each cylinder's center Z from the cumulative heights.\n"
        f"Output all {n_layers} cylinders in the JSON array."
    )
    meta = {"n": n_layers, "coord_range": cr, "radii": radii, "heights": heights}
    return prompt, gt, meta


# ── G4: MIRROR ASSEMBLY ──
def gen_mirror_assembly(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)
    n, cr = _wave_params(wave)
    n_base = max(2, min(n // 2, 3 + wave))

    mirror_axis = rng.choice(["X", "Y", "Z"])
    axis_idx = {"X": 0, "Y": 1, "Z": 2}[mirror_axis]

    def mirror_center(c):
        mc = list(c)
        mc[axis_idx] = round(-mc[axis_idx], 3)
        return mc

    shape_types = [rng.choice(["cylinder", "sphere", "box"]) for _ in range(n_base)]
    base_shapes = []
    for i, st in enumerate(shape_types):
        c = [round(rng.uniform(5, cr * 0.4), 2) for _ in range(3)]
        c[axis_idx] = round(abs(c[axis_idx]) + 5, 2)
        r = round(rng.uniform(3, 15), 2)
        h = round(rng.uniform(r * 1.5, r * 3), 2)
        if st == "sphere":
            base_shapes.append(_sph(i, c[0], c[1], c[2], r))
        elif st == "cylinder":
            base_shapes.append(_cyl(i, c[0], c[1], c[2], r, h))
        else:
            base_shapes.append(_box(i, c[0], c[1], c[2], r*2, r*2, h))

    mirrored = []
    for i, s in enumerate(base_shapes):
        mc = mirror_center(s["center"])
        ms = dict(s)
        ms["id"] = n_base + i
        ms["center"] = mc
        mirrored.append(ms)

    gt = base_shapes + mirrored

    shape_desc = "\n".join(
        (f"  id={s['id']}: {s['type']} at center=({s['center'][0]}, {s['center'][1]}, {s['center'][2]})"
         + (f", radius={s['radius']}" if 'radius' in s else '')
         + (f", height={s['height']}" if 'height' in s else '')
         + (f", size={s['size']}" if 'size' in s else ''))
        for s in base_shapes)

    prompt = (
        f"Given these {n_base} shapes:\n{shape_desc}\n\n"
        f"Mirror ALL of them across the {mirror_axis}=0 plane "
        f"(negate the {mirror_axis} coordinate of each center).\n"
        f"Output all {len(gt)} shapes (original + mirrored) in the JSON array.\n"
        f"Mirrored shapes start at id={n_base}."
    )
    meta = {"n": len(gt), "coord_range": cr, "n_base": n_base, "mirror_axis": mirror_axis}
    return prompt, gt, meta


# ── G5: ALTERNATING LATTICE ──
def gen_alternating_lattice(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)
    n, cr = _wave_params(wave)

    side = max(2, round(n ** (1/3)))
    nx = side; ny = side; nz = max(1, min(side, n // (side * side)))
    spacing = round(rng.uniform(18, 30), 1)
    r_sph = round(rng.uniform(4, spacing * 0.3), 2)
    r_cyl = round(r_sph * 0.7, 2)
    h_cyl = round(spacing * 0.6, 2)

    gt = []
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                cx = round((ix - (nx-1)/2) * spacing, 3)
                cy = round((iy - (ny-1)/2) * spacing, 3)
                cz = round((iz - (nz-1)/2) * spacing, 3)
                if (ix + iy + iz) % 2 == 0:
                    gt.append(_sph(idx, cx, cy, cz, r_sph))
                else:
                    gt.append(_cyl(idx, cx, cy, cz, r_cyl, h_cyl))
                idx += 1

    total = nx * ny * nz
    prompt = (
        f"Generate a {nx}x{ny}x{nz} 3D lattice with alternating shape types.\n"
        f"  - Spacing: {spacing}mm in all directions\n"
        f"  - Lattice centered at origin\n"
        f"  - Rule: if (ix + iy + iz) is EVEN -> sphere (radius={r_sph}mm)\n"
        f"           if (ix + iy + iz) is ODD  -> cylinder (radius={r_cyl}mm, height={h_cyl}mm)\n"
        f"  - ix, iy, iz are zero-based grid indices (0 to {nx-1}, 0 to {ny-1}, 0 to {nz-1})\n"
        f"  - Total shapes: {total}\n"
        f"Output all {total} shapes in the JSON array."
    )
    meta = {"n": total, "coord_range": cr, "grid": (nx, ny, nz), "spacing": spacing}
    return prompt, gt, meta


# ── G6: SUPPORTED PYRAMID (v5.0) ──
def gen_supported_pyramid(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)
    n_layers = min(5, 2 + wave // 5)

    widths = [30 + 15*i for i in range(n_layers)]
    widths.reverse()

    gt = []
    z = 0.0
    idx = 0
    for layer in range(n_layers):
        n_boxes = n_layers - layer
        width_per_box = widths[layer]
        height = 15.0
        depth = 20.0
        total_width = n_boxes * width_per_box
        start_x = -total_width / 2 + width_per_box / 2

        for box_i in range(n_boxes):
            cx = round(start_x + box_i * width_per_box, 3)
            cy = 0.0
            cz = round(z + height / 2, 3)
            gt.append(_box(idx, cx, cy, cz, width_per_box, depth, height))
            idx += 1
        z += height

    total = len(gt)
    w_list = ", ".join(str(w) for w in widths)
    prompt = (
        f"Build a supported pyramid of {n_layers} layers using boxes.\n"
        f"  - Bottom layer: {n_layers} boxes side-by-side\n"
        f"  - Each subsequent layer: one fewer box, centered on layer below\n"
        f"  - All boxes height: 15mm, depth: 20mm\n"
        f"  - Layer box widths (largest to smallest): {w_list}mm\n"
        f"  - Boxes stack vertically (no gap)\n"
        f"  - CRITICAL: Each box must rest on boxes below (gravity support)\n"
        f"Output all {total} boxes in the JSON array."
    )
    meta = {"n": total, "coord_range": 100, "n_layers": n_layers, "widths": widths}
    return prompt, gt, meta


# ── G7: CONNECTED BRIDGE (v5.0) ──
def gen_connected_bridge(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)

    pillar_h = 60.0; pillar_r = 4.0; deck_h = 5.0; brace_r = 3.0
    gt = []
    idx = 0

    left_x = -30.0; right_x = 30.0
    pillar_z = pillar_h / 2

    gt.append(_cyl(idx, left_x, 0, pillar_z, pillar_r, pillar_h)); idx += 1
    gt.append(_cyl(idx, right_x, 0, pillar_z, pillar_r, pillar_h)); idx += 1

    deck_z = pillar_h
    for i, dy in enumerate([0, -10, 10]):
        ax = [1.0, 0.0, 0.0]
        gt.append({"id": idx, "type": "cylinder",
                   "center": [0.0, round(dy, 3), round(deck_z, 3)],
                   "radius": 3.0, "height": 60.0, "axis": ax})
        idx += 1

    gt.append(_cyl(idx, left_x, -5, (pillar_h + deck_z)/2, brace_r,
                   math.sqrt((15)**2 + (30)**2))); idx += 1
    gt.append(_cyl(idx, right_x, 5, (pillar_h + deck_z)/2, brace_r,
                   math.sqrt((15)**2 + (30)**2))); idx += 1

    total = len(gt)
    prompt = (
        f"Build a bridge structure with connectivity:\n"
        f"  - 2 vertical pillars: radius={pillar_r}mm, height={pillar_h}mm\n"
        f"    left pillar center: ({left_x}, 0, {pillar_z})\n"
        f"    right pillar center: ({right_x}, 0, {pillar_z})\n"
        f"  - 3 horizontal deck beams: radius=3mm, connecting pillars at z={deck_z}\n"
        f"    beams at y=0, y=-10, y=10\n"
        f"  - 2 diagonal braces: radius={brace_r}mm (crisscross)\n"
        f"  - CRITICAL: Every beam must physically touch another beam (no floating parts)\n"
        f"Output all {total} cylinders in the JSON array."
    )
    meta = {"n": total, "coord_range": 70}
    return prompt, gt, meta


# ── G8: PROPORTIONED TOWER (v5.0) ──
def gen_proportioned_tower(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)

    w1 = 80.0
    w2 = round(w1 * 0.60, 2)
    w3 = round(w1 * 0.35, 2)
    w4 = round(w1 * 0.15, 2)
    heights = [20.0, 20.0, 20.0, 20.0]

    gt = []
    z = 0.0
    widths = [w1, w2, w3, w4]
    for sec, w in enumerate(widths):
        cz = round(z + heights[sec] / 2, 3)
        gt.append(_box(sec, 0, 0, cz, w, w, heights[sec]))
        z += heights[sec]

    prompt = (
        f"Build a proportioned tower (like Eiffel Tower) with 4 sections:\n"
        f"  - Section 0 (base): width={w1}mm x {w1}mm, height={heights[0]}mm\n"
        f"  - Section 1: width={w2}mm x {w2}mm (60% of base), height={heights[1]}mm\n"
        f"  - Section 2: width={w3}mm x {w3}mm (35% of base), height={heights[2]}mm\n"
        f"  - Section 3: width={w4}mm x {w4}mm (15% of base), height={heights[3]}mm\n"
        f"  - All sections centered at x=0, y=0\n"
        f"  - Sections stack vertically from z=0\n"
        f"  - CRITICAL: WIDTH RATIOS must be EXACT (0.60, 0.35, 0.15)\n"
        f"Output all 4 boxes in the JSON array."
    )
    meta = {"n": 4, "coord_range": 100, "w1": w1, "w2": w2, "w3": w3, "w4": w4,
            "expected_ratios": [("w2/w1", 0.60), ("w3/w1", 0.35), ("w4/w1", 0.15)]}
    return prompt, gt, meta


# ── G9: CANTILEVER ASSEMBLY (v5.0) ──
def gen_cantilever_assembly(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)

    gt = []
    gt.append(_box(0, 0, 0, 10, 30, 30, 20))
    gt.append(_box(1, 25, 0, 25, 50, 10, 8))

    brace_start = [15, 0, 20]
    brace_end = [25, 0, 25]
    brace_mid = [(brace_start[i] + brace_end[i])/2 for i in range(3)]
    brace_len = math.sqrt(sum((brace_end[i] - brace_start[i])**2 for i in range(3)))
    gt.append(_cyl(2, round(brace_mid[0], 3), round(brace_mid[1], 3),
                   round(brace_mid[2], 3), 2.5, round(brace_len, 2)))

    prompt = (
        f"Build a cantilever assembly:\n"
        f"  - Base block: 30mm x 30mm x 20mm, centered at (0, 0, 10)\n"
        f"  - Arm: 50mm long x 10mm wide x 8mm tall, extending at (25, 0, 25)\n"
        f"  - Brace: cylinder radius=2.5mm, connecting base to arm\n"
        f"    brace must touch both the base and the arm\n"
        f"  - CRITICAL: Arm must be connected to base via brace\n"
        f"Output all 3 shapes in the JSON array."
    )
    meta = {"n": 3, "coord_range": 80}
    return prompt, gt, meta


# ── G10: AERODYNAMIC HULL (v5.0) ──
def gen_aerodynamic_hull(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    rng = random.Random(seed)

    n_sections = 5
    gt = []
    z_positions = [10 + i*15 for i in range(n_sections)]
    widths = [40 - i*6 for i in range(n_sections)]
    heights = [30 - i*4 for i in range(n_sections)]

    for i in range(n_sections):
        gt.append(_box(i, 0, 0, z_positions[i], widths[i], widths[i], heights[i]))

    prompt = (
        f"Build an aerodynamic hull with {n_sections} cross-sections:\n"
        f"  - Sections aligned along Z-axis (nose to tail)\n"
        f"  - Section positions (z): {z_positions}\n"
        f"  - Widths taper: {widths}\n"
        f"  - Heights taper: {heights}\n"
        f"  - All sections centered at x=0, y=0\n"
        f"  - CRITICAL: Sections must be continuous\n"
        f"Output all {n_sections} boxes in the JSON array."
    )
    meta = {"n": n_sections, "coord_range": 100, "n_sections": n_sections,
            "z_positions": z_positions, "widths": widths, "heights": heights}
    return prompt, gt, meta


# ── G11: ORIENTED PIPE NETWORK (v5.1) ──
def gen_oriented_pipes(wave: int, seed: int) -> Tuple[str, List[dict], dict]:
    """Cylinders at non-trivial angles — tests orientation reasoning."""
    rng = random.Random(seed)
    n, cr = _wave_params(wave)
    n_pipes = max(3, min(n, 30))

    gt = []
    pipe_specs = []

    for i in range(n_pipes):
        cx = round(rng.uniform(-cr * 0.6, cr * 0.6), 1)
        cy = round(rng.uniform(-cr * 0.6, cr * 0.6), 1)
        cz = round(rng.uniform(0, cr * 0.5), 1)
        radius = round(rng.uniform(2.0, 6.0), 1)
        height = round(rng.uniform(15.0, 50.0), 1)

        theta_deg = rng.randint(0, 359)
        phi_deg = rng.randint(15, 75)
        theta = math.radians(theta_deg)
        phi = math.radians(phi_deg)

        ax = round(math.sin(phi) * math.cos(theta), 4)
        ay = round(math.sin(phi) * math.sin(theta), 4)
        az = round(math.cos(phi), 4)

        gt.append({"id": i, "type": "cylinder", "center": [cx, cy, cz],
                   "radius": radius, "height": height, "axis": [ax, ay, az]})
        pipe_specs.append(
            f"  Pipe {i}: center=[{cx},{cy},{cz}], r={radius}, h={height}, "
            f"axis=[{ax},{ay},{az}] (azimuth={theta_deg}, elevation={phi_deg})")

    prompt = (
        f"Build a pipe network with {n_pipes} cylinders at various orientations.\n"
        f"CRITICAL: Each cylinder has a specific axis direction vector.\n"
        f"Pipe specifications:\n" + "\n".join(pipe_specs) + "\n"
        f"Output all {n_pipes} cylinders as a JSON array."
    )
    meta = {"n": n_pipes, "coord_range": cr, "n_pipes": n_pipes}
    return prompt, gt, meta


# ── GENERATOR LIST ──
GENERATORS = [
    gen_uniform_grid,
    gen_radial_array,
    gen_stacked_tower,
    gen_mirror_assembly,
    gen_alternating_lattice,
    gen_supported_pyramid,
    gen_connected_bridge,
    gen_proportioned_tower,
    gen_cantilever_assembly,
    gen_aerodynamic_hull,
    gen_oriented_pipes,
]


def generate_stress_level(wave: int, master_seed: int) -> tuple:
    """Generate a stress test wave. Returns (info_dict, ground_truth_list)."""
    gen_idx = (wave - 1) % len(GENERATORS)
    gen_fn = GENERATORS[gen_idx]
    gen_name = GENERATOR_NAMES[gen_idx]
    wave_seed = (master_seed * 997 + wave * 31) % (2**31)

    prompt, gt, meta = gen_fn(wave, wave_seed)
    _, cr = _wave_params(wave)

    info = {
        "wave": wave,
        "name": f"S{wave}: {gen_name}",
        "skill": gen_name,
        "prompt": prompt,
        "seed": wave_seed,
        "n_parts": meta.get("n", len(gt)),
        "coord_range": cr,
        "generator_name": gen_name,
    }
    return info, gt
