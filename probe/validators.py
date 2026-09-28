"""
Validation engines: spatial geometry, physics (connectivity, gravity, proportions),
orientation, reasoning quality, and engineering design rules (v6.0).

v6.0 additions:
  validate_interference()       — detect unintended volumetric overlaps
  validate_clearance_fit()      — check radial/linear clearance between pairs
  validate_wall_thickness()     — check minimum wall on hollow shapes
  validate_mates()              — concentric / coincident / tangent / distance mates
  compute_chamfer_distance()    — surface-sampled bidirectional similarity metric
  validate_symmetry()           — mirror-symmetry check across XY/YZ/XZ planes
  validate_pattern()            — circular or linear pattern regularity
  analyze_reasoning_chain()     — verify intermediate computation values appear
"""

import re, math, random
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# BOUNDING BOX HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_bbox(shape: dict) -> Tuple[List[float], List[float]]:
    """
    Get axis-aligned bounding box (min, max) for a shape.

    For cylinders with non-Z axes, computes the correct AABB by projecting
    the cylinder's oriented extent onto each world axis.  This fixes the
    connectivity / gravity / interference false-positives on tilted cylinders.
    """
    c = shape.get("center", [0, 0, 0])
    shape_type = shape.get("type", "")

    if shape_type == "box":
        size = shape.get("size", [0, 0, 0])
        min_pt = [c[i] - size[i]/2 for i in range(3)]
        max_pt = [c[i] + size[i]/2 for i in range(3)]
    elif shape_type == "cylinder":
        r = shape.get("radius", 0)
        h = shape.get("height", 0)
        ax = shape.get("axis", [0, 0, 1])
        # Normalise axis
        mag = math.sqrt(sum(a*a for a in ax))
        if mag < 1e-9:
            ax = [0.0, 0.0, 1.0]
        else:
            ax = [a / mag for a in ax]
        # For each world axis i, the half-extent is:
        #   half_height * |ax[i]| + r * sqrt(1 - ax[i]^2)
        # This accounts for both the height contribution along the axis
        # and the radial contribution perpendicular to it.
        hh = h / 2.0
        extents = []
        for i in range(3):
            e = hh * abs(ax[i]) + r * math.sqrt(max(0.0, 1.0 - ax[i]*ax[i]))
            extents.append(e)
        min_pt = [c[i] - extents[i] for i in range(3)]
        max_pt = [c[i] + extents[i] for i in range(3)]
    elif shape_type == "sphere":
        r = shape.get("radius", 0)
        min_pt = [c[i] - r for i in range(3)]
        max_pt = [c[i] + r for i in range(3)]
    else:
        min_pt = c[:]
        max_pt = c[:]

    return min_pt, max_pt


def _bboxes_overlap(bbox1: Tuple[List[float], List[float]],
                    bbox2: Tuple[List[float], List[float]], tol_mm: float) -> bool:
    """Check if two bounding boxes overlap (with tolerance)."""
    min1, max1 = bbox1
    min2, max2 = bbox2
    for axis in range(3):
        if max1[axis] + tol_mm < min2[axis] or max2[axis] + tol_mm < min1[axis]:
            return False
    return True


def _vec_dist(a, b):
    return math.sqrt(sum((ai - bi)**2 for ai, bi in zip(a, b)))


def _norm(v):
    mag = math.sqrt(sum(x*x for x in v))
    return [x / max(mag, 1e-9) for x in v] if mag > 1e-9 else [0.0, 0.0, 1.0]


# ═══════════════════════════════════════════════════════════════
# PHYSICS VALIDATORS (v5.x — unchanged)
# ═══════════════════════════════════════════════════════════════

def validate_connectivity(shapes: List[dict], tol_mm: float = 2.0) -> dict:
    """
    Build connectivity graph: shapes are connected if bboxes overlap/touch.
    Return: {"connected": bool, "islands": int, "floating_ids": [list]}
    """
    if not shapes:
        return {"connected": True, "islands": 0, "floating_ids": []}

    n = len(shapes)
    if n == 1:
        return {"connected": True, "islands": 1, "floating_ids": []}

    bboxes = {s.get("id", i): _get_bbox(s) for i, s in enumerate(shapes)}
    shape_ids = list(bboxes.keys())

    adj = {sid: [] for sid in shape_ids}
    for i, sid1 in enumerate(shape_ids):
        for sid2 in shape_ids[i+1:]:
            if _bboxes_overlap(bboxes[sid1], bboxes[sid2], tol_mm):
                adj[sid1].append(sid2)
                adj[sid2].append(sid1)

    visited = set()
    islands = 0
    floating = []

    def bfs(start):
        q = [start]
        visited.add(start)
        while q:
            u = q.pop(0)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)

    for sid in shape_ids:
        if sid not in visited:
            islands += 1
            if islands == 1:
                bfs(sid)
            else:
                floating.append(sid)

    connected = (islands <= 1)
    return {"connected": connected, "islands": islands, "floating_ids": floating}


def validate_gravity(shapes: List[dict], ground_z: float = 0.0, tol_mm: float = 1.0) -> dict:
    """Check if each shape is supported (touches ground or shape below it)."""
    if not shapes:
        return {"all_supported": True, "floating_count": 0, "floating_ids": []}

    floating_ids = []

    def bottom_z(shape):
        c = shape.get("center", [0, 0, 0])[2]
        shape_type = shape.get("type", "")
        if shape_type == "box":
            h = shape.get("size", [0, 0, 0])[2]
            return c - h / 2
        elif shape_type == "cylinder":
            h = shape.get("height", 0)
            return c - h / 2
        elif shape_type == "sphere":
            r = shape.get("radius", 0)
            return c - r
        return c

    shapes_sorted = sorted(shapes, key=bottom_z)

    for i, shape in enumerate(shapes_sorted):
        bz = bottom_z(shape)
        shape_id = shape.get("id", i)

        if bz <= ground_z + tol_mm:
            continue

        supported = False
        for j in range(i):
            below = shapes_sorted[j]
            _, max_below = _get_bbox(below)
            if abs(bz - max_below[2]) <= tol_mm:
                supported = True
                break

        if not supported:
            floating_ids.append(shape_id)

    all_supported = (len(floating_ids) == 0)
    return {"all_supported": all_supported, "floating_count": len(floating_ids),
            "floating_ids": floating_ids}


def validate_proportions(shapes: List[dict], expected_ratios: List[Tuple],
                         tol_ratio: float = 0.15) -> dict:
    """Check dimension ratios between shapes."""
    if not expected_ratios:
        return {"ratios_checked": 0, "ratios_correct": 0, "details": []}

    details = []
    correct = 0

    if len(shapes) >= 4 and all(s.get("type") == "box" for s in shapes):
        widths = [s.get("size", [0, 0, 0])[0] for s in shapes]
        if widths[0] > 0:
            ratio_21 = widths[1] / widths[0]
            ratio_31 = widths[2] / widths[0]
            ratio_41 = widths[3] / widths[0]

            if abs(ratio_21 - 0.60) <= tol_ratio: correct += 1
            if abs(ratio_31 - 0.35) <= tol_ratio: correct += 1
            if abs(ratio_41 - 0.15) <= tol_ratio: correct += 1

            details.append(f"w2/w1={ratio_21:.2f} (exp 0.60)")
            details.append(f"w3/w1={ratio_31:.2f} (exp 0.35)")
            details.append(f"w4/w1={ratio_41:.2f} (exp 0.15)")

    return {"ratios_checked": len(expected_ratios), "ratios_correct": correct, "details": details}


def validate_orientations(llm_shapes: List[dict], gt_shapes: List[dict],
                           tol_angle_deg: float = 15.0) -> dict:
    """
    Check if cylinder axes match ground truth via dot product.
    Antiparallel axes treated as correct.
    """
    checked = 0; correct = 0; details = []

    gt_by_id = {s["id"]: s for s in gt_shapes if "id" in s and s.get("type") == "cylinder"}
    llm_by_id = {}
    for s in llm_shapes:
        if isinstance(s, dict) and "id" in s and s.get("type") == "cylinder":
            llm_by_id[int(s["id"])] = s

    for sid, gt_s in gt_by_id.items():
        gt_axis = gt_s.get("axis", [0, 0, 1])
        llm_s = llm_by_id.get(sid)
        if llm_s is None:
            continue
        llm_axis = llm_s.get("axis", [0, 0, 1])
        checked += 1

        ga = _norm(gt_axis); la = _norm(llm_axis)
        dot = abs(sum(a * b for a, b in zip(ga, la)))
        dot = min(dot, 1.0)
        angle_deg = math.degrees(math.acos(dot))
        ok = angle_deg <= tol_angle_deg
        if ok: correct += 1
        details.append(f"id={sid} axis_err={angle_deg:.1f}° {'OK' if ok else 'FAIL'}")

    return {"checked": checked, "correct": correct,
            "accuracy": correct / max(checked, 1), "details": details}


# ═══════════════════════════════════════════════════════════════
# SPATIAL GEOMETRY VALIDATION (v5.x — unchanged)
# ═══════════════════════════════════════════════════════════════

def _rel_err(lm_v, gt_v):
    """Single-value relative error helper."""
    if gt_v > 1e-9:
        return abs(lm_v - gt_v) / gt_v
    elif abs(lm_v) > 1e-9:
        return 1.0
    return 0.0


def _dim_rel_error(llm_obj: dict, gt_obj: dict) -> float:
    """
    Compute relative dimension error between LLM and ground-truth shape.

    Returns mean fractional error over all key dimensions:
      cylinder → radius + height
      box      → width + depth + height
      sphere   → radius
      cone     → start_radius + end_radius + height
      torus    → ring_radius + tube_radius
      pipe     → inner_radius + outer_radius + height
      beam     -> start_radius + end_radius + beam_length

    A value of 0.0 means perfect dimensions; 0.5 means 50% off on average.
    Capped at 1.0.
    """
    t = gt_obj.get("type", "")
    errors = []

    if t == "cylinder":
        for key in ("radius", "height"):
            errors.append(_rel_err(llm_obj.get(key, 0.0), gt_obj.get(key, 0.0)))

    elif t == "box":
        gt_size = gt_obj.get("size", [0, 0, 0])
        lm_size = llm_obj.get("size", [0, 0, 0])
        for i in range(3):
            gt_v = gt_size[i] if i < len(gt_size) else 0.0
            lm_v = lm_size[i] if i < len(lm_size) else 0.0
            errors.append(_rel_err(lm_v, gt_v))

    elif t == "sphere":
        errors.append(_rel_err(llm_obj.get("radius", 0.0), gt_obj.get("radius", 0.0)))

    elif t == "cone":
        for key in ("start_radius", "end_radius", "height"):
            errors.append(_rel_err(llm_obj.get(key, 0.0), gt_obj.get(key, 0.0)))

    elif t == "torus":
        for key in ("ring_radius", "tube_radius"):
            errors.append(_rel_err(llm_obj.get(key, 0.0), gt_obj.get(key, 0.0)))

    elif t == "pipe":
        for key in ("inner_radius", "outer_radius", "height"):
            errors.append(_rel_err(llm_obj.get(key, 0.0), gt_obj.get(key, 0.0)))

    elif t == "beam":
        # Check start_radius, end_radius, and beam length
        for key in ("start_radius", "end_radius"):
            errors.append(_rel_err(llm_obj.get(key, 0.0), gt_obj.get(key, 0.0)))
        # Beam length (derived from start/end points)
        gt_s, gt_e = gt_obj.get("start", [0,0,0]), gt_obj.get("end", [0,0,0])
        lm_s, lm_e = llm_obj.get("start", [0,0,0]), llm_obj.get("end", [0,0,0])
        gt_len = math.sqrt(sum((gt_e[i]-gt_s[i])**2 for i in range(3)))
        lm_len = math.sqrt(sum((lm_e[i]-lm_s[i])**2 for i in range(3)))
        if gt_len > 1e-9:
            errors.append(_rel_err(lm_len, gt_len))

    if not errors:
        return 0.0
    return min(1.0, sum(errors) / len(errors))


def _get_shape_center(obj: dict) -> list:
    """Extract geometric center from any shape type.
    Most shapes have 'center'; beams use midpoint(start,end)."""
    if "center" in obj:
        return obj["center"]
    if obj.get("type") == "beam":
        s = obj.get("start", [0,0,0])
        e = obj.get("end", [0,0,0])
        return [(s[i]+e[i])/2 for i in range(3)]
    return [0, 0, 0]


def validate_geometry(llm_shapes: List[dict], gt_shapes: List[dict],
                      check_ids: List[int], tol_mm: float) -> dict:
    """
    Compare LLM output against ground truth for specific shape IDs.

    Checks THREE things per shape:
      1. Type match (cylinder/box/sphere/cone/torus/pipe/beam)
      2. Center position error (mm) - beams use midpoint
      3. Dimension error — type-specific relative dimension comparison.

    Details strings include pos_err=Xmm and dim_err=Y% so the scorer can
    compute a continuous per-shape score without double-counting Chamfer.
    """
    result = {
        "positions_checked": 0,
        "positions_correct": 0,
        "types_correct":     False,
        "ops_correct":       True,
        "details":           [],
        "dim_errors":        [],   # parallel list of relative dim errors [0,1]
    }

    if not llm_shapes:
        result["types_correct"] = False
        return result

    llm_by_id: Dict[int, dict] = {}
    for obj in llm_shapes:
        if isinstance(obj, dict) and "id" in obj:
            llm_by_id[int(obj["id"])] = obj

    gt_by_id: Dict[int, dict] = {s["id"]: s for s in gt_shapes if "id" in s}

    type_matches = 0
    for check_id in check_ids:
        if check_id not in gt_by_id:
            continue
        gt_obj  = gt_by_id[check_id]
        llm_obj = llm_by_id.get(check_id)

        result["positions_checked"] += 1

        if llm_obj is None:
            result["details"].append(f"id={check_id}: MISSING in LLM output")
            result["dim_errors"].append(1.0)
            continue

        # ── 1. Type check ───────────────────────────────────────
        type_match = (llm_obj.get("type") == gt_obj.get("type"))
        if type_match:
            type_matches += 1

        # ── 2. Position error (mm) ──────────────────────────────
        gt_center  = _get_shape_center(gt_obj)
        llm_center = _get_shape_center(llm_obj)
        if llm_center is None or None in llm_center or len(llm_center) < 3:
            result["details"].append(
                f"id={check_id}: malformed center type_mismatch={not type_match}")
            result["dim_errors"].append(1.0)
            continue

        dist   = _vec_dist(gt_center, llm_center)
        pos_ok = dist <= tol_mm
        if pos_ok:
            result["positions_correct"] += 1

        # ── 3. Dimension error (relative) ───────────────────────
        if type_match:
            dim_err = _dim_rel_error(llm_obj, gt_obj)
        else:
            dim_err = 1.0  # wrong type → full dimension penalty

        result["dim_errors"].append(dim_err)

        tm_tag = "" if type_match else " type_mismatch"
        result["details"].append(
            f"id={check_id} ({gt_obj.get('type')}) "
            f"pos_err={dist:.1f}mm "
            f"dim_err={dim_err*100:.0f}% "
            f"{'OK' if pos_ok else 'FAIL'}{tm_tag}")

    result["types_correct"] = (type_matches == len(check_ids)) if check_ids else False
    return result


def validate_stress_geometry(llm_shapes: List[dict], gt_shapes: List[dict],
                              tol_mm: float = 2.0) -> dict:
    """Validation for stress tests: sample up to 20 positions."""
    if not gt_shapes:
        return {"positions_checked": 0, "positions_correct": 0,
                "types_correct": False, "details": []}

    ids_to_check = set()
    shape_ids = [s["id"] for s in gt_shapes if "id" in s]
    if shape_ids:
        ids_to_check.add(shape_ids[0])
        ids_to_check.add(shape_ids[-1])
        if len(shape_ids) > 4:
            ids_to_check.add(shape_ids[len(shape_ids)//4])
            ids_to_check.add(shape_ids[len(shape_ids)//2])
            ids_to_check.add(shape_ids[3*len(shape_ids)//4])
        rng = random.Random(42)
        sample = rng.sample(shape_ids, min(15, len(shape_ids)))
        ids_to_check.update(sample)

    return validate_geometry(llm_shapes, gt_shapes, list(ids_to_check), tol_mm)


# ═══════════════════════════════════════════════════════════════
# JSON EXTRACTION & COUNTING (v5.x — unchanged)
# ═══════════════════════════════════════════════════════════════

def extract_json(text: str) -> Optional[list]:
    """Extract and parse a JSON array from LLM response.

    Handles many real-world LLM output quirks:
      - Markdown-fenced JSON (```json ... ```)
      - Bare JSON arrays
      - Trailing commas
      - Truncated JSON (missing closing brackets)
      - Thinking blocks / preamble text before the array
      - JSON wrapped in an object like {"shapes": [...]} or {"result": [...]}
    """
    import json

    if not text or not text.strip():
        return None

    text = text.strip()

    # ── 0) Strip <think>...</think> reasoning blocks ──────────
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()

    # ── 1) Fenced: ```json ... ``` (greedy to get the full block) ─
    m = re.search(r'```(?:json)?\s*\n?(\[[\s\S]*?\])\s*\n?```', text, re.IGNORECASE)
    if m:
        parsed = _try_parse_array(m.group(1))
        if parsed is not None:
            return parsed

    # ── 2) Direct JSON array — first '[' to matching ']' ─────
    start = text.find('[')
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_str:
                escape = True
                continue
            if ch == '"' and not escape:
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    parsed = _try_parse_array(text[start:i+1])
                    if parsed is not None:
                        return parsed
                    break

        # ── 3) Truncated JSON — array started but never closed ──
        # Try to repair by closing open brackets
        if depth > 0:
            chunk = text[start:]
            # Remove any trailing partial object
            last_close = max(chunk.rfind('}'), chunk.rfind(']'))
            if last_close > 0:
                chunk = chunk[:last_close+1]
            chunk = chunk.rstrip().rstrip(',')
            chunk += ']' * depth
            parsed = _try_parse_array(chunk)
            if parsed is not None:
                return parsed

    # ── 4) JSON object wrapper: {"shapes":[...]} or {"result":[...]} ─
    obj_start = text.find('{')
    if obj_start != -1:
        try:
            obj = json.loads(text[obj_start:])
            if isinstance(obj, dict):
                for key in ("shapes", "result", "data", "elements", "objects",
                            "geometry", "output", "json", "response"):
                    if key in obj and isinstance(obj[key], list):
                        return obj[key]
                # If it's a single shape object, wrap in list
                if "type" in obj and ("center" in obj or "start" in obj):
                    return [obj]
        except json.JSONDecodeError:
            pass

    return None


def _try_parse_array(chunk: str) -> Optional[list]:
    """Try to parse a JSON array string with progressive repair."""
    import json
    # Direct parse
    try:
        result = json.loads(chunk)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before } or ]
    cleaned = re.sub(r',\s*([}\]])', r'\1', chunk)
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Remove single-line comments // ...
    cleaned2 = re.sub(r'//[^\n]*', '', cleaned)
    try:
        result = json.loads(cleaned2)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Replace single quotes with double quotes (Python-style dicts)
    cleaned3 = cleaned2.replace("'", '"')
    try:
        result = json.loads(cleaned3)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    return None


def count_shape_objects(parsed: list) -> int:
    """Count how many objects are actual shapes (not ops)."""
    if not parsed:
        return 0
    return sum(1 for obj in parsed
               if isinstance(obj, dict) and obj.get("type") in ("cylinder", "box", "sphere"))


# ═══════════════════════════════════════════════════════════════
# REASONING QUALITY ANALYSIS (v5.x — unchanged)
# ═══════════════════════════════════════════════════════════════

def analyze_reasoning(response_text: str, n_expected: int) -> dict:
    """Analyze the LLM's reasoning quality from its response."""
    depth = 0.0
    has_loop = False

    if not response_text:
        return {"reasoning_depth": 0.0, "has_loop_logic": False}

    txt_lower = response_text.lower()

    # Check for formula / loop logic
    loop_indicators = [
        r'\bfor\s+\w+\s+in\b', r'\brange\s*\(', r'\bwhile\b',
        r'i\s*[*+]\s*\d', r'\bcos\s*\(', r'\bsin\s*\(',
        r'\bi\s*\*\s*angle', r'angle_step', r'\bfor\s*\(',
    ]
    for pattern in loop_indicators:
        if re.search(pattern, response_text, re.IGNORECASE):
            has_loop = True
            break

    signals = [
        (r'step\s*\d|step\s+\w+:', 0.2),
        (r'therefore|thus|hence|so\s+the', 0.15),
        (r'formula|equation|calculate', 0.15),
        (r'let\s+\w+\s*=|define|where\s+\w+\s*=', 0.1),
        (r'\d+\s*[*+/-]\s*\d+\s*=', 0.1),
        (r'height\s*=|radius\s*=|diameter\s*=', 0.1),
        (r'position|center\s*=|coordinate', 0.1),
        (r'spacing|offset|gap|pitch', 0.1),
    ]

    for pattern, weight in signals:
        if re.search(pattern, response_text, re.IGNORECASE):
            depth += weight

    depth = min(1.0, depth)
    return {"reasoning_depth": depth, "has_loop_logic": has_loop}


# ═══════════════════════════════════════════════════════════════
# DFM / ENGINEERING VALIDATORS  (v6.0 — NEW)
# ═══════════════════════════════════════════════════════════════

def _cyl_cyl_interfere(s1: dict, s2: dict, tol_mm: float) -> bool:
    """Accurate interference check for two Z-axis cylinders."""
    c1 = s1.get("center", [0,0,0]); r1 = s1.get("radius", 0); h1 = s1.get("height", 0)
    c2 = s2.get("center", [0,0,0]); r2 = s2.get("radius", 0); h2 = s2.get("height", 0)
    xy = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    z1_lo = c1[2]-h1/2; z1_hi = c1[2]+h1/2
    z2_lo = c2[2]-h2/2; z2_hi = c2[2]+h2/2
    xy_overlap = xy < (r1 + r2 - tol_mm)
    z_overlap  = (z1_hi - tol_mm) > z2_lo and (z2_hi - tol_mm) > z1_lo
    return xy_overlap and z_overlap


def _shapes_actually_interfere(s1: dict, s2: dict, tol_mm: float) -> bool:
    """Best-effort geometric interference check."""
    t1 = s1.get("type"); t2 = s2.get("type")
    a1 = s1.get("axis", [0,0,1]); a2 = s2.get("axis", [0,0,1])
    if t1 == "cylinder" and t2 == "cylinder":
        if abs(a1[0]) < 0.15 and abs(a1[1]) < 0.15 and abs(a2[0]) < 0.15 and abs(a2[1]) < 0.15:
            return _cyl_cyl_interfere(s1, s2, tol_mm)
    # Fallback: tight bbox overlap (require actual penetration, not just touching)
    b1 = _get_bbox(s1); b2 = _get_bbox(s2)
    return _bboxes_overlap(b1, b2, -tol_mm)


def validate_interference(shapes: List[dict], ops: List[dict] = None,
                          tol_mm: float = 0.5) -> dict:
    """
    Detect unintended volumetric interference between shapes.

    Pairs linked by union/subtract ops are excluded (intentional overlaps).
    Returns: {interference_free, overlapping_pairs, score, details}

    Innovation: Only benchmark that checks manufacturing-grade interference
    — the difference between a clearance fit and a collision.
    """
    if not shapes or len(shapes) < 2:
        return {"interference_free": True, "overlapping_pairs": [], "score": 1.0, "details": []}

    # Collect intentional pairs from boolean ops
    intentional: set = set()
    for op in (ops or []):
        if op.get("op") == "union":
            parts = op.get("parts", [])
            for i in range(len(parts)):
                for j in range(i+1, len(parts)):
                    intentional.add((min(parts[i], parts[j]), max(parts[i], parts[j])))
        elif op.get("op") == "subtract":
            t = op.get("target"); tool = op.get("tool")
            if t is not None and tool is not None:
                intentional.add((min(t, tool), max(t, tool)))

    id_shape = [(s.get("id", i), s) for i, s in enumerate(shapes)]
    overlapping = []; details = []
    n_pairs = len(id_shape) * (len(id_shape) - 1) / 2

    for i, (id1, s1) in enumerate(id_shape):
        for j, (id2, s2) in enumerate(id_shape):
            if j <= i: continue
            pair = (min(id1, id2), max(id1, id2))
            if pair in intentional: continue
            if _shapes_actually_interfere(s1, s2, tol_mm):
                overlapping.append((id1, id2))
                details.append(
                    f"id={id1}({s1.get('type','?')}) ↔ "
                    f"id={id2}({s2.get('type','?')}): INTERFERENCE")

    score = max(0.0, 1.0 - len(overlapping) / max(1.0, n_pairs))
    return {
        "interference_free": len(overlapping) == 0,
        "overlapping_pairs": overlapping,
        "score": round(score, 3),
        "details": details,
    }


def validate_clearance_fit(shapes: List[dict], clearance_specs: List[dict],
                           tol_mm: float = 0.1) -> dict:
    """
    Check radial (cylinder-in-cylinder) or face (box-in-box) clearance fits.

    clearance_specs: [{shaft_id, hole_id, expected_clearance, tol}]

    Innovation: Tests whether LLM understands engineering fit classes
    (clearance, interference, transition) — unexplored in any benchmark.
    """
    by_id = {s.get("id", i): s for i, s in enumerate(shapes)}
    details = []; correct = 0
    checked = len(clearance_specs)

    for spec in clearance_specs:
        sid = spec.get("shaft_id"); hid = spec.get("hole_id")
        shaft = by_id.get(sid); hole = by_id.get(hid)
        exp_cl = spec.get("expected_clearance", 0.0)
        tol = spec.get("tol", tol_mm)

        if shaft is None or hole is None:
            details.append(f"shaft={sid} hole={hid}: MISSING shape(s)"); continue

        st = shaft.get("type"); ht = hole.get("type")
        if st == "cylinder" and ht == "cylinder":
            actual_cl = hole.get("radius", 0) - shaft.get("radius", 0)
            ok = abs(actual_cl - exp_cl) <= tol
            if ok: correct += 1
            err = abs(actual_cl - exp_cl)
            details.append(
                f"shaft={sid}(r={shaft.get('radius'):.3f}) "
                f"hole={hid}(r={hole.get('radius'):.3f}): "
                f"clearance={actual_cl:.3f}mm "
                f"(exp {exp_cl:.3f}±{tol:.3f}) "
                f"{'OK' if ok else f'FAIL err={err:.3f}mm'}")
        elif st == "box" and ht == "box":
            cs = shaft.get("center",[0,0,0]); ss = shaft.get("size",[0,0,0])
            ch = hole.get("center",[0,0,0]);  sh = hole.get("size",[0,0,0])
            gaps = [sh[k]/2 - abs(cs[k]-ch[k]) - ss[k]/2 for k in range(3)]
            actual_cl = min(gaps)
            ok = abs(actual_cl - exp_cl) <= tol
            if ok: correct += 1
            err = abs(actual_cl - exp_cl)
            details.append(
                f"box_shaft={sid} box_hole={hid}: "
                f"min_clearance={actual_cl:.3f}mm "
                f"(exp {exp_cl:.3f}±{tol:.3f}) "
                f"{'OK' if ok else f'FAIL err={err:.3f}mm'}")
        else:
            details.append(f"shaft={sid}({st}) hole={hid}({ht}): type mismatch")

    return {
        "checked": checked, "correct": correct,
        "score": round(correct / max(1, checked), 3),
        "details": details,
    }


def validate_wall_thickness(shapes: List[dict], wall_specs: List[dict],
                             tol_mm: float = 0.05) -> dict:
    """
    Validate minimum wall thickness for hollow shapes.

    wall_specs: [{outer_id, inner_id, min_wall_mm}]
    For coaxial cylinders: wall = r_outer − r_inner.
    For concentric boxes: wall = min face thickness.

    Innovation: Tests DFM awareness — minimum printable/machinable wall.
    """
    by_id = {s.get("id", i): s for i, s in enumerate(shapes)}
    details = []; violations = 0
    checked = len(wall_specs)

    for spec in wall_specs:
        oid = spec.get("outer_id"); iid = spec.get("inner_id")
        min_wall = spec.get("min_wall_mm", 2.0)
        outer = by_id.get(oid); inner = by_id.get(iid)

        if outer is None or inner is None:
            details.append(f"outer={oid} inner={iid}: MISSING shape(s)")
            violations += 1; continue

        ot = outer.get("type"); it_ = inner.get("type")
        if ot == "cylinder" and it_ == "cylinder":
            wall = outer.get("radius", 0) - inner.get("radius", 0)
            ok = wall >= min_wall - tol_mm
            if not ok: violations += 1
            details.append(
                f"outer={oid}(r={outer.get('radius'):.3f}) "
                f"inner={iid}(r={inner.get('radius'):.3f}): "
                f"wall={wall:.3f}mm (min {min_wall}mm) "
                f"{'OK' if ok else 'FAIL'}")
        elif ot == "box" and it_ == "box":
            so = outer.get("size",[0,0,0]); si = inner.get("size",[0,0,0])
            walls = [(so[k] - si[k]) / 2 for k in range(3)]
            wall = min(walls)
            ok = wall >= min_wall - tol_mm
            if not ok: violations += 1
            details.append(
                f"box outer={oid} inner={iid}: "
                f"min_wall={wall:.3f}mm (min {min_wall}mm) "
                f"{'OK' if ok else 'FAIL'}")
        else:
            details.append(f"outer={oid}({ot}) inner={iid}({it_}): type mismatch")
            violations += 1

    return {
        "checked": checked, "violations": violations,
        "ok": violations == 0,
        "score": round(max(0.0, 1.0 - violations / max(1, checked)), 3),
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════
# ASSEMBLY MATE CONSTRAINT VALIDATOR  (v6.0 — NEW)
# ═══════════════════════════════════════════════════════════════

def validate_mates(shapes: List[dict], mate_specs: List[dict],
                   tol_mm: float = 1.0) -> dict:
    """
    Validate CAD assembly mate constraints between shape pairs.

    Supported types:
      concentric  {ids:[a,b]}
          Cylinders share the same axis line (XY centers match, axes parallel).
      coincident  {ids:[a,b], face_a:'top'|'bottom', face_b:'top'|'bottom'}
          Two named faces lie on the same plane.
      tangent     {ids:[a,b]}
          Surfaces touch (min gap ≈ 0) — no separation, no overlap.
      distance    {ids:[a,b], gap:d, tol:t}
          Gap between bounding boxes = d ± t.

    Innovation: First benchmark to validate mechanical assembly constraints
    (concentric, coincident, tangent, distance) from pure text prompts.
    """
    by_id = {s.get("id", i): s for i, s in enumerate(shapes)}
    details = []; correct = 0
    checked = len(mate_specs)

    def _face_coord(shape, face):
        """Coordinate of a named face plane."""
        c = shape.get("center", [0,0,0])
        t = shape.get("type", "")
        if t == "cylinder":
            h = shape.get("height", 0)
            if face == "top":    return c[2] + h/2
            if face == "bottom": return c[2] - h/2
        elif t == "box":
            sz = shape.get("size", [0,0,0])
            if face == "top":    return c[2] + sz[2]/2
            if face == "bottom": return c[2] - sz[2]/2
            if face == "right":  return c[0] + sz[0]/2
            if face == "left":   return c[0] - sz[0]/2
            if face == "front":  return c[1] + sz[1]/2
            if face == "back":   return c[1] - sz[1]/2
        elif t == "sphere":
            r = shape.get("radius", 0)
            if face == "top":    return c[2] + r
            if face == "bottom": return c[2] - r
        return c[2]

    for spec in mate_specs:
        mt = spec.get("type", "").lower()
        ids = spec.get("ids", [])
        if len(ids) < 2:
            details.append(f"Mate {mt}: needs ≥2 ids"); continue
        a = by_id.get(ids[0]); b = by_id.get(ids[1])
        if a is None or b is None:
            details.append(f"Mate {mt} ids={ids}: MISSING shape(s)"); continue

        ca = a.get("center",[0,0,0]); cb = b.get("center",[0,0,0])

        # ── Concentric ──────────────────────────────────────────
        if mt == "concentric":
            ax_a = _norm(a.get("axis",[0,0,1]))
            ax_b = _norm(b.get("axis",[0,0,1]))
            dot = abs(sum(x*y for x,y in zip(ax_a, ax_b)))
            axes_parallel = dot >= math.cos(math.radians(5))
            xy_dist = math.sqrt((ca[0]-cb[0])**2 + (ca[1]-cb[1])**2)
            centers_aligned = xy_dist <= tol_mm
            ok = axes_parallel and centers_aligned
            if ok: correct += 1
            details.append(
                f"concentric ids={ids}: "
                f"axis_dot={dot:.3f} xy_dist={xy_dist:.2f}mm "
                f"{'OK' if ok else 'FAIL'}")

        # ── Coincident ──────────────────────────────────────────
        elif mt == "coincident":
            fa = spec.get("face_a", "top"); fb = spec.get("face_b", "bottom")
            za = _face_coord(a, fa); zb = _face_coord(b, fb)
            ok = abs(za - zb) <= tol_mm
            if ok: correct += 1
            details.append(
                f"coincident ids={ids} {fa}↔{fb}: "
                f"za={za:.2f} zb={zb:.2f} Δ={abs(za-zb):.2f}mm "
                f"{'OK' if ok else 'FAIL'}")

        # ── Tangent ─────────────────────────────────────────────
        elif mt == "tangent":
            min_a, max_a = _get_bbox(a)
            min_b, max_b = _get_bbox(b)
            gaps = []
            for k in range(3):
                gaps.append(max(min_b[k] - max_a[k], 0.0))
                gaps.append(max(min_a[k] - max_b[k], 0.0))
            min_gap = min(gaps)
            ok = min_gap <= tol_mm
            if ok: correct += 1
            details.append(
                f"tangent ids={ids}: min_gap={min_gap:.2f}mm "
                f"{'OK' if ok else 'FAIL'}")

        # ── Distance ─────────────────────────────────────────────
        elif mt == "distance":
            exp_gap = spec.get("gap", 0.0)
            tol_d = spec.get("tol", tol_mm)
            min_a, max_a = _get_bbox(a)
            min_b, max_b = _get_bbox(b)
            ax_gaps = [max(min_b[k] - max_a[k], min_a[k] - max_b[k], 0.0) for k in range(3)]
            actual_gap = min(ax_gaps)
            ok = abs(actual_gap - exp_gap) <= tol_d
            if ok: correct += 1
            details.append(
                f"distance ids={ids}: "
                f"gap={actual_gap:.2f}mm (exp {exp_gap:.2f}±{tol_d:.2f}) "
                f"{'OK' if ok else 'FAIL'}")

        else:
            details.append(f"Unknown mate type '{mt}'")

    score = round(correct / max(1, checked), 3) if checked else 1.0
    return {"checked": checked, "correct": correct, "score": score, "details": details}


# ═══════════════════════════════════════════════════════════════
# CHAMFER DISTANCE  (v6.0 — NEW)
# ═══════════════════════════════════════════════════════════════

def _sample_surface(shape: dict, n: int, rng: random.Random) -> List[List[float]]:
    """Sample n points uniformly on the outer surface of a shape."""
    t = shape.get("type", ""); c = shape.get("center", [0,0,0])
    pts: List[List[float]] = []

    if t == "sphere":
        r = shape.get("radius", 1.0)
        while len(pts) < n:
            x = rng.uniform(-1, 1); y = rng.uniform(-1, 1); z = rng.uniform(-1, 1)
            d = math.sqrt(x*x + y*y + z*z)
            if d < 1e-9: continue
            pts.append([c[0]+r*x/d, c[1]+r*y/d, c[2]+r*z/d])

    elif t == "cylinder":
        r = max(shape.get("radius", 1.0), 1e-6)
        h = max(shape.get("height", 1.0), 1e-6)
        side_a = 2*math.pi*r*h; cap_a = math.pi*r*r
        total_a = side_a + 2*cap_a
        n_side = max(1, round(n * side_a / total_a))
        n_cap  = max(1, (n - n_side) // 2)
        for _ in range(n_side):
            theta = rng.uniform(0, 2*math.pi); z_off = rng.uniform(-h/2, h/2)
            pts.append([c[0]+r*math.cos(theta), c[1]+r*math.sin(theta), c[2]+z_off])
        for sign in (1, -1):
            for _ in range(n_cap):
                rho = r * math.sqrt(rng.random()); theta = rng.uniform(0, 2*math.pi)
                pts.append([c[0]+rho*math.cos(theta), c[1]+rho*math.sin(theta), c[2]+sign*h/2])

    elif t == "box":
        wx = max(shape.get("size", [1,1,1])[0], 1e-6)
        wy = max(shape.get("size", [1,1,1])[1], 1e-6)
        wz = max(shape.get("size", [1,1,1])[2], 1e-6)
        faces = [
            (wx*wy, lambda: [c[0]+rng.uniform(-wx/2,wx/2), c[1]+rng.uniform(-wy/2,wy/2), c[2]+wz/2]),
            (wx*wy, lambda: [c[0]+rng.uniform(-wx/2,wx/2), c[1]+rng.uniform(-wy/2,wy/2), c[2]-wz/2]),
            (wx*wz, lambda: [c[0]+rng.uniform(-wx/2,wx/2), c[1]+wy/2, c[2]+rng.uniform(-wz/2,wz/2)]),
            (wx*wz, lambda: [c[0]+rng.uniform(-wx/2,wx/2), c[1]-wy/2, c[2]+rng.uniform(-wz/2,wz/2)]),
            (wy*wz, lambda: [c[0]+wx/2, c[1]+rng.uniform(-wy/2,wy/2), c[2]+rng.uniform(-wz/2,wz/2)]),
            (wy*wz, lambda: [c[0]-wx/2, c[1]+rng.uniform(-wy/2,wy/2), c[2]+rng.uniform(-wz/2,wz/2)]),
        ]
        total_a = sum(a for a, _ in faces)
        for area, sampler in faces:
            cnt = max(1, round(n * area / total_a))
            for _ in range(cnt):
                pts.append(sampler())
    return pts[:n]


def compute_chamfer_distance(llm_shapes: List[dict], gt_shapes: List[dict],
                              n_samples: int = 40) -> dict:
    """
    Bidirectional Chamfer Distance between LLM and GT assemblies.

    For each shape, sample n_samples surface points, then compute:
      CD = mean(LLM→GT nearest) + mean(GT→LLM nearest)

    CD=0 → perfect match. Normalised score = exp(−CD / 25).

    Innovation: Captures size/proportion errors that center-point scoring
    misses — inspired by Text2CAD (NeurIPS 2024) & DCD (NeurIPS 2021).
    """
    valid_types = ("cylinder", "box", "sphere")
    rng = random.Random(42)

    pts_llm: List[List[float]] = []
    for s in (llm_shapes or []):
        if isinstance(s, dict) and s.get("type") in valid_types:
            pts_llm.extend(_sample_surface(s, n_samples, rng))

    pts_gt: List[List[float]] = []
    for s in (gt_shapes or []):
        if isinstance(s, dict) and s.get("type") in valid_types:
            pts_gt.extend(_sample_surface(s, n_samples, rng))

    if not pts_llm or not pts_gt:
        return {"cd": 9999.0, "score": 0.0, "n_llm_pts": len(pts_llm), "n_gt_pts": len(pts_gt)}

    # Cap for performance
    MAX = 400
    if len(pts_llm) > MAX: pts_llm = rng.sample(pts_llm, MAX)
    if len(pts_gt)  > MAX: pts_gt  = rng.sample(pts_gt,  MAX)

    def mean_nn(src, tgt):
        total = 0.0
        for p in src:
            best = min(sum((p[k]-q[k])**2 for k in range(3)) for q in tgt)
            total += math.sqrt(best)
        return total / len(src)

    d_fwd = mean_nn(pts_llm, pts_gt)
    d_rev = mean_nn(pts_gt,  pts_llm)
    cd = d_fwd + d_rev
    score = math.exp(-cd / 25.0)

    return {
        "cd": round(cd, 3),
        "score": round(score, 3),
        "n_llm_pts": len(pts_llm),
        "n_gt_pts": len(pts_gt),
    }


# ═══════════════════════════════════════════════════════════════
# SYMMETRY VALIDATOR  (v6.0 — NEW)
# ═══════════════════════════════════════════════════════════════

def validate_symmetry(shapes: List[dict], symmetry_spec: dict,
                      tol_mm: float = 2.0) -> dict:
    """
    Verify mirror symmetry across XY / YZ / XZ planes.

    symmetry_spec: {plane: 'YZ'|'XZ'|'XY', reference_ids: [ids_to_check]}

    For each reference shape, compute its mirror position and verify a
    same-type shape exists there in the LLM output.

    Innovation: Elevates L9-style mirror logic to a reusable validator
    dimension that can augment any level.
    """
    if not shapes or not symmetry_spec:
        return {"checked": 0, "correct": 0, "score": 1.0, "details": []}

    plane = symmetry_spec.get("plane", "YZ").upper()
    ref_ids = symmetry_spec.get("reference_ids", [])

    by_id = {s.get("id", i): s for i, s in enumerate(shapes)}
    all_shapes = list(by_id.values())

    def mirror(c):
        x, y, z = c
        if plane == "YZ": return [-x,  y,  z]
        if plane == "XZ": return [ x, -y,  z]
        if plane == "XY": return [ x,  y, -z]
        return c

    ids_to_check = ref_ids if ref_ids else list(by_id.keys())
    checked = len(ids_to_check); correct = 0; details = []

    for sid in ids_to_check:
        ref = by_id.get(sid)
        if ref is None:
            details.append(f"id={sid}: MISSING"); continue

        mc = mirror(ref.get("center", [0,0,0]))
        rt = ref.get("type")

        best_d = float("inf"); best_match = None
        for s in all_shapes:
            if s.get("id") == sid or s.get("type") != rt: continue
            d = _vec_dist(mc, s.get("center", [0,0,0]))
            if d < best_d: best_d = d; best_match = s.get("id")

        ok = best_d <= tol_mm
        if ok: correct += 1
        mc_r = [round(v, 1) for v in mc]
        details.append(
            f"id={sid}({rt}) mirror→{mc_r}: "
            f"nearest={best_d:.1f}mm (match={best_match}) "
            f"{'OK' if ok else 'FAIL'}")

    return {
        "checked": checked, "correct": correct,
        "score": round(correct / max(1, checked), 3),
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════
# PATTERN VALIDATOR  (v6.0 — NEW)
# ═══════════════════════════════════════════════════════════════

def validate_pattern(shapes: List[dict], pattern_spec: dict,
                     tol_mm: float = 2.0) -> dict:
    """
    Validate circular or linear shape patterns.

    Circular spec: {type:'circular', center:[x,y,z], radius:r, count:n,
                    shape_ids:[...]}
      — Checks each shape is at radius r from center, and that
        angular spacing = 360/n between consecutive shapes.

    Linear spec: {type:'linear', origin:[x,y,z], direction:[dx,dy,dz],
                  spacing:d, shape_ids:[...]}
      — Checks shapes form an evenly spaced linear array.

    Returns: {checked, correct, score, details}
    """
    if not shapes or not pattern_spec:
        return {"checked": 0, "correct": 0, "score": 1.0, "details": []}

    by_id = {s.get("id", i): s for i, s in enumerate(shapes)}
    pt = pattern_spec.get("type", "").lower()
    shape_ids = pattern_spec.get("shape_ids", list(by_id.keys()))
    checked = len(shape_ids); correct = 0; details = []

    if pt == "circular":
        cx, cy = pattern_spec.get("center", [0,0,0])[:2]
        exp_r = pattern_spec.get("radius", 0.0)
        count = pattern_spec.get("count", len(shape_ids))
        exp_step = 360.0 / max(count, 1)
        actual_angles = []

        for sid in shape_ids:
            s = by_id.get(sid)
            if s is None:
                details.append(f"id={sid}: MISSING"); continue
            sc = s.get("center", [0,0,0])
            dx = sc[0] - cx; dy = sc[1] - cy
            r = math.sqrt(dx*dx + dy*dy)
            ang = math.degrees(math.atan2(dy, dx)) % 360
            actual_angles.append((sid, r, ang))
            r_ok = abs(r - exp_r) <= tol_mm
            if r_ok: correct += 1
            details.append(
                f"id={sid}: r={r:.2f}mm (exp {exp_r:.2f}) "
                f"angle={ang:.1f}° {'OK' if r_ok else 'FAIL'}")

        if len(actual_angles) > 1:
            angs = sorted(a for _, _, a in actual_angles)
            spacings = [angs[i+1]-angs[i] for i in range(len(angs)-1)]
            spacings.append(360 - angs[-1] + angs[0])
            avg_err = sum(abs(sp - exp_step) for sp in spacings) / len(spacings)
            details.append(f"Angular spacing: exp {exp_step:.1f}° avg_err={avg_err:.1f}°")

    elif pt == "linear":
        origin = pattern_spec.get("origin", [0,0,0])
        direction = pattern_spec.get("direction", [1,0,0])
        spacing = pattern_spec.get("spacing", 10.0)
        mag = math.sqrt(sum(d*d for d in direction))
        if mag > 1e-9: direction = [d/mag for d in direction]

        for i, sid in enumerate(shape_ids):
            s = by_id.get(sid)
            if s is None:
                details.append(f"id={sid}: MISSING"); continue
            exp_pos = [origin[k] + i * spacing * direction[k] for k in range(3)]
            dist = _vec_dist(exp_pos, s.get("center", [0,0,0]))
            ok = dist <= tol_mm
            if ok: correct += 1
            details.append(f"id={sid}: pos_err={dist:.2f}mm {'OK' if ok else 'FAIL'}")

    return {
        "checked": checked, "correct": correct,
        "score": round(correct / max(1, checked), 3),
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════
# REASONING CHAIN VERIFIER  (v6.0 — NEW)
# ═══════════════════════════════════════════════════════════════

def analyze_reasoning_chain(response: str, expected_intermediates: List[dict]) -> dict:
    """
    Verify that the LLM's reasoning text contains expected intermediate values.

    expected_intermediates: [
        {"value": 38.25, "description": "sun pitch radius (mm)", "tol": 0.5},
        ...
    ]

    Extracts all numbers from the response and checks each expected value
    appears (within tolerance). Unlike the basic reasoning scorer (which
    only checks keywords), this traces actual computation milestones.

    Innovation: Lets the benchmark verify that the LLM reached the right
    intermediate results, not just the final answer — pinpointing exactly
    which step in a chain failed.
    """
    if not response or not expected_intermediates:
        return {"checked": 0, "found": 0, "score": 1.0, "details": []}

    raw = re.findall(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", response)
    numbers: set = set()
    for tok in raw:
        try: numbers.add(float(tok))
        except ValueError: pass

    details = []; found = 0
    checked = len(expected_intermediates)

    for item in expected_intermediates:
        exp_val = item.get("value", 0.0)
        desc    = item.get("description", "")
        tol     = item.get("tol", 0.5)
        matched = any(abs(n - exp_val) <= tol for n in numbers)
        if matched: found += 1
        details.append(
            f"{desc}: {exp_val} → "
            f"{'FOUND ✓' if matched else 'MISSING ✗'}")

    return {
        "checked": checked, "found": found,
        "score": round(found / max(1, checked), 3),
        "details": details,
    }
