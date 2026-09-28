"""Shape normalization and coverage/geometry scoring module.

This module provides utilities for normalizing LLM-generated shape representations
into a canonical schema and computing coverage/geometry accuracy scores using
nearest-neighbor matching with strict scoring policies.
"""

import math


# ═══════════════════════════════════════════════════════════════
# SHAPE NORMALIZER — map LLM field variants to canonical schema
# ═══════════════════════════════════════════════════════════════

# LLM type names → canonical type
_TYPE_MAP = {
    "box": "box", "cube": "box", "cuboid": "box", "rectangular_prism": "box",
    "cylinder": "cylinder", "cyl": "cylinder",
    "sphere": "sphere", "ball": "sphere",
    "beam": "beam", "bar": "beam", "rod": "beam",
    "cone": "cone",
    "torus": "torus", "ring": "torus",
    "pipe": "pipe", "tube": "pipe", "hollow_cylinder": "pipe",
}

def _infer_type(s):
    """Guess shape type from fields if 'type' is missing."""
    if "radius" in s and "height" in s:
        return "cylinder"
    if "radius" in s:
        return "sphere"
    if "size" in s or "dimensions" in s or "dim" in s or "dims" in s:
        return "box"
    if "start" in s and "end" in s:
        return "beam"
    if "inner_radius" in s and "outer_radius" in s:
        return "pipe"
    if "ring_radius" in s or "tube_radius" in s:
        return "torus"
    if "start_radius" in s or "end_radius" in s:
        return "cone"
    if any(k in s for k in ("width", "w", "length", "l", "depth", "d")):
        return "box"
    return ""


def _to_float(v, default=0.0):
    """Safely cast a value to float — handles strings, ints, None."""
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

def _to_float_list(v, expected_len=3):
    """Safely cast a list/tuple/dict of values to float list.
    Handles: [1,2,3], {"x":1,"y":2,"z":3}, {"0":1,"1":2,"2":3}."""
    if isinstance(v, dict):
        # LLMs often return {"x":0, "y":5, "z":10} instead of [0,5,10]
        if "x" in v and "y" in v and "z" in v:
            try:
                return [float(v["x"]), float(v["y"]), float(v["z"])]
            except (ValueError, TypeError):
                return None
        # Or {"width":10, "height":5, "depth":3} for size
        if "width" in v or "length" in v:
            try:
                return [float(v.get("width", v.get("length", 0))),
                        float(v.get("depth", v.get("height", 0))),
                        float(v.get("height", v.get("depth", 0)))]
            except (ValueError, TypeError):
                return None
        return None
    if not isinstance(v, (list, tuple)) or len(v) < expected_len:
        return None
    try:
        return [float(x) for x in v[:expected_len]]
    except (ValueError, TypeError):
        return None


def _normalize_shape(s):
    """Convert LLM shape dict to canonical schema used by validators."""
    if not isinstance(s, dict):
        return None
    out = dict(s)  # shallow copy

    # Normalize type — infer if missing
    raw_type = str(out.get("type", "")).lower().strip()
    canonical_type = _TYPE_MAP.get(raw_type, raw_type)
    if not canonical_type:
        canonical_type = _infer_type(out)
    out["type"] = canonical_type

    # Normalize center: accept "position", "pos", "origin", "center", "location"
    # Also handle center_x/center_y/center_z, x/y/z, pos_x/pos_y/pos_z
    if "center" not in out:
        for key in ("position", "pos", "origin", "location", "centre"):
            if key in out:
                out["center"] = out[key]
                break
    if "center" not in out:
        # Try component fields
        cx = out.get("center_x", out.get("x", out.get("pos_x", None)))
        cy = out.get("center_y", out.get("y", out.get("pos_y", None)))
        cz = out.get("center_z", out.get("z", out.get("pos_z", None)))
        if cx is not None and cy is not None and cz is not None:
            out["center"] = [float(cx), float(cy), float(cz)]

    # ── Cast all numeric fields to float (LLMs sometimes return strings) ──
    if "center" in out:
        cast = _to_float_list(out["center"], 3)
        if cast:
            out["center"] = cast
    for num_key in ("radius", "height", "inner_radius", "outer_radius",
                    "ring_radius", "tube_radius", "major_radius", "minor_radius",
                    "base_radius", "top_radius", "start_radius", "end_radius",
                    "width", "length", "depth"):
        if num_key in out:
            out[num_key] = _to_float(out[num_key])
    if "size" in out:
        cast = _to_float_list(out["size"], 3)
        if cast:
            out["size"] = cast
    if "axis" in out:
        ax = out["axis"]
        # Handle string axis like "X", "Y", "Z", "+X", "-Z" etc.
        if isinstance(ax, str):
            _AXIS_MAP = {
                "x": [1,0,0], "+x": [1,0,0], "-x": [-1,0,0],
                "y": [0,1,0], "+y": [0,1,0], "-y": [0,-1,0],
                "z": [0,0,1], "+z": [0,0,1], "-z": [0,0,-1],
            }
            out["axis"] = _AXIS_MAP.get(ax.lower().strip(), [0,0,1])
        else:
            cast = _to_float_list(ax, 3)
            if cast:
                out["axis"] = cast
    for pt_key in ("start", "end"):
        if pt_key in out:
            cast = _to_float_list(out[pt_key], 3)
            if cast:
                out[pt_key] = cast

    # Normalize box dimensions: accept "dimensions", "dim", "size"
    if out["type"] == "box" and "size" not in out:
        for key in ("dimensions", "dim", "dims"):
            if key in out:
                val = out[key]
                cast = _to_float_list(val, 3)  # handles list, tuple, AND dict {"x":..}
                if cast:
                    out["size"] = cast
                break
        # Also accept width/height/depth or length/width/height as separate keys
        if "size" not in out:
            w = _to_float(out.get("width", out.get("w", 0)))
            h = _to_float(out.get("height", out.get("h", 0)))
            d = _to_float(out.get("depth", out.get("d", out.get("length", out.get("l", 0)))))
            if w or h or d:
                out["size"] = [w, d, h]

    # Normalize cylinder: accept "length" as "height"
    if out["type"] == "cylinder" and "height" not in out:
        for key in ("length", "h", "len"):
            if key in out:
                out["height"] = out[key]
                break

    # Normalize beam: accept "from"/"to" as "start"/"end"
    if out["type"] == "beam":
        if "start" not in out:
            for key in ("from", "from_point", "p1", "point1"):
                if key in out:
                    out["start"] = out[key]; break
        if "end" not in out:
            for key in ("to", "to_point", "p2", "point2"):
                if key in out:
                    out["end"] = out[key]; break

    # Normalize cone: accept "base_radius"/"top_radius" or "bottom_radius"/"radius_top"
    if out["type"] == "cone":
        if "base_radius" not in out:
            for key in ("bottom_radius", "radius_bottom", "radius_base", "start_radius", "radius"):
                if key in out:
                    out["base_radius"] = out[key]; break
        if "top_radius" not in out:
            for key in ("radius_top", "end_radius", "tip_radius"):
                if key in out:
                    out["top_radius"] = out[key]; break
        if "top_radius" not in out:
            out["top_radius"] = 0  # default to pointed cone
        if "height" not in out:
            for key in ("length", "h", "len"):
                if key in out:
                    out["height"] = out[key]; break

    # Normalize torus: accept "major_radius"/"minor_radius"
    if out["type"] == "torus":
        if "ring_radius" not in out:
            for key in ("major_radius", "R", "large_radius", "radius"):
                if key in out:
                    out["ring_radius"] = out[key]; break
        if "tube_radius" not in out:
            for key in ("minor_radius", "r", "small_radius", "cross_radius"):
                if key in out:
                    out["tube_radius"] = out[key]; break

    # Normalize pipe: accept "inner_radius"/"outer_radius" variants
    if out["type"] == "pipe":
        if "inner_radius" not in out:
            for key in ("radius_inner", "ri", "bore_radius"):
                if key in out:
                    out["inner_radius"] = out[key]; break
        if "outer_radius" not in out:
            for key in ("radius_outer", "ro"):
                if key in out:
                    out["outer_radius"] = out[key]; break
        if "height" not in out:
            for key in ("length", "h", "len"):
                if key in out:
                    out["height"] = out[key]; break

    return out


def _is_degenerate(s):
    """Return True if a normalized shape is degenerate (zero-size, NaN, or
    missing critical spatial data).  Degenerate shapes are noise — they should
    be discarded *before* scoring so they cannot inflate coverage or geometry."""
    import math as _m

    def _bad(v):
        """True if v is None, NaN, or effectively zero."""
        if v is None:
            return True
        try:
            f = float(v)
            return _m.isnan(f) or _m.isinf(f)
        except (ValueError, TypeError):
            return True

    def _bad_pt(pt):
        if not isinstance(pt, (list, tuple)) or len(pt) < 3:
            return True
        return any(_bad(x) for x in pt[:3])

    # Must have a resolvable center
    center = _shape_center(s)
    if center is None or _bad_pt(center):
        return True

    t = s.get("type", "")

    if t == "cylinder":
        r = s.get("radius", 0)
        h = s.get("height", 0)
        if _bad(r) or _bad(h) or float(r) <= 1e-9 or float(h) <= 1e-9:
            return True
    elif t == "box":
        sz = s.get("size", [0, 0, 0])
        if not isinstance(sz, (list, tuple)) or len(sz) < 3:
            return True
        if any(_bad(x) or abs(float(x)) <= 1e-9 for x in sz[:3]):
            return True
    elif t == "sphere":
        r = s.get("radius", 0)
        if _bad(r) or float(r) <= 1e-9:
            return True
    elif t == "beam":
        st = s.get("start")
        en = s.get("end")
        if _bad_pt(st) or _bad_pt(en):
            return True
        length = _dist3(st, en)
        if length <= 1e-9:
            return True
    elif t == "pipe":
        ir = s.get("inner_radius", 0)
        orr = s.get("outer_radius", 0)
        h = s.get("height", s.get("length", 0))
        if _bad(ir) or _bad(orr) or _bad(h):
            return True
        if float(orr) <= 1e-9 or float(h) <= 1e-9:
            return True
    elif t == "cone":
        br = s.get("base_radius", 0)
        h = s.get("height", 0)
        if _bad(br) or _bad(h) or float(br) <= 1e-9 or float(h) <= 1e-9:
            return True
    elif t == "torus":
        rr = s.get("ring_radius", 0)
        tr = s.get("tube_radius", 0)
        if _bad(rr) or _bad(tr) or float(rr) <= 1e-9 or float(tr) <= 1e-9:
            return True
    elif not t:
        # No recognizable type at all — degenerate
        return True

    return False


def _normalize_shapes(shapes):
    """Normalize a list of LLM shapes, discarding degenerate ones."""
    normed = []
    for s in shapes:
        n = _normalize_shape(s)
        if n and not _is_degenerate(n):
            normed.append(n)
    return normed


def _shape_center(s):
    """Extract position from a normalized shape."""
    if "center" in s:
        c = s["center"]
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            try:
                return [float(c[0]), float(c[1]), float(c[2])]
            except (ValueError, TypeError):
                return None
    if s.get("type") == "beam":
        st = s.get("start", [0,0,0])
        en = s.get("end", [0,0,0])
        try:
            return [(float(st[i])+float(en[i]))/2.0 for i in range(3)]
        except (ValueError, TypeError, IndexError):
            return None
    return None


def _dist3(a, b):
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))


def _assembly_scale(golden):
    """Compute the characteristic size of the golden assembly for relative tolerance."""
    all_pts = []
    for g in golden:
        c = _shape_center(g)
        if c:
            all_pts.append(c)
    if len(all_pts) < 2:
        return 10.0  # fallback
    mins = [min(p[i] for p in all_pts) for i in range(3)]
    maxs = [max(p[i] for p in all_pts) for i in range(3)]
    diag = math.sqrt(sum((maxs[i]-mins[i])**2 for i in range(3)))
    return max(diag, 1.0)


# ═══════════════════════════════════════════════════════════════
# SCORING — nearest-neighbor matching, shape-aware weights
# ═══════════════════════════════════════════════════════════════

def eval_cov_geom(llm_shapes_raw, golden_json):
    """Score Coverage and Geometry accuracy (all phases).

    Uses nearest-neighbor matching (not ID-based) so LLM shapes don't need
    matching IDs.  Tolerance is relative to the assembly's bounding diagonal.

    STRICT scoring policy (v2):
      - Excess shapes penalized (hallucination penalty)
      - Type mismatch → zero credit on type dimension
      - Steeper position falloff (1.5× tolerance → already 0)
      - Shapes outside the assembly's bounding region penalized
    """
    if not golden_json:
        return 0, 0

    llm_shapes = _normalize_shapes(llm_shapes_raw)

    # ── Coverage (with excess penalty) ────────────────────────
    normed_golden = _normalize_shapes(golden_json)
    expected = len(normed_golden) if normed_golden else len(golden_json)
    actual = len(llm_shapes)

    if expected == 0:
        return 0, 0

    # Base coverage: how many of the expected shapes are present
    base_cov = min(1.0, actual / float(expected))

    # Excess penalty: producing way too many shapes is hallucination, not accuracy.
    # Penalty kicks in when actual > 1.5× expected, and gets harsh above 2×.
    if actual > expected * 1.5:
        excess_ratio = actual / float(expected)
        # penalty: 1.0 at 1.5×, drops to 0.5 at 3×, 0.25 at 5×
        excess_penalty = 1.0 / (1.0 + (excess_ratio - 1.5))
        cov = base_cov * excess_penalty
    else:
        cov = base_cov

    # ── Geometry — two-pass nearest-neighbor matching (STRICT) ─────────
    # Pass 1: match type-identical pairs only.  This prevents shapes of
    # one type from stealing LLM slots that belong to a different type,
    # which would otherwise cause every GT sphere to get zero when the
    # model produced spheres-only but the GT has mostly beams.
    # Pass 2: for GT shapes still unmatched, fall back to any remaining
    # LLM shape (type-mismatch fallback, penalised heavily).
    scale = _assembly_scale(golden_json)
    pos_tol = max(2.0, scale * 0.05)

    geom_scores = [None] * len(golden_json)
    used_llm = set()
    unmatched_gt = []   # GT indices that found no same-type partner in pass 1

    # ── Pass 1: same-type matching ──────────────────────────────
    for gi, gt in enumerate(golden_json):
        gt_center = _shape_center(gt)
        if gt_center is None:
            geom_scores[gi] = 0.0
            continue
        gt_type = gt.get("type", "")

        best_idx, best_dist = None, float('inf')
        for li, ls in enumerate(llm_shapes):
            if li in used_llm:
                continue
            if ls.get("type") != gt_type:
                continue
            lc = _shape_center(ls)
            if lc is None:
                continue
            d = _dist3(gt_center, lc)
            if d < best_dist:
                best_idx, best_dist = li, d

        if best_idx is None:
            unmatched_gt.append(gi)   # will retry in pass 2
        else:
            used_llm.add(best_idx)
            ls = llm_shapes[best_idx]
            pos_score = max(0.0, 1.0 - best_dist / (1.5 * pos_tol))
            dim_score = _eval_dimensions(ls, gt)
            geom_scores[gi] = pos_score * 0.4 + 1.0 * 0.2 + dim_score * 0.4

    # ── Pass 2: type-mismatch fallback for unmatched GT shapes ──
    for gi in unmatched_gt:
        gt = golden_json[gi]
        gt_center = _shape_center(gt)
        if gt_center is None:
            geom_scores[gi] = 0.0
            continue

        best_idx, best_dist = None, float('inf')
        for li, ls in enumerate(llm_shapes):
            if li in used_llm:
                continue
            lc = _shape_center(ls)
            if lc is None:
                continue
            d = _dist3(gt_center, lc)
            if d < best_dist:
                best_idx, best_dist = li, d

        if best_idx is None:
            geom_scores[gi] = 0.0
        else:
            used_llm.add(best_idx)
            ls = llm_shapes[best_idx]
            # Type mismatch: pos credit only (0.4 weight), no type or dim credit
            pos_score = max(0.0, 1.0 - best_dist / (1.5 * pos_tol))
            geom_scores[gi] = pos_score * 0.4

    geom_scores = [s for s in geom_scores if s is not None]
    geom = (sum(geom_scores) / len(geom_scores)) if geom_scores else 0.0

    return round(cov * 100), round(geom * 100)


def _eval_dimensions(llm_s, gt_s):
    """Shape-type-aware dimension comparison. Returns 0.0–1.0 (1.0 = perfect)."""
    gt_type = gt_s.get("type", "")
    errors = []

    def rel_err(a, b):
        a, b = float(a or 0), float(b or 0)
        if b < 1e-9:
            return 0.0 if a < 1e-9 else 1.0
        return min(1.0, abs(a - b) / b)

    if gt_type == "cylinder":
        errors.append(rel_err(llm_s.get("radius"), gt_s.get("radius")))
        errors.append(rel_err(llm_s.get("height"), gt_s.get("height")))
    elif gt_type == "box":
        gt_size = gt_s.get("size", [0,0,0])
        lm_size = llm_s.get("size", [0,0,0])
        if isinstance(lm_size, (list,tuple)) and len(lm_size) >= 3:
            gs = sorted([abs(x) for x in gt_size], reverse=True)
            ls = sorted([abs(x) for x in lm_size], reverse=True)
            for i in range(3):
                errors.append(rel_err(ls[i], gs[i]))
        else:
            errors.append(1.0)
    elif gt_type == "sphere":
        errors.append(rel_err(llm_s.get("radius"), gt_s.get("radius")))
    elif gt_type == "beam":
        gt_s_pt = gt_s.get("start", [0,0,0])
        gt_e_pt = gt_s.get("end", [0,0,0])
        lm_s_pt = llm_s.get("start", [0,0,0])
        lm_e_pt = llm_s.get("end", [0,0,0])
        gt_len = _dist3(gt_s_pt, gt_e_pt)
        lm_len = _dist3(lm_s_pt, lm_e_pt)
        if gt_len > 1e-9:
            errors.append(rel_err(lm_len, gt_len))
        errors.append(rel_err(llm_s.get("width", llm_s.get("thickness", 0)),
                              gt_s.get("width", gt_s.get("thickness", 0))))
        if gt_len > 1e-9 and lm_len > 1e-9:
            gt_dir = [(gt_e_pt[i]-gt_s_pt[i])/gt_len for i in range(3)]
            lm_dir = [(lm_e_pt[i]-lm_s_pt[i])/lm_len for i in range(3)]
            dot = abs(sum(gt_dir[i]*lm_dir[i] for i in range(3)))
            errors.append(1.0 - min(1.0, dot))
    elif gt_type == "pipe":
        errors.append(rel_err(llm_s.get("inner_radius"), gt_s.get("inner_radius")))
        errors.append(rel_err(llm_s.get("outer_radius"), gt_s.get("outer_radius")))
        errors.append(rel_err(llm_s.get("height", llm_s.get("length")),
                              gt_s.get("height", gt_s.get("length"))))
    elif gt_type == "cone":
        errors.append(rel_err(llm_s.get("base_radius"), gt_s.get("base_radius")))
        errors.append(rel_err(llm_s.get("top_radius"), gt_s.get("top_radius")))
        errors.append(rel_err(llm_s.get("height"), gt_s.get("height")))
    elif gt_type == "torus":
        errors.append(rel_err(llm_s.get("ring_radius"), gt_s.get("ring_radius")))
        errors.append(rel_err(llm_s.get("tube_radius"), gt_s.get("tube_radius")))

    if not errors:
        return 0.5
    return max(0.0, 1.0 - sum(errors) / len(errors))
