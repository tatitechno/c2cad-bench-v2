#!/usr/bin/env python3
"""
C2CAD-Bench unified live benchmark runner
==============================================
Runs any registered LLM against any test at any scale, scores the output,
and writes results into showcase_db.js for the HTML visualiser.

Usage examples:
    # Run ALL tests × ALL default models
    python run_unified.py --all

    # Run all tests for a single model (exact API model ID)
    python run_unified.py --all --model claude-sonnet-4-6

    # Run one test family at one scale for one model
    python run_unified.py --test "Planetary" --scale 12 --model deepseek-chat

    # Run a full phase for two models
    python run_unified.py --phase 2 --model claude-opus-4-6 --model deepseek-reasoner

    # Add a model to an existing DB (only fills missing slots)
    python run_unified.py --all --model claude-opus-4-6 --skip-existing

    # Redo only zero-score (failed/crashed) tests for a model
    python run_unified.py --all --model deepseek-chat --redo

    # Mark all un-run tests as zero (no API calls)
    python run_unified.py --all --model gpt-5.4 --zero-missing

    # List every supported model string
    python run_unified.py --list-models
"""

import sys
import os
import json
import time
import math
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = os.path.dirname(os.path.abspath(__file__))
# V2 stage paths MUST come before probe root to avoid V1 generator shadowing
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '../stages/phase1_basic')))
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '../stages/phase2_advanced')))
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '../stages/phase3_semantic')))
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '../stages/phase4_bioinspired')))
sys.path.append(os.path.abspath(os.path.join(base_dir, '..')))

from probe.llm import call_llm, list_models, GEMINI_ALIASES, CLAUDE_ALIASES, OPENAI_ALIASES, DEEPSEEK_ALIASES, MISTRAL_ALIASES, GROQ_ALIASES, KIMI_ALIASES
from probe.validators import extract_json, validate_geometry, validate_gravity, validate_connectivity, validate_interference
from probe.validators import validate_wall_thickness, validate_clearance_fit, validate_mates

from generate_staircase import generate_staircase
from generate_pyramid import generate_pyramid
from generate_rubiks import generate_rubiks
from generate_stonehenge import generate_stonehenge
from generate_dna import generate_dna
from generate_bridge_v2 import generate_bridge
from generate_planetary_v2 import generate_planetary
from generate_truss_v2 import generate_truss
from generate_fractal_v2 import generate_fractal
from generate_bcc_v2 import generate_bcc
from generate_human_furniture import generate_furniture
from generate_human_manifold import generate_manifold
from generate_human_axle import generate_axle
from generate_phyllotaxis import generate_phyllotaxis
from generate_compound_eye import generate_compound_eye
from generate_diatom import generate_diatom
from generate_honeycomb import generate_honeycomb
from generate_radiolarian import generate_radiolarian
from generate_vertebral import generate_vertebral
from generate_flange_bolt_circle import generate_flange
from generate_ball_bearing import generate_ball_bearing
from generate_human_clock import generate_clock
from generate_human_gantry import generate_gantry
from generate_cochlea import generate_cochlea
from generate_radiolarian_skeleton import generate_radiolarian as generate_radiolarian_skel


# ═══════════════════════════════════════════════════════════════
# MODEL DEFAULTS  (used when --all is invoked without --model)
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODELS = [
    # Exact API model IDs — no alias translation needed
    # Gemini (Google Generative Language API)
    "gemini-2.5-flash",                # Gemini 2.5 Flash
    "gemini-2.5-pro",                  # Gemini 2.5 Pro
    "gemini-3.1-flash-lite-preview",   # Gemini 3.1 Flash-Lite
    "gemini-3-flash-preview",          # Gemini 3 Flash
    "gemini-3.1-pro-preview",          # Gemini 3.1 Pro
    # Claude (Anthropic Messages API)
    "claude-opus-4-6",                 # Claude Opus 4.6
    "claude-sonnet-4-6",               # Claude Sonnet 4.6
    # DeepSeek (DeepSeek API)
    # NOTE: deepseek-chat   → DeepSeek-V3.2 (non-thinking)
    #       deepseek-reasoner → DeepSeek-R1  (thinking / chain-of-thought)
    "deepseek-chat",                   # DeepSeek V3.2
    "deepseek-reasoner",               # DeepSeek R1
    # OpenAI (OpenAI API — requires OPENAI_API_KEY)
    "gpt-4.1",                         # GPT-4.1
    "gpt-5.4",                         # GPT-5.4
    "gpt-5.4-mini",                    # GPT-5.4 Mini
    # Kimi (Moonshot AI — requires MOONSHOT_API_KEY)
    "kimi-k2.5",                       # Kimi K2.5
]

# All recognised model aliases (used for --model validation)
ALL_KNOWN = set()
ALL_KNOWN.update(GEMINI_ALIASES.keys())
ALL_KNOWN.update(CLAUDE_ALIASES.keys())
ALL_KNOWN.update(OPENAI_ALIASES.keys())
ALL_KNOWN.update(DEEPSEEK_ALIASES.keys())
ALL_KNOWN.update(MISTRAL_ALIASES.keys())
ALL_KNOWN.update(GROQ_ALIASES.keys())
ALL_KNOWN.update(KIMI_ALIASES.keys())
# Supplement: ensure DEFAULT_MODELS are always accepted even if
# probe aliases haven't been refreshed
ALL_KNOWN.update(DEFAULT_MODELS)


# ═══════════════════════════════════════════════════════════════
# TEST REGISTRY
# ═══════════════════════════════════════════════════════════════

ALL_TESTS = [
    {"family": "Spiral Staircase",   "func": generate_staircase, "scales": [10, 24, 50],   "phase": 1},
    {"family": "Cannonball Pyramid",  "func": generate_pyramid,   "scales": [3, 4, 5],      "phase": 1},
    {"family": "Voxel Grid",          "func": generate_rubiks,    "scales": [2, 3, 4],      "phase": 1},
    {"family": "Domino Ring",         "func": generate_stonehenge,"scales": [5, 10, 20],    "phase": 1},
    {"family": "DNA Helix",           "func": generate_dna,       "scales": [5, 10, 20],    "phase": 1},
    {"family": "Suspension Bridge",   "func": generate_bridge,    "scales": [5, 10, 20],    "phase": 2},
    {"family": "Planetary Array",     "func": generate_planetary,  "scales": [6, 12, 18],    "phase": 2},
    {"family": "Cross-Braced Truss",  "func": generate_truss,     "scales": [2, 4, 8],      "phase": 2},
    {"family": "Fractal Y-Tree",      "func": generate_fractal,   "scales": [2, 3, 4],      "phase": 2},
    {"family": "BCC Lattice",         "func": generate_bcc,       "scales": [2, 3, 4],      "phase": 2},
    {"family": "Furniture Assembly",  "func": generate_furniture,  "scales": [2, 3, 4],      "phase": 3},
    {"family": "Pipe Manifold",       "func": generate_manifold,   "scales": [2, 3, 5],      "phase": 3},
    {"family": "Axle Bearing",         "func": generate_axle,       "scales": [1, 2, 3],      "phase": 3},
    {"family": "Phyllotaxis Disc",    "func": generate_phyllotaxis,   "scales": [21, 55, 89],   "phase": 4},
    {"family": "Compound Eye",        "func": generate_compound_eye,  "scales": [2, 3, 4],      "phase": 4},
    {"family": "Diatom Frustule",     "func": generate_diatom,        "scales": [4, 7, 10],     "phase": 4},
    {"family": "Honeycomb Lattice",  "func": generate_honeycomb,     "scales": [1, 2, 3],      "phase": 4},
    {"family": "Armillary Sphere",   "func": generate_radiolarian,   "scales": [2, 3, 4],      "phase": 3},
    {"family": "Vertebral Column",   "func": generate_vertebral,     "scales": [7, 12, 19],    "phase": 4},
    {"family": "Flanged Pipe Joint", "func": generate_flange,        "scales": [4, 8, 16],     "phase": 1},
    {"family": "Ball Bearing Assembly","func": generate_ball_bearing, "scales": [6, 12, 20],    "phase": 2},
    {"family": "Clock Tower Mechanism","func": generate_clock,        "scales": [4, 12, 24],    "phase": 3},
    {"family": "Gantry Crane Assembly","func": generate_gantry,       "scales": [2, 4, 6],      "phase": 3},
    {"family": "Cochlear Spiral",     "func": generate_cochlea,       "scales": [12, 24, 36],   "phase": 4},
    {"family": "Radiolarian Skeleton","func": generate_radiolarian_skel,"scales": [1, 2, 3],    "phase": 4},
]


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


# ═══════════════════════════════════════════════════════════════
# SEMANTIC SCORING — subsumes Physics + structural constraints
# ═══════════════════════════════════════════════════════════════
# Each family defines weighted checks. Physics (gravity, connectivity,
# interference) are base checks; family-specific structural rules sit
# on top. Weights are tuned per family so the score reflects what
# matters most for that particular assembly.

def _apply_physics_gate(raw, shapes, tol_mm=3.0):
    """Universal physics gate — crush semantic when basic physics are violated.

    Computes avg(gravity, connectivity). If < 0.4, multiplies raw score by
    (physics_avg / 0.4), smoothly driving it toward 0 for physically invalid
    outputs. Evaluators can opt in by calling this on their final raw score.
    """
    if not shapes or len(shapes) < 2:
        return 0.0
    g = _phys_gravity(shapes, tol_mm=tol_mm)
    c = _phys_connectivity(shapes, tol_mm=tol_mm)
    physics_avg = (g + c) / 2.0
    if physics_avg < 0.4:
        return raw * (physics_avg / 0.4)
    return raw


def _phys_gravity(shapes, tol_mm=2.0):
    """Proportion of shapes that are gravity-supported. 0.0–1.0."""
    if not shapes:
        return 0.0
    res = validate_gravity(shapes, tol_mm=tol_mm)
    n = max(1, len(shapes))
    unsup = len(res.get("floating_ids", []))
    return max(0.0, 1.0 - unsup / n)

def _phys_connectivity(shapes, tol_mm=2.0):
    """1/num_components — fully connected=1.0. 0.0–1.0."""
    if not shapes:
        return 0.0
    res = validate_connectivity(shapes, tol_mm=tol_mm)
    n_comp = res.get("islands", 1)
    return 1.0 / max(1, n_comp)

def _phys_interference(shapes, tol_mm=1.0):
    """Proportion of non-interfering pairs. 0.0–1.0."""
    if not shapes or len(shapes) < 2:
        return 1.0
    res = validate_interference(shapes, tol_mm=tol_mm)
    n_interf = len(res.get("overlapping_pairs", []))
    n_pairs = max(1, len(shapes) * (len(shapes) - 1) // 2)
    return max(0.0, 1.0 - n_interf / n_pairs)

def _check_uniform_z_spacing(shapes_subset):
    """How uniformly spaced are shapes in Z? 0.0–1.0."""
    centers = [_shape_center(s) for s in shapes_subset]
    z_vals = sorted([c[2] for c in centers if c])
    if len(z_vals) < 3:
        return 1.0
    gaps = [z_vals[i+1] - z_vals[i] for i in range(len(z_vals)-1)]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap < 0.1:
        return 0.0
    dev = sum(abs(g - mean_gap) / mean_gap for g in gaps) / len(gaps)
    return max(0.0, 1.0 - dev)

def _check_angular_regularity(shapes_subset, cx=0.0, cy=0.0):
    """How evenly are shapes distributed angularly around (cx,cy)? 0.0–1.0."""
    angles = []
    for s in shapes_subset:
        c = _shape_center(s)
        if c:
            dx, dy = c[0] - cx, c[1] - cy
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                angles.append(math.atan2(dy, dx))
    if len(angles) < 2:
        return 1.0
    angles.sort()
    n = len(angles)
    gaps = []
    for i in range(n):
        g = angles[(i+1) % n] - angles[i]
        if g <= 0:
            g += 2 * math.pi
        gaps.append(g)
    mean_gap = 2 * math.pi / n
    dev = sum(abs(g - mean_gap) / mean_gap for g in gaps) / n
    return max(0.0, 1.0 - dev)

def _check_coplanar_z(shapes_subset):
    """Are all shapes at the same Z? 0.0–1.0."""
    z_vals = []
    for s in shapes_subset:
        c = _shape_center(s)
        if c:
            z_vals.append(c[2])
    if len(z_vals) < 2:
        return 1.0
    spread = max(z_vals) - min(z_vals)
    return max(0.0, 1.0 - spread / max(1.0, abs(sum(z_vals)/len(z_vals))))

def _check_constant_radius(shapes_subset, cx=0.0, cy=0.0):
    """Are all shapes at the same XY distance from (cx,cy)? 0.0–1.0."""
    radii = []
    for s in shapes_subset:
        c = _shape_center(s)
        if c:
            r = math.sqrt((c[0]-cx)**2 + (c[1]-cy)**2)
            radii.append(r)
    if len(radii) < 2:
        return 1.0
    mean_r = sum(radii) / len(radii)
    if mean_r < 0.1:
        return 1.0
    dev = sum(abs(r - mean_r) / mean_r for r in radii) / len(radii)
    return max(0.0, 1.0 - dev)

def _check_uniform_sizes(shapes_subset, field="radius"):
    """Are all shapes the same size for a given field? 0.0–1.0."""
    vals = [s.get(field, 0) for s in shapes_subset if s.get(field)]
    if len(vals) < 2:
        return 1.0
    mean_v = sum(vals) / len(vals)
    if mean_v < 1e-9:
        return 1.0
    dev = sum(abs(v - mean_v) / mean_v for v in vals) / len(vals)
    return max(0.0, 1.0 - dev)

def _check_beam_connects_pair(beam, shape_a, shape_b, tol_mm=3.0):
    """Does this beam connect shape_a center to shape_b center? 0.0 or 1.0."""
    bs = beam.get("start", [0,0,0])
    be = beam.get("end", [0,0,0])
    ca = _shape_center(shape_a)
    cb = _shape_center(shape_b)
    if not ca or not cb:
        return 0.0
    # Check both orientations
    d1 = _dist3(bs, ca) + _dist3(be, cb)
    d2 = _dist3(bs, cb) + _dist3(be, ca)
    return 1.0 if min(d1, d2) <= tol_mm * 2 else 0.0

def _check_planarity_y0(shapes_subset):
    """Are all shapes in the Y=0 plane? 0.0–1.0."""
    y_devs = []
    for s in shapes_subset:
        c = _shape_center(s)
        if c:
            y_devs.append(abs(c[1]))
        if s.get("type") == "beam":
            for pt in [s.get("start"), s.get("end")]:
                if pt:
                    y_devs.append(abs(pt[1]))
    if not y_devs:
        return 1.0
    max_dev = max(y_devs)
    return max(0.0, 1.0 - max_dev / 5.0)  # penalize if Y > 5mm

def _check_tangent_spheres(shapes_subset, expected_distance=None):
    """Are adjacent spheres tangent? Average tangency quality. 0.0–1.0."""
    spheres = [s for s in shapes_subset if s.get("type") == "sphere"]
    if len(spheres) < 2:
        return 1.0
    R = spheres[0].get("radius", 1.0)
    exp_d = expected_distance or (2.0 * R)
    # Find nearest-neighbor distances
    scores = []
    for i, s1 in enumerate(spheres):
        c1 = _shape_center(s1)
        if not c1:
            continue
        min_d = float('inf')
        for j, s2 in enumerate(spheres):
            if i == j:
                continue
            c2 = _shape_center(s2)
            if c2:
                d = _dist3(c1, c2)
                min_d = min(min_d, d)
        if min_d < float('inf'):
            err = abs(min_d - exp_d) / exp_d
            scores.append(max(0.0, 1.0 - err))
    return sum(scores) / len(scores) if scores else 1.0


# ── Per-family semantic evaluators ────────────────────────────

def _sem_staircase(shapes, golden, scale):
    """Spiral Staircase: pillar+steps spiral with uniform rise and rotation.

    Physics-first scoring (v2): gravity, connectivity, and radial attachment
    are prerequisites — not optional bonuses. A staircase with floating,
    disconnected beams that happen to be evenly spaced is NOT a valid staircase.

    Physics gate: if the average of gravity + connectivity + radial < 0.4,
    the entire semantic score is multiplied by (physics_avg / 0.4), crushing
    outputs that fail basic structural requirements regardless of pattern quality.
    """
    beams = [s for s in shapes if s.get("type") == "beam"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    tol = max(3.0, scale * 0.3)

    # ── Physics checks (dominant: 60% total weight) ──────────
    gravity_s      = _phys_gravity(shapes)
    connectivity_s = _phys_connectivity(shapes, tol_mm=tol)
    interference_s = _phys_interference(shapes)

    # Radial attachment: beam starts near pillar surface
    if cylinders and beams:
        pillar = max(cylinders, key=lambda s: s.get("height", 0))
        pr = pillar.get("radius", 5.0)
        pc = _shape_center(pillar) or [0,0,0]
        errs = []
        for b in beams:
            st = b.get("start", [0,0,0])
            d_xy = math.sqrt((st[0]-pc[0])**2 + (st[1]-pc[1])**2)
            errs.append(min(1.0, abs(d_xy - pr) / max(pr, 1)))
        radial_s = max(0, 1.0 - sum(errs)/len(errs)) if errs else 0.0
    else:
        radial_s = 0.0

    # ── Pattern checks (secondary: 40% total weight) ─────────
    z_spacing_s = _check_uniform_z_spacing(beams)

    beam_data = []
    for b in beams:
        c = _shape_center(b)
        if c and (abs(c[0]) > 0.1 or abs(c[1]) > 0.1):
            beam_data.append((c[2], math.atan2(c[1], c[0])))
    beam_data.sort()
    if len(beam_data) >= 3:
        diffs = []
        for i in range(len(beam_data)-1):
            d = beam_data[i+1][1] - beam_data[i][1]
            while d > math.pi: d -= 2*math.pi
            while d < -math.pi: d += 2*math.pi
            diffs.append(abs(d))
        mean_d = sum(diffs)/len(diffs) if diffs else 0
        dev = sum(abs(x - mean_d)/max(mean_d, 0.01) for x in diffs)/len(diffs) if diffs and mean_d > 0.01 else 1.0
        angle_reg_s = max(0, 1.0 - dev)
    else:
        angle_reg_s = 0.0

    # ── Weighted sum ─────────────────────────────────────────
    raw = (gravity_s      * 0.20 +
           connectivity_s * 0.15 +
           radial_s       * 0.25 +
           interference_s * 0.10 +
           z_spacing_s    * 0.15 +
           angle_reg_s    * 0.15)

    # ── Physics gate ─────────────────────────────────────────
    # If gravity + connectivity + radial are poor, the structure is physically
    # invalid — crush the score regardless of pattern quality.
    physics_avg = (gravity_s + connectivity_s + radial_s) / 3.0
    if physics_avg < 0.4:
        raw *= physics_avg / 0.4

    return raw

def _sem_pyramid(shapes, golden, scale):
    """Cannonball Pyramid: dense-packed sphere stacking."""
    spheres = [s for s in shapes if s.get("type") == "sphere"]
    w = {}
    w['gravity']      = (_phys_gravity(shapes, tol_mm=12.0), 0.20)
    w['interference']  = (_phys_interference(shapes, tol_mm=2.0), 0.10)
    w['tangency']     = (_check_tangent_spheres(shapes), 0.25)
    z_vals = sorted(set(round(_shape_center(s)[2], 1) for s in spheres if _shape_center(s)))
    expected_layers = scale
    layer_score = min(1.0, len(z_vals) / max(1, expected_layers)) if z_vals else 0.0
    w['layers']       = (layer_score, 0.20)
    w['size_uniform'] = (_check_uniform_sizes(spheres, "radius"), 0.25)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=12.0)

def _sem_voxel(shapes, golden, scale):
    """Voxel Grid: regular 3D box lattice with uniform gaps."""
    boxes = [s for s in shapes if s.get("type") == "box"]
    w = {}
    w['gravity']      = (_phys_gravity(shapes), 0.15)
    w['interference']  = (_phys_interference(shapes, tol_mm=0.5), 0.10)
    for axis_name, axis_idx in [('x', 0), ('y', 1), ('z', 2)]:
        vals = sorted(set(round(_shape_center(b)[axis_idx], 1) for b in boxes if _shape_center(b)))
        if len(vals) >= 2:
            gaps = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
            mean_g = sum(gaps)/len(gaps)
            dev = sum(abs(g-mean_g)/max(mean_g, 0.01) for g in gaps)/len(gaps) if mean_g > 0.01 else 1.0
            w[f'grid_{axis_name}'] = (max(0, 1.0 - dev), 0.20)
        else:
            w[f'grid_{axis_name}'] = (0.5, 0.20)
    if boxes:
        sizes = [tuple(sorted(s.get("size", [0,0,0]))) for s in boxes if s.get("size")]
        if sizes:
            ref = sizes[0]
            matches = sum(1 for s in sizes if all(abs(s[i]-ref[i]) < 0.5 for i in range(3)))
            w['size_uniform'] = (matches / len(sizes), 0.15)
        else:
            w['size_uniform'] = (0.0, 0.15)
    else:
        w['size_uniform'] = (0.0, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes)

def _sem_stonehenge(shapes, golden, scale):
    """Domino Ring / Stonehenge: circular archway array."""
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    w['gravity']      = (_phys_gravity(shapes), 0.18)
    w['connectivity']  = (_phys_connectivity(shapes, tol_mm=5.0), 0.17)
    w['interference']  = (_phys_interference(shapes), 0.08)
    # Circular distribution of pillars
    w['angular_reg']  = (_check_angular_regularity(cylinders), 0.20)
    # Constant radius from center
    w['const_radius'] = (_check_constant_radius(cylinders), 0.15)
    # Header beams connect pillar tops — check beams at correct Z
    if beams and cylinders:
        pillar_top_z = max(c.get("height", 0) for c in cylinders)
        beam_z_scores = []
        for b in beams:
            bstart = b.get("start", [0,0,0])
            bend = b.get("end", [0,0,0])
            avg_z = (bstart[2] + bend[2]) / 2.0
            beam_z_scores.append(max(0, 1.0 - abs(avg_z - pillar_top_z) / max(pillar_top_z, 1)))
        w['header_z'] = (sum(beam_z_scores)/len(beam_z_scores) if beam_z_scores else 0.0, 0.15)
    else:
        w['header_z'] = (0.0, 0.15)
    w['pillar_size']  = (_check_uniform_sizes(cylinders, "radius"), 0.12)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=5.0)

def _sem_dna(shapes, golden, scale):
    """DNA Helix: double-helix with uniform pitch and anti-parallel pairing."""
    spheres = [s for s in shapes if s.get("type") == "sphere"]
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    # DNA has intentional close contacts at rungs — relaxed tolerance
    w['interference']  = (_phys_interference(shapes, tol_mm=2.0), 0.10)
    # Uniform Z pitch for spheres
    w['z_pitch']      = (_check_uniform_z_spacing(spheres), 0.20)
    # Backbone radius consistency
    w['const_radius'] = (_check_constant_radius(spheres), 0.20)
    # Anti-parallel pairing: spheres at same Z should be ~180° apart
    z_groups = {}
    for s in spheres:
        c = _shape_center(s)
        if c:
            zk = round(c[2], 1)
            z_groups.setdefault(zk, []).append(c)
    pair_scores = []
    for zk, pts in z_groups.items():
        if len(pts) == 2:
            a1 = math.atan2(pts[0][1], pts[0][0])
            a2 = math.atan2(pts[1][1], pts[1][0])
            diff = abs(a1 - a2)
            diff = min(diff, 2*math.pi - diff)
            pair_scores.append(max(0, 1.0 - abs(diff - math.pi) / math.pi))
    w['antiparallel'] = (sum(pair_scores)/len(pair_scores) if pair_scores else 0.0, 0.20)
    # Sphere size uniformity
    w['size_uniform'] = (_check_uniform_sizes(spheres, "radius"), 0.10)
    # Rung beams connect sphere pairs at same Z
    if beams and spheres:
        rung_ok = 0
        for b in beams:
            bs, be = b.get("start", [0,0,0]), b.get("end", [0,0,0])
            if abs(bs[2] - be[2]) < 2.0:  # horizontal rung
                rung_ok += 1
        w['rung_horiz'] = (rung_ok / max(1, len(beams)), 0.20)
    else:
        w['rung_horiz'] = (0.0, 0.20)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=3.0)

def _sem_bridge(shapes, golden, scale):
    """Suspension Bridge: deck + towers + symmetric cables."""
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    w['gravity']      = (_phys_gravity(shapes), 0.18)
    w['connectivity']  = (_phys_connectivity(shapes, tol_mm=5.0), 0.17)
    w['interference']  = (_phys_interference(shapes), 0.10)
    # Deck colinearity: longest beam should be ~horizontal
    if beams:
        deck = max(beams, key=lambda b: _dist3(b.get("start",[0,0,0]), b.get("end",[0,0,0])))
        ds, de = deck.get("start", [0,0,0]), deck.get("end", [0,0,0])
        z_diff = abs(ds[2] - de[2])
        horiz_len = math.sqrt((ds[0]-de[0])**2 + (ds[1]-de[1])**2)
        w['deck_horiz'] = (max(0, 1.0 - z_diff / max(horiz_len, 1)), 0.15)
    else:
        w['deck_horiz'] = (0.0, 0.15)
    # Tower grounding: cylinders base at Z=0
    if cylinders:
        grounded = sum(1 for c in cylinders
                       if (_shape_center(c) or [0,0,0])[2] - c.get("height",0)/2 < 2.0)
        w['tower_ground'] = (grounded / len(cylinders), 0.15)
    else:
        w['tower_ground'] = (0.0, 0.15)
    # Cable symmetry: check L/R cable count balance
    cable_beams = [b for b in beams if b != max(beams, key=lambda b: _dist3(b.get("start",[0,0,0]), b.get("end",[0,0,0])))] if beams else []
    left_cables = [b for b in cable_beams if (_shape_center(b) or [0,0,0])[0] < 0]
    right_cables = [b for b in cable_beams if (_shape_center(b) or [0,0,0])[0] > 0]
    nl, nr = len(left_cables), len(right_cables)
    sym = 1.0 - abs(nl - nr) / max(nl + nr, 1)
    w['cable_sym'] = (sym, 0.15)
    # Cable endpoints evenly spaced on deck
    if cable_beams:
        deck_xs = sorted([(_shape_center(b) or [0,0,0])[0] for b in cable_beams])
        if len(deck_xs) >= 3:
            # check spacing in left and right halves independently
            w['cable_dist'] = (0.5 * _spacing_regularity([x for x in deck_xs if x < 0])
                              + 0.5 * _spacing_regularity([x for x in deck_xs if x > 0]), 0.15)
        else:
            w['cable_dist'] = (0.5, 0.15)
    else:
        w['cable_dist'] = (0.0, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=5.0)

def _spacing_regularity(vals):
    """Helper: how regular are sorted values spaced? 0.0–1.0."""
    if len(vals) < 2:
        return 1.0
    vals = sorted(vals)
    gaps = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
    if not gaps:
        return 1.0
    mean_g = sum(gaps)/len(gaps)
    if mean_g < 0.01:
        return 0.0
    dev = sum(abs(g - mean_g)/mean_g for g in gaps)/len(gaps)
    return max(0.0, 1.0 - dev)

def _sem_planetary(shapes, golden, scale):
    """Planetary Array: tangential sun+planet gear arrangement."""
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    w = {}
    w['interference']  = (_phys_interference(shapes, tol_mm=0.3), 0.15)
    # Identify sun (largest radius at center) and planets
    if cylinders:
        sun = max(cylinders, key=lambda c: c.get("radius", 0))
        planets = [c for c in cylinders if c is not sun]
    else:
        sun, planets = None, []
    # Tangential contact: each planet center distance = sun_r + planet_r
    if sun and planets:
        sr = sun.get("radius", 10)
        sc = _shape_center(sun) or [0,0,0]
        tang_scores = []
        for p in planets:
            pr = p.get("radius", 3)
            pc = _shape_center(p) or [0,0,0]
            dist = math.sqrt((pc[0]-sc[0])**2 + (pc[1]-sc[1])**2)
            expected = sr + pr
            err = abs(dist - expected) / max(expected, 1)
            tang_scores.append(max(0, 1.0 - err))
        w['tangency'] = (sum(tang_scores)/len(tang_scores), 0.25)
    else:
        w['tangency'] = (0.0, 0.25)
    # Even angular distribution of planets
    w['angular_reg'] = (_check_angular_regularity(planets, *((_shape_center(sun) or [0,0])[:2])) if sun else 0.0, 0.25)
    # Coplanarity (all in same Z plane)
    w['coplanar']    = (_check_coplanar_z(cylinders), 0.20)
    # Planet size uniformity
    w['planet_size'] = (_check_uniform_sizes(planets, "radius") if planets else 0.0, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=3.0)

def _sem_truss(shapes, golden, scale):
    """Cross-Braced Truss: stories of pillars + X-diagonal braces."""
    beams = [s for s in shapes if s.get("type") == "beam"]
    tol = max(3.0, scale * 1.5)
    w = {}
    w['gravity']      = (_phys_gravity(shapes, tol_mm=2.0), 0.15)
    w['connectivity']  = (_phys_connectivity(shapes, tol_mm=tol), 0.20)
    w['interference']  = (_phys_interference(shapes), 0.10)
    # Story stacking: Z endpoints cluster at uniform heights
    z_endpoints = set()
    for b in beams:
        for pt in [b.get("start"), b.get("end")]:
            if pt:
                z_endpoints.add(round(pt[2], 0))
    z_levels = sorted(z_endpoints)
    if len(z_levels) >= 3:
        gaps = [z_levels[i+1]-z_levels[i] for i in range(len(z_levels)-1)]
        mean_g = sum(gaps)/len(gaps)
        dev = sum(abs(g-mean_g)/max(mean_g, 0.01) for g in gaps)/len(gaps) if mean_g > 0.01 else 1.0
        w['story_height'] = (max(0, 1.0 - dev), 0.20)
    else:
        w['story_height'] = (0.5, 0.20)
    # Square footprint: XY endpoints should cluster at 4 corner positions
    xy_pts = set()
    for b in beams:
        for pt in [b.get("start"), b.get("end")]:
            if pt:
                xy_pts.add((round(pt[0], 0), round(pt[1], 0)))
    n_xy = len(xy_pts)
    # For a perfect truss, exactly 4 unique XY positions
    w['square_foot'] = (min(1.0, 4.0 / max(n_xy, 1)), 0.20)
    # Beam count per story should be 12 (4 pillars + 8 diagonals)
    expected_per_story = 12
    expected_total = scale * expected_per_story
    count_score = min(1.0, len(beams) / max(expected_total, 1))
    w['beam_count'] = (count_score, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=tol)

def _sem_fractal(shapes, golden, scale):
    """Fractal Y-Tree: recursive binary branching in XZ plane."""
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    w['interference']  = (_phys_interference(shapes, tol_mm=0.5), 0.10)
    # Planarity: all beams in XZ plane (Y ≈ 0)
    w['planarity']    = (_check_planarity_y0(beams), 0.20)
    # Parent-child continuity: each beam end should match another beam start
    if len(beams) >= 2:
        endpoints = []
        startpoints = []
        for b in beams:
            endpoints.append(tuple(b.get("end", [0,0,0])))
            startpoints.append(tuple(b.get("start", [0,0,0])))
        matches = 0
        for ep in endpoints:
            for sp in startpoints:
                if _dist3(list(ep), list(sp)) < 1.0:
                    matches += 1
                    break
        # Root has no parent, so max matches = len-1
        w['continuity'] = (matches / max(1, len(beams) - 1), 0.20)
    else:
        w['continuity'] = (0.0, 0.20)
    # Length halving: child beams should be ~half their parent's length
    beam_lengths = sorted([_dist3(b.get("start",[0,0,0]), b.get("end",[0,0,0])) for b in beams], reverse=True)
    if len(beam_lengths) >= 3:
        # Group by approximate length level
        ratios = []
        for i in range(len(beam_lengths)-1):
            if beam_lengths[i] > 1e-3:
                r = beam_lengths[i+1] / beam_lengths[i]
                if 0.3 < r < 0.8:  # should be ~0.5
                    ratios.append(abs(r - 0.5) / 0.5)
        w['halving'] = (max(0, 1.0 - sum(ratios)/len(ratios)) if ratios else 0.5, 0.25)
    else:
        w['halving'] = (0.5, 0.25)
    # Expected beam count: 2^(depth+1) - 1
    expected = 2**(scale+1) - 1
    count_score = min(1.0, len(beams) / max(expected, 1))
    w['beam_count'] = (count_score, 0.25)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=2.0)

def _sem_bcc(shapes, golden, scale):
    """BCC Lattice: body-centered cubic crystal structure."""
    spheres = [s for s in shapes if s.get("type") == "sphere"]
    beams = [s for s in shapes if s.get("type") == "beam"]
    w = {}
    w['interference']  = (_phys_interference(shapes, tol_mm=0.3), 0.10)
    w['connectivity']  = (_phys_connectivity(shapes, tol_mm=8.0), 0.15)
    # Lattice regularity: sphere positions on regular grid
    if spheres:
        # Check if positions cluster to grid-aligned values
        for axis_idx, axis_name in enumerate(('x', 'y', 'z')):
            vals = sorted(set(round((_shape_center(s) or [0,0,0])[axis_idx], 0) for s in spheres))
            if len(vals) >= 2:
                gaps = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
                # BCC should have 5-unit spacing (corners at 0,10,20 and centers at 5,15)
                w[f'grid_{axis_name}'] = (_spacing_regularity(vals), 0.10)
            else:
                w[f'grid_{axis_name}'] = (0.5, 0.10)
    else:
        for axis_name in ('x', 'y', 'z'):
            w[f'grid_{axis_name}'] = (0.0, 0.10)
    # Center-to-corner beams: expected 8*N^3 beams
    expected_beams = 8 * (scale ** 3)
    beam_score = min(1.0, len(beams) / max(expected_beams, 1))
    w['beam_count'] = (beam_score, 0.20)
    # Sphere count: (N+1)^3 corners + N^3 centers (with dedup)
    expected_spheres = (scale+1)**3 + scale**3
    sphere_score = min(1.0, len(spheres) / max(expected_spheres, 1))
    w['sphere_count'] = (sphere_score, 0.15)
    raw = sum(s*wt for s, wt in w.values())
    # Physics gate: only apply when beams are present.
    # Missing beams = incomplete model, not physically broken —
    # don't crush the node-placement score just because connectivity = 0.
    if beams:
        return _apply_physics_gate(raw, shapes, tol_mm=8.0)
    # No beams: partial credit for node structure only, capped at 45%
    # (a nodes-only output can never be a correct BCC lattice)
    return min(raw, 0.45)

def _sem_furniture(shapes, ops, specs):
    """Furniture Assembly (Phase 3): table with support legs."""
    w = {}
    w['gravity'] = (_phys_gravity(shapes), 0.15)
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=3.0), 0.15)
    w['interference'] = (_phys_interference(shapes), 0.10)
    # Mates: legs coincident to tabletop
    if "mates" in specs:
        res = validate_mates(shapes, specs["mates"])
        chk = max(1, res.get("checked", 1))
        w['mates'] = (res.get("correct", 0) / chk, 0.30)
    else:
        w['mates'] = (0.5, 0.30)
    # Leg angular symmetry
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    w['leg_symmetry'] = (_check_angular_regularity(cylinders), 0.15)
    # Leg size uniformity
    w['leg_uniform'] = (_check_uniform_sizes(cylinders, "radius"), 0.15)
    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=3.0)

def _sem_manifold(shapes, ops, specs):
    """Pipe Manifold (Phase 3): header pipe with branches, flanges, valves, brackets.

    Strengthened evaluator — additional structural checks:
      - Branch count matches expected N
      - End-cap presence (2 cylinders closing header ends)
      - Wall/mounting surface presence
      - Flange count and flange-branch pairing
      - Valve count per branch
      - Header elevation above ground
      - Branch-header connectivity (branches must touch header)
    """
    w = {}
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    boxes     = [s for s in shapes if s.get("type") == "box"]

    # --- Helper: get normalised axis vector for a shape (default Z-up) ---
    def _get_axis(s):
        ax = s.get("axis", [0, 0, 1])
        if isinstance(ax, list) and len(ax) == 3:
            mag = math.sqrt(sum(v*v for v in ax))
            return [v / mag for v in ax] if mag > 1e-9 else [0, 0, 1]
        return [0, 0, 1]

    def _dot(a, b):
        return sum(x*y for x, y in zip(a, b))

    # Identify header (longest pipe)
    header = None
    header_axis = None
    if pipes:
        header = max(pipes, key=lambda p: p.get("height", 0))
        header_axis = _get_axis(header)

    # Identify branch pipes (shorter pipes, perpendicular to header)
    branch_pipes = []
    flange_pipes = []
    if header and header_axis:
        non_header = [p for p in pipes if p is not header]
        for p in non_header:
            pax = _get_axis(p)
            dot_val = abs(_dot(pax, header_axis))
            h = p.get("height", 0)
            # Branches: perpendicular to header, reasonably long
            if dot_val < 0.3 and h > 15:
                branch_pipes.append(p)
            # Flanges: perpendicular, short (< 10mm)
            elif dot_val < 0.3 and h <= 15:
                flange_pipes.append(p)

    # --- Physics (12%) ---
    w['gravity']      = (_phys_gravity(shapes), 0.04)
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=5.0), 0.04)
    w['interference'] = (_phys_interference(shapes, tol_mm=0.5), 0.04)

    # --- Header orientation (8%) ---
    header_orient_score = 0.0
    if header_axis:
        z_component = abs(header_axis[2])
        header_orient_score = max(0.0, 1.0 - z_component)
    w['header_orient'] = (header_orient_score, 0.08)

    # --- Header elevation (5%) ---
    # Header should be elevated above ground (Z > 20mm typically)
    header_elev_score = 0.0
    if header:
        hc = _shape_center(header)
        if hc and hc[2] > 15.0:
            header_elev_score = min(1.0, hc[2] / 50.0)
    w['header_elevation'] = (header_elev_score, 0.05)

    # --- Branch count (8%) ---
    # Infer expected N from reference shapes
    ref = specs.get("reference", [])
    ref_pipes = [s for s in ref if s.get("type") == "pipe"]
    # In reference: 1 header + N branches + N flanges = 1 + 2N pipes
    expected_branches = max(1, (len(ref_pipes) - 1) // 2) if ref_pipes else 2
    branch_count_score = 0.0
    if expected_branches > 0:
        ratio = len(branch_pipes) / expected_branches
        if 0.8 <= ratio <= 1.2:
            branch_count_score = 1.0
        elif ratio > 0:
            branch_count_score = max(0.0, 1.0 - abs(ratio - 1.0))
    w['branch_count'] = (branch_count_score, 0.08)

    # --- Branch perpendicularity and consistency (10%) ---
    branch_perp_score = 0.0
    branch_consistency_score = 0.0
    if branch_pipes and header_axis:
        perp_scores = []
        branch_axes = []
        for bp in branch_pipes:
            bax = _get_axis(bp)
            branch_axes.append(bax)
            dot_val = abs(_dot(bax, header_axis))
            perp_scores.append(max(0.0, 1.0 - dot_val * 2.0))
        if perp_scores:
            branch_perp_score = sum(perp_scores) / len(perp_scores)
        if len(branch_axes) >= 2:
            ref_ax = branch_axes[0]
            consist_scores = [abs(_dot(bax, ref_ax)) for bax in branch_axes[1:]]
            branch_consistency_score = sum(consist_scores) / len(consist_scores)
        else:
            branch_consistency_score = 1.0
    w['branch_perp'] = (branch_perp_score, 0.06)
    w['branch_consistency'] = (branch_consistency_score, 0.04)

    # --- Flange count (5%) ---
    # Should have one flange per branch
    flange_score = 0.0
    if expected_branches > 0:
        flange_ratio = len(flange_pipes) / expected_branches
        flange_score = min(1.0, flange_ratio)
    w['flange_count'] = (flange_score, 0.05)

    # --- End-cap presence (5%) ---
    # 2 short cylinders at header ends, aligned with header axis
    endcap_score = 0.0
    if header_axis and cylinders:
        endcaps = []
        for c in cylinders:
            cax = _get_axis(c)
            # Must be aligned with header (|dot| > 0.8) and short
            if abs(_dot(cax, header_axis)) > 0.7 and c.get("height", 999) < 15.0:
                endcaps.append(c)
        endcap_score = min(1.0, len(endcaps) / 2.0)
    w['endcap_count'] = (endcap_score, 0.05)

    # --- Valve count (4%) ---
    # Should have N valve cylinders (one per branch), perpendicular to header
    valve_cyls = []
    if header_axis and cylinders:
        for c in cylinders:
            cax = _get_axis(c)
            if abs(_dot(cax, header_axis)) < 0.3:  # perpendicular to header
                valve_cyls.append(c)
    valve_score = min(1.0, len(valve_cyls) / max(1, expected_branches))
    w['valve_count'] = (valve_score, 0.04)

    # --- Mates (10%) ---
    if "mates" in specs:
        res = validate_mates(shapes, specs["mates"])
        chk = max(1, res.get("checked", 1))
        w['mates'] = (res.get("correct", 0) / chk, 0.10)
    else:
        w['mates'] = (0.5, 0.10)

    # --- Clearance fit (8%) ---
    if "clearance_fit" in specs:
        res = validate_clearance_fit(shapes, specs["clearance_fit"])
        chk = max(1, res.get("checked", 1))
        w['clearance'] = (res.get("correct", 0) / chk, 0.08)
    else:
        w['clearance'] = (0.5, 0.08)

    # --- Branch spacing uniformity (8%) ---
    if len(branch_pipes) >= 2:
        # Project branch centres onto header axis for spacing check
        branch_centres = [_shape_center(p) for p in branch_pipes if _shape_center(p)]
        if header_axis and len(branch_centres) >= 2:
            # Project onto header axis direction
            projections = sorted(_dot(c, header_axis) for c in branch_centres)
            gaps = [projections[i+1] - projections[i] for i in range(len(projections)-1)]
            mean_gap = sum(gaps) / len(gaps) if gaps else 1.0
            if mean_gap > 0.1:
                dev = sum(abs(g - mean_gap) / mean_gap for g in gaps) / len(gaps)
                w['spacing'] = (max(0.0, 1.0 - dev), 0.08)
            else:
                w['spacing'] = (0.0, 0.08)
        else:
            w['spacing'] = (0.0, 0.08)
    else:
        w['spacing'] = (0.5, 0.08)

    # --- Type variety (3%) ---
    types = set(s.get("type") for s in shapes)
    expected_types = {"box", "cylinder", "pipe"}
    w['type_variety'] = (len(types & expected_types) / len(expected_types), 0.03)

    # --- Bracket ground contact (5%) ---
    # Brackets should sit on Z=0 ground and reach up to header
    bracket_score = 0.0
    if boxes:
        grounded = 0
        for b in boxes:
            bc = _shape_center(b)
            sz = b.get("size", [0, 0, 0])
            if bc and isinstance(sz, list) and len(sz) == 3:
                base_z = bc[2] - sz[2] / 2.0
                if abs(base_z) < 3.0:
                    grounded += 1
        bracket_score = grounded / len(boxes)
    w['bracket_ground'] = (bracket_score, 0.05)

    # --- Wall presence (4%) ---
    # Should have a back wall (large box behind header)
    wall_score = 0.0
    if boxes:
        for b in boxes:
            sz = b.get("size", [0, 0, 0])
            if isinstance(sz, list) and len(sz) == 3:
                # Wall is thin in one dimension, large in the other two
                dims = sorted(sz)
                if dims[0] < 10.0 and dims[1] > 50.0 and dims[2] > 30.0:
                    wall_score = 1.0
                    break
    w['wall_presence'] = (wall_score, 0.04)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=5.0)

def _sem_axle(shapes, ops, specs):
    """Axle Bearing (Phase 3): rotational shaft with bearings in support block.

    Strengthened evaluator — structural checks beyond clearance/mates:
      - Block presence and orientation
      - Shaft horizontal alignment and length vs block width
      - Bore full penetration through block
      - Bearing count and symmetrical placement
      - Shaft protrusion beyond block (functional requirement)
    """
    boxes     = [s for s in shapes if s.get("type") == "box"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    w = {}

    # ── Physics (15%) ────────────────────────────────────────
    w['gravity']      = (_phys_gravity(shapes), 0.05)
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=3.0), 0.05)
    if "interference_check" in specs:
        ir = validate_interference(shapes, ops, tol_mm=0.2)
        w['interference'] = (1.0 if ir.get("interference_free", False) else 0.0, 0.05)
    else:
        w['interference'] = (_phys_interference(shapes), 0.05)

    # ── Clearance fit (15%) ──────────────────────────────────
    if "clearance_fit" in specs:
        res = validate_clearance_fit(shapes, specs["clearance_fit"])
        chk = max(1, res.get("checked", 1))
        w['clearance'] = (res.get("correct", 0) / chk, 0.15)
    else:
        w['clearance'] = (0.5, 0.15)

    # ── Mates / concentricity (15%) ──────────────────────────
    if "mates" in specs:
        res = validate_mates(shapes, specs["mates"])
        chk = max(1, res.get("checked", 1))
        w['mates'] = (res.get("correct", 0) / chk, 0.15)
    else:
        w['mates'] = (0.5, 0.15)

    # ── Shape count and variety (10%) ────────────────────────
    # Expect: 1 box (block), 2 cylinders (bore + shaft), 2 pipes (bearings)
    types = set(s.get("type") for s in shapes)
    expected_types = {"box", "cylinder", "pipe"}
    w['type_variety'] = (len(types & expected_types) / len(expected_types), 0.05)
    # Exact shape count: 5 shapes
    count_score = min(1.0, len(shapes) / 5.0) if len(shapes) <= 7 else max(0.0, 1.0 - (len(shapes) - 5) / 10.0)
    w['shape_count'] = (count_score, 0.05)

    # ── Support block check (8%) ─────────────────────────────
    # Must have exactly 1 box sitting on ground (Z≈0 base)
    block_score = 0.0
    block = None
    if boxes:
        # Find the box with the largest volume (the support block)
        def _box_vol(b):
            sz = b.get("size", [0, 0, 0])
            return sz[0] * sz[1] * sz[2] if isinstance(sz, list) and len(sz) == 3 else 0
        block = max(boxes, key=_box_vol)
        c = _shape_center(block)
        sz = block.get("size", [0, 0, 0])
        if c and isinstance(sz, list) and len(sz) == 3:
            # Block bottom should be near Z=0
            base_z = c[2] - sz[2] / 2.0
            if abs(base_z) < 5.0:
                block_score += 0.5
            # Block should be roughly cubic or rectangular (all dims > 10mm)
            if all(d > 10.0 for d in sz):
                block_score += 0.5
    w['block_check'] = (block_score, 0.08)

    # ── Shaft alignment (10%) ────────────────────────────────
    # Shaft should be horizontal (axis ≈ X), longer than block width
    shaft_score = 0.0
    shaft = None
    if cylinders:
        # Shaft = longest cylinder
        shaft = max(cylinders, key=lambda c: c.get("height", 0))
        ax = shaft.get("axis", [0, 0, 1])
        if isinstance(ax, list) and len(ax) == 3:
            mag = math.sqrt(sum(v*v for v in ax))
            if mag > 1e-9:
                ax_norm = [v / mag for v in ax]
                # Horizontal: |x-component| should be near 1
                shaft_score += abs(ax_norm[0]) * 0.5
                # Shaft should protrude beyond block (shaft length > block width)
                if block:
                    block_w = block.get("size", [0, 0, 0])[0]
                    if shaft.get("height", 0) > block_w * 1.0:
                        shaft_score += 0.5
    w['shaft_alignment'] = (shaft_score, 0.10)

    # ── Bearing count and symmetry (12%) ─────────────────────
    # Should have exactly 2 bearings (pipes), placed symmetrically about X=0
    bearing_score = 0.0
    if len(pipes) == 2:
        bearing_score += 0.4
        c1 = _shape_center(pipes[0])
        c2 = _shape_center(pipes[1])
        if c1 and c2:
            # Symmetry: X coords should be equal and opposite (or mirrored)
            if abs(c1[0] + c2[0]) < 5.0:  # mirror about X=0
                bearing_score += 0.3
            # Both bearings horizontal (axis ≈ X)
            for p in pipes:
                ax = p.get("axis", [0, 0, 1])
                if isinstance(ax, list) and len(ax) == 3:
                    mag = math.sqrt(sum(v*v for v in ax))
                    if mag > 1e-9 and abs(ax[0] / mag) > 0.8:
                        bearing_score += 0.15
    elif len(pipes) >= 1:
        bearing_score += 0.2  # at least some bearings
    w['bearing_symmetry'] = (bearing_score, 0.12)

    # ── Bore penetration (10%) ───────────────────────────────
    # The bore cylinder should fully penetrate the block (bore length ≥ block width)
    bore_score = 0.0
    if block and len(cylinders) >= 2:
        # Bore = the cylinder that's NOT the shaft (shorter one, or one with bigger radius)
        bore_candidates = [c for c in cylinders if c is not shaft]
        if bore_candidates:
            bore = bore_candidates[0]
            bore_h = bore.get("height", 0)
            block_w = block.get("size", [0, 0, 0])[0]
            if bore_h >= block_w * 0.95:
                bore_score += 0.5
            # Bore must be coaxial with shaft
            if shaft:
                bc = _shape_center(bore)
                sc = _shape_center(shaft)
                if bc and sc:
                    # Y and Z coords should match
                    yz_dist = math.sqrt((bc[1] - sc[1])**2 + (bc[2] - sc[2])**2)
                    bore_score += max(0.0, 0.5 - yz_dist / 20.0)
    w['bore_penetration'] = (bore_score, 0.10)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=3.0)


# ── Phase 4 Bio-Inspired semantic evaluators ─────────────────

def _sem_phyllotaxis(shapes, golden, scale):
    """Phyllotaxis Disc: sunflower golden-angle Fibonacci spiral of spheres."""
    spheres = [s for s in shapes if s.get("type") == "sphere"]
    w = {}
    # Base physics
    w['gravity']      = (_phys_gravity(shapes, tol_mm=5.0), 0.05)
    w['interference']  = (_phys_interference(shapes, tol_mm=1.0), 0.10)

    # Golden angle regularity: sort spheres by angle, check angular increments ≈ 137.508°
    GOLDEN_ANGLE = math.radians(137.50776405003785)
    sphere_data = []
    for s in spheres:
        c = _shape_center(s)
        if c and (abs(c[0]) > 0.01 or abs(c[1]) > 0.01):
            angle = math.atan2(c[1], c[0])
            r = math.sqrt(c[0]**2 + c[1]**2)
            sphere_data.append((r, angle))
    # Sort by radius (inner→outer = seed index order in a Fermat spiral)
    sphere_data.sort(key=lambda x: x[0])
    if len(sphere_data) >= 3:
        # Check consecutive angular increment ≈ golden angle (mod 2π)
        angle_errs = []
        for i in range(len(sphere_data) - 1):
            diff = sphere_data[i+1][1] - sphere_data[i][1]
            # Normalize to [0, 2π)
            diff = diff % (2*math.pi)
            # Closest to golden angle
            err1 = abs(diff - GOLDEN_ANGLE) / GOLDEN_ANGLE
            err2 = abs(diff - (2*math.pi - GOLDEN_ANGLE)) / GOLDEN_ANGLE
            angle_errs.append(min(err1, err2))
        mean_err = sum(angle_errs) / len(angle_errs)
        w['golden_angle'] = (max(0.0, 1.0 - mean_err), 0.30)
    else:
        w['golden_angle'] = (0.0, 0.30)

    # Fermat spiral radius check: r(n) ∝ √n → r²(n) should be linearly spaced
    if len(sphere_data) >= 3:
        r_sq = [d[0]**2 for d in sphere_data]
        gaps = [r_sq[i+1] - r_sq[i] for i in range(len(r_sq)-1)]
        mean_gap = sum(gaps) / len(gaps) if gaps else 1.0
        if mean_gap > 0.01:
            dev = sum(abs(g - mean_gap) / mean_gap for g in gaps) / len(gaps)
            w['spiral_spacing'] = (max(0.0, 1.0 - dev * 0.5), 0.25)
        else:
            w['spiral_spacing'] = (0.0, 0.25)
    else:
        w['spiral_spacing'] = (0.0, 0.25)

    # Planarity (all seeds at same Z)
    w['planarity'] = (_check_coplanar_z(spheres), 0.10)

    # Seed count
    expected = scale
    actual = len(spheres)
    w['count'] = (min(1.0, actual / max(1, expected)), 0.10)

    # Size uniformity (all seeds same radius)
    w['size_uniform'] = (_check_uniform_sizes(spheres, "radius"), 0.10)

    raw = sum(s*wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=5.0)


def _sem_compound_eye(shapes, golden, scale):
    """Compound Eye Array: hemispherical dome with ommatidia in concentric rings."""
    spheres   = [s for s in shapes if s.get("type") == "sphere"]
    cones     = [s for s in shapes if s.get("type") == "cone"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    tori      = [s for s in shapes if s.get("type") == "torus"]
    w = {}

    N_rings = scale
    n_omm = 1 + sum(6 * k for k in range(1, N_rings + 1))
    dome_R = 50.0

    # Type variety — should have at least 4 of 6 required types
    type_set = set(s.get("type") for s in shapes)
    required = {"sphere", "cone", "cylinder", "pipe", "torus"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.10)

    # Component counts — each ommatidium = 1 sphere + 1 cone + 1 cylinder
    # Plus 1 dome sphere + 1 pipe + 1 torus as supports
    expected_lens = n_omm + 1  # dome sphere + lens spheres
    expected_cone = n_omm
    expected_cyl  = n_omm
    w['lens_count'] = (min(1.0, len(spheres) / max(1, expected_lens)), 0.08)
    w['cone_count'] = (min(1.0, len(cones) / max(1, expected_cone)), 0.08)
    w['cyl_count']  = (min(1.0, len(cylinders) / max(1, expected_cyl)), 0.08)

    # Dome presence — should be a large sphere near origin
    dome_found = any(s.get("radius", 0) > 30 for s in spheres)
    w['dome_present'] = (1.0 if dome_found else 0.0, 0.06)

    # Hemispherical distribution — lens spheres should be on a dome surface
    # Check they lie at roughly dome_R from origin
    lens_radii = []
    for s in spheres:
        c = _shape_center(s)
        if c and s.get("radius", 0) < 10:  # not the dome itself
            r = math.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
            lens_radii.append(r)
    if lens_radii:
        mean_r = sum(lens_radii) / len(lens_radii)
        r_err = abs(mean_r - dome_R) / dome_R
        w['dome_radius'] = (max(0.0, 1.0 - r_err * 2), 0.12)
    else:
        w['dome_radius'] = (0.0, 0.12)

    # Ring structure — check concentric ring counts (ring k has 6k units)
    # Cluster lens spheres by polar angle from +Z axis
    lens_thetas = []
    for s in spheres:
        c = _shape_center(s)
        if c and s.get("radius", 0) < 10:
            r = math.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
            if r > 1:
                theta = math.acos(max(-1, min(1, c[2] / r)))
                lens_thetas.append(theta)
    if len(lens_thetas) >= 5:
        # Sort and cluster by theta
        lens_thetas.sort()
        clusters = [[lens_thetas[0]]]
        for t in lens_thetas[1:]:
            if t - clusters[-1][-1] < math.radians(8):
                clusters[-1].append(t)
            else:
                clusters.append([t])
        expected_rings = N_rings + 1  # ring 0..N
        w['ring_count'] = (min(1.0, len(clusters) / max(1, expected_rings)), 0.15)
    else:
        w['ring_count'] = (0.0, 0.15)

    # Upper hemisphere — most ommatidia should have Z > 0
    above = sum(1 for t in lens_thetas if t < math.pi / 2)
    w['upper_hemi'] = (above / max(1, len(lens_thetas)) if lens_thetas else 0, 0.08)

    # Radial alignment — cones and cylinders should point inward (toward origin)
    inward_count = 0
    total_axes = 0
    for s in cones + cylinders:
        c = _shape_center(s)
        ax = s.get("axis")
        if c and ax and any(abs(v) > 0.01 for v in c):
            # Inward direction = from surface toward origin = -normalize(c)
            r = math.sqrt(sum(v**2 for v in c))
            if r > 1:
                inward = [-v/r for v in c]
                # Dot product with axis (either direction is fine for alignment)
                dot = abs(sum(inward[i] * ax[i] for i in range(3)))
                if dot > 0.5:
                    inward_count += 1
                total_axes += 1
    w['axis_alignment'] = (inward_count / max(1, total_axes), 0.15)

    # Connectivity — shapes should form connected groups
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=15.0), 0.10)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=15.0)


def _sem_diatom(shapes, golden, scale):
    """Diatom Frustule: bilateral symmetric silica shell with costae, areolae, girdle bands."""
    boxes     = [s for s in shapes if s.get("type") == "box"]
    beams     = [s for s in shapes if s.get("type") == "beam"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    spheres   = [s for s in shapes if s.get("type") == "sphere"]
    tori      = [s for s in shapes if s.get("type") == "torus"]
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    w = {}

    N = scale  # costae per half-valve

    # Type variety — should have at least 5 of 6 required types
    type_set = set(s.get("type") for s in shapes)
    required = {"box", "beam", "cylinder", "sphere", "torus", "pipe"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.10)

    # Valve count — expect 2 boxes (top and bottom valves)
    w['valve_count'] = (min(1.0, len(boxes) / 2.0), 0.08)

    # Valve symmetry — two valves should be at ±Z, symmetric about Z=0
    if len(boxes) >= 2:
        zs = sorted([_shape_center(b)[2] for b in boxes if _shape_center(b)])
        if len(zs) >= 2:
            # Check that the two extreme Z values are symmetric
            sym_err = abs(zs[0] + zs[-1]) / max(1, abs(zs[-1] - zs[0]))
            w['valve_symmetry'] = (max(0.0, 1.0 - sym_err * 2), 0.08)
        else:
            w['valve_symmetry'] = (0.0, 0.08)
    else:
        w['valve_symmetry'] = (0.0, 0.08)

    # Raphe count — expect 2 beams
    w['raphe_count'] = (min(1.0, len(beams) / 2.0), 0.06)

    # Costa count — expect 4 × N × 2 (mirror) = 8N costae
    expected_costae = 8 * N
    w['costa_count'] = (min(1.0, len(cylinders) / max(1, expected_costae)), 0.10)

    # Costa regularity — costae should be evenly spaced along X
    costa_x = sorted([abs(_shape_center(c)[0]) for c in cylinders
                      if _shape_center(c) and abs(_shape_center(c)[0]) > 0.5])
    if len(costa_x) >= 4:
        # Check spacing uniformity
        diffs = [costa_x[i+1] - costa_x[i] for i in range(len(costa_x) - 1)]
        if diffs:
            mean_d = sum(diffs) / len(diffs)
            if mean_d > 0.01:
                var = sum((d - mean_d)**2 for d in diffs) / len(diffs)
                cv = math.sqrt(var) / mean_d  # coefficient of variation
                w['costa_regularity'] = (max(0.0, 1.0 - cv * 2), 0.12)
            else:
                w['costa_regularity'] = (0.0, 0.12)
        else:
            w['costa_regularity'] = (0.0, 0.12)
    else:
        w['costa_regularity'] = (0.0, 0.12)

    # Bilateral symmetry of costae — X positions should be mirrored
    if len(costa_x) >= 2:
        pos_x = sorted(set(round(abs(_shape_center(c)[0]), 1) for c in cylinders
                           if _shape_center(c)))
        # For each |x|, count costae with +x and -x
        mirror_score = 0
        for xv in pos_x:
            n_pos = sum(1 for c in cylinders if _shape_center(c) and abs(_shape_center(c)[0] - xv) < 1)
            n_neg = sum(1 for c in cylinders if _shape_center(c) and abs(_shape_center(c)[0] + xv) < 1)
            if n_pos > 0 and n_neg > 0:
                mirror_score += min(n_pos, n_neg) / max(n_pos, n_neg)
        w['bilateral_sym'] = (mirror_score / max(1, len(pos_x)), 0.12)
    else:
        w['bilateral_sym'] = (0.0, 0.12)

    # Areolae count — expect 8 × N small spheres (minus the nodule)
    small_spheres = [s for s in spheres if s.get("radius", 0) < 2.0]
    expected_areolae = 8 * N
    w['areola_count'] = (min(1.0, len(small_spheres) / max(1, expected_areolae)), 0.08)

    # Girdle bands — expect 2 torus rings
    w['girdle_count'] = (min(1.0, len(tori) / 2.0), 0.06)

    # Mantle wall — expect 1 pipe
    w['mantle_count'] = (min(1.0, len(pipes) / 1.0) if pipes else 0.0, 0.05)

    # Central nodule — expect 1 large sphere near origin
    nodule_found = any(s.get("radius", 0) > 1.5 and
                       all(abs(v) < 5 for v in (_shape_center(s) or [99,99,99]))
                       for s in spheres)
    w['nodule'] = (1.0 if nodule_found else 0.0, 0.05)

    # Overall connectivity
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=12.0), 0.10)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=12.0)


# ── NEW BIO-INSPIRED EVALUATORS ──────────────────────────────

def _sem_honeycomb(shapes, golden, scale):
    """Honeycomb Lattice: hex grid of pipe cells with caps, base, and frame."""
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    tori      = [s for s in shapes if s.get("type") == "torus"]
    w = {}

    N_rings = scale
    n_cells = 1 + 3 * N_rings * (N_rings + 1)
    hex_size = 10.0  # cell_outer_r * 2

    # Physics
    w['gravity']      = (_phys_gravity(shapes, tol_mm=5.0), 0.05)
    w['interference']  = (_phys_interference(shapes, tol_mm=1.0), 0.05)

    # Cell count — expect n_cells pipes
    w['cell_count'] = (min(1.0, len(pipes) / max(1, n_cells)), 0.15)

    # Cap count — expect n_cells small cylinders (caps)
    small_cyls = [s for s in cylinders if s.get("height", 999) < 3.0]
    w['cap_count'] = (min(1.0, len(small_cyls) / max(1, n_cells)), 0.10)

    # Hexagonal spacing check: for each pipe, find nearest pipe neighbor
    # and check distance ≈ hex_size
    pipe_centres = [_shape_center(p) for p in pipes if _shape_center(p)]
    if len(pipe_centres) >= 3:
        nn_dists = []
        for i, c1 in enumerate(pipe_centres):
            min_d = float('inf')
            for j, c2 in enumerate(pipe_centres):
                if i == j:
                    continue
                d = math.sqrt(sum((a - b)**2 for a, b in zip(c1, c2)))
                if d < min_d:
                    min_d = d
            nn_dists.append(min_d)
        mean_nn = sum(nn_dists) / len(nn_dists)
        if mean_nn > 0.01:
            dev = sum(abs(d - mean_nn) / mean_nn for d in nn_dists) / len(nn_dists)
            w['hex_regularity'] = (max(0.0, 1.0 - dev * 2.0), 0.25)
        else:
            w['hex_regularity'] = (0.0, 0.25)
    else:
        w['hex_regularity'] = (0.0, 0.25)

    # Vertical alignment — all pipes should have same Z centre
    if pipe_centres:
        z_vals = [c[2] for c in pipe_centres]
        z_mean = sum(z_vals) / len(z_vals)
        z_dev = sum(abs(z - z_mean) for z in z_vals) / len(z_vals) if z_vals else 0
        w['z_alignment'] = (max(0.0, 1.0 - z_dev / 5.0), 0.10)
    else:
        w['z_alignment'] = (0.0, 0.10)

    # Base plate — expect 1 large cylinder
    base_cyls = [s for s in cylinders if s.get("radius", 0) > 20.0]
    w['base_plate'] = (min(1.0, len(base_cyls)), 0.05)

    # Frame — expect 1 torus
    w['frame_ring'] = (min(1.0, len(tori)), 0.05)

    # Type variety
    type_set = set(s.get("type") for s in shapes)
    required = {"pipe", "cylinder", "torus"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.10)

    # Connectivity
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=12.0), 0.10)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=8.0)


def _sem_radiolarian(shapes, golden, scale):
    """Armillary Sphere: concentric great-circle ring shells (torus wireframes) + icosahedral axis rods + girdle rings."""
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    tori      = [s for s in shapes if s.get("type") == "torus"]
    w = {}

    N_shells = scale
    n_spines = 12 * (N_shells - 1)
    # Ground truth: 3 shell tori per shell + 1 girdle torus per shell
    n_expected_tori = N_shells * 3 + N_shells  # shell lattice + girdles

    # Shell lattice tori — origin-centred tori (3 per shell × N_shells + N_shells girdles)
    origin_tori = [s for s in tori
                   if _shape_center(s) and
                   all(abs(v) < 2.0 for v in _shape_center(s))]
    # Count distinct ring_radius values to determine how many shells are represented
    shell_ring_radii = sorted(set(round(s.get("ring_radius", 0), 1) for s in origin_tori))
    w['shell_count'] = (min(1.0, len(shell_ring_radii) / max(1, N_shells)), 0.15)

    # Shell radius progression: should be r_base × 2^k (geometric ratio ≈ 2.0)
    if len(shell_ring_radii) >= 2:
        ratios = [shell_ring_radii[i+1] / shell_ring_radii[i]
                  for i in range(len(shell_ring_radii)-1) if shell_ring_radii[i] > 0.01]
        if ratios:
            ratio_err = sum(abs(r - 2.0) / 2.0 for r in ratios) / len(ratios)
            w['shell_progression'] = (max(0.0, 1.0 - ratio_err), 0.15)
        else:
            w['shell_progression'] = (0.0, 0.15)
    else:
        w['shell_progression'] = (0.0, 0.15)

    # Spine count
    w['spine_count'] = (min(1.0, len(cylinders) / max(1, n_spines)), 0.15)

    # Icosahedral symmetry: spine axes should point radially (axis ≈ normalized centre)
    radial_score = 0.0
    n_checked = 0
    for cyl in cylinders:
        c = _shape_center(cyl)
        axis = cyl.get("axis", [0, 0, 1])
        if c and any(abs(v) > 1.0 for v in c):
            c_len = math.sqrt(sum(v**2 for v in c))
            if c_len > 0.01:
                c_norm = [v / c_len for v in c]
                dot = abs(sum(a * b for a, b in zip(axis, c_norm)))
                radial_score += dot
                n_checked += 1
    if n_checked > 0:
        w['radial_alignment'] = (radial_score / n_checked, 0.15)
    else:
        w['radial_alignment'] = (0.0, 0.15)

    # Torus count (shell lattice tori + girdle tori)
    w['torus_count'] = (min(1.0, len(origin_tori) / max(1, n_expected_tori)), 0.10)

    # Concentricity — all tori at origin
    if tori:
        origin_count = sum(1 for s in tori
                           if _shape_center(s) and
                           all(abs(v) < 3.0 for v in _shape_center(s)))
        w['concentricity'] = (origin_count / len(tori), 0.10)
    else:
        w['concentricity'] = (0.0, 0.10)

    # Type variety — expect torus and cylinder (no solid spheres)
    type_set = set(s.get("type") for s in shapes)
    required = {"cylinder", "torus"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.10)

    # Connectivity
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=25.0), 0.10)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=25.0)


def _sem_vertebral(shapes, golden, scale):
    """Vertebral Column: linear articulated chain with physiological curvature."""
    boxes     = [s for s in shapes if s.get("type") == "box"]
    cones     = [s for s in shapes if s.get("type") == "cone"]
    beams     = [s for s in shapes if s.get("type") == "beam"]
    cylinders = [s for s in shapes if s.get("type") == "cylinder"]
    pipes     = [s for s in shapes if s.get("type") == "pipe"]
    w = {}

    N = scale
    n_discs = N - 1

    # Vertebral body count (boxes)
    w['body_count'] = (min(1.0, len(boxes) / max(1, N)), 0.10)

    # Spinous process count (cones)
    w['process_count'] = (min(1.0, len(cones) / max(1, N)), 0.08)

    # Transverse process count (beams) — expect 2 per vertebra
    w['trans_count'] = (min(1.0, len(beams) / max(1, 2 * N)), 0.08)

    # Disc count (cylinders, small height)
    disc_cyls = [s for s in cylinders if s.get("height", 999) < 10.0]
    w['disc_count'] = (min(1.0, len(disc_cyls) / max(1, n_discs)), 0.08)

    # Spinal canal (segmented pipe — N-1 segments following curvature)
    n_canal_expected = N - 1
    w['canal'] = (min(1.0, len(pipes) / max(1, n_canal_expected)), 0.06)

    # Curvature check: vertebral bodies should NOT be in a straight line
    box_centres = sorted([_shape_center(b) for b in boxes if _shape_center(b)],
                         key=lambda c: c[2])  # sort by Z (superior)
    if len(box_centres) >= 3:
        first = box_centres[0]
        last = box_centres[-1]
        chord_dy = last[1] - first[1]
        chord_dz = last[2] - first[2]
        chord_len = math.sqrt(chord_dy**2 + chord_dz**2)
        if chord_len > 0.01:
            max_dev = 0.0
            for c in box_centres:
                t = ((c[1] - first[1]) * chord_dy + (c[2] - first[2]) * chord_dz) / (chord_len**2)
                proj_y = first[1] + t * chord_dy
                proj_z = first[2] + t * chord_dz
                dev = math.sqrt((c[1] - proj_y)**2 + (c[2] - proj_z)**2)
                if dev > max_dev:
                    max_dev = dev
            w['curvature'] = (min(1.0, max_dev / 10.0), 0.15)
        else:
            w['curvature'] = (0.0, 0.15)
    else:
        w['curvature'] = (0.0, 0.15)

    # Regularity: spacing between consecutive bodies should be roughly uniform
    if len(box_centres) >= 2:
        spacings = []
        for i in range(len(box_centres) - 1):
            d = math.sqrt(sum((a - b)**2 for a, b in zip(box_centres[i], box_centres[i+1])))
            spacings.append(d)
        mean_sp = sum(spacings) / len(spacings)
        if mean_sp > 0.01:
            dev = sum(abs(s - mean_sp) / mean_sp for s in spacings) / len(spacings)
            w['spacing_regularity'] = (max(0.0, 1.0 - dev * 2.0), 0.10)
        else:
            w['spacing_regularity'] = (0.0, 0.10)
    else:
        w['spacing_regularity'] = (0.0, 0.10)

    # Bilateral symmetry: beams should be mirrored in X
    beam_centres = [_shape_center(b) for b in beams if _shape_center(b)]
    if beam_centres:
        x_vals = [c[0] for c in beam_centres]
        pos_x = [x for x in x_vals if x > 1.0]
        neg_x = [x for x in x_vals if x < -1.0]
        if pos_x and neg_x:
            symmetry = min(len(pos_x), len(neg_x)) / max(len(pos_x), len(neg_x))
            w['bilateral_symmetry'] = (symmetry, 0.10)
        else:
            w['bilateral_symmetry'] = (0.0, 0.10)
    else:
        w['bilateral_symmetry'] = (0.0, 0.10)

    # Type variety
    type_set = set(s.get("type") for s in shapes)
    required = {"box", "cone", "beam", "cylinder", "pipe"}
    w['type_variety'] = (len(type_set & required) / len(required), 0.10)

    # Physics
    w['connectivity'] = (_phys_connectivity(shapes, tol_mm=30.0), 0.10)
    w['interference']  = (_phys_interference(shapes, tol_mm=2.0), 0.05)

    raw = sum(s * wt for s, wt in w.values())
    return _apply_physics_gate(raw, shapes, tol_mm=15.0)


def _sem_flanged_pipe(shapes, golden, scale):
    """
    Flanged Pipe Joint semantic evaluator — strict geometric arrangement.

    A flanged pipe joint is an AXIAL assembly: every pipe body and every bolt
    cylinder must be explicitly oriented along the assembly axis (e.g. [1,0,0]).
    Omitting the axis field is a critical failure — the 3D renderer defaults to
    Z-up, making all pipes render as vertical columns instead of a horizontal
    pipe run, which is visually and semantically wrong.

    Weights:
      15% — pipe presence   (hollow pipes required)
      35% — axis direction  (pipes + bolts must declare axis matching assembly)
      25% — bolt-circle regularity  (N cylinders evenly spaced in the plane ⊥ axis)
      15% — bolt-flange colocation  (bolts at the same axial position as flanges)
      10% — nut/torus count  (one torus per bolt)

    Hard gates:
      0 pipes                         → cap at 0.20
      1 pipe                          → cap at 0.35
      axis_fraction < 0.5 (most shapes lack correct axis) → cap at 0.38
    """
    from collections import Counter
    if not shapes:
        return 0.0

    N_bolts = scale   # L1=4, L2=8, L3=16
    n_pipes_req = 4   # 2 pipe bodies + 2 flange rings
    types = Counter(s.get("type") for s in shapes)
    n_pipes = types.get("pipe", 0)
    n_cyls  = types.get("cylinder", 0)
    n_tori  = types.get("torus", 0)

    # ── 1. Pipe presence (15%) ──────────────────────────────────────────────
    if n_pipes == 0:
        pipe_score = 0.0
    elif n_pipes == 1:
        pipe_score = 0.15
    elif n_pipes < n_pipes_req:
        pipe_score = 0.50
    else:
        pipe_score = min(n_pipes / n_pipes_req, 1.0)

    # ── 2. Axis direction (35%) ─────────────────────────────────────────────
    # The assembly axis is inferred from the spatial spread of pipe centers.
    # Each pipe AND each bolt cylinder must declare an 'axis' field that aligns
    # with this assembly axis.  Missing axis → wrong rendering → zero credit.
    pipe_shapes = [s for s in shapes if s.get("type") == "pipe"]
    cyl_shapes  = [s for s in shapes if s.get("type") == "cylinder"]

    # Infer assembly axis from pipe-center spread (or default to X)
    if len(pipe_shapes) >= 2:
        centers = [s.get("center", [0, 0, 0]) for s in pipe_shapes]
        try:
            spreads = [max(c[i] for c in centers) - min(c[i] for c in centers)
                       for i in range(3)]
            asm_ax_idx = spreads.index(max(spreads))  # 0=X, 1=Y, 2=Z
        except Exception:
            asm_ax_idx = 0
    else:
        asm_ax_idx = 0  # default X

    def _axis_aligned(ax):
        """Return True if axis vector is aligned (|dot| > 0.7) with assembly axis."""
        if not isinstance(ax, (list, tuple)) or len(ax) < 3:
            return False
        try:
            dot = abs(float(ax[asm_ax_idx]))   # component along assembly axis
            mag = math.sqrt(sum(float(v)**2 for v in ax[:3]))
            return (dot / max(mag, 1e-9)) > 0.7
        except (TypeError, ValueError):
            return False

    # Count axial shapes (pipes + bolt cylinders) with correct axis
    n_axial_total   = len(pipe_shapes) + len(cyl_shapes)
    n_axial_correct = (sum(1 for p in pipe_shapes if _axis_aligned(p.get("axis")))
                       + sum(1 for c in cyl_shapes if _axis_aligned(c.get("axis"))))

    axis_fraction = n_axial_correct / max(n_axial_total, 1)
    axis_score    = axis_fraction  # 0 → 1.0

    # ── 3. Bolt-circle regularity (25%) ────────────────────────────────────
    # Bolts must be off-axis (on the bolt circle) and evenly spaced.
    # Determine which two axes are perpendicular to the assembly axis.
    perp = [i for i in range(3) if i != asm_ax_idx]  # e.g. [1,2] if asm=X

    bolt_cyls = []
    for s in cyl_shapes:
        c = s.get("center", [0, 0, 0])
        try:
            perp_r = math.sqrt(float(c[perp[0]])**2 + float(c[perp[1]])**2)
        except (TypeError, ValueError, IndexError):
            perp_r = 0
        if perp_r > 8.0:     # must be off-axis (bolt-circle threshold)
            bolt_cyls.append(s)

    if len(bolt_cyls) >= 2:
        angles = []
        for s in bolt_cyls:
            c = s.get("center", [0, 0, 0])
            try:
                angles.append(math.atan2(float(c[perp[1]]), float(c[perp[0]])))
            except (TypeError, ValueError):
                pass
        angles.sort()
        if len(angles) >= 2:
            expected_gap = 2 * math.pi / len(angles)
            diffs = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi)
                     for i in range(len(angles))]
            regularity  = max(0.0, 1.0 - sum(abs(d - expected_gap) for d in diffs)
                              / (2 * math.pi))
            count_ratio = min(len(bolt_cyls) / N_bolts, 1.0)
            bolt_score  = 0.55 * regularity + 0.45 * count_ratio
        else:
            bolt_score = 0.0
    else:
        bolt_score = max(0.0, n_cyls / N_bolts * 0.20)

    # ── 4. Bolt-flange colocation (15%) ────────────────────────────────────
    # Bolts should sit at the same axial coordinate as the flange rings.
    # Flange rings = pipes with larger outer_r; bolt cylinders share their X/Y/Z.
    if pipe_shapes and bolt_cyls:
        outer_rs = [s.get("outer_radius", 0) or 0 for s in pipe_shapes]
        if outer_rs:
            mean_or = sum(outer_rs) / len(outer_rs)
            flange_pipes = [s for s in pipe_shapes
                            if (s.get("outer_radius") or 0) >= mean_or * 0.9]
        else:
            flange_pipes = pipe_shapes

        flange_ax_vals = []
        for fp in flange_pipes:
            c = fp.get("center", [0, 0, 0])
            try:
                flange_ax_vals.append(float(c[asm_ax_idx]))
            except (TypeError, ValueError, IndexError):
                pass

        if flange_ax_vals:
            coloc_scores = []
            for bc in bolt_cyls:
                c = bc.get("center", [0, 0, 0])
                try:
                    bx = float(c[asm_ax_idx])
                except (TypeError, ValueError, IndexError):
                    coloc_scores.append(0.0)
                    continue
                closest = min(abs(bx - fx) for fx in flange_ax_vals)
                coloc_scores.append(max(0.0, 1.0 - closest / 20.0))
            bolt_flange_score = sum(coloc_scores) / len(coloc_scores)
        else:
            bolt_flange_score = 0.5
    else:
        bolt_flange_score = 0.0 if not bolt_cyls else 0.5

    # ── 5. Nut/torus count (10%) ────────────────────────────────────────────
    torus_score = min(n_tori / max(N_bolts, 1), 1.0)

    raw = (pipe_score * 0.15
           + axis_score  * 0.35
           + bolt_score  * 0.25
           + bolt_flange_score * 0.15
           + torus_score * 0.10)

    # ── Hard gates ──────────────────────────────────────────────────────────
    if n_pipes == 0:
        raw = min(raw, 0.20)   # no pipes → fundamentally wrong
    elif n_pipes == 1:
        raw = min(raw, 0.35)   # only one pipe → near-failure

    # Missing axis = wrong 3D rendering → cannot exceed 0.38 regardless of
    # how many pipes or how regular the bolt circle is.
    if axis_fraction < 0.5:
        raw = min(raw, 0.38)

    return max(0.0, min(1.0, raw))


def _sem_ball_bearing(shapes, golden, scale):
    """
    Ball Bearing Assembly semantic evaluator.

    Four structural checks:
      40% — race completeness  (MUST have inner + outer race as pipes)
      35% — pitch-circle regularity  (N spheres equidistant from centre, equally spaced)
      15% — sphere count accuracy
      10% — race concentricity  (inner and outer pipe share the same XY centre)

    Hard gates:
      0 pipes → capped at 0.20  (no races = completely wrong)
      1 pipe  → capped at 0.45  (missing one race = major structural failure)
    """
    from collections import Counter
    if not shapes:
        return 0.0

    N_balls = scale   # L1=6, L2=12, L3=20
    types   = Counter(s.get("type") for s in shapes)
    n_pipes  = types.get("pipe", 0)
    n_spheres = types.get("sphere", 0)

    # ── 1. Race completeness (40%) ──────────────────────────────────────────
    # The inner + outer race are the load-bearing skeleton.  Without them
    # the assembly is structurally meaningless regardless of other shapes.
    if n_pipes == 0:
        race_score = 0.0          # no races at all
    elif n_pipes == 1:
        race_score = 0.25         # only one race — missing inner or outer
    else:
        race_score = min(n_pipes / 2.0, 1.0)

    # ── 2. Pitch-circle regularity (35%) ───────────────────────────────────
    sphere_shapes = [s for s in shapes if s.get("type") == "sphere"]
    if len(sphere_shapes) >= 2:
        pts = [(s.get("center", [0, 0, 0])[0], s.get("center", [0, 0, 0])[1])
               for s in sphere_shapes]
        radii = [math.sqrt(x ** 2 + y ** 2) for x, y in pts]
        avg_r = sum(radii) / len(radii) if radii else 0
        if avg_r > 1.0:
            r_variance = sum(abs(r - avg_r) for r in radii) / (avg_r * len(radii))
            radius_score = max(0.0, 1.0 - r_variance * 3.0)
        else:
            radius_score = 0.0

        angles = sorted(math.atan2(y, x) for x, y in pts)
        if len(angles) >= 2:
            diffs = [(angles[(i + 1) % len(angles)] - angles[i]) % (2 * math.pi)
                     for i in range(len(angles))]
            expected = 2 * math.pi / len(angles)
            regularity = max(0.0, 1.0 - sum(abs(d - expected) for d in diffs) / (2 * math.pi))
        else:
            regularity = 0.5

        circle_score = 0.50 * radius_score + 0.50 * regularity
    else:
        circle_score = 0.0

    # ── 3. Sphere count (15%) ──────────────────────────────────────────────
    sphere_ratio = min(n_spheres / max(N_balls, 1), 1.0)
    if n_spheres > N_balls * 1.5:
        sphere_ratio *= 0.6       # heavy penalty for massive over-generation

    # ── 4. Race concentricity (10%) ─────────────────────────────────────────
    pipe_list = [s for s in shapes if s.get("type") == "pipe"]
    if len(pipe_list) >= 2:
        c0 = pipe_list[0].get("center", [0, 0, 0])
        spread_xy = max(
            math.sqrt((s.get("center", [0, 0, 0])[0] - c0[0]) ** 2
                      + (s.get("center", [0, 0, 0])[1] - c0[1]) ** 2)
            for s in pipe_list
        )
        concentric_score = 1.0 if spread_xy < 3.0 else max(0.0, 1.0 - spread_xy / 30.0)
    else:
        concentric_score = 0.0

    raw = (race_score * 0.40 + circle_score * 0.35
           + sphere_ratio * 0.15 + concentric_score * 0.10)

    # Apply hard gates
    if n_pipes == 0:
        raw = min(raw, 0.20)
    elif n_pipes == 1:
        raw = min(raw, 0.45)

    return max(0.0, min(1.0, raw))


def _sem_engineering(shapes, golden, specs):
    """
    Generic semantic evaluator for new engineering families
    (Clock Tower, Gantry Crane, Cochlear Spiral, Radiolarian Skeleton).

    Lighter than the family-specific evaluators — uses geometry-based
    matching since these families don't have simple hard-gate constraints:
      45% — type distribution match (correct primitive mix)
      30% — count accuracy  (penalises missing AND excess shapes)
      15% — concentric / mate constraints from specs
      10% — no-interference
    """
    from collections import Counter
    if not shapes:
        return 0.0

    # 1. Type distribution match
    g_types = Counter(s.get("type") for s in golden)
    s_types = Counter(s.get("type") for s in shapes)
    all_types = set(g_types.keys()) | set(s_types.keys())
    if all_types:
        type_score = max(0.0,
            1.0 - sum(abs(g_types.get(t, 0) - s_types.get(t, 0))
                      for t in all_types) / max(sum(g_types.values()), 1))
    else:
        type_score = 0.0

    # 2. Count accuracy (penalise over-generation at half rate)
    ratio = min(len(shapes), len(golden)) / max(len(golden), 1)
    over_penalty = max(0, len(shapes) - len(golden)) / max(len(golden), 1) * 0.5
    count_score = max(0.0, ratio - over_penalty)

    # 3. Mate / concentric constraints
    mates = specs.get("mates", []) if isinstance(specs, dict) else []
    if mates:
        mate_scores = []
        for mate in mates:
            if mate.get("type") == "concentric":
                ids = mate.get("ids", [])
                centers = []
                for sid in ids:
                    for s in shapes:
                        if s.get("id") == sid:
                            c = s.get("center")
                            if c and len(c) >= 2:
                                centers.append(c)
                            break
                if len(centers) >= 2:
                    xs = [c[0] for c in centers]
                    ys = [c[1] for c in centers]
                    spread = max(max(xs) - min(xs), max(ys) - min(ys))
                    mate_scores.append(1.0 if spread < 5.0
                                       else max(0.0, 1.0 - spread / 50.0))
        mate_score = sum(mate_scores) / max(len(mate_scores), 1) if mate_scores else 0.5
    else:
        mate_score = 0.5  # neutral

    # 4. Interference
    interf_score = _phys_interference(shapes, tol_mm=2.0)

    raw = (type_score * 0.45 + count_score * 0.30
           + mate_score * 0.15 + interf_score * 0.10)
    return max(0.0, min(1.0, raw))


# ── Semantic dispatcher ───────────────────────────────────────

_SEM_DISPATCH = {
    "Spiral Staircase":   lambda s, g, sc, sp, o: _sem_staircase(s, g, sc),
    "Cannonball Pyramid":  lambda s, g, sc, sp, o: _sem_pyramid(s, g, sc),
    "Voxel Grid":          lambda s, g, sc, sp, o: _sem_voxel(s, g, sc),
    "Domino Ring":         lambda s, g, sc, sp, o: _sem_stonehenge(s, g, sc),
    "DNA Helix":           lambda s, g, sc, sp, o: _sem_dna(s, g, sc),
    "Suspension Bridge":   lambda s, g, sc, sp, o: _sem_bridge(s, g, sc),
    "Planetary Array":     lambda s, g, sc, sp, o: _sem_planetary(s, g, sc),
    "Cross-Braced Truss":  lambda s, g, sc, sp, o: _sem_truss(s, g, sc),
    "Fractal Y-Tree":      lambda s, g, sc, sp, o: _sem_fractal(s, g, sc),
    "BCC Lattice":         lambda s, g, sc, sp, o: _sem_bcc(s, g, sc),
    "Furniture Assembly":  lambda s, g, sc, sp, o: _sem_furniture(s, o, sp),
    "Pipe Manifold":       lambda s, g, sc, sp, o: _sem_manifold(s, o, sp),
    "Axle Bearing":         lambda s, g, sc, sp, o: _sem_axle(s, o, sp),
    "Phyllotaxis Disc":    lambda s, g, sc, sp, o: _sem_phyllotaxis(s, g, sc),
    "Compound Eye":        lambda s, g, sc, sp, o: _sem_compound_eye(s, g, sc),
    "Diatom Frustule":     lambda s, g, sc, sp, o: _sem_diatom(s, g, sc),
    "Honeycomb Lattice":   lambda s, g, sc, sp, o: _sem_honeycomb(s, g, sc),
    "Armillary Sphere":    lambda s, g, sc, sp, o: _sem_radiolarian(s, g, sc if sc else 2),
    "Vertebral Column":    lambda s, g, sc, sp, o: _sem_vertebral(s, g, sc),
    "Flanged Pipe Joint":  lambda s, g, sc, sp, o: _sem_flanged_pipe(s, g, sc),
    "Ball Bearing Assembly":lambda s, g, sc, sp, o: _sem_ball_bearing(s, g, sc),
    "Clock Tower Mechanism":lambda s, g, sc, sp, o: _sem_engineering(s, g, sp),
    "Gantry Crane Assembly":lambda s, g, sc, sp, o: _sem_engineering(s, g, sp),
    "Cochlear Spiral":     lambda s, g, sc, sp, o: _sem_engineering(s, g, sp),
    "Radiolarian Skeleton":lambda s, g, sc, sp, o: _sem_engineering(s, g, sp),
}

# ── Golden baseline cache for Sem normalization ──────────────
# The raw Sem score depends on validator limitations (e.g. bbox-based
# interference flags tangent contacts, gravity checker misses nested
# pockets).  To make scores interpretable across families we normalize:
#   Sem_normalized = raw_llm / raw_golden * 100
# so the golden reference always benchmarks at 100%.
_SEM_BASELINE = {}   # key: (family, scale) → raw float [0..1]

def _sem_raw(family, shapes, golden, scale, specs=None, ops=None):
    """Raw semantic score (0.0–1.0) before golden-baseline normalization."""
    fn = _SEM_DISPATCH.get(family)
    if not fn:
        print(f"  [WARN] No semantic evaluator for family '{family}' — skipping Sem score")
        return 0.0
    try:
        raw = fn(shapes, golden, scale, specs or {}, ops or [])
        return max(0.0, min(1.0, raw))
    except Exception as e:
        print(f"  [WARN] Semantic evaluator for '{family}' raised {type(e).__name__}: {e}")
        return 0.0

def _get_sem_baseline(family, golden, scale, specs=None):
    """Compute and cache the golden reference's raw Sem score."""
    key = (family, scale)
    if key not in _SEM_BASELINE:
        normed_golden = _normalize_shapes(golden)
        baseline = _sem_raw(family, normed_golden, golden, scale, specs)
        if baseline < 0.01:
            print(f"  [WARN] Golden baseline for '{family}' scale={scale} is {baseline:.4f} — "
                  f"setting to 1.0 (check evaluator). Golden has {len(normed_golden)} shapes.")
            baseline = 1.0
        _SEM_BASELINE[key] = baseline
    return _SEM_BASELINE[key]

def eval_sem(family, shapes, golden, scale, specs=None, ops=None, geom_score=0):
    """Unified Semantic score, normalized against golden baseline.
    Golden reference always → 100%.  LLM output scored proportionally.

    STRICT gating (v2):
      - Coverage gate: exponent raised to 1.5 (was 0.6) — missing shapes
        now devastate the semantic score instead of getting a free pass.
        5% coverage → ~1% sem, 30% → ~16% sem, 70% → ~59% sem.
      - Geometry quality gate: if geometry accuracy is below 20%, the output
        is spatially incoherent — semantic structural checks are meaningless.
        geom < 10%  → sem multiplied by 0.0 (complete garbage)
        geom 10-20% → linear ramp from 0.0 to 1.0
        geom >= 20% → no penalty from this gate
    Returns 0–100."""
    raw = _sem_raw(family, shapes, golden, scale, specs, ops)
    baseline = _get_sem_baseline(family, golden, scale, specs)
    normalized = raw / baseline

    # ── Coverage gate (STRICT) ────────────────────────────────
    n_actual = len(shapes)
    n_golden = len(_normalize_shapes(golden)) if golden else 1
    coverage_ratio = min(1.0, n_actual / max(1, n_golden))
    # Power > 1 means partial coverage gets crushed hard.
    coverage_gate = coverage_ratio ** 1.5

    # ── Geometry quality gate ─────────────────────────────────
    # If the geometry score is terrible, the model didn't produce a real
    # 3D assembly — semantic checks on random shapes are meaningless noise.
    geom_frac = geom_score / 100.0  # convert 0-100 → 0-1
    if geom_frac < 0.10:
        geom_gate = 0.0   # below 10% geom → semantic is zero
    elif geom_frac < 0.20:
        geom_gate = (geom_frac - 0.10) / 0.10  # linear ramp 10%-20%
    else:
        geom_gate = 1.0

    gated = normalized * coverage_gate * geom_gate
    return round(max(0.0, min(1.0, gated)) * 100)


def eval_global(cov, geom, sem):
    """Global composite score with structural validity gate.

    Weights: Cov 20%, Geom 30%, Sem 50%.
    Calibration: power-curve (s/100)^1.3 × 100 compresses the upper
    range so that even strong outputs rarely exceed ~70%.

    Structural validity gate:
      geom < 5%   → global × 0.0  (no real output)
      geom 5–15%  → linear ramp 0.0 → 1.0
      geom >= 15% → no penalty from validity gate

    Returns 0–100."""
    raw_global = cov * 0.20 + geom * 0.30 + sem * 0.50

    # Structural validity gate — geometry is the hard proof of real output
    geom_frac = geom / 100.0
    if geom_frac < 0.05:
        validity = 0.0
    elif geom_frac < 0.15:
        validity = (geom_frac - 0.05) / 0.10
    else:
        validity = 1.0

    gated = raw_global * validity
    # Calibration curve: (x/100)^1.3 × 100
    calibrated = ((gated / 100.0) ** 1.3) * 100.0
    return round(calibrated)


# ═══════════════════════════════════════════════════════════════
# LLM INVOCATION
# ═══════════════════════════════════════════════════════════════

_JSON_SUFFIX = """

CRITICAL OUTPUT RULES — read carefully:
1. Your ENTIRE response must be valid JSON: either a raw array [ {...}, {...}, ... ] or an object {"shapes": [ {...}, {...}, ... ]}.
2. Do NOT wrap in ```json``` or any markdown fence.
3. Do NOT include ANY text before or after the JSON — no explanations, no commentary, no "Here is the result", no thinking.
4. Do NOT use trailing commas in the JSON.
5. Every shape object in the array MUST have "id" (integer), "type" (string), and positional fields.
6. All coordinates must be arrays [x, y, z], NOT objects {"x":..., "y":..., "z":...}."""

def _needed_tokens(n_shapes: int) -> int:
    """
    Compute the minimum output token budget for a response containing n_shapes.
    Each JSON shape object averages ~90 tokens; add 25% buffer.
    Returns at least 8 192 tokens (floor) — never exceeds 65 536 (ceiling).
    """
    return max(8192, min(int(n_shapes * 90 * 1.25), 65536))


def run_single_api(model, prompt, max_tokens: int = None):
    """Call an LLM with retry logic. Returns the raw text response."""
    cp = prompt + _JSON_SUFFIX
    timeout = 300
    delay = 10
    for attempt in range(3):
        try:
            resp = call_llm(cp, model, timeout=timeout, max_tokens=max_tokens)
            # Quick sanity: if response is clearly an error or empty, retry
            if not resp or resp.strip() in ("", "Crash", "[TIMEOUT]"):
                if attempt < 2:
                    print(f"    ⏳ Empty response ({model}), retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
            # If response starts with [ERROR], retry with backoff
            if resp and resp.strip().startswith("[ERROR]"):
                if attempt < 2:
                    print(f"    ⏳ API error ({model}): {resp[:80]}, retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
            return resp
        except Exception as e:
            if "exhausted" in str(e).lower() or "429" in str(e):
                print(f"    ⏳ Rate-limited ({model}), retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                if attempt < 2:
                    print(f"    ⚠ Exception ({model}): {e}, retrying...")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    return "Crash"
    return "Crash"


def process_task(model, test_meta, scale, diff, max_retries=3):
    """Run one (model, test, scale) combination and return scored results.
    Retries up to max_retries times if the result is all-zero (empty/parse fail).

    New scoring architecture (all phases):
      Cov   — coverage (shape count)
      Geom  — geometry accuracy (position + type + dimensions)
      Sem   — semantic (subsumes physics + structural constraints per family)
      Global — weighted composite: Cov 20% + Geom 30% + Sem 50%
    """
    DEBUG = os.environ.get("C2CAD_DEBUG", os.environ.get("CG3D_DEBUG", "0")) == "1"

    for attempt in range(1, max_retries + 1):
        # Call the generator — may return (prompt, shapes_list) or (prompt, specs_dict)
        _gen_result = test_meta["func"](scale)
        prompt, _second = _gen_result
        _has_specs = isinstance(_second, dict) and "reference" in _second
        if not _has_specs:
            golden = _second
            llm_resp = run_single_api(model, prompt,
                                      max_tokens=_needed_tokens(len(golden)))
            if DEBUG:
                preview = (llm_resp or "")[:300].replace("\n", "\\n")
                print(f"  [DEBUG] {model} | {test_meta['family']} | attempt={attempt} | resp_len={len(llm_resp or '')} | preview: {preview}")
            parsed = extract_json(llm_resp) if llm_resp not in ["Crash", None, ""] else None
            raw_shapes = parsed if isinstance(parsed, list) else []
            shapes = _normalize_shapes(raw_shapes)
            if DEBUG:
                print(f"  [DEBUG] parsed {len(shapes)} shapes (golden={len(golden)})")
            if shapes:
                c, g = eval_cov_geom(raw_shapes, golden)
                sem = eval_sem(test_meta["family"], shapes, golden, scale, geom_score=g)
                gl = eval_global(c, g, sem)
                if c > 0 or g > 0 or sem > 0:
                    if attempt > 1:
                        print(f"    ↳ retry {attempt} succeeded for {model} | {test_meta['family']}")
                    return {"model": model, "family": test_meta["family"], "diff": diff,
                            "shapes": shapes, "cov": c, "geom": g, "sem": sem, "glob": gl,
                            "phase": test_meta["phase"]}
            if attempt < max_retries:
                print(f"    ↳ attempt {attempt} returned 0 — retrying {model} | {test_meta['family']} ...")
        else:
            # Has specs dict with engineering constraints (Phase 3 / new engineering families)
            specs = _second
            ref_shapes = specs.get("reference", [])
            llm_resp = run_single_api(model, prompt,
                                      max_tokens=_needed_tokens(len(ref_shapes)))
            parsed = extract_json(llm_resp) if llm_resp not in ["Crash", None, ""] else None
            raw = parsed if isinstance(parsed, list) else []
            shapes = _normalize_shapes(raw)
            ops = [r for r in raw if isinstance(r, dict) and "op" in r]
            # Cov + Geom against reference shapes
            if ref_shapes and shapes:
                c, g = eval_cov_geom(raw, ref_shapes)
            else:
                c, g = 0, 0
            # Sem (includes physics + engineering constraints)
            sem = eval_sem(test_meta["family"], shapes, ref_shapes, scale, specs, ops, geom_score=g)
            gl = eval_global(c, g, sem)
            total = c + g + sem
            if total > 0:
                if attempt > 1:
                    print(f"    ↳ retry {attempt} succeeded for {model} | {test_meta['family']}")
                return {"model": model, "family": test_meta["family"], "diff": diff,
                        "shapes": shapes, "cov": c, "geom": g, "sem": sem, "glob": gl,
                        "phase": test_meta["phase"]}
            if attempt < max_retries:
                print(f"    ↳ attempt {attempt} returned 0 — retrying {model} | {test_meta['family']} ...")

    # All retries exhausted — return zero result
    return {"model": model, "family": test_meta["family"], "diff": diff,
            "shapes": [], "cov": 0, "geom": 0, "sem": 0, "glob": 0,
            "phase": test_meta["phase"]}


# ═══════════════════════════════════════════════════════════════
# DATABASE I/O
# ═══════════════════════════════════════════════════════════════

def load_db(db_path, target_models):
    """
    Load or create the showcase_db.js database.
    The 'models' dict is keyed by model alias (the string passed to --model).
    New model keys are added dynamically — no hard-coded model list.
    """
    # 1. Build goldens
    goldens = []
    for test in ALL_TESTS:
        for idx, scale in enumerate(test["scales"]):
            diff = idx + 1
            prompt, _second = test["func"](scale)
            if isinstance(_second, dict) and "reference" in _second:
                ref_shapes = _second.get("reference", [])
            else:
                ref_shapes = _second
            goldens.append({
                "family": test["family"],
                "difficultyLabel": f"Level {diff} (Scale {scale})",
                "difficultyID": diff,
                "shapes": ref_shapes,
                "prompt": prompt,
                "phase": test["phase"]
            })

    db = {"golden": goldens, "models": {}}

    # 2. Load existing DB from disk (preserve all previous model results)
    if os.path.exists(db_path):
        with open(db_path, "r") as f:
            try:
                content = f.read().replace("window.SHOWCASE_DB = ", "").replace(";\n", "")
                disk_db = json.loads(content)
                for m, results in disk_db.get("models", {}).items():
                    db["models"][m] = [None] * len(goldens)
                    for disk_res in (results or []):
                        if disk_res is None:
                            continue
                        for i, g in enumerate(goldens):
                            if g["family"] == disk_res.get("family") and g["difficultyID"] == disk_res.get("difficultyID"):
                                db["models"][m][i] = disk_res
                                break
            except Exception:
                pass

    # 3. Ensure every target model has a slot array
    for m in target_models:
        if m not in db["models"]:
            db["models"][m] = [None] * len(goldens)
        while len(db["models"][m]) < len(goldens):
            db["models"][m].append(None)

    # 4. Fill empty slots with zeroed placeholders
    for m in db["models"]:
        while len(db["models"][m]) < len(goldens):
            db["models"][m].append(None)
        for i, res in enumerate(db["models"][m]):
            if res is None:
                g = goldens[i]
                db["models"][m][i] = {
                    "family": g["family"], "difficultyLabel": g["difficultyLabel"],
                    "difficultyID": g["difficultyID"], "shapes": [],
                    "score_cov": 0, "score_geom": 0, "score_sem": 0, "score_global": 0,
                    "phase": g["phase"]
                }

    return db


def save_db(db, db_path, quiet=False):
    """Write the DB as a JS global for the HTML pages, then split by phase.

    Uses file-locking so concurrent runs (different models in separate
    terminals) never overwrite each other's results.
    If quiet=True, suppress the "Database saved" message (used for
    incremental saves after every single test).
    """
    import fcntl, time as _time

    lock_path = db_path + ".lock"
    lock_fd = open(lock_path, "w")
    for _attempt in range(10):
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except (IOError, OSError):
            _time.sleep(0.5)
    else:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

    try:
        # Re-read disk to merge in results from other concurrent runs
        disk_db = None
        if os.path.exists(db_path):
            try:
                with open(db_path, "r") as f:
                    content = f.read().replace("window.SHOWCASE_DB = ", "").rstrip().rstrip(";")
                disk_db = json.loads(content)
            except Exception:
                pass

        goldens = db["golden"]
        merged_models = {}
        all_model_keys = set(db["models"].keys())
        if disk_db:
            all_model_keys |= set(disk_db.get("models", {}).keys())

        for m in all_model_keys:
            mem_results = db["models"].get(m, [])
            disk_results = disk_db.get("models", {}).get(m, []) if disk_db else []

            merged = [None] * len(goldens)
            for i, g in enumerate(goldens):
                mem_entry = mem_results[i] if i < len(mem_results) else None
                disk_entry = disk_results[i] if i < len(disk_results) else None

                mem_score = (mem_entry or {}).get("score_global", 0)
                disk_score = (disk_entry or {}).get("score_global", 0)

                if mem_score >= disk_score:
                    merged[i] = mem_entry
                else:
                    merged[i] = disk_entry

                if merged[i] is None:
                    merged[i] = {
                        "family": g["family"], "difficultyLabel": g["difficultyLabel"],
                        "difficultyID": g["difficultyID"], "shapes": [],
                        "score_cov": 0, "score_geom": 0, "score_sem": 0, "score_global": 0,
                        "phase": g["phase"]
                    }
            merged_models[m] = merged

        final_db = {"golden": goldens, "models": merged_models}

        tmp_path = db_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write("window.SHOWCASE_DB = ")
            json.dump(final_db, f, separators=(',', ':'))
            f.write(";\n")
        os.replace(tmp_path, db_path)

        if not quiet:
            print(f"\n✅ Database saved (merge-safe) → {db_path}")

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            os.remove(lock_path)
        except OSError:
            pass

    # Split into per-phase files for lazy loading in the visualiser
    try:
        import importlib.util
        splitter_path = os.path.join(os.path.dirname(__file__), "split_database.py")
        spec = importlib.util.spec_from_file_location("split_database", splitter_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.split()
    except Exception as e:
        if not quiet:
            print(f"⚠  Phase split warning: {e}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="C2CAD-Bench unified live benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_unified.py --all                             # All tests × all default models
  python run_unified.py --all --model claude-sonnet       # All tests, Claude only
  python run_unified.py --test "Bridge" --model deepseek  # One test, DeepSeek
  python run_unified.py --phase 1 --model claude-haiku    # Phase 1 only
  python run_unified.py --list-models                     # Show all supported models
        """)

    parser.add_argument("--all", action="store_true",
                        help="Run all tests (default models unless --model given)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="Run only Phase 1, 2, or 3")
    parser.add_argument("--test", type=str,
                        help="Run a specific test family (substring match, e.g. 'Bridge')")
    parser.add_argument("--scale", type=int,
                        help="Target a specific scale level within a test")
    parser.add_argument("--model", type=str, action="append", dest="models",
                        help="Target model(s). Can be repeated: --model claude --model deepseek")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tests where the model already has non-zero scores in the DB")
    parser.add_argument("--redo", action="store_true",
                        help="Re-run only tests that have zero scores (failed/crashed). "
                             "Complements --skip-existing: skip successes, redo failures.")
    parser.add_argument("--zero-missing", action="store_true",
                        help="Fill in missing/empty DB slots with zero scores instead of running them. "
                             "Use to mark un-run tests as 0 without calling the API.")
    parser.add_argument("--workers", type=str, default="auto",
                        help="Max parallel API workers. Default 'auto': 5 for Gemini-only runs, 2 for all others. Pass 'max' for one worker per task.")
    parser.add_argument("--list-models", action="store_true",
                        help="Print all supported model strings and exit")

    args = parser.parse_args()

    if args.list_models:
        print(list_models())
        return

    # Resolve worker count ("max" = one per task, resolved after task list is built)
    _workers_raw = args.workers

    # Resolve target models
    if args.models:
        target_models = []
        for m in args.models:
            if m in ALL_KNOWN:
                target_models.append(m)
            else:
                # Try as a prefix match
                matches = [k for k in ALL_KNOWN if k.startswith(m)]
                if matches:
                    target_models.extend(matches)
                else:
                    print(f"⚠️  Unknown model '{m}'. Passing through as custom CLI.")
                    target_models.append(m)
    else:
        target_models = DEFAULT_MODELS

    db_path = os.path.abspath(os.path.join(base_dir, "../results/showcase_db.js"))
    db = load_db(db_path, target_models)

    # Filter tests
    target_tests = ALL_TESTS
    if not args.all and not args.phase and not args.test:
        target_tests = []
    else:
        if args.phase:
            target_tests = [t for t in target_tests if t["phase"] == args.phase]
        if args.test:
            target_tests = [t for t in target_tests if args.test.lower() in t["family"].lower()]

    # Build task queue
    goldens = db["golden"]
    tasks = []

    # ── --zero-missing: fill empty DB slots with zero scores, no API calls ──
    if args.zero_missing:
        filled = 0
        for t in target_tests:
            scales = [args.scale] if args.scale and args.scale in t["scales"] else t["scales"]
            for scale in scales:
                diff = t["scales"].index(scale) + 1
                for m in target_models:
                    gidx = next(i for i, g in enumerate(goldens)
                                if g["family"] == t["family"] and g["difficultyID"] == diff)
                    model_data = db["models"].get(m, [None] * len(goldens))
                    if m not in db["models"]:
                        db["models"][m] = [None] * len(goldens)
                        model_data = db["models"][m]
                    existing = model_data[gidx]
                    if not existing or not existing.get("family"):
                        g = goldens[gidx]
                        db["models"][m][gidx] = {
                            "family": g["family"],
                            "difficultyLabel": g["difficultyLabel"],
                            "difficultyID": g["difficultyID"],
                            "shapes": [],
                            "score_cov": 0, "score_geom": 0, "score_sem": 0, "score_global": 0,
                            "phase": g["phase"],
                        }
                        filled += 1
        print(f"  Filled {filled} missing slots with zero scores.")
        save_db(db, db_path)
        return

    for t in target_tests:
        scales = [args.scale] if args.scale and args.scale in t["scales"] else t["scales"]
        for scale in scales:
            diff = t["scales"].index(scale) + 1
            for m in target_models:
                gidx = next(i for i, g in enumerate(goldens)
                            if g["family"] == t["family"] and g["difficultyID"] == diff)

                existing = db["models"].get(m, [None] * len(goldens))[gidx]
                if existing:
                    has_score = (existing.get("score_cov", 0) > 0 or
                                 existing.get("score_geom", 0) > 0 or
                                 existing.get("score_sem", 0) > 0 or
                                 existing.get("score_global", 0) > 0)

                    # --skip-existing: skip anything with a real score
                    if args.skip_existing and has_score:
                        continue

                    # --redo: ONLY run tests that have zero scores (failures)
                    if args.redo and has_score:
                        continue
                    # --redo without existing score: run it (it's missing)
                    if args.redo and not has_score:
                        pass  # fall through to append

                    # Default (no flags): run everything
                    if not args.skip_existing and not args.redo:
                        pass  # fall through to append

                tasks.append((m, t, scale, diff, gidx))

    if not tasks:
        print("No executable tasks. Use --all, --phase, or --test to select targets.")
        print(f"Database has {len(db['models'])} model(s): {', '.join(db['models'].keys())}")
        save_db(db, db_path)
        return

    # Resolve worker count now that we know task count
    if _workers_raw == "max":
        n_workers = len(tasks)
    elif _workers_raw == "auto":
        # 5 workers for Gemini-only runs, 2 for all others (including mixed)
        all_gemini = all(m.startswith("gemini") for m, *_ in tasks)
        n_workers = 5 if all_gemini else 2
    else:
        n_workers = int(_workers_raw)

    # Print execution plan
    model_set = sorted(set(m for m, *_ in tasks))
    test_set = sorted(set(t["family"] for _, t, *_ in tasks))
    print(f"\n{'='*65}")
    print(f"  C2CAD-Bench live benchmark")
    print(f"  Models : {', '.join(model_set)}")
    print(f"  Tests  : {', '.join(test_set)}")
    print(f"  Tasks  : {len(tasks)} total | {n_workers} parallel workers")
    print(f"{'='*65}\n")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_task, m, t, scale, diff): (m, gidx)
                   for m, t, scale, diff, gidx in tasks}
        completed = 0
        for future in as_completed(futures):
            m_key, gidx = futures[future]
            try:
                res = future.result()
            except Exception as e:
                print(f"  ❌ {m_key} task crashed: {e}")
                continue
            completed += 1

            out_obj = {
                "family": res["family"],
                "difficultyLabel": db["golden"][gidx]["difficultyLabel"],
                "difficultyID": res["diff"],
                "shapes": res["shapes"],
                "phase": res["phase"]
            }
            out_obj["score_cov"]    = res.get("cov", 0)
            out_obj["score_geom"]   = res.get("geom", 0)
            out_obj["score_sem"]    = res.get("sem", 0)
            out_obj["score_global"] = res.get("glob", 0)
            status = f"Cov:{res.get('cov',0)}% Geom:{res.get('geom',0)}% Sem:{res.get('sem',0)}% Global:{res.get('glob',0)}%"

            print(f"  [{completed:02d}/{len(tasks)}] {res['model']:24s} | {res['family']:20s} L{res['diff']} → {status}")

            # Store in DB under the model alias the user passed
            if res["model"] not in db["models"]:
                db["models"][res["model"]] = [None] * len(goldens)
                for i, g in enumerate(goldens):
                    db["models"][res["model"]][i] = {
                        "family": g["family"], "difficultyLabel": g["difficultyLabel"],
                        "difficultyID": g["difficultyID"], "shapes": [],
                        "score_cov": 0, "score_geom": 0, "score_sem": 0, "score_global": 0,
                        "phase": g["phase"]
                    }
            db["models"][res["model"]][gidx] = out_obj

            # Save immediately after each result (incremental, merge-safe)
            save_db(db, db_path, quiet=True)

    save_db(db, db_path)

    # Print summary
    print(f"\n{'-'*65}")
    print(f"  Summary  ({completed}/{len(tasks)} completed)")
    print(f"{'-'*65}")
    for m in model_set:
        data = db["models"].get(m, [])
        def _has_score(d):
            return (d.get("score_cov",0) + d.get("score_geom",0) + d.get("score_sem",0) + d.get("score_global",0)) > 0
        for phase, label in [(1, "Phase 1"), (2, "Phase 2"), (3, "Phase 3"), (4, "Phase 4")]:
            # Include ALL entries for the phase (zeros count — consistent with HTML display)
            pdata = [d for d in data if d and d.get("phase") == phase]
            # Skip phase only if there are no entries at all (model wasn't run on this phase)
            if not pdata:
                continue
            avg_cov  = round(sum(d.get("score_cov", 0)    for d in pdata) / len(pdata))
            avg_geom = round(sum(d.get("score_geom", 0)   for d in pdata) / len(pdata))
            avg_sem  = round(sum(d.get("score_sem", 0)    for d in pdata) / len(pdata))
            avg_glob = round(sum(d.get("score_global", 0) for d in pdata) / len(pdata))
            print(f"  {m:24s}  {label}: Cov {avg_cov:3d}% | Geom {avg_geom:3d}% | Sem {avg_sem:3d}% | Global {avg_glob:3d}%")
        # Overall model average — all phases, zeros included
        all_data = [d for d in data if d]
        if all_data:
            oa_cov  = round(sum(d.get("score_cov", 0)    for d in all_data) / len(all_data))
            oa_geom = round(sum(d.get("score_geom", 0)   for d in all_data) / len(all_data))
            oa_sem  = round(sum(d.get("score_sem", 0)    for d in all_data) / len(all_data))
            oa_glob = round(sum(d.get("score_global", 0) for d in all_data) / len(all_data))
            print(f"  {m:24s}  OVERALL: Cov {oa_cov:3d}% | Geom {oa_geom:3d}% | Sem {oa_sem:3d}% | Global {oa_glob:3d}%")
        print()
    print()


if __name__ == "__main__":
    main()
