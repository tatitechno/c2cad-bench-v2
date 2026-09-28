"""
Configuration, data structures, system prompt, and shape helper functions.
"""

import os, sys, math, textwrap
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class Config:
    def __init__(self):
        self.PROBE_DIR   = Path(__file__).parent.parent.resolve()
        self.RESULTS_DIR = self.PROBE_DIR / "results"
        self.CODES_DIR   = self.PROBE_DIR / "codes"

        if sys.platform == "win32":
            self.CONDA_ACT = r'call "C:\ProgramData\anaconda3\Scripts\activate.bat" base && '
        else:
            self.CONDA_ACT = ""

        context_json = os.environ.get("C2CAD_CONTEXT_JSON")
        self.CONTEXT_JSON = Path(context_json) if context_json else self.PROBE_DIR / "data" / "context.json"

    def ensure_dirs(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.CODES_DIR.mkdir(parents=True, exist_ok=True)

CFG = Config()


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL SYSTEM PROMPT  (v8.0 — expanded shape protocol)
# ═══════════════════════════════════════════════════════════════

SYSTEM_PREFIX = textwrap.dedent("""\
You are a 3D geometry reasoning engine. Your task is to express a 3D assembly
as a structured JSON array of shape objects.

OUTPUT RULES (strictly follow):
- Print ONLY a valid JSON array as your final output.
- Wrap it in ```json ... ``` code fences.
- Each element is one of the following 7 shape types:

  SOLID PRIMITIVES:
    {"id": <int>, "type": "cylinder", "center": [x,y,z], "radius": <float>, "height": <float>, "axis": [0,0,1]}
    {"id": <int>, "type": "box",      "center": [x,y,z], "size": [width, depth, height]}
    {"id": <int>, "type": "sphere",   "center": [x,y,z], "radius": <float>}
    {"id": <int>, "type": "cone",     "center": [x,y,z], "start_radius": <float>, "end_radius": <float>, "height": <float>, "axis": [0,0,1]}

  HOLLOW / COMPOUND PRIMITIVES:
    {"id": <int>, "type": "torus",    "center": [x,y,z], "ring_radius": <float>, "tube_radius": <float>, "axis": [0,0,1]}
    {"id": <int>, "type": "pipe",     "center": [x,y,z], "inner_radius": <float>, "outer_radius": <float>, "height": <float>, "axis": [0,0,1]}

  LATTICE / BEAM ELEMENTS:
    {"id": <int>, "type": "beam",     "start": [x,y,z], "end": [x,y,z], "start_radius": <float>, "end_radius": <float>}

- All dimensions in millimeters. All coordinates as floats (3 decimal places).
- "center" is the geometric centroid of the shape.
- "axis" is the unit vector along the shape's main symmetry axis.
- A "cone" with end_radius=0 is a sharp cone; with end_radius>0 it is a frustum.
- A "torus" is a ring (doughnut): ring_radius = distance from center to tube center.
- A "pipe" is a hollow cylinder with inner_radius < outer_radius.
- A "beam" is a tapered strut from start to end; its center = midpoint(start, end).
- IDs are sequential starting from 0.

BOOLEAN OPERATIONS (optional, append after shapes):
    {"op": "union",     "parts": [id, id, ...]}
    {"op": "subtract",  "target": id, "tool": id}
    {"op": "intersect", "parts": [id, id, ...]}

Think step by step about the positions, then output the JSON.
""")


# ═══════════════════════════════════════════════════════════════
# SHARED DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class LevelResult:
    level_id:   int
    level_name: str
    skill:      str
    # — LLM response —
    response_ok:   bool  = False
    response_len:  int   = 0
    # — JSON extraction —
    json_ok:       bool  = False
    json_text:     str   = ""
    part_count:    int   = 0          # how many shape objects in LLM output
    expected_parts: int  = 0          # how many the probe expected
    # — Spatial validation —
    positions_checked: int   = 0
    positions_correct: int   = 0
    position_accuracy: float = 0.0    # 0-1
    types_correct:     bool  = False
    ops_correct:       bool  = False
    # — Physics validation (v5.0) —
    connectivity_ok:   bool  = False
    gravity_ok:        bool  = False
    proportions_ok:    bool  = False
    connectivity_score: float = 0.0   # 0-1
    gravity_score:      float = 0.0   # 0-1
    # — Orientation validation (v5.1) —
    orientations_checked: int   = 0
    orientations_correct: int   = 0
    orientation_accuracy: float = 0.0  # 0-1
    # — Reasoning quality —
    reasoning_depth:   float = 0.0
    has_loop_logic:    bool  = False   # LLM used a loop / formula in its thinking
    # — Stress params —
    wave:           int   = 0
    n_parts:        int   = 0
    coord_range:    float = 0.0
    generator_name: str   = ""
    seed:           int   = 0
    # — Aggregate —
    score:   float = 0.0
    failure: str   = ""
    elapsed: float = 0.0
    # — Detailed validation data (for HTML report drill-down) —
    position_details:          list = field(default_factory=list)  # per-shape position check details
    dim_errors:                list = field(default_factory=list)  # per-shape relative dimension errors [0,1]
    orientation_details:       list = field(default_factory=list)  # per-cylinder orientation details
    connectivity_islands:      int  = 0                            # number of disconnected islands
    connectivity_floating_ids: list = field(default_factory=list)  # shape IDs in isolated islands
    gravity_floating_ids:      list = field(default_factory=list)  # shape IDs floating in mid-air
    # Exportable ground truth for visual validation
    ground_truth_json: str = ""   # the correct JSON for rendering or visual validation
    llm_json:          str = ""   # what the LLM produced
    # — v6.0 Engineering validators —
    interference_score:      float = 0.0
    interference_details:    list  = field(default_factory=list)
    clearance_score:         float = 0.0
    clearance_details:       list  = field(default_factory=list)
    wall_ok:                 bool  = True
    wall_details:            list  = field(default_factory=list)
    mate_score:              float = 0.0
    mate_details:            list  = field(default_factory=list)
    chamfer_distance:        float = -1.0   # -1 means not computed
    chamfer_score:           float = 0.0
    symmetry_score:          float = 0.0
    symmetry_details:        list  = field(default_factory=list)
    pattern_score:           float = 0.0
    pattern_details:         list  = field(default_factory=list)
    reasoning_chain_score:   float = 0.0
    reasoning_chain_details: list  = field(default_factory=list)


@dataclass
class FixedLevel:
    id:           int
    name:         str
    skill:        str
    prompt:       str
    ground_truth: List[dict]           # the correct JSON
    check_positions: List[int] = field(default_factory=list)  # which IDs to check
    tol_mm: float = 1.0                # positional tolerance in mm
    # — v6.0 engineering spec fields (optional; activate advanced scoring when present) —
    difficulty_tier:        int  = 1    # 1=foundation, 2=intermediate, 3=advanced
    mate_specs:             list = field(default_factory=list)   # assembly mate constraints
    clearance_specs:        list = field(default_factory=list)   # clearance fit specs
    wall_specs:             list = field(default_factory=list)   # min wall thickness specs
    symmetry_spec:          dict = field(default_factory=dict)   # symmetry plane + ref IDs
    pattern_spec:           dict = field(default_factory=dict)   # circular or linear pattern
    expected_intermediates: list = field(default_factory=list)   # intermediate calc values for reasoning chain


# ═══════════════════════════════════════════════════════════════
# SHAPE HELPER FUNCTIONS  (7 canonical shape types)
# ═══════════════════════════════════════════════════════════════

# ── Original 3 primitives ─────────────────────────────────────

def _cyl(id, cx, cy, cz, r, h, ax=None):
    return {"id": id, "type": "cylinder",
            "center": [round(cx,3), round(cy,3), round(cz,3)],
            "radius": round(r,3), "height": round(h,3),
            "axis": ax or [0,0,1]}

def _box(id, cx, cy, cz, w, d, h):
    return {"id": id, "type": "box",
            "center": [round(cx,3), round(cy,3), round(cz,3)],
            "size": [round(w,3), round(d,3), round(h,3)]}

def _sph(id, cx, cy, cz, r):
    return {"id": id, "type": "sphere",
            "center": [round(cx,3), round(cy,3), round(cz,3)],
            "radius": round(r,3)}

# Cone / Frustum

def _cone(id, cx, cy, cz, r_start, r_end, h, ax=None):
    """Cone or frustum. r_end=0 → sharp cone."""
    return {"id": id, "type": "cone",
            "center": [round(cx,3), round(cy,3), round(cz,3)],
            "start_radius": round(r_start,3),
            "end_radius": round(r_end,3),
            "height": round(h,3),
            "axis": ax or [0,0,1]}

# Torus / Ring

def _torus(id, cx, cy, cz, ring_r, tube_r, ax=None):
    """Torus: ring_radius = major, tube_radius = minor."""
    return {"id": id, "type": "torus",
            "center": [round(cx,3), round(cy,3), round(cz,3)],
            "ring_radius": round(ring_r,3),
            "tube_radius": round(tube_r,3),
            "axis": ax or [0,0,1]}

# Pipe / Hollow Cylinder

def _pipe(id, cx, cy, cz, r_inner, r_outer, h, ax=None):
    """Hollow cylinder: inner_radius < outer_radius."""
    return {"id": id, "type": "pipe",
            "center": [round(cx,3), round(cy,3), round(cz,3)],
            "inner_radius": round(r_inner,3),
            "outer_radius": round(r_outer,3),
            "height": round(h,3),
            "axis": ax or [0,0,1]}

# Tapered Beam / Strut

def _beam(id, sx, sy, sz, ex, ey, ez, r_start, r_end):
    """Tapered cylindrical beam from start to end."""
    return {"id": id, "type": "beam",
            "start": [round(sx,3), round(sy,3), round(sz,3)],
            "end":   [round(ex,3), round(ey,3), round(ez,3)],
            "start_radius": round(r_start,3),
            "end_radius":   round(r_end,3)}

# ── Convenience: compute beam center for validation ──────────

def beam_center(b):
    """Return the geometric center (midpoint) of a beam shape dict."""
    s, e = b["start"], b["end"]
    return [round((s[i]+e[i])/2, 3) for i in range(3)]
