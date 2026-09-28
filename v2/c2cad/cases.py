"""Build the C2CAD-Bench v2 case set from the (v2-prompt) generators in stages/.

The generators are NOT modified. Every deviation from their output is an explicit,
documented patch in PATCHES / PROMPT_PATCHES below, so the paper can list each one.

Each case:
  case_id, family, phase, level, scale,
  prompt_body            task text from the generator (without the generator's output suffix)
  prompt                 prompt_body + SCHEMA_CONTRACT (what models receive)
  reference              canonical JSON parts, ids 0..n-1 in the order the prompt prescribes
  symbolic_ids           parts the prompt declares symbolic/embedded (may overlap other parts)
  cut_ids                subtractive envelopes (excluded from occupied-volume metrics)
  patches                list of patch names applied
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from .geom import normalize

ROOT = Path(__file__).resolve().parents[2]
for sub in ("phase1_basic", "phase2_advanced", "phase3_semantic", "phase4_bioinspired"):
    p = str(ROOT / "stages" / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

GENERATOR_SUFFIX_MARKER = "Use millimetres and the canonical"

SCHEMA_CONTRACT = """
OUTPUT CONTRACT (identical for every task)
Return only JSON: an array of part objects, or {"shapes": [...]}. No markdown, no prose.
Every part has an integer "id" and a "type". Units are millimetres. Directions are unit vectors.
  box      {"id","type":"box","center":[x,y,z],"size":[sx,sy,sz]}  sides along world X, Y, Z
  cylinder {"id","type":"cylinder","center":[x,y,z],"axis":[ax,ay,az],"radius":r,"height":h}
  pipe     {"id","type":"pipe","center":[x,y,z],"axis":[ax,ay,az],"inner_radius":ri,"outer_radius":ro,"height":h}
  cone     {"id","type":"cone","center":[x,y,z],"axis":[ax,ay,az],"base_radius":rb,"top_radius":rt,"height":h}
  sphere   {"id","type":"sphere","center":[x,y,z],"radius":r}
  torus    {"id","type":"torus","center":[x,y,z],"axis":[ax,ay,az],"ring_radius":R,"tube_radius":r}
  beam     {"id","type":"beam","start":[x,y,z],"end":[x,y,z],"width":w,"height":t}
Conventions: "center" is the midpoint of the part's extent along its axis (for a cone, the point
halfway between base face and tip face, not the volume centroid). A cone's axis points from its
base face to its tip face. "height" of cylinder, pipe and cone is the length along the axis.
A torus axis is normal to its ring plane. A beam is a straight rectangular bar whose centerline
runs from start to end; a square section s means width = height = s.
"""

# ---------------------------------------------------------------------------
# Family registry (same generators, scales and phases as the v1 runner)
# ---------------------------------------------------------------------------
from generate_staircase import generate_staircase                      # noqa: E402
from generate_pyramid import generate_pyramid                          # noqa: E402
from generate_rubiks import generate_rubiks                            # noqa: E402
from generate_stonehenge import generate_stonehenge                    # noqa: E402
from generate_dna import generate_dna                                  # noqa: E402
from generate_bridge_v2 import generate_bridge                         # noqa: E402
from generate_planetary_v2 import generate_planetary                   # noqa: E402
from generate_truss_v2 import generate_truss                           # noqa: E402
from generate_fractal_v2 import generate_fractal                       # noqa: E402
from generate_bcc_v2 import generate_bcc                               # noqa: E402
from generate_human_furniture import generate_furniture                # noqa: E402
from generate_human_manifold import generate_manifold                  # noqa: E402
from generate_human_axle import generate_axle                          # noqa: E402
from generate_phyllotaxis import generate_phyllotaxis                  # noqa: E402
from generate_compound_eye import generate_compound_eye                # noqa: E402
from generate_diatom import generate_diatom                            # noqa: E402
from generate_honeycomb import generate_honeycomb                      # noqa: E402
from generate_radiolarian import generate_radiolarian                  # noqa: E402
from generate_vertebral import generate_vertebral                      # noqa: E402
from generate_flange_bolt_circle import generate_flange                # noqa: E402
from generate_ball_bearing import generate_ball_bearing                # noqa: E402
from generate_human_clock import generate_clock                        # noqa: E402
from generate_human_gantry import generate_gantry                      # noqa: E402
from generate_cochlea import generate_cochlea                          # noqa: E402
from generate_radiolarian_skeleton import generate_radiolarian as generate_radiolarian_skel  # noqa: E402

FAMILIES = [
    ("Spiral Staircase", generate_staircase, [10, 24, 50], 1),
    ("Cannonball Pyramid", generate_pyramid, [3, 4, 5], 1),
    ("Voxel Grid", generate_rubiks, [2, 3, 4], 1),
    ("Domino Ring", generate_stonehenge, [5, 10, 20], 1),
    ("DNA Helix", generate_dna, [5, 10, 20], 1),
    ("Flanged Pipe Joint", generate_flange, [4, 8, 16], 1),
    ("Suspension Bridge", generate_bridge, [5, 10, 20], 2),
    ("Planetary Array", generate_planetary, [6, 12, 18], 2),
    ("Cross-Braced Truss", generate_truss, [2, 4, 8], 2),
    ("Fractal Y-Tree", generate_fractal, [2, 3, 4], 2),
    ("BCC Lattice", generate_bcc, [2, 3, 4], 2),
    ("Ball Bearing Assembly", generate_ball_bearing, [6, 12, 20], 2),
    ("Furniture Assembly", generate_furniture, [2, 3, 4], 3),
    ("Pipe Manifold", generate_manifold, [2, 3, 5], 3),
    ("Axle Bearing", generate_axle, [1, 2, 3], 3),
    ("Armillary Sphere", generate_radiolarian, [2, 3, 4], 3),
    ("Clock Tower Mechanism", generate_clock, [4, 12, 24], 3),
    ("Gantry Crane Assembly", generate_gantry, [2, 4, 6], 3),
    ("Phyllotaxis Disc", generate_phyllotaxis, [21, 55, 89], 4),
    ("Compound Eye", generate_compound_eye, [2, 3, 4], 4),
    ("Diatom Frustule", generate_diatom, [4, 7, 10], 4),
    ("Honeycomb Lattice", generate_honeycomb, [1, 2, 3], 4),
    ("Vertebral Column", generate_vertebral, [7, 12, 19], 4),
    ("Cochlear Spiral", generate_cochlea, [12, 24, 36], 4),
    ("Radiolarian Skeleton", generate_radiolarian_skel, [1, 2, 3], 4),
]
FAMILY_INDEX = {f[0]: f for f in FAMILIES}


def case_id(family: str, level: int) -> str:
    return family.lower().replace(" ", "_").replace("-", "_") + f"_level_{level}"


# ---------------------------------------------------------------------------
# Symbolic / cut declarations, taken from the prompt wording of each family.
# Expressed as predicates over (reference index, part dict) after patching.
# ---------------------------------------------------------------------------
def _ids_where(ref, pred):
    return [i for i, s in enumerate(ref) if pred(i, s)]


SYMBOLIC = {
    # "The bore cylinder is a symbolic cut envelope"
    "Axle Bearing": lambda ref: [1],
    # "The valve bodies are symbolic embedded solids"
    "Pipe Manifold": lambda ref: _ids_where(ref, lambda i, s: s["type"] == "cylinder" and s.get("radius") == 10.0),
    # "The full sphere is a symbolic dome carrier ... optical parts may be embedded"
    "Compound Eye": lambda ref: list(range(len(ref))),
    # "Pores and ribs are embedded symbolic geometry"
    "Diatom Frustule": lambda ref: list(range(3, len(ref))),
    # "The frame and links are symbolic reinforcing centerlines"
    "Honeycomb Lattice": lambda ref: _ids_where(ref, lambda i, s: s["type"] in ("torus", "beam")),
    # "Discs and canal are symbolic embedded components"
    "Vertebral Column": lambda ref: _ids_where(ref, lambda i, s: s["type"] in ("cylinder", "pipe")),
    # "Nuts and bolts are symbolic uncut parts"
    "Flanged Pipe Joint": lambda ref: list(range(4, len(ref))),
    # "Dial and markers are symbolic embedded face components"
    "Clock Tower Mechanism": lambda ref: [3] + _ids_where(ref, lambda i, s: s["type"] == "beam" and abs(s["start"][2]) < 1e-9),
    # "Crossed centerlines are symbolic structural joints" (braces and rails/bridge meet at centerlines)
    "Gantry Crane Assembly": lambda ref: _ids_where(ref, lambda i, s: s["type"] == "beam"),
    # "Membranes are symbolic internal components"
    "Cochlear Spiral": lambda ref: _ids_where(ref, lambda i, s: s["type"] == "box"),
    # "This is a symbolic uncaged bearing" (balls touch races)
    "Ball Bearing Assembly": lambda ref: [],
}
CUT = {"Axle Bearing": lambda ref: [1]}


# ---------------------------------------------------------------------------
# Reference patches (each fixes a verified disagreement between prompt and reference)
# ---------------------------------------------------------------------------
def _patch_bridge(ref, scale):
    """Prompt: 'Index deck, negative-X tower, positive-X tower, then cables' and deck has
    'square section 1'. Generator emitted towers before the deck and a 2 x 1 deck section."""
    towers = [s for s in ref if s["type"] == "cylinder"]
    beams = [s for s in ref if s["type"] == "beam"]
    deck = next(b for b in beams if abs(b["start"][2] - 10) < 1e-9 and abs(b["end"][2] - 10) < 1e-9
                and abs(abs(b["start"][0]) - 50) < 1e-9 and abs(abs(b["end"][0]) - 50) < 1e-9)
    deck = dict(deck, width=1, height=1)
    cables = [b for b in beams if b is not deck and not (b.get("width") == 2)]
    towers.sort(key=lambda s: s["center"][0])
    return [deck] + towers + cables, ["bridge_id_order_deck_first", "bridge_deck_square_section_1"]


def _patch_flange(ref, scale):
    """Nut tori had no axis; the prompt says 'its axis is parallel to X'."""
    out = []
    for s in ref:
        if s["type"] == "torus" and "axis" not in s:
            s = dict(s, axis=[1, 0, 0])
        out.append(s)
    return out, ["flange_nut_axis_x"]


def _patch_axle(ref, scale):
    """The generator placed bearings (outer radius 14) inside the block, flush with its end faces,
    while the bore radius is shaft + clearance (8.5-9.5): the bearings overlap block material.
    v2 places each bearing outside the block, abutting one end face, and lets the shaft protrude
    20 so it carries the full bearing width (see PROMPT_PATCHES)."""
    out = []
    for s in ref:
        s = copy.deepcopy(s)
        if s["id"] == 2:                     # shaft: 80 + 2*20
            s["height"] = 120.0
        if s["id"] in (3, 4):                # bearings: centers at +-(40 + 6)
            sign = -1 if s["id"] == 3 else 1
            s["center"] = [sign * 46.0, 0.0, 25.0]
        out.append(s)
    return out, ["axle_bearings_outside_block", "axle_shaft_protrusion_20"]


def _patch_honeycomb(ref, scale):
    """Prompt: 'Link IDs follow increasing endpoint-ID pairs'. The generator emitted links in a
    different order; reorder them by (lower cell id, higher cell id) of their endpoint cells."""
    pipes = [s for s in ref if s["type"] == "pipe"]
    cell_of = lambda p: min(range(len(pipes)), key=lambda i: sum((pipes[i]["center"][k] - p[k]) ** 2 for k in range(3)))
    first_link = next(i for i, s in enumerate(ref) if s["type"] == "beam" and abs(s["start"][2] - s["end"][2]) < 1e-9
                      and abs(s["start"][2] - pipes[0]["center"][2]) < 1e-9)
    links = [s for s in ref[first_link:] if s["type"] == "beam" and s.get("width") == 1.0]
    rest = [s for s in ref[first_link:] if not (s["type"] == "beam" and s.get("width") == 1.0)]
    keyed = []
    for s in links:
        a, b = sorted((cell_of(s["start"]), cell_of(s["end"])))
        keyed.append(((a, b), dict(s, start=pipes[a]["center"], end=pipes[b]["center"])))
    keyed.sort(key=lambda t: t[0])
    return ref[:first_link] + [s for _, s in keyed] + rest, ["honeycomb_link_id_order"]


PATCHES = {
    "Suspension Bridge": _patch_bridge,
    "Flanged Pipe Joint": _patch_flange,
    "Axle Bearing": _patch_axle,
    "Honeycomb Lattice": _patch_honeycomb,
}

PROMPT_PATCHES = {
    "Axle Bearing": [
        ("protrudes 10 beyond each\nblock end face", "protrudes 20 beyond each\nblock end face"),
        ("Bearings lie inside the block,\nwith their outer end faces flush with its opposite X end faces, and are\nconcentric with the shaft.",
         "Bearings sit outside the block,\neach with one end face abutting one of the block's X end faces, and are\nconcentric with the shaft."),
    ],
    # The central cell's elevation was not stated anywhere; the reference stands it on the base plate.
    "Honeycomb Lattice": [
        ("All pipe centroids\nlie in the central vertical cell's mid-height plane.",
         "The central cell stands on the base plate's top face. All pipe centroids\nlie in the central vertical cell's mid-height plane."),
    ],
}


def _apply_prompt_patches(family: str, body: str) -> tuple[str, list[str]]:
    applied = []
    for old, new in PROMPT_PATCHES.get(family, []):
        if old not in body:
            raise ValueError(f"prompt patch for {family} no longer matches generator text: {old[:40]!r}")
        body = body.replace(old, new)
        applied.append("prompt:" + old.split()[0] + "..")
    return body, applied


def _canonical_reference(raw_ref: list) -> list[dict]:
    shapes, rep = normalize(raw_ref)
    if rep.n_kept != len(raw_ref):
        raise ValueError(f"reference contains parts the v2 normalizer drops: {rep}")
    out = []
    for i, s in enumerate(shapes):
        d = s.to_json()
        d["id"] = i
        out.append(_round(d))
    return out


def _round(obj, nd=6):
    if isinstance(obj, float):
        v = round(obj, nd)
        return 0.0 if v == 0 else v
    if isinstance(obj, list):
        return [_round(x, nd) for x in obj]
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    return obj


def build_case(family: str, level: int) -> dict:
    fam, fn, scales, phase = FAMILY_INDEX[family]
    scale = scales[level - 1]
    with contextlib.redirect_stdout(io.StringIO()):
        prompt, second = fn(scale)
    specs = second if isinstance(second, dict) and "reference" in second else None
    raw_ref = copy.deepcopy(specs["reference"] if specs else second)
    body = prompt.split(GENERATOR_SUFFIX_MARKER)[0].rstrip()
    body, ppatch = _apply_prompt_patches(family, body)
    patches = list(ppatch)
    if family in PATCHES:
        raw_ref, names = PATCHES[family](raw_ref, scale)
        patches += names
    ref = _canonical_reference(raw_ref)
    sym = sorted(set(SYMBOLIC.get(family, lambda r: [])(ref)))
    cut = sorted(set(CUT.get(family, lambda r: [])(ref)))
    full_prompt = body + "\n" + SCHEMA_CONTRACT
    return {
        "case_id": case_id(family, level), "family": family, "phase": phase,
        "level": level, "scale": scale, "n_parts": len(ref),
        "prompt_body": body, "prompt": full_prompt,
        "prompt_sha256": hashlib.sha256(full_prompt.encode()).hexdigest()[:16],
        "reference": ref, "symbolic_ids": sym, "cut_ids": cut, "patches": patches,
    }


def build_all() -> list[dict]:
    return [build_case(f[0], lvl) for f in FAMILIES for lvl in (1, 2, 3)]


def export(path: Path | None = None) -> Path:
    path = path or ROOT / "v2" / "data" / "cases_v2.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for c in build_all():
            fh.write(json.dumps(c) + "\n")
    return path


def load(path: Path | None = None) -> list[dict]:
    path = path or ROOT / "v2" / "data" / "cases_v2.jsonl"
    return [json.loads(l) for l in open(path)]


# ---------------------------------------------------------------------------
# Scale sweep: the same prompt template at many counts (for per-model "breaking size" curves).
# Only families whose generator and constraints are valid beyond the three levels (checked by
# v2/tests: every sweep reference satisfies all its constraints).
SWEEP_SCALES = {
    "Spiral Staircase": [5, 10, 16, 24, 36, 50, 72, 100], "Domino Ring": [3, 5, 8, 10, 15, 20, 30, 40],
    "DNA Helix": [3, 5, 10, 15, 20, 30, 40], "Flanged Pipe Joint": [4, 6, 8, 12, 16, 24],
    "Suspension Bridge": [3, 5, 10, 15, 20, 30, 40], "Planetary Array": [3, 6, 9, 12, 18, 24],
    "Cross-Braced Truss": [1, 2, 3, 4, 6, 8, 12], "Fractal Y-Tree": [1, 2, 3, 4, 5, 6],
    "Ball Bearing Assembly": [4, 6, 8, 12, 16, 20, 24], "Pipe Manifold": [1, 2, 3, 4, 5, 6, 8],
    "Clock Tower Mechanism": [4, 6, 8, 12, 24, 36, 48], "Gantry Crane Assembly": [1, 2, 3, 4, 6, 8, 10],
    "Phyllotaxis Disc": [8, 21, 34, 55, 89, 144, 233], "Cannonball Pyramid": [2, 3, 4, 5, 6, 7],
    "Voxel Grid": [2, 3, 4, 5, 6], "Compound Eye": [1, 2, 3, 4, 5], "Diatom Frustule": [2, 4, 7, 10, 12, 16],
    "Honeycomb Lattice": [1, 2, 3, 4, 5], "BCC Lattice": [1, 2, 3, 4],
}


def build_sweep() -> list[dict]:
    out = []
    for fam, scales in SWEEP_SCALES.items():
        f, fn, orig, ph = FAMILY_INDEX[fam]
        for s in scales:
            FAMILY_INDEX[fam] = (f, fn, [s, s, s], ph)
            try:
                c = build_case(fam, 1)
            finally:
                FAMILY_INDEX[fam] = (f, fn, orig, ph)
            c["case_id"] = case_id(fam, 0).replace("_level_0", f"_scale_{s}")
            c["level"] = 0
            c["split"] = "sweep"
            out.append(c)
    return out


def export_sweep(path: Path | None = None) -> Path:
    path = path or ROOT / "v2" / "data" / "sweep_v2.jsonl"
    with open(path, "w") as fh:
        for c in build_sweep():
            fh.write(json.dumps(c) + "\n")
    return path
