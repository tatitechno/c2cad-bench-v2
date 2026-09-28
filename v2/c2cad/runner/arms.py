"""Experimental conditions ("arms"). Every arm asks for the same assembly and is scored by the same
evaluator; each differs from `json` in one factor.

  json       v2 prompt (relations + base values) -> the model writes coordinates directly
  neutral    same, with object/domain nouns replaced by neutral part names               (recall)
  tool       same prompt; the model writes a Python program that prints the JSON         (arithmetic)
  v1         the released v1 prompt (formulas / coordinates leaked) + the same contract   (copying)
  mates      the model declares parts, mates and patterns in CML; a solver places them   (pose/pattern derivation)
  schema     json, with the provider's constrained decoding to the output schema         (format channel closed)
  cadquery   same prompt; the model writes a CadQuery program, run in a CAD kernel        (code vs coordinates)
  probe      same prompt; the model returns only three named parts                        (serialisation load)
  repair_generic / repair_verifier
             multi-turn: the json answer (sample 0) is sent back with a generic "check and fix" request, or with
             the reference-free verifier's report (violated prompt sentences, missing ids, part count); up to
             R rounds. Run after the json arm (they reuse its answer as round 0).
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from ..cases import SCHEMA_CONTRACT
from . import neutral

REPO = Path(__file__).resolve().parents[3]

SYSTEM_JSON = ("You design 3D assemblies from written engineering specifications. "
               "Answer with the JSON the output contract asks for and nothing else.")
SYSTEM_TOOL = ("You design 3D assemblies from written engineering specifications by writing a short Python program "
               "that computes the assembly. Answer with the program only.")
TOOL_INSTRUCTION = """
TOOL MODE
Do not write the JSON yourself. Write one self-contained Python 3 program that computes the parts and prints
the JSON described by the output contract to standard output (for example with print(json.dumps(parts))).
You may import only math, numpy, itertools and json. Return the program in a single ```python block.
"""

SYSTEM_MATES = ("You design 3D assemblies from written engineering specifications the way a CAD user does: you declare parts, "
                "attach them to features of other parts and use patterns. A solver computes the coordinates. "
                "Answer with the program only.")

SYSTEM_CAD = ("You design 3D assemblies from written engineering specifications by writing CadQuery code. "
              "Answer with the program only.")
CAD_INSTRUCTION = """
CADQUERY MODE
Write one self-contained CadQuery 2.x (Python) program that builds the assembly. Units are millimetres; use the
world origin and axes the specification states.
Build every part as its own solid, one solid per part id, and store them in a dict named `parts` that maps each
integer id to its solid (a cadquery Solid, or a Workplane holding exactly one solid), for example
parts[0] = cq.Workplane("XY").box(10, 10, 2). Use the ids the specification prescribes.
Keep parts separate: do not union parts together and do not cut one part with another. A single part may use a
cut (a pipe is a cylinder with a coaxial bore).
Part shapes: box (rectangular block with faces parallel to the world axes), cylinder, pipe (hollow cylinder),
cone or truncated cone, sphere, torus, and beam (straight bar of rectangular section, in any orientation).
You may import only cadquery, math, numpy, itertools and json. Do not read or write files.
Return the program in a single ```python block.
"""

PROBE_INSTRUCTION = """
PROBE MODE
Do not write the whole assembly. Return only the parts with ids {ids}, each exactly as it would appear in the
complete answer (same id, type, position, orientation and dimensions). Return only JSON: an array of those
{n} part objects, as described by the output contract.
"""

REPAIR_GENERIC = ("Check your answer against the specification above. If any part is missing, misplaced, mis-sized, "
                  "wrongly oriented or of the wrong type, return the corrected complete assembly; if it is correct, "
                  "return it unchanged. Follow the same output contract: JSON only.")
REPAIR_VERIFIER_HEAD = ("An automatic checker read your answer against the specification. It uses only the "
                        "specification's own statements and the part ids you gave; it has not seen any reference "
                        "answer. It found:")
REPAIR_VERIFIER_TAIL = "Return the corrected complete assembly. Follow the same output contract: JSON only."

MATES_INSTRUCTION = """
MATES MODE
Do not write coordinates part by part. Write a program in the assembly language below; a deterministic solver
computes every part from it. Use the part ids the task prescribes. Return only the JSON program {"steps": [...]}.

"""


def _cml_reference():
    from .. import dsl
    doc = dsl.__doc__
    return doc[doc.index("Program   "):].rstrip()


# Two worked examples on assemblies that are not in the benchmark (validated by tests).
MATES_EXAMPLES = [
    ("A wheel stands on the ground: a hub cylinder of radius 10 and height 8 on the Z axis, a torus rim of ring radius 40 and "
     "tube radius 3 in the hub's mid-plane, and 8 spoke beams of square section 2 running radially from the hub surface to the "
     "rim circle in that plane, equally spaced, the first along +Y. IDs: hub 0, rim 1, spokes 2-9 counterclockwise.",
     {"steps": [
         {"id": 0, "type": "cylinder", "radius": 10, "height": 8, "axis": "+z", "bottom_at": "origin"},
         {"id": 1, "type": "torus", "ring_radius": 40, "tube_radius": 3, "axis": "+z", "at": {"of": 0, "feature": "center"}},
         {"id": 2, "type": "beam", "width": 2, "height": 2, "start": {"of": 0, "feature": "surface", "deg": 90},
          "end": {"of": 1, "feature": "center", "then": [{"along": {"radial": {"about": {"axis_of": 0}, "deg": 90}}, "by": {"dim": [1, "ring_radius"]}}]}},
         {"pattern": "circular", "seed": [2], "count": 8, "about": {"axis_of": 0}, "full_circle": True, "first_id": 3}]}),
    ("A shelf: two vertical cylinder posts of radius 2 and height 30 stand on the ground 40 apart along X, centered on the origin; "
     "a box board of size 60 x 20 x 2 rests on the post tops, centered over them; a ball of radius 3 sits on the board directly "
     "above the origin, touching it. IDs: posts 0 and 1 (negative X first), board 2, ball 3.",
     {"steps": [
         {"id": 0, "type": "cylinder", "radius": 2, "height": 30, "axis": "+z", "bottom_at": "origin"},
         {"pattern": "linear", "seed": [0], "count": 2, "direction": "+x", "step": 40, "centered": True, "first_id": 1},
         {"id": 2, "type": "box", "size": [60, 20, 2], "align": {"x": ["center", "origin"], "y": ["center", "origin"],
                                                                 "z": ["-", {"of": 0, "feature": "top"}]}},
         {"id": 3, "type": "sphere", "radius": 3, "at": {"of": 2, "feature": "top"}, "rest_on": {"face": [2, "top"]}}]}),
]


def _examples_text():
    out = ["EXAMPLES (unrelated to the task)"]
    for i, (task, prog) in enumerate(MATES_EXAMPLES, 1):
        out.append(f"Example {i} task: {task}")
        out.append(f"Example {i} program: {json.dumps(prog, separators=(',', ':'))}")
    return "\n".join(out)


SINGLE_TURN_ARMS = ("json", "neutral", "tool", "v1", "mates", "schema", "cadquery", "probe")
REPAIR_ARMS = ("repair_generic", "repair_verifier")
ARMS = SINGLE_TURN_ARMS + REPAIR_ARMS
COORDINATE_ARMS = ("json", "neutral", "v1", "schema")     # the model writes every coordinate itself
PROGRAM_ARMS = ("tool", "mates", "cadquery")
V1_EXCLUDED = {"Furniture Assembly", "Axle Bearing"}   # v1 prompts left dimensions free


@lru_cache(maxsize=1)
def _v1_prompts():
    d = {}
    for line in open(REPO / "data" / "cases.jsonl"):
        c = json.loads(line)
        d[(c["family"], c["difficulty_id"])] = c["prompt"]
    return d


def probe_ids(case: dict) -> list[int]:
    """Three part ids per case, fixed in advance: the middle id, the last id (the end of every pattern), and one
    id drawn with a seed derived from the case id. All ids when the case has three parts or fewer."""
    n = case["n_parts"]
    if n <= 3:
        return list(range(n))
    ids = {n // 2, n - 1}
    seed = int(hashlib.sha256(case["case_id"].encode()).hexdigest()[:8], 16)
    rest = [i for i in range(n) if i not in ids]
    ids.add(rest[seed % len(rest)])
    return sorted(ids)


def build(arm: str, case: dict) -> tuple[str, str]:
    """(system, user) for this single-turn arm and case. Repair arms start from the json prompt."""
    if arm in ("json", "schema") or arm in REPAIR_ARMS:
        return SYSTEM_JSON, case["prompt_body"] + "\n" + SCHEMA_CONTRACT
    if arm == "neutral":
        return SYSTEM_JSON, neutral.twin(case["family"], case["prompt_body"]) + "\n" + SCHEMA_CONTRACT
    if arm == "tool":
        return SYSTEM_TOOL, case["prompt_body"] + "\n" + SCHEMA_CONTRACT + TOOL_INSTRUCTION
    if arm == "v1":
        return SYSTEM_JSON, _v1_prompts()[(case["family"], case["level"])].rstrip() + "\n" + SCHEMA_CONTRACT
    if arm == "mates":
        return SYSTEM_MATES, (case["prompt_body"] + "\n" + MATES_INSTRUCTION + _cml_reference() + "\n\n"
                              + _examples_text() + "\n\nNow write the program for the task above.\n")
    if arm == "cadquery":
        return SYSTEM_CAD, case["prompt_body"] + "\n" + CAD_INSTRUCTION
    if arm == "probe":
        ids = probe_ids(case)
        names = ", ".join(str(i) for i in ids[:-1]) + (" and " if len(ids) > 1 else "") + str(ids[-1])
        return SYSTEM_JSON, (case["prompt_body"] + "\n" + SCHEMA_CONTRACT
                             + PROBE_INSTRUCTION.format(ids=names, n=len(ids)))
    raise ValueError(arm)


def repair_feedback(arm: str, report: dict | None, parsed_ok: bool) -> str | None:
    """The user turn that follows a model answer in a repair arm. None = nothing to report (verifier is clean)."""
    if arm == "repair_generic":
        return REPAIR_GENERIC
    items = []
    if not parsed_ok or report is None:
        items.append("- Your answer could not be read as JSON.")
    else:
        n, N = report["n_parts"], report["n_required"]
        if n != N:
            items.append(f"- The specification defines {N} parts (ids 0 to {N - 1}); your answer has {n}.")
        if not report["ids_usable"]:
            items.append("- Part ids are missing, repeated or not integers, so the parts could not be matched to the "
                         "specification's numbering.")
        else:
            miss = report["missing_ids"]
            if miss:
                shown = ", ".join(str(i) for i in miss[:20]) + (f" and {len(miss) - 20} more" if len(miss) > 20 else "")
                items.append(f"- These ids are missing: {shown}.")
            if report["extra_ids"]:
                items.append(f"- These ids are outside 0 to {N - 1}: "
                             + ", ".join(str(i) for i in report["extra_ids"][:20]) + ".")
            for v in report["violations"][:12]:
                ids = v["ids"]
                where = ((" (part " if len(ids) == 1 else " (parts ") + ", ".join(str(i) for i in ids[:10])
                         + (" ..." if len(ids) > 10 else "") + ")" if ids else "")
                items.append(f'- Not satisfied: "{v["sentence"]}"{where}.')
            if len(report["violations"]) > 12:
                items.append(f"- ... and {len(report['violations']) - 12} more statements of the specification.")
    if not items:
        return None
    return REPAIR_VERIFIER_HEAD + "\n" + "\n".join(items) + "\n" + REPAIR_VERIFIER_TAIL


def applicable(arm: str, case: dict) -> bool:
    if arm == "v1":      # released v1 prompts exist only for the main split, and two of them left dimensions free
        return case.get("split", "main") == "main" and case["family"] not in V1_EXCLUDED
    return True


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Constrained decoding: the output contract as a JSON schema (strict-mode compatible: every object closed,
# every property required, no numeric or array-length keywords, which some providers reject).
_NUM = {"type": "number"}
_VEC = {"type": "array", "items": _NUM}


def _part(t: str, fields: dict) -> dict:
    props = {"id": {"type": "integer"}, "type": {"type": "string", "enum": [t]}, **fields}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


ASSEMBLY_SCHEMA = {
    "type": "object",
    "properties": {"shapes": {"type": "array", "items": {"anyOf": [
        _part("box", {"center": _VEC, "size": _VEC}),
        _part("cylinder", {"center": _VEC, "axis": _VEC, "radius": _NUM, "height": _NUM}),
        _part("pipe", {"center": _VEC, "axis": _VEC, "inner_radius": _NUM, "outer_radius": _NUM, "height": _NUM}),
        _part("cone", {"center": _VEC, "axis": _VEC, "base_radius": _NUM, "top_radius": _NUM, "height": _NUM}),
        _part("sphere", {"center": _VEC, "radius": _NUM}),
        _part("torus", {"center": _VEC, "axis": _VEC, "ring_radius": _NUM, "tube_radius": _NUM}),
        _part("beam", {"start": _VEC, "end": _VEC, "width": _NUM, "height": _NUM}),
    ]}}},
    "required": ["shapes"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
def parse_json(text: str):
    """Return (value or None, status in {"ok","extracted","repaired","fail"})."""
    if not text:
        return None, "fail"
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(t), "ok"
    except json.JSONDecodeError:
        pass
    for block in re.findall(r"```(?:json)?\s*\n(.*?)```", t, flags=re.S):
        try:
            return json.loads(block), "extracted"
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\[{]", t):
        try:
            v, _ = dec.raw_decode(t[m.start():])
            if isinstance(v, list) or (isinstance(v, dict) and any(isinstance(x, list) for x in v.values())):
                return v, "extracted"
        except json.JSONDecodeError:
            continue
    # truncated output: keep complete objects of the first array
    i = t.find("[")
    if i >= 0:
        j = t.rfind("}")
        while j > i:
            try:
                v = json.loads(t[i:j + 1] + "]")
                if isinstance(v, list):
                    return v, "repaired"
            except json.JSONDecodeError:
                pass
            j = t.rfind("}", i, j)
    return None, "fail"
