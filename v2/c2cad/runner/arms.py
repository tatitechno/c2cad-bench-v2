"""Experimental conditions ("arms"). Every arm asks for the same assembly and is scored by the same
evaluator; they differ in one factor each.

  json      v2 prompt (relations + base values) -> the model writes coordinates directly
  neutral   same, with object/domain nouns replaced by neutral part names  (tests recall)
  tool      same prompt, but the model writes a Python program that prints the JSON (tests arithmetic)
  v1        the released v1 prompt (formulas / coordinates leaked) + the same output contract (tests copying)
  mates     (see dsl.py) the model declares parts, mates and patterns; a solver places them
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


def _cml_reference():
    from .. import dsl
    doc = dsl.__doc__
    return doc[doc.index("Program   "):].rstrip()


MATES_INSTRUCTION = """
MATES MODE
Do not write coordinates part by part. Write a program in the assembly language below; a deterministic solver
computes every part from it. Use the part ids the task prescribes. Return only the JSON program {"steps": [...]}.

"""

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

ARMS = ("json", "neutral", "tool", "v1", "mates")
V1_EXCLUDED = {"Furniture Assembly", "Axle Bearing"}   # v1 prompts left dimensions free


@lru_cache(maxsize=1)
def _v1_prompts():
    d = {}
    for line in open(REPO / "data" / "cases.jsonl"):
        c = json.loads(line)
        d[(c["family"], c["difficulty_id"])] = c["prompt"]
    return d


def build(arm: str, case: dict) -> tuple[str, str]:
    """(system, user) for this arm and case."""
    if arm == "json":
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
    raise ValueError(arm)


def applicable(arm: str, case: dict) -> bool:
    if arm == "v1":      # released v1 prompts exist only for the main split, and two of them left dimensions free
        return case.get("split", "main") == "main" and case["family"] not in V1_EXCLUDED
    return True


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


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
