"""Offline mock providers: exercise every arm, the resume logic and the analysis without API calls.

  mock:reference      the reference answer in the arm's format (a correct model)
  mock:jitter-<s>     reference with Gaussian position noise sigma = s mm
  mock:beam-box       every beam replaced by an axis-aligned box at its midpoint
  mock:empty          an empty answer
  mock:noisy          a deterministic, family- and arm-dependent mix of failures (drops, jitter, beam->box,
                      misnumbering), weaker on later repair rounds, so the analysis has variance to report
  mock:noisy-<tag>    as noisy with a different skill offset (a second mock model)
Usage fields imitate a provider's (characters / 4), so the budget guard can be tested.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from .. import cadcode, cml_programs
from . import arms as A
from .providers import Response


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _jitter(ref, s, rng):
    for p in ref:
        for k in ("center", "start", "end"):
            if k in p:
                p[k] = (np.array(p[k]) + rng.normal(0, s, 3)).tolist()


def _beam_box(ref):
    for i, p in enumerate(ref):
        if p["type"] == "beam":
            a, b = np.array(p["start"]), np.array(p["end"])
            ref[i] = {"id": p["id"], "type": "box", "center": ((a + b) / 2).tolist(),
                      "size": [float(np.linalg.norm(b - a)), p["width"], p["height"]]}


ARM_SKILL = {"json": 0.0, "neutral": 0.05, "v1": -0.15, "schema": 0.0, "tool": -0.2, "mates": -0.1,
             "cadquery": -0.1, "probe": -0.3, "repair_generic": 0.0, "repair_verifier": 0.0}


def answer(kind: str, arm: str, case: dict, sample: int, round_: int = 0) -> list[dict]:
    ref = copy.deepcopy(case["reference"])
    rng = np.random.default_rng(_h(kind, arm, case["case_id"], sample, round_))
    if kind.startswith("jitter-"):
        _jitter(ref, float(kind.split("-")[1]), rng)
    elif kind == "beam-box":
        _beam_box(ref)
    elif kind == "empty":
        ref = []
    elif kind.startswith("noisy"):
        fam = _h(case["family"]) % 100 / 100.0             # family difficulty in [0, 1)
        size = min(1.0, case["n_parts"] / 200.0)
        model = (_h(kind) % 21 - 10) / 100.0                # a per-mock-model skill offset in [-0.1, 0.1]
        p_fail = min(0.95, max(0.02, 0.15 + 0.5 * fam + 0.3 * size + ARM_SKILL.get(arm, 0.0) + model
                                - (0.25 * round_ if arm == "repair_verifier" else 0.05 * round_)))
        r = rng.random()
        if r < p_fail * 0.35:
            keep = int(len(ref) * rng.uniform(0.5, 0.95))
            ref = ref[:keep]
        elif r < p_fail * 0.6:
            _jitter(ref, rng.uniform(0.5, 4.0), rng)
        elif r < p_fail * 0.8:
            _beam_box(ref)
        elif r < p_fail:
            for p in ref:
                p["id"] = int(p["id"]) + 1
            _jitter(ref, 0.2, rng)
    return ref


def response(spec: str, arm: str, case: dict, sample: int, history: list | None = None) -> Response:
    kind = spec.split(":", 1)[1]
    round_ = 0 if not history else sum(1 for m in history if m["role"] == "assistant")
    parts = answer(kind, arm, case, sample, round_)
    if arm == "probe":
        want = set(A.probe_ids(case))
        text = json.dumps([p for p in parts if int(p.get("id", -1)) in want])
    elif arm == "mates":
        fn = cml_programs.PROGRAMS.get(case["family"])
        text = json.dumps(fn(case)) if (fn and kind == "reference") else json.dumps({"steps": []})
    elif arm == "tool":
        text = "```python\nimport json\nparts = " + json.dumps(parts) + "\nprint(json.dumps(parts))\n```"
    elif arm == "cadquery":
        text = "```python\n" + (cadcode.to_cadquery_code(parts) if parts else "parts = {}\n") + "```"
    elif arm == "schema":
        text = json.dumps({"shapes": parts})
    else:
        text = json.dumps(parts)
    usage = {"prompt_tokens": 1000, "completion_tokens": len(text) // 4}
    enforcement = "strict_schema" if arm == "schema" else "none"
    return Response(text, True, returned_model=spec, finish_reason="stop", usage=usage, enforcement=enforcement,
                    request_meta={"mock": True})
