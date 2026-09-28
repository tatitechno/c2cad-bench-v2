"""One entry point: evaluate a model output (raw JSON value) against a v2 case.

Global_v2 = mean(Coverage, Geometry, Semantic); no gates, no calibration exponent.
Every component is also reported on its own; the paper leads with the components.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from . import constraints as K
from .constraints import trace as T
from .geom import normalize
from .score import agreement, assignment_map

WEIGHTS = (1 / 3, 1 / 3, 1 / 3)


@dataclass
class Result:
    case_id: str
    n_ref: int
    n_out: int
    coverage: float
    geometry: float
    geometry_equiv: float
    type_fidelity: float
    position: float
    semantic: float
    semantic_flat: float
    semantic_kind: float
    semantic_sentence: float
    semantic_id_binding: float | None
    id_binding_valid: bool
    constraint_pass_rate: float
    by_kind: dict
    global_v2: float
    exact: bool
    normalize_report: dict = field(default_factory=dict)
    top_failures: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@lru_cache(maxsize=None)
def _ref_and_constraints(case_key: str):
    case = _CASES[case_key]
    ref, _ = normalize(case["reference"])
    c2s = {k: [s for s in v] for k, v in T.clause_to_sentences(case).items()}
    return ref, K.build(case, ref), c2s


_CASES: dict[str, dict] = {}


def register(case: dict):
    _CASES[case["case_id"]] = case


def id_binding(out, n_ref: int):
    """Bind by the output's own integer ids (0- or 1-based). None if ids are unusable."""
    ids = []
    for s in out:
        try:
            v = s.id
            if isinstance(v, bool):
                return None
            iv = int(v)
            if float(v) != iv:
                return None
            ids.append(iv)
        except (TypeError, ValueError):
            return None
    if len(set(ids)) != len(ids) or not ids:
        return None
    shift = 1 if (min(ids) == 1 and 0 not in ids) else 0
    return {i - shift: s for i, s in zip(ids, out) if 0 <= i - shift < n_ref}


def evaluate(case: dict, raw_output: Any, weights=WEIGHTS) -> Result:
    if case["case_id"] not in _CASES:
        register(case)
    ref, cons, c2s = _ref_and_constraints(case["case_id"])
    out, rep = normalize(raw_output)
    ag = agreement(ref, out)
    amap = assignment_map(ref, out)
    parts = {r: out[j] for r, j in amap.items()}
    sem = K.evaluate(cons, parts, clause_sentences=c2s)
    idb = id_binding(out, len(ref))
    sem_id = K.evaluate(cons, idb).sem if idb is not None else None
    g = weights[0] * ag.coverage + weights[1] * ag.geometry + weights[2] * sem.sem
    exact = sem.n_satisfied == sem.n_constraints and ag.coverage >= 99.999 and len(out) == len(ref)
    return Result(case["case_id"], len(ref), len(out), ag.coverage, ag.geometry, ag.geometry_equiv, ag.type_fidelity,
                  ag.position, sem.sem, sem.sem_flat, sem.sem_kind, sem.sem_sentence, sem_id, idb is not None,
                  100.0 * sem.n_satisfied / max(1, sem.n_constraints), sem.by_kind, g, exact,
                  asdict(rep), [(n, r if r != float("inf") else "missing/inapplicable", t, u) for n, r, t, u in sem.failures[:10]])
