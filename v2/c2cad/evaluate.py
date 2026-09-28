"""One entry point: evaluate a model output (raw JSON value) against a v2 case.

Global_v2 = mean(Coverage, Geometry, Semantic); no gates, no calibration exponent.
Every component is also reported on its own; the paper leads with the components.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from . import constraints as K
from . import prism as PR
from .constraints import core as KC
from .constraints import trace as T
from .geom import normalize, unwrap
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


def _resolve_prisms(ref, raw_output):
    items = unwrap(raw_output) if raw_output is not None else []
    if not any(isinstance(p, dict) and p.get("type") == "prism" for p in items):
        return raw_output, 0
    return PR.resolve(ref, items)


def shapes_for(case: dict, raw_output: Any):
    """(normalized reference, normalized output after prism resolution) for metrics outside evaluate()."""
    if case["case_id"] not in _CASES:
        register(case)
    ref, _, _ = _ref_and_constraints(case["case_id"])
    raw_output, _ = _resolve_prisms(ref, raw_output)
    out, _ = normalize(raw_output)
    return ref, out


def verify(case: dict, raw_output: Any) -> dict:
    """Reference-free check used as feedback by the repair_verifier arm.

    Parts are bound to roles by the output's own ids (the numbering the prompt prescribes), so no reference
    coordinates enter: the report lists the prompt sentences whose constraints fail, with the offending ids,
    the ids the specification defines but the output lacks, and the part count the specification implies."""
    if case["case_id"] not in _CASES:
        register(case)
    ref, cons, c2s = _ref_and_constraints(case["case_id"])
    out, _ = normalize(raw_output)
    n_req = len(ref)
    rep = {"n_parts": len(out), "n_required": n_req, "ids_usable": False, "missing_ids": [], "extra_ids": [],
           "violations": [], "n_failed": 0, "n_constraints": len(cons)}
    idb = id_binding(out, n_req)
    if idb is None:
        return rep
    rep["ids_usable"] = True
    rep["missing_ids"] = [i for i in range(n_req) if i not in idb]
    shift = 1 if out and all(isinstance(s.id, (int, float)) for s in out) and min(int(s.id) for s in out) == 1 \
        and 0 not in [int(s.id) for s in out] else 0
    rep["extra_ids"] = sorted(int(s.id) - shift for s in out if not 0 <= int(s.id) - shift < n_req)
    by_sentence: dict[str, dict] = {}
    for c in cons:
        if c.residual(idb) <= c.tol * KC.TOL_SCALE:
            continue
        rep["n_failed"] += 1
        for s in c2s.get(c.clause.split(":", 1)[-1], []):
            d = by_sentence.setdefault(s, {"n": 0, "ids": set()})
            d["n"] += 1
            d["ids"].update(r for r in c.roles if r in idb)
    order = {s: i for i, s in enumerate(T.sentences(case["prompt_body"]))}
    rep["violations"] = [{"sentence": s, "n_failed": d["n"], "ids": sorted(d["ids"])}
                         for s, d in sorted(by_sentence.items(), key=lambda kv: order.get(kv[0], 1e9))]
    return rep


def evaluate(case: dict, raw_output: Any, weights=WEIGHTS) -> Result:
    if case["case_id"] not in _CASES:
        register(case)
    ref, cons, c2s = _ref_and_constraints(case["case_id"])
    raw_output, n_prisms = _resolve_prisms(ref, raw_output)
    out, rep = normalize(raw_output)
    if n_prisms:
        rep.prisms_resolved = n_prisms
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
