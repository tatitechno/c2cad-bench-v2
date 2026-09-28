"""Constraint engine for reference-free semantic validity (Sem v2).

A family builder turns a case into a list of Constraint objects. Each constraint
  * names the reference roles (ids 0..n-1, in the order the prompt prescribes) it involves,
  * computes a non-negative residual from the bound output parts (mm, degrees or relative),
  * quotes the prompt clause it enforces (`clause`), and
  * belongs to a kind: dimension | anchor | mate | pattern | orientation | topology.

Sem = 100 * mean over clauses of (fraction of that clause's constraints satisfied).
A constraint whose roles are not all bound is unsatisfied. No gates, no reference
normalization: the reference must satisfy every constraint (tested), and a model output is
judged only on the relations the prompt states.

Binding (which output part plays which role):
  primary   optimal assignment against the reference (robust to id bookkeeping errors);
  secondary the output's own ids (reported as instruction following).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from ..geom import Shape

INF = float("inf")
ANG_TOL_DEG = 0.5
REL_TOL = 0.01
LEN_TOL_MIN = 0.25
LEN_TOL_FRAC = 0.002
TOL_SCALE = 1.0          # sensitivity analysis only: multiplies every constraint tolerance at evaluation time


class Missing(Exception):
    """Raised by helpers when a bound part lacks the geometry a constraint needs."""


@dataclass
class Constraint:
    name: str
    kind: str
    roles: tuple
    fn: Callable[[dict], float]
    tol: float
    unit: str
    clause: str

    def residual(self, parts: dict) -> float:
        if any(r not in parts for r in self.roles):
            return INF
        try:
            v = float(self.fn(parts))
        except (Missing, ValueError, ZeroDivisionError, FloatingPointError, IndexError, KeyError, TypeError):
            return INF
        return v if math.isfinite(v) else INF


@dataclass
class SemResult:
    sem: float                      # 0-100, clause-averaged
    n_constraints: int
    n_satisfied: int
    by_kind: dict = field(default_factory=dict)       # kind -> pass fraction (0-1)
    by_clause: dict = field(default_factory=dict)     # clause -> pass fraction (0-1)
    failures: list = field(default_factory=list)      # (name, residual, tol, unit)
    sem_flat: float = 0.0           # mean over individual constraints
    sem_kind: float = 0.0           # mean over constraint kinds
    sem_sentence: float | None = None   # mean over prompt sentences (needs a clause->sentences map)


def evaluate(constraints: list[Constraint], parts: dict[int, Shape], keep_failures: int = 50,
             clause_sentences: dict | None = None) -> SemResult:
    clause_hits, kind_hits = defaultdict(list), defaultdict(list)
    fails, n_ok = [], 0
    for c in constraints:
        r = c.residual(parts)
        ok = r <= c.tol * TOL_SCALE
        n_ok += ok
        clause_hits[c.clause].append(ok)
        kind_hits[c.kind].append(ok)
        if not ok and len(fails) < keep_failures:
            fails.append((c.name, r, c.tol, c.unit))
    by_clause = {k: sum(v) / len(v) for k, v in clause_hits.items()}
    by_kind = {k: sum(v) / len(v) for k, v in kind_hits.items()}
    sem = 100.0 * (sum(by_clause.values()) / len(by_clause)) if by_clause else 0.0
    flat = 100.0 * n_ok / len(constraints) if constraints else 0.0
    kind = 100.0 * sum(by_kind.values()) / len(by_kind) if by_kind else 0.0
    sent = None
    if clause_sentences is not None:
        per_sent = defaultdict(list)
        for key, hits in clause_hits.items():
            for sid in clause_sentences.get(key.split(":", 1)[-1], []):
                per_sent[sid].extend(hits)
        sent = 100.0 * sum(sum(v) / len(v) for v in per_sent.values()) / len(per_sent) if per_sent else 0.0
    return SemResult(sem, len(constraints), n_ok, by_kind, by_clause, fails, flat, kind, sent)


# ---------------------------------------------------------------------------
# Builder context: tolerances and a small vocabulary of constraint constructors
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, case: dict, ref: list[Shape]):
        from ..geom import assembly_diagonal
        self.case, self.ref = case, ref
        self.D = assembly_diagonal(ref)
        self.tl = max(LEN_TOL_MIN, LEN_TOL_FRAC * self.D)     # length tolerance (mm)
        self.ta = ANG_TOL_DEG
        self.out: list[Constraint] = []

    # generic adders --------------------------------------------------------
    def add(self, name, kind, roles, fn, clause, tol=None, unit="mm"):
        tol = self.tl if tol is None and unit == "mm" else (self.ta if tol is None and unit == "deg" else
                                                           (REL_TOL if tol is None else tol))
        self.out.append(Constraint(name, kind, tuple(roles), fn, tol, unit, clause))

    def eq_len(self, name, kind, roles, f, value, clause):
        """|f(parts) - value| <= length tolerance."""
        self.add(name, kind, roles, lambda P: abs(f(P) - value), clause, unit="mm")

    def eq_rel(self, name, kind, roles, f, value, clause):
        """relative error of a dimension <= 1%."""
        self.add(name, kind, roles, lambda P: abs(f(P) - value) / max(abs(value), 1e-9), clause, unit="rel")

    def zero_len(self, name, kind, roles, f, clause):
        self.add(name, kind, roles, lambda P: abs(f(P)), clause, unit="mm")

    def zero_deg(self, name, kind, roles, f, clause):
        self.add(name, kind, roles, lambda P: abs(f(P)), clause, unit="deg")

    def type_is(self, rid, t, clause):
        self.add(f"type[{rid}]={t}", "dimension", (rid,), lambda P: 0.0 if P[rid].type == t else INF,
                 clause, tol=0.0, unit="bool")


# ---------------------------------------------------------------------------
# Geometry helpers (all raise Missing when the part cannot supply the quantity)
# ---------------------------------------------------------------------------
AXIAL = ("cylinder", "pipe", "cone")


def V(*x) -> np.ndarray:
    return np.array(x, float)


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise Missing("zero vector")
    return v / n


def ang(u, v, signed=True) -> float:
    """Angle in degrees between directions (unsigned: line angle in [0,90])."""
    d = float(np.dot(unit(u), unit(v)))
    if not signed:
        d = abs(d)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def need(s: Shape, *types) -> Shape:
    if s.type not in types:
        raise Missing(f"{s.type} not in {types}")
    return s


def radius(s: Shape) -> float:
    if s.type in ("cylinder", "sphere"):
        return s.radius
    if s.type == "pipe":
        return s.outer_radius
    if s.type == "cone":
        return s.base_radius
    raise Missing("radius")


def axis(s: Shape) -> np.ndarray:
    if s.type in AXIAL + ("torus",):
        return s.axis
    if s.type == "beam":
        return s.direction
    raise Missing("axis")


def ends(s: Shape):
    """(p0, p1): beam start/end, or axial end-face centers (cone: base, tip)."""
    if s.type == "beam":
        return s.start, s.end
    if s.type in AXIAL:
        return s.end_centers()
    raise Missing("ends")


def beam_ends_unordered(s: Shape):
    need(s, "beam")
    return s.start, s.end


def dist_point_line(p, a, d) -> float:
    d = unit(d)
    w = np.asarray(p, float) - np.asarray(a, float)
    return float(np.linalg.norm(w - np.dot(w, d) * d))


def endpoints_match(s: Shape, p, q) -> float:
    """Residual for a beam whose (unordered) endpoints must be p and q."""
    a, b = beam_ends_unordered(s)
    return min(max(np.linalg.norm(a - p), np.linalg.norm(b - q)),
               max(np.linalg.norm(a - q), np.linalg.norm(b - p)))


def zmin(s: Shape) -> float:
    return float(s.bbox()[0][2])


def zmax(s: Shape) -> float:
    return float(s.bbox()[1][2])


def polar_deg(p) -> float:
    return math.degrees(math.atan2(p[1], p[0]))


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def coaxial_residual(s: Shape, t: Shape) -> tuple[float, float]:
    """(angle between axes in deg (line), distance of t's center from s's axis line)."""
    a1, a2 = axis(s), axis(t)
    return ang(a1, a2, signed=False), dist_point_line(t.center, s.center, a1)


def all_pairs(ids: Iterable[int]):
    ids = list(ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            yield ids[i], ids[j]
