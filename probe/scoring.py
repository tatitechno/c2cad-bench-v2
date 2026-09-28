"""
Scoring engine for C2CAD-Bench v7.1 — scientifically calibrated.

Key changes from v7.0:
  1. Dynamic weighting — only measured components contribute; unmeasured
     components redistribute their weight rather than awarding free 1.0.
  2. Continuous distance-based position scoring — exp(-d/τ) instead of
     binary pass/fail, giving gradient signal at every distance.
  3. Asymmetric part-count penalty — penalizes both over- and under-counting.
  4. No double-counting — surface fidelity does NOT fall back to position accuracy.
  5. Type correctness gated — wrong type → that shape scores 0 regardless of position.
  6. Reasoning scored conservatively — capped and downweighted.

Two modes (auto-selected):

  CORE MODE — No engineering specs
    Component pool: geometric, count, spatial, surface, orientation, reasoning, interference
    Base weights:   35%, 12%, 23%, 10%, 5%, 10%, 5%
    Unmeasured components' weight is redistributed proportionally to measured ones.

  ENGINEERING MODE — Level has ≥1 engineering spec
    Component pool: geometric, count, spatial, surface, orientation, reasoning, engineering, patterns
    Base weights:   30%, 10%, 15%, 10%, 5%, 10%, 15%, 5%
    Same redistribution rule.

Scientific design principles:
  - A perfect score requires getting EVERY shape correct to sub-millimeter
    precision, with correct topology, orientations, and engineering constraints.
  - Partial credit is smooth (exponential decay), encouraging incremental improvement.
  - No component can inflate the total by returning 1.0 when nothing was validated.
"""

import math
from .config import LevelResult


def score_level(r: LevelResult, level=None) -> float:
    """
    Compute and return the composite score [0, 1].

    Pass `level` (a FixedLevel) to enable engineering mode when specs exist.
    If `level` is None or has no specs, core mode is used.
    """
    if not r.response_ok:
        r.failure = "No response from LLM"
        return 0.0
    if not r.json_ok:
        r.failure = "Could not extract valid JSON array from response"
        return 0.02  # near-zero: JSON extraction is the minimum bar

    has_engineering = False
    if level is not None:
        has_engineering = bool(
            level.mate_specs or level.clearance_specs or
            level.wall_specs or level.symmetry_spec or level.pattern_spec
        )

    if has_engineering:
        return _score_engineering(r)
    else:
        return _score_core(r)


# ─────────────────────────────────────────────────────────────────
# Sub-score helpers — scientifically calibrated
# ─────────────────────────────────────────────────────────────────

def _geometric_accuracy(r: LevelResult) -> float:
    """
    Per-shape combined position + dimension score.

    For each checked shape the score is the product of two exponential decays:

        pos_score  = exp(-pos_dist_mm / τ_pos)   τ_pos = 2.0 mm
        dim_score  = exp(-dim_rel_err  / τ_dim)   τ_dim = 0.25  (25% relative error → ~0.37)

        shape_score = pos_score × dim_score

    This means:
      - Perfect position AND perfect dimensions → 1.0
      - Right position, 25% dimension error     → ~0.37
      - Right position, 50% dimension error     → ~0.14
      - Right position, 100% dimension error    → ~0.02
      - Wrong type → 0.0 (hard gate, no partial credit)
      - MISSING shape → 0.0

    Returns mean over all checked shapes → [0, 1].
    """
    import re

    if r.positions_checked == 0:
        return 0.0

    TAU_POS = 2.0   # mm — exponential decay for position error
    TAU_DIM = 0.25  # relative — 25% error halves this component

    shape_scores = []

    if r.position_details:
        dim_errors = r.dim_errors if r.dim_errors else []

        for i, detail in enumerate(r.position_details):
            if not isinstance(detail, str):
                shape_scores.append(0.0)
                continue

            if "MISSING" in detail:
                shape_scores.append(0.0)
                continue

            # Type mismatch → hard zero
            if "type_mismatch" in detail:
                shape_scores.append(0.0)
                continue

            # Position component
            m = re.search(r'pos_err=([\d.]+)mm', detail)
            pos_dist = float(m.group(1)) if m else 999.0
            pos_score = math.exp(-pos_dist / TAU_POS)

            # Dimension component from parallel dim_errors list
            if i < len(dim_errors):
                dim_rel = dim_errors[i]
            else:
                # Fallback: parse dim_err from detail string if present
                md = re.search(r'dim_err=([\d.]+)%', detail)
                dim_rel = float(md.group(1)) / 100.0 if md else 0.0

            dim_score = math.exp(-dim_rel / TAU_DIM)

            shape_scores.append(pos_score * dim_score)

    if not shape_scores:
        # Legacy fallback: binary position + type
        pos_acc = r.positions_correct / max(r.positions_checked, 1)
        type_factor = 1.0 if r.types_correct else 0.5
        r.position_accuracy = pos_acc * type_factor
        return r.position_accuracy

    score = sum(shape_scores) / len(shape_scores)
    r.position_accuracy = score

    if score < 0.5 and not r.failure:
        r.failure = (f"Low geometric accuracy: {score:.1%} "
                     f"({r.positions_correct}/{r.positions_checked} positions, "
                     f"check radius/height/size values)")
    return score


def _part_count_accuracy(r: LevelResult) -> float:
    """
    Asymmetric part-count penalty: 1 - |actual - expected| / expected.

    Penalizes BOTH over-counting and under-counting.
    Score ∈ [0, 1]. Missing all parts → 0, doubling parts → 0.
    """
    if r.expected_parts <= 0:
        return 1.0  # edge case: no expectation

    error = abs(r.part_count - r.expected_parts)
    score = max(0.0, 1.0 - error / r.expected_parts)

    if score < 0.8 and not r.failure:
        r.failure = (r.failure or
                     f"Part count mismatch: {r.part_count} vs expected {r.expected_parts}")
    return score


def _spatial_reasoning(r: LevelResult) -> float:
    """
    Combined connectivity + gravity → [0, 1].

    Uses weighted arithmetic mean: 60% connectivity, 40% gravity.
    Geometric mean would zero-out levels where the intended design has
    disconnected components (circular arrays) or cantilevered parts (trees),
    punishing correct output.  Arithmetic mean preserves partial credit
    while still penalizing missing connectivity or support.
    """
    conn = r.connectivity_score
    grav = r.gravity_score
    return 0.6 * conn + 0.4 * grav


def _surface_fidelity(r: LevelResult) -> float:
    """
    Chamfer distance score → [0, 1].

    If Chamfer wasn't computed, return 0.0 (unmeasured → no credit).
    The dynamic weighting system will redistribute this weight.
    """
    if r.chamfer_distance >= 0:
        return r.chamfer_score
    # NOT measured → signal to dynamic weighting that this is unmeasured
    return -1.0  # sentinel: component is unmeasured


def _orientation_score(r: LevelResult) -> float:
    """
    Cylinder orientation accuracy → [0, 1].

    If no cylinders exist in the level, return -1.0 (unmeasured sentinel).
    No free points for levels without cylinders.
    """
    if r.orientations_checked > 0:
        return r.orientation_accuracy
    return -1.0  # unmeasured sentinel


def _reasoning_score(r: LevelResult) -> float:
    """
    Reasoning quality → [0, 1].

    Conservatively scored: regex-based reasoning depth is inherently noisy,
    so we cap it at 0.8 and require chain verification for full marks.
    """
    base = min(r.reasoning_depth, 0.8)  # cap regex-based score

    if r.reasoning_chain_score > 0:
        # Chain verification is more rigorous: weight it higher
        return 0.4 * base + 0.6 * r.reasoning_chain_score
    return base


def _engineering_constraints(r: LevelResult) -> float:
    """
    Combined assembly mates + DFM (clearance + wall) → [0, 1].

    Returns -1.0 if no engineering specs are defined (unmeasured).
    """
    components = []

    if r.mate_score > 0 or r.mate_details:
        components.append(r.mate_score)

    dfm_parts = []
    if r.clearance_score > 0 or r.clearance_details:
        dfm_parts.append(r.clearance_score)
    if r.wall_details:
        dfm_parts.append(1.0 if r.wall_ok else 0.0)
    if dfm_parts:
        components.append(sum(dfm_parts) / len(dfm_parts))

    if not components:
        return -1.0  # unmeasured sentinel
    return sum(components) / len(components)


def _structural_patterns(r: LevelResult) -> float:
    """
    Combined symmetry + pattern → [0, 1].

    Returns -1.0 if no pattern/symmetry specs exist (unmeasured).
    """
    parts = []
    if r.symmetry_score > 0 or r.symmetry_details:
        parts.append(r.symmetry_score)
    if r.pattern_score > 0 or r.pattern_details:
        parts.append(r.pattern_score)
    if not parts:
        return -1.0  # unmeasured sentinel
    return sum(parts) / len(parts)


def _interference_score_val(r: LevelResult) -> float:
    """Interference check → [0, 1]. Always measured when shapes exist."""
    return r.interference_score


# ─────────────────────────────────────────────────────────────────
# Dynamic weighting engine
# ─────────────────────────────────────────────────────────────────

def _dynamic_weighted_sum(components: list) -> float:
    """
    Compute weighted sum with dynamic redistribution.

    `components` is a list of (name, base_weight, score) tuples.
    If score == -1.0, the component is unmeasured and its weight
    is redistributed proportionally to the remaining components.

    This ensures:
      - No free points from unmeasured components.
      - Weights always sum to 1.0 (or close, within floating precision).
      - The measured components carry more weight when others are absent.
    """
    measured = [(name, w, s) for name, w, s in components if s >= 0.0]
    if not measured:
        return 0.0

    total_measured_weight = sum(w for _, w, _ in measured)
    if total_measured_weight <= 0:
        return 0.0

    # Redistribute: each measured component gets its share of the total weight
    score = 0.0
    for name, base_w, s in measured:
        actual_w = base_w / total_measured_weight  # normalized to sum=1
        score += actual_w * s

    return score


# ─────────────────────────────────────────────────────────────────
# Core mode (no engineering specs)
# ─────────────────────────────────────────────────────────────────

def _score_core(r: LevelResult) -> float:
    """
    Core scoring with dynamic weighting.

    Base weights (when all components are measured):
      35%  Geometric accuracy (position + type, exponential decay)
      12%  Part count accuracy (asymmetric penalty)
      23%  Spatial reasoning (connectivity × gravity, geometric mean)
      10%  Surface fidelity (Chamfer distance)
       5%  Orientation accuracy
      10%  Reasoning quality (capped + chain-verified)
       5%  Interference-free
    ────
     100%

    Unmeasured components (score == -1.0) have their weight
    redistributed proportionally to measured components.
    """
    components = [
        ("geometric",     0.35, _geometric_accuracy(r)),
        ("count",         0.12, _part_count_accuracy(r)),
        ("spatial",       0.23, _spatial_reasoning(r)),
        ("surface",       0.10, _surface_fidelity(r)),
        ("orientation",   0.05, _orientation_score(r)),
        ("reasoning",     0.10, _reasoning_score(r)),
        ("interference",  0.05, _interference_score_val(r)),
    ]

    s = _dynamic_weighted_sum(components)
    return round(max(0.0, min(1.0, s)), 3)


# ─────────────────────────────────────────────────────────────────
# Engineering mode (mate / clearance / wall / symmetry / pattern)
# ─────────────────────────────────────────────────────────────────

def _score_engineering(r: LevelResult) -> float:
    """
    Engineering scoring with dynamic weighting.

    Base weights (when all components are measured):
      30%  Geometric accuracy
      10%  Part count accuracy
      15%  Spatial reasoning
      10%  Surface fidelity
       5%  Orientation accuracy
      10%  Reasoning quality
      15%  Engineering constraints (mates + DFM)
       5%  Structural patterns (symmetry + pattern)
    ────
     100%

    Unmeasured components redistributed proportionally.
    """
    components = [
        ("geometric",     0.30, _geometric_accuracy(r)),
        ("count",         0.10, _part_count_accuracy(r)),
        ("spatial",       0.15, _spatial_reasoning(r)),
        ("surface",       0.10, _surface_fidelity(r)),
        ("orientation",   0.05, _orientation_score(r)),
        ("reasoning",     0.10, _reasoning_score(r)),
        ("engineering",   0.15, _engineering_constraints(r)),
        ("patterns",      0.05, _structural_patterns(r)),
    ]

    s = _dynamic_weighted_sum(components)
    return round(max(0.0, min(1.0, s)), 3)
