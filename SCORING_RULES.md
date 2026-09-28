# C2CAD-Bench Scoring Rules

This document describes the scoring system used by C2CAD-Bench to evaluate LLM-generated 3D assemblies against ground-truth references. The scorer uses strict gating mechanisms that prevent inflated scores when models fail to produce meaningful 3D output.

## Overview

Every test case produces four scores on a 0–100 scale:

| Metric | Weight in Global | What it measures |
|--------|-----------------|------------------|
| **Coverage (Cov)** | 20% | Did the model produce the right number of shapes? |
| **Geometry (Geom)** | 30% | Are the shapes in the right places, of the right type, with correct dimensions? |
| **Semantic (Sem)** | 50% | Does the assembly satisfy structural, physical, and family-specific constraints? |
| **Global** | — | Weighted composite of the three above, with a structural validity gate and calibration |

The raw global score formula before the validity gate and calibration is:

```
Global_raw = Cov × 0.20 + Geom × 0.30 + Sem × 0.50
```

The reported Global score applies the structural validity gate and then the
implemented calibration curve:

```
Global = 100 * ((Global_raw * validity) / 100) ^ 1.3
```

---

## Pre-Processing: Degenerate Shape Filter

Before any scoring happens, all LLM output shapes pass through a degenerate shape filter. A shape is discarded if any of the following are true:

- It has no resolvable 3D center (missing or NaN position).
- It has no recognized type (not cylinder, box, sphere, beam, pipe, cone, or torus).
- It has zero or near-zero dimensions for its type:
  - Cylinder: radius ≤ 0 or height ≤ 0.
  - Box: any size dimension ≤ 0.
  - Sphere: radius ≤ 0.
  - Beam: start and end points are identical (zero length).
  - Pipe: outer radius ≤ 0 or height ≤ 0.
  - Cone: base radius ≤ 0 or height ≤ 0.
  - Torus: ring radius ≤ 0 or tube radius ≤ 0.
- Any numeric field is NaN or Infinity.

Degenerate shapes are removed before coverage counting, geometry matching, and semantic evaluation. They cannot inflate any score.

---

## Coverage Score (Cov)

Coverage measures whether the model produced the expected number of shapes.

### Base Coverage

```
base_cov = min(1.0, actual_shapes / expected_shapes)
```

Producing fewer shapes than expected reduces coverage proportionally.

### Excess Shape Penalty (Hallucination Penalty)

Producing far more shapes than expected is treated as hallucination, not accuracy. If the model outputs more than 1.5× the expected count, an excess penalty is applied:

```
if actual > expected × 1.5:
    excess_ratio = actual / expected
    excess_penalty = 1.0 / (1.0 + (excess_ratio − 1.5))
    cov = base_cov × excess_penalty
```

| Actual / Expected | Excess Penalty | Final Coverage |
|-------------------|---------------|----------------|
| 1.0× | 1.00 | 100% |
| 1.5× | 1.00 | 100% |
| 2.0× | 0.67 | 67% |
| 3.0× | 0.40 | 40% |
| 5.0× | 0.22 | 22% |
| 10.0× | 0.11 | 11% |

---

## Geometry Score (Geom)

Geometry evaluates the spatial accuracy of each shape through deterministic greedy one-to-one nearest-neighbor matching against the golden reference. No matching IDs are required.

### Matching

For each golden shape, the best matching LLM shape is found in two passes:
1. Pass 1 traverses golden shapes in reference order and assigns the nearest unused LLM shape of the same canonical type.
2. Pass 2 revisits golden shapes that had no same-type partner and assigns the nearest unused LLM shape of any type.
3. Each LLM shape can be used at most once. Output order and LLM-provided IDs are ignored.

This is not a global Hungarian assignment; it is the scorer implemented in `runners/run_unified.py`. The type-first pass prevents a pool of wrong-type shapes from stealing same-type matches, while the fallback pass still records type-substitution errors.

### Per-Shape Geometry Score

Each matched pair is scored on three dimensions:

**Position accuracy (40% of shape score):**
```
pos_tol = max(2.0mm, assembly_diagonal × 0.05)
pos_score = max(0.0, 1.0 − distance / (1.5 × pos_tol))
```
The falloff reaches zero at 1.5× the tolerance radius. Shapes beyond this distance contribute nothing.

**Type accuracy (20% of shape score):**
```
type_score = 1.0  if types match
type_score = 0.0  if types differ
```
Wrong type receives zero credit. There is no partial credit for type mismatch.

**Dimension accuracy (40% of shape score):**
```
dim_score = 1.0 − mean(relative_errors)
```
Dimension comparison is type-aware (cylinders compare radius and height, boxes compare sorted size triplets, beams compare length, width, and endpoint-vector direction, pipes compare inner/outer radius and height, cones compare base/top radius and height, and tori compare ring/tube radii). If the type does not match, dimension score is forced to 0.0 since comparing dimensions across different shape types is meaningless.

**Combined per-shape geometry:**
```
shape_geom = pos_score × 0.4 + type_score × 0.2 + dim_score × 0.4
```

### Unmatched Golden Shapes

Golden shapes with no LLM match receive a geometry score of 0.0.

### Final Geometry Score

```
Geom = mean(all per-shape geometry scores) × 100
```

---

## Semantic Score (Sem)

Semantic evaluates whether the output satisfies structural, physical, and family-specific constraints (gravity support, connectivity, interference, symmetry, regularity, spacing patterns, etc.).

### Raw Semantic

Each test family has a dedicated evaluator that checks family-specific properties. The raw score (0.0–1.0) is normalized against the golden reference's own raw score so that the golden always receives 100%:

```
normalized_sem = raw_llm / raw_golden
```

### Coverage Gate

If most shapes are missing, structural checks on the few remaining shapes are meaningless. A harsh coverage gate is applied:

```
coverage_ratio = min(1.0, actual_shapes / expected_shapes)
coverage_gate = coverage_ratio ^ 1.5
```

| Coverage | Gate Value | Effect on Sem |
|----------|-----------|---------------|
| 5% | 1.1% | Nearly zeroed |
| 10% | 3.2% | Nearly zeroed |
| 20% | 8.9% | Heavily penalized |
| 30% | 16.4% | Heavily penalized |
| 50% | 35.4% | Significantly reduced |
| 70% | 58.6% | Moderate penalty |
| 90% | 85.4% | Mild penalty |
| 100% | 100.0% | No penalty |

### Geometry Quality Gate

If the geometry score is very low, the model did not produce a spatially coherent assembly. Semantic checks on randomly placed shapes would produce noise scores. A geometry quality gate prevents this:

```
if Geom < 10%:   geom_gate = 0.0
if 10% ≤ Geom < 20%:  geom_gate = (Geom − 10%) / 10%
if Geom ≥ 20%:   geom_gate = 1.0
```

| Geometry | Gate Value |
|----------|-----------|
| 0% | 0.0 |
| 5% | 0.0 |
| 10% | 0.0 |
| 15% | 0.5 |
| 20% | 1.0 |
| 50% | 1.0 |

### Final Semantic Score

```
Sem = normalized_sem × coverage_gate × geom_gate × 100
```

---

## Global Score

The global score is a weighted composite with a structural validity gate and final calibration curve.

### Weighted Composite

```
Global_raw = Cov × 0.20 + Geom × 0.30 + Sem × 0.50
```

### Structural Validity Gate

Geometry is the hard proof that a model produced a real 3D assembly. If geometry is below a minimum threshold, the global score is driven toward zero regardless of coverage or semantic:

```
if Geom < 5%:    validity = 0.0
if 5% ≤ Geom < 15%:  validity = (Geom − 5%) / 10%
if Geom ≥ 15%:   validity = 1.0

Global_gated = Global_raw × validity
```

| Geometry | Validity Gate | Effect |
|----------|-------------|--------|
| 0% | 0.0 | Global forced to 0 |
| 3% | 0.0 | Global forced to 0 |
| 5% | 0.0 | Global forced to 0 |
| 8% | 0.3 | Global reduced to 30% |
| 10% | 0.5 | Global halved |
| 15% | 1.0 | No penalty |
| 50% | 1.0 | No penalty |

### Calibration Curve

The implementation reports a calibrated Global score:

```
Global = 100 * (Global_gated / 100) ^ 1.3
```

This monotonic curve preserves model ordering for a fixed raw score comparison while compressing partially correct assemblies. All paper tables and figures use the calibrated value written to `score_global`.

---

## Example Scenarios

### Model produces perfect output
```
Cov=100  Geom=100  Sem=100  Global=100
```
All gates pass. Full marks.

### Model produces empty output (parse failure, crash)
```
Cov=0  Geom=0  Sem=0  Global=0
```
Nothing to score.

### Model produces shapes of wrong types at random locations
```
Cov=100  Geom=0  Sem=0  Global=0
```
Coverage looks fine (right count), but geometry is zero (wrong types, wrong positions). The geometry quality gate zeros semantic. The structural validity gate zeros global. The inflated coverage cannot save the score.

### Model produces degenerate shapes (zero-size, NaN)
```
Cov=0  Geom=0  Sem=0  Global=0
```
Degenerate shapes are filtered out before scoring. Effective output is empty.

### Model produces 10× more shapes than expected
```
Cov=11  Geom=100  Sem=80  Global=66 (approximately, after calibration)
```
Even if the correct shapes exist within the excess, the hallucination penalty crushes coverage. The model is penalized for not understanding the assembly specification.

### Model produces 1 correct shape out of 3 expected
```
Cov=33  Geom=33  Sem≈15  Global≈16
```
Partial credit is fair but harsh. The steep coverage gate (1.5 exponent) reduces semantic substantially.

---

## Comparing Two JSON Files

Every benchmark run produces two JSON files that are compared pair-by-pair to compute all scores.

### File Roles

| File | Role | Produced by |
|------|------|-------------|
| `showcase_golden.json` | Ground-truth reference assemblies | Parametric generators (deterministic) |
| `showcase_llm.json` | LLM-generated assemblies | Model inference |

Both files are JSON arrays. Each element represents one test case. The two arrays are aligned by `family` and `difficultyID` — not by array index.

### Entry Schema

A golden entry looks like this:

```json
{
  "family": "Spiral Staircase",
  "difficultyLabel": "Level 1 (Scale 10)",
  "difficultyID": 1,
  "shapes": [
    {
      "id": 0,
      "type": "cylinder",
      "center": [0.0, 0.0, 25.0],
      "radius": 5.0,
      "height": 50.0,
      "axis": [0.0, 0.0, 1.0]
    },
    {
      "id": 1,
      "type": "beam",
      "start": [5.0, 0.0, 1.0],
      "end": [25.0, 0.0, 1.0],
      "width": 5.0,
      "height": 2.0
    }
  ]
}
```

An LLM entry has the same outer schema (`family`, `difficultyLabel`, `difficultyID`, `shapes`). The `shapes` array is whatever the model output — it may use different field names, extra fields, wrong types, or missing fields. Normalization handles this before scoring.

After scoring, each LLM entry in the database also carries the four computed scores:

```json
{
  "family": "Spiral Staircase",
  "difficultyLabel": "Level 1 (Scale 10)",
  "difficultyID": 1,
  "shapes": [...],
  "score_cov": 85,
  "score_geom": 72,
  "score_sem": 61,
  "score_global": 69
}
```

### Matching: How Entries Are Paired

The two files are matched on `family` + `difficultyID`. This means:

- Entries do not need to be in the same array order.
- A golden entry with `"family": "Spiral Staircase", "difficultyID": 2` is compared only against the LLM entry with the same `family` and `difficultyID`.
- If no matching LLM entry exists for a golden entry, the result is zeroed (`score_cov=0`, etc.).

### Matching: How Shapes Are Paired

Within a matched entry, the golden `shapes` array and the LLM `shapes` array are compared shape-by-shape using **nearest-neighbor matching**, not ID matching:

- IDs in the LLM output are ignored for scoring purposes.
- For each golden shape, the closest LLM shape (by 3D Euclidean distance) of the same type is found.
- Each LLM shape can be matched to at most one golden shape.
- Unmatched golden shapes score 0 on geometry.

This means the LLM does not need to output shapes in any particular order, and does not need to preserve the golden IDs.

### Shape Field Normalization

LLMs often use non-standard field names. Before comparison, every LLM shape is normalized to a canonical schema:

| Canonical field | Also accepted as |
|----------------|-----------------|
| `center` | `position`, `pos`, `origin`, `location`, `centre`, `x`/`y`/`z` components |
| `size` (box) | `dimensions`, `dim`, `dims`, `width`/`height`/`depth` |
| `height` (cylinder) | `length`, `h`, `len` |
| `start` / `end` (beam) | `from`/`to`, `p1`/`p2`, `from_point`/`to_point` |
| `ring_radius` (torus) | `major_radius`, `R`, `large_radius` |
| `tube_radius` (torus) | `minor_radius`, `r`, `small_radius` |
| `inner_radius` (pipe) | `radius_inner`, `ri`, `bore_radius` |
| `outer_radius` (pipe) | `radius_outer`, `ro` |

Shape type aliases are also resolved:

| Input type | Canonical type |
|-----------|---------------|
| `cube`, `cuboid`, `rectangular_prism` | `box` |
| `cyl` | `cylinder` |
| `ball` | `sphere` |
| `bar`, `rod` | `beam` |
| `ring` | `torus` |
| `tube`, `hollow_cylinder` | `pipe` |

Normalization happens before degenerate filtering and before any scoring. A shape that normalizes successfully but is then found to be degenerate (zero size, NaN, etc.) is discarded.

### End-to-End Comparison Flow

```
showcase_golden.json          showcase_llm.json
        │                             │
        │   match by family +         │
        └──────── difficultyID ───────┘
                      │
              For each matched pair:
                      │
              ┌───────▼────────┐
              │  Normalize LLM │  (field aliases, type aliases, float casting)
              │  shapes        │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Filter        │  (remove degenerate shapes)
              │  degenerate    │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Coverage      │  actual vs expected count + excess penalty
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Geometry      │  nearest-neighbor match → pos + type + dim
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Semantic      │  family evaluator → coverage gate → geom gate
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Global        │  weighted sum → validity gate → calibration
              └───────┴────────┘
                      │
              score_cov / score_geom / score_sem / score_global
              written back into showcase_db.js
```

### Output Database

After scoring, results are written into `results/showcase_db.js` as a JavaScript global (`window.SHOWCASE_DB`) consumed by the HTML visualiser. The database is split by phase into `showcase_db_p1.js`, `showcase_db_p2.js`, `showcase_db_p3.js`, and `showcase_db_p4.js` for faster page loads. Existing results for other models are preserved across runs — only the target model's entries are updated.

---

## Design Rationale

The v2 scoring system is designed around one principle: **geometry is truth**. A model that cannot place the right shapes at the right positions in 3D space has not understood the task, regardless of how many shapes it outputs or how well a few isolated shapes happen to satisfy structural constraints.

The three-layer gating system (coverage gate on semantic, geometry gate on semantic, validity gate on global) ensures that each higher-level metric is anchored by the lower-level ones. This prevents the score inflation observed in v1, where models with no recognizable 3D output could still achieve 50–70% global scores by producing the right number of shapes or getting lucky on physics checks applied to randomly placed objects.
