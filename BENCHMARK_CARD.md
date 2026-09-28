# C2CAD-Bench Benchmark Card

## Intended Use

C2CAD-Bench is a diagnostic benchmark for coordinate-level 3D spatial reasoning in language models. A model receives a natural-language CAD-style specification and returns a JSON array of geometric primitives. The scorer compares the output to deterministic golden references using coverage, geometry, and semantic scores.

The benchmark is intended for:

- measuring primitive assembly reasoning separately from CAD API syntax;
- identifying failure modes such as wrong coordinates, wrong primitive type, missing topology, and violated clearances;
- comparing model behavior on a fixed set of released prompts and scoring rules;
- developing new CAD-oriented prompting, parsing, or model-training methods.

## Not Intended Use

C2CAD-Bench does not certify that a generated CAD design is manufacturable, structurally safe, physically complete, or suitable for engineering deployment. A high score means agreement with this benchmark's primitive references, not professional CAD validity.

## Task Format

Inputs are natural-language engineering prompts. Outputs are JSON arrays using seven primitive types:

```text
box, cylinder, sphere, cone, torus, pipe, beam
```

All dimensions are in millimeters. See `docs/schema.md` for canonical fields and accepted aliases.

## Dataset Scope

The released benchmark contains 25 test families, 3 difficulty levels per family, and 75 test cases. Families are grouped into four phases:

- P1: geometric forms
- P2: complex structures
- P3: engineering constraints
- P4: bio-inspired assemblies

## Evaluation

Scores are deterministic and computed without rendering or code execution:

- Coverage: expected versus produced primitive count.
- Geometry: position, type, and dimension agreement after one-to-one matching.
- Semantic: family-specific checks such as connectivity, clearance, symmetry, and pattern regularity.

Semantic and global scores are gated by coverage and geometry to prevent random or incomplete outputs from receiving inflated structural credit.

## Known Caveats

- The current v1.0 released prompts are the exact evaluated prompts. Some include explicit counts, vectors, axes, or formula-like construction hints. The repository includes `scripts/audit_prompts.py` and `data/prompt_audit.csv` to make these scaffolding signals visible.
- The benchmark uses a finite primitive vocabulary. Freeform surfaces, sketches, feature trees, chamfers, fillets, and constraint-solver workflows are outside scope.
- Commercial model APIs can change over time. Use the released raw outputs and `results/run_manifest_neurips2026.json` for the fixed paper snapshot.
- The WebGL viewer is for qualitative inspection only; scores are computed arithmetically.

## Responsible Reporting

When reporting results, include:

- repository commit SHA;
- model IDs and provider dates;
- runner settings, especially temperature, timeout, retries, and output token cap;
- whether results use the released outputs or live re-evaluation;
- any prompt or parser changes.
