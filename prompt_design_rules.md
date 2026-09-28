# Zero-Scaffolding Prompt Design Rules

This document summarizes the prompt construction rules used by C2CAD-Bench.
The goal is to test coordinate-level 3D spatial reasoning rather than whether a
model can copy formulas or derived coordinates from the prompt.

## Benchmark Scope

C2CAD-Bench contains 25 parametric test families across four phases:

| Phase | Families | Main demand |
| --- | ---: | --- |
| P1: Geometric Forms | 6 | Trigonometry, repetition, bolt-circle patterns |
| P2: Complex Structures | 6 | Hidden formula derivation and lattice connectivity |
| P3: Engineering Constraints | 6 | Mates, clearances, interference, mechanism layout |
| P4: Bio-Inspired Assemblies | 7 | Biological morphology and large-scale patterned structures |

Each family has three difficulty levels, yielding 75 total test cases.

## Rule 1: Describe Physical Intent, Not Derived Coordinates

Prompts should describe what the assembly must satisfy in physical terms:
tangency, adjacency, concentricity, radial alignment, gravity support, and
clearance. They should not state the derived coordinate or distance that the
model is expected to compute.

Example: prefer "the planet just touches the outer surface of the sun cylinder"
over "the planet center is 13 mm from the sun center" when 13 mm is the sum of
the two radii.

## Rule 2: Provide Base Variables Only

Independent input parameters such as radii, counts, pitch constants, clearances,
and component dimensions may appear in the prompt. Derived intermediates should
not appear.

Do not state quantities such as total height, angular increment, pitch-circle
radius, or center coordinates when they can be computed from other prompt
values.

## Rule 3: Anchor the Assembly

Each prompt must remove avoidable translation and rotation ambiguity. It should
define the world origin, relevant axes, and deterministic indexing order when
component IDs or repeated structures are expected.

Semantic-only engineering tests may allow multiple valid dimensions, but they
still need enough anchoring to make constraints testable.

## Rule 4: Use Domain Concepts Without Worked Solutions

Prompts may use standard terms such as helix, tangency, concentric, body-centered
cubic, golden angle, clearance fit, and interference. They should not explain the
formula or step-by-step construction method implied by those terms.

## Rule 5: Match the JSON Schema

The canonical schema uses seven primitive types:

```text
box, cylinder, sphere, cone, torus, pipe, beam
```

All dimensions are in millimetres. For primitives with a `center` field, that
field denotes the geometric centroid. Beams use `start` and `end` endpoints.
Surface-contact language is allowed when it describes a physical relationship
rather than leaking a coordinate.

## Prompt Audit Checklist

1. List every numeric literal in the prompt and confirm that it is a base
   parameter, not a derived answer.
2. Try to solve the task by copying numbers directly from the prompt. If that
   nearly solves it, the prompt leaks too much.
3. Confirm that the assembly has a deterministic frame of reference.
4. Remove worked formulas, procedural construction hints, and hidden answer keys.
5. Confirm that the expected output can be expressed using the seven canonical
   primitive types.

The deterministic Python generators compute the golden references from the same
base parameters given to the model. The model never receives the generator code
or golden coordinates during evaluation.

## Release Audit

The v1.0 artifact preserves the exact prompts used for the released model
outputs. Some prompts contain explicit shape counts, coordinate vectors, axes, or
formula-like construction hints. These are not silently edited after evaluation,
because doing so would invalidate the released model scores.

Run the audit below to inspect scaffolding signals:

```bash
python scripts/audit_prompts.py
```

For a future strict zero-scaffold split, use:

```bash
python scripts/audit_prompts.py --strict
```
