# C2CAD-Bench v2: lab notebook

Running record of the rebuild for the resubmission. Every number below is printed by the script named next to it; its full output is in `v2/reports/results/`. The v1 artifact (`runners/`, `data/`, `results/`, and the reference code in `stages/`) is frozen and untouched. The v1 tests still pass (`pytest -q tests`: 11 passed).

Layout:

| Path | Contents |
|---|---|
| `v2/c2cad/geom.py` | Shape model and one normalizer, applied to both reference and output |
| `v2/c2cad/score.py` | Coverage and Geometry, using optimal (Hungarian) assignment |
| `v2/c2cad/constraints/` | Reference-free semantic constraints, one builder per family, each tied to a prompt clause |
| `v2/c2cad/cases.py` | v2 case set: generator prompts + output contract + documented patches |
| `v2/c2cad/evaluate.py` | One evaluation entry point |
| `v2/c2cad/stats.py` | Family-clustered bootstrap |
| `v2/experiments/eNN_*.py` | One script per experiment |
| `v2/c2cad/runner/` | Live-run machinery: providers, arms, sandbox, mock, runner, offline rescore |
| `v2/c2cad/cadkernel.py`, `cadcode.py`, `prism.py` | CAD-kernel bridge (converter, B-Rep recovery), CadQuery emitter, reading of label-free prisms |
| `v2/c2cad/partlevel.py` | Named-part metrics (probe arm) |
| `v2/config/models.json` | Model registry: ids, output caps, reasoning settings, prices, verification status |
| `v2/analysis/` | Pre-registered analysis (`reports/ANALYSIS_PLAN.md`), `python -m analysis.make_all` |
| `v2/RUNBOOK.md` | How to run everything, with the cost table |
| `v2/tests/` | `test_v2_validity.py` (scorer) and `test_v2_runner.py` (runner, arms, kernel, analysis); `pytest -q v2/tests` → 1,092 passed at the last update |

---

## E01. The v1 semantic validators contradict their own references
Script: `v2/experiments/e01_v1_validator_golden_audit.py` → `results/e01_v1_validator_golden_audit.md`

**18 of 25 families** have a reference answer scoring below 0.999 on its own v1 semantic validator at one or more levels. v1 hid this by dividing every model's Sem by the reference's raw score. That division lets a model score better by reproducing whatever the validator wrongly rewards.

Every failing sub-check could be read on the 11 instrumentable evaluators. Each is a validator defect, not a reference defect:

| Family | Failing sub-check | Why the check is wrong |
|---|---|---|
| DNA Helix | `z_pitch` = 0 | Z-spacing uniformity is computed over paired spheres that share a Z, so gaps of 0 occur |
| Suspension Bridge | `deck_horiz` = 0 | The deck is taken to be the longest beam, but the cables (up to 133) are longer than the deck (100) |
| Suspension Bridge, Domino Ring | `connectivity` ≈ 1/n | The connectivity helper treats touching beams and cylinders as disconnected |
| Domino Ring | `angular_reg` | Assumes a single-pillar ring, but pillars come in pairs |
| Fractal Y-Tree | `continuity` = 0.5 | Divides matches by (n−1); leaves have no children, so the maximum is about 0.5 |
| BCC Lattice | raw = 0.9 | The sub-check weights sum to 0.9 |
| Cannonball Pyramid, Planetary Array | `interference` | Tangent contacts, which the prompt requires, are counted as interference |
| Furniture Assembly | `leg_symmetry` | Checks angular symmetry that the prompt does not require for the extra legs |

Pipe Manifold (0.62), Axle Bearing (0.64), Gantry/Cochlea/Radiolarian (0.925, constant) and Diatom (0.88, constant) could not be instrumented. Their constant values across levels point to the same kind of defect.

## Reference-scoring bugs in v1 (verified earlier this session)
- 16/75 references score below 100 geometry against themselves. The v1 normalizer runs on the output only, so reference tori written as `major_radius`/`minor_radius` and cones written with `start_radius` or `radius` fail the dimension comparison.
- 15 further cases lose points when the reference order is reversed. Greedy matching breaks ties by order, and X-braces share midpoints.
- The released scores use a calibration exponent of 1.0 in 19 families and 1.3 in 6. 511/975 released Global values cannot be reproduced from the released code.
- The paper's type-confusion table does not reproduce; see E03.

## v2 case set
Built by: `v2/c2cad/cases.py` → `v2/data/cases_v2.jsonl` (75 cases)

The prompt bodies are the user's v2 generator prompts, with their per-generator output suffix replaced by one shared **output contract**. The contract lists field names per primitive and defines `center` as the axis-extent midpoint, so for cones it is not the volume centroid. Neither v1 nor v2 prompts ever gave field names. The paper claimed a "universal system prefix that defines the JSON schema", but the actual system prompt only says "JSON-only generator".

Documented patches. Each fixes a verified disagreement; the user may veto the ones marked (P), which change prompt text:

| Family | Patch | Evidence |
|---|---|---|
| Suspension Bridge | Deck is listed first in the ids | Prompt: "Index deck, negative-X tower, …"; the reference listed the towers first |
| Suspension Bridge | Deck section 1 × 1 | Prompt: "square section 1"; the reference had 2 × 1. The v1 prompt also said 1 |
| Flanged Pipe Joint | Nut torus axis set to X | Prompt: "its axis is parallel to X"; the reference had no axis |
| Axle Bearing (P) | Bearings outside the block, abutting its end faces; shaft protrudes 20 | Bearing outer radius 14 sat inside a bore of radius 8.5–9.5, so the reference violated its own no-interference constraint. The v1 prompt said "one on each side of the block" |
| Honeycomb (P) | Adds "The central cell stands on the base plate's top face." | The central cell's elevation was stated nowhere |
| Honeycomb | Link ids ordered by endpoint-id pairs | Prompt rule: "Link IDs follow increasing endpoint-ID pairs" |

## v2 scoring (spec, final)
- **Coverage** = 100 · min(n, N) / max(n, N).
- **Geometry**: optimal one-to-one assignment. A pair's score is pos × ori × (0.4 + 0.2·type + 0.4·dims), a "pose gate".
  - pos = max(0, 1 − d / (1.5 τ)), with τ = max(2 mm, 5% of the reference bounding diagonal).
  - ori = 1 − min(1, angle / 45°). Cones are compared as directed axes; other axes as unsigned lines. Spheres and boxes have ori = 1, and box orientation is checked through the per-axis sides.
  - dims = 1 − mean relative dimension error.
  - Type and shape credit only count where, and as, the part is placed. Under v1's additive form, randomly placed parts kept 60% of the geometry score.
  - The geometry-equivalent view also scores an axis-aligned box as a beam. It never lowers a pair's score; a test enforces this.
- **Normalizer**:
  - reads Euler `rotation` fields (XYZ order, degrees) when no axis is given, and counts them;
  - ignores box rotation (boxes are axis-aligned by the schema) and counts it;
  - drops out-of-schema types and counts them.
- **Semantic** = mean over prompt clauses of the fraction of that clause's constraints satisfied.
  - Tolerances: length max(0.25 mm, 0.2% of the diagonal), angle 0.5°, dimensions 1% relative.
  - Parts are bound to roles by the optimal assignment. Binding by the output's own ids is reported separately.
  - Constraints involving a missing role fail. There are no gates and no normalisation by the reference.
- **Global_v2** = mean(Cov, Geom, Sem). **exact** = every constraint satisfied and the part count exact.

## Validity checks (all pass)
Script: `pytest -q v2/tests` → 842 passed (at the time of the last notebook update)
- 75/75 references score 100 on every axis.
- 75/75 are invariant to shuffling and renumbering.
- 24,579 constraints in total. Each has a clause, a kind and valid roles, and the reference satisfies all of them.
- Geometry decreases monotonically under jitter.
- The equivalence view is never below the label-strict view.
- The `{"shapes": [...]}` wrapper is accepted. Out-of-schema types are dropped and counted.

## E02. Response to controlled corruptions (final scorer)
Script: `v2/experiments/e02_perturbation_validity.py` → `results/e02_perturbation_validity.md`

| Corruption | Cov | Geom | GeomEq | TypeFid | Sem | Global | anchor | mate | pattern |
|---|---|---|---|---|---|---|---|---|---|
| identity / shuffle / 1-based ids | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| translate 2 mm | 100 | 75.7 | 75.7 | 100 | 81.3 | 85.7 | 46.9 | 92.7 | 57.1 |
| rotate about Z 10° | 100 | 48.6 | 48.6 | 99.1 | 79.5 | 76.0 | 75.4 | 83.6 | 57.9 |
| jitter σ = 0.5 mm | 100 | 89.2 | 89.2 | 100 | 58.3 | 82.5 | 28.7 | 25.6 | 28.5 |
| jitter σ = 2 mm | 100 | 60.6 | 60.6 | 99.2 | 45.9 | 68.8 | 5.6 | 10.6 | 6.5 |
| delete 25% | 75.0 | 75.0 | 75.0 | 75.0 | 64.4 | 71.5 | 72.7 | 49.5 | 59.3 |
| duplicate 50% | 66.8 | 100 | 100 | 100 | 100 | 88.9 | 100 | 100 | 100 |
| beam → axis-aligned box | 100 | 80.4 | 83.7 | 67.4 | 73.5 | 84.7 | 88.1 | 57.9 | 69.4 |
| mirror Y (handedness) | 100 | 71.2 | 71.2 | 98.0 | 91.0 | 87.4 | 89.3 | 89.6 | 83.8 |
| random positions, types kept | 100 | 9.0 | 9.0 | 73.6 | 26.2 | 45.1 | 3.5 | 4.2 | 0.7 |
| count only (random unit boxes) | 100 | 6.3 | 6.6 | 9.2 | 3.4 | 36.6 | 2.5 | 3.5 | 0.6 |

What the table shows:
- A rigid move fails only the anchor constraints; mates and patterns are relational.
- Sub-millimetre jitter barely moves Geometry (89.2 at σ = 0.5 mm) but halves the Semantic pass rate (58.3). The constraint layer is the part that checks engineering exactness.
- Global stays at 36.6 for count-only output because Coverage is 100, so the paper leads with the components.

## E03. The 975 released v1 outputs rescored with v2
Script: `v2/experiments/e03_rescore_v1_outputs.py` → `results/e03_*.md`, `e03_rows.csv`

> **Caveat on the released v1 outputs.** `data/model_outputs.jsonl` stores shapes *after* v1 normalization and degenerate filtering, and the raw model responses are not in the artifact. Parts that v1 dropped but v2 would accept cannot be recovered. Findings about "missing parts" therefore describe the released record, not necessarily what the model produced. Ask the user whether the raw responses exist.


Furniture and Axle are excluded because their v1 prompts said "choose the dimensions freely", leaving 897 outputs.

- **Rankings.** Rank agreement between v1 Global and v2 Global is Spearman ρ = 0.868 (Kendall τ = 0.718).
  - The top three are unchanged: gemini-3.1-pro 84.1 [74.9, 91.9], gpt-5.4 82.2 [74.9, 89.0] and gemini-2.5-pro 81.5 [73.6, 88.6], with P(top-3) of 98%, 96% and 99% under the family-clustered bootstrap.
  - The middle reorders: gemini-3-flash moves from rank 10 to 6, gpt-4.1 from 4 to 7, deepseek-chat from 9 to 11, deepseek-reasoner from 13 to 10.
- **Exact assemblies** (every stated constraint met and the part count exact): from 4.3% (deepseek-chat) to 42.0% (gemini-2.5-pro).
- **Co-located type substitutions:** beam→box 2,049 of 2,541 matched pairs, mostly in Truss (864), Staircase (573) and Clock (205); beam→sphere 237 (Radiolarian only); beam→pipe 106 (Honeycomb); cone→cylinder 91 (Compound Eye 80). The paper's "beam→sphere 8,644" does not reproduce.
- **Hardest families**, by mean Global_v2 over all 975 outputs: Radiolarian 30.3, Axle 34.9 (excluded), Pipe Manifold 44.9, Furniture 52.0 (excluded), Compound Eye 52.1, Vertebral 54.0. Easiest: Planetary 97.2, Pyramid 94.8, Voxel 89.7.
- *History:* before the E07 normalizer fixes, ρ was 0.797. The difference comes from parts v2 had wrongly discarded.

## E04. Sensitivity to every free scorer choice
Script: `v2/experiments/e04_metric_sensitivity.py` → `results/e04_metric_sensitivity.md`

20 variants were tested:
- pair form (additive or pose-gated);
- coverage form;
- position tolerance at 2.5% or 10% of the diagonal;
- orientation saturation at 15° or 90°;
- constraint tolerances × 0.5, 2 or 5;
- six weightings, including Geometry-only and Semantic-only;
- all v1-like choices together;
- three ways of grouping constraints into Sem.

The top-3 set is the same in all 20. The lowest rank agreement with the default is Spearman ρ = 0.940, and the lowest Kendall τ is 0.795. This replaces the rebuttal's "32 configurations, ρ ≥ 0.95", which was computed with the buggy v1 scorer.

## E05. Agreement with standard 3D shape metrics
Script: `v2/experiments/e05_convergent_validity.py` → `results/e05_convergent_validity.md`, `.csv`

> **Caveat on the released v1 outputs.** `data/model_outputs.jsonl` stores shapes *after* v1 normalization and degenerate filtering, and the raw model responses are not in the artifact. Parts that v1 dropped but v2 would accept cannot be recovered. Findings about "missing parts" therefore describe the released record, not necessarily what the model produced. Ask the user whether the raw responses exist.


Metrics compared: 96³ voxel IoU, Chamfer distance divided by the reference diagonal, F-score at 2% of the diagonal (the self-comparison noise floor makes 1% unreliable), and orientation error over matched parts.

- **Case level** (Spearman, n = 897): Geometry vs IoU 0.807, vs F@2% 0.816, vs Chamfer 0.753. Semantic vs IoU 0.823. Coverage vs IoU only 0.484.
- **Model level** (ρ against Global_v2): IoU 0.896, F@2% 0.951, Chamfer 0.918, orientation error 0.736.
- **Where they disagree, and why component-level scoring matters:**
  - 31.0% of outputs have F@2% ≥ 0.9 but are not exact: a stated constraint is violated or the part count differs.
  - 6.0% have IoU ≥ 0.8 with Sem < 60. Example: the *released record* for kimi-k2.5 on clock L1 contains only 2 of 12 parts (back plate and shaft), yet scores IoU 0.92 and F@2% 1.00, because the plate dominates the volume. Whether the raw response had more parts is unknown (see the caveat above).
  - 2.3% have IoU < 0.3 with Geom ≥ 80. Example: ball-bearing L2, where the released records of five models contain all 12 balls, exactly placed, but neither race (2 parts, about 90% of the volume): IoU 0.09, Geom 85.7.

### E04 addendum: how constraints are grouped into Sem
The semantic score averages over clauses, so the grouping acts as a hidden weight. Three alternatives were tested: the mean over constraints, over kinds, and over prompt sentences. All three keep the default top-3 set; they are included among the 20 variants in E04.

## Traceability: prompt sentences and constraints
Module: `v2/c2cad/constraints/trace.py`

Every sentence of every one of the 75 prompts is mapped to one or more of the following, and a test enforces it:
- the constraint keys that enforce it;
- `ids`: an ordering rule, evaluated through the output's own ids;
- `symbolic`: a declaration that affects interference and volume metrics only;
- `note`: a clarification with no separate requirement.

A second test checks that every constraint key a builder emits is stated in at least one prompt sentence.

Wording for the paper: "every constraint is mapped to a prompt sentence". Two caveats must stay with that sentence:
- The map and the builders were written by the same author (Claude, in this session). An independent audit is still to do.
- For Pyramid tangency pairs, Truss joints and the Fractal parent tree, *which roles relate* is taken from the reference. Residuals never use reference coordinates.

Do not claim "exactly the relations the prompt states and nothing else".

## Experiment runner (live runs need API keys)
Package: `v2/c2cad/runner/`; entry point `python -m c2cad.runner.run`.

- **Providers.** OpenAI, Anthropic, Google, DeepSeek and Moonshot, plus OpenRouter and Together for open-weight models, all over plain HTTPS. Keys come from the environment or `.env` and are never logged.
- **Mock providers** (`mock:reference|jitter-σ|beam-box|empty`) test the pipeline offline.
- **Records.** Every request is appended to `runs/<run>/responses.jsonl`: model, returned model version, settings, system and user prompt hashes, raw text, usage, latency, finish reason. Every score goes to `scores.jsonl`. The manifest also holds the code hash, the case-file hash and argv. Runs are resumable, and k samples are drawn per case.
- **Arms.** Each changes one factor:

| Arm | What the model receives | What it isolates |
|---|---|---|
| `json` | v2 prompt + output contract | baseline: the model writes coordinates |
| `neutral` | v2 prompt with object and domain nouns replaced + contract | recall of known objects |
| `tool` | v2 prompt + contract + "write a Python program that prints the JSON" | arithmetic, which is offloaded to an interpreter |
| `v1` | released v1 prompt (formulas and coordinates leaked) + contract | the released prompt. This is not a pure scaffolding ablation: its wording differs, and for Clock so does its specification ("10 o'clock (120°)") |
| `mates` | v2 prompt + CML reference (parts, mates, patterns; a solver computes coordinates) | derivation of poses and patterns |

- **Neutral twins** (`runner/neutral.py`). Tests check all 75: no banned domain word survives, and the set of numbers is unchanged. The one listed exception is the Clock, where "eleven o'clock" becomes "30 degrees counterclockwise from +Y".
- **Tool sandbox** (`runner/sandbox.py`).
  - An AST allow-list permits only the math, numpy, itertools and json imports, and refuses file, process and network calls, reflection, and dunder access.
  - The program runs in an isolated interpreter with CPU and memory limits and a 30 s wall clock.
  - Verified: numpy runs; `os`, `open` and dunder access are rejected; an infinite loop is killed.
- **Parser.** Handles raw, fenced and wrapped JSON (`{"shapes": [...]}`), and truncated arrays (repaired, with the status recorded).
- **Offline self-test.** 1,764 mock requests (3 mock models × 4 arms × k = 2) scored as expected. A dry run confirms that 2 real models × 4 arms × k = 3 is 1,764 requests.

## CML: the mates-and-patterns interface ("a way forward")
Modules: `v2/c2cad/dsl.py` (language reference and interpreter), `v2/c2cad/cml_programs.py` (hand-written gate programs)

- **Design.** The model declares parts, attaches them to features of other parts, and uses patterns. A deterministic interpreter places parts in program order with closed-form geometry only.
  - Features: faces, axes, surface points and beam faces.
  - Mates: tangency (`touch`), face and plane contact (`rest_on`, `top_at`, `bottom_at`, `top_through`), coaxial placement (`axis_of`), and sphere nesting in pockets.
  - Patterns: circular, helical, linear (optionally centered) and mirror.
- **Limits.** There are no free-form expressions: dimensions are literals or references to declared dimensions. There are no family macros. Invalid programs raise explicit errors, which count as their own failure category.
- **Gate.** The hand-written programs reproduce the reference **exactly for 16 of 25 families (48/48 cases)**: every constraint is satisfied and Geometry ≥ 99.97, the shortfall being rounding in the references. A test enforces this. Wording for the paper: "verified against the reference", not "written from the prompt text", because the author had seen the references. The literal-provenance check is what supports the prompt-only claim.
- **What the model would still have to derive**, from a check of every numeric literal against the prompt:

| Category | Families |
|---|---|
| Prompt numbers only (plus the constants 0, ½, 1, 2) | Staircase, Voxel, DNA, Flanged, Planetary, Fractal, Ball Bearing, Manifold, Axle |
| Plus counts (row sizes, n+1 columns) | Pyramid, BCC, Gantry |
| Plus simple fractions or halves | Domino (the half-angle 3°), Bridge (i/(n−1) attachment fractions), Furniture (k/(P+1) stations) |
| Plus domain knowledge | Clock ("eleven o'clock" → 120°) |
| Not yet written | Phyllotaxis (Fermat law), Cochlea (taper law), Radiolarian (geodesic refinement), Vertebral (accumulated tilt), Compound Eye (70k/R polar angles), Armillary (φ-based directions), Diatom and Honeycomb (many derived offsets) |

- **Summary of the table.** 9 families need only prompt numbers, 6 also need counts or simple fractions, and 1 (Clock) needs one domain convention (11 o'clock → 120°). The remaining 9 families are not yet written.
- **Why this matters for the paper.** Generic CAD vocabulary covers regular engineering assemblies with prompt numbers alone; growth laws are where it stops helping. Comparing the `mates` arm with the `tool` arm will show whether computing coordinates is the bottleneck (mates ≈ tool > json) or whether a relational interface helps beyond arithmetic (mates > tool).

## E07. Cross-check against an independently written matcher
Script: `v2/experiments/e07_cross_implementation.py` → `results/e07_cross_implementation.md`

The June 2026 `runners/matching.py` in the author's `C2CAD` working copy is an independent optimal-assignment scorer. v2 was not derived from it, and it passes its own 22 tests. Both scored the 897 comparable released outputs.

- **Like-for-like** (v2 in its additive / v1-coverage configuration):
  - the model geometry ranking is identical (Spearman 1.000, Kendall 1.000);
  - Coverage agrees in 100% of cases;
  - case-level Geometry Spearman is 0.988, with a mean absolute difference of 0.87 and 89.9% of cases within 2 points;
  - the remaining gaps (at most 14.6) are Flanged outputs that give orientation as Euler angles, which the June code does not read.
- **The cross-check found two gaps in v2's normalizer**, now fixed and tested:
  1. Beams with no width or height were dropped as degenerate. deepseek's Fractal outputs, with correct centerlines, scored Geom 0. Such beams are now kept, and their section is scored as wrong.
  2. Euler angles given under the name `orientation`, e.g. `[0, 90, 0]`, were ignored, so X-axis pipes were read as Z-axis. Now a unit vector is read as a direction and anything else as Euler degrees.

  E02–E05 were re-run after these fixes, and the numbers above are the updated ones.
- **v2 default vs June** (pose-gated vs additive): the model ranking agrees at ρ = 0.918. The largest differences are intended. Example: deepseek-chat's Voxel L1 output puts cube *centers* on the lattice corners. The structure is right but everything is shifted by half a cube, so v2 gives Geom 0 and Sem 80 (every gap, alignment and size constraint passes, only the anchor fails), while the additive June score gives 60.

## The author's `C2CAD` working copy (last modified 2026-06-13)
- **No CadQuery, OpenSCAD, IoU or Chamfer code is present**, here or anywhere under `~/Documents`. The rebuttal's format study must be rebuilt. CadQuery 2.8.0 is installed in the git-ignored `v2/.venv-cad`, a Python 3.12 environment made with uv.
- **Useful contents:**
  - the independent matcher (used in E07);
  - `manual_eval.py`;
  - four new families not in the release: `phase5_kinematics/generate_fourbar.py` and `generate_fourbar_synth.py` (four-bar linkage analysis and synthesis), and `phase6_engineering/generate_stackup.py` and `generate_swept.py`, with tests;
  - `C2CAD_paper_v3.pdf`, a June draft;
  - `results/LANGUAGE_AGNOSTIC_METHODOLOGY.md`.
- **Unsupported claim** in that methodology note: "switching to Hungarian matching changes fewer than 0.3% of shape pairings". It has no supporting script; do not reuse it.

## Run settings decided
- **Temperature:** one explicit temperature, 1.0, for every model and arm (user decision, 2026-09-28). It is recorded in every run manifest.

## Hardening for live runs (done before any API spend)
- **Output budget.** Output tokens are sized per case: about 90 tokens × parts × 1.6 + 2K for coordinate arms, and 12K for program arms, plus optional reasoning headroom, capped by the model's documented output limit.
  - `--output-cap` is required for real providers.
  - Cases that cannot fit are recorded as `infeasible`, never as failures.
  - Truncated responses (`finish_reason` of length) get their own status, `truncated`.
- **Sandbox.** Code runs in an empty temporary directory with `RLIMIT_FSIZE = 0`. numpy and json file-I/O names (`save`, `tofile`, `loadtxt`, `fromfile`, `memmap`, `f2py`, `dump`, ...) are rejected, with tests for each.
- **Mates arm fairness.** The prompt includes two validated worked examples on assemblies outside the benchmark (a wheel and a shelf). Analysis must report the invalid-program rate separately, and must report the 16 gated families separately from the 9 ungated ones.
- **BCC constraint.** `strut_diag` mixed millimetres and degrees; it is now two constraints, one on direction and one on length.

## Scale-sweep split
Built by: `cases.build_sweep` → `v2/data/sweep_v2.jsonl`

121 cases across 19 families, from 3 to 701 parts. These are the same prompt templates at counts not used in the main set. All references satisfy all constraints, with a test. Outside the main scales, 23/25 families generate valid cases; Vertebral (fixed anatomical regions), Radiolarian (generator stops at refinement level 2) and Cochlea's half-revolution are excluded. Purpose: per-model breaking-size curves. Per c9s1 and the AC, this is scale, not generalization.

## Held-out generalization split
Built by: `v2/c2cad/heldout.py` → `v2/data/heldout_v2.jsonl`

48 cases: 8 families × 2 variants × 3 levels. The families are Staircase, Planetary, DNA, Domino, Voxel, Ball Bearing, Flanged and Bridge.
- Each variant changes every base dimension. Most also mirror the handedness (clockwise, left-handed helix) and rotate the first-element anchor (+Y, −X, −Y, or a bolt first on +Z).
- The parametric generators reproduce the user's prompt text *exactly* and the reference to Geometry ≥ 99.995 at the default parameters (a test).
- All held-out references satisfy all constraints (a test).
- **The memorized default answer fails every held-out case** (not exact, Sem < 90; a test). The split therefore rewards reading the spec, not recalling the template.

## Prior art (see `reports/PRIOR_ART.md`)
- The mates or solver interface for LLM CAD already exists (AIDL, CGF 2025; AssemCAD, Jul 2026; Embodied CAD, Jun 2026).
- Multi-format output with part-level scoring already exists (P3D-Bench, Jun 2026).
- **Claim instead:** the controlled attribution on exactly specified tasks, the clause-traced constraints, the metric validity evidence, held-out and scale generalization, and the map of where CAD-style interfaces stop helping.

## Run readiness (2026-09-28, second session)

The rebuttal commitments were read in full (the four OpenReview posts and the long `.docx` response in `~/Documents/replay to reviews/`). Every one is mapped to v2 in `reports/REBUTTAL_TRACKER.md`. The rebuttal's own code was not in that folder (it holds a copy of the v1 repository only), so its CadQuery and IoU pipeline was rebuilt (E08, a05).

### E00. Prompt scaffolding audit (a saved script for the outline's numbers)
Script: `v2/experiments/e00_prompt_audit.py` → `results/e00_prompt_audit.md`
- **Regex flags.** The released v1 regexes flag 60/75 v1 prompts (48 shape count, 27 coordinate vector, 9 formula or assignment, 18 trig or Cartesian). They flag 0/75 v2 task texts, and 0/75 for the task-specific text of every v2 arm.
- **Shared instruction blocks** are audited once:
  - the output contract matches only through the literal `{"shapes": [...]}`;
  - the CadQuery instruction matches only through `parts[0]`;
  - the mates reference contains coordinate vectors in its two worked examples, which lie outside the benchmark.

  None of these is a task value.
- **Verbatim coordinate overlap**, counted by occurrence: 22.7% for v1 and 13.2% for v2. Counted by distinct value per case: 10.5% and 7.2%.
  - Highest in v2: Gantry 71.4%, Pipe Manifold 51.2%, Suspension Bridge 50.0% and Honeycomb 40.8%. These still need the hand review.
- **CI gate.** `tests/test_v2_runner.py::test_audit_gate_task_text_of_every_arm_has_no_scaffolding_flag`.

### E02 additions: decomposition and rescale (Reviewer 3)
Same script; the rows already in the table are unchanged.

| rewrite of the reference | Cov | Geom | Sem | Global | exact % | dimension | mate | orientation | topology |
|---|---|---|---|---|---|---|---|---|---|
| split every box into two abutting halves | 93.9 | 94.6 | 95.4 | 94.6 | 68.0 | 97.6 | 88.0 | 100 | 100 |
| split every beam into two collinear halves | 80.7 | 80.2 | 90.6 | 83.8 | 44.0 | 96.6 | 67.3 | 99.7 | 93.9 |
| scale × 1.10 about the origin | 100 | 55.9 | 66.8 | 74.3 | 0 | 39.5 | 89.6 | 98.3 | 100 |
| scale × 1.25 | 100 | 20.5 | 60.7 | 60.4 | 0 | 37.2 | 70.1 | 87.5 | 100 |
| scale × 1.50 | 100 | 8.9 | 60.2 | 56.4 | 0 | 36.6 | 68.7 | 88.1 | 100 |

- **The rescale result differs from the rebuttal.** The rebuttal reported Sem ≈ 99 at 1.5× under v1. Under v2, every prompt states its dimensions, so a rescaled answer really is wrong: dimension constraints fail (about 37%) while orientation and topology hold. The rebuttal's sentence "Semantic generalises to alternative valid designs" must not be reused as written.
- **Decomposition.** Finer decompositions cost 5.4 points (boxes) and 16.2 (beams). v2 prompts prescribe each part and its id, so a split is a deviation from the specification.
- **Open decision:** the merge pass promised to Reviewer 3 is not implemented (see the tracker).

### E08. CAD-kernel round trip
Script: `v2/experiments/e08_kernel_roundtrip.py` → `results/e08_kernel_roundtrip.md`
- **Converter** (`c2cad/cadkernel.py`): 5,042 of 5,042 reference primitives build as valid OpenCascade solids, and all 5,042 are recovered from their B-Rep faces.
  - The prisms among them (2,502) come back label-free.
  - By type: beam 2,265, box 237, cone 446, cylinder 689, pipe 160, sphere 1,160, torus 85.
- **The full cadquery-arm path** (program → sandbox → kernel → recovery → evaluator) gives 75/75 cases exact. The lowest Geometry is 99.999998, which is rounding.
- **Rule for label-free prisms** (`c2cad/prism.py`). A rectangular CAD solid carries no box/beam label and no centerline, so the evaluator takes the reading that best fits the reference: an axis-aligned box, or a beam along one of its three axes. The occupied volume is identical for every reading. This is the same principle as the geometry-equivalent view.
  - Why it is needed: the two types cannot be told apart by shape. The Domino lintels are beams shorter than they are wide, and some Manifold boxes are 4.5:1.

### Runner hardening (before any paid request)
- **Resume.** Previously an `api_error` was never re-requested. Now only ok and infeasible responses count as done.
- **Retries.** 408/409/425/429/5xx/529, connection errors and timeouts get exponential backoff that honours `retry-after`. Other 4xx errors are not retried.
- **Streaming (SSE)** for every provider. The Anthropic path uses the official SDK. The idle timeout replaces the fixed 600 s request timeout.
- **Temperature.** Claude Opus 5/5.5, Sonnet 5 and Fable 5.x return a 400 when sent a temperature. The registry flag `send_temperature` now omits it, and the record states that 1.0 was requested but not sent.
- **Scoring is separate from requesting.** The paid response is written first, so a scoring crash is recorded as `harness_error` and nothing is lost. `python -m c2cad.runner.rescore` re-scores offline and reproduces the stored scores exactly (checked on 456 mock responses).
- **Programs.** Any exception from a model-written CML program is that program's `invalid_program`, not a runner crash.
- **Budget.** `--max-usd` is metered on the providers' own usage (hidden reasoning included) for the listed profiles.
- **Concurrency.** Appends are file-locked. Three processes writing 900 lines of 200 KB concurrently left every line intact.
- **Infeasible** now means that the answer alone cannot fit the cap. The reasoning reserve is cut back first.
- **`--smoke`** sends the smallest and largest feasible case per arm and checks the finish reason, usage, returned model, parse and truncation.
- **Model registry** (`config/models.json`): ids, output caps and prices checked on 2026-09-28, with the source of each.
  - DeepSeek's `deepseek-chat` and `deepseek-reasoner` aliases were retired on 2026-07-24, according to third-party pages; the official docs refused the connection.
  - The DeepSeek, Kimi and OpenRouter entries are marked `verified: false`.

### New arms (frozen before the runs)
| arm | design | rebuttal commitment |
|---|---|---|
| schema | the json prompt with provider-native JSON-schema decoding (OpenAI strict json_schema, Anthropic `output_config.format`, Google `responseJsonSchema`, OpenRouter with `require_parameters`). The enforcement actually used is recorded per response. A test checks that all 75 references validate against the schema | R1 and AC: constrained decoding |
| cadquery | the prompt; the model writes CadQuery with one solid per id (`parts = {id: solid}`), which runs in `.venv-cad` under the same sandbox policy (plus a ban on CadQuery file I/O). Primitives are recovered from the faces. Unions and non-primitive solids are counted as unrecognised | R1, R2: the generator-side code-vs-JSON experiment |
| probe | the prompt; the model returns only three named parts, fixed per case (middle id, last id, one seeded id). The same parts are scored inside every full answer, under id binding and under assignment binding | R1 W2: serialisation load |
| repair_generic / repair_verifier | multi-turn. Seeds are non-exact json answers (sample 0). Up to 2 rounds of either "check and fix" or the reference-free verifier (`evaluate.verify`): constraints bound by the answer's own ids, violated prompt sentences quoted, missing ids and the part count listed, no reference coordinates. The verifier is silent on all 75 references (test) | R2: iterative refinement |

### Analysis plan and pipeline
- **The plan** (`reports/ANALYSIS_PLAN.md`) was committed before any live run. It covers:
  - estimands and the handling of each response status;
  - H1–H8 with decision rules;
  - tiers and variance decomposition;
  - breaking size, with infeasible and truncated responses censored;
  - held-out gap and memorization index;
  - validity on live outputs;
  - v1 continuity.
- **The pipeline** is `v2/analysis/` (a01–a06 plus figures), run as `python -m analysis.make_all`. It writes `paper_numbers.tex` macros. It was tested end to end on about 8,300 mock records: 3 mock models, every arm, main + sweep + held-out.

## Open items
- [x] E04, E05 done (above).
- [x] Runner, arms (json, neutral, tool, v1, mates, schema, cadquery, probe, repair_*), registry, smoke mode, analysis pipeline.
- [x] The rebuttal's CadQuery and IoU code was rebuilt (E08, a05); OpenSCAD is not installed and has no arm (user decision).
- [x] Atlas: `reports/atlas/` has 75 reference PNGs, 4 phase contact sheets, and the Staircase panel with its three levels and prompts.
- [ ] Live runs (`RUNBOOK.md`): confirm the unverified registry entries, smoke each profile, then run main, repair, sweep and held-out.
- [ ] Decisions: sign-off on the (P) patches and the contract; the Axle Bearing levels (still 5 parts each); the decomposition merge pass; the phase-5/6 families; the roster and reasoning settings; the budget.
- [ ] Render symbolic parts translucent (Compound Eye's dome hides the units).
- [ ] Remaining CML programs (8 families) to complete the expressibility table (offline; does not affect the runs).
- [ ] Human rating kit (sample, renders, form) and an independent audit of the trace map.
- [ ] Hand review of the high-overlap prompts (E00).
- [ ] Paper draft from `PAPER_OUTLINE.md`.
