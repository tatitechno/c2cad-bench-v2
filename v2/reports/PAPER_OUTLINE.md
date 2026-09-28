# Working outline for the new paper

**Working title.** "Where Do LLMs Fail at Building 3D Assemblies from Text? Controlled Attribution on Exactly Specified CAD Tasks"

**Thesis.** LLM failures in text-to-CAD assembly can be attributed to specific sub-skills: copying, recall, arithmetic, output format, serialisation, and pose and pattern derivation. We measure each with a matched control on exactly specified tasks. The results show which interface helps (coordinates, code, CAD-style mates, constrained decoding, verifier feedback) and where CAD vocabularies stop helping. That gives an evidence-based way forward for LLM CAD assistants.

Each section names the evidence behind it and the file that produces it.
- **Status:** ✅ done offline; ⏳ needs live runs (all tooling built and tested on mock runs); 🧑 needs people.
- **Pre-registration.** The hypotheses and statistics are fixed in `ANALYSIS_PLAN.md`, committed before any paid request.
- **Reviewer commitments** are tracked in `REBUTTAL_TRACKER.md`.

## 1. Introduction
- **The problem.** Text-to-CAD needs construction at the level of coordinates.
  - Current evaluations check whether code executes or whether shapes look similar (Chamfer, IoU, VLM judges). Neither can say *why* an assembly is wrong.
  - Build success is not correctness. The kernel-build rate of json answers is set against their exactness (a05) and the cadquery arm's "builds but wrong" rate (a02, H6). ⏳
- **Contributions** (see `PRIOR_ART.md` for what is *not* claimed):
  1. matched-control attribution;
  2. clause-traced, reference-free constraint validity;
  3. a metric validated before use;
  4. held-out and scale generalization;
  5. an expressibility map of CAD-style interfaces.
- **Scope.** Coordinate-level layout of rigid assemblies of primitives. It is not feature-based CAD.

## 2. Related work
Five strands:
- **CAD code benchmarks:** Text2CAD-Bench, BenchCAD, CADPrompt, P3D-Bench.
- **LLM plus solver or mates:** AIDL, AssemCAD, Embodied CAD, Holodeck.
- **Coordinate-level 3D generation:** LayoutGPT, SceneCraft, LLaMA-Mesh.
- **Spatial-reasoning QA** (perception, not generation).
- **Template-generated benchmarks**, used against contamination.

Read the full texts of P3D-Bench, AssemCAD, AIDL and ExpConCAD before citing them.

## 3. Benchmark
- **Task.** 25 families and 75 cases. Prompts carry relations and base values only. There is one output contract. ✅
- **Prompt audit (E00).** ✅
  - The released v1 regexes flag 60/75 v1 prompts and 0/75 v2 prompts. The same holds for the task text of every arm, and the audit is a CI gate.
  - Verbatim coordinate overlap falls from 22.7% to 13.2%.
  - Prompt sufficiency ⏳🧑: the a05 list of never-solved cases, plus a human check.
- **Reference corrections**, each documented and with the evidence behind it. ✅
- **Splits:** main (75), scale sweep (121), held-out (48; the memorized answer is rejected). ✅
- **Figure:** the atlas contact sheets, and the three levels of one family with their prompts. ✅

## 4. Measurement and its validity
- **Scorer:** Coverage, Geometry (optimal assignment with a pose gate) and Semantic (24,579 clause-traced constraints). ✅
- **Validity evidence:**
  - references score 100 against themselves, under permutation and under renumbering (tests) ✅;
  - the CAD-kernel round trip: 5,042/5,042 primitives build and are recovered, and the cadquery path is exact on 75/75 (E08) ✅;
  - responses to controlled corruptions, including decomposition and rescale (E02) ✅;
  - sensitivity across 20 scorer variants: the top 3 never changes and ρ ≥ 0.94 (E04) ✅;
  - convergent validity against IoU, Chamfer and F-score, on the released outputs (E05) ✅ and on live outputs, with the OBB orientation error and published per-case IoU (a05) ⏳;
  - an independently written matcher reproduces the model ranking exactly (E07) ✅;
  - agreement with human engineers 🧑.
- **Interpretation key:** the E02 table (oracle, displacement, beam → box, random positions, count only). ✅
- **Lesson from v1** (short, appendix): the v1 validators contradicted their own references in 18/25 families (E01), and the v1 released scores are not reproducible. ✅

## 5. Attribution experiment (core) ⏳
- **Arms** (`RUNBOOK.md` §7): json, neutral, tool, v1, mates, schema, cadquery, probe, repair_generic and repair_verifier.
  - Models: frontier and open-weight profiles from `config/models.json`; k = 3 for the single-turn arms.
  - Both the first attempt and best-of-k are reported.
- **Measures (a02).**
  - Effect of each arm against json, with a family-clustered bootstrap and Holm-corrected sign-flip tests.
  - Broken down per family and per constraint kind (anchor, mate, pattern, orientation, dimension, topology).
- **Hypotheses stated before running** (ANALYSIS_PLAN §4):

| # | contrast | reading if supported |
|---|---|---|
| H1 | tool > json and mates > json | computing and writing numbers is the bottleneck |
| H2 | mates > tool (gated vs ungated) | a relational interface helps beyond arithmetic |
| H3 | neutral ≈ json (within ±5 pp) | recall is not what drives scores |
| H4 | v1 > json (per family) | how much the released prompts helped |
| H5 | schema ≈ json | the output-format channel is not the bottleneck |
| H6 | cadquery vs json; the "builds but wrong" rate | code versus coordinates, measured on model-written code |
| H7 | probe > the same parts in full answers | the cost of producing the whole assembly |
| H8 | repair_verifier > repair_generic, against the oracle channel bounds (a04) | constraint feedback as a way forward |

- **Mates** are reported separately for the 16 gated and 9 ungated families, with the invalid-program rate.
- **Models:** tiers, not ranks, plus the variance decomposition (a01).

## 6. Generalization and scale ⏳
- **Held-out minus main,** per model × arm, with a memorization index: the share of held-out answers closer to the default answer (a03).
- **Breaking size:** the part count at which P(exact) falls below 50%, per model × arm. Infeasible and truncated responses are censored (a03).

## 7. Where CAD-style interfaces stop helping, and what does help
- **Expressibility map.** ✅
  - 16/25 families are exact with generic mates and patterns. Of these, 9 need only prompt numbers, 6 also need counts or fractions, and 1 needs a domain convention.
  - The remaining families need growth laws. Programs for 8 of these families are still to write.
- **Combined with section 5**, this tells designers of CAD copilots four things (⏳):
  - where mates suffice;
  - where code or tools are needed;
  - whether constrained decoding matters;
  - whether a reference-free verifier in the loop repairs what one pass gets wrong.

## 8. Limitations
- Primitive assemblies only, with no B-Rep or feature trees.
- The trace map and the builders have one author.
- Role topology comes from the reference in 3 families.
- Axle Bearing's three levels all have 5 parts (a parameter variant, not a scale level).
- The v2 prompts prescribe the decomposition (E02 measures the cost of alternative decompositions).
- API drift. Returned versions and dates are recorded.

## Appendix material ready now
- `reports/LAB_NOTEBOOK.md`, `REBUTTAL_TRACKER.md`, `ANALYSIS_PLAN.md`
- `results/e00..e08`
- `atlas/`
- `PRIOR_ART.md`
- 1,097 tests
- after the runs: `results/analysis/` (a01–a06, the figures, `paper_numbers.tex`)
