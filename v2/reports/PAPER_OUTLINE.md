# Working outline for the new paper

**Working title.** "Where Do LLMs Fail at Building 3D Assemblies from Text? Controlled Attribution on Exactly Specified CAD Tasks"

**Thesis.** LLM failures in text-to-CAD assembly can be attributed to specific sub-skills: copying, recall, arithmetic, and pose and pattern derivation. We measure each with matched controls on exactly specified tasks. The results show which interface (coordinates, code, or CAD-style mates) helps, and where CAD vocabularies stop helping. That gives an evidence-based way forward for LLM CAD assistants.

Each section below names the evidence behind it. ✅ = done offline; ⏳ = needs live runs; 🧑 = needs people.

## 1. Introduction
- The problem: text-to-CAD needs coordinate-level construction. Current evaluations check whether code executes or whether shapes look similar (Chamfer, IoU, VLM judges), which cannot say *why* an assembly is wrong.
- Contributions (see `PRIOR_ART.md` for what is *not* claimed):
  1. matched-control attribution;
  2. clause-traced, reference-free constraint validity;
  3. a metric validated before use;
  4. held-out and scale generalization;
  5. an expressibility map of CAD-style interfaces.

## 2. Related work
Five strands:
- **CAD code benchmarks:** Text2CAD-Bench, BenchCAD, CADPrompt, P3D-Bench.
- **LLM plus solver or mates:** AIDL, AssemCAD, Embodied CAD, Holodeck.
- **Coordinate-level 3D generation:** LayoutGPT, SceneCraft, LLaMA-Mesh.
- **Spatial-reasoning QA** (perception, not generation).
- **Template-generated benchmarks**, used against contamination.

## 3. Benchmark
- **Task.** 25 families and 75 cases. Prompts carry relations and base values only. There is one output contract. ✅
- **Prompt audit.**
  - Regex flags fall from 60 to 0 of 75. ✅
  - Verbatim coordinate overlap falls from 22.7% to 13.2%. ✅
  - Prompt-sufficiency test: an independent reconstruction of each reference from its prompt. ⏳ Can be done by a tool-arm run with human check, or by an independent person 🧑.
- **Reference corrections**, each documented and with the evidence behind it. ✅
- **Splits:** main (75), scale sweep (121), held-out (48; the memorized answer is rejected). ✅
- **Figure:** the atlas contact sheets, and the three levels of one family with their prompts. ✅

## 4. Measurement and its validity
- **Scorer:** Coverage, Geometry (optimal assignment with a pose gate) and Semantic (24,579 clause-traced constraints). ✅
- **Validity evidence:**
  - references score 100 against themselves, under permutation and under renumbering (378+ tests) ✅;
  - responses to controlled corruptions (E02) ✅;
  - sensitivity across 20 variants of the scorer: the top 3 never changes and ρ ≥ 0.90 (E04) ✅;
  - convergent validity against IoU, Chamfer and F-score, where model-level ρ is 0.92–0.98 (E05) ✅;
  - cases where volume metrics fail: 31% of outputs look near-perfect by F-score but are not exact, and a released record with 10 of 12 parts missing still scores IoU 0.92. Caveat: the released v1 records are post-parsing, so live v2 runs must re-establish these cases ✅/⏳;
  - agreement with human engineers 🧑.
- **Lesson from v1** (short, appendix): the v1 validators contradicted their own references in 18/25 families (E01), and the v1 released scores are not reproducible. ✅

## 5. Attribution experiment (core)
- **Arms:** json, neutral, tool, v1 (the released prompt), mates. Several frontier and open-weight models, k = 3; first-attempt and best-of-k reported. ⏳
- **Measures:** effect sizes per arm, with a family-clustered bootstrap. Each arm is compared with json per family and per constraint kind (anchor, mate, pattern, orientation, dimension, topology). ⏳
- **Expected shape of the result, a hypothesis stated before running:**
  - If tool ≈ mates > json, arithmetic and serialization are the bottleneck.
  - If mates > tool, relational interfaces help beyond arithmetic.
  - If neutral ≈ json, recall is not driving scores.
  - The v1 − json gap per family is how much the released prompts helped.
- **The mates analysis is split** into the 16 gated families and the 9 ungated ones, with the invalid-program rate reported separately. ✅ design, ⏳ data.

## 6. Generalization and scale
- Held-out minus main, per model: do scores survive new values, handedness and anchors? ⏳
- Scale sweep: the breaking size for each model and arm, i.e. the part count at which the exact rate falls below 50%. ⏳

## 7. Where CAD-style interfaces stop helping
- Expressibility map: 16/25 families are exact with generic mates and patterns. Of these, 9 need only prompt numbers, 6 also need counts or fractions, and 1 needs a domain convention. The remaining families need growth laws. ✅
- Combined with section 5, this tells designers of CAD copilots where mates suffice and where code or tools are needed. ⏳

## 8. Limitations
- Primitive assemblies only, with no B-Rep or feature trees.
- The trace map and the builders have one author.
- Role topology comes from the reference in 3 families.
- API drift.

## Appendix material ready now
- `reports/LAB_NOTEBOOK.md`
- `results/e01..e05`
- `atlas/`
- `PRIOR_ART.md`
- 840 tests
