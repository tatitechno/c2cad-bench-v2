# Prior art and positioning (draft, 2026-09-28)

**Status: from abstracts and machine summaries of the HTML pages, not from reading the full papers.** Descriptions below come from web-search snippets and automated page summaries. Confirm every statement from the full text before citing it. Specific figures such as success rates and F1 values have been removed until they are read in the papers themselves.

## Closest work

| Work | What it does | Overlap with us | What it does not do |
|---|---|---|---|
| **P3D-Bench** (arXiv 2606.11152, Jun 2026) | Parametric 3D generation by MLLMs. Tasks: text-to-3D (400), image-to-3D (400) and Assembly-3D (203 annotated assemblies from Text2CAD and Fusion 360 Gallery). Outputs: minimal JSON, OpenSCAD, CadQuery or Three.js. Metrics: Chamfer, F-score, IoU, topology, an MLLM judge for semantic and parametric QA, and part-level F1 (PartMatchF1). Reportedly ablates output format, thinking effort and multi-turn feedback. | Multi-format output including JSON; part-level scoring; assemblies; the finding that semantics is much better than geometry. | Inputs include images. References are real CAD models with no exact constraint specification, and constraint checks are done by an MLLM judge. It does not ablate copying, recall, arithmetic or pose derivation, and has no generalization split. (A format ranking is reported; read the paper before citing it.) |
| **Text2CAD-Bench** (arXiv 2605.18430, May 2026) | 600 curated text-to-CadQuery tasks at four levels; Chamfer, invalidity rate, IoU, and a VLM judge on L4. Ablates the description style (geometric vs sequence). | Graded text-to-CAD; representation ablation. | Code only; no part-level or constraint-level scoring; no attribution of failures. |
| **BenchCAD** (arXiv 2605.10865, May 2026) | Programmatic CAD with feature operations (extrude, cut, sweep, fillet, pattern). | CAD programs with patterns. | Feature-based single parts; no assembly constraints; no attribution. |
| **AssemCAD** (arXiv 2607.05123, Jul 2026) | The LLM writes an AssemblyDraft of typed ports and mates; a library applies closed-form mate transforms. It reportedly uses a Fusion-360-derived assembly set scored by success rate, assembly preservation rate and a VLM judge. | **The same idea as our mates arm**: the LLM declares mates and deterministic transforms place the parts. | A system paper. No exact references (it deliberately drops Chamfer), no matched Python-tool or coordinate arm on identical tasks, no per-constraint scoring against prompt clauses. |
| **AIDL** (Computer Graphics Forum 2025, arXiv 2502.09819) | A solver-aided hierarchical DSL that offloads spatial reasoning to a constraint solver; beats OpenSCAD on closeness to the prompt. | The same motivation as the mates arm. | Evaluated against OpenSCAD on visual closeness; no exact references or attribution design. |
| **Embodied CAD** (arXiv 2606.31252, Jun 2026) | Solver-grounded LLM agents for B-Rep assembly modeling with a skill library and solver feedback. | Solver-grounded assembly. | An agent system, not a controlled measurement. |
| **Holodeck** (CVPR 2024, arXiv 2312.09067) | GPT-4 writes spatial relational constraints and a solver lays out 3D scenes. | LLM relations with a solver. | Scene layout from asset retrieval; no exact engineering references. |
| **CADPrompt / CADCodeVerify** (ICLR 2025, arXiv 2410.05340) | 200 prompts with expert CAD code; point-cloud distance and compile rate; VLM feedback loop. | Geometric metrics on CAD code. | Code only; the geometric metrics are volume or surface based. |
| **LayoutGPT** (NeurIPS 2023), **SceneCraft** (ICML 2024), **LLaMA-Mesh** (2024) | LLMs emit layouts or meshes as text, or scene graphs, which are turned into Blender code. | Coordinate-level generation by LLMs. | No engineering constraints; no attribution. |

## What we can claim (and what we cannot)

**Not novel. Do not claim:**
- JSON or primitive output from LLMs;
- multi-format comparison, which P3D-Bench already does;
- LLM plus solver or mates, which AssemCAD, AIDL and Holodeck already do;
- part-level scoring, which P3D-Bench already does.

**Claimable, verified against the list above:**
1. **Attribution by matched controls on identical, exactly specified tasks.** The same 75 cases are run under five conditions:
   - the leaked released prompt (copying);
   - a neutral-vocabulary twin (recall);
   - a Python tool (arithmetic);
   - a mates interface (pose and pattern derivation);
   - direct coordinates.

   Each condition changes one factor. Per their abstracts, none of the works above isolate these factors; confirm this from the full texts.
2. **Reference-free, clause-traced semantic validity.** 24,579 machine-checked constraints, each mapped to a prompt sentence. References satisfy 100% of them at the main, sweep and held-out scales. This contrasts with MLLM-judge QA (P3D-Bench) and VLM judges (AssemCAD, Text2CAD-Bench).
3. **Metric validity evidence.** Reference self-consistency, invariance tests, responses to corruption, sensitivity across 20 scorer variants, and convergent and discriminant validity against Chamfer, IoU and F-score. This includes documented cases where volume metrics hide missing parts. Among the works above, only AssemCAD argues against Chamfer; none validate their metric this way.
4. **Generalization and scale.** A held-out split (new values, mirrored handedness, rotated anchors; the memorized default answer is rejected) and a 121-case scale sweep for per-model breaking size.
5. **Where CAD-style interfaces stop helping.** A gated expressibility study: 16/25 families are exactly expressible with generic mates and patterns from prompt numbers alone or with simple counts and fractions. Growth laws (Fermat spiral, taper, geodesic refinement, accumulated tilt) are not.

**Novelty sentence to use:** "Recent systems let LLMs place CAD parts through mates or solvers (AIDL, AssemCAD) and recent benchmarks compare output formats (P3D-Bench); we provide the controlled measurement those comparisons lack: on exactly specified assemblies with clause-traced constraints, we isolate how much of LLMs' 3D construction error is due to copying, recall, arithmetic, and pose derivation, and map where CAD-style interfaces stop helping."

## Still to check before submission
- The full text of P3D-Bench's Assembly-3D protocol, in case it has a text-only condition.
- AIDL's evaluation section.
- ExpConCAD (arXiv 2608.24760, "implicit spatial constraints"): the PDF did not extract; read the HTML version.
- CADWorld (arXiv 2609.16251), for its computer-use framing.
- LLM spatial-reasoning QA benchmarks to cite as "perception, not generation": SpatialEval (NeurIPS 2024), GeoGramBench, 3DSRBench, Spatial457.

## Sources
- [P3D-Bench](https://arxiv.org/html/2606.11152v1)
- [Text2CAD-Bench](https://arxiv.org/html/2605.18430v1)
- [BenchCAD](https://arxiv.org/pdf/2605.10865)
- [AssemCAD](https://arxiv.org/html/2607.05123)
- [AIDL](https://arxiv.org/abs/2502.09819)
- [Embodied CAD](https://arxiv.org/abs/2606.31252v1)
- [Holodeck](https://arxiv.org/abs/2312.09067)
- [CADPrompt / CADCodeVerify](https://arxiv.org/abs/2410.05340)
- [LayoutGPT](https://arxiv.org/abs/2305.15393)
- [SceneCraft](https://arxiv.org/pdf/2403.01248)
- [LLaMA-Mesh](https://arxiv.org/abs/2411.09595)
- [ExpConCAD](https://arxiv.org/pdf/2608.24760): the PDF did not extract; not yet read
- [CADWorld](https://arxiv.org/html/2609.16251v2)
