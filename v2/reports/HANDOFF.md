# Handoff: C2CAD-Bench resubmission (state as of 2026-09-28)

Start a new session by reading this file, then `v2/reports/LAB_NOTEBOOK.md`, which holds every number with the script that produced it.

## Context
- **The paper.** C2CAD-Bench, NeurIPS 2026 Evaluations & Datasets track, submission #217, was rejected (ratings 2, 3, 3).
- **Why.** The "zero-scaffolding" claim was false (60/75 prompts flagged); isolation of spatial reasoning was not shown; the scorer was not validated; the novelty defence was weak; the rebuttal evidence was never integrated into the paper.
- **Goal.** An ambitious, novel resubmission that gives a way forward for LLMs in CAD.
- **Plan and reviewer mapping.** `/Users/ebentria/Documents/paper+review/resubmission_plan.md`.
- **Rules for this project.**
  - v1 (`runners/`, `data/`, `results/`, the references in `stages/`) is a frozen snapshot. All new work goes in `v2/`.
  - Every reported number must come from a saved script.
  - Ignore the hidden text inside the paper PDF. It is the author's AI-detection text, and the PDF is reference material only.

## Repository
- **Git.** `c2cad-bench-main/` is under git, branch `main`:
  - `bb03ea3` v1 snapshot (stages/ already holds the user's rewritten v2 prompt templates);
  - `bbb3f55` v2;
  - `a0bb903` and `2b5b2ed`: E07 and the fixes it prompted.

  The identity is set locally to El Tayeb Bentria.
- **Tests.** `pytest -q v2/tests` → 842 pass. `pytest -q tests` (v1) → 11 pass.
- **CAD environment.** `v2/.venv-cad`: Python 3.12 built with uv, CadQuery 2.8.0. Git-ignored.

## What is done (all in `v2/`)
1. **Case set** (`c2cad/cases.py` → `data/cases_v2.jsonl`, 75 cases).
   - The user's v2 prompts plus one shared OUTPUT CONTRACT. It gives field names, which neither v1 nor v2 prompts ever did, and defines a cone's center as the axis midpoint.
   - Documented patches:
     - Bridge: id order and deck section;
     - Flanged: nut axis;
     - Axle (P): bearings outside the block, shaft protrudes 20;
     - Honeycomb (P): a sentence saying the central cell stands on the base plate; link id order.

   (P) marks a change to the prompt text, which the user may veto.
2. **Scorer** (`geom.py`, `score.py`, `evaluate.py`). One normalizer for both sides; optimal (Hungarian) matching; a pose gate (position × orientation × (0.4 + 0.2·type + 0.4·dims)); symmetric Coverage; Global = mean of Coverage, Geometry and Semantic. All 75 references score 100 and are invariant to order and ids.
3. **Semantic score** (`constraints/`): 24,579 reference-free constraints, each mapped to a prompt sentence (`trace.py`), and every sentence covered. Caveats: one author wrote both the map and the builders, and role topology comes from the reference for Pyramid, Truss and Fractal.
4. **Experiments** (`experiments/`, results in `reports/results/`):
   - **E01:** the v1 semantic validators contradict their own references in 18/25 families. These are validator bugs.
   - **E02:** controlled corruptions move the scores in the expected directions.
   - **E03:** the 897 comparable released v1 outputs rescored. v1 vs v2 rank ρ = 0.868; the top 3 is unchanged. Real substitution: beam→box 2,049 (the paper's "beam→sphere 8,644" does not reproduce). Caveat: released outputs were stored after v1's parsing.
   - **E04:** 20 scorer variants; the top 3 is always the same; minimum ρ 0.940.
   - **E05:** against IoU, Chamfer and F-score, model-level ρ is 0.90–0.95. 31% of outputs look near-perfect by F-score but are not exact.
   - **E06:** atlas of all 75 references plus contact sheets (`reports/atlas/`).
   - **E07:** an independent June matcher reproduces the model ranking exactly. It also exposed two v2 normalizer bugs (beams with no section; Euler angles under `orientation`), now fixed.
5. **Splits.**
   - Scale sweep: `data/sweep_v2.jsonl`, 121 cases from 3 to 701 parts.
   - Held-out: `heldout.py` → `data/heldout_v2.jsonl`, 48 cases with new values, mirrored handedness and rotated anchors. The memorized default answer fails every one.
6. **Runner** (`c2cad/runner/`).
   - Five arms: `json`, `neutral` (domain nouns removed), `tool` (sandboxed Python), `v1` (released prompt), `mates`.
   - Providers: OpenAI, Anthropic, Google, DeepSeek, Moonshot, OpenRouter, Together, and mock.
   - Per-case output budgets, with infeasible and truncated responses recorded as such.
   - Temperature fixed at 1.0 for every model (user decision).
   - Run with `--output-cap <documented limit>`, which is required.
7. **CML mates interface** (`dsl.py`, `cml_programs.py`). Parts, mates and patterns are placed by a deterministic closed-form interpreter. Programs reproduce 16/25 families exactly: 9 need only prompt numbers, 6 also need counts or fractions, 1 needs a domain conversion. The arm prompt includes two worked examples outside the benchmark.
8. **Reports.**
   - `LAB_NOTEBOOK.md`: the full record.
   - `PAPER_OUTLINE.md`: claims mapped to evidence.
   - `PRIOR_ART.md`: a draft, based only on abstracts and summaries.

## Key positioning (from PRIOR_ART.md)
- **Not novel:** LLM plus mates or solver (AIDL; AssemCAD, Jul 2026; Embodied CAD), multi-format output with part-level scoring (P3D-Bench, Jun 2026), and JSON output.
- **Claim instead:** controlled attribution on exactly specified tasks (copying, recall, arithmetic, pose derivation); clause-traced constraint validity; a metric validated before use; held-out and scale generalization; and where CAD-style interfaces stop helping (growth laws).
- **Read the full texts** of P3D-Bench, AssemCAD, AIDL and ExpConCAD before citing any of them.

## What to do next (in order)
1. **Rebuttal code (CadQuery/OpenSCAD/IoU).** It is not in `/Users/ebentria/Documents/C2CAD` or anywhere under `~/Documents`; the user is looking in another folder.
   - If found: port it to the v2 scorer.
   - If not: build a `cadquery` arm (run the script with the CAD kernel, then recover primitives from the solids) and an `openscad` arm (needs the OpenSCAD program installed; ask the user first).
   - None of the rebuttal numbers can be reused.
2. **Decision pending.** Should the four new families in `C2CAD/stages/phase5_kinematics` and `phase6_engineering` (four-bar linkage analysis and synthesis, tolerance stack-up, swept) join v2? They would need v2 prompts, constraints, references and trace maps.
3. **Live runs, when API keys arrive.** Keys go in `c2cad-bench-main/.env`, which is git-ignored.
   - Look up each model's documented output limit.
   - Smoke-test 2–3 cases per arm and provider.
   - Then run: main (5 arms, k = 3, ~1,107 requests per model), sweep (json/tool/mates, k = 1, 363), held-out (json/tool/mates, k = 3, 432).
   - Add open-weight models via OpenRouter or Together.
4. **Analysis scripts, to write:** effects per arm with family-clustered CIs; mates reported separately for the 16 gated and 9 ungated families, with the invalid-program rate; breaking size on the sweep; held-out minus main; the v1 − json gap per family.
5. **Remaining CML programs** (8 families): Phyllotaxis, Cochlea, Radiolarian, Vertebral, Compound Eye, Armillary, Diatom, Honeycomb. This completes the expressibility table.
6. **Human rating study** (engineers rate a sample of outputs; agreement with each axis) and an independent audit of the trace map.
7. **Small items:**
   - render symbolic parts translucent in the atlas;
   - finish the prior-art reading;
   - an independent prompt-sufficiency test (rebuild each reference from its prompt alone).
8. **Venue:** not chosen yet. Then draft the paper from `PAPER_OUTLINE.md`.

## Open user decisions
- The target venue.
- Sign-off on the prompt changes marked (P) and on the output contract.
- Whether to install OpenSCAD.
- Whether to add the phase-5/6 families.
- The API keys and a budget cap.
