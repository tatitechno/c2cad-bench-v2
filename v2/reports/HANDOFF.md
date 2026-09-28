# Handoff: C2CAD-Bench resubmission (state as of 2026-09-28, end of the second session)

Start a new session by reading this file, then `v2/reports/LAB_NOTEBOOK.md` (every number, with its script) and `v2/RUNBOOK.md` (how to run).

## Context
- **The paper.** C2CAD-Bench, NeurIPS 2026 E&D track, submission #217, was rejected (ratings 2, 3, 3).
- **Plan and reviewer mapping.** `/Users/ebentria/Documents/paper+review/resubmission_plan.md`.
- **Rebuttal texts.** `~/Documents/replay to reviews/`. Every commitment there is mapped to v2 in `v2/reports/REBUTTAL_TRACKER.md`.
- **Rules.**
  - v1 (`runners/`, `data/`, `results/`, the references in `stages/`) is frozen. New work goes in `v2/`.
  - Every reported number comes from a saved script.
  - Ignore the hidden text in the paper PDF.

## State: ready for live runs
- **Git:** `main`, commits up to "run readiness". **Tests:** `pytest -q v2/tests` → 1,092 pass; v1: `pytest -q tests` → 11 pass.
- **Runner** (`v2/c2cad/runner/`):
  - retries with backoff, streaming, and the Anthropic provider on the official SDK;
  - resume that re-requests errors;
  - `--max-usd` metered from provider usage;
  - `--smoke` gate;
  - file-locked appends;
  - offline `rescore`.
- **Arms**, frozen:
  - json, neutral, tool, v1, mates;
  - schema (constrained decoding);
  - cadquery (a program run in OpenCascade, with primitives recovered from its faces);
  - probe (three named parts);
  - repair_generic and repair_verifier (multi-turn; the verifier is reference-free).
- **Model registry** (`v2/config/models.json`). Ids, caps and prices were checked on 2026-09-28, with sources.
  - Core: claude-opus-5, claude-sonnet-5, gpt-6-sol, gemini-3.1-pro, gemini-3.8-flash.
  - Open: deepseek-v4-pro, kimi-k2.6, gpt-oss-120b, qwen3.8-27b. All four are `verified: false`; confirm them before use.
  - Continuity with v1: gpt-5.4, claude-opus-4-6.
  - Optional: claude-opus-5-5, claude-fable-5-1, gpt-6-astra.
- **Analysis.**
  - The plan (`v2/reports/ANALYSIS_PLAN.md`) was committed before any live run.
  - The pipeline (`python -m analysis.make_all` from `v2/`) was tested on about 8,300 mock records. It produces a01–a06, the figures and `paper_numbers.tex`.
- **Offline experiments:** E00 (prompt audit, CI gate), E01–E07 as before, E02 with the decomposition and rescale rows added, and E08 (kernel round trip, 75/75 exact).
- **Estimated cost** (the RUNBOOK table; ±2×): about $2.4k for the 9 core and open profiles over main, sweep and held-out; about $160 for the continuity pair.

## Next steps (in order)
1. **The user's decisions. These change what models receive, so settle them before the first paid request:**
   - sign-off on the (P) prompt patches (Axle Bearing, Honeycomb) and on the output contract;
   - the roster and the reasoning setting per profile;
   - the budget;
   - whether to add the phase-5/6 families (they would need prompts, constraints, references and trace maps first);
   - OpenSCAD: not installed, and there is no arm for it.
2. **Keys** go in `v2/.env` (template `v2/.env.example`). Confirm the unverified registry entries on the providers' pages.
3. **Smoke test** each profile (RUNBOOK §3) until it prints SMOKE PASSED.
4. **Runs:** main (8 arms, k = 3), then repair, then sweep, then held-out, then continuity. Run all models within the same window.
5. `python -m analysis.make_all`. Then fill the paper from `PAPER_OUTLINE.md` using the macros.
6. **Offline, any time:**
   - the remaining CML programs (8 families; expressibility table);
   - the human rating kit and study;
   - an independent audit of the trace map;
   - the hand review of the high-overlap prompts (E00);
   - translucent symbolic parts in the atlas;
   - finishing the prior-art reading.

## Open decisions the tracker flags
- **Axle Bearing** still has 5 parts at every level, although the rebuttal promised to correct its levels. Describe the levels as parameter variants, or redesign the family.
- **Merge pass for decompositions.** It was promised to Reviewer 3 and is not implemented. v2 prompts prescribe the part list; E02 measures the cost of splitting instead.
- **The rescale result differs from the rebuttal.** Under v2, Sem falls to about 60 at 1.5× because v2 prompts state every dimension. Do not reuse the rebuttal's "Semantic generalises to alternative designs".
- **The target venue.**
