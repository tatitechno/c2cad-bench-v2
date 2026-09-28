# Rebuttal commitments → v2 status

**Sources.** The four OpenReview posts (`~/Documents/replay to reviews/OpenReview_Responses*.md`) and the full author response (`C2CAD_Bench_Author_Response.docx`). The AC's main complaint was that rebuttal evidence was never integrated into the paper. This table maps every commitment to the v2 artifact that carries it.

**Status key.** ✅ done offline · ⏳ built and tested, waiting for live runs · 📝 paper text · 🧑 needs people · ❗ open decision.

**Numbers.** None of the rebuttal's numbers is reused. They came from the v1 scorer, whose bugs are listed in `LAB_NOTEBOOK.md`. Every figure below is the v2 re-measurement, and each has a script.

## Reviewer 1 (c9s1)
| commitment | status | where |
|---|---|---|
| Rewrite the prompts that state totals or closed forms | ✅ | v2 prompts. E00: 60/75 → 0/75 flagged; verbatim coordinate overlap 22.7% → 13.2% (`experiments/e00_prompt_audit.py`) |
| The audit becomes a strict CI gate | ✅ | `tests/test_v2_runner.py::test_audit_gate_task_text_of_every_arm_has_no_scaffolding_flag`, which covers every arm's task text |
| Report compliant and non-compliant prompts separately | ✅ design, ⏳ data | All v2 prompts comply. The released leaky prompts are the `v1` arm, and v1 − json is reported per family (H4, `analysis/a02`) |
| Re-run the models | ⏳ | registry `config/models.json`; `RUNBOOK.md`. The v1 models still served are continuity profiles (gpt-5.4, gemini-3.1-pro, claude-opus-4-6). The deepseek-chat/reasoner aliases were retired on 2026-07-24; the Gemini 2.5 models are limited-access |
| The generator-side code-vs-JSON experiment (the model writes CadQuery) | ⏳ | `cadquery` arm: the program runs in OpenCascade and the primitives are recovered from B-Rep faces. E08: a correct program scores 100 (75/75). a02 reports the rates of "runs", "builds" and "builds but wrong" (H6) |
| The mechanical JSON → CadQuery converter | ✅ | `c2cad/cadkernel.py`, `c2cad/cadcode.py`. E08: 5,042/5,042 reference primitives build and are recovered. a05 builds every json answer in the kernel |
| Non-LLM controls: oracle, beam → box, random positions, count only | ✅ | E02 (v2 numbers: identity 100; beam → box Global 84.7; random positions 45.1; count only 36.6, the Coverage floor) |
| Couple the semantic checks to geometry | ✅ | v2 Semantic binds roles by optimal assignment. Randomised positions now keep Sem 26.2, against 75.3 under v1 (E02) |
| Sensitivity to every weight, exponent and gate | ✅ | E04: 20 variants, top-3 always the same, Spearman ρ ≥ 0.940 |
| A constrained-decoding condition | ⏳ | `schema` arm: provider-native JSON schema, with the enforcement recorded per response. H5 |
| Serialisation load ("cannot serialise 805 primitives") | ⏳ | `probe` arm: three named parts asked alone, compared with the same parts inside the full answer (H7). Breaking size on the sweep, with infeasible and truncated responses censored (a03) |
| A held-out split | ✅ data built, ⏳ runs | `data/heldout_v2.jsonl`: 48 cases, new values, mirrored handedness, rotated anchors; the memorized answer fails every case. The a03 memorization index |
| Scope: coordinate-level layout of rigid assemblies | 📝 | `PAPER_OUTLINE.md` §1, §8 |
| One documented scoring function, a regression test, no hidden calibration exponent | ✅ | Global_v2 = mean(Cov, Geom, Sem). References score 100 on every axis (tests). E03 re-scores the released outputs |
| Tiers, not ranks | ⏳ | a01: pairwise family-level sign-flip tests, Holm correction, greedy tiers, variance decomposition |
| Per-case diagnostics: parse failure, truncation and schema violation apart from geometry | ✅ design | `scores.jsonl` statuses; the a01 status columns; the a05 failure taxonomy |

## Reviewer 2 (V8E8)
| commitment | status | where |
|---|---|---|
| A visual atlas of all 75 references | ✅ | E06, `reports/atlas/` (contact sheets and per-case renders) |
| The three levels of one family shown side by side, with prompts | ✅ | the atlas Staircase panel |
| Narrow the "code-free" claim; release the converter | ✅ / 📝 | converter as above; wording in the outline |
| The primitive representation as the primary limitation | 📝 | `PAPER_OUTLINE.md` §8 |
| An interpretation key for metric values | ✅ | E02 table (v2), to sit beside the metric definition |
| Figure 5: failures marked on the renders | ⏳ | not yet built. The renderer (`c2cad/render.py`) and the failure taxonomy exist; pick the examples from a05 after the runs |
| Iterative refinement / feedback | ⏳ | the `repair_generic` and `repair_verifier` arms (multi-turn, reference-free verifier). The a04 oracle single-channel upper bounds (types, positions, missing) on the same seeds |

## Reviewer 3 (z3QJ)
| commitment | status | where |
|---|---|---|
| Sensitivity to the canonical decomposition | ✅ measured | E02: splitting every box costs 5.4 Global points; splitting every beam costs 16.2 |
| ❗ The promised merge pass (collinear beams, coplanar boxes) | open | Not implemented. v2 prompts prescribe the part list and ids, so a split part violates the specification. Decide: keep as is and state it, or add merging as a sensitivity variant |
| IoU / Chamfer / F-score comparison, with the disagreements explained | ✅ v1 outputs, ⏳ v2 | E05 on the released outputs; a05 on live outputs, with quadrants and examples |
| Per-case IoU published | ⏳ | `analysis/a05` → `a05_per_case_metrics.csv` |
| Orientation error over the oriented bounding box of every matched pair | ✅ | `metrics3d.orientation_error_obb`, reported in a05 |
| Functional equivalence / rescaled goldens | ✅ measured | E02: rescaling by 1.1, 1.25 and 1.5 gives Geom 55.9 / 20.5 / 8.9 and Sem 66.8 / 60.7 / 60.2. **This differs from the rebuttal** (Sem about 99 under v1): v2 prompts state every dimension, so a rescaled answer is wrong. Relational kinds hold (orientation 88%, topology 100%) and dimension constraints fail (37%) |
| Report axes separately for free-dimension families | ✅ | v2 has no free-dimension family: Axle is patched, and the Furniture prompt now gives every dimension. Every axis is reported separately anyway |
| Narrow the novelty claims | 📝 | `PRIOR_ART.md`, `PAPER_OUTLINE.md` |

## Area chair and further commitments in the full response
| commitment | status | where |
|---|---|---|
| Integrate the evidence into the paper | 📝 | `PAPER_OUTLINE.md`; `analysis/make_all` writes `paper_numbers.tex` macros |
| Scripts for every number, one command | ✅ | `python -m analysis.make_all`; experiments E00–E08 |
| A perturbation axis beyond scale (orientation, handedness, anchors) | ✅ partial | the held-out variants |
| ❗ Correct the Axle Bearing level definitions (5 parts at every level) | open | Still 5 parts at all three levels in v2. Axle is excluded from the sweep. Decide: describe its levels as parameter variants, or redesign the family |
| Expand the case count | ✅ | sweep (121) + held-out (48) on top of main (75) |
| First attempt and best-of-k; family-clustered bootstrap | ✅ design | a01 |
| Record model identity: version, date, settings | ✅ | every response record; `manifest.json` |
| Human agreement study | 🧑 | not built; see the open items in `HANDOFF.md` |
| An independent prompt-sufficiency check | ⏳ / 🧑 | the a05 list of never-solved cases for review, and the tool-arm solved count |
| Anonymity: the patent reference and the grant acknowledgment | 📝 | the paper source |
