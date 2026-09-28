# C2CAD-Bench v2: analysis plan (fixed before any live run)

This plan is committed before the first paid request; the git history dates it. Every quantity below is computed by `v2/analysis/` (entry point `python -m analysis.make_all`, run from `v2/`), which was tested end to end on mock runs before the live runs. Deviations after data collection must be listed in the paper as such.

## 1. Data and units
- **Splits.**
  - main: 75 cases, 25 families × 3 scale levels;
  - sweep: 121 cases, 19 families, 3–701 parts;
  - held-out: 48 cases, 8 families × 2 variants × 3 levels.
- **Unit of analysis:** a case. **Cluster:** a family. The 3 levels of a family share a template and are not independent, so every confidence interval resamples families (cluster bootstrap, 2,000 draws, percentile 95% intervals).
- **Record kept per key.** The last score per (model profile, arm, split, case, sample, round).
- **Model identity.**
  - A model is a registry profile: model id plus reasoning settings.
  - The returned model versions and dates are reported per profile.
  - Temperature is 1.0 wherever the API accepts it. Where the API rejects sampling parameters, the record states that temperature was not sent.

## 2. Response statuses
| status | meaning | primary analysis | reported |
|---|---|---|---|
| complete | normal finish | scored | – |
| truncated | output cap reached (reasoning + answer) | scored as returned (usually a failure) | rate per model × arm; sensitivity analysis excluding them |
| refusal | provider safety stop | scored as returned | count |
| infeasible | the answer alone exceeds the model's documented output cap | **excluded** (censored), from every arm of that model × case pairing | count |
| api_error | the request failed after 6 attempts | **excluded** (re-requested until none remain) | count |

Parse failures, program errors (tool, cadquery), invalid CML programs (mates) and harness errors score 0. Each has its own rate.

## 3. Outcomes
- **Primary: exact.** Every stated constraint is satisfied and the part count is exact (binary).
  - **First attempt:** sample 0.
  - **Expected single attempt:** the mean over the k samples.
  - **pass@k:** any of the k samples is exact.
- **Secondary:**
  - Global_v2 = mean(Coverage, Geometry, Semantic), and each axis separately;
  - Geometry-equivalent, which is the cross-format geometry;
  - pass rates by constraint kind (anchor, mate, pattern, orientation, dimension, topology).
- **Probe arm:** part-exact and pair score of the three named parts. They are compared, paired by (model, case, part), with the same parts inside that model's full json answers, under both id binding and assignment binding.

## 4. Attribution (main split). Hypotheses stated before running
Effects are arm − json, per model and pooled over models. The rule:
1. Compute the per-case difference of the expected single-attempt exact rate.
2. Average over cases.
3. Take the CI from the family-cluster bootstrap (the same family draw for every model when pooling).

The same is reported for Global_v2, per family, and per constraint kind.

A verdict is "supported" when the 95% CI excludes 0 in the stated direction. "Equivalent" requires the CI to lie inside ±5 percentage points. Verdicts are three-valued (supported / contradicted / inconclusive); see amendment (a) at the end.

| # | Contrast | Reading if supported |
|---|---|---|
| H1 | tool − json > 0 and mates − json > 0 | computing numbers and writing them out is a bottleneck |
| H2 | mates − tool > 0 (reported separately for the 16 gated and 9 ungated families) | a relational interface helps beyond arithmetic |
| H3 | neutral − json equivalent to 0 | recall of named objects is not what drives scores |
| H4 | v1 − json > 0, per family | how much the released (leaky) prompts helped: scaffolding sensitivity |
| H5 | schema − json equivalent to 0 (strict_schema responses only) | the output-format channel is not the bottleneck |
| H6 | cadquery − json, plus the conditional rate: among cadquery programs that build every part as a recognised solid, the share that is not exact and the share with Geometry-equivalent < 70 | build success does not imply spatial correctness (measured on model-written code) |
| H7 | probe part-exact − same parts in the full json answer > 0 | the cost of producing the whole assembly (length/serialisation) |
| H8 | repair_verifier − repair_generic at the last round > 0 (chains seeded by the same non-exact json answers; generic runs all rounds, the verifier stops only when its own report is clean) | constraint-level feedback helps beyond a generic request to check. Also compared with oracle upper bounds: fix types only, positions only, add missing parts |

**Mates-specific reporting.** The invalid-program rate, and gated families (hand-written CML programs reproduce the reference) versus ungated families.

**Multiplicity.** The eight hypotheses are reported with their CIs. The per-model contrasts are descriptive. Holm-adjusted p-values (family-level sign-flip permutation, 10,000 draws) are given for the pooled contrasts.

## 5. Models: tiers, not ranks
- **Pairwise tests.** Json arm, expected single-attempt exact rate and Global_v2. Every pair of models is compared with a paired, family-level sign-flip permutation test, Holm-corrected across pairs.
- **Tiers.** Models are sorted by mean. A tier starts at its best model, and later models join it while they are not significantly worse than that best model.
- **Also reported:**
  - the family-cluster bootstrap distribution of ranks;
  - the variance decomposition of case-level Global_v2 into model, family, model × family and residual (sums of squares).

## 6. Generalization and scale
- **Held-out.** The held-out exact rate minus the main exact rate on the same 8 families and levels, per model × arm, with a family-cluster bootstrap.
  - Memorization index: the share of held-out answers that agree better with the default (main-split) reference than with the held-out reference (Global_v2 against both).
- **Breaking size (sweep).** Per model × arm, a logistic model with log2(part count) as the slope and L2-penalised family offsets, fitted on exact.
  - The breaking size n50 is the part count at which the family-averaged P(exact) = 0.5. Its CI comes from a family bootstrap.
  - Infeasible and truncated responses are censored (excluded) and counted, so the curve does not measure the output cap.
  - Per family, the largest scale solved is also reported.

## 7. Measurement validity on live outputs
- **Convergent validity.** Voxel IoU (96³), Chamfer distance / diagonal, F-score at 2% of the diagonal, and orientation error over matched parts, each against every axis (case-level Spearman, model-level Spearman).
- **Disagreement quadrants, with examples:**
  - F@2% ≥ 0.9 but not exact;
  - IoU ≥ 0.8 but Sem < 60;
  - IoU < 0.3 but Geom ≥ 80.
- **Per-case IoU** is published as a CSV.
- **Kernel build of json answers.** The share that build as valid solids, set against their Geometry and exact rates. This is the rebuttal's "builds but misplaced" figure, now measured on v2 outputs.
- **Failure taxonomy.** api / parse / program / count / type substitution / placement, plus a co-located type-confusion matrix.
- **Prompt sufficiency (review list).**
  - Cases that no model and arm solved exactly are flagged for human review.
  - Cases solved exactly by at least one tool-arm answer count as evidence that the prompt determines the reference.

## 8. v1 → v2 continuity
For the profiles that match released v1 models (gpt-5.4, gemini-3.1-pro, claude-opus-4-6): the v2 json exact rate against the E03 re-scored released v1 outputs of the same model on the same cases (released records were stored after v1 parsing; stated as a caveat).

## 9. What is not claimed
- No leaderboard.
- No claim of "isolating spatial reasoning". Claims are restricted to the attribution contrasts above.
- Primitive assemblies only.

## Amendments (all made before any live run; the git history dates them)
- **2026-09-28 (a) Verdicts are three-valued.**
  - For a directional hypothesis: *supported* when the 95% CI excludes 0 in the stated direction, *contradicted* when it excludes 0 in the opposite direction, and *inconclusive* otherwise.
  - For an equivalence hypothesis (H3, H5): *supported* when the CI lies inside ±5 pp, *contradicted* when it excludes 0 without lying inside the margin, and *inconclusive* otherwise. This replaces "supported / not supported", which could read an underpowered interval as evidence against a hypothesis.
- **2026-09-28 (b) Repair chains use no reference information after seeding.**
  - repair_generic always runs all R rounds, because a deployed "check and fix" loop cannot know when it is done.
  - repair_verifier stops only when its own reference-free report is clean.
  - H8 compares the two at round R. The round at which an answer first becomes exact is derived afterwards.
- **2026-09-28 (c) H7 bindings.** The primary binding is the output's own ids, where a full answer numbered from 1 is shifted to 0-based by the same rule as `evaluate.id_binding` (probe answers name their ids and are not shifted). Assignment binding is the sensitivity analysis. H7 is added to the verdict table.
