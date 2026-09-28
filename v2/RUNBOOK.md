# C2CAD-Bench v2: runbook for the live runs

Every command runs from `v2/` unless stated otherwise. Nothing here spends money until step 3. Every paid response is written to disk before it is scored. A run can be stopped at any time and resumed with the same command.

## 0. One-time setup
```bash
pip install -r requirements.txt
```
- **CAD environment** (needed by the `cadquery` arm and the kernel checks) is already built in `v2/.venv-cad`: Python 3.12, CadQuery 2.8.0. To rebuild it:
  ```bash
  uv venv .venv-cad --python 3.12 && uv pip install --python .venv-cad/bin/python cadquery==2.8.0
  ```
- **Keys.** Copy `.env.example` to `.env` (git-ignored) and fill in only the providers you will run.
- **Offline check.** It needs no keys, takes about a minute, and runs about 1,100 tests:
  ```bash
  cd .. && python -m pytest -q v2/tests && cd v2
  ```

## 1. Check the model registry (`config/models.json`)
Each profile fixes the model id, the documented output cap, the reasoning settings, whether the API accepts a temperature, the schema support and the price. They were checked on 2026-09-28 against the providers' pages; the source of each is recorded in the entry.

**Entries with `"verified": false` must be confirmed on the provider's page before use:**
- `deepseek-v4-pro`: id, output cap, how thinking mode is switched on. The official docs refused the connection when checked; the `deepseek-chat` and `deepseek-reasoner` aliases were retired on 2026-07-24.
- `kimi-k2.6`: output cap and base URL.
- `gpt-oss-120b` and `qwen3.8-27b` (OpenRouter): price and output cap.

**Decisions that change what models receive.** These have to be settled before the first paid request; changing them later means re-running:
- sign-off on the prompt patches marked (P) in `LAB_NOTEBOOK.md` (Axle Bearing, Honeycomb) and on the shared output contract;
- the reasoning setting per profile (the defaults are `high` effort or level on the frontier models, and the provider default for the continuity models);
- the roster itself.

## 2. Dry run: requests and estimated cost (no keys needed)
```bash
python -m c2cad.runner.run --profile gpt-6-sol --arms json,neutral,tool,v1,mates,schema,cadquery,probe,repair_generic,repair_verifier --k 3 --run main --dry-run
```
The estimates below come from the dry runs with the registry prices. Hidden reasoning is assumed at `est_reasoning_tokens` per request, which is only a rough guess. **Treat the estimates as ±2×.** The real spend is metered from the providers' own usage fields.

| profile | tier | main (10 arms, k=3) | sweep (json/tool/mates, k=1) | held-out (json/tool/mates, k=3) | total |
|---|---|---|---|---|---|
| claude-opus-5 | core | 2,082 req, ~$673 | 363, ~$111 | 432, ~$119 | ~$903 |
| claude-sonnet-5 | core | 2,082, ~$269 | 363, ~$44 | 432, ~$48 | ~$361 |
| gpt-6-sol | core | 2,082, ~$269 | 363, ~$44 | 432, ~$48 | ~$361 |
| gemini-3.1-pro | core | 2,082, ~$321 | 363, ~$53 | 432, ~$57 | ~$431 |
| gemini-3.8-flash | core | 2,082, ~$85 | 363, ~$14 | 432, ~$15 | ~$114 |
| deepseek-v4-pro | open | 2,082, ~$76 | 363, ~$12 | 432, ~$12 | ~$100 |
| kimi-k2.6 | open | 2,082, ~$58 | 363, ~$9 | 432, ~$9 | ~$76 |
| gpt-oss-120b | open | 2,082, ~$9 | 363, ~$1.5 | 432, ~$1.5 | ~$12 |
| qwen3.8-27b | open | 2,082, ~$15 | 363, ~$2.4 | 432, ~$2.4 | ~$20 |
| gpt-5.4 | continuity (json, tool, schema, v1 only) | 882, ~$73 | – | – | ~$73 |
| claude-opus-4-6 | continuity (json, tool, v1; no schema support) | 657, ~$87 | – | – | ~$87 |

**Totals.**
- The 9 core and open profiles come to about $2.4k.
- The continuity pair adds about $160.
- The optional profiles are claude-opus-5-5 (~$870), claude-fable-5-1 and gpt-6-astra (~$2.1k each).

**Infeasible cases.** On the main split, the json arm cannot fit the 701- and 805-part cases for the 64K-cap Gemini models (2 cases) or the 32K-cap models (3 cases). These are recorded as `infeasible` and excluded, never counted as failures.

**Cheaper variants.**
- `--k 2` cuts about a third of the main cost.
- `--families` restricts a pilot.
- Dropping `neutral` and `v1` for the open models keeps H1, H2, H5, H6 and H7 intact.

## 3. Smoke test per profile (about 20 requests each; the gate before any full run)
```bash
python -m c2cad.runner.run --profile gpt-6-sol --arms json,neutral,tool,v1,mates,schema,cadquery,probe,repair_generic,repair_verifier --smoke --run smoke_gpt-6-sol
```
The smoke run sends each arm the smallest case and the largest case that fits. For every response it checks five things:
- the request succeeded and was not truncated;
- the finish reason, usage and returned model version were recorded;
- the answer parsed or the program ran.

It must end with `SMOKE PASSED`. Typical fixes:

| symptom | fix in `config/models.json` |
|---|---|
| HTTP 400 mentioning temperature | `"send_temperature": false` |
| HTTP 400 on the schema arm | `"schema_mode": "json_object"`, or `"none"` (the arm is then skipped), or for Google `"google_schema_field": "responseSchema"` |
| HTTP 404 / unknown model | the model id |
| `truncated` on the large case | raise `reasoning_headroom`, or lower the reasoning effort |
| thinking parameter rejected | move or remove it in `extra_body`, `thinking` or `reasoning_effort` |

Registry changes after a smoke test are fine. Change the registry again after the main run has started only if you intend to re-run that model.

## 4. Full runs (one terminal per provider; `--max-usd` caps the listed profiles' spend in that run directory)
Run all models within the same short window, because provider model versions drift. The returned version and the time are recorded per request.

```bash
# main split: 8 single-turn arms, k = 3
python -m c2cad.runner.run --profile gpt-6-sol --arms json,neutral,tool,v1,mates,schema,cadquery,probe --k 3 --run main --workers 6 --max-usd 400

# multi-turn repair (needs the json answers above; up to 2 feedback rounds on the non-exact sample-0 answers)
python -m c2cad.runner.run --profile gpt-6-sol --arms repair_generic,repair_verifier --run main --workers 6 --max-usd 450

# scale sweep and held-out split
python -m c2cad.runner.run --profile gpt-6-sol --split sweep --arms json,tool,mates --k 1 --run sweep --workers 6 --max-usd 100
python -m c2cad.runner.run --profile gpt-6-sol --split heldout --arms json,tool,mates --k 3 --run heldout --workers 6 --max-usd 100

# continuity with v1 (models that were in the rejected paper)
python -m c2cad.runner.run --profile gpt-5.4 --profile claude-opus-4-6 --arms json,tool,schema,v1 --k 3 --run main --workers 6 --max-usd 200
```
- **Several profiles** can share a run directory, and `--profile` can be repeated. Several processes can write to the same run directory safely, because every append is file-locked. Keep one run directory per split (`main`, `sweep`, `heldout`); the analysis reads them all.
- **Budget.** `--max-usd` counts only the listed profiles' recorded cost in that directory, so the repair step's cap includes what the main step already spent for that profile.
- **Resuming.** Re-running the same command continues where it stopped. Requests that failed with an API error are retried, and completed ones are never re-bought.
- **Budget stop.** If `--max-usd` stops a run, raise it and re-run.
- **Rate limits.** 429 responses are retried with backoff. If many requests end as `api_error`, lower `--workers` and re-run.

## 5. Analysis (no API calls)
```bash
python -m analysis.make_all
```
- **Inputs.** It reads every run in `runs/` except `selftest*` and `smoke*`.
- **Outputs**, in `reports/results/analysis/`:
  - `a01`–`a05` reports (`.md` and `.json`);
  - the figures;
  - `a05_per_case_metrics.csv`;
  - `paper_numbers.tex`, which holds a LaTeX macro for every number the paper quotes.
- **Plan.** The estimands, hypotheses and decision rules are fixed in `reports/ANALYSIS_PLAN.md`, committed before any live run.

If the scorer changes after the runs, re-score every recorded response without new requests, then re-run the analysis:
```bash
python -m c2cad.runner.rescore --run main && python -m c2cad.runner.rescore --run sweep && python -m c2cad.runner.rescore --run heldout
```

## 6. What gets recorded
| file | content |
|---|---|
| `runs/<run>/responses.jsonl` | per request: profile, model id, arm, split, case, sample, round, the settings actually sent (including whether the temperature was sent), the returned model version, finish reason, usage, cost, latency, attempts, the schema enforcement used, the repair feedback text and the raw answer |
| `runs/<run>/scores.jsonl` | per response: status, parse, program and CAD statistics, Coverage, Geometry, Geometry-equivalent, Semantic (clause, kind, sentence and id-binding variants), exact, pass rate per constraint kind, named-part (probe) metrics, top failures |
| `runs/<run>/manifest.json` | per invocation: git commit, code hash, case-file hash, registry entries, argv, time |

Real run directories are committed; `selftest*` ones are git-ignored.

## 7. Arms (what each isolates)
| arm | the model gets | isolates |
|---|---|---|
| json | the v2 prompt and the output contract | baseline: writes every coordinate |
| neutral | the same, with object and domain nouns replaced | recall of known objects |
| tool | the same, but writes a Python program that prints the JSON | arithmetic |
| v1 | the released v1 prompt (leaked formulas and coordinates) and the contract | copying (scaffolding sensitivity) |
| mates | the prompt and the CML language (parts, mates, patterns); a solver places the parts | pose and pattern derivation |
| schema | json with the provider's constrained decoding | the output-format channel |
| cadquery | the prompt; the model writes CadQuery, which is executed in OpenCascade and the primitives recovered | code versus coordinates; "builds but wrong" |
| probe | the prompt; the model returns only three named parts | the cost of producing the whole assembly |
| repair_generic / repair_verifier | multi-turn: "check and fix", or the reference-free verifier's report (violated prompt sentences, missing ids, part count) | whether constraint feedback repairs answers |
