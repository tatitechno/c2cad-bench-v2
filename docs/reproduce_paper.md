# Reproducing the Artifact

## Fresh Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with:

```powershell
.venv\Scripts\activate
```

## Offline Checks

These commands do not call model APIs:

```bash
python runners/check_artifact.py
python scripts/validate_croissant.py
python scripts/audit_prompts.py
python scripts/reproduce_tables.py
pytest -q
```

Expected artifact counts:

```text
75 golden test cases
13 models
975 model-case results
```

## Regenerating Lightweight Tables

```bash
python scripts/reproduce_tables.py
```

This writes `results/reproduced_tables.md` from `data/model_summary.csv` and `data/scores.csv`.

## Live Model Re-Evaluation

Live re-evaluation requires provider API keys in `.env`. Copy `.env.example` and fill only the providers you intend to run.

```bash
cp .env.example .env
python runners/run_unified.py --all --model gpt-4.1
```

Commercial APIs can change. Always report the commit SHA, model IDs, provider date, retry policy, temperature, timeout, and token cap.
