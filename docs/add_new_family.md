# Adding a New Test Family

1. Create a deterministic generator under the relevant `stages/phase*/` directory.
2. Return a natural-language prompt and a golden list of canonical primitive objects.
3. Use only the seven canonical primitive types unless the schema is intentionally extended.
4. Add the family to `ALL_TESTS` in `runners/run_unified.py`.
5. Add or update a family-specific semantic validator when structural constraints matter.
6. Run:

```bash
python runners/check_artifact.py
python scripts/audit_prompts.py
pytest -q
```

## Prompt Review

For a strict zero-scaffold split, run:

```bash
python scripts/audit_prompts.py --strict
```

If the audit flags the prompt, review whether the prompt leaks derived coordinates, shape counts, formulas, or construction steps that the model should infer.
