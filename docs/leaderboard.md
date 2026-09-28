# Released Leaderboard

The source of record is `data/model_summary.csv`. Regenerate this markdown summary with:

```bash
python scripts/reproduce_tables.py
```

For new results, report:

- commit SHA;
- model ID and provider;
- evaluation date;
- temperature, timeout, max token cap, and retry policy;
- whether prompts, parser, normalizer, or scoring changed.
