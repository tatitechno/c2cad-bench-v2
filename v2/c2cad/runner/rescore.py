"""Re-score every recorded response offline (no API calls), e.g. after a scorer fix.

  python -m c2cad.runner.rescore --run main [--workers 8]

Reads v2/runs/<run>/responses.jsonl, keeps the last record per (model, arm, split, case, sample, round), re-runs
post-processing (sandboxed programs included) and scoring, and writes a fresh scores.jsonl. The previous file is
kept as scores.<UTC time>.bak.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from c2cad.runner.run import SPLIT_FILES, Runner, postprocess, score_record  # noqa: E402

_CASES: dict = {}


def _cases():
    if not _CASES:
        for split, f in SPLIT_FILES.items():
            for line in open(ROOT / "data" / f):
                c = json.loads(line)
                c.setdefault("split", split)
                _CASES[(split, c["case_id"])] = c
    return _CASES


def _one(rec):
    case = _cases()[(rec.get("split", "main"), rec["case_id"])]
    post = {"value": None, "parse_status": "fail", "exec_status": "", "exec_error": "", "cad_stats": None}
    if rec.get("ok"):
        post = postprocess(rec["arm"], case, rec["text"])
    rec.setdefault("model_spec", rec["model"])
    rec.setdefault("round", 0)
    rec.setdefault("split", "main")
    return score_record(rec, case, post)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    run_dir = ROOT / "runs" / a.run
    latest = {}
    for line in open(run_dir / "responses.jsonl"):
        r = json.loads(line)
        latest[Runner.key(r)] = r
    recs = list(latest.values())
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        scores = list(ex.map(_one, recs, chunksize=4))
    sp = run_dir / "scores.jsonl"
    if sp.exists():
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sp.rename(run_dir / f"scores.{stamp}.bak.jsonl")
    with open(sp, "w") as fh:
        for s in scores:
            fh.write(json.dumps(s, default=float) + "\n")
    print(f"rescored {len(scores)} responses -> {sp}")


if __name__ == "__main__":
    main()
