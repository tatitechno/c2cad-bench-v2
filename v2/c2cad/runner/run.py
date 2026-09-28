"""Run models on v2 cases under one or more arms; every request and score is appended to JSONL.

  python -m c2cad.runner.run --model openai:gpt-5.4 --arms json,neutral,tool --k 3 --run pilot
  python -m c2cad.runner.run --model mock:reference --arms json,tool --run selftest

Resumable: an existing (model, arm, case, sample) record is never re-requested.
Output: v2/runs/<run>/{manifest.json, responses.jsonl, scores.jsonl}
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import platform
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from c2cad import cases as CASES  # noqa: E402
from c2cad.evaluate import evaluate  # noqa: E402
from c2cad.runner import arms as A, providers as P, sandbox  # noqa: E402
from c2cad import dsl, cml_programs  # noqa: E402

LOCK = threading.Lock()

TOKENS_PER_PART = 90          # v1 runner's measured average for one JSON part
JSON_ARMS = ("json", "neutral", "v1")
TRUNCATED = {"length", "max_tokens", "MAX_TOKENS"}


def output_budget(arm: str, n_parts: int, reasoning_headroom: int) -> int:
    """Output tokens to request: coordinate arms scale with the part count; program arms are short."""
    body = int(n_parts * TOKENS_PER_PART * 1.6) + 2048 if arm in JSON_ARMS else 12000
    return body + reasoning_headroom


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "c2cad").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def mock_response(spec: str, arm: str, case: dict, sample: int) -> P.Response:
    kind = spec.split(":", 1)[1]
    ref = copy.deepcopy(case["reference"])
    rng = np.random.default_rng(sample)
    if kind.startswith("jitter-"):
        s = float(kind.split("-")[1])
        for p in ref:
            for k in ("center", "start", "end"):
                if k in p:
                    p[k] = (np.array(p[k]) + rng.normal(0, s, 3)).tolist()
    elif kind == "beam-box":
        for i, p in enumerate(ref):
            if p["type"] == "beam":
                a, b = np.array(p["start"]), np.array(p["end"])
                ref[i] = {"id": p["id"], "type": "box", "center": ((a + b) / 2).tolist(),
                          "size": [float(np.linalg.norm(b - a)), p["width"], p["height"]]}
    elif kind == "empty":
        ref = []
    text = json.dumps(ref)
    if arm == "mates":
        fn = cml_programs.PROGRAMS.get(case["family"])
        text = json.dumps(fn(case)) if (fn and kind == "reference") else json.dumps({"steps": []})
    if arm == "tool":
        text = "```python\nimport json\nparts = " + text + "\nprint(json.dumps(parts))\n```"
    return P.Response(text, True, returned_model=spec, finish_reason="stop", usage={"mock": True})


def one(args, run_dir, model, arm, case, sample, settings):
    system, user = A.build(arm, case)
    need = output_budget(arm, case["n_parts"], args.reasoning_headroom)
    settings = P.Settings(**{**settings.__dict__, "max_tokens": min(need, args.output_cap)})
    if need > args.output_cap:          # cannot fit: record as infeasible, never as a model failure
        resp = P.Response("", False, f"infeasible: needs ~{need} output tokens, cap {args.output_cap}")
    elif model.startswith("mock:"):
        resp = mock_response(model, arm, case, sample)
    else:
        resp = P.call(model, system, user, settings)
    rec = {"model": model, "arm": arm, "case_id": case["case_id"], "family": case["family"], "level": case["level"],
           "sample": sample, "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "system_sha": A.sha(system), "user_sha": A.sha(user), "settings": settings.__dict__,
           "request_meta": resp.request_meta, "returned_model": resp.returned_model, "finish_reason": resp.finish_reason,
           "usage": resp.usage, "latency_s": round(resp.latency_s, 2), "ok": resp.ok, "error": resp.error, "text": resp.text}
    exec_status, exec_err = "", ""
    text = resp.text
    if arm == "tool" and resp.ok:
        ok, out, err = sandbox.run(sandbox.extract_code(text))
        exec_status, exec_err, text = ("ok" if ok else "error"), err, out
    value, parse_status = A.parse_json(text) if resp.ok else (None, "fail")
    if arm == "mates" and value is not None:
        try:
            value = dsl.run(value)
            exec_status = "ok"
        except (dsl.CMLError, KeyError, TypeError, ValueError, IndexError) as e:
            exec_status, exec_err, value = "invalid_program", f"{type(e).__name__}: {e}", None
    r = evaluate(case, value if value is not None else [])
    status = ("infeasible" if resp.error.startswith("infeasible") else "api_error" if not resp.ok
              else "truncated" if resp.finish_reason in TRUNCATED else "complete")
    score = {"model": model, "arm": arm, "case_id": case["case_id"], "family": case["family"], "phase": case["phase"],
             "level": case["level"], "sample": sample, "api_ok": resp.ok, "response_status": status,
             "max_tokens": settings.max_tokens, "exec_status": exec_status, "exec_error": exec_err[:300],
             "parse_status": parse_status, **{k: v for k, v in r.to_dict().items() if k not in ("top_failures",)},
             "top_failures": r.top_failures[:5]}
    with LOCK:
        with open(run_dir / "responses.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        with open(run_dir / "scores.jsonl", "a") as fh:
            fh.write(json.dumps(score, default=float) + "\n")
    return score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--arms", default="json")
    ap.add_argument("--families", default="all")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--output-cap", type=int, required=False, default=None,
                    help="the model's documented maximum output tokens (verify per model; required for real providers)")
    ap.add_argument("--reasoning-headroom", type=int, default=0,
                    help="extra output tokens reserved for hidden reasoning (reasoning models)")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--thinking-budget", type=int, default=None)
    ap.add_argument("--split", default="main", choices=["main", "sweep", "heldout"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    arms = a.arms.split(",")
    for arm in arms:
        if arm not in A.ARMS:
            ap.error(f"unknown arm {arm}")
    levels = {int(x) for x in a.levels.split(",")}
    fams = None if a.families == "all" else set(a.families.split(","))
    if a.split in ("sweep", "heldout"):
        cs = [json.loads(l) for l in open(ROOT / "data" / f"{a.split}_v2.jsonl")]
        cs = [c for c in cs if fams is None or c["family"] in fams]
    else:
        cs = [c for c in CASES.load() if c["level"] in levels and (fams is None or c["family"] in fams)]
    if a.output_cap is None:
        if any(not m.startswith("mock:") for m in a.model):
            ap.error("--output-cap is required for real providers (use the model's documented output limit)")
        a.output_cap = 1_000_000
    settings = P.Settings(max_tokens=a.output_cap, temperature=a.temperature,
                          reasoning_effort=a.reasoning_effort, thinking_budget=a.thinking_budget)
    run_dir = ROOT / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if (run_dir / "scores.jsonl").exists():
        for line in open(run_dir / "scores.jsonl"):
            s = json.loads(line)
            done.add((s["model"], s["arm"], s["case_id"], s["sample"]))
    jobs = [(m, arm, c, k) for m in a.model for arm in arms for c in cs for k in range(a.k)
            if A.applicable(arm, c) and (m, arm, c["case_id"], k) not in done]
    missing = sorted({m.split(":")[0] for m in a.model if not P.available(m.split(":")[0])})
    print(f"{len(jobs)} requests to make ({len(done)} already done); providers without keys: {missing or 'none'}")
    if jobs:
        est_in = sum(len("".join(A.build(arm, c))) / 4 for (_, arm, c, _) in jobs)
        est_out = sum(min(output_budget(arm, c["n_parts"], 0), a.output_cap) / 1.6 for (_, arm, c, _) in jobs)
        infeas = sum(output_budget(arm, c["n_parts"], a.reasoning_headroom) > a.output_cap for (_, arm, c, _) in jobs)
        print(f"estimated input ~{est_in / 1e6:.2f}M tokens, visible output ~{est_out / 1e6:.2f}M tokens "
              f"(+ hidden reasoning, model-dependent); infeasible under the cap: {infeas}")
    if a.dry_run or not jobs:
        return
    if missing:
        sys.exit(f"missing API keys for: {missing}")
    manifest = {"run": a.run, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "models": a.model, "arms": arms, "levels": sorted(levels), "families": a.families, "k": a.k,
                "settings": settings.__dict__, "code_sha": code_hash(),
                "cases_sha": hashlib.sha256((ROOT / "data/cases_v2.jsonl").read_bytes()).hexdigest()[:16],
                "python": platform.python_version(), "argv": sys.argv}
    mf = run_dir / "manifest.json"
    history = json.loads(mf.read_text()) if mf.exists() else []
    mf.write_text(json.dumps((history if isinstance(history, list) else [history]) + [manifest], indent=1))
    n = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, a, run_dir, m, arm, c, k, settings) for (m, arm, c, k) in jobs]
        for f in as_completed(futs):
            s = f.result(); n += 1
            if n % 25 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}  last: {s['model']} {s['arm']} {s['case_id']} s{s['sample']} "
                      f"global={s['global_v2']:.1f} parse={s['parse_status']} exec={s['exec_status']}", flush=True)


if __name__ == "__main__":
    main()
