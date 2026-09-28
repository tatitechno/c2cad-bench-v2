"""Run models on v2 cases under one or more arms; every request and every score is appended to JSONL.

  python -m c2cad.runner.run --profile gpt-5.4 --arms json,neutral,tool,v1,mates,schema,cadquery,probe --k 3 --run main
  python -m c2cad.runner.run --profile gpt-5.4 --arms repair_generic,repair_verifier --run main   # after json
  python -m c2cad.runner.run --profile gpt-5.4 --split sweep --arms json,tool,mates --run sweep
  python -m c2cad.runner.run --profile gpt-5.4 --smoke --run smoke
  python -m c2cad.runner.run --model mock:noisy --arms json,tool --run selftest                 # offline

Profiles come from v2/config/models.json: model id, documented output cap, reasoning settings, whether the API
accepts a temperature, schema support and price. `--model provider:id` runs an ad-hoc model (then --output-cap
is required).

Records in v2/runs/<run>/:
  responses.jsonl  every request: model (profile), model_spec, arm, case, sample, round, the settings actually sent,
                   returned model version, finish reason, usage, cost, latency, attempts, enforcement, raw text
  scores.jsonl     one score per response. Scoring is separate from requesting and can be re-run offline
                   (python -m c2cad.runner.rescore --run <run>) after any scorer change, at no API cost.
  manifest.json    one entry per invocation: code / case-file hashes, git commit, registry entries, argv, time
Resume: a (model, arm, case, sample, round) whose response is ok or infeasible is never requested again; api
errors are requested again on the next invocation. Analysis keeps the last record per key.
Budget: --max-usd stops new requests once the run directory's recorded cost (from the providers' own usage
fields, hidden reasoning included) reaches the cap; requests already in flight finish.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from c2cad import cases as CASES  # noqa: E402
from c2cad import dsl, partlevel  # noqa: E402
from c2cad.evaluate import evaluate, shapes_for, verify  # noqa: E402
from c2cad.runner import arms as A, mock as MOCK, providers as P, sandbox  # noqa: E402
from c2cad.score import assignment_map  # noqa: E402

LOCK = threading.Lock()
TOKENS_PER_PART = 90          # v1 runner's measured average for one JSON part
PROGRAM_BUDGET = 12000        # visible tokens for program arms (tool, mates, cadquery)
TRUNCATED = {"length", "max_tokens", "MAX_TOKENS"}
REGISTRY = ROOT / "config" / "models.json"
SPLIT_FILES = {"main": "cases_v2.jsonl", "sweep": "sweep_v2.jsonl", "heldout": "heldout_v2.jsonl"}


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ModelCfg:
    name: str                 # recorded as "model" (profile name, unique per settings)
    spec: str                 # provider:model id
    output_cap: int
    headroom: int             # output tokens reserved for hidden reasoning
    settings: P.Settings
    price_in: float = 0.0     # USD per 1M input tokens
    price_out: float = 0.0    # USD per 1M output tokens (reasoning included)
    entry: dict = dataclasses.field(default_factory=dict)


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text())["models"]


def settings_from_entry(e: dict, temperature: float | None) -> P.Settings:
    return P.Settings(max_tokens=e["output_cap"], temperature=temperature,
                      send_temperature=e.get("send_temperature", True), reasoning_effort=e.get("reasoning_effort"),
                      thinking=e.get("thinking"), thinking_budget=e.get("thinking_budget"), stream=e.get("stream", True),
                      idle_timeout=e.get("idle_timeout", 1800), schema_mode=e.get("schema_mode", "json_schema"),
                      google_schema_field=e.get("google_schema_field", "responseJsonSchema"),
                      extra_body=e.get("extra_body", {}))


def model_cfgs(a) -> list[ModelCfg]:
    out = []
    reg = load_registry() if a.profile else {}
    for name in a.profile or []:
        if name not in reg:
            sys.exit(f"unknown profile {name!r}; known: {', '.join(sorted(reg))}")
        e = reg[name]
        out.append(ModelCfg(name, e["spec"], e["output_cap"], e.get("reasoning_headroom", 0),
                            settings_from_entry(e, a.temperature), e.get("price_in", 0.0), e.get("price_out", 0.0), e))
    for spec in a.model or []:
        if not spec.startswith("mock:") and a.output_cap is None:
            sys.exit("--output-cap is required for an ad-hoc real model (use the model's documented output limit)")
        cap = a.output_cap or 1_000_000
        st = P.Settings(max_tokens=cap, temperature=a.temperature, reasoning_effort=a.reasoning_effort,
                        thinking=a.thinking, thinking_budget=a.thinking_budget, stream=not a.no_stream)
        out.append(ModelCfg(spec, spec, cap, a.reasoning_headroom, st, a.price_in, a.price_out, {"spec": spec, "adhoc": True}))
    return out


def output_budget(arm: str, case: dict, headroom: int) -> int:
    """Output tokens to request: coordinate arms scale with the part count; program arms are short."""
    if arm in A.PROGRAM_ARMS:
        body = PROGRAM_BUDGET
    elif arm == "probe":
        body = int(len(A.probe_ids(case)) * TOKENS_PER_PART * 1.6) + 2048
    else:
        body = int(case["n_parts"] * TOKENS_PER_PART * 1.6) + 2048
    return body + headroom


# ---------------------------------------------------------------------------
# response -> parts -> score (shared with rescore.py)
# ---------------------------------------------------------------------------
def postprocess(arm: str, case: dict, text: str) -> dict:
    info = {"exec_status": "", "exec_error": "", "cad_stats": None}
    if arm == "tool":
        ok, out, err = sandbox.run(sandbox.extract_code(text))
        info.update(exec_status="ok" if ok else "error", exec_error=err[:300])
        text = out
    elif arm == "cadquery":
        ok, res, err = sandbox.run_cadquery(sandbox.extract_code(text))
        info.update(exec_status="ok" if ok else "error", exec_error=err[:300])
        if ok:
            info["cad_stats"] = res["stats"]
            return dict(info, value=res["parts"], parse_status="ok")
        return dict(info, value=None, parse_status="fail")
    value, ps = A.parse_json(text)
    if arm == "mates" and value is not None:
        try:
            value = dsl.run(value)
            info["exec_status"] = "ok"
        except RecursionError as e:
            info.update(exec_status="invalid_program", exec_error=f"RecursionError: {e}"[:300])
            value = None
        except Exception as e:   # any failure of a model-written program is that program's failure
            info.update(exec_status="invalid_program", exec_error=f"{type(e).__name__}: {e}"[:300])
            value = None
    return dict(info, value=value, parse_status=ps)


def response_status(rec: dict) -> str:
    if rec.get("error", "").startswith("infeasible"):
        return "infeasible"
    if not rec.get("ok"):
        return "api_error"
    if rec.get("finish_reason") in TRUNCATED:
        return "truncated"
    if rec.get("finish_reason") == "refusal":
        return "refusal"
    return "complete"


KEY_FIELDS = ("model", "model_spec", "arm", "split", "case_id", "family", "level", "sample", "round")


def score_record(rec: dict, case: dict, post: dict) -> dict:
    arm = rec["arm"]
    s = {k: rec.get(k) for k in KEY_FIELDS}
    s.update(phase=case["phase"], n_parts=case["n_parts"], response_status=response_status(rec),
             max_tokens=rec["settings"].get("max_tokens"), enforcement=rec.get("enforcement", "none"),
             exec_status=post["exec_status"], exec_error=post["exec_error"], parse_status=post["parse_status"],
             cad_stats=post["cad_stats"], cost_usd=rec.get("cost_usd", 0.0))
    value = post["value"]
    ids = A.probe_ids(case)
    if arm == "probe":
        ref, out = shapes_for(case, value if value is not None else [])
        m = partlevel.named_parts(ref, out, ids)
        s.update(n_ref=len(ref), n_out=len(out), probe=m, probe_pair=float(np.mean(m["pair_id"])),
                 probe_exact=float(np.mean(m["exact_id"])))
        return s
    r = evaluate(case, value if value is not None else [])
    d = r.to_dict()
    d["top_failures"] = d["top_failures"][:5]
    s.update(d)
    ref, out = shapes_for(case, value if value is not None else [])
    s["probe"] = partlevel.named_parts(ref, out, ids, assignment_map(ref, out))
    return s


# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, a, run_dir: Path, models: list[ModelCfg]):
        self.a, self.run_dir, self.models = a, run_dir, models
        self.spent = 0.0
        self.stopped = False
        self.latest: dict[tuple, dict] = {}
        self.session: list[tuple] = []          # keys requested by this invocation (smoke report)
        rp = run_dir / "responses.jsonl"
        if rp.exists():
            for line in open(rp):
                r = json.loads(line)
                self.latest[self.key(r)] = r
                self.spent += r.get("cost_usd", 0.0) or 0.0

    @staticmethod
    def key(r: dict) -> tuple:
        return (r["model"], r["arm"], r.get("split", "main"), r["case_id"], r["sample"], r.get("round", 0))

    def done(self, key) -> bool:
        r = self.latest.get(key)
        return r is not None and response_status(r) != "api_error"

    def _write(self, name: str, rec: dict):
        with LOCK:
            with open(self.run_dir / name, "a") as fh:
                fh.write(json.dumps(rec, default=float) + "\n")

    def request(self, mc: ModelCfg, arm: str, case: dict, sample: int, round_: int = 0, history=None, feedback=None):
        """One request and its score. Returns (response record, post-processed parts) or None if the budget stops it."""
        system, user = A.build(arm, case)
        msgs = history if history is not None else [{"role": "user", "content": user}]
        body = output_budget(arm, case, 0)
        need = body + mc.headroom         # the reasoning reserve is cut back before a case is declared infeasible
        st = dataclasses.replace(mc.settings, max_tokens=min(need, mc.output_cap),
                                 schema=A.ASSEMBLY_SCHEMA if arm == "schema" else None)
        if body > mc.output_cap:          # the answer alone cannot fit: infeasible, never a model failure
            resp = P.Response("", False, f"infeasible: the answer needs ~{body} output tokens, cap {mc.output_cap}")
        else:
            with LOCK:
                if self.a.max_usd is not None and self.spent >= self.a.max_usd:
                    self.stopped = True
                    return None
            if mc.spec.startswith("mock:"):
                resp = MOCK.response(mc.spec, arm, case, sample, history)
            else:
                resp = P.call(mc.spec, system, msgs, st)
        cost = P.cost_usd(mc.spec, resp.usage, mc.price_in, mc.price_out) if resp.ok else 0.0
        with LOCK:
            self.spent += cost
        hist_sha = A.sha(json.dumps(msgs)) if history is not None else A.sha(user)
        rec = {"model": mc.name, "model_spec": mc.spec, "arm": arm, "split": case.get("split", "main"),
               "case_id": case["case_id"], "family": case["family"], "level": case["level"], "sample": sample,
               "round": round_, "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "system_sha": A.sha(system), "user_sha": hist_sha, "settings": st.record(),
               "request_meta": resp.request_meta, "returned_model": resp.returned_model,
               "finish_reason": resp.finish_reason, "usage": resp.usage, "cost_usd": round(cost, 6),
               "latency_s": round(resp.latency_s, 2), "attempts": resp.attempts, "enforcement": resp.enforcement,
               "http_status": resp.http_status, "ok": resp.ok, "error": resp.error, "feedback": feedback,
               "text": resp.text}
        self._write("responses.jsonl", rec)          # the paid response is saved before anything can fail
        with LOCK:
            self.latest[self.key(rec)] = rec
            self.session.append(self.key(rec))
        post = {"value": None, "parse_status": "fail", "exec_status": "", "exec_error": "", "cad_stats": None}
        try:
            if resp.ok:
                post = postprocess(arm, case, resp.text)
            sc = score_record(rec, case, post)
        except Exception:
            sc = {k: rec.get(k) for k in KEY_FIELDS}
            sc.update(response_status="harness_error", harness_error=traceback.format_exc()[-1500:])
        self._write("scores.jsonl", sc)
        return rec, post, sc

    def repair_chain(self, mc: ModelCfg, arm: str, case: dict, sample: int):
        """Round 0 is the json answer for the same (model, case, sample); rounds 1..R add feedback."""
        split = case.get("split", "main")
        j = self.latest.get((mc.name, "json", split, case["case_id"], sample, 0))
        if j is None or response_status(j) != "complete":
            return "no_json"
        post = postprocess("json", case, j["text"])
        if evaluate(case, post["value"] if post["value"] is not None else []).exact:
            return "already_exact"
        _, user = A.build("json", case)
        history = [{"role": "user", "content": user}, {"role": "assistant", "content": j["text"]}]
        value, parsed = post["value"], post["value"] is not None
        for rnd in range(1, self.a.repair_rounds + 1):
            report = verify(case, value) if parsed else None
            fb = A.repair_feedback(arm, report, parsed)
            if fb is None:
                return f"clean_after_{rnd - 1}"
            history = history + [{"role": "user", "content": fb}]
            key = (mc.name, arm, split, case["case_id"], sample, rnd)
            if self.done(key):
                rec = self.latest[key]
                post = postprocess(arm, case, rec["text"]) if rec["ok"] else {"value": None}
            else:
                got = self.request(mc, arm, case, sample, rnd, history=history, feedback=fb)
                if got is None:
                    return "budget"
                rec, post, _ = got
            if response_status(rec) != "complete":
                return f"stopped_{response_status(rec)}_at_{rnd}"
            history = history + [{"role": "assistant", "content": rec["text"]}]
            value, parsed = post["value"], post["value"] is not None
            if parsed and evaluate(case, value).exact:
                return f"exact_at_{rnd}"
        return "rounds_done"


# ---------------------------------------------------------------------------
def load_cases(split: str, families, levels) -> list[dict]:
    cs = [json.loads(l) for l in open(ROOT / "data" / SPLIT_FILES[split])]
    for c in cs:
        c.setdefault("split", split)
    if split == "main":
        cs = [c for c in cs if c["level"] in levels]
    return [c for c in cs if families is None or c["family"] in families]


def smoke_cases(cs, mc: ModelCfg, arm: str) -> list[dict]:
    """The smallest case and the largest case that fits the model's output cap."""
    ok = [c for c in cs if A.applicable(arm, c) and output_budget(arm, c, 0) <= mc.output_cap]
    ok.sort(key=lambda c: (c["n_parts"], c["case_id"]))
    return [ok[0], ok[-1]] if len(ok) > 1 else ok


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "c2cad").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def estimate(jobs, repair_jobs, a, models) -> dict:
    """Pre-run estimate (not a bill): input from prompt length, visible output from the part count, hidden
    reasoning from the registry's est_reasoning_tokens per request."""
    by = {}
    for mc, arm, c, _ in jobs:
        system, user = A.build(arm, c)
        tin = len(system + user) / 3.5
        if arm in A.PROGRAM_ARMS:
            vis = 3000
        elif arm == "probe":
            vis = len(A.probe_ids(c)) * TOKENS_PER_PART
        else:
            vis = c["n_parts"] * TOKENS_PER_PART
        tout = vis + mc.entry.get("est_reasoning_tokens", 0)
        d = by.setdefault((mc.name, arm), {"requests": 0, "in": 0.0, "out": 0.0, "usd": 0.0, "infeasible": 0})
        d["requests"] += 1
        d["in"] += tin
        d["out"] += tout
        d["usd"] += tin * mc.price_in / 1e6 + tout * mc.price_out / 1e6
        d["infeasible"] += output_budget(arm, c, 0) > mc.output_cap
    for mc, arm, c, _ in repair_jobs:      # upper bound: every chain runs all rounds with a growing history
        system, user = A.build("json", c)
        vis = c["n_parts"] * TOKENS_PER_PART
        d = by.setdefault((mc.name, arm), {"requests": 0, "in": 0.0, "out": 0.0, "usd": 0.0, "infeasible": 0})
        for rnd in range(1, a.repair_rounds + 1):
            tin = len(system + user) / 3.5 + rnd * (vis + 300)
            tout = vis + mc.entry.get("est_reasoning_tokens", 0)
            d["requests"] += 1
            d["in"] += tin
            d["out"] += tout
            d["usd"] += tin * mc.price_in / 1e6 + tout * mc.price_out / 1e6
    return by


def print_estimate(by: dict):
    tot = {"requests": 0, "usd": 0.0}
    print(f"{'model':28s} {'arm':16s} {'requests':>8s} {'in Mtok':>8s} {'out Mtok':>9s} {'USD':>9s} {'infeas':>6s}")
    for (m, arm), d in sorted(by.items()):
        print(f"{m:28s} {arm:16s} {d['requests']:8d} {d['in'] / 1e6:8.2f} {d['out'] / 1e6:9.2f} {d['usd']:9.2f} "
              f"{d['infeasible']:6d}")
        tot["requests"] += d["requests"]
        tot["usd"] += d["usd"]
    print(f"TOTAL requests {tot['requests']}, estimated USD {tot['usd']:.2f} (repair arms counted as if every chain "
          f"ran all rounds; prices from the registry, verify before spending)")


SMOKE_CHECKS = ("ok", "not_truncated", "finish_reason", "usage", "returned_model", "parsed")


def smoke_report(runner: Runner, keys) -> bool:
    good = True
    print("\nSMOKE CHECKS (each must be True before a full run)")
    print(f"{'model':24s} {'arm':16s} {'case':34s} " + " ".join(f"{c:>14s}" for c in SMOKE_CHECKS))
    scores = {}
    sp = runner.run_dir / "scores.jsonl"
    for line in open(sp):
        s = json.loads(line)
        scores[(s["model"], s["arm"], s.get("split", "main"), s["case_id"], s["sample"], s.get("round", 0))] = s
    for key in keys:
        r = runner.latest.get(key)
        if r is None:
            continue
        s = scores.get(key, {})
        infeasible = response_status(r) == "infeasible"
        parsed = s.get("parse_status") not in (None, "fail") and s.get("exec_status") not in ("error", "invalid_program")
        checks = {"ok": r["ok"] or infeasible, "not_truncated": response_status(r) != "truncated",
                  "finish_reason": bool(r["finish_reason"]) or infeasible, "usage": bool(r["usage"]) or infeasible,
                  "returned_model": bool(r["returned_model"]) or infeasible, "parsed": parsed or infeasible}
        good &= all(checks.values())
        print(f"{r['model']:24s} {r['arm']:16s} {r['case_id']:34s} " + " ".join(f"{str(checks[c]):>14s}" for c in SMOKE_CHECKS))
        if not r["ok"]:
            print(f"    error: {r['error'][:300]}")
    return good


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", action="append", help="model profile from v2/config/models.json (repeatable)")
    ap.add_argument("--model", action="append", help="ad-hoc provider:model spec, or mock:<kind> (repeatable)")
    ap.add_argument("--arms", default="json")
    ap.add_argument("--families", default="all")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--k", type=int, default=1, help="samples per case (single-turn arms)")
    ap.add_argument("--repair-rounds", type=int, default=2)
    ap.add_argument("--repair-samples", type=int, default=1, help="json samples (0..n-1) that seed repair chains")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="main", choices=list(SPLIT_FILES))
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="one temperature for every model and arm (decision 2026-09-28); not sent to APIs that "
                         "reject sampling parameters (registry send_temperature=false), which is recorded")
    ap.add_argument("--output-cap", type=int, default=None, help="ad-hoc models: documented maximum output tokens")
    ap.add_argument("--reasoning-headroom", type=int, default=0, help="ad-hoc models: tokens reserved for reasoning")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--thinking", default=None, choices=[None, "adaptive", "disabled"])
    ap.add_argument("--thinking-budget", type=int, default=None)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--price-in", type=float, default=0.0)
    ap.add_argument("--price-out", type=float, default=0.0)
    ap.add_argument("--max-usd", type=float, default=None, help="stop new requests once the run's recorded cost reaches this")
    ap.add_argument("--smoke", action="store_true", help="per (model, arm): smallest and largest feasible case, k=1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.profile and not a.model:
        ap.error("give --profile and/or --model")
    arms = a.arms.split(",")
    for arm in arms:
        if arm not in A.ARMS:
            ap.error(f"unknown arm {arm}; arms: {', '.join(A.ARMS)}")
    models = model_cfgs(a)
    levels = {int(x) for x in a.levels.split(",")}
    fams = None if a.families == "all" else set(a.families.split(","))
    cs = load_cases(a.split, fams, levels)
    run_dir = ROOT / "runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(a, run_dir, models)
    single = [x for x in arms if x in A.SINGLE_TURN_ARMS]
    repair = [x for x in arms if x in A.REPAIR_ARMS]
    if a.smoke:
        a.k, a.repair_rounds = 1, 1
    for mc in models:
        if "schema" in single and mc.settings.schema_mode == "none":
            print(f"note: {mc.name} has no native schema support (registry schema_mode=none); schema arm skipped")
        if "cadquery" in single and not sandbox.cad_available():
            sys.exit(f"cadquery arm needs the CAD environment at {sandbox.CAD_PYTHON} (see v2/RUNBOOK.md)")
    jobs, repair_jobs = [], []
    for mc in models:
        for arm in single:
            if arm == "schema" and mc.settings.schema_mode == "none":
                continue
            pool = smoke_cases(cs, mc, arm) if a.smoke else [c for c in cs if A.applicable(arm, c)]
            for c in pool:
                for k in range(a.k):
                    if not runner.done((mc.name, arm, a.split, c["case_id"], k, 0)):
                        jobs.append((mc, arm, c, k))
        for arm in repair:
            pool = smoke_cases(cs, mc, "json") if a.smoke else cs
            for c in pool:
                for k in range(a.repair_samples):
                    repair_jobs.append((mc, arm, c, k))
    missing = sorted({mc.spec.split(":")[0] for mc in models if not P.available(mc.spec.split(":")[0])})
    print(f"{len(jobs)} single-turn requests to make; {len(repair_jobs)} repair chains (up to {a.repair_rounds} "
          f"rounds each); recorded cost so far ${runner.spent:.2f}; providers without keys: {missing or 'none'}")
    by = estimate(jobs, repair_jobs, a, models)
    if by:
        print_estimate(by)
    if a.dry_run or (not jobs and not repair_jobs):
        return 0
    if missing:
        sys.exit(f"missing API keys for: {missing}")
    manifest = {"run": a.run, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "split": a.split, "models": {mc.name: {"spec": mc.spec, "entry": mc.entry, "settings": mc.settings.record(),
                                                       "output_cap": mc.output_cap, "headroom": mc.headroom}
                                             for mc in models},
                "arms": arms, "levels": sorted(levels), "families": a.families, "k": a.k,
                "repair_rounds": a.repair_rounds, "repair_samples": a.repair_samples, "temperature": a.temperature,
                "max_usd": a.max_usd, "smoke": a.smoke, "code_sha": code_hash(), "git_commit": git_commit(),
                "cases_sha": hashlib.sha256((ROOT / "data" / SPLIT_FILES[a.split]).read_bytes()).hexdigest()[:16],
                "python": platform.python_version(), "platform": platform.platform(), "argv": sys.argv}
    mf = run_dir / "manifest.json"
    history = json.loads(mf.read_text()) if mf.exists() else []
    mf.write_text(json.dumps((history if isinstance(history, list) else [history]) + [manifest], indent=1))
    n = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(runner.request, mc, arm, c, k) for (mc, arm, c, k) in jobs]
        for f in as_completed(futs):
            got = f.result()
            n += 1
            if got and (n % 25 == 0 or n == len(jobs)):
                rec, _, s = got
                print(f"  {n}/{len(jobs)}  {rec['model']} {rec['arm']} {rec['case_id']} s{rec['sample']} "
                      f"status={s.get('response_status')} global={s.get('global_v2', s.get('probe_pair', float('nan'))):.1f} "
                      f"spent=${runner.spent:.2f}", flush=True)
    outcomes = {}
    if repair_jobs:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(runner.repair_chain, mc, arm, c, k): (mc.name, arm) for (mc, arm, c, k) in repair_jobs}
            for f in as_completed(futs):
                o = f.result()
                key = futs[f] + (o.split("_at_")[0] if "_at_" in o else o.split("_after_")[0],)
                outcomes[key] = outcomes.get(key, 0) + 1
        print("repair chains:", json.dumps({" / ".join(k): v for k, v in sorted(outcomes.items())}, indent=1))
    if runner.stopped:
        print(f"STOPPED by --max-usd {a.max_usd}: recorded cost ${runner.spent:.2f}. Re-run to continue after raising it.")
    print(f"recorded cost for this run directory: ${runner.spent:.2f}")
    if a.smoke:
        ok = smoke_report(runner, sorted(set(runner.session), key=str))
        print("SMOKE PASSED" if ok else "SMOKE FAILED: fix the failures above before a full run")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
