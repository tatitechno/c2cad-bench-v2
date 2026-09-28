"""Guarantees for the live-run machinery (no API calls). Run: pytest -q v2/tests"""
import copy
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "scripts"))
from c2cad import cadcode, cases, prism  # noqa: E402
from c2cad.evaluate import evaluate, verify  # noqa: E402
from c2cad.runner import arms as A, providers as P, run as RUN, sandbox as SB  # noqa: E402

CASES = cases.load()
IDS = [c["case_id"] for c in CASES]
CAD = pytest.mark.skipif(not SB.cad_available(), reason="CAD environment v2/.venv-cad missing")


# ---------------------------------------------------------------------------
# providers: retries, request bodies
# ---------------------------------------------------------------------------
def test_transient_errors_are_retried_and_counted(monkeypatch):
    calls = []

    def fake(provider, model, system, msgs, st):
        calls.append(1)
        if len(calls) < 3:
            raise P._Transient("HTTP 529: overloaded", status=529)
        return P.Response("[]", True, finish_reason="stop")

    monkeypatch.setattr(P, "_openai_like", fake)
    r = P.call("openai:x", "s", "u", P.Settings(), sleep=lambda s: None)
    assert r.ok and r.attempts == 3


def test_fatal_errors_are_not_retried(monkeypatch):
    calls = []

    def fake(provider, model, system, msgs, st):
        calls.append(1)
        raise P._Fatal("HTTP 400: bad", status=400)

    monkeypatch.setattr(P, "_openai_like", fake)
    r = P.call("openai:x", "s", "u", P.Settings(), sleep=lambda s: None)
    assert not r.ok and len(calls) == 1 and r.http_status == 400


class _FakeResp:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_openai_body_temperature_and_schema(monkeypatch):
    seen = {}

    def fake_post(url, headers, body, st, stream):
        seen["body"] = body
        return _FakeResp({"model": "m", "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 2}})

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setattr(P, "_post", fake_post)
    st = P.Settings(max_tokens=100, temperature=1.0, send_temperature=False, stream=False, schema=A.ASSEMBLY_SCHEMA,
                    reasoning_effort="high")
    r = P._openai_like("openai", "gpt-x", "sys", [{"role": "user", "content": "u"}], st)
    b = seen["body"]
    assert "temperature" not in b and b["reasoning_effort"] == "high" and b["max_completion_tokens"] == 100
    assert b["response_format"]["json_schema"]["strict"] is True and r.enforcement == "strict_schema"
    assert r.request_meta["response_format"]["schema"] == "<assembly schema>"      # schema not copied into records
    st2 = dataclasses_replace(st, send_temperature=True, schema=None)
    P._openai_like("deepseek", "d", "sys", [{"role": "user", "content": "u"}], st2)
    assert seen["body"]["temperature"] == 1.0 and "max_tokens" in seen["body"] and "response_format" not in seen["body"]


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def test_google_body_schema_and_thinking(monkeypatch):
    seen = {}

    def fake_post(url, headers, body, st, stream):
        seen["url"], seen["body"] = url, body
        return _FakeResp({"modelVersion": "g", "candidates": [{"content": {"parts": [{"text": "[]"}]},
                                                                "finishReason": "STOP"}], "usageMetadata": {}})

    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setattr(P, "_post", fake_post)
    st = P.Settings(max_tokens=50, temperature=1.0, stream=False, schema=A.ASSEMBLY_SCHEMA,
                    extra_body={"generationConfig": {"thinkingConfig": {"thinkingLevel": "high"}}})
    r = P._google("gemini-x", "sys", [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}], st)
    g = seen["body"]["generationConfig"]
    assert g["responseJsonSchema"] == A.ASSEMBLY_SCHEMA and g["thinkingConfig"]["thinkingLevel"] == "high"
    assert g["maxOutputTokens"] == 50 and g["temperature"] == 1.0
    assert [c["role"] for c in seen["body"]["contents"]] == ["user", "model"] and r.text == "[]"


def test_anthropic_request_omits_rejected_parameters(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    seen = {}

    class FakeMsg:
        content = [types.SimpleNamespace(type="text", text="[]")]
        model, stop_reason = "claude-x", "end_turn"
        usage = types.SimpleNamespace(model_dump=lambda: {"input_tokens": 1, "output_tokens": 2})

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return FakeMsg()

    class FakeClient:
        def __init__(self, **kw):
            self.messages = types.SimpleNamespace(stream=self.stream)

        def stream(self, **kw):
            seen.update(kw)
            return FakeStream()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    st = P.Settings(max_tokens=64, temperature=1.0, send_temperature=False, reasoning_effort="high",
                    schema=A.ASSEMBLY_SCHEMA)
    r = P._anthropic("claude-x", "sys", [{"role": "user", "content": "u"}], st)
    assert "temperature" not in seen and "thinking" not in seen["extra_body"]
    oc = seen["extra_body"]["output_config"]
    assert oc["effort"] == "high" and oc["format"]["type"] == "json_schema" and r.enforcement == "strict_schema"
    assert P.billed_tokens("anthropic:x", r.usage) == (1, 2)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_reference_conforms_to_the_constrained_decoding_schema(case):
    jsonschema = pytest.importorskip("jsonschema")
    ref = {"shapes": [{k: v for k, v in p.items() if k != "symbolic"} for p in case["reference"]]}
    jsonschema.validate(ref, A.ASSEMBLY_SCHEMA)


def test_schema_is_strict_mode_compatible():
    def walk(s):
        if s.get("type") == "object":
            assert s.get("additionalProperties") is False and set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
        for k in ("items",):
            if k in s:
                walk(s[k])
        for v in s.get("anyOf", []):
            walk(v)
        assert not {"minItems", "maxItems", "minimum", "maximum"} & set(s)
    walk(A.ASSEMBLY_SCHEMA)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_probe_ids_fixed_distinct_in_range_and_named_in_prompt(case):
    ids = A.probe_ids(case)
    assert ids == A.probe_ids(copy.deepcopy(case))
    assert len(ids) == min(3, case["n_parts"]) and len(set(ids)) == len(ids)
    assert all(0 <= i < case["n_parts"] for i in ids) and case["n_parts"] - 1 in ids
    assert all(str(i) in A.build("probe", case)[1].split("PROBE MODE")[1] for i in ids)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_verifier_is_silent_on_the_reference(case):
    rep = verify(case, case["reference"])
    assert rep["ids_usable"] and rep["n_failed"] == 0 and not rep["missing_ids"] and not rep["violations"]
    assert A.repair_feedback("repair_verifier", rep, True) is None


def test_verifier_reports_sentences_ids_and_count():
    case = CASES[0]
    out = copy.deepcopy(case["reference"])[:-1]
    out[3]["start"][2] += 5
    rep = verify(case, out)
    fb = A.repair_feedback("repair_verifier", rep, True)
    assert f"defines {case['n_parts']} parts" in fb and "missing" in fb and "Not satisfied" in fb
    assert "reference" not in fb.split("It found:")[1]
    assert A.repair_feedback("repair_verifier", None, False).count("could not be read as JSON") == 1
    assert A.repair_feedback("repair_generic", rep, True) == A.REPAIR_GENERIC


def test_every_arm_builds_for_every_case():
    for arm in A.SINGLE_TURN_ARMS:
        for c in CASES:
            if A.applicable(arm, c):
                system, user = A.build(arm, c)
                assert system and c["prompt_body"][:40] in user or arm == "v1" or arm == "neutral"


def test_audit_gate_task_text_of_every_arm_has_no_scaffolding_flag():
    """Strict CI gate (promised in the rebuttal): the released v1 audit regexes find nothing in any v2 task text."""
    import audit_prompts as AP
    sys.path.insert(0, str(ROOT / "experiments"))
    import e00_prompt_audit as E00
    for arm in A.SINGLE_TURN_ARMS:
        if arm == "v1":
            continue
        for c in CASES:
            if A.applicable(arm, c):
                row = AP.audit_case({"prompt": E00.task_text(arm, c)})
                assert not row.high_risk_flag, (arm, c["case_id"])


# ---------------------------------------------------------------------------
# prisms and the CAD kernel
# ---------------------------------------------------------------------------
def test_prism_readings():
    aligned = {"type": "prism", "center": [0, 0, 0], "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "extents": [4, 2, 1]}
    r = prism.readings(aligned)
    assert r[0]["type"] == "box" and r[0]["size"] == [4, 2, 1] and sum(x["type"] == "beam" for x in r) == 3
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    rotated = dict(aligned, axes=[[c, s, 0], [-s, c, 0], [0, 0, 1]])
    assert all(x["type"] == "beam" for x in prism.readings(rotated))


@CAD
@pytest.mark.parametrize("cid", ["spiral_staircase_level_1", "flanged_pipe_joint_level_1", "compound_eye_level_1",
                                 "domino_ring_level_1", "honeycomb_lattice_level_1"])
def test_cadquery_sandbox_round_trip_is_exact(cid):
    case = next(c for c in CASES if c["case_id"] == cid)
    post = RUN.postprocess("cadquery", case, "```python\n" + cadcode.to_cadquery_code(case["reference"]) + "```")
    assert post["exec_status"] == "ok" and post["cad_stats"]["n_unrecognized"] == 0
    r = evaluate(case, post["value"])
    assert r.exact and r.geometry > 99.99


@CAD
def test_every_reference_primitive_builds_and_is_recovered():
    """Mechanical converter + recovery on all 75 references (5,042 primitives), in one CAD process."""
    code = ("import json, sys\nsys.path.insert(0, %r)\nfrom c2cad import cadkernel as K\n"
            "bad = 0\nfor line in open(%r):\n    c = json.loads(line)\n"
            "    st = K.recover_all({p['id']: K.to_solid(p) for p in c['reference']})['stats']\n"
            "    bad += st['n_unrecognized'] + st['n_invalid'] + (st['n_solids'] != len(c['reference']))\n"
            "print(bad)\n") % (str(ROOT), str(ROOT / "data" / "cases_v2.jsonl"))
    p = subprocess.run([str(SB.CAD_PYTHON), "-c", code], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-500:]
    assert p.stdout.strip() == "0"


@CAD
def test_cadquery_sandbox_rejects_file_io_and_reports_unions():
    assert SB.run_cadquery("import cadquery as cq\ncq.exporters.export(cq.Workplane().box(1,1,1), 'x.step')")[0] is False
    ok, res, _ = SB.run_cadquery("import cadquery as cq\nparts = {0: cq.Workplane().box(2,2,2).union(cq.Workplane().sphere(1.5))}")
    assert ok and res["stats"]["n_unrecognized"] == 1 and res["parts"] == []


# ---------------------------------------------------------------------------
# runner: resume, budget, infeasible, crash isolation
# ---------------------------------------------------------------------------
def _args(**kw):
    base = dict(max_usd=None, repair_rounds=2, temperature=1.0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _mc(spec="mock:reference", cap=1_000_000, headroom=0):
    return RUN.ModelCfg(spec, spec, cap, headroom, P.Settings(max_tokens=cap), 1.0, 4.0, {})


def test_resume_requests_api_errors_again(tmp_path):
    rec = {"model": "m", "arm": "json", "split": "main", "case_id": "c", "sample": 0, "round": 0, "ok": False,
           "error": "HTTP 503", "finish_reason": "", "cost_usd": 0.0}
    (tmp_path / "responses.jsonl").write_text(json.dumps(rec) + "\n")
    r = RUN.Runner(_args(), tmp_path, [])
    assert not r.done(("m", "json", "main", "c", 0, 0))
    rec.update(ok=True, error="", finish_reason="stop")
    (tmp_path / "responses.jsonl").write_text(json.dumps(rec) + "\n")
    assert RUN.Runner(_args(), tmp_path, []).done(("m", "json", "main", "c", 0, 0))


def test_budget_guard_stops_new_requests(tmp_path):
    r = RUN.Runner(_args(max_usd=0.0), tmp_path, [])
    assert r.request(_mc(), "json", dict(CASES[0], split="main"), 0) is None and r.stopped


def test_infeasible_is_recorded_not_scored_as_failure(tmp_path):
    big = max(CASES, key=lambda c: c["n_parts"])
    r = RUN.Runner(_args(), tmp_path, [])
    rec, post, sc = r.request(_mc(cap=20000, headroom=4000), "json", dict(big, split="main"), 0)
    assert sc["response_status"] == "infeasible" and rec["cost_usd"] == 0.0
    small = min(CASES, key=lambda c: c["n_parts"])
    rec, _, sc = r.request(_mc(cap=RUN.output_budget("json", small, 0) + 10, headroom=4000), "json",
                           dict(small, split="main"), 0)
    assert sc["response_status"] == "complete" and rec["settings"]["max_tokens"] <= RUN.output_budget("json", small, 0) + 10


def test_broken_model_programs_are_recorded_not_raised():
    case = CASES[0]
    for prog in ('{"steps": [{"pattern": "circular", "seed": [0], "count": 0}]}', '{"steps": 5}', '{"steps": [null]}'):
        post = RUN.postprocess("mates", case, prog)
        assert post["value"] is None and post["exec_status"] == "invalid_program"
    post = RUN.postprocess("tool", case, "```python\nprint(1/0)\n```")
    assert post["exec_status"] == "error" and post["value"] is None


def test_scoring_crash_keeps_the_paid_response(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("scorer bug")
    monkeypatch.setattr(RUN, "score_record", boom)
    r = RUN.Runner(_args(), tmp_path, [])
    rec, _, sc = r.request(_mc(), "json", dict(CASES[0], split="main"), 0)
    assert sc["response_status"] == "harness_error" and "scorer bug" in sc["harness_error"]
    assert json.loads((tmp_path / "responses.jsonl").read_text().splitlines()[0])["text"] == rec["text"]


def test_registry_entries_are_complete():
    reg = RUN.load_registry()
    for name, e in reg.items():
        assert e["spec"].split(":")[0] in set(P.OPENAI_COMPATIBLE) | {"anthropic", "google", "mock"}, name
        assert e["output_cap"] > 0 and "price_in" in e and "price_out" in e and "verified" in e, name
        assert e.get("schema_mode", "json_schema") in ("json_schema", "json_object", "none"), name
        if e["spec"].startswith("anthropic:claude-opus-5") or e["spec"].startswith(("anthropic:claude-sonnet-5",
                                                                                    "anthropic:claude-fable")):
            assert e["send_temperature"] is False, name       # these APIs return 400 for sampling parameters
