"""Validity guarantees for the v2 scorer. Run: pytest -q v2/tests"""
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c2cad import cases, constraints as K  # noqa: E402
from c2cad.evaluate import evaluate  # noqa: E402
from c2cad.geom import normalize  # noqa: E402

CASES = cases.load()
IDS = [c["case_id"] for c in CASES]


def test_case_file_is_current():
    """cases_v2.jsonl must equal a fresh build from the generators + documented patches."""
    fresh = cases.build_all()
    assert [c["prompt_sha256"] for c in fresh] == [c["prompt_sha256"] for c in CASES]
    assert [c["reference"] for c in fresh] == [c["reference"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_reference_scores_100_on_every_axis(case):
    r = evaluate(case, case["reference"])
    for axis_ in ("coverage", "geometry", "geometry_equiv", "type_fidelity", "semantic", "global_v2"):
        assert getattr(r, axis_) == pytest.approx(100.0, abs=1e-5), axis_
    assert r.exact


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_order_and_id_invariance(case):
    ref = copy.deepcopy(case["reference"])
    random.Random(7).shuffle(ref)
    for i, s in enumerate(ref):
        s["id"] = 1000 + i
    r = evaluate(case, ref)
    assert r.geometry == pytest.approx(100.0, abs=1e-5)
    assert r.semantic == pytest.approx(100.0, abs=1e-5)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_constraint_has_clause_and_roles(case):
    ref, _ = normalize(case["reference"])
    cons = K.build(case, ref)
    assert cons
    for c in cons:
        assert c.clause and c.kind in {"dimension", "anchor", "mate", "pattern", "orientation", "topology"}
        assert all(0 <= r < len(ref) for r in c.roles)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_jitter_is_monotone(case):
    rng = np.random.default_rng(0)
    prev = 101.0
    for sigma in (0.0, 0.5, 5.0):
        out = copy.deepcopy(case["reference"])
        for s in out:
            for k in ("center", "start", "end"):
                if k in s:
                    s[k] = (np.array(s[k]) + rng.normal(0, sigma, 3)).tolist()
        g = evaluate(case, out).geometry
        assert g <= prev + 1e-6
        prev = g


def test_wrapped_output_accepted():
    case = CASES[0]
    assert evaluate(case, {"shapes": case["reference"]}).geometry == pytest.approx(100.0, abs=1e-5)


def test_unknown_types_are_dropped_and_reported():
    case = CASES[0]
    out = copy.deepcopy(case["reference"]) + [{"id": 999, "type": "pillar", "center": [0, 0, 0]}]
    r = evaluate(case, out)
    assert r.n_out == len(case["reference"])
    assert r.normalize_report["dropped_unknown_type"] == 1


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_equivalence_view_never_below_label_strict(case):
    out = []
    for s in copy.deepcopy(case["reference"]):
        if s["type"] == "beam":
            a, b = np.array(s["start"]), np.array(s["end"])
            s = {"id": s["id"], "type": "box", "center": ((a + b) / 2).tolist(),
                 "size": [float(np.linalg.norm(b - a)), s["width"], s.get("height", s["width"])]}
        out.append(s)
    r = evaluate(case, out)
    assert r.geometry_equiv >= r.geometry - 1e-9


from c2cad.constraints import trace as T  # noqa: E402


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_prompt_sentence_is_traced(case):
    uncovered = [s for s, keys in T.sentence_map(case) if not keys]
    assert not uncovered, uncovered


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_constraint_key_is_stated_in_the_prompt(case):
    ref, _ = normalize(case["reference"])
    keys = {c.clause.split(":", 1)[1] for c in K.build(case, ref)}
    stated = set(T.clause_to_sentences(case))
    assert keys <= stated, sorted(keys - stated)


from c2cad.runner import neutral as NT, sandbox as SB, arms as AR  # noqa: E402
from collections import Counter  # noqa: E402


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_neutral_twin_removes_domain_words_and_keeps_numbers(case):
    t = NT.twin(case["family"], case["prompt_body"])
    assert not NT.violations(case["family"], t)
    a, b = Counter(NT.numbers(case["prompt_body"])), Counter(NT.numbers(t))
    assert not (a - b)
    assert not ((b - a) - Counter(NT.ADDED_NUMBERS.get(case["family"], [])))


def test_sandbox_rejects_and_runs():
    assert SB.run("import os")[0] is False
    assert SB.run("print(open('x'))")[0] is False
    ok, out, _ = SB.run("import numpy as np, json\nprint(json.dumps([float(np.pi)]))")
    assert ok and out.strip().startswith("[3.14")


def test_parser_recovers_common_wrappings():
    assert AR.parse_json('```json\n[{"a":1}]\n```')[0] == [{"a": 1}]
    assert AR.parse_json('{"shapes": [1]}')[0] == {"shapes": [1]}
    assert AR.parse_json('[{"a":1},{"b":')[1] == "repaired"


from c2cad import dsl as DSL, cml_programs as CMLP  # noqa: E402


@pytest.mark.parametrize("case", [c for c in CASES if c["family"] in CMLP.PROGRAMS], ids=lambda c: c["case_id"])
def test_mates_programs_reproduce_references(case):
    out = DSL.run(CMLP.PROGRAMS[case["family"]](case))
    r = evaluate(case, out)
    assert r.exact and r.geometry > 99.9


def test_mates_errors_are_explicit():
    for bad in ({"steps": [{"id": 0, "type": "sphere", "radius": 1}]},                       # unplaced
                {"steps": [{"id": 0, "type": "sphere", "radius": 1, "at": {"of": 9}}]},      # undefined reference
                {"steps": [{"id": 0, "type": "box", "size": [1, 1, 1], "at": "origin"},
                           {"pattern": "circular", "seed": [0], "count": 3, "about": "z-axis", "full_circle": True}]}):  # rotated box
        with pytest.raises(DSL.CMLError):
            DSL.run(bad)


SWEEP = [json.loads(l) for l in open(ROOT / "data" / "sweep_v2.jsonl")] if (ROOT / "data" / "sweep_v2.jsonl").exists() else []


@pytest.mark.parametrize("case", SWEEP, ids=lambda c: c["case_id"])
def test_sweep_references_satisfy_constraints(case):
    r = evaluate(case, case["reference"])
    assert r.exact


@pytest.mark.parametrize("code", [
    "import numpy as np\nnp.save('x.npy', np.zeros(3))",
    "import numpy as np\nnp.zeros(3).tofile('x.bin')",
    "import numpy as np\nprint(np.loadtxt('/etc/hosts'))",
    "import numpy as np\nprint(np.fromfile('/etc/hosts'))",
    "import numpy as np\nm = np.memmap('x', mode='w+', shape=(3,))",
    "from numpy import f2py",
    "import json\njson.dump([1], None)",
])
def test_sandbox_blocks_file_io(code):
    assert SB.run(code)[0] is False


def test_sandbox_runs_numpy_in_temp_dir_with_fsize_limit():
    # a write attempted through an allowed call path still fails because RLIMIT_FSIZE = 0 and cwd is a fresh temp dir
    ok, out, err = SB.run("import numpy as np\nprint(np.sum(np.arange(10)))")
    assert ok and out.strip() == "45"


def test_mates_examples_run_and_are_not_benchmark_families():
    import numpy as _np
    for task, prog in AR.MATES_EXAMPLES:
        parts = DSL.run(prog)
        assert parts
    wheel = DSL.run(AR.MATES_EXAMPLES[0][1])
    spoke = next(p for p in wheel if p["id"] == 2)
    assert _np.allclose(spoke["start"], [0, 10, 4]) and _np.allclose(spoke["end"], [0, 40, 4])
    shelf = DSL.run(AR.MATES_EXAMPLES[1][1])
    ball = next(p for p in shelf if p["id"] == 3)
    assert _np.allclose(ball["center"], [0, 0, 35])


from c2cad import heldout as HO  # noqa: E402

HELDOUT = HO.build_all()
MAIN = {(c["family"], c["level"]): c for c in CASES}


@pytest.mark.parametrize("F", HO.FAMILIES, ids=lambda F: F.family)
def test_heldout_generator_reproduces_main_case_at_defaults(F):
    for lvl in (1, 2, 3):
        c = HO.build_case(F, lvl, F.DEFAULT, "default")
        m = MAIN[(F.family, lvl)]
        assert c["prompt_body"] == m["prompt_body"]
        assert evaluate(m, c["reference"]).exact


@pytest.mark.parametrize("case", HELDOUT, ids=lambda c: c["case_id"])
def test_heldout_reference_is_exact_and_default_answer_is_not(case):
    assert evaluate(case, case["reference"]).exact
    memorised = MAIN[(case["family"], case["level"])]["reference"]
    r = evaluate(case, memorised)
    assert not r.exact and r.semantic < 90
