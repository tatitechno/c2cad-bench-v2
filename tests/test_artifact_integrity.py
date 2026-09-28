import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_release_artifact_counts():
    cases = _jsonl(ROOT / "data" / "cases.jsonl")
    outputs = _jsonl(ROOT / "data" / "model_outputs.jsonl")
    scores = list(csv.DictReader((ROOT / "data" / "scores.csv").open(encoding="utf-8")))
    summary = list(csv.DictReader((ROOT / "data" / "model_summary.csv").open(encoding="utf-8")))

    assert len(cases) == 75
    assert len(outputs) == 975
    assert len(scores) == 975
    assert len(summary) == 13


def test_case_ids_are_unique_and_referenced():
    cases = _jsonl(ROOT / "data" / "cases.jsonl")
    outputs = _jsonl(ROOT / "data" / "model_outputs.jsonl")
    case_ids = {case["case_id"] for case in cases}

    assert len(case_ids) == len(cases)
    assert {row["case_id"] for row in outputs} == case_ids


def test_golden_references_parse_and_match_counts():
    for case in _jsonl(ROOT / "data" / "cases.jsonl"):
        shapes = json.loads(case["golden_shapes_json"])
        assert isinstance(shapes, list), case["case_id"]
        assert len(shapes) == int(case["expected_shape_count"]), case["case_id"]
        for shape in shapes:
            assert isinstance(shape, dict)
            assert shape.get("type") in {"box", "cylinder", "sphere", "cone", "torus", "pipe", "beam"}


def test_model_outputs_parse():
    for row in _jsonl(ROOT / "data" / "model_outputs.jsonl"):
        shapes = json.loads(row["output_shapes_json"])
        assert isinstance(shapes, list)
        for score_key in ("score_cov", "score_geom", "score_sem", "score_global"):
            value = float(row[score_key])
            assert 0.0 <= value <= 100.0
