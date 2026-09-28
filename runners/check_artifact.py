"""Lightweight integrity checks for the C2CAD-Bench release artifact."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "results" / "showcase_db.js"
CASES_PATH = ROOT / "data" / "cases.jsonl"
OUTPUTS_PATH = ROOT / "data" / "model_outputs.jsonl"
SCORES_PATH = ROOT / "data" / "scores.csv"
SUMMARY_PATH = ROOT / "data" / "model_summary.csv"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return records


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    cases = _load_jsonl(CASES_PATH)
    outputs = _load_jsonl(OUTPUTS_PATH)
    scores = _load_csv(SCORES_PATH)
    summary = _load_csv(SUMMARY_PATH)

    text = DB_PATH.read_text(encoding="utf-8")
    match = re.search(r"window\.SHOWCASE_DB\s*=\s*(\{.*\});?\s*$", text, re.S)
    if not match:
        raise SystemExit(f"Could not parse {DB_PATH}")

    db = json.loads(match.group(1))
    golden = db.get("golden", [])
    models = db.get("models", {})
    result_count = sum(len(v) for v in models.values())

    print(f"{len(golden)} golden test cases")
    print(f"{len(models)} models")
    print(f"{result_count} model-case results")

    expected = (75, 13, 975)
    actual = (len(golden), len(models), result_count)
    if actual != expected:
        raise SystemExit(f"Unexpected artifact size: expected {expected}, got {actual}")

    if (len(cases), len(outputs), len(scores), len(summary)) != (75, 975, 975, 13):
        raise SystemExit(
            "Unexpected data table sizes: "
            f"cases={len(cases)}, outputs={len(outputs)}, scores={len(scores)}, summary={len(summary)}"
        )

    case_ids = {case["case_id"] for case in cases}
    if len(case_ids) != len(cases):
        raise SystemExit("Duplicate case_id values in data/cases.jsonl")
    if {row["case_id"] for row in outputs} != case_ids:
        raise SystemExit("data/model_outputs.jsonl case_id set does not match data/cases.jsonl")

    for case in cases:
        shapes = json.loads(case["golden_shapes_json"])
        if len(shapes) != int(case["expected_shape_count"]):
            raise SystemExit(
                f"{case['case_id']}: expected_shape_count={case['expected_shape_count']} "
                f"but golden_shapes_json has {len(shapes)} shapes"
            )


if __name__ == "__main__":
    main()
