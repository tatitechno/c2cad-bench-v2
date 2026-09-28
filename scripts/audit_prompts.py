#!/usr/bin/env python3
"""Audit C2CAD-Bench prompts for scaffolding signals.

The audit is intentionally conservative: it flags patterns that deserve human
review, not proof that a prompt is invalid. Use --strict when curating a
zero-scaffold split where any high-risk flag should fail CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "data" / "cases.jsonl"
DEFAULT_OUT = ROOT / "data" / "prompt_audit.csv"


NUMERIC_VECTOR_RE = re.compile(r"\[[\s0-9.,+\-eE]+\]")
ASSIGNMENT_RE = re.compile(
    r"\b(?:axis|centre|center|rod centre|rod center)\s*=",
    re.IGNORECASE,
)
SHAPE_COUNT_RE = re.compile(
    r"(?:total shapes|total\s+\d+\s+shapes|\(total\s+\d+|exactly\s+\d+\s+(?:shapes|steps|pipes|beams|spheres|cylinders|cones|tori))",
    re.IGNORECASE,
)
FORMULA_HINT_RE = re.compile(
    r"(?:cartesian|sin\(|cos\(|sqrt|phi|normalis[ez]e|divide by|computed as|formula)",
    re.IGNORECASE,
)
AXIS_ANCHOR_RE = re.compile(
    r"(?:origin|positive x-axis|negative x-axis|positive y-axis|negative y-axis|z-axis|axis\s*=\s*\[)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AuditRow:
    case_id: str
    family: str
    phase: int
    difficulty_id: int
    expected_shape_count: int
    explicit_shape_count: bool
    coordinate_vector: bool
    formula_or_assignment: bool
    trig_or_cartesian_hint: bool
    origin_axis_anchor: bool

    @property
    def high_risk_flag(self) -> bool:
        return (
            self.explicit_shape_count
            or self.coordinate_vector
            or self.formula_or_assignment
            or self.trig_or_cartesian_hint
        )

    def as_dict(self) -> dict[str, object]:
        data = {
            "case_id": self.case_id,
            "family": self.family,
            "phase": self.phase,
            "difficulty_id": self.difficulty_id,
            "expected_shape_count": self.expected_shape_count,
            "explicit_shape_count": int(self.explicit_shape_count),
            "coordinate_vector": int(self.coordinate_vector),
            "formula_or_assignment": int(self.formula_or_assignment),
            "trig_or_cartesian_hint": int(self.trig_or_cartesian_hint),
            "origin_axis_anchor": int(self.origin_axis_anchor),
            "high_risk_flag": int(self.high_risk_flag),
        }
        return data


def load_cases(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def audit_case(case: dict) -> AuditRow:
    prompt = case.get("prompt", "")
    return AuditRow(
        case_id=str(case.get("case_id", "")),
        family=str(case.get("family", "")),
        phase=int(case.get("phase", 0)),
        difficulty_id=int(case.get("difficulty_id", 0)),
        expected_shape_count=int(case.get("expected_shape_count", 0)),
        explicit_shape_count=bool(SHAPE_COUNT_RE.search(prompt)),
        coordinate_vector=bool(NUMERIC_VECTOR_RE.search(prompt)),
        formula_or_assignment=bool(ASSIGNMENT_RE.search(prompt)),
        trig_or_cartesian_hint=bool(FORMULA_HINT_RE.search(prompt)),
        origin_axis_anchor=bool(AXIS_ANCHOR_RE.search(prompt)),
    )


def write_csv(rows: list[AuditRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].as_dict().keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def summarize(rows: list[AuditRow]) -> str:
    total = len(rows)
    high = sum(row.high_risk_flag for row in rows)
    counts = {
        "explicit_shape_count": sum(row.explicit_shape_count for row in rows),
        "coordinate_vector": sum(row.coordinate_vector for row in rows),
        "formula_or_assignment": sum(row.formula_or_assignment for row in rows),
        "trig_or_cartesian_hint": sum(row.trig_or_cartesian_hint for row in rows),
        "origin_axis_anchor": sum(row.origin_axis_anchor for row in rows),
    }
    lines = [
        f"Audited {total} prompts",
        f"High-risk scaffolding flags: {high}",
    ]
    lines.extend(f"{name}: {value}" for name, value in counts.items())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any high-risk scaffolding signal is present.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the summary without writing CSV.",
    )
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    rows = [audit_case(case) for case in cases]
    if not args.summary_only:
        write_csv(rows, args.out)
    print(summarize(rows))

    if args.strict and any(row.high_risk_flag for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
