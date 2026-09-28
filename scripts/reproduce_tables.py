#!/usr/bin/env python3
"""Regenerate lightweight markdown tables from released CSV artifacts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reproduced_tables.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def model_rankings() -> str:
    rows = read_csv(ROOT / "data" / "model_summary.csv")
    rows.sort(key=lambda row: float(row["mean_global"]), reverse=True)
    lines = [
        "## Model Rankings",
        "",
        "| Rank | Model | Mean Global | Std |",
        "| ---: | --- | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | `{row['model_id']}` | {float(row['mean_global']):.2f} | {float(row['std_global']):.2f} |"
        )
    return "\n".join(lines)


def phase_table() -> str:
    rows = read_csv(ROOT / "data" / "scores.csv")
    by_model_phase: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_model_phase[(row["model_id"], row["phase"])].append(float(row["score_global"]))
    models = sorted({model for model, _ in by_model_phase})
    phases = sorted({phase for _, phase in by_model_phase}, key=lambda p: int(float(p)))
    lines = [
        "## Phase Means",
        "",
        "| Model | " + " | ".join(f"P{phase}" for phase in phases) + " |",
        "| --- | " + " | ".join("---:" for _ in phases) + " |",
    ]
    for model in models:
        values = []
        for phase in phases:
            vals = by_model_phase[(model, phase)]
            values.append(f"{sum(vals) / len(vals):.2f}" if vals else "")
        lines.append(f"| `{model}` | " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    OUT.write_text(
        model_rankings() + "\n\n" + phase_table() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
