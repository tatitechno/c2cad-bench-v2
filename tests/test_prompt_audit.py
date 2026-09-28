import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_audit_generates_expected_rows(tmp_path):
    out = tmp_path / "prompt_audit.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_prompts.py"),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = list(csv.DictReader(out.open(encoding="utf-8")))

    assert "Audited 75 prompts" in result.stdout
    assert len(rows) == 75
    assert any(row["high_risk_flag"] == "1" for row in rows)
