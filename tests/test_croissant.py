import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_croissant_metadata_is_current():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_croissant.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
