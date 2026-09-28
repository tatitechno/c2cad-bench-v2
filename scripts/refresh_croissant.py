#!/usr/bin/env python3
"""Refresh Croissant repository URLs and MD5 hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CROISSANT = ROOT / "croissant.json"
REPO_URL = "https://github.com/c2cad-bench/c2cad-bench"
RAW_PREFIX = "https://raw.githubusercontent.com/c2cad-bench/c2cad-bench/main/"


def md5(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() not in {".png", ".pdf", ".zip"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.md5(data).hexdigest()


def main() -> int:
    meta = json.loads(CROISSANT.read_text(encoding="utf-8"))
    meta["url"] = REPO_URL
    meta["citeAs"] = REPO_URL
    for entry in meta.get("distribution", []):
        name = entry["name"]
        entry["contentUrl"] = RAW_PREFIX + name.replace("\\", "/")
        entry["md5"] = md5(ROOT / name)
    CROISSANT.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Refreshed {CROISSANT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
