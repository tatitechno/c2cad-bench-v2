#!/usr/bin/env python3
"""Validate Croissant metadata links and file hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROISSANT = ROOT / "croissant.json"
DEFAULT_RAW_PREFIX = "https://raw.githubusercontent.com/c2cad-bench/c2cad-bench/main/"
DEFAULT_REPO_URL = "https://github.com/c2cad-bench/c2cad-bench"
CORE_SPEC_URI = "http://mlcommons.org/croissant/1.0"
RAI_SPEC_URI = "http://mlcommons.org/croissant/RAI/1.0"
RAI_NAMESPACE = "http://mlcommons.org/croissant/RAI/"
REQUIRED_RAI_FIELDS = (
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
)


def md5(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() not in {".png", ".pdf", ".zip"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.md5(data).hexdigest()


def as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def has_content(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(has_content(item) for item in value)
    return value is not None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--croissant", type=Path, default=DEFAULT_CROISSANT)
    parser.add_argument("--raw-prefix", default=DEFAULT_RAW_PREFIX)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    args = parser.parse_args(argv)

    meta = json.loads(args.croissant.read_text(encoding="utf-8"))
    errors: list[str] = []

    context = meta.get("@context", {})
    conforms_to = as_list(meta.get("conformsTo") or meta.get("dct:conformsTo"))

    if meta.get("url") != args.repo_url:
        errors.append(f"url should be {args.repo_url!r}, got {meta.get('url')!r}")
    if meta.get("citeAs") != args.repo_url:
        errors.append(f"citeAs should be {args.repo_url!r}, got {meta.get('citeAs')!r}")
    if not isinstance(context, dict):
        errors.append("@context should be a JSON object")
    elif context.get("rai") != RAI_NAMESPACE:
        errors.append(f"@context.rai should be {RAI_NAMESPACE!r}, got {context.get('rai')!r}")
    if CORE_SPEC_URI not in conforms_to:
        errors.append(f"conformsTo should include {CORE_SPEC_URI!r}")
    if RAI_SPEC_URI not in conforms_to:
        errors.append(f"conformsTo should include {RAI_SPEC_URI!r}")

    for field in REQUIRED_RAI_FIELDS:
        if field not in meta:
            errors.append(f"missing required RAI field {field}")
            continue
        if not has_content(meta[field]):
            errors.append(f"required RAI field {field} is empty")
    if "rai:hasSyntheticData" in meta and not isinstance(meta["rai:hasSyntheticData"], bool):
        errors.append("rai:hasSyntheticData should be a boolean")

    for entry in meta.get("distribution", []):
        name = entry.get("name")
        if not name:
            errors.append("distribution entry missing name")
            continue
        local = ROOT / name
        if not local.exists():
            errors.append(f"{name}: local file missing")
            continue
        actual_hash = md5(local)
        expected_hash = entry.get("md5")
        if actual_hash != expected_hash:
            errors.append(f"{name}: md5 mismatch metadata={expected_hash} actual={actual_hash}")
        expected_url = args.raw_prefix + name.replace("\\", "/")
        if entry.get("contentUrl") != expected_url:
            errors.append(f"{name}: contentUrl should be {expected_url!r}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Croissant metadata is consistent with local files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
