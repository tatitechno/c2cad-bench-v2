#!/usr/bin/env python3
"""
Database I/O module for C2CAD-Bench V2.
Handles loading and saving the showcase database.

Merge-safe: save_db() re-reads the disk before writing, so multiple
concurrent runs (e.g. different models in separate terminals) never
overwrite each other's results.
"""

import os
import json
import fcntl
import time


def _read_disk_db(db_path):
    """Read and parse the JS-wrapped DB from disk. Returns dict or None."""
    if not os.path.exists(db_path):
        return None
    try:
        with open(db_path, "r") as f:
            content = f.read().replace("window.SHOWCASE_DB = ", "").replace(";\n", "")
        return json.loads(content)
    except Exception:
        return None


def _build_goldens(all_tests):
    """Build the golden reference list from ALL_TESTS definitions."""
    goldens = []
    for test in all_tests:
        for idx, scale in enumerate(test["scales"]):
            diff = idx + 1
            if test["phase"] in [1, 2, 4]:
                prompt, golden = test["func"](scale)
                goldens.append({
                    "family": test["family"],
                    "difficultyLabel": f"Level {diff} (Scale {scale})",
                    "difficultyID": diff,
                    "shapes": golden,
                    "prompt": prompt,
                    "phase": test["phase"]
                })
            else:
                prompt, specs = test["func"](scale)
                ref_shapes = specs.get("reference", [])
                goldens.append({
                    "family": test["family"],
                    "difficultyLabel": f"Level {diff} (Scale {scale})",
                    "difficultyID": diff,
                    "shapes": ref_shapes,
                    "prompt": prompt,
                    "phase": test["phase"]
                })
    return goldens


def _map_disk_results(disk_results, goldens):
    """Map a list of disk results to golden indices by (family, difficultyID)."""
    slots = [None] * len(goldens)
    for disk_res in (disk_results or []):
        if disk_res is None:
            continue
        for i, g in enumerate(goldens):
            if g["family"] == disk_res.get("family") and g["difficultyID"] == disk_res.get("difficultyID"):
                slots[i] = disk_res
                break
    return slots


def _zero_placeholder(golden_entry):
    """Create a zeroed-out placeholder for a golden entry."""
    return {
        "family": golden_entry["family"],
        "difficultyLabel": golden_entry["difficultyLabel"],
        "difficultyID": golden_entry["difficultyID"],
        "shapes": [],
        "score_cov": 0, "score_geom": 0, "score_sem": 0, "score_global": 0,
        "phase": golden_entry["phase"]
    }


def load_db(db_path, target_models, all_tests):
    """
    Load or create the showcase_db.js database.
    The 'models' dict is keyed by model alias (the string passed to --model).
    New model keys are added dynamically — no hard-coded model list.

    Args:
        db_path (str): Path to the database file
        target_models (list): List of model aliases to initialize
        all_tests (list): List of all test definitions (replaces global ALL_TESTS)
    """
    goldens = _build_goldens(all_tests)
    db = {"golden": goldens, "models": {}}

    # Load existing DB from disk (preserve all previous model results)
    disk_db = _read_disk_db(db_path)
    if disk_db:
        for m, results in disk_db.get("models", {}).items():
            db["models"][m] = _map_disk_results(results, goldens)

    # Ensure every target model has a slot array
    for m in target_models:
        if m not in db["models"]:
            db["models"][m] = [None] * len(goldens)

    # Fill empty slots with zeroed placeholders
    for m in db["models"]:
        while len(db["models"][m]) < len(goldens):
            db["models"][m].append(None)
        for i, res in enumerate(db["models"][m]):
            if res is None:
                db["models"][m][i] = _zero_placeholder(goldens[i])

    return db


def save_db(db, db_path):
    """Merge-safe save: re-reads the disk, merges in new results, then writes.

    This allows multiple terminals to run different models concurrently.
    For each (model, test) slot, the version with the higher score_global wins.
    A file lock prevents two processes from writing at the exact same instant.

    Args:
        db (dict): The in-memory database with new results
        db_path (str): Path where the database file should be written
    """
    goldens = db["golden"]
    lock_path = db_path + ".lock"

    # Acquire file lock so two processes don't write at the same instant
    lock_fd = open(lock_path, "w")
    for attempt in range(10):
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except (IOError, OSError):
            time.sleep(0.5)
    else:
        # After 5s of retries, force-acquire (blocking)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

    try:
        # Re-read disk to get any results written by other concurrent runs
        disk_db = _read_disk_db(db_path)

        if disk_db:
            disk_models = disk_db.get("models", {})
        else:
            disk_models = {}

        merged_models = {}

        # Collect all model keys from both disk and in-memory
        all_model_keys = set(db["models"].keys()) | set(disk_models.keys())

        for m in all_model_keys:
            mem_results  = db["models"].get(m, [])
            disk_results = _map_disk_results(disk_models.get(m, []), goldens) if m in disk_models else []

            merged = [None] * len(goldens)
            for i, g in enumerate(goldens):
                mem_entry  = mem_results[i]  if i < len(mem_results)  else None
                disk_entry = disk_results[i] if i < len(disk_results) else None

                mem_score  = (mem_entry  or {}).get("score_global", 0)
                disk_score = (disk_entry or {}).get("score_global", 0)

                # Keep whichever has the higher score (non-zero wins over zero)
                if mem_score >= disk_score:
                    merged[i] = mem_entry
                else:
                    merged[i] = disk_entry

                # Fill any remaining None with zero placeholder
                if merged[i] is None:
                    merged[i] = _zero_placeholder(g)

            merged_models[m] = merged

        # Build final DB
        final_db = {"golden": goldens, "models": merged_models}

        # Write atomically: write to temp file first, then rename
        tmp_path = db_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write("window.SHOWCASE_DB = ")
            json.dump(final_db, f, separators=(',', ':'))
            f.write(";\n")
        os.replace(tmp_path, db_path)  # atomic on POSIX

        print(f"\n✅ Database saved (merge-safe) → {db_path}")

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            os.remove(lock_path)
        except OSError:
            pass

    # Split into per-phase files for lazy loading in the visualiser
    try:
        import importlib.util
        splitter_path = os.path.join(os.path.dirname(__file__), "split_database.py")
        spec = importlib.util.spec_from_file_location("split_database", splitter_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.split()
    except Exception as e:
        print(f"⚠  Phase split warning: {e}")
