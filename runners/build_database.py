#!/usr/bin/env python3
"""
Build the showcase_db.js database with simulated LLM failures for demo/testing.
Includes Gemini, Claude, and DeepSeek model entries.

Usage:
    python build_database.py               # Build with all default models
    python build_database.py --models gemini-2.5-flash claude-sonnet deepseek   # Specific models only
"""

import sys
import os
import json
import argparse

base_dir = os.path.dirname(os.path.abspath(__file__))
# V2 stage paths MUST come before probe root to avoid V1 generator shadowing
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '../stages/phase1_basic')))
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '../stages/phase2_advanced')))
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '../stages/phase3_semantic')))
sys.path.insert(0, os.path.abspath(os.path.join(base_dir, '../stages/phase4_bioinspired')))
sys.path.append(os.path.abspath(os.path.join(base_dir, '../..')))

from probe.validators import validate_geometry, validate_gravity, validate_connectivity, validate_interference
from probe.validators import validate_wall_thickness, validate_clearance_fit, validate_mates

from generate_staircase import generate_staircase
from generate_pyramid import generate_pyramid
from generate_rubiks import generate_rubiks
from generate_stonehenge import generate_stonehenge
from generate_dna import generate_dna
from generate_bridge_v2 import generate_bridge
from generate_planetary_v2 import generate_planetary
from generate_truss_v2 import generate_truss
from generate_fractal_v2 import generate_fractal
from generate_bcc_v2 import generate_bcc
from generate_human_furniture import generate_furniture
from generate_human_enclosure import generate_enclosure
from generate_human_axle import generate_axle
from generate_compound_eye import generate_compound_eye
from generate_diatom import generate_diatom

from generate_showcase_failures import degrade_shapes

TESTS = [
    # Phase 1 — Basic Parametric Scaling
    {"family": "Spiral Staircase",   "func": generate_staircase, "scales": [10, 24, 50],  "phase": 1},
    {"family": "Cannonball Pyramid",  "func": generate_pyramid,   "scales": [3, 4, 5],     "phase": 1},
    {"family": "Voxel Grid",          "func": generate_rubiks,    "scales": [2, 3, 4],     "phase": 1},
    {"family": "Domino Ring",         "func": generate_stonehenge,"scales": [5, 10, 20],   "phase": 1},
    {"family": "DNA Helix",           "func": generate_dna,       "scales": [5, 10, 20],   "phase": 1},
    # Phase 2 — Advanced Hidden-Formula
    {"family": "Suspension Bridge",   "func": generate_bridge,    "scales": [5, 10, 20],   "phase": 2},
    {"family": "Planetary Array",     "func": generate_planetary,  "scales": [6, 12, 18],   "phase": 2},
    {"family": "Cross-Braced Truss",  "func": generate_truss,     "scales": [2, 4, 8],     "phase": 2},
    {"family": "Fractal Y-Tree",      "func": generate_fractal,   "scales": [2, 3, 4],     "phase": 2},
    {"family": "BCC Lattice",         "func": generate_bcc,       "scales": [2, 3, 4],     "phase": 2},
    # Phase 3 — Semantic Constraint Satisfaction
    {"family": "Furniture Assembly",  "func": generate_furniture,  "scales": [2, 3, 4],     "phase": 3},
    {"family": "Machined Enclosure",  "func": generate_enclosure,  "scales": [2, 3, 5],     "phase": 3},
    {"family": "Axle Bearing",         "func": generate_axle,       "scales": [1, 2, 3],     "phase": 3},
    # Phase 4 — Bio-Inspired
    {"family": "Compound Eye",        "func": generate_compound_eye, "scales": [2, 3, 4],   "phase": 4},
    {"family": "Diatom Frustule",     "func": generate_diatom,       "scales": [4, 7, 10],  "phase": 4},
]

# Model simulation profiles: (model_alias, severity_offset)
# severity_offset is added to the base difficulty level.
# Higher = more degraded = worse scores.
# Negative = better than baseline.
DEFAULT_MODELS = {
    # Exact API model IDs — no alias translation
    # severity_offset: higher = more degraded (worse), negative = better than baseline

    # Gemini family (Google Generative Language API)
    "gemini-2.5-flash":              2,    # Older, weaker
    "gemini-2.5-pro":                0,    # Strong reasoning model
    "gemini-3.1-flash-lite-preview": 1,    # Medium
    "gemini-3-flash-preview":        0,    # Good
    "gemini-3.1-pro-preview":       -1,    # Best Gemini

    # Claude family (Anthropic Messages API)
    "claude-opus-4-6":              -1,    # Most capable Claude
    "claude-sonnet-4-6":             0,    # Strong reasoning

    # DeepSeek family (DeepSeek API)
    # deepseek-chat     = DeepSeek-V3.2 (non-thinking mode)
    # deepseek-reasoner = DeepSeek-R1   (thinking / chain-of-thought)
    "deepseek-chat":                 1,    # DeepSeek V3.2
    "deepseek-reasoner":             0,    # DeepSeek R1 — strong reasoning

    # OpenAI family (OpenAI API — requires OPENAI_API_KEY)
    "gpt-4.1":                       0,    # GPT-4.1 — strong instruction following
    "gpt-5.4":                      -1,    # GPT-5.4 — flagship reasoning
    "gpt-5.4-mini":                  1,    # GPT-5.4 Mini — fast/cheap
    "gpt-5.4-pro":                  -1,    # GPT-5.4 Pro — max reasoning

    # Kimi family (Moonshot AI API — requires MOONSHOT_API_KEY)
    "kimi-k2.5":                     0,    # Kimi K2.5 — 1T MoE, strong reasoning
}


def compute_score(llm_shapes, golden_json):
    """Compute simulated Cov/Geom scores for demo DB.
    Sem and Global are simulated based on severity in the caller."""
    expected_count = len(golden_json)
    actual_count   = len(llm_shapes)
    coverage = min(1.0, actual_count / float(max(1, expected_count)))

    try:
        check_ids = [s["id"] for s in llm_shapes if "id" in s and type(s["id"]) == int]
        check_ids = [i for i in check_ids if any(g["id"] == i for g in golden_json)]
        geom_res = validate_geometry(llm_shapes, golden_json, check_ids, tol_mm=2.0)
        pos_correct = geom_res.get("positions_correct", 0)
        checked     = geom_res.get("positions_checked", 1)
        pos_accuracy = pos_correct / float(checked) if checked > 0 else 0.0
        dim_errors   = geom_res.get("dim_errors", [])
        dim_accuracy = 1.0 - (sum(dim_errors) / len(dim_errors)) if len(dim_errors) > 0 else 0.0
        final_geom_score = (pos_accuracy + dim_accuracy) / 2.0
    except Exception:
        final_geom_score = 0.0

    return coverage, final_geom_score


def build_database(model_filter=None):
    model_profiles = DEFAULT_MODELS
    if model_filter:
        model_profiles = {k: v for k, v in DEFAULT_MODELS.items() if k in model_filter}
        if not model_profiles:
            print(f"No matching models found. Available: {', '.join(DEFAULT_MODELS.keys())}")
            return

    model_list = list(model_profiles.keys())
    print(f"Building showcase_db.js for {len(model_list)} models: {', '.join(model_list)}")
    print(f"Total tests: {len(TESTS)} families x 3 scales = {len(TESTS) * 3} test cases\n")

    db = {"golden": [], "models": {m: [] for m in model_list}}

    for test in TESTS:
        for idx, scale in enumerate(test["scales"]):
            diff = idx + 1
            phase = test["phase"]

            if phase in [1, 2, 4]:
                prompt, golden = test["func"](scale)
                golden_shapes = golden
            else:
                prompt, specs = test["func"](scale)
                golden_shapes = specs.get("reference", [])

            print(f"  Phase {phase} | {test['family']:22s} L{diff} (scale={scale}) — {len(golden_shapes)} shapes")

            db["golden"].append({
                "family": test["family"],
                "difficultyLabel": f"Level {diff} (Scale {scale})",
                "difficultyID": diff,
                "shapes": golden_shapes,
                "prompt": prompt,
                "phase": phase,
            })

            for m in model_list:
                severity_offset = model_profiles[m]
                failure_severity = max(1, min(5, diff + severity_offset))

                if phase in [1, 2, 4]:
                    simulated_shapes = degrade_shapes(golden_shapes, failure_severity)
                    cov, geom = compute_score(simulated_shapes, golden_shapes)
                    cov_pct  = round(cov  * 100)
                    geom_pct = round(geom * 100)
                else:
                    simulated_shapes = golden_shapes  # Use reference shapes for visualization
                    cov_pct  = max(0, min(100, 100 - failure_severity * 8))
                    geom_pct = max(0, min(100, 100 - failure_severity * 12))

                # Simulate Sem based on severity (lower severity = better)
                sem_pct = max(0, min(100, 100 - failure_severity * 16))
                # Global = Cov*20% + Geom*30% + Sem*50%
                glob_pct = round(cov_pct * 0.20 + geom_pct * 0.30 + sem_pct * 0.50)

                db["models"][m].append({
                    "family":          test["family"],
                    "difficultyLabel": f"Level {diff} (Scale {scale})",
                    "difficultyID":    diff,
                    "shapes":          simulated_shapes,
                    "score_cov":       cov_pct,
                    "score_geom":      geom_pct,
                    "score_sem":       sem_pct,
                    "score_global":    glob_pct,
                    "phase":           phase,
                })

    out_file = os.path.join(os.path.dirname(__file__), "..", "results", "showcase_db.js")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write("window.SHOWCASE_DB = ")
        json.dump(db, f, separators=(',', ':'))
        f.write(";\n")

    print(f"\n  Database written to {out_file}")
    print(f"  Models: {', '.join(model_list)}")
    print(f"  Golden entries: {len(db['golden'])}")
    for m in model_list:
        count = len([d for d in db['models'][m] if d])
        print(f"  {m}: {count} test results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the C2CAD-Bench demo database")
    parser.add_argument("--models", nargs="+", help="Specific models to include (default: all)")
    args = parser.parse_args()

    build_database(model_filter=args.models)

    # Auto-split into per-phase files after building
    try:
        import importlib.util, os
        splitter_path = os.path.join(os.path.dirname(__file__), "split_database.py")
        spec = importlib.util.spec_from_file_location("split_database", splitter_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.split()
    except Exception as e:
        print(f"[build_database] Warning: phase split failed: {e}")
