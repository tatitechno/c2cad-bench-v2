"""
Phase 2 Advanced runner — live LLM inference for Suspension Bridge,
Planetary Array, Cross-Braced Truss, Fractal Y-Tree, BCC Lattice.

Usage:
    python runners/run_advanced_live.py
    C2CAD_DEBUG=1 python runners/run_advanced_live.py  # verbose LLM output
"""
import sys
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_dir, '../..')))
sys.path.append(os.path.abspath(os.path.join(base_dir, '../stages/phase2_advanced')))

from probe.llm import call_llm
from probe.validators import extract_json
from build_database import compute_score

from generate_bridge_v2 import generate_bridge
from generate_planetary_v2 import generate_planetary
from generate_truss_v2 import generate_truss
from generate_fractal_v2 import generate_fractal
from generate_bcc_v2 import generate_bcc

MODELS = ["gemini-2.5-flash", "gemini-3-flash", "gemini-3-flash-lite"]

ADVANCED_TESTS = [
    {"family": "Suspension Bridge",  "func": generate_bridge,    "scales": [5, 10, 20]},
    {"family": "Planetary Array",    "func": generate_planetary, "scales": [6, 12, 18]},
    {"family": "Cross-Braced Truss", "func": generate_truss,     "scales": [2, 4, 8]},
    {"family": "Fractal Y-Tree",     "func": generate_fractal,   "scales": [2, 3, 4]},
    {"family": "BCC Lattice",        "func": generate_bcc,       "scales": [2, 3, 4]},
]


def run_single_api(model, prompt):
    """Call LLM with exponential back-off to handle rate limits."""
    clean_prompt = prompt + "\nOutput exactly raw JSON representing these shapes without any markdown blocks or explanations."
    max_retries = 3
    delay = 10
    for attempt in range(max_retries):
        try:
            return call_llm(clean_prompt, model, timeout=120)
        except Exception as e:
            if "exhausted" in str(e).lower() or "429" in str(e) or "quota" in str(e).lower():
                time.sleep(delay)
                delay *= 2
            else:
                return "Crash"
    return "Crash"


def process_task(model, family, scale, diff, prompt, golden):
    DEBUG = os.environ.get("C2CAD_DEBUG", os.environ.get("CG3D_DEBUG", "0")) == "1"
    llm_resp = run_single_api(model, prompt)
    if llm_resp == "Crash":
        return model, family, scale, diff, [], 0.0, 0.0, 0.0

    if DEBUG:
        preview = (llm_resp or "")[:300].replace("\n", "\\n")
        print(f"  [DEBUG] {model} | {family} L{diff} | len={len(llm_resp or '')} | {preview}")

    shapes = extract_json(llm_resp)
    if not shapes:
        return model, family, scale, diff, [], 0.0, 0.0, 0.0

    cov, geom, phys = compute_score(shapes, golden)
    return model, family, scale, diff, shapes, cov, geom, phys


def run_tasks():
    print("Starting Live API Inference — Phase 2 Advanced Structural Suite...")

    # Pre-compute golden JSONs
    db = {"golden": [], "models": {m: [] for m in MODELS}}
    golden_map = {}

    for test in ADVANCED_TESTS:
        for idx, scale in enumerate(test["scales"]):
            diff = idx + 1
            prompt, golden = test["func"](scale)
            suite_meta = {
                "family":         test["family"],
                "difficultyLabel": f"Level {diff} (Scale {scale})",
                "difficultyID":    diff,
                "shapes":          golden,
                "prompt":          prompt,
            }
            db["golden"].append(suite_meta)
            golden_map[f"{test['family']}_{scale}"] = (prompt, golden, diff)

    tasks = []
    for m in MODELS:
        for test in ADVANCED_TESTS:
            for scale in test["scales"]:
                tkey = f"{test['family']}_{scale}"
                prompt, golden, diff = golden_map[tkey]
                tasks.append((m, test["family"], scale, diff, prompt, golden))

    sim_outputs = {m: {} for m in MODELS}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_task, *args) for args in tasks]
        completed, total = 0, len(tasks)
        for future in as_completed(futures):
            model, family, scale, diff, extracted_shapes, cov, geom, phys = future.result()
            completed += 1
            print(f"[{completed:02d}/{total}] {model[:12]} | {family} L{diff} -> "
                  f"Cov:{cov*100:.0f}% Geom:{geom*100:.0f}% Phys:{phys*100:.0f}%")
            sim_outputs[model][f"{family}_{scale}"] = {
                "family":          family,
                "difficultyLabel": f"Level {diff} (Scale {scale})",
                "difficultyID":    diff,
                "shapes":          extracted_shapes,
                "score_cov":       round(cov  * 100),
                "score_geom":      round(geom * 100),
                "score_phys":      round(phys * 100),
            }

    # Reconstruct DB preserving golden order
    for m in MODELS:
        for m_golden in db["golden"]:
            try:
                sk = f"{m_golden['family']}_{m_golden['difficultyLabel'].split('Scale ')[1].replace(')', '')}"
                db["models"][m].append(sim_outputs[m][sk])
            except Exception:
                db["models"][m].append({
                    "family":          m_golden["family"],
                    "difficultyLabel": m_golden["difficultyLabel"],
                    "shapes": [], "score_cov": 0, "score_geom": 0, "score_phys": 0,
                })

    out_file = os.path.join(base_dir, "..", "results", "showcase_db.js")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write("window.SHOWCASE_DB = ")
        json.dump(db, f, separators=(',', ':'))
        f.write(";\n")

    print(f"\nLive results written to {out_file}")
    print("Open ui/visualizer.html to inspect hallucination patterns.")


if __name__ == "__main__":
    run_tasks()
