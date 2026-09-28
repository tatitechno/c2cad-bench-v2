"""
Phase 3 Semantic runner — live LLM inference for Furniture Assembly,
Machined Enclosure, and Axle Bearing semantic constraint challenges.

Usage:
    python runners/run_v3_semantic_benchmark.py
"""
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

base_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_dir, '../..')))
sys.path.append(os.path.abspath(os.path.join(base_dir, '../stages/phase3_semantic')))

from probe.llm import call_llm
from probe.validators import (extract_json, validate_gravity,
                               validate_wall_thickness, validate_clearance_fit,
                               validate_interference, validate_mates)

from generate_human_furniture import generate_furniture
from generate_human_enclosure import generate_enclosure
from generate_human_axle import generate_axle

MODELS = ["gemini-2.5-flash", "gemini-3-flash", "gemini-3-flash-lite"]

V3_TESTS = [
    {"family": "Furniture Assembly", "func": generate_furniture, "scales": [2, 3, 4]},
    {"family": "Machined Enclosure", "func": generate_enclosure, "scales": [2, 3, 5]},
    {"family": "Axle Bearing",       "func": generate_axle,      "scales": [1, 2, 3]},
]


def eval_semantic_constraints(shapes, ops, specs):
    score_points = 0
    total_checks = 0
    details = []

    if "gravity_check" in specs:
        total_checks += 1
        res = validate_gravity(shapes)
        if res.get("all_supported", False):
            score_points += 1
        details.append(f"Gravity: {'OK' if res.get('all_supported', False) else 'FAIL'}")

    if "mates" in specs:
        res = validate_mates(shapes, specs["mates"])
        checked = res.get("checked", 0)
        correct = res.get("correct", 0)
        total_checks += checked
        score_points += correct
        details.extend(res.get("details", []))

    if "wall_thickness" in specs:
        res = validate_wall_thickness(shapes, specs["wall_thickness"])
        checked    = res.get("checked", 0)
        violations = res.get("violations", 0)
        total_checks += checked
        score_points += (checked - violations)
        details.extend(res.get("details", []))

    if "clearance_fit" in specs:
        res = validate_clearance_fit(shapes, specs["clearance_fit"])
        checked = res.get("checked", 0)
        correct = res.get("correct", 0)
        total_checks += checked
        score_points += correct
        details.extend(res.get("details", []))

    if "interference_check" in specs:
        total_checks += 1
        res = validate_interference(shapes, ops, tol_mm=0.2)
        if res.get("interference_free", False):
            score_points += 1
        else:
            details.extend(res.get("details", []))

    final_score = (score_points / total_checks) if total_checks > 0 else 0.0
    return final_score, details


def run_single_api(model, prompt):
    max_retries = 3
    delay = 10
    for attempt in range(max_retries):
        try:
            return call_llm(prompt, model, timeout=120)
        except Exception as e:
            if "exhausted" in str(e).lower() or "429" in str(e):
                time.sleep(delay)
                delay *= 2
            else:
                return "Crash"
    return "Crash"


def process_task(model, family, scale, prompt, specs):
    llm_resp = run_single_api(model, prompt)
    if llm_resp == "Crash":
        return model, family, scale, 0.0, ["API Crash"]

    raw = extract_json(llm_resp)
    if not raw:
        return model, family, scale, 0.0, ["Parse Fail"]

    shapes = [r for r in raw if r.get("type") in ["box", "cylinder", "sphere", "pipe", "beam"]]
    ops    = [r for r in raw if "op" in r]

    try:
        final_score, details = eval_semantic_constraints(shapes, ops, specs)
        return model, family, scale, final_score, details
    except Exception as e:
        return model, family, scale, 0.0, [f"Eval Error: {str(e)[:40]}"]


def run_tasks():
    print("Starting Semantic V3 Benchmark...")
    tasks = []

    for m in MODELS:
        for t in V3_TESTS:
            for scale in t["scales"]:
                prompt, specs = t["func"](scale)
                tasks.append((m, t["family"], scale, prompt, specs))

    results_dict = {m: {} for m in MODELS}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures   = [executor.submit(process_task, *args) for args in tasks]
        completed, total = 0, len(tasks)
        for future in as_completed(futures):
            model, family, scale, score, details = future.result()
            completed += 1
            status = f"Score: {score*100:.0f}%"
            print(f"[{completed:02d}/{total}] {model[:12]} | {family} Scale {scale} -> {status}")
            if details:
                for d in details[:3]:        # show first 3 detail lines
                    print(f"          {d}")
            results_dict[model][f"{family}_{scale}"] = status

    lines = ["# V3 Semantic Constraints Benchmark Results",
             "\n*Evaluates human-intent relational logic without rigid Golden coordinates. "
             "Tests DFM clearances, wall thicknesses, spatial mates, and interference-free boolean logic.*",
             "\n| Semantic Structure Challenge | " + " | ".join(MODELS) + " |",
             "|---| " + " | ".join(["---"] * len(MODELS)) + " |"]

    for t in V3_TESTS:
        for scale in t["scales"]:
            row_title = f"{t['family']} Scale {scale}"
            rower = f"| **{row_title}** |"
            for m in MODELS:
                status = results_dict[m].get(f"{t['family']}_{scale}", "Missing")
                rower += f" {status} |"
            lines.append(rower)

    out_file = os.path.join(base_dir, "..", "results", "v3_semantic_results.md")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    run_tasks()
