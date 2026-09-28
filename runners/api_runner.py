#!/usr/bin/env python3
"""
API runner module for C2CAD-Bench.
Handles LLM invocation with advanced retry logic for rate limits and errors.
"""

import os
import time
import threading
from probe.llm import call_llm
from probe.validators import extract_json
from scoring_utils import _normalize_shapes, eval_cov_geom
from semantic_evaluators import eval_sem, eval_global


# ── Per-provider concurrency semaphores ──────────────────────────────────────
# The global thread pool can have up to 20 workers, but each provider only
# allows N concurrent API calls at once.  Semaphores enforce this without
# slowing down other providers.
#
#   Global pool: 5 workers max.
#   Claude:   4  — thinking disabled + 4K max_tokens = low TPM
#   DeepSeek: 4  — moderate rate limits
#   Kimi:     5  — thinking disabled, faster now
#   Others:   5  — capped at global pool size
# ─────────────────────────────────────────────────────────────────────────────
_PROVIDER_SEMAPHORES = {
    "claude":   threading.Semaphore(4),
    "deepseek": threading.Semaphore(4),
    "kimi":     threading.Semaphore(5),
    "openai":   threading.Semaphore(5),
    "gemini":   threading.Semaphore(5),
    "mistral":  threading.Semaphore(5),
    "groq":     threading.Semaphore(5),
}

def _provider_of(model_id):
    """Detect provider from model ID string."""
    if model_id.startswith("claude"):   return "claude"
    if model_id.startswith("gemini"):   return "gemini"
    if model_id.startswith("deepseek"): return "deepseek"
    if model_id.startswith("kimi") or model_id.startswith("moonshot"): return "kimi"
    if any(model_id.startswith(p) for p in ("gpt", "o1", "o3", "o4", "openai")): return "openai"
    if model_id.startswith("mistral") or model_id.startswith("codestral"): return "mistral"
    if model_id.startswith("groq"):     return "groq"
    return "other"


_JSON_SUFFIX = """

CRITICAL OUTPUT RULES — read carefully:
1. Your ENTIRE response must be valid JSON: either a raw array [ {...}, {...}, ... ] or an object {"shapes": [ {...}, {...}, ... ]}.
2. Do NOT wrap in ```json``` or any markdown fence.
3. Do NOT include ANY text before or after the JSON — no explanations, no commentary, no "Here is the result", no thinking.
4. Do NOT use trailing commas in the JSON.
5. Every shape object in the array MUST have "id" (integer), "type" (string), and positional fields.
6. All coordinates must be arrays [x, y, z], NOT objects {"x":..., "y":..., "z":...}."""


def run_single_api(model, prompt):
    """Call an LLM with retry logic. Returns the raw text response.

    Acquires a per-provider semaphore before each API call so multiple
    providers can run concurrently at their own limits without blocking
    each other.

    Intelligent backoff:
    - "concurrent" or "connections" in [ERROR] → wait 30s (concurrent request limit)
    - "output" + ("rate limit" or "exceed") in [ERROR] → wait 65s (TPM window reset)
    - Other [ERROR] → exponential backoff up to 120s
    - Empty responses → exponential backoff up to 120s
    """
    cp = prompt + _JSON_SUFFIX
    sem = _PROVIDER_SEMAPHORES.get(_provider_of(model))
    # Reasoning models (gpt-5.4-pro, deepseek-reasoner, o3) need longer timeouts
    timeout = 360 if any(tag in model for tag in ("pro", "reasoner", "o3", "opus")) else 180
    delay = 10
    max_attempts = 6

    for attempt in range(max_attempts):
        try:
            if sem:
                sem.acquire()
            try:
                resp = call_llm(cp, model, timeout=timeout)
            finally:
                if sem:
                    sem.release()
            # Quick sanity: if response is clearly an error or empty, retry
            if not resp or resp.strip() in ("", "Crash", "[TIMEOUT]"):
                if attempt < max_attempts - 1:
                    print(f"    ⏳ Empty response ({model}), retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
            # If response starts with [ERROR], apply intelligent backoff
            if resp and resp.strip().startswith("[ERROR]"):
                error_msg = resp.lower()
                # Check for concurrent/connections error
                if "concurrent" in error_msg or "connections" in error_msg:
                    wait_time = 30
                    if attempt < max_attempts - 1:
                        print(f"    ⏳ Concurrent limit ({model}): {resp[:80]}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        delay = 10  # Reset delay for next backoff
                        continue
                # Check for output rate limit / TPM exceeded
                elif "output" in error_msg and ("rate limit" in error_msg or "exceed" in error_msg):
                    wait_time = 65
                    if attempt < max_attempts - 1:
                        print(f"    ⏳ TPM limit ({model}): {resp[:80]}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        delay = 10  # Reset delay for next backoff
                        continue
                # Other [ERROR] messages: exponential backoff
                else:
                    if attempt < max_attempts - 1:
                        print(f"    ⏳ API error ({model}): {resp[:80]}, retrying in {delay}s...")
                        time.sleep(delay)
                        delay = min(delay * 2, 120)
                        continue
            return resp
        except Exception as e:
            if "exhausted" in str(e).lower() or "429" in str(e):
                print(f"    ⏳ Rate-limited ({model}), retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                if attempt < max_attempts - 1:
                    print(f"    ⚠ Exception ({model}): {e}, retrying...")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                else:
                    return "Crash"
    return "Crash"


def process_task(model, test_meta, scale, diff, max_retries=3):
    """Run one (model, test, scale) combination and return scored results.
    Retries up to max_retries times if the result is all-zero (empty/parse fail).

    New scoring architecture (all phases):
      Cov   — coverage (shape count)
      Geom  — geometry accuracy (position + type + dimensions)
      Sem   — semantic (subsumes physics + structural constraints per family)
      Global — weighted composite: Cov 20% + Geom 30% + Sem 50%
    """
    DEBUG = os.environ.get("C2CAD_DEBUG", os.environ.get("CG3D_DEBUG", "0")) == "1"

    for attempt in range(1, max_retries + 1):
        if test_meta["phase"] in [1, 2, 4]:
            prompt, golden = test_meta["func"](scale)
            llm_resp = run_single_api(model, prompt)
            if DEBUG:
                preview = (llm_resp or "")[:300].replace("\n", "\\n")
                print(f"  [DEBUG] {model} | {test_meta['family']} | attempt={attempt} | resp_len={len(llm_resp or '')} | preview: {preview}")
            parsed = extract_json(llm_resp) if llm_resp not in ["Crash", None, ""] else None
            raw_shapes = parsed if isinstance(parsed, list) else []
            shapes = _normalize_shapes(raw_shapes)
            if DEBUG:
                print(f"  [DEBUG] parsed {len(shapes)} shapes (golden={len(golden)})")
            if shapes:
                c, g = eval_cov_geom(raw_shapes, golden)
                sem = eval_sem(test_meta["family"], shapes, golden, scale, geom_score=g)
                gl = eval_global(c, g, sem)
                if c > 0 or g > 0 or sem > 0:
                    if attempt > 1:
                        print(f"    ↳ retry {attempt} succeeded for {model} | {test_meta['family']}")
                    return {"model": model, "family": test_meta["family"], "diff": diff,
                            "shapes": shapes, "cov": c, "geom": g, "sem": sem, "glob": gl,
                            "phase": test_meta["phase"]}
            if attempt < max_retries:
                print(f"    ↳ attempt {attempt} returned 0 — retrying {model} | {test_meta['family']} ...")
        else:
            # Phase 3: has specs dict with engineering constraints
            prompt, specs = test_meta["func"](scale)
            ref_shapes = specs.get("reference", [])
            llm_resp = run_single_api(model, prompt)
            parsed = extract_json(llm_resp) if llm_resp not in ["Crash", None, ""] else None
            raw = parsed if isinstance(parsed, list) else []
            shapes = _normalize_shapes(raw)
            ops = [r for r in raw if isinstance(r, dict) and "op" in r]
            # Cov + Geom against reference shapes
            if ref_shapes and shapes:
                c, g = eval_cov_geom(raw, ref_shapes)
            else:
                c, g = 0, 0
            # Sem (includes physics + engineering constraints)
            sem = eval_sem(test_meta["family"], shapes, ref_shapes, scale, specs, ops, geom_score=g)
            gl = eval_global(c, g, sem)
            total = c + g + sem
            if total > 0:
                if attempt > 1:
                    print(f"    ↳ retry {attempt} succeeded for {model} | {test_meta['family']}")
                return {"model": model, "family": test_meta["family"], "diff": diff,
                        "shapes": shapes, "cov": c, "geom": g, "sem": sem, "glob": gl,
                        "phase": test_meta["phase"]}
            if attempt < max_retries:
                print(f"    ↳ attempt {attempt} returned 0 — retrying {model} | {test_meta['family']} ...")

    # All retries exhausted — return zero result
    return {"model": model, "family": test_meta["family"], "diff": diff,
            "shapes": [], "cov": 0, "geom": 0, "sem": 0, "glob": 0,
            "phase": test_meta["phase"]}
