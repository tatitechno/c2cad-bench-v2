"""
split_database.py
Reads the monolithic showcase_db.js and writes per-phase JS files:
  results/showcase_db_p1.js  ->  window.CG3D_P1 = { golden:[...], models:{...} }
  results/showcase_db_p2.js  ->  window.CG3D_P2 = ...
  results/showcase_db_p3.js  ->  window.CG3D_P3 = ...
Call this after run_unified.py or build_database.py.
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)
RESULTS    = os.path.join(ROOT, "results")
SRC        = os.path.join(RESULTS, "showcase_db.js")


def split():
    if not os.path.exists(SRC):
        print(f"[split_database] {SRC} not found — run build_database.py first.")
        return

    with open(SRC, encoding="utf-8") as f:
        raw = f.read().strip()

    # Strip JS wrapper
    if raw.startswith("window.SHOWCASE_DB = "):
        raw = raw[len("window.SHOWCASE_DB = "):]
    if raw.endswith(";"):
        raw = raw[:-1]

    data = json.loads(raw)
    golden_all = data["golden"]
    models_all = data.get("models", {})

    for phase in [1, 2, 3, 4]:
        # Golden entries for this phase
        golden_phase = [e for e in golden_all if e.get("phase") == phase]

        # Model entries for this phase (preserve index alignment via family+diff match)
        golden_keys = {(e["family"], e["difficultyID"]) for e in golden_phase}
        models_phase = {}
        for model_id, entries in models_all.items():
            phase_entries = [
                e for e in entries
                if e and e.get("phase") == phase
            ]
            if phase_entries:
                models_phase[model_id] = phase_entries

        payload = {"golden": golden_phase, "models": models_phase}
        out_js   = json.dumps(payload, separators=(",", ":"))
        out_path = os.path.join(RESULTS, f"showcase_db_p{phase}.js")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"window.CG3D_P{phase} = {out_js};")

        kb = os.path.getsize(out_path) // 1024
        print(f"[split_database] Phase {phase}: {len(golden_phase)} golden entries, "
              f"{len(models_phase)} models → {out_path} ({kb} KB)")


if __name__ == "__main__":
    split()
