"""E09: inventory of the v2 benchmark (the counts quoted in reports and the paper).

Per split: cases, families, part counts (min / median / max / total), primitive types. For every split:
constraints in total and by kind, and whether every reference satisfies all of them. For the main split:
prompt sentences and how each is traced (constraint keys, ids rule, symbolic declaration, note), and the
families with a hand-written CML gate program.
Output: v2/reports/results/e09_inventory.{md,json}
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c2cad import cml_programs, constraints as K  # noqa: E402
from c2cad.constraints import trace as T  # noqa: E402
from c2cad.evaluate import evaluate  # noqa: E402
from c2cad.geom import normalize  # noqa: E402

SPLITS = {"main": "cases_v2.jsonl", "sweep": "sweep_v2.jsonl", "heldout": "heldout_v2.jsonl"}


def main():
    out, L = {}, ["# E09. Benchmark inventory", "", "Script: `v2/experiments/e09_inventory.py`.", ""]
    L += ["| split | cases | families | parts min | median | max | total | constraints | references exact |",
          "|---|---|---|---|---|---|---|---|---|"]
    for split, f in SPLITS.items():
        cs = [json.loads(l) for l in open(ROOT / "data" / f)]
        n = np.array([c["n_parts"] for c in cs])
        kinds, types, n_cons, exact = Counter(), Counter(), 0, 0
        for c in cs:
            ref, _ = normalize(c["reference"])
            cons = K.build(c, ref)
            n_cons += len(cons)
            kinds.update(x.kind for x in cons)
            types.update(p["type"] for p in c["reference"])
            exact += evaluate(c, c["reference"]).exact
        out[split] = {"cases": len(cs), "families": len({c["family"] for c in cs}), "parts_min": int(n.min()),
                      "parts_median": float(np.median(n)), "parts_max": int(n.max()), "parts_total": int(n.sum()),
                      "constraints": n_cons, "constraints_by_kind": dict(kinds), "types": dict(types),
                      "references_exact": exact}
        L.append(f"| {split} | {len(cs)} | {out[split]['families']} | {n.min()} | {np.median(n):g} | {n.max()} | "
                 f"{n.sum()} | {n_cons} | {exact}/{len(cs)} |")
    L += ["", "Constraints by kind (main split): " + ", ".join(f"{k} {v}" for k, v in
                                                          sorted(out["main"]["constraints_by_kind"].items()))]
    L += ["Primitive types (main split): " + ", ".join(f"{k} {v}" for k, v in sorted(out["main"]["types"].items()))]
    main_cases = [json.loads(l) for l in open(ROOT / "data" / SPLITS["main"])]
    sent = Counter()
    n_sent = 0
    for c in main_cases:
        for s, keys in T.sentence_map(c):
            n_sent += 1
            if keys - T.TAGS:
                sent["constraint keys"] += 1
            else:
                for t in ("ids", "symbolic", "note"):
                    if t in keys:
                        sent[t] += 1
                        break
    out["sentences"] = {"total": n_sent, **sent}
    L += ["", f"Prompt sentences (main split): {n_sent}; traced to constraint keys {sent['constraint keys']}, "
              f"to an id-ordering rule only {sent['ids']}, to a symbolic declaration only {sent['symbolic']}, "
              f"to a note only {sent['note']}; untraced 0 (enforced by tests)."]
    gated = sorted(cml_programs.PROGRAMS)
    out["cml_gated_families"] = gated
    L += ["", f"Families with a CML gate program ({len(gated)}): " + ", ".join(gated)]
    res = ROOT / "reports" / "results"
    (res / "e09_inventory.json").write_text(json.dumps(out, indent=1))
    (res / "e09_inventory.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
