"""E01: Which v1 semantic sub-checks does each golden reference fail?

Instruments every v1 family evaluator (runners/run_unified.py, unchanged) so the
weighted sub-check dict `w` is captured, then evaluates each golden against its
own validator. Output: v2/reports/results/e01_v1_validator_golden_audit.{json,md}
"""
import sys, io, json, re, inspect, contextlib, textwrap
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "runners")]
with contextlib.redirect_stdout(io.StringIO()):
    import runners.run_unified as R

CAPTURE = {}
def instrument(fn):
    src = textwrap.dedent(inspect.getsource(fn))
    pat = "raw = sum(s*wt for s, wt in w.values())"
    if pat not in src:
        return None
    src = src.replace(pat, pat + "\n    _CAPTURE['w'] = dict(w)")
    ns = dict(R.__dict__); ns["_CAPTURE"] = CAPTURE
    exec(src, ns)
    return ns[fn.__name__]

rows = []
for t in R.ALL_TESTS:
    fam = t["family"]
    disp = R._SEM_DISPATCH[fam]
    # dispatch entries are lambdas wrapping the _sem_* function; find the target by name
    target_name = None
    for name in re.findall(r"(_sem_\w+)", inspect.getsource(disp)):
        target_name = name; break
    target = getattr(R, target_name)
    inst = instrument(target)
    for lvl, scale in enumerate(t["scales"], 1):
        with contextlib.redirect_stdout(io.StringIO()):
            p, g = t["func"](scale)
        specs = g if isinstance(g, dict) and "reference" in g else None
        ref = specs["reference"] if specs else g
        shapes = R._normalize_shapes(ref)
        with contextlib.redirect_stdout(io.StringIO()):
            raw = R._sem_raw(fam, shapes, ref, scale, specs)
        failing = {}
        if inst is not None:
            CAPTURE.clear()
            old = getattr(R, target_name); setattr(R, target_name, inst)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    R._sem_raw(fam, shapes, ref, scale, specs)
            finally:
                setattr(R, target_name, old)
            w = CAPTURE.get("w", {})
            failing = {k: round(v[0], 4) for k, v in w.items() if v[0] < 0.999}
        rows.append({"family": fam, "level": lvl, "evaluator": target_name,
                     "raw_golden": round(raw, 4), "instrumented": inst is not None,
                     "failing_subchecks": failing})

out = ROOT / "v2/reports/results"
json.dump(rows, open(out / "e01_v1_validator_golden_audit.json", "w"), indent=1)
lines = ["| Family | L | raw golden Sem | failing sub-checks (score) |", "|---|---|---|---|"]
for r in rows:
    fs = ", ".join(f"{k}={v}" for k, v in r["failing_subchecks"].items()) if r["instrumented"] else "(not instrumentable)"
    if r["raw_golden"] < 0.999 or r["failing_subchecks"]:
        lines.append(f"| {r['family']} | {r['level']} | {r['raw_golden']} | {fs} |")
n_fam = len({r['family'] for r in rows if r['raw_golden'] < 0.999})
lines.append(f"\nFamilies with raw golden Sem < 0.999 at >=1 level: {n_fam}/25")
open(out / "e01_v1_validator_golden_audit.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
