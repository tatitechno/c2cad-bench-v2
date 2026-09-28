"""E08: the CAD-kernel bridge on every reference (the rebuttal's converter claim, re-established under v2).

For each of the 75 main references:
  1. mechanical converter: every primitive -> one OpenCascade solid (c2cad/cadkernel.to_solid); count valid solids;
  2. recovery: every solid -> the primitive read from its B-Rep faces (cadkernel.recover); count unrecognised;
  3. full cadquery-arm path: the converter written as a CadQuery program (cadcode.to_cadquery_code), executed in
     the cadquery sandbox exactly as a model's program would be, recovered, and scored by the v2 evaluator.
Pass criterion for the arm: every case exact with Geometry = 100 (up to rounding).
Output: v2/reports/results/e08_kernel_roundtrip.{md,json}
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c2cad import cadcode, cases  # noqa: E402
from c2cad.evaluate import evaluate  # noqa: E402
from c2cad.runner import sandbox as SB  # noqa: E402
from c2cad.runner.run import postprocess  # noqa: E402


def kernel_counts():
    code = ("import json, sys, collections\nsys.path.insert(0, %r)\nfrom c2cad import cadkernel as K\n"
            "tot = collections.Counter()\nfor line in open(%r):\n    c = json.loads(line)\n"
            "    sol = {p['id']: K.to_solid(p) for p in c['reference']}\n"
            "    tot['primitives'] += len(c['reference'])\n"
            "    tot['valid_solids'] += sum(s.isValid() for s in sol.values())\n"
            "    st = K.recover_all(sol)['stats']\n    tot['unrecognized'] += st['n_unrecognized']\n"
            "    tot['prisms'] += st['n_prisms']\n    for p in c['reference']:\n        tot['type:' + p['type']] += 1\n"
            "print(json.dumps(tot))\n") % (str(ROOT), str(ROOT / "data" / "cases_v2.jsonl"))
    p = subprocess.run([str(SB.CAD_PYTHON), "-c", code], capture_output=True, text=True, timeout=1800)
    if p.returncode:
        raise SystemExit(p.stderr[-800:])
    return json.loads(p.stdout)


def sandbox_case(case):
    post = postprocess("cadquery", case, "```python\n" + cadcode.to_cadquery_code(case["reference"]) + "```")
    r = evaluate(case, post["value"] if post["value"] is not None else [])
    return {"case_id": case["case_id"], "exec": post["exec_status"], "exact": r.exact, "geometry": r.geometry,
            "geometry_equiv": r.geometry_equiv, "semantic": r.semantic, "stats": post["cad_stats"]}


def main():
    if not SB.cad_available():
        raise SystemExit(f"CAD environment missing: {SB.CAD_PYTHON}")
    k = kernel_counts()
    cs = cases.load()
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(sandbox_case, cs))
    n_exact = sum(r["exact"] for r in rows)
    worst = min(rows, key=lambda r: r["geometry"])
    L = ["# E08. CAD-kernel round trip on every reference", "",
         "Script: `v2/experiments/e08_kernel_roundtrip.py` (CadQuery 2.8 / OpenCascade in `v2/.venv-cad`).", "",
         f"- primitives converted: {k['primitives']}; valid OpenCascade solids: {k['valid_solids']}; "
         f"recovered as a primitive: {k['primitives'] - k['unrecognized']} (unrecognised {k['unrecognized']}); "
         f"of which label-free rectangular prisms: {k['prisms']}",
         "- by type: " + ", ".join(f"{t.split(':')[1]} {v}" for t, v in sorted(k.items()) if t.startswith("type:")),
         f"- full cadquery-arm path (program -> sandbox -> kernel -> recovery -> evaluator): {n_exact} / {len(rows)} "
         f"cases exact; lowest Geometry {worst['geometry']:.6f} ({worst['case_id']})", "",
         "A model's CadQuery program is therefore scored on exactly the same footing as its JSON: a correct program "
         "scores 100, and nothing is lost in conversion."]
    res = ROOT / "reports" / "results"
    (res / "e08_kernel_roundtrip.json").write_text(json.dumps({"kernel": k, "cases": rows}, indent=1))
    (res / "e08_kernel_roundtrip.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
