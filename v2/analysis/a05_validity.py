"""A05: measurement validity on live outputs, failure taxonomy, kernel build of json answers, sufficiency list
(plan §7). Uses the json arm, sample 0, main split."""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import V2, macro_name, md_table, write

CASES = None


def _cases():
    global CASES
    if CASES is None:
        CASES = {c["case_id"]: c for c in (json.loads(l) for l in open(V2 / "data" / "cases_v2.jsonl"))}
    return CASES


def json_answers(r_dirs: list[Path]) -> dict:
    out = {}
    for d in r_dirs:
        p = d / "responses.jsonl"
        if p.exists():
            for line in open(p):
                rec = json.loads(line)
                if (rec["arm"] == "json" and rec.get("split", "main") == "main" and rec["sample"] == 0
                        and rec.get("round", 0) == 0 and rec.get("ok")):
                    out[(rec["model"], rec["case_id"])] = rec["text"]
    return out


def _metrics(item):
    (model, cid), text = item
    from c2cad import metrics3d as M
    from c2cad.evaluate import evaluate
    from c2cad.geom import normalize
    from c2cad.runner.arms import parse_json
    from c2cad.score import agreement, assignment_map
    case = _cases()[cid]
    val, ps = parse_json(text)
    raw = val if val is not None else []
    r = evaluate(case, raw)
    ref, _ = normalize(case["reference"])
    out, _ = normalize(raw)
    row = dict(model=model, case_id=cid, family=case["family"], n_ref=len(ref), n_out=len(out), parse=ps,
               coverage=r.coverage, geometry=r.geometry, geometry_equiv=r.geometry_equiv, semantic=r.semantic,
               global_v2=r.global_v2, exact=r.exact, type_fidelity=r.type_fidelity,
               iou=np.nan, chamfer=np.nan, f2=np.nan, orient_err=np.nan, orient_obb=np.nan, confusions={})
    if out:
        row["iou"] = M.iou(ref, out)
        ch, fs = M.chamfer_fscore(ref, out, n=20000, thresholds=(0.02,))
        row["chamfer"], row["f2"] = ch, fs[0.02]
        amap = assignment_map(ref, out)
        row["orient_err"] = M.orientation_error(ref, out, amap)
        row["orient_obb"] = M.orientation_error_obb(ref, out, amap)
        ag = agreement(ref, out)
        from c2cad.score import POS_TOL_FRAC, POS_TOL_MIN
        from c2cad.geom import assembly_diagonal
        tol = max(POS_TOL_MIN, POS_TOL_FRAC * assembly_diagonal(ref))
        row["confusions"] = dict(Counter(f"{rt}->{ot}" for _, _, dist, _, rt, ot in ag.pairs if rt != ot and dist <= tol))
    return row


def kernel_build(items: dict) -> dict:
    """(model, case) -> True if every part of the json answer builds as a valid OpenCascade solid."""
    from c2cad.runner import sandbox
    if not sandbox.cad_available() or not items:
        return {}
    from c2cad.geom import normalize
    from c2cad.runner.arms import parse_json
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "answers.jsonl"
        with open(src, "w") as fh:
            for (m, cid), text in items.items():
                val, _ = parse_json(text)
                parts = [s.to_json() for s in normalize(val if val is not None else [])[0]]
                fh.write(json.dumps({"k": [m, cid], "parts": parts}) + "\n")
        code = ("import json, sys\nsys.path.insert(0, %r)\nfrom c2cad import cadkernel as K\n"
                "for line in open(%r):\n    a = json.loads(line)\n    ok = bool(a['parts'])\n"
                "    for p in a['parts']:\n        try:\n            ok = ok and K.to_solid(p).isValid()\n"
                "        except Exception:\n            ok = False\n"
                "    print(json.dumps([a['k'], ok]))\n") % (str(V2), str(src))
        p = subprocess.run([str(sandbox.CAD_PYTHON), "-c", code], capture_output=True, text=True, timeout=7200)
    res = {}
    for line in p.stdout.splitlines():
        k, ok = json.loads(line)
        res[tuple(k)] = ok
    return res


def taxonomy(row) -> str:
    if row["parse"] == "fail":
        return "parse failure"
    if row["n_out"] == 0:
        return "empty"
    if row["exact"]:
        return "exact"
    if row["n_out"] != row["n_ref"]:
        return "wrong part count"
    if row["type_fidelity"] < 99.999:
        return "type substitution"
    if row["geometry"] < 99.0:
        return "placement or size"
    return "sub-tolerance (constraints only)"


def run(s: pd.DataFrame, r: pd.DataFrame, out_dir, r_dirs=None) -> dict:
    items = json_answers(r_dirs or [])
    L = ["# A05. Measurement validity on live outputs (json arm, sample 0, main split)", ""]
    if not items:
        write(out_dir, "a05_validity", L + ["No json answers found."], {})
        return {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(_metrics, list(items.items()), chunksize=2))
    kb = kernel_build(items)
    for row in rows:
        row["builds"] = kb.get((row["model"], row["case_id"]))
        row["taxonomy"] = taxonomy(row)
    df = pd.DataFrame(rows)
    with open(out_dir / "a05_per_case_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "confusions"])
        w.writeheader()
        w.writerows([{k: v for k, v in x.items() if k != "confusions"} for x in rows])
    axes = ["coverage", "geometry", "geometry_equiv", "semantic", "global_v2"]
    alt = [("iou", 1), ("f2", 1), ("chamfer", -1), ("orient_err", -1), ("orient_obb", -1)]
    L += [f"Case-level Spearman (n = {len(df)} answers; Chamfer and orientation error sign-flipped):", ""]
    body = []
    for ax in axes:
        cells = [ax]
        for a, sg in alt:
            x = df[[ax, a]].dropna()
            cells.append(f"{spearmanr(x[ax], sg * x[a]).statistic:.3f}" if len(x) > 2 else "n/a")
        body.append(cells)
    L += md_table(["axis"] + [a for a, _ in alt], body)
    per = df.groupby("model")[axes + [a for a, _ in alt]].mean()
    if len(per) > 2:
        L += ["", "Model-level Spearman with Global_v2: " + ", ".join(
            f"{a} {spearmanr(per['global_v2'], sg * per[a]).statistic:.3f}" for a, sg in alt)]
    q = {"F@2% >= 0.9 but not exact": df[(df["f2"] >= 0.9) & (~df["exact"])],
         "IoU >= 0.8 but Sem < 60": df[(df["iou"] >= 0.8) & (df["semantic"] < 60)],
         "IoU < 0.3 but Geom >= 80": df[(df["iou"] < 0.3) & (df["geometry"] >= 80)]}
    L += ["", "## Disagreement quadrants", ""]
    for k, v in q.items():
        L.append(f"- {k}: {len(v)} / {len(df)} ({100 * len(v) / len(df):.1f}%)")
        for x in v.sort_values("semantic").head(3).itertuples():
            L.append(f"    - {x.model} / {x.case_id}: IoU {x.iou:.2f}, F@2% {x.f2:.2f}, Geom {x.geometry:.1f}, "
                     f"Sem {x.semantic:.1f}, parts {x.n_out}/{x.n_ref}")
    if kb:
        b = df[df["builds"] == True]  # noqa: E712
        L += ["", "## Kernel build of json answers (every part a valid OpenCascade solid)", "",
              f"- build: {len(b)} / {len(df)} ({100 * len(b) / len(df):.1f}%)",
              f"- of those, not exact: {100 * (~b['exact']).mean():.1f}%; Geometry < 90: {100 * (b['geometry'] < 90).mean():.1f}%;"
              f" Geometry < 70: {100 * (b['geometry'] < 70).mean():.1f}%"]
    L += ["", "## Failure taxonomy (first failing check, in order)", ""]
    tax = df.groupby(["model", "taxonomy"]).size().unstack(fill_value=0)
    L += md_table(["model"] + list(tax.columns), [[m] + [int(v) for v in tax.loc[m]] for m in tax.index])
    conf = Counter()
    for c in df["confusions"]:
        conf.update(c)
    L += ["", "## Co-located type substitutions (reference -> answer, within the position tolerance)", ""]
    L += [f"- {k}: {v}" for k, v in conf.most_common(12)] or ["- none"]
    # sufficiency review list: cases never solved exactly by any model / arm / sample on the main split
    m = s[(s["split"] == "main") & (s["round"] == 0) & (s["arm"] != "probe")]
    solved = m.groupby("case_id")["exact"].any()
    tool_solved = m[m["arm"] == "tool"].groupby("case_id")["exact"].any()
    never = sorted(solved[~solved].index)
    L += ["", "## Prompt-sufficiency review list", "",
          f"- cases solved exactly by at least one tool-arm answer: {int(tool_solved.sum())} / {len(tool_solved)}",
          f"- cases never solved by any model, arm or sample ({len(never)}; review each prompt against its reference):"]
    L += [f"    - {c}" for c in never]
    macros = {"\\ValidityNAnswers": str(len(df))}
    if kb:
        b = df[df["builds"] == True]  # noqa: E712
        macros["\\KernelBuildPct"] = f"{100 * len(b) / len(df):.1f}"
        macros["\\KernelBuildNotExactPct"] = f"{100 * (~b['exact']).mean():.1f}"
    write(out_dir, "a05_validity", L, {"quadrants": {k: len(v) for k, v in q.items()}, "confusions": dict(conf),
                                       "never_solved": never, "tool_solved": int(tool_solved.sum()),
                                       "taxonomy": tax.to_dict()})
    return macros
