"""E00: scaffolding audit of the prompts, v1 against v2 (and every v2 arm).

1. The released v1 regex audit (scripts/audit_prompts.py: explicit shape count, coordinate vector, formula or
   assignment, trigonometric / Cartesian hint) applied to
     - the released v1 prompts (data/cases.jsonl),
     - the v2 task text (prompt bodies),
     - the full user prompt of every v2 arm (for mates: the task text plus the instruction header; the CML
       language reference and the two worked examples are identical for every task and are audited once).
2. Verbatim coordinate overlap: the share of nonzero reference coordinate components (centers, beam ends)
   whose value appears as a number in the prompt, under two definitions:
     occurrences  every component counts (|value| rounded to 3 decimals, matched against |numbers| in the prompt)
     distinct     each distinct |value| per case counts once
   reported overall and per family (the families with the highest v2 overlap need a hand review: what remains
   is mostly base inputs that coincide with coordinates, e.g. a deck height of 10).
Output: v2/reports/results/e00_prompt_audit.{md,json}. The CI gate is tests/test_v2_runner.py.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "scripts"))
import audit_prompts as AP  # noqa: E402  (the released v1 audit, unchanged)
from c2cad import cases  # noqa: E402
from c2cad.runner import arms as A  # noqa: E402

FLAGS = ("explicit_shape_count", "coordinate_vector", "formula_or_assignment", "trig_or_cartesian_hint")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def flags(text: str) -> dict:
    row = AP.audit_case({"prompt": text})
    return {f: bool(getattr(row, f)) for f in FLAGS}


def shared_blocks() -> dict:
    """Instruction text that is identical for every task (audited once, removed from the per-task audit)."""
    return {"output contract": A.SCHEMA_CONTRACT, "tool instruction": A.TOOL_INSTRUCTION,
            "cadquery instruction": A.CAD_INSTRUCTION,
            "mates instruction + language reference + examples": A.MATES_INSTRUCTION + A._cml_reference() + "\n\n"
            + A._examples_text() + "\n\nNow write the program for the task above.\n"}


def task_text(arm, case) -> str:
    """The part of an arm's user prompt that varies by task."""
    text = A.build(arm, case)[1]
    for block in shared_blocks().values():
        text = text.replace(block, "")
    return text


def numbers(text: str) -> set:
    return {round(abs(float(x)), 3) for x in NUM.findall(text)}


def coord_values(ref) -> list:
    vals = []
    for p in ref:
        for k in ("center", "start", "end"):
            if k in p:
                vals += [round(abs(float(v)), 3) for v in p[k] if abs(float(v)) > 1e-9]
    return vals


def main():
    v2 = cases.load()
    v1 = {(c["family"], c["difficulty_id"]): c for c in (json.loads(l) for l in open(REPO / "data/cases.jsonl"))}
    out = {"v1": {}, "v2_body": {}, "arms": {}}
    L = ["# E00. Prompt scaffolding audit", "", "Script: `v2/experiments/e00_prompt_audit.py` (regexes: the released "
         "`scripts/audit_prompts.py`, unchanged).", ""]

    def tally(texts):
        per = [flags(t) for t in texts]
        return {"n": len(per), "any_flag": sum(any(d.values()) for d in per),
                **{f: sum(d[f] for d in per) for f in FLAGS}}

    out["v1"] = tally([v1[(c["family"], c["level"])]["prompt"] for c in v2])
    out["v2_body"] = tally([c["prompt_body"] for c in v2])
    for arm in A.SINGLE_TURN_ARMS:
        if arm == "v1":
            continue
        out["arms"][arm] = tally([task_text(arm, c) for c in v2 if A.applicable(arm, c)])
    out["shared"] = {name: flags(block) for name, block in shared_blocks().items()}
    L += ["## Regex flags (high-risk = any of the four)", "",
          "| prompt set | prompts | high-risk | shape count | coordinate vector | formula/assignment | trig/Cartesian |",
          "|---|---|---|---|---|---|---|"]
    for name, d in [("v1 released", out["v1"]), ("v2 task text", out["v2_body"])] + \
                   [(f"v2 arm `{k}` (task-specific text)", v) for k, v in out["arms"].items()]:
        L.append(f"| {name} | {d['n']} | {d['any_flag']} | " + " | ".join(str(d[f]) for f in FLAGS) + " |")
    L += ["", "Shared instruction blocks (identical for every task; audited once):", ""]
    for name, d in out["shared"].items():
        hit = [f for f, v in d.items() if v]
        L.append(f"- {name}: {', '.join(hit) if hit else 'no flag'}")
    L += ["", "These matches are not task values: the regex's character class admits the output contract's literal "
              "`{\"shapes\": [...]}` and the CadQuery instruction's dict index `parts[0]`; the mates reference and "
              "its two worked examples (assemblies outside the benchmark) contain coordinate vectors by design."]

    # verbatim coordinate overlap
    ov = {}
    for tag, getp in (("v1", lambda c: v1[(c["family"], c["level"])]["prompt"]), ("v2", lambda c: c["prompt_body"])):
        occ_hit = occ_n = dist_hit = dist_n = 0
        fam = defaultdict(lambda: [0, 0])
        for c in v2:
            nums = numbers(getp(c))
            vals = coord_values(c["reference"])
            hit = sum(v in nums for v in vals)
            occ_hit += hit
            occ_n += len(vals)
            dv = set(vals)
            dist_hit += sum(v in nums for v in dv)
            dist_n += len(dv)
            fam[c["family"]][0] += hit
            fam[c["family"]][1] += len(vals)
        ov[tag] = {"occurrences": 100 * occ_hit / occ_n, "distinct": 100 * dist_hit / dist_n,
                   "by_family": {f: 100 * h / n for f, (h, n) in fam.items() if n}}
    out["overlap"] = ov
    L += ["", "## Verbatim coordinate overlap (% of nonzero reference coordinate components found as numbers in the prompt)",
          "", "| prompts | by occurrence | by distinct value per case |", "|---|---|---|",
          f"| v1 released | {ov['v1']['occurrences']:.1f} | {ov['v1']['distinct']:.1f} |",
          f"| v2 | {ov['v2']['occurrences']:.1f} | {ov['v2']['distinct']:.1f} |", "",
          "Highest v2 families (hand review: are these base inputs that coincide with coordinates, or derived values?):", ""]
    for f, v in sorted(ov["v2"]["by_family"].items(), key=lambda kv: -kv[1])[:8]:
        L.append(f"- {f}: {v:.1f}% (v1: {ov['v1']['by_family'].get(f, float('nan')):.1f}%)")
    res = ROOT / "reports/results"
    (res / "e00_prompt_audit.json").write_text(json.dumps(out, indent=1))
    (res / "e00_prompt_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
