"""E06: atlas of all 75 v2 reference assemblies (one PNG each), per-phase contact sheets, and a
three-level panel for one family with its prompts. Output: v2/reports/atlas/"""
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c2cad import cases, render  # noqa: E402
from c2cad.geom import normalize  # noqa: E402

OUT = ROOT / "reports/atlas"


def one(c):
    for attempt in range(3):      # matplotlib 3D occasionally raises on autoscale inside worker processes
        try:
            render.render_file(c["reference"], OUT / f"{c['case_id']}.png", title=f"{c['family']} L{c['level']} ({c['n_parts']} parts)")
            return c["case_id"]
        except ValueError:
            continue
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C = cases.load()
    with ProcessPoolExecutor(max_workers=10) as ex:
        failed = [c for c, r in zip(C, ex.map(one, C)) if r is None]
    for c in failed:          # retry serially
        one(c)
    phase_names = {1: "Phase 1: geometric forms", 2: "Phase 2: complex structures", 3: "Phase 3: engineering constraints", 4: "Phase 4: bio-inspired"}
    for ph in (1, 2, 3, 4):
        fams = [f for f in dict.fromkeys(c["family"] for c in C if c["phase"] == ph)]
        fig = plt.figure(figsize=(9, 3 * len(fams)), dpi=110)
        for i, f in enumerate(fams):
            for l in (1, 2, 3):
                c = next(x for x in C if x["family"] == f and x["level"] == l)
                ax = fig.add_subplot(len(fams), 3, 3 * i + l, projection="3d")
                render.draw(ax, normalize(c["reference"])[0], title=f"{f} L{l} ({c['n_parts']})")
        fig.legend(handles=render.legend_handles(), loc="lower center", ncol=7, fontsize=8)
        fig.suptitle(phase_names[ph], fontsize=12)
        fig.tight_layout(rect=(0, 0.03, 1, 0.98))
        fig.savefig(OUT / f"contact_phase{ph}.png")
        plt.close(fig)
    # three levels of one family with their prompts
    fam = "Spiral Staircase"
    fig = plt.figure(figsize=(12, 7.5), dpi=120)
    for l in (1, 2, 3):
        c = next(x for x in C if x["family"] == fam and x["level"] == l)
        ax = fig.add_subplot(2, 3, l, projection="3d")
        render.draw(ax, normalize(c["reference"])[0], title=f"Level {l}: {c['n_parts']} parts")
        tax = fig.add_subplot(2, 3, 3 + l); tax.axis("off")
        tax.text(0, 1, "\n".join(textwrap.wrap(c["prompt_body"], 58)), va="top", fontsize=6.3, family="monospace")
    fig.suptitle(f"{fam}: the three levels differ only in the tread count (10, 24, 50)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "levels_spiral_staircase.png")
    plt.close(fig)
    print("atlas written to", OUT)


if __name__ == "__main__":
    main()
