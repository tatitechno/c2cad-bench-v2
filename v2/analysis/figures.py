"""Figures for the paper, drawn from the analysis outputs (vector PDF + PNG preview)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import included  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
ARM_ORDER = ["json", "neutral", "v1", "schema", "tool", "mates", "cadquery"]


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


def attribution(out_dir):
    p = out_dir / "a02_attribution.json"
    if not p.exists():
        return
    C = json.loads(p.read_text())["contrasts"]
    keys = [k for k in C if "|" not in k]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(keys) + 1))
    for i, k in enumerate(keys):
        pt, lo, hi = C[k]["pooled"]
        ax.errorbar(100 * pt, i, xerr=[[100 * (pt - lo)], [100 * (hi - pt)]], fmt="o", color="k", capsize=2)
        for mo, (mp, _, _) in C[k]["per_model"].items():
            ax.plot(100 * mp, i + 0.18, "|", color="0.55", ms=6)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.axvspan(-5, 5, color="0.93", zorder=0)
    ax.set_yticks(range(len(keys)), keys)
    ax.invert_yaxis()
    ax.set_xlabel("difference in exact rate (percentage points); pooled 95% CI, ticks = models; band = +-5 pp")
    _save(fig, out_dir, "fig_attribution")


def family_arm(out_dir):
    p = out_dir / "a02_attribution.json"
    if not p.exists():
        return
    fam = pd.DataFrame(json.loads(p.read_text()).get("family_arm_exact", {}))
    if fam.empty:
        return
    cols = [c for c in ARM_ORDER if c in fam.columns]
    fam = fam[cols].sort_values("json" if "json" in cols else cols[0])
    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 2.5, 0.22 * len(fam) + 1))
    im = ax.imshow(100 * fam.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(fam)), fam.index)
    fig.colorbar(im, ax=ax, label="exact %")
    _save(fig, out_dir, "fig_family_arm")


def scale(s, out_dir):
    sw = included(s[(s["split"] == "sweep") & (s["round"] == 0)])
    sw = sw[sw["response_status"] != "truncated"]
    if sw.empty:
        return
    arms = [a for a in ("json", "tool", "mates") if a in set(sw["arm"])]
    fig, axs = plt.subplots(1, len(arms), figsize=(2.4 * len(arms), 2.2), sharey=True, squeeze=False)
    bins = [0, 8, 16, 32, 64, 128, 256, 1024]
    for ax, arm in zip(axs[0], arms):
        g = sw[sw["arm"] == arm].copy()
        g["bin"] = pd.cut(g["n_parts"], bins)
        for mo, gm in g.groupby("model"):
            r = gm.groupby("bin", observed=True)["exact"].mean()
            ax.plot([b.right for b in r.index], 100 * r.to_numpy(float), marker="o", ms=3, label=mo)
        ax.set_xscale("log", base=2)
        ax.axhline(50, color="0.7", lw=0.7, ls="--")
        ax.set_title(arm)
        ax.set_xlabel("parts (bin upper edge)")
    axs[0][0].set_ylabel("exact %")
    axs[0][-1].legend(fontsize=6, frameon=False)
    _save(fig, out_dir, "fig_scale")


def repair(out_dir):
    p = out_dir / "a04_repair.json"
    if not p.exists():
        return
    tr = pd.DataFrame(json.loads(p.read_text()).get("trajectories", []))
    if tr.empty:
        return
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for (mo, arm), g in tr.groupby(["model", "arm"]):
        ax.plot(g["round"], 100 * g["exact"], marker="o", ms=3, ls="-" if arm == "repair_verifier" else ":",
                label=f"{mo} {arm.replace('repair_', '')}")
    ax.set_xlabel("round")
    ax.set_ylabel("exact % of seeds")
    ax.legend(fontsize=5, frameon=False)
    _save(fig, out_dir, "fig_repair")


def run(s, out_dir):
    attribution(out_dir)
    family_arm(out_dir)
    scale(s, out_dir)
    repair(out_dir)
