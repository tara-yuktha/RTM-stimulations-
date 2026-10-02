from pathlib import Path as _P
_RES = _P(__file__).resolve().parents[1]
_ROOT = _RES.parent
"""WAG - continuous (net pp) against regional Darcy flux, injection-driven model (compiled2.json)."""
import json
from pathlib import Path as _P
RES = _RES
CODE = _P(__file__).resolve().parents[2]     # the model code
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = json.load(open(RES / "compiled2.json"))
FLUX5 = [(0.0, "A1"), (0.01, "A1_amb0.01"), (0.03, "A1_amb0.03"), (0.1, "A1_amb0.1"), (0.3, "A1_amb0.3")]
TRUNC = [(1.0, "A1_amb1")]                       # 5-m domain at 1 m/yr: 16-47 % carried out of the domain
L50 = [(0.3, "A1_amb0.3_L50"), (1.0, "A1_amb1_L50")]
XLAB = ["0", "0.01", "0.03", "0.1", "0.3", "1"]
XPOS = {0.0: 0, 0.01: 1, 0.03: 2, 0.1: 3, 0.3: 4, 1.0: 5}
INK, INK2, GRID, SPINE, TIE = "#0b0b0b", "#52514e", "#e6e5e0", "#b9b8b2", "#f0efec"
SITES = [("MHOW", "Mhow", "o", "solid", -0.06), ("CRBG", "GRB", "s", (0, (4, 2)), 0.06)]


def diff(cfg, reg, key):
    try:
        a = C[cfg][f"{reg}_pulsed"][key]; b = C[cfg][f"{reg}_continuous"][key]
    except KeyError:
        return None
    return None if a is None or b is None else a - b


def pts(lst, reg, key, off):
    out = [(XPOS[f] + off, diff(cfg, reg, key)) for f, cfg in lst]
    return [(x, y) for x, y in out if y is not None]


fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), sharey=True)
for ax, (key, title) in zip(axes, [("net_1yr", "After 1 year"), ("net_2yr", "After 2 years"), ("net", "After 5 years")]):
    ax.axhspan(-1, 1, color=TIE, zorder=0)
    ax.axhline(0, color=SPINE, lw=0.8, zorder=1)
    for reg, lab, mk, ls, off in SITES:
        p = pts(FLUX5, reg, key, off)
        if p:
            x, y = zip(*p)
            ax.plot(x, y, color=INK, lw=1.6, ls=ls, marker=mk, ms=5.5, label=f"{lab}, 5-m domain", zorder=3)
        q = pts(L50, reg, key, off)
        if q:
            x, y = zip(*q)
            ax.plot(x, y, ls="none", marker=mk, ms=7, mfc="white", mec=INK, mew=1.3, label=f"{lab}, whole 50-m column", zorder=4)
        r = pts(TRUNC, reg, key, off)
        if r:
            x, y = zip(*r)
            ax.plot(x, y, ls="none", marker="x", ms=6, mew=1.3, color=INK2,
                    label="5-m domain at 1 m/yr (CO₂ lost across boundary)" if reg == "MHOW" else None, zorder=4)
    ax.set_xticks(range(len(XLAB))); ax.set_xticklabels(XLAB)
    ax.set_xlim(-0.5, len(XLAB) - 0.5)
    ax.set_title(title, fontsize=13, color=INK, loc="left")
    ax.set_xlabel("Regional Darcy flux (m/yr)", fontsize=12, color=INK2)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SPINE)
    ax.tick_params(colors=INK2, labelsize=8)
axes[0].set_ylabel("WAG − continuous (pp of injected CO₂)", fontsize=12, color=INK2)
hl = {}
for ax in axes:
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        hl.setdefault(l_, h_)
order = ["Mhow, 5-m domain", "GRB, 5-m domain", "Mhow, whole 50-m column", "GRB, whole 50-m column",
         "5-m domain at 1 m/yr (CO₂ lost across boundary)"]
labs = [o for o in order if o in hl]
fig.legend([hl[o] for o in labs], labs, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
fig.text(0.5, 0.01, "Injection-driven model, reactive surface ×1, net of each run's no-injection control. "
         "Shaded: ±1 pp tie band. Positive = WAG mineralises more.", ha="center", fontsize=10.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.84), w_pad=1.5)
fig.savefig(str(_RES / 'figures' / 'regime.png'), dpi=220)
print("saved")
