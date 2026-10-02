"""Round-2 time series: net efficiency (A0, A1) and A1 carbon partitioning (free / dissolved / mineral / exported).

Runs every scenario of the overpressure (A0) and injection-driven (A1) models plus the A1 cyclic
injection without water (A1s, pulsed only), each with its own no-injection control, caches the
time series and the carbon-log ledger, and draws two figures.
"""
import sys, os, json, multiprocessing as mp
from pathlib import Path as _P
RES = _P(__file__).resolve().parents[1]      # round2_results/
CODE = _P(__file__).resolve().parents[2]     # the model code
import numpy as np
sys.path.insert(0, str(_P(__file__).resolve().parent))
os.environ.setdefault("RTM_OUTPUT_ROOT", str(CODE / "RTM_Output"))
import harness2 as H

REG = ("MHOW", "CRBG"); MOD = ("single", "continuous", "pulsed")
A1_MC = {"flow_model": "injection_driven", "domain_length_m": 5.0}
A1_EX = {"transport/n_cells": 50}
for r in REG:
    A1_EX[f"region_params/{r}/transport_n_cells"] = 50
    A1_EX[f"region_params/{r}/transport_disp_m"] = 0.1
MODELS = {"A0": ({}, {}), "A1": (A1_MC, A1_EX),
          "A1s": (dict(A1_MC, pulsed_off_mode="shut_in"), A1_EX)}
MODES_OF = {"A0": MOD, "A1": MOD, "A1s": ("pulsed",)}
CACHE = RES / "ts_round2.npz"
LEDGER = RES / "ledger_round2.json"
LOGKEYS = ("t_yr", "C_injected", "C_free", "C_aqueous", "C_mineral_net", "C_export", "C_inflow")


def work(item):
    model, reg, mode, ctrl = item
    mc, ex = MODELS[model]
    cfg = H.make_cfg(mc=mc, control=ctrl, extra=ex)
    s, r, _ = H.run(reg, mode, cfg, f"ts_{model}", keep=True)
    log = r.get("carbon_log", [])
    series = {k: np.array([row[k] for row in log], dtype=float) for k in LOGKEYS} if log else {}
    s.pop("census", None)
    return (model, reg, mode, ctrl), r["time"], r["efficiency_arr"], series, s


def compute():
    items = [(m, r, mo, c) for m in MODELS for r in REG for mo in MODES_OF[m] for c in (False, True)]
    out, led = {}, {}
    with mp.Pool(2) as pool:
        for key, t, e, series, s in pool.imap_unordered(work, items):
            k = "|".join(map(str, key))
            out[k + "|t"] = t; out[k + "|e"] = e
            for name, arr in series.items():
                out[k + "|" + name] = arr
            led[k] = s
            print("done", k, round(s["eff"], 2), s["runtime_s"], flush=True)
    np.savez(CACHE, **out)
    json.dump(led, open(LEDGER, "w"), indent=1, default=str)


INK, INK2, GRID, SPINE, SHADE = "#0b0b0b", "#52514e", "#e6e5e0", "#b9b8b2", "#f0efec"
COLORS = {"single": "#2a78d6", "continuous": "#eb6834", "pulsed": "#1baf7a", "shut": "#eda100"}  # validated slots 1-4
STYLES = {"single": (0, (4, 2)), "continuous": "solid", "pulsed": "solid", "shut": (0, (1, 1.6))}
LABELS = {"single": "Single-burst", "continuous": "Continuous", "pulsed": "WAG", "shut": "Cyclic, no water"}
SITE = {"MHOW": "Mhow", "CRBG": "GRB"}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SPINE)
    ax.tick_params(colors=INK2, labelsize=8)


def _labels(ax, items, ymax):
    """Direct end labels, nudged apart so they never overlap."""
    items = sorted(items, key=lambda x: x[1])
    gap = ymax * 0.075; placed = []
    for text, y, color in items:
        yy = y if not placed else max(y, placed[-1] + gap)
        placed.append(yy)
        ax.annotate(text, xy=(5.0, y), xytext=(5.08, yy), textcoords="data", va="center", fontsize=7.5,
                    color=INK2, annotation_clip=False)


def plot_net(png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(CACHE)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.0), sharex=True)
    titles = {"A0": "Overpressure-driven flow", "A1": "Injection-driven flow"}
    for i, model in enumerate(("A0", "A1")):
        for j, reg in enumerate(REG):
            ax = axes[i, j]
            ax.axvspan(0.0, 0.25, color=SHADE, zorder=0); ax.axvspan(0.5, 0.75, color=SHADE, zorder=0)
            ymax, labs = 0.0, []
            for mode in MOD:
                t = d[f"{model}|{reg}|{mode}|False|t"]; e = d[f"{model}|{reg}|{mode}|False|e"]
                tc = d[f"{model}|{reg}|{mode}|True|t"]; ec = d[f"{model}|{reg}|{mode}|True|e"]
                net = e - np.interp(t, tc, ec)
                ax.plot(t, net, color=COLORS[mode], lw=2.0, ls=STYLES[mode], label=LABELS[mode], zorder=3)
                labs.append((f"{LABELS[mode]} {net[-1]:.0f}%", float(net[-1]), COLORS[mode]))
                ymax = max(ymax, float(np.nanmax(net)))
            top = max(10.0, ymax * 1.12)
            ax.set_xlim(0, 5.0); ax.set_ylim(0, top); _style(ax)
            _labels(ax, labs, top)
            ax.set_title(f"{titles[model]} – {SITE[reg]}", fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("Net mineralised (% of injected CO₂)", fontsize=8.5, color=INK2)
            if i == 1:
                ax.set_xlabel("Time (years)", fontsize=8.5, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "Shaded: WAG CO₂ pulses (0–0.25 and 0.5–0.75 yr). Net = run minus its no-injection control "
             "(same flow history, no CO₂).", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.025, 0.90, 0.95), w_pad=5.0)
    fig.savefig(png, dpi=220); print("saved", png)


def plot_free(png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(CACHE)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5), sharey=True)
    series = [("A1", "single", "single"), ("A1", "continuous", "continuous"), ("A1", "pulsed", "pulsed"),
              ("A1s", "pulsed", "shut")]
    for j, reg in enumerate(REG):
        ax = axes[j]
        ax.axvspan(0.0, 0.25, color=SHADE, zorder=0); ax.axvspan(0.5, 0.75, color=SHADE, zorder=0)
        labs = []
        for model, mode, key in series:
            k = f"{model}|{reg}|{mode}|False"
            t = d[k + "|t_yr"]; inj = d[k + "|C_injected"]; free = d[k + "|C_free"]
            tot = float(inj[-1])
            pct = np.where(tot > 0, free / tot * 100.0, 0.0)
            ax.plot(t, pct, color=COLORS[key], lw=2.0, ls=STYLES[key], label=LABELS[key], zorder=3)
            labs.append((f"{LABELS[key]} {pct[-1]:.0f}%", float(pct[-1]), COLORS[key]))
        ax.set_xlim(0, 5.0); ax.set_ylim(0, 100); _style(ax); _labels(ax, labs, 100.0)
        ax.set_title(f"Injection-driven flow – {SITE[reg]}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Time (years)", fontsize=8.5, color=INK2)
        if j == 0:
            ax.set_ylabel("Undissolved CO₂ (% of total injected)", fontsize=8.5, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.01, "Free-phase CO₂ remaining in the inlet cell, as % of the total CO₂ injected (600 t Mhow, "
             "810 t GRB). Shaded: WAG CO₂ pulses.", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 0.90, 0.90), w_pad=5.0)
    fig.savefig(png, dpi=220); print("saved", png)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        compute()
    plot_net(RES / "fig_round2_net.png")
    plot_free(RES / "fig_round2_free.png")
