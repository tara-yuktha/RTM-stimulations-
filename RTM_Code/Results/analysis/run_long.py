"""20-yr injection-driven runs (continuous vs WAG, uniform time grid) to test whether WAG
mineralises more CO2 or the same CO2 sooner.  Caches time series and ledgers; draws a figure."""
import sys, os, json, multiprocessing as mp
from pathlib import Path as _P
RES = _P(__file__).resolve().parents[1]      # round2_results/
CODE = _P(__file__).resolve().parents[2]     # the model code
import numpy as np
sys.path.insert(0, str(_P(__file__).resolve().parent))
os.environ.setdefault("RTM_OUTPUT_ROOT", str(CODE / "RTM_Output"))
import harness2 as H

REG = ("MHOW", "CRBG"); MOD = ("continuous", "pulsed")
MC = {"flow_model": "injection_driven", "domain_length_m": 5.0}
EX = {"transport/n_cells": 50, "injection/T_sim_yr": 20.0, "injection/n_steps": 3200, "injection/time_grid": "uniform"}
for r in REG:
    EX[f"region_params/{r}/transport_n_cells"] = 50
    EX[f"region_params/{r}/transport_disp_m"] = 0.1
CACHE = RES / "ts_long.npz"; LEDGER = RES / "ledger_long.json"
KEYS = ("t_yr", "C_injected", "C_free", "C_aqueous", "C_mineral_net", "C_export")


def work(item):
    reg, mode, ctrl = item
    s, r, _ = H.run(reg, mode, H.make_cfg(mc=MC, control=ctrl, extra=EX), "long", keep=True)
    log = r.get("carbon_log", [])
    ser = {k: np.array([row[k] for row in log], dtype=float) for k in KEYS}
    s.pop("census", None)
    return item, r["time"], r["efficiency_arr"], ser, s


def compute():
    items = [(r, m, c) for r in REG for m in MOD for c in (False, True)]
    out, led = {}, {}
    with mp.Pool(2) as pool:
        for key, t, e, ser, s in pool.imap_unordered(work, items):
            k = "|".join(map(str, key)); out[k + "|t"] = t; out[k + "|e"] = e
            for n, a in ser.items():
                out[k + "|" + n] = a
            led[k] = s; print("done", k, round(s["eff"], 2), s["runtime_s"], flush=True)
    np.savez(CACHE, **out); json.dump(led, open(LEDGER, "w"), indent=1, default=str)


def plot(png=RES / "fig_round2_long.png"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = np.load(CACHE)
    COL = {"continuous": "#eb6834", "pulsed": "#1baf7a"}; LAB = {"continuous": "Continuous", "pulsed": "WAG"}
    INK, INK2, GRID, SPINE = "#0b0b0b", "#52514e", "#e6e5e0", "#b9b8b2"
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5), sharey=True)
    for ax, (reg, site) in zip(axes, (("MHOW", "Mhow"), ("CRBG", "GRB"))):
        for mode in MOD:
            t = d[f"{reg}|{mode}|False|t"]; e = d[f"{reg}|{mode}|False|e"]
            tc = d[f"{reg}|{mode}|True|t"]; ec = d[f"{reg}|{mode}|True|e"]
            net = e - np.interp(t, tc, ec)
            ax.plot(t, net, color=COL[mode], lw=2.0, label=LAB[mode], zorder=3)
            for yr in (5, 10, 20):
                v = float(np.interp(yr, t, net))
                ax.plot([yr], [v], marker="o", ms=4, color=COL[mode], zorder=4)
            ax.annotate(f"{LAB[mode]} {net[-1]:.0f}%", xy=(t[-1], net[-1]), xytext=(4, 6 if mode == "pulsed" else -6),
                        textcoords="offset points", fontsize=7.5, color=INK2, va="center")
        ax.axvline(5, color=SPINE, lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.text(5.2, 2, "5-yr horizon\nof the paper", fontsize=7, color=INK2, va="bottom")
        ax.set_xlim(0, 20); ax.set_ylim(0, 105); ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_title(f"Injection-driven flow, no regional flow – {site}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Time (years)", fontsize=8.5, color=INK2)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(SPINE)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("Net mineralised (% of injected CO₂)", fontsize=8.5, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.01, "20-yr runs on a uniform time grid (3200 steps). Markers at 5, 10 and 20 yr. Net of each run's "
             "no-injection control.", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 0.95, 0.9), w_pad=2.5)
    fig.savefig(png, dpi=220); print("saved", png)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        compute()
    plot()
