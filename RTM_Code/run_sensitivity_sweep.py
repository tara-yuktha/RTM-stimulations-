"""
run_sensitivity_sweep.py  (v31, revision R2 corrected)
======================================================
Cross-mode ranking sensitivity sweep.

PURPOSE
-------
Reviewer R1#2 and R1#7: TEST whether the Pulsed > Continuous > Single ordering of
mineralisation efficiency holds when uncertain parameters are varied.

[FIX-14] The previous version described the sweep as demonstrating that the
ordering is "emergent and robust" under a "mode-independent CARB_CAP".  In that
code CARB_CAP carried a mode-specific multiplier, so the sweep could not test
the ordering.  With the corrected (mass-conservative) engine the swept
parameters are:
  1. access_fraction : fraction of mineral surface that reacts (reactive
                       surface area), factors [0.01, 0.1, 0.5, 1, 2]
  2. permeability    : k multiplied by [0.1, 0.5, 1, 2] (sets the Darcy
                       velocity and therefore the residence time)
With the legacy engine (revision_fixes all False) the original parameters
carb_cap_base and log_pco2_off_floor are swept instead.

For every value all 6 scenarios are run together with their no-injection
controls; the ordering is evaluated on the control-corrected (net) efficiency
and reported as "holds" / "does not hold" for each site.

OUTPUTS
-------
- Excel files:
    sensitivity_ranking_carb_cap.xlsx
    sensitivity_ranking_pco2_floor.xlsx
    sensitivity_ranking_summary.xlsx
- Visualization:
    sensitivity_ranking_heatmap.png
"""

import sys, os, copy, argparse, time, multiprocessing
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from pathlib import Path

# [FIX-14] portable path (was a hard-coded Windows path)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_rtm import CONFIG, run_simulation

OUTPUT_DIR = Path(os.environ.get("RTM_OUTPUT_ROOT", str(HERE / "RTM_Output"))) / "sensitivity_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _worker_sim(args):
    region, mode, cfg, val_idx, val_label, actual_val, param_name = args
    import io, sys
    # Suppress console printing during parallel execution
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        r = run_simulation(region, mode, cfg)
        eff = float(r["efficiency_arr"][-1])
        # [FIX-15] no-injection control with the same settings -> net efficiency
        eff_ctrl = float("nan")
        if cfg.get("revision_fixes", {}).get("mass_conservative_engine"):
            cfg_c = copy.deepcopy(cfg); cfg_c["control_no_injection"] = True
            eff_ctrl = float(run_simulation(region, mode, cfg_c)["efficiency_arr"][-1])
        ph_nadir = float(r["pH"].min())
        ph_fin = float(r["pH"][-1])
        por_fin = float(r["porosity"][-1])
        mb_err = float(r.get("mass_balance_err_pct", 0.0))
        return {
            "val_idx": val_idx,
            "region": region,
            "mode": mode,
            "param": param_name,
            "setting": val_label,
            "value": actual_val,
            "eff": eff,
            "eff_ctrl": eff_ctrl,
            "eff_net": eff - eff_ctrl if eff_ctrl == eff_ctrl else eff,
            "pH_nadir": ph_nadir,
            "pH_fin": ph_fin,
            "por_fin": por_fin,
            "mb_err": mb_err,
            "success": True,
        }
    except Exception as e:
        return {
            "val_idx": val_idx,
            "region": region,
            "mode": mode,
            "param": param_name,
            "setting": val_label,
            "value": actual_val,
            "eff": float("nan"),
            "eff_ctrl": float("nan"),
            "eff_net": float("nan"),
            "pH_nadir": float("nan"),
            "pH_fin": float("nan"),
            "por_fin": float("nan"),
            "mb_err": float("nan"),
            "success": False,
            "error": str(e),
        }
    finally:
        sys.stdout = old_stdout


def _apply(cfg, param_name, val, is_factor, base_val):
    """Set one swept parameter in cfg; returns (actual value, label)."""
    if param_name == "carb_cap_base":              # legacy engine only
        actual = base_val * val if is_factor else val
        cfg["kinetics"]["carb_cap_base"] = actual
        return actual, (f"{val:.2f}x ({actual:.2e})" if is_factor else f"{actual:.2e}")
    if param_name == "log_pco2_off_floor":         # legacy engine only
        cfg["geochemistry"]["log_pco2_off_floor"] = val
        return val, f"{val:.2f}"
    if param_name == "access_fraction":            # reactive surface area
        actual = base_val * val
        cfg["kinetics"]["access_fraction"] = actual
        return actual, f"{val:g}x ({actual:.3g})"
    if param_name == "permeability":               # Darcy velocity / residence time
        for reg in ("MHOW", "CRBG"):
            cfg["region_params"][reg]["permeability_mD"] *= val
        return val, f"k x{val:g}"
    raise ValueError(f"Unknown param: {param_name}")


def run_single_sweep(param_name, param_values, is_factor=False, base_val=None, n_workers=2):
    """
    Run sweep for a single parameter across both regions and all 3 modes using parallel processes.
    """
    print("\n" + "="*78)
    print(f"STARTING PARALLEL SENSITIVITY SWEEP: {param_name}")
    print(f"Values to test: {param_values} (is_factor={is_factor}) | Workers: {n_workers}")
    print("="*78)

    tasks = []
    val_metadata = []
    for val_idx, val in enumerate(param_values):
        cfg = copy.deepcopy(CONFIG)
        actual_val, label_val = _apply(cfg, param_name, val, is_factor, base_val)

        val_metadata.append({
            "val_idx": val_idx,
            "param": param_name,
            "setting": label_val,
            "value": actual_val,
        })

        for region in ["MHOW", "CRBG"]:
            for mode in ["single", "continuous", "pulsed"]:
                tasks.append((region, mode, cfg, val_idx, label_val, actual_val, param_name))

    print(f"Queueing {len(tasks)} simulation runs across {n_workers} CPU worker processes...", flush=True)
    t0 = time.time()
    with multiprocessing.Pool(processes=n_workers) as pool:
        sim_results = pool.map(_worker_sim, tasks)
    elapsed = time.time() - t0
    print(f"All {len(tasks)} runs completed in {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per simulation avg)!\n")

    # Group results by val_idx
    results_by_val = {v["val_idx"]: dict(v) for v in val_metadata}
    for res in sim_results:
        idx = res["val_idx"]
        reg = res["region"]
        mod = res["mode"]
        for key in ("eff", "eff_ctrl", "eff_net", "pH_nadir", "pH_fin", "por_fin", "mb_err"):
            results_by_val[idx][f"{reg}_{mod}_{key}"] = res[key]

    rows = []
    for val_idx in sorted(results_by_val.keys()):
        row = results_by_val[val_idx]
        for region in ["MHOW", "CRBG"]:
            p = row.get(f"{region}_pulsed_eff_net", float("nan"))
            c = row.get(f"{region}_continuous_eff_net", float("nan"))
            s = row.get(f"{region}_single_eff_net", float("nan"))
            rank_ok = bool(p > c and c > s)
            row[f"{region}_ranked_ok"] = rank_ok
            row[f"{region}_p_minus_c"] = p - c
            row[f"{region}_c_minus_s"] = c - s
            print(f"  [{row['setting']:18s}] {region}: net Pulsed {p:6.1f}% | Cont {c:6.1f}% | Single {s:6.1f}%"
                  f"  P-C {p - c:+.1f} pp, C-S {c - s:+.1f} pp (strict P>C>S {'holds' if rank_ok else 'does not hold'})")
        row["all_ranked_ok"] = row["MHOW_ranked_ok"] and row["CRBG_ranked_ok"]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_ranking_summary(dfs, save_path):
    """
    Net efficiency of the three modes versus each swept parameter (one row per
    parameter, one column per site).  [FIX-14] neutral title: the figure shows
    whether the ordering holds, it does not assume it.
    """
    names = list(dfs.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(14, 5 * len(names)), squeeze=False)
    fig.subplots_adjust(hspace=0.40, wspace=0.25)
    for i, pname in enumerate(names):
        df = dfs[pname]
        x_labels = [str(r["setting"]) for _, r in df.iterrows()]
        x_idx = np.arange(len(x_labels))
        for j, region in enumerate(["MHOW", "CRBG"]):
            ax = axes[i, j]
            ax.plot(x_idx, df[f"{region}_pulsed_eff_net"], 'o-', color='#2ca02c', label='Pulsed (WAG)', lw=2)
            ax.plot(x_idx, df[f"{region}_continuous_eff_net"], 's-', color='#1f77b4', label='Continuous', lw=2)
            ax.plot(x_idx, df[f"{region}_single_eff_net"], '^-', color='#d62728', label='Single-burst', lw=2)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=9)
            ax.set_title(f"{region}: net efficiency vs {pname}", fontsize=11, fontweight='bold')
            ax.set_ylabel("Net mineralisation efficiency (%)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='best', fontsize=9)
    plt.suptitle("Sensitivity of mode ordering (control-corrected efficiency)",
                 fontsize=13, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[FIGURE SAVED] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-mode ranking sensitivity sweep")
    parser.add_argument("--quick", action="store_true", help="Run 3 points per parameter instead of full set")
    parser.add_argument("--param", default="all",
                        help="all | access_fraction | permeability | carb_cap_base | log_pco2_off_floor")
    # [FIX-14] the README documented --workers but the parser did not accept it
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel worker processes")
    args = parser.parse_args()

    use_mc = CONFIG.get("revision_fixes", {}).get("mass_conservative_engine", False)
    if use_mc:
        sweeps = {
            "access_fraction": ([0.01, 0.1, 0.5, 1.0, 2.0] if not args.quick else [0.1, 1.0, 2.0],
                                True, CONFIG["kinetics"]["access_fraction"]),
            "permeability":    ([0.1, 0.5, 1.0, 2.0] if not args.quick else [0.1, 1.0],
                                True, 1.0),
        }
    else:
        sweeps = {
            "carb_cap_base":      ([0.50, 0.75, 1.00, 1.25, 1.50, 2.00] if not args.quick else [0.50, 1.00, 1.50],
                                   True, CONFIG["kinetics"]["carb_cap_base"]),
            "log_pco2_off_floor": ([-3.0, -2.5, -2.0, -1.5] if not args.quick else [-3.0, -2.5, -2.0],
                                   False, None),
        }
    if args.param != "all":
        sweeps = {args.param: sweeps[args.param]}

    dfs = {}
    for pname, (vals, is_factor, base) in sweeps.items():
        df = run_single_sweep(pname, vals, is_factor=is_factor, base_val=base, n_workers=args.workers)
        xlsx = OUTPUT_DIR / f"sensitivity_ranking_{pname}.xlsx"
        df.to_excel(xlsx, index=False)
        print(f"[SAVED] {xlsx}")
        dfs[pname] = df

    plot_path = OUTPUT_DIR / "sensitivity_ranking.png"
    plot_ranking_summary(dfs, plot_path)
    with pd.ExcelWriter(OUTPUT_DIR / "sensitivity_ranking_summary.xlsx") as writer:
        for pname, df in dfs.items():
            df.to_excel(writer, sheet_name=pname[:31], index=False)
    print(f"[SAVED] {OUTPUT_DIR / 'sensitivity_ranking_summary.xlsx'}")

    print("\n" + "="*78)
    print("SENSITIVITY SWEEP RUN COMPLETE!")
    print("="*78)


if __name__ == "__main__":
    main()
