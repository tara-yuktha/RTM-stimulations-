"""
run_sensitivity_sweep.py  (v30)
================================
Cross-mode ranking-robustness sensitivity sweep.

PURPOSE
-------
Addresses Reviewer R1#2 and R1#7: demonstrate that the Pulsed > Continuous > Single
mineralisation-efficiency ordering is emergent and robust across:
  1. carb_cap_base: base kinetic carbonate precipitation throttle
     Swept over factors: [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
     (spanning 4.5e-5 to 1.8e-4 mol/kgw/step)
  2. log_pco2_off_floor: residual dissolved CO2 in porewater during monitoring / flush
     Swept over values: [-3.0, -2.5, -2.0, -1.5]

For every parameter value, all 6 simulations (MHOW × 3 modes + CRBG × 3 modes)
are executed with the physics-derived mode-independent CARB_CAP formula.
The ranking condition (Pulsed > Continuous > Single) is evaluated for each region.

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

HERE = Path(r"E:\Whats Next\Manuscript submission\Injection Code")
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_rtm import CONFIG, run_simulation

OUTPUT_DIR = HERE / "RTM_Output" / "sensitivity_sweep"
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
            "pH_nadir": float("nan"),
            "pH_fin": float("nan"),
            "por_fin": float("nan"),
            "mb_err": float("nan"),
            "success": False,
            "error": str(e),
        }
    finally:
        sys.stdout = old_stdout


def run_single_sweep(param_name, param_values, is_factor=False, base_val=None, n_workers=6):
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
        if param_name == "carb_cap_base":
            actual_val = base_val * val if is_factor else val
            cfg["kinetics"]["carb_cap_base"] = actual_val
            label_val = f"{val:.2f}x ({actual_val:.2e})" if is_factor else f"{actual_val:.2e}"
        elif param_name == "log_pco2_off_floor":
            actual_val = val
            cfg["geochemistry"]["log_pco2_off_floor"] = actual_val
            label_val = f"{actual_val:.2f}"
        else:
            raise ValueError(f"Unknown param: {param_name}")

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
        eff = res["eff"]
        results_by_val[idx][f"{reg}_{mod}_eff"] = eff
        results_by_val[idx][f"{reg}_{mod}_pH_nadir"] = res["pH_nadir"]
        results_by_val[idx][f"{reg}_{mod}_pH_fin"] = res["pH_fin"]
        results_by_val[idx][f"{reg}_{mod}_por_fin"] = res["por_fin"]
        results_by_val[idx][f"{reg}_{mod}_mb_err"] = res["mb_err"]

    rows = []
    for val_idx in sorted(results_by_val.keys()):
        row = results_by_val[val_idx]
        for region in ["MHOW", "CRBG"]:
            p = row.get(f"{region}_pulsed_eff", float("nan"))
            c = row.get(f"{region}_continuous_eff", float("nan"))
            s = row.get(f"{region}_single_eff", float("nan"))
            rank_ok = bool(p > c and c > s)
            row[f"{region}_ranked_ok"] = rank_ok
            row[f"{region}_p_minus_c"] = p - c
            row[f"{region}_c_minus_s"] = c - s
            print(f"  [{row['setting']:15s}] {region}: Pulsed({p:.1f}%) > Cont({c:.1f}%) > Single({s:.1f}%): "
                  f"{'[PASS]' if rank_ok else '[FAIL]'}")
        row["all_ranked_ok"] = row["MHOW_ranked_ok"] and row["CRBG_ranked_ok"]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_ranking_summary(df_carb, df_pco2, save_path):
    """
    Generate comprehensive multi-panel publication-grade summary plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. CARB_CAP Base - MHOW
    ax = axes[0, 0]
    x_labels = [r["setting"] for _, r in df_carb.iterrows()]
    x_idx = np.arange(len(x_labels))
    ax.plot(x_idx, df_carb["MHOW_pulsed_eff"], 'o-', color='#2ca02c', label='Pulsed (WAG)', lw=2)
    ax.plot(x_idx, df_carb["MHOW_continuous_eff"], 's-', color='#1f77b4', label='Continuous', lw=2)
    ax.plot(x_idx, df_carb["MHOW_single_eff"], '^-', color='#d62728', label='Single-burst', lw=2)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=9)
    ax.set_title("MHOW Basalt: Sensitivity to CARB_CAP Base", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mineralisation Efficiency (%)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)

    # 2. CARB_CAP Base - CRBG
    ax = axes[0, 1]
    ax.plot(x_idx, df_carb["CRBG_pulsed_eff"], 'o-', color='#2ca02c', label='Pulsed (WAG)', lw=2)
    ax.plot(x_idx, df_carb["CRBG_continuous_eff"], 's-', color='#1f77b4', label='Continuous', lw=2)
    ax.plot(x_idx, df_carb["CRBG_single_eff"], '^-', color='#d62728', label='Single-burst', lw=2)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=9)
    ax.set_title("CRBG Basalt: Sensitivity to CARB_CAP Base", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mineralisation Efficiency (%)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)

    # 3. log_pco2_off_floor - MHOW
    ax = axes[1, 0]
    x_labels_p = [f"{r['value']:.2f}" for _, r in df_pco2.iterrows()]
    x_idx_p = np.arange(len(x_labels_p))
    ax.plot(x_idx_p, df_pco2["MHOW_pulsed_eff"], 'o-', color='#2ca02c', label='Pulsed (WAG)', lw=2)
    ax.plot(x_idx_p, df_pco2["MHOW_continuous_eff"], 's-', color='#1f77b4', label='Continuous', lw=2)
    ax.plot(x_idx_p, df_pco2["MHOW_single_eff"], '^-', color='#d62728', label='Single-burst', lw=2)
    ax.set_xticks(x_idx_p)
    ax.set_xticklabels(x_labels_p, fontsize=10)
    ax.set_xlabel(r"Residual $\log(p\mathrm{CO}_2)$ Floor (bar)", fontsize=10)
    ax.set_title(r"MHOW Basalt: Sensitivity to $\log(p\mathrm{CO}_2)$ Off Floor", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mineralisation Efficiency (%)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)

    # 4. log_pco2_off_floor - CRBG
    ax = axes[1, 1]
    ax.plot(x_idx_p, df_pco2["CRBG_pulsed_eff"], 'o-', color='#2ca02c', label='Pulsed (WAG)', lw=2)
    ax.plot(x_idx_p, df_pco2["CRBG_continuous_eff"], 's-', color='#1f77b4', label='Continuous', lw=2)
    ax.plot(x_idx_p, df_pco2["CRBG_single_eff"], '^-', color='#d62728', label='Single-burst', lw=2)
    ax.set_xticks(x_idx_p)
    ax.set_xticklabels(x_labels_p, fontsize=10)
    ax.set_xlabel(r"Residual $\log(p\mathrm{CO}_2)$ Floor (bar)", fontsize=10)
    ax.set_title(r"CRBG Basalt: Sensitivity to $\log(p\mathrm{CO}_2)$ Off Floor", fontsize=11, fontweight='bold')
    ax.set_ylabel("Mineralisation Efficiency (%)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=9)

    plt.suptitle("Cross-Mode Ranking Robustness (Pulsed > Continuous > Single)\n"
                 "Mode-Independent XRF-Derived Mineralisation Throttle",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[FIGURE SAVED] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Mode Ranking Robustness Sensitivity Sweep")
    parser.add_argument("--quick", action="store_true", help="Run 3 points per parameter instead of full set")
    parser.add_argument("--param", choices=["all", "carb_cap", "pco2_floor"], default="all")
    args = parser.parse_args()

    carb_factors = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00] if not args.quick else [0.50, 1.00, 1.50]
    pco2_values  = [-3.0, -2.5, -2.0, -1.5] if not args.quick else [-3.0, -2.5, -2.0]

    df_carb = None
    df_pco2 = None

    if args.param in ["all", "carb_cap"]:
        df_carb = run_single_sweep(
            param_name="carb_cap_base",
            param_values=carb_factors,
            is_factor=True,
            base_val=CONFIG["kinetics"]["carb_cap_base"]
        )
        carb_xlsx = OUTPUT_DIR / "sensitivity_ranking_carb_cap.xlsx"
        df_carb.to_excel(carb_xlsx, index=False)
        print(f"[SAVED] {carb_xlsx}")

    if args.param in ["all", "pco2_floor"]:
        df_pco2 = run_single_sweep(
            param_name="log_pco2_off_floor",
            param_values=pco2_values,
            is_factor=False
        )
        pco2_xlsx = OUTPUT_DIR / "sensitivity_ranking_pco2_floor.xlsx"
        df_pco2.to_excel(pco2_xlsx, index=False)
        print(f"[SAVED] {pco2_xlsx}")

    if df_carb is not None and df_pco2 is not None:
        plot_path = OUTPUT_DIR / "sensitivity_ranking_heatmap.png"
        plot_ranking_summary(df_carb, df_pco2, plot_path)

        with pd.ExcelWriter(OUTPUT_DIR / "sensitivity_ranking_summary.xlsx") as writer:
            df_carb.to_excel(writer, sheet_name="carb_cap_base", index=False)
            df_pco2.to_excel(writer, sheet_name="log_pco2_off_floor", index=False)
        print(f"[SAVED] {OUTPUT_DIR / 'sensitivity_ranking_summary.xlsx'}")

    print("\n" + "="*78)
    print("SENSITIVITY SWEEP RUN COMPLETE!")
    print("="*78)


if __name__ == "__main__":
    main()
