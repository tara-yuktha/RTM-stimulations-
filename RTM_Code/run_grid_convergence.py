"""
run_grid_convergence.py
=======================
Spatial and temporal discretisation study for Reactive Transport Model.

Addresses:
  Reviewer 1, Major Comment 8  ("a basic grid-convergence comparison, for
                                example 10, 20 and 50 cells")
  Reviewer 2, Comment 8        ("grid and time-step independence ... should be
                                demonstrated")

Sweeps:
  (a) SPATIAL  : n_cells = 10, 20, 50 at fixed n_steps = 800
  (b) TEMPORAL : n_steps = 400, 800, 1600 at fixed n_cells = 10

Outputs:
  - Prints summary convergence table
  - Saves RTM_Output/results_grid_convergence.csv

Usage:
  python run_grid_convergence.py [--quick] [--site MHOW|CRBG|ALL]
"""

import sys
import os
import copy
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from run_rtm import CONFIG, run_simulation, PRIMARY_MINERALS

OUTPUT_DIR = HERE / "RTM_Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUTPUT_DIR / "results_grid_convergence.csv"


def run_single_grid_case(region, mode, n_cells, n_steps, sweep_type):
    cfg = copy.deepcopy(CONFIG)
    cfg["transport"]["n_cells"] = n_cells
    cfg["injection"]["n_steps"] = n_steps
    for rp_k in cfg["region_params"]:
        cfg["region_params"][rp_k]["transport_n_cells"] = n_cells
        cfg["region_params"][rp_k]["n_steps"] = n_steps

    t0 = time.time()
    try:
        r = run_simulation(region, mode, cfg)
        dt_sim = time.time() - t0
        eff = float(r["efficiency_arr"][-1])
        ph_nadir = float(r["pH"].min())
        d_phi = float(r["porosity"][-1] - r["porosity"][0])
        
        # Total primary dissolved moles
        prim_diss = sum(float(r["dissolved"][m][-1]) for m in PRIMARY_MINERALS)
        
        # Transport steps percentage
        census = r.get("solver_census", {})
        n_tot = census.get("n_steps", n_steps)
        n_transp = census.get("n_transport", 0)
        transp_pct = (n_transp / max(n_tot, 1)) * 100.0
        
        # Carbon closure error
        cb = r.get("carbon_balance")
        closure_err = cb.closure_error_pct if cb is not None else float(r.get("mass_balance_err_pct", 0.0))

        return {
            "sweep": sweep_type,
            "region": region,
            "mode": mode,
            "cells_or_steps": n_cells if sweep_type == "spatial" else n_steps,
            "efficiency_pct": round(eff, 2),
            "pH_nadir": round(ph_nadir, 2),
            "delta_porosity_pct": round(d_phi, 4),
            "total_primary_dissolved_mol": f"{prim_diss:.4e}",
            "transport_steps_pct": round(transp_pct, 1),
            "C_closure_err_pct": round(closure_err, 2),
            "n_steps": n_steps,
            "runtime_s": int(dt_sim),
            "status": "OK"
        }
    except Exception as e:
        dt_sim = time.time() - t0
        return {
            "sweep": sweep_type,
            "region": region,
            "mode": mode,
            "cells_or_steps": n_cells if sweep_type == "spatial" else n_steps,
            "efficiency_pct": float("nan"),
            "pH_nadir": float("nan"),
            "delta_porosity_pct": float("nan"),
            "total_primary_dissolved_mol": float("nan"),
            "transport_steps_pct": float("nan"),
            "C_closure_err_pct": float("nan"),
            "n_steps": n_steps,
            "runtime_s": int(dt_sim),
            "status": f"FAIL: {e}"
        }


def _grid_worker(args):
    region, mode, n_cells, n_steps, sweep_type = args
    return run_single_grid_case(region, mode, n_cells, n_steps, sweep_type)


def main():
    import multiprocessing as mp
    parser = argparse.ArgumentParser(description="Run RTM grid and time-step convergence study")
    parser.add_argument("--quick", action="store_true", help="Run quick 2-point check")
    parser.add_argument("--site", choices=["MHOW", "CRBG", "ALL"], default="ALL")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel worker processes")
    args = parser.parse_args()

    sites = ["MHOW", "CRBG"] if args.site == "ALL" else [args.site]
    modes = ["single", "continuous", "pulsed"]

    if args.quick:
        spatial_cells = [10, 20]
        temporal_steps = [200, 400]
        base_steps = 400
        base_cells = 10
    else:
        spatial_cells = [10, 20, 50]
        temporal_steps = [400, 800, 1600]
        base_steps = 800
        base_cells = 10

    print("=" * 80)
    print("REACTIVE TRANSPORT MODEL — GRID & TIME-STEP INDEPENDENCE STUDY")
    print("=" * 80)
    print(f"Sites: {sites} | Modes: {modes} | Workers: {args.workers}")
    print(f"Spatial cell sweep:  {spatial_cells} (at fixed n_steps = {base_steps})")
    print(f"Temporal step sweep: {temporal_steps} (at fixed n_cells = {base_cells})")
    print("=" * 80)

    # Build work queue
    work_items = []
    # 1. Spatial sweep
    for site in sites:
        for mode in modes:
            for n_c in spatial_cells:
                work_items.append((site, mode, n_c, base_steps, "spatial"))

    # 2. Temporal sweep
    for site in sites:
        for mode in modes:
            for n_s in temporal_steps:
                work_items.append((site, mode, base_cells, n_s, "temporal"))

    print(f"Queueing {len(work_items)} simulation runs across {args.workers} worker processes...")
    t_start = time.time()
    results = []

    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_grid_worker, work_items), 1):
                results.append(res)
                print(f"[{i:2d}/{len(work_items)}] [{res['sweep'].upper():8s}] {res['region']} | {res['mode']:10s} | {res['cells_or_steps']:4d} -> Eff={res['efficiency_pct']}% | pH={res['pH_nadir']} | closure={res['C_closure_err_pct']}% ({res['runtime_s']}s)")
    else:
        for i, item in enumerate(work_items, 1):
            res = _grid_worker(item)
            results.append(res)
            print(f"[{i:2d}/{len(work_items)}] [{res['sweep'].upper():8s}] {res['region']} | {res['mode']:10s} | {res['cells_or_steps']:4d} -> Eff={res['efficiency_pct']}% | pH={res['pH_nadir']} | closure={res['C_closure_err_pct']}% ({res['runtime_s']}s)")

    t_total = time.time() - t_start
    print(f"\nAll {len(work_items)} runs completed in {t_total:.1f}s ({t_total/len(work_items):.2f}s per run avg)!")

    # Sort results for tidy presentation
    df = pd.DataFrame(results)
    df.sort_values(by=["sweep", "region", "mode", "cells_or_steps"], inplace=True)
    df.to_csv(CSV_OUT, index=False)
    print("\n" + "=" * 80)
    print(f"[SUCCESS] Grid convergence study complete. Saved: {CSV_OUT}")
    print("=" * 80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
