"""
run_carbon_balance.py
=====================
Carbon budget and control-corrected mineralisation efficiency for the six
scenarios (Reviewer 1, comment 10).  [FIX-05/FIX-15]

For each site and injection mode the script runs
  (a) the scenario, and
  (b) a no-injection CONTROL with the same pressure and flow history,
and writes, per scenario, every term of the carbon ledger kept by the
mass-conservative engine (phreeqc_engine_mc.py):

    initial (pore water + minerals at t = 0) + injected + inflow (native water)
        = aqueous + mineral + free-phase CO2 + exported + residual

Efficiencies
    eff_gross   = (mineral C at 5 yr - mineral C at t = 0) / scheduled CO2
    eff_control = the same quantity for the control run
    eff_net     = eff_gross - eff_control   (carbon mineralised BECAUSE of the injection)

Usage
    python run_carbon_balance.py [--workers 2] [--flow overpressure_darcy|injection_driven]
Outputs
    RTM_Output/carbon_balance/carbon_balance.xlsx  and  carbon_balance.csv
"""

import sys, os, io, copy, argparse, contextlib, multiprocessing
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from run_rtm import CONFIG, run_simulation, OUTPUT_ROOT

OUT_DIR = Path(OUTPUT_ROOT) / "carbon_balance"
SCENARIOS = [(r, m) for r in ("MHOW", "CRBG") for m in ("single", "continuous", "pulsed")]


def _run(args):
    region, mode, control = args
    cfg = copy.deepcopy(CONFIG)
    cfg["control_no_injection"] = control
    with contextlib.redirect_stdout(io.StringIO()):
        r = run_simulation(region, mode, cfg)
    cb = r["carbon_balance"]
    t = r["time"]; e = r["efficiency_arr"]
    return {
        "region": region, "mode": mode, "control": control,
        "engine": r.get("engine_name"),
        "CO2_scheduled_mol": r.get("co2_scheduled_mol", float("nan")),
        "C_injected_mol": cb.injected_mol,
        "C_initial_mol": getattr(cb, "initial_mol", 0.0),
        "C_inflow_mol": getattr(cb, "inflow_mol", 0.0),
        "C_aqueous_mol": cb.aqueous_mol,
        "C_mineral_mol": cb.mineral_mol,
        "C_free_phase_mol": cb.gas_phase_mol,
        "C_exported_mol": cb.exported_mol,
        "closure_error_pct": cb.closure_error_pct,
        "eff_5yr_pct": float(e[-1]),
        "eff_1yr_pct": float(np.interp(1.0, t, e)),
        "eff_2yr_pct": float(np.interp(2.0, t, e)),
        "pH_min": float(np.min(r["pH"])),
        "pH_final": float(r["pH"][-1]),
        "dphi_pp": float(r["porosity"][-1] - r["porosity"][0]),
        # [FIX-20] hydraulic comparability (Reviewer 1, comment 9)
        **{f"hyd_{k}": v for k, v in (r.get("hydraulics") or {}).items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--flow", default=None, choices=["overpressure_darcy", "injection_driven"],
                    help="[FIX-17] flow model; injection_driven uses the refined 5-m, 50-cell domain")
    args = ap.parse_args()
    if args.flow == "injection_driven":
        CONFIG["mc_engine"].update({"flow_model": "injection_driven", "domain_length_m": 5.0})
        CONFIG["transport"]["n_cells"] = 50
        for reg in ("MHOW", "CRBG"):
            CONFIG["region_params"][reg]["transport_n_cells"] = 50
            CONFIG["region_params"][reg]["transport_disp_m"] = 0.1
    tasks = [(r, m, c) for (r, m) in SCENARIOS for c in (False, True)]
    with multiprocessing.Pool(args.workers) as pool:
        rows = pool.map(_run, tasks)
    df = pd.DataFrame(rows)
    ctrl = df[df.control].set_index(["region", "mode"])
    run = df[~df.control].set_index(["region", "mode"]).copy()
    for key in ("eff_5yr_pct", "eff_1yr_pct", "eff_2yr_pct"):
        run[key.replace("eff", "eff_control")] = ctrl[key]
        run[key.replace("eff", "eff_net")] = run[key] - ctrl[key]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_DIR / "carbon_balance.xlsx") as w:
        run.reset_index().to_excel(w, sheet_name="scenarios", index=False)
        ctrl.reset_index().to_excel(w, sheet_name="controls", index=False)
    run.reset_index().to_csv(OUT_DIR / "carbon_balance.csv", index=False)
    cols = ["eff_5yr_pct", "eff_control_5yr_pct", "eff_net_5yr_pct", "eff_net_1yr_pct",
            "C_exported_mol", "closure_error_pct"]
    print(run[cols].to_string(float_format=lambda v: f"{v:,.3g}"))
    hcols = [c for c in ("hyd_m_co2_t", "hyd_V_co2_m3", "hyd_m_water_t", "hyd_V_water_m3",
                         "hyd_pore_volumes_displaced", "hyd_residence_time_column_d") if c in run]
    if hcols:
        print("\nHydraulic comparability (R1-9):")
        print(run[hcols].to_string(float_format=lambda v: f"{v:,.4g}"))
    for region in ("MHOW", "CRBG"):
        s, c, p = (run.loc[(region, m), "eff_net_5yr_pct"] for m in ("single", "continuous", "pulsed"))
        print(f"{region}: net 5-yr efficiency single {s:.1f} | continuous {c:.1f} | pulsed {p:.1f}"
              f" -> pulsed-continuous {p - c:+.1f} pp, continuous-single {c - s:+.1f} pp"
              f" (strict ordering P>C>S {'holds' if p > c > s else 'does not hold'})")
    print(f"[SAVED] {OUT_DIR / 'carbon_balance.xlsx'}")


if __name__ == "__main__":
    main()
