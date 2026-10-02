"""
run_sensitivity_matrix.py
=========================
Pre-registered sensitivity and verification matrix (revision R2, round 2).
Reviewer 1 comments 2-9 and 17 plus the requested sensitivity matrix;
Reviewer 2 comments 5, 8, 13 and 14.  [FIX-21]

Every configuration below is run for the requested sites and modes, each
together with its own no-injection control (same pressure and flow history,
no CO2).  The quantity compared across modes is the NET efficiency
(run minus control), i.e. carbon mineralised because of the injection.

The settings were fixed before any of these runs was made and are applied
identically to all modes; see round2_plan.md.

Usage
    python run_sensitivity_matrix.py --workers 2              # everything
    python run_sensitivity_matrix.py --only A0 A1 A1_sa0.1    # selected rows
    python run_sensitivity_matrix.py --list                   # show the matrix
Outputs
    RTM_Output/sensitivity_matrix/matrix_runs.jsonl   (one line per run, appended)
    RTM_Output/sensitivity_matrix/matrix_summary.csv  (net efficiencies)
"""

import sys, os, io, json, copy, time, argparse, contextlib, multiprocessing, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from run_rtm import CONFIG, run_simulation, OUTPUT_ROOT

OUT_DIR = Path(OUTPUT_ROOT) / "sensitivity_matrix"
REGIONS = ("MHOW", "CRBG")
MODES = ("single", "continuous", "pulsed")

# ── building blocks ──────────────────────────────────────────────────────────
SA01 = {"kinetics/access_fraction": 0.025}        # reactive surface x0.1
SA001 = {"kinetics/access_fraction": 0.0025}      # reactive surface x0.01
# Injection-driven flow, refined near-inlet domain (plume advances < 1 m)
A1_MC = {"flow_model": "injection_driven", "domain_length_m": 5.0}
A1_EX = {"transport/n_cells": 50}
for _r in REGIONS:
    A1_EX[f"region_params/{_r}/transport_n_cells"] = 50
    A1_EX[f"region_params/{_r}/transport_disp_m"] = 0.1
SHUT = {"pulsed_off_mode": "shut_in"}
DESIGNS = {  # same CO2 mass, same 1-yr window
    "D4":     {"injection/T_on_yr": 0.125, "injection/T_off_yr": 0.125},
    "D1":     {"injection/T_on_yr": 0.5,   "injection/T_off_yr": 0.5},
    "Dshort": {"injection/T_on_yr": 0.125, "injection/T_off_yr": 0.375},
    "Dlong":  {"injection/T_on_yr": 0.375, "injection/T_off_yr": 0.125},
}


def _merge(*ds):
    out = {}
    for d in ds:
        out.update(d or {})
    return out


def build_matrix():
    M = {}

    def add(name, desc, mc=None, extra=None, modes=MODES, group=""):
        M[name] = dict(desc=desc, mc=mc or {}, extra=extra or {}, modes=tuple(modes), group=group)

    # flow model x reactive surface
    add("A0", "Overpressure-driven flow (round-1 corrected default)", group="base")
    add("A0_sa0.1", "A0, reactive surface x0.1", extra=SA01, group="base")
    add("A0_sa0.01", "A0, reactive surface x0.01", extra=SA001, group="base")
    add("A1", "Injection-driven flow (WAG water adds throughput), 5-m refined domain",
        mc=A1_MC, extra=A1_EX, group="flow")
    add("A1_sa0.1", "A1, reactive surface x0.1", mc=A1_MC, extra=_merge(A1_EX, SA01), group="flow")
    add("A1_sa0.01", "A1, reactive surface x0.01", mc=A1_MC, extra=_merge(A1_EX, SA001), group="flow")
    # pulsed off-period hydraulics (shut-in instead of water)
    add("A0_shutin", "A0, cyclic injection with shut-in off-periods", mc=SHUT, modes=("pulsed",), group="pulsed")
    add("A0_sa0.1_shutin", "A0 SA x0.1, shut-in off-periods", mc=SHUT, extra=SA01, modes=("pulsed",), group="pulsed")
    add("A1_shutin", "A1, shut-in off-periods", mc=_merge(A1_MC, SHUT), extra=A1_EX, modes=("pulsed",), group="pulsed")
    add("A1_sa0.1_shutin", "A1 SA x0.1, shut-in off-periods", mc=_merge(A1_MC, SHUT),
        extra=_merge(A1_EX, SA01), modes=("pulsed",), group="pulsed")
    # pulse design (R1-2)
    for dn, dd in DESIGNS.items():
        add(f"A0_{dn}", f"A0, pulse design {dn}", extra=dd, modes=("pulsed",), group="pulsed")
        add(f"A0_sa0.1_{dn}", f"A0 SA x0.1, pulse design {dn}", extra=_merge(SA01, dd), modes=("pulsed",), group="pulsed")
        add(f"A1_{dn}", f"A1, pulse design {dn}", mc=A1_MC, extra=_merge(A1_EX, dd), modes=("pulsed",), group="pulsed")
        add(f"A1_sa0.1_{dn}", f"A1 SA x0.1, pulse design {dn}", mc=A1_MC,
            extra=_merge(A1_EX, SA01, dd), modes=("pulsed",), group="pulsed")
    # CO2:H2O in-situ volume ratio (only matters with injection-driven flow)
    for rv in (0.5, 2.0):
        add(f"A1_w{rv}", f"A1, WAG water:CO2 volume ratio {rv}", mc=A1_MC,
            extra=_merge(A1_EX, {"injection/wag_water_co2_volume_ratio": rv}), modes=("pulsed",), group="pulsed")
        add(f"A1_sa0.1_w{rv}", f"A1 SA x0.1, WAG water:CO2 volume ratio {rv}", mc=A1_MC,
            extra=_merge(A1_EX, SA01, {"injection/wag_water_co2_volume_ratio": rv}), modes=("pulsed",), group="pulsed")
    # A1 numerical and hydraulic checks (added after the first A1 results; see
    # round2_plan.md): inlet-grid resolution and post-injection regional flow
    for nc in (25, 100):
        ex = dict(A1_EX); ex["transport/n_cells"] = nc
        ex.update({f"region_params/{r}/transport_n_cells": nc for r in REGIONS})
        add(f"A1_grid{nc}", f"A1 with {nc} cells in the 5-m domain", mc=A1_MC, extra=ex, group="flow")
    for amb in (0.1, 1.0):
        add(f"A1_amb{amb:g}", f"A1 with {amb:g} m/yr regional Darcy flux (during and after injection)",
            mc=_merge(A1_MC, {"ambient_darcy_m_yr": amb}), extra=A1_EX, group="flow")
    # [FIX-25] uniform time grid (same step for every mode) at 800 and 1600 steps: checks
    # that the mode-specific adaptive grid does not bias the A1 comparison
    for ns in (800, 1600):
        add(f"A1_tu{ns}", f"A1, uniform time grid, {ns} steps", mc=A1_MC,
            extra=_merge(A1_EX, {"injection/time_grid": "uniform", "injection/n_steps": ns}),
            modes=("continuous", "pulsed"), group="flow")
    # [FIX-24] continuous CO2 with the same water volume co-injected (1:1): separates the
    # effect of the water from that of the alternation (added after the design results)
    add("A1_coinj", "A1, continuous CO2 with 1:1 water co-injected", mc=A1_MC,
        extra=_merge(A1_EX, {"injection/continuous_water_co2_volume_ratio": 1.0}), modes=("continuous",), group="pulsed")
    add("A1_sa0.1_coinj", "A1 SA x0.1, continuous CO2 with 1:1 water co-injected", mc=A1_MC,
        extra=_merge(A1_EX, SA01, {"injection/continuous_water_co2_volume_ratio": 1.0}), modes=("continuous",), group="pulsed")
    # regional-flow sweep and a 50-m domain check (added after the 0.1 and 1 m/yr results,
    # to locate where the ranking changes and to test the 5-m domain truncation at high flux)
    for amb in (0.01, 0.03, 0.3):
        add(f"A1_amb{amb:g}", f"A1 with {amb:g} m/yr regional Darcy flux (during and after injection)",
            mc=_merge(A1_MC, {"ambient_darcy_m_yr": amb}), extra=A1_EX, modes=("continuous", "pulsed"), group="flow")
    L50 = {"transport/n_cells": 100}
    L50.update({f"region_params/{r}/transport_n_cells": 100 for r in REGIONS})
    L50.update({f"region_params/{r}/transport_disp_m": 0.1 for r in REGIONS})
    for amb in (0.3, 1.0):
        add(f"A1_amb{amb:g}_L50", f"A1, whole 50-m column (100 cells), {amb:g} m/yr regional Darcy flux",
            mc={"flow_model": "injection_driven", "domain_length_m": 50.0, "ambient_darcy_m_yr": amb},
            extra=L50, modes=("continuous", "pulsed"), group="flow")
    # [FIX-23] Dshort/Dlong with the TOTAL water volume held at 1:1 (the default "rate"
    # rule gives 3:1 and 1:3); the water volume does not enter the A0 flow, so A1 only
    for dn in ("Dshort", "Dlong"):
        vol = _merge(DESIGNS[dn], {"injection/wag_water_rule": "volume"})
        add(f"A1_{dn}_vol", f"A1, pulse design {dn}, total water volume 1:1", mc=A1_MC,
            extra=_merge(A1_EX, vol), modes=("pulsed",), group="pulsed")
        add(f"A1_sa0.1_{dn}_vol", f"A1 SA x0.1, pulse design {dn}, total water volume 1:1", mc=A1_MC,
            extra=_merge(A1_EX, SA01, vol), modes=("pulsed",), group="pulsed")
    # sensitivity matrix on A0 (R1 matrix, R1-3/4, R2-14, R2-8)
    add("S_glass0.5", "Glass abundance x0.5", mc={"modal_factors": {"BasaltGlass": 0.5}}, group="matrix")
    add("S_glass1.5", "Glass abundance x1.5", mc={"modal_factors": {"BasaltGlass": 1.5}}, group="matrix")
    add("S_diop0.5", "Diopside abundance x0.5", mc={"modal_factors": {"Diopside": 0.5}}, group="matrix")
    add("S_diop1.5", "Diopside abundance x1.5", mc={"modal_factors": {"Diopside": 1.5}}, group="matrix")
    add("S_glassk0.1", "Glass rate constants x0.1", mc={"rate_factors": {"BasaltGlass": 0.1}}, group="matrix")
    add("S_glassk10", "Glass rate constants x10", mc={"rate_factors": {"BasaltGlass": 10.0}}, group="matrix")
    add("S_logk-0.5", "P&K log k of crystalline minerals -0.5", mc={"logk_shift": -0.5}, group="matrix")
    add("S_logk+0.5", "P&K log k of crystalline minerals +0.5", mc={"logk_shift": 0.5}, group="matrix")
    add("S_disp0.5", "Dispersivity 0.5 m", extra={f"region_params/{r}/transport_disp_m": 0.5 for r in REGIONS}, group="matrix")
    add("S_disp2", "Dispersivity 2.0 m", extra={f"region_params/{r}/transport_disp_m": 2.0 for r in REGIONS}, group="matrix")
    add("S_fe0.001", "Porewater Fe(II) 0.001 mmol/kgw", mc={"fe2_porewater_mmol": 0.001}, group="matrix")
    add("S_fe0.1", "Porewater Fe(II) 0.1 mmol/kgw", mc={"fe2_porewater_mmol": 0.1}, group="matrix")
    add("S_dic1", "Initial DIC 1 mmol/kgw", extra={f"region_params/{r}/initial_DIC_mmol": 1.0 for r in REGIONS}, group="matrix")
    add("S_dic5", "Initial DIC 5 mmol/kgw", extra={f"region_params/{r}/initial_DIC_mmol": 5.0 for r in REGIONS}, group="matrix")
    g50 = {"transport/n_cells": 50}
    g50.update({f"region_params/{r}/transport_n_cells": 50 for r in REGIONS})
    add("S_grid50", "50 cells", extra=g50, group="matrix")
    add("S_dt400", "400 time steps", extra={"injection/n_steps": 400}, group="matrix")
    # the same matrix on A1, continuous vs WAG (added after the A1 results: the WAG
    # advantage appears only with injection-driven flow; A1 dispersivity default 0.1 m)
    a1m = {
        "glass0.5": ({"modal_factors": {"BasaltGlass": 0.5}}, {}),
        "glass1.5": ({"modal_factors": {"BasaltGlass": 1.5}}, {}),
        "diop0.5": ({"modal_factors": {"Diopside": 0.5}}, {}),
        "diop1.5": ({"modal_factors": {"Diopside": 1.5}}, {}),
        "glassk0.1": ({"rate_factors": {"BasaltGlass": 0.1}}, {}),
        "glassk10": ({"rate_factors": {"BasaltGlass": 10.0}}, {}),
        "logk-0.5": ({"logk_shift": -0.5}, {}),
        "logk+0.5": ({"logk_shift": 0.5}, {}),
        "disp0.05": ({}, {f"region_params/{r}/transport_disp_m": 0.05 for r in REGIONS}),
        "disp0.2": ({}, {f"region_params/{r}/transport_disp_m": 0.2 for r in REGIONS}),
        "fe0.001": ({"fe2_porewater_mmol": 0.001}, {}),
        "fe0.1": ({"fe2_porewater_mmol": 0.1}, {}),
        "dic1": ({}, {f"region_params/{r}/initial_DIC_mmol": 1.0 for r in REGIONS}),
        "dic5": ({}, {f"region_params/{r}/initial_DIC_mmol": 5.0 for r in REGIONS}),
        "dt400": ({}, {"injection/n_steps": 400}),
    }
    for k, (mc, ex) in a1m.items():
        add(f"A1m_{k}", f"A1 matrix: {k}", mc=_merge(A1_MC, mc), extra=_merge(A1_EX, ex),
            modes=("continuous", "pulsed"), group="matrix_A1")
    # round-1 sensitivities (pulsed re-run with the round-2 time grid)
    r1 = {
        "R_noaff": ("Glass affinity term off", {"glass_affinity": "none"}, {}),
        "R_noaff_sa0.1": ("Glass affinity off + SA x0.1", {"glass_affinity": "none"}, SA01),
        "R_k0.1": ("Permeability x0.1", {}, {f"region_params/MHOW/permeability_mD": 3.0,
                                              f"region_params/CRBG/permeability_mD": 20.0}),
        "R_chl": ("Clinochlore allowed (LLNL)", {"secondary_phases": ["Calcite", "Siderite", "Kaolinite",
                   "Saponite-Mg", "SiO2(am)", "Fe(OH)3", "Clinochlore-14A"]}, {}),
        "R_zeo": ("Zeolites allowed", {"secondary_phases": ["Calcite", "Siderite", "Kaolinite", "Saponite-Mg",
                   "SiO2(am)", "Fe(OH)3", "Stilbite", "Analcime", "Mesolite"]}, {}),
        "R_allphases": ("Submitted mineral list", {"secondary_phases": ["Calcite", "Siderite", "Magnesite",
                        "Dolomite", "Ankerite", "Saponite-Mg", "Clinochlore-14A", "Kaolinite", "Muscovite"]}, {}),
        "R_grid20": ("20 cells", {}, dict({"transport/n_cells": 20},
                     **{f"region_params/{r}/transport_n_cells": 20 for r in REGIONS})),
        "R_dt1600": ("1600 time steps", {}, {"injection/n_steps": 1600}),
    }
    # Single-burst and continuous runs are unaffected by the round-2 changes (verified
    # bit-for-bit against round 1), so by default only the pulsed runs are repeated here;
    # pass --r1-all-modes to repeat all three modes.
    r1_modes = MODES if "--r1-all-modes" in sys.argv else ("pulsed",)
    for name, (desc, mc, ex) in r1.items():
        add(name, desc, mc=mc, extra=ex, modes=r1_modes, group="round1")
    add("R_legacyrates", "Submitted (non-P&K) activation energies", group="round1", modes=r1_modes,
        extra={"revision_fixes/pk2004_rate_parameters": False})
    return M


MATRIX = build_matrix()


def make_cfg(spec, control):
    cfg = copy.deepcopy(CONFIG)
    cfg["mc_engine"].update(copy.deepcopy(spec["mc"]))
    for path, val in spec["extra"].items():
        d = cfg
        keys = path.split("/")
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = val
    cfg["control_no_injection"] = control
    return cfg


def _one(args):
    name, region, mode, control = args
    spec = MATRIX[name]
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = run_simulation(region, mode, make_cfg(spec, control))
        t = r["time"]; e = r["efficiency_arr"]
        cb = r["carbon_balance"]
        hyd = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in (r.get("hydraulics") or {}).items()}
        return dict(config=name, region=region, mode=mode, control=control,
                    eff=float(e[-1]), eff_1yr=float(np.interp(1.0, t, e)),
                    eff_2yr=float(np.interp(2.0, t, e)),
                    eff_5yr=float(np.interp(5.0, t, e)), eff_10yr=float(np.interp(10.0, t, e)),
                    pH_min=float(np.min(r["pH"])), pH_final=float(r["pH"][-1]),
                    dphi=float(r["porosity"][-1] - r["porosity"][0]),
                    closure=float(cb.closure_error_pct),
                    exported_mol=float(cb.exported_mol),
                    co2_sched_mol=float(r.get("co2_scheduled_mol", 0.0)),
                    minerals={m: float(v[-1]) for m, v in r["minerals"].items() if v[-1] > 1e-6},
                    hydraulics=hyd, runtime_s=round(time.time() - t0, 1))
    except Exception as exc:
        return dict(config=name, region=region, mode=mode, control=control,
                    error=repr(exc), tb=traceback.format_exc()[-1500:])


def summarise(rows):
    """Net efficiency per configuration/site/mode (run minus its control)."""
    import pandas as pd
    df = pd.DataFrame([r for r in rows if "error" not in r])
    if df.empty:
        return df
    key = ["config", "region", "mode"]
    run = df[~df.control].drop_duplicates(key, keep="last").set_index(key)
    ctl = df[df.control].drop_duplicates(key, keep="last").set_index(key)
    out = run[["eff", "eff_1yr", "eff_2yr", "pH_min", "dphi", "closure", "exported_mol", "co2_sched_mol"]].copy()
    out["eff_ctrl"] = ctl["eff"]
    out["net"] = out["eff"] - out["eff_ctrl"]
    out["net_1yr"] = out["eff_1yr"] - ctl["eff_1yr"]
    out["net_2yr"] = out["eff_2yr"] - ctl["eff_2yr"]
    out["export_pct"] = out["exported_mol"] / out["co2_sched_mol"] * 100.0
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--r1-all-modes", action="store_true",
                    help="repeat the round-1 sensitivities for all modes, not only pulsed")
    args = ap.parse_args()
    names = args.only or list(MATRIX)
    if args.list:
        for n in names:
            print(f"{n:22s} {MATRIX[n]['group']:8s} modes={','.join(MATRIX[n]['modes']):28s} {MATRIX[n]['desc']}")
        return
    tasks = [(n, r, m, c) for n in names for r in REGIONS for m in MATRIX[n]["modes"] for c in (False, True)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "matrix_runs.jsonl"
    print(f"{len(tasks)} runs on {args.workers} workers -> {log}")
    with multiprocessing.Pool(args.workers) as pool, open(log, "a") as fh:
        for res in pool.imap_unordered(_one, tasks):
            fh.write(json.dumps(res, default=str) + "\n"); fh.flush()
            tag = "ctrl" if res["control"] else "run "
            print(f"  {res['config']:20s} {res['region']} {res['mode']:10s} {tag} "
                  + (f"eff={res['eff']:.2f} ({res['runtime_s']} s)" if "eff" in res else res["error"]), flush=True)
    rows = [json.loads(l) for l in open(log)]
    summ = summarise(rows)
    summ.to_csv(OUT_DIR / "matrix_summary.csv", index=False)
    print(f"[SAVED] {OUT_DIR / 'matrix_summary.csv'}")


if __name__ == "__main__":
    main()
