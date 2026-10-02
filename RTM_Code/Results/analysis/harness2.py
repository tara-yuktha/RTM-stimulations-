import sys, os, io, time, json, copy, contextlib
from pathlib import Path as _P
RES = _P(__file__).resolve().parents[1]      # round2_results/
CODE = _P(__file__).resolve().parents[2]     # the model code
FIXED = str(CODE)
os.environ.setdefault("RTM_OUTPUT_ROOT", str(CODE / "RTM_Output"))
sys.path.insert(0, FIXED)
import run_rtm
from run_rtm import CONFIG, run_simulation

ALL_OFF = {k: False for k in CONFIG["revision_fixes"]}

def make_cfg(fixes=None, mc=None, control=False, extra=None):
    cfg = copy.deepcopy(CONFIG)
    cfg["phreeqc_db"] = os.path.join(FIXED, "llnl.dat")
    if fixes is not None:
        cfg["revision_fixes"] = dict(fixes)
    if mc:
        cfg["mc_engine"].update(mc)
    cfg["control_no_injection"] = control
    if extra:
        for path, val in extra.items():
            d = cfg
            keys = path.split("/")
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = val
    return cfg

def summarize(r):
    cb = r["carbon_balance"]
    mins = r["minerals"]
    out = dict(region=r["region"], mode=r["mode"], control=r.get("control_no_injection"),
               eff=float(r["efficiency_arr"][-1]),
               pH_min=float(r["pH"].min()), pH_final=float(r["pH"][-1]),
               por0=float(r["porosity"][0]), por_final=float(r["porosity"][-1]),
               dphi=float(r["porosity"][-1] - r["porosity"][0]),
               closure=float(r["mass_balance_err_pct"]),
               cb_closure=float(cb.closure_error_pct),
               C_inj=cb.injected_mol, C_aq=cb.aqueous_mol, C_min=cb.mineral_mol,
               C_exp=cb.exported_mol, C_free=cb.gas_phase_mol,
               C_init=getattr(cb, "initial_mol", 0.0), C_inflow=getattr(cb, "inflow_mol", 0.0),
               census=r["solver_census"], carb_cap=float(r["carb_cap_eff"]),
               eta=float(r["sweep_efficiency"]),
               diss_total=float(sum(r["dissolved"][m][-1] for m in r["dissolved"])),
               diss={m: float(r["dissolved"][m][-1]) for m in r["dissolved"]},
               minerals_final={m: float(mins[m][-1]) for m in mins},
               minerals_max={m: float(max(mins[m])) for m in mins},
               n_steps=len(r["time"]), vel_min=float(min(r["vel_arr"])), vel_max=float(max(r["vel_arr"])),
               co2_sched=float(r.get("co2_scheduled_mol", 0.0)),
               engine=r.get("engine_name"))
    out["hyd"] = r.get("hydraulics", {})
    if "C_mineral_net_arr" in r and r["revision_fixes"].get("mass_conservative_engine"):
        t = r["time"]; e = r["efficiency_arr"]
        import numpy as np
        out["eff_1yr"] = float(np.interp(1.0, t, e)); out["eff_2yr"] = float(np.interp(2.0, t, e))
        out["eff_peak"] = float(max(e))
        out["C_exp_frac"] = float(cb.exported_mol / max(r.get("co2_scheduled_mol", 1.0), 1.0))
        out["pH_mean_min"] = float(np.nanmin(r["pH_mean_arr"]))
    return out

def run(region, mode, cfg, tag="", keep=False):
    buf = io.StringIO(); t0 = time.time()
    with contextlib.redirect_stdout(buf):
        r = run_simulation(region, mode, cfg)
    s = summarize(r); s["tag"] = tag; s["runtime_s"] = round(time.time() - t0, 1)
    log = buf.getvalue(); s["n_warn"] = log.count("[WARN]")
    return (s, r, log) if keep else (s, None, log)

if __name__ == "__main__":
    region, mode, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    fx = None if tag.startswith("mc") else ALL_OFF
    s, r, log = run(region, mode, make_cfg(fixes=fx), tag)
    open(f"log_{tag}_{region}_{mode}.txt", "w").write(log)
    print(json.dumps(s, default=str))
