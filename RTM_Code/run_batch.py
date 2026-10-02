import sys, os
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

import copy
from run_rtm import CONFIG, run_simulation, make_plots, save_excel, compute_schedules

# [FIX-15] each scenario is also run as a no-injection control (same flow and
# pressure history, no CO2) so that the reported efficiency is net of carbon
# mineralised from native formation water.  Set RUN_CONTROLS = False to skip.
RUN_CONTROLS = True

scenarios = [
    ("MHOW", "single"),
    ("MHOW", "continuous"),
    ("MHOW", "pulsed"),
    ("CRBG", "single"),
    ("CRBG", "continuous"),
    ("CRBG", "pulsed"),
]

results = {}

for region, mode in scenarios:
    print(f"\n{'='*72}\nSTARTING SIMULATION: {region} | {mode.upper()}\n{'='*72}")
    s = compute_schedules(region, CONFIG)
    r = run_simulation(region, mode, CONFIG)
    make_plots(r, CONFIG["xrf"][region])
    save_excel(r, CONFIG["xrf"][region])
    eff = r['efficiency_arr'][-1]
    eff_ctrl = float('nan')
    if RUN_CONTROLS:
        cfg_c = copy.deepcopy(CONFIG); cfg_c["control_no_injection"] = True
        rc = run_simulation(region, mode, cfg_c)
        save_excel(rc, CONFIG["xrf"][region])
        eff_ctrl = rc['efficiency_arr'][-1]
    results[(region, mode)] = {
        'eff': eff,
        'eff_ctrl': eff_ctrl,
        'eff_net': eff - eff_ctrl if eff_ctrl == eff_ctrl else float('nan'),
        'pH_nadir': r['pH'].min(),
        'pH_final': r['pH'][-1],
        'por_init': r['porosity'][0],
        'por_final': r['porosity'][-1],
        'mb_err': r['mass_balance_err_pct']
    }
    print(f"\n[DONE] {region} | {mode.upper()} -> Efficiency: {eff:.1f}%, pH nadir: {r['pH'].min():.2f}, pH final: {r['pH'][-1]:.2f}")

print("\n" + "="*72)
print("BATCH RUN COMPLETE SUMMARY:")
print(f"Engine: {r.get('engine_name', '?')}")
print(f"{'Region':<8} {'Mode':<12} {'Eff_%':>7} {'Ctrl_%':>7} {'Net_%':>7} {'pH_min':>7} {'pH_fin':>7} {'dPhi_pp':>8} {'C_err_%':>9}")
print("-" * 84)
for (region, mode), d in results.items():
    print(f"{region:<8} {mode:<12} {d['eff']:>7.1f} {d['eff_ctrl']:>7.1f} {d['eff_net']:>7.1f} "
          f"{d['pH_nadir']:>7.2f} {d['pH_final']:>7.2f} {d['por_final']-d['por_init']:>8.3f} {d['mb_err']:>9.2e}")
print("="*84)
