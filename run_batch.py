import sys, os
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from pathlib import Path

HERE = Path(r"E:\Whats Next\Manuscript submission\Injection Code")
sys.path.insert(0, str(HERE))

from run_rtm import CONFIG, run_simulation, make_plots, save_excel, compute_schedules

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
    results[(region, mode)] = {
        'eff': eff,
        'pH_nadir': r['pH'].min(),
        'pH_final': r['pH'][-1],
        'por_init': r['porosity'][0],
        'por_final': r['porosity'][-1],
        'mb_err': r['mass_balance_err_pct']
    }
    print(f"\n[DONE] {region} | {mode.upper()} -> Efficiency: {eff:.1f}%, pH nadir: {r['pH'].min():.2f}, pH final: {r['pH'][-1]:.2f}")

print("\n" + "="*72)
print("BATCH RUN COMPLETE SUMMARY:")
print(f"{'Region':<8} {'Mode':<12} {'Eff_%':<8} {'pH_min':<8} {'pH_fin':<8} {'Por_%':<12} {'MB_err_%':<8}")
print("-" * 72)
for (region, mode), d in results.items():
    print(f"{region:<8} {mode:<12} {d['eff']:<8.1f} {d['pH_nadir']:<8.2f} {d['pH_final']:<8.2f} {d['por_init']:.1f}->{d['por_final']:.1f}%   {d['mb_err']:<8.2f}")
print("="*72)
