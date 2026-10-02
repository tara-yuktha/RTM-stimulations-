import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import harness2 as H
from run_sensitivity_matrix import A1_MC, A1_EX
base = {"MHOW": 0.35, "CRBG": 0.25}
out = HERE.parents[0] / "tau_results.jsonl"
done = set()
if out.exists():
    for l in open(out): j = json.loads(l); done.add((j['f'], j['region'], j['mode']))
for f in (0.5, 2.0):
    for reg in ("MHOW", "CRBG"):
        for mode in ("single", "continuous", "pulsed"):
            if (f, reg, mode) in done: continue
            ex = dict(A1_EX); ex[f"injection/tau_p_decay/{reg}"] = base[reg] * f
            t0 = time.time()
            s, r, lg = H.run(reg, mode, H.make_cfg(mc=A1_MC, extra=ex), "tau")
            with open(out, "a") as fh:
                fh.write(json.dumps(dict(f=f, region=reg, mode=mode, eff=s['eff'], eff_1yr=s.get('eff_1yr'), eff_2yr=s.get('eff_2yr'), closure=s['closure'])) + "\n")
            print("done", f, reg, mode, round(s['eff'], 2), round(time.time() - t0), flush=True)
print("ALL DONE TAU", flush=True)
