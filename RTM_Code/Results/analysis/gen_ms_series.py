"""Regenerate time series for the base injection-driven (A1) cases used in the revised manuscript figures."""
import sys, os, pickle, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
import harness2 as H
from run_sensitivity_matrix import A1_MC, A1_EX
OUT = HERE.parents[0] / "cache"
def tonp(v):
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float, np.floating, np.integer)):
        return np.asarray(v, float)
    if isinstance(v, dict):
        return {k: tonp(x) for k, x in v.items()}
    return v
for reg in ("MHOW", "CRBG"):
    for mode in ("single", "continuous", "pulsed"):
        for ctrl in (False, True):
            f = OUT / f"A1_{reg}_{mode}_{'ctrl' if ctrl else 'inj'}.pkl"
            if f.exists(): continue
            t0 = time.time()
            s, r, log = H.run(reg, mode, H.make_cfg(mc=A1_MC, control=ctrl, extra=A1_EX), "ms", keep=True)
            keep = {}
            for k, v in r.items():
                try:
                    pickle.dumps(v); keep[k] = tonp(v)
                except Exception:
                    pass
            keep["_summary"] = s
            keep["_log_tail"] = log[-3000:]
            pickle.dump(keep, open(f, "wb"))
            print("done", f.name, round(s["eff"], 3), round(time.time() - t0), flush=True)
print("ALL DONE", flush=True)
