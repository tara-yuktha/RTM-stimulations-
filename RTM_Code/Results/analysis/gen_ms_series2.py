"""Second batch: A0 time series (12 runs) + Mhow dissolution_boost=1.0 sensitivity on A1 (continuous & WAG, with controls)."""
import sys, os, pickle, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import numpy as np
import harness2 as H
from run_sensitivity_matrix import A1_MC, A1_EX
OUT = HERE.parents[0] / "cache"
# wait for batch 1
log = HERE.parents[0] / "gen_ms.log"
while "ALL DONE" not in (log.read_text() if log.exists() else ""):
    time.sleep(10)
def tonp(v):
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float, np.floating, np.integer)):
        return np.asarray(v, float)
    if isinstance(v, dict):
        return {k: tonp(x) for k, x in v.items()}
    return v
def save(f, s, r, log_):
    keep = {}
    for k, v in r.items():
        try:
            pickle.dumps(v); keep[k] = tonp(v)
        except Exception:
            pass
    keep["_summary"] = s; keep["_log_tail"] = log_[-3000:]
    pickle.dump(keep, open(f, "wb"))
for reg in ("MHOW", "CRBG"):
    for mode in ("single", "continuous", "pulsed"):
        for ctrl in (False, True):
            f = OUT / f"A0_{reg}_{mode}_{'ctrl' if ctrl else 'inj'}.pkl"
            if f.exists(): continue
            t0 = time.time()
            s, r, lg = H.run(reg, mode, H.make_cfg(control=ctrl), "ms0", keep=True)
            save(f, s, r, lg); print("done", f.name, round(s["eff"], 3), round(time.time() - t0), flush=True)
for mode in ("continuous", "pulsed"):
    for ctrl in (False, True):
        f = OUT / f"A1boost1_MHOW_{mode}_{'ctrl' if ctrl else 'inj'}.pkl"
        if f.exists(): continue
        t0 = time.time()
        ex = dict(A1_EX); ex["region_params/MHOW/dissolution_boost"] = 1.0
        s, r, lg = H.run("MHOW", mode, H.make_cfg(mc=A1_MC, control=ctrl, extra=ex), "boost", keep=True)
        save(f, s, r, lg); print("done", f.name, round(s["eff"], 3), round(time.time() - t0), flush=True)
print("ALL DONE 2", flush=True)
