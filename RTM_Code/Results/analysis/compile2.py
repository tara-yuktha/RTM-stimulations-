"""Compile round-2 matrix runs into net-efficiency tables (reusing round-1 single/continuous for R_* configs)."""
import json, collections
from pathlib import Path as _P
RES = _P(__file__).resolve().parents[1]      # round2_results/
CODE = _P(__file__).resolve().parents[2]     # the model code
MATRIX_LOG = RES / 'matrix_runs.jsonl'
R1_LOG = RES / 'round1_results.jsonl'
REG = ('MHOW', 'CRBG'); MOD = ('single', 'continuous', 'pulsed')
runs = {}
for l in open(MATRIX_LOG):
    d = json.loads(l)
    if 'eff' in d:
        runs[(d['config'], d['region'], d['mode'], d['control'])] = d
# round-1 reuse map: R_* config -> round-1 tag (single/continuous only)
R1MAP = {'R_noaff': 'sens_noaff', 'R_noaff_sa0.1': 'sens_noaff_sa0.1', 'R_k0.1': 'sens_k0.1', 'R_chl': 'sens_chl',
         'R_zeo': 'sens_zeo', 'R_allphases': 'sens_allphases', 'R_grid20': 'sens_grid20', 'R_dt1600': 'sens_dt1600',
         'R_legacyrates': 'sens_legacyrates'}
r1 = {}
for l in open(R1_LOG):
    d = json.loads(l)
    if 'eff' in d:
        tag = d['tag']; ctrl = tag.endswith('_ctrl'); base = tag[:-5] if ctrl else tag
        r1[(base, d['region'], d['mode'], ctrl)] = d
for cfg, tag in R1MAP.items():
    for reg in REG:
        for m in ('single', 'continuous'):
            for c in (False, True):
                d = r1.get((tag, reg, m, c))
                if d and (cfg, reg, m, c) not in runs:
                    runs[(cfg, reg, m, c)] = dict(config=cfg, region=reg, mode=m, control=c, eff=d['eff'],
                        eff_1yr=d.get('eff_1yr'), eff_2yr=d.get('eff_2yr'), pH_min=d['pH_min'], dphi=d['dphi'],
                        closure=d['closure'], exported_mol=d.get('C_exp', 0.0), co2_sched_mol=d.get('co2_sched', 0.0),
                        reused_round1=True)
def net(cfg, reg, m, key='eff'):
    a = runs.get((cfg, reg, m, False)); b = runs.get((cfg, reg, m, True))
    if a is None or b is None or a.get(key) is None or b.get(key) is None:
        return None
    return a[key] - b[key]
configs = sorted({k[0] for k in runs}, key=lambda x: list(dict.fromkeys(k[0] for k in runs)).index(x))
table = collections.OrderedDict()
for cfg in configs:
    row = {}
    for reg in REG:
        for m in MOD:
            v = net(cfg, reg, m)
            if v is not None:
                a = runs[(cfg, reg, m, False)]
                row[f'{reg}_{m}'] = dict(net=v, net_1yr=net(cfg, reg, m, 'eff_1yr'), net_2yr=net(cfg, reg, m, 'eff_2yr'),
                                         gross=a['eff'], ctrl=runs[(cfg, reg, m, True)]['eff'], pH_min=a['pH_min'],
                                         dphi=a['dphi'], closure=a['closure'],
                                         export_pct=(a.get('exported_mol', 0) / a['co2_sched_mol'] * 100 if a.get('co2_sched_mol') else None),
                                         hyd=a.get('hydraulics', {}), minerals=a.get('minerals', {}))
    table[cfg] = row
json.dump(table, open(RES / 'compiled2.json', 'w'), indent=1, default=str)
def fmt(v): return '   -  ' if v is None else f'{v:6.1f}'
print(f"{'config':20s} " + " ".join(f"{r[:4]}-{m[:4]:4s}" for r in REG for m in MOD))
for cfg, row in table.items():
    print(f"{cfg:20s} " + " ".join(fmt(row.get(f'{r}_{m}', {}).get('net')) for r in REG for m in MOD))
