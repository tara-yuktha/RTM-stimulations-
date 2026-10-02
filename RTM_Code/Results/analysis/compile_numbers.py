import json, pandas as pd, numpy as np
from pathlib import Path as _P
_RES = _P(__file__).resolve().parents[1]
_ROOT = _RES.parent
P=str(_RES) + '/'
df=pd.read_csv(P+'matrix_summary.csv')
L=json.load(open(P+'ledger_round2.json')); LL=json.load(open(P+'ledger_long.json'))
def net(cfg,reg,mode,col='net'):
    r=df[(df.config==cfg)&(df.region==reg)&(df['mode']==mode)]
    return float(r[col].iloc[0]) if len(r) else None
out={}
# base tables from ledger
def base(cfg):
    T={}
    for reg in ('MHOW','CRBG'):
        for mode in ('single','continuous','pulsed'):
            e=L[f'{cfg}|{reg}|{mode}|False']; c=L[f'{cfg}|{reg}|{mode}|True']
            s=e['co2_sched']
            T[f'{reg}|{mode}']=dict(gross=e['eff'],ctrl=c['eff'],net=e['eff']-c['eff'],
              net1=net(cfg,reg,mode,'net_1yr'),net2=net(cfg,reg,mode,'net_2yr'),
              pHmin=e['pH_min'],pHfin=e['pH_final'],dphi=e['dphi'],closure=e['closure'],
              C_min_pct=e['C_min']/s*100,C_aq_pct=e['C_aq']/s*100,C_free_pct=e['C_free']/s*100,C_exp_pct=e['C_exp']/s*100,
              C_init=e['C_init'],C_inflow=e['C_inflow'],C_inj=e['C_inj'],C_min=e['C_min'],C_aq=e['C_aq'],C_free=e['C_free'],C_exp=e['C_exp'],sched=s,
              diss=e['diss'],diss_total=e['diss_total'],minerals=e['minerals_final'],hyd=e['hyd'],
              ctrl_minerals=c['minerals_final'])
    return T
out['A1']=base('A1'); out['A0']=base('A0')
json.dump(out,open(_RES / 'base_numbers.json','w'),indent=1,default=str)
def row(cfg,modes=('single','continuous','pulsed')):
    r=[]
    for reg in ('MHOW','CRBG'):
        r.append('/'.join('%5.1f'%net(cfg,reg,m) if net(cfg,reg,m) is not None else '  -  ' for m in modes))
    return ' | '.join(r)
print('A0 net (single/cont/pulsed) MHOW | CRBG'); print(row('A0'))
for k in ('MHOW','CRBG'):
    for m in ('single','continuous','pulsed'):
        t=out['A0'][f'{k}|{m}']; h=t['hyd']
        print('A0',k,m,'gross %.1f ctrl %.1f net %.1f'%(t['gross'],t['ctrl'],t['net']),'PV displaced %.1f'%h['pore_volumes_displaced'],'exp%% %.1f'%t['C_exp_pct'],'pHmin %.2f'%t['pHmin'],'dphi %.3f'%t['dphi'],'vel', )
print()
print('A1 partition % of scheduled (min/aq/free/exp), pHmin, dphi')
for k in ('MHOW','CRBG'):
    for m in ('single','continuous','pulsed'):
        t=out['A1'][f'{k}|{m}']
        print(k,m,'net %.1f (1y %.1f, 2y %.1f)'%(t['net'],t['net1'],t['net2']),'| gross %.1f ctrl %.2f'%(t['gross'],t['ctrl']),'|',' / '.join('%.1f'%t[x] for x in ('C_min_pct','C_aq_pct','C_free_pct','C_exp_pct')),'| pH %.2f -> %.2f | dphi %.3f'%(t['pHmin'],t['pHfin'],t['dphi']))
print()
cfgs=[c for c in df.config.unique()]
print('20-yr (gross, ctrl, net):')
for k in ('MHOW','CRBG'):
    for m in ('continuous','pulsed'):
        e=LL[f'{k}|{m}|False']; c=LL[f'{k}|{m}|True']
        print(k,m,'gross %.2f ctrl %.2f net %.2f'%(e['eff'],c['eff'],e['eff']-c['eff']),'aq %.1f free %.1f exp %.2f'%(e['C_aq']/e['co2_sched']*100,e['C_free']/e['co2_sched']*100,e['C_exp']/e['co2_sched']*100))
