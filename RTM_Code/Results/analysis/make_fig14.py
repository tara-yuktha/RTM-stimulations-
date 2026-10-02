import sys, pickle, numpy as np, os
from pathlib import Path as _P
_RES = _P(__file__).resolve().parents[1]
_ROOT = _RES.parent
sys.path.insert(0,str(_ROOT))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
C=str(_RES / 'cache') + '/'; OUT=str(_RES / 'figures') + '/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False})
COL={'single':'#2d7dd2','continuous':'#e8683a','pulsed':'#1aa37a'}; LAB={'single':'Single-burst','continuous':'Continuous','pulsed':'Pulsed WAG'}
def ld(cfg,reg,mode,ctrl): return pickle.load(open(C+f"{cfg}_{reg}_{mode}_{'ctrl' if ctrl else 'inj'}.pkl",'rb'))
def net(cfg,reg,mode):
    a=ld(cfg,reg,mode,False); b=ld(cfg,reg,mode,True)
    return a['time'], a['efficiency_arr']-np.interp(a['time'],b['time'],b['efficiency_arr'])
have_a0=all(os.path.exists(C+f'A0_{r}_{m}_{c}.pkl') for r in ('MHOW','CRBG') for m in ('single','continuous','pulsed') for c in ('inj','ctrl'))
cfgs=[('A0','Overpressure-driven flow'),('A1','Injection-driven flow')] if have_a0 else [('A1','Injection-driven flow')]
fig,axes=plt.subplots(len(cfgs),2,figsize=(9.6,3.9*len(cfgs)),sharex=True,squeeze=False)
for i,(cfg,name) in enumerate(cfgs):
    for j,reg in enumerate(('MHOW','CRBG')):
        ax=axes[i][j]
        for a,b in ((0,0.25),(0.5,0.75)): ax.axvspan(a,b,color='#f4c7c3',alpha=0.5,lw=0)
        for m in ('single','continuous','pulsed'):
            t,y=net(cfg,reg,m); ax.plot(t,y,color=COL[m],lw=2.4,label=LAB[m],ls='--' if m=='single' else '-')
        ax.set_title(f"{name} – {'Mhow' if reg=='MHOW' else 'GRB'}",loc='left',fontsize=13); ax.set_xlim(0,5); ax.grid(alpha=0.25,lw=0.6)
        if cfg=='A0': ax.set_ylim(0,112)
        if j==0: ax.set_ylabel('Net mineralised (% of CO$_2$)')
        if i==len(cfgs)-1: ax.set_xlabel('Time (yr)')
h,l=axes[0][0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.02))
fig.tight_layout(rect=(0,0,1,0.96)); fig.savefig(OUT+'net.png',dpi=200); plt.close(fig)
# free-phase
fig,axes=plt.subplots(1,2,figsize=(9.6,3.9),sharey=True)
for j,reg in enumerate(('MHOW','CRBG')):
    ax=axes[j]
    for a,b in ((0,0.25),(0.5,0.75)): ax.axvspan(a,b,color='#f4c7c3',alpha=0.5,lw=0)
    for m in ('single','continuous','pulsed'):
        r=ld('A1',reg,m,False); t=np.array([x['t_yr'] for x in r['carbon_log']]); f=np.array([x['C_free'] for x in r['carbon_log']])/float(r['co2_scheduled_mol'])*100
        ax.plot(t,f,color=COL[m],lw=2.4,label=LAB[m],ls='--' if m=='single' else '-')
    ax.set_title('Injection-driven flow – '+('Mhow' if reg=='MHOW' else 'GRB'),loc='left',fontsize=13); ax.set_xlim(0,5); ax.set_ylim(0,100); ax.set_xlabel('Time (yr)'); ax.grid(alpha=0.25,lw=0.6)
    if j==0: ax.set_ylabel('Undissolved CO$_2$ (% of scheduled)')
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.03))
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(OUT+'free.png',dpi=200); plt.close(fig)
print('fig14/15 done; A0 included:',have_a0)
