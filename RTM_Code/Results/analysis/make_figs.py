import sys, pickle, numpy as np, os
from pathlib import Path as _P
_RES = _P(__file__).resolve().parents[1]
_ROOT = _RES.parent
sys.path.insert(0, str(_ROOT))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
CACHE = str(_RES / 'cache') + '/'
OUT = str(_RES / 'figures') + '/'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelsize': 13, 'xtick.labelsize': 12,
                     'ytick.labelsize': 12, 'legend.fontsize': 10.5, 'axes.spines.top': False, 'axes.spines.right': False})
MODES = [('single', 'a  Single-burst'), ('continuous', 'b  Continuous'), ('pulsed', 'c  Pulsed WAG')]
WIN = {'single': [(0, 0.0821)], 'continuous': [(0, 1.0)], 'pulsed': [(0, 0.25), (0.5, 0.75)]}
SITE = {'MHOW': 'Mhow', 'CRBG': 'GRB'}
def load(cfg, reg, mode, ctrl=False):
    return pickle.load(open(CACHE + f"{cfg}_{reg}_{mode}_{'ctrl' if ctrl else 'inj'}.pkl", 'rb'))
def panel_fig(reg, ylabel, drawer, fname, logy=False, cfg='A1', ylim=None, legend_in=0):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
    for k, (ax, (mode, title)) in enumerate(zip(axes, MODES)):
        r = load(cfg, reg, mode); r['_ctrl'] = load(cfg, reg, mode, True)
        for a, b in WIN[mode]:
            ax.axvspan(a, b, color='#f4c7c3', alpha=0.55, lw=0)
        drawer(ax, r)
        ax.set_title(title, loc='left', fontsize=13, fontweight='bold'); ax.set_xlim(0, 5); ax.set_xlabel('Time (yr)')
        if logy: ax.set_yscale('log')
        if ylim: ax.set_ylim(*ylim)
        if k == 0: ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.6)
    h, l = axes[legend_in].get_legend_handles_labels()
    if h: axes[legend_in].legend(handles=h, labels=l, loc='best', frameon=False, ncol=1)
    fig.tight_layout(); fig.savefig(OUT + fname, dpi=200); plt.close(fig)
def d_ph(ax, r):
    t = r['time']; ax.plot(t, r['pH'], color='#1f4e9c', lw=2, label='Inlet cell, injection run')
    c = r['_ctrl']; ax.plot(c['time'], c['pH'], color='#7a7a7a', lw=1.6, ls='--', label='Inlet cell, no-injection control')
DCOL = {'Anorthite': '#1f77b4', 'Diopside': '#d62728', 'Albite': '#9467bd', 'Magnetite': '#8c564b', 'Ilmenite': '#ff7f0e', 'BasaltGlass': '#2ca02c'}
def d_diss(ax, r):
    t = r['time']
    for m, c in DCOL.items():
        y = np.maximum(np.asarray(r['dissolved'][m], float), 1e-2)
        ax.plot(t, y, color=c, lw=1.8, label={'BasaltGlass': 'Basalt glass'}.get(m, m))
def d_carb(ax, r):
    t = r['time']
    for m, c in (('Calcite', '#e08a00'), ('Siderite', '#7b3f00'), ('Dolomite', '#4b8b3b'), ('Magnesite', '#888888')):
        y = np.asarray(r['minerals'][m], float)
        if y.max() > 1e-8: ax.plot(t, np.maximum(y, 1e-9), color=c, lw=2, label=m)
def d_clay(ax, r):
    t = r['time']
    for m, c, ls in (('Kaolinite', '#1f77b4', '-'), ('Saponite-Mg', '#d62728', '-'), ('Clinochlore-14A', '#2ca02c', '-'), ('Muscovite', '#9467bd', '-'), ('SiO2(am)', '#555555', '--')):
        y = np.asarray(r['minerals'][m], float)
        if y.max() > 1e-8: ax.plot(t, np.maximum(y, 1e-9), color=c, lw=2, ls=ls, label='Amorphous silica' if m == 'SiO2(am)' else m)
def d_por(ax, r):
    t = r['time']; ax.plot(t, r['porosity'], color='#111', lw=2); ax.axhline(r['porosity'][0], color='#999', lw=0.8, ls=':')
for reg in ('MHOW', 'CRBG'):
    tag = reg
    try:
        panel_fig(reg, 'pH', d_ph, f'ph_{tag}.png', ylim=(3.8, 11.5))
        panel_fig(reg, 'Cumulative dissolution (mol)', d_diss, f'diss_{tag}.png', logy=True, ylim=(1e-2, 5e7), legend_in=2)
        panel_fig(reg, 'Carbonate inventory (mol kgw$^{-1}$)', d_carb, f'carb_{tag}.png', logy=True, ylim=(1e-6, 1))
        panel_fig(reg, 'Clay and silica inventory (mol kgw$^{-1}$)', d_clay, f'clay_{tag}.png', logy=True, ylim=(1e-6, 1))
        panel_fig(reg, 'Bulk porosity (%)', d_por, f'por_{tag}.png')
        print('ok', reg)
    except FileNotFoundError as e:
        print('missing', e)
