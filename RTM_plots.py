# -*- coding: utf-8 -*-
"""
Publication-quality RTM plots — v3
Aligned with run_rtm.py v29:
  - Paths updated to E:\\Whats Next\\Manuscript submission\\Injection Code\\RTM_Output
  - Excel files now live in per-region/mode subfolders (RTM_{region}_{Mode}\\)
  - Sheet name: "{P:.0f}bar_TimeSeries"  (e.g. "8bar_TimeSeries")
  - Run_Params columns match v29 save_excel() exactly
  - REGION_META updated to match CONFIG values (CRBG phi0=12, T=50; FIBG T=25.3)
  - Guard clauses added: skips missing files gracefully
  - Efficiency plot added as a new figure type
  - No gridlines anywhere
  - Porosity y-axis: tight auto-range around data
  - All curves: dashed lines with small markers
  - No grey used for any mineral or curve
"""

import os, sys, time
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
# Root output directory — matches OUTPUT_ROOT in run_rtm.py
_RTM_ROOT = r"E:\Whats Next\Manuscript submission\Injection Code\RTM_Output"

# All plots saved here
OUT_DIR = os.path.join(_RTM_ROOT, "RTM_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Injection pressure used in all runs — matches CONFIG["injection"]["P_INJ_bar"]
_P_BAR = 8

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':           'Arial',
    'font.size':             14,
    'axes.titlesize':        15,
    'axes.titleweight':      'bold',
    'axes.labelsize':        15,
    'axes.labelweight':      'bold',
    'xtick.labelsize':       13,
    'ytick.labelsize':       13,
    'legend.fontsize':       13,
    'legend.title_fontsize': 13,
    'figure.dpi':            150,
    'savefig.dpi':           300,
    'axes.facecolor':        '#F7FAFA',
    'figure.facecolor':      'white',
    'axes.grid':             False,
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    'axes.spines.left':      True,
    'axes.spines.bottom':    True,
    'axes.linewidth':        1.4,
    'xtick.major.width':     1.4,
    'ytick.major.width':     1.4,
    'xtick.major.size':      6,
    'ytick.major.size':      6,
})

# ── Region metadata — aligned with CONFIG["region_params"] in run_rtm.py ──────
#   CRBG: T_C=50.0, initial_porosity=12.0, q0=27 t/day, M_total=810 t
#   MHOW: T_C=45.0, initial_porosity=10.0, q0=20 t/day, M_total=600 t
#   FIBG: T_C=25.3, initial_porosity=10.0, q0=20 t/day, M_total=600 t
REGION_META = {
    'MHOW': dict(long='Deccan Trap Basalt (Mhow)',       T_C=45.0, q0=20, M=600,  phi0=10.0),
    'CRBG': dict(long='Columbia River Basalt (CRBG)',     T_C=50.0, q0=27, M=810,  phi0=12.0),
    'FIBG': dict(long='Faroe Island Basalt Group (FIBG)', T_C=25.3, q0=20, M=600,  phi0=10.0),
}

SCENARIO_META = {
    'continuous': 'Continuous',
    'pulsed':     'Pulsed',
    'single':     'Single',
}

# ── Color scheme — NO grey anywhere ───────────────────────────────────────────
DISS_COLORS = {
    'Diopside_dissolved':    '#C0392B',
    'Anorthite_dissolved':   '#1565C0',
    'Albite_dissolved':      '#2E7D32',
    'Magnetite_dissolved':   '#6A1E9E',
    'Ilmenite_dissolved':    '#E65C00',
    'BasaltGlass_dissolved': '#00838F',
}
DISS_LABELS = {
    'Diopside_dissolved':    'Diopside',
    'Anorthite_dissolved':   'Anorthite',
    'Albite_dissolved':      'Albite',
    'Magnetite_dissolved':   'Magnetite',
    'Ilmenite_dissolved':    'Ilmenite',
    'BasaltGlass_dissolved': 'Basalt Glass',
}

CARB_COLORS = {
    'Calcite_mol_kgw':   '#E6A817',
    'Siderite_mol_kgw':  '#C62828',
    'Magnesite_mol_kgw': '#1565C0',
    'Dolomite_mol_kgw':  '#6A1E9E',
    'Ankerite_mol_kgw':  '#2E7D32',
}
CARB_LABELS = {
    'Calcite_mol_kgw':   'Calcite',
    'Siderite_mol_kgw':  'Siderite',
    'Magnesite_mol_kgw': 'Magnesite',
    'Dolomite_mol_kgw':  'Dolomite',
    'Ankerite_mol_kgw':  'Ankerite',
}

CLAY_COLORS = {
    'Saponite-Mg_mol_kgw':      '#E65C00',
    'Clinochlore-14A_mol_kgw':  '#1565C0',
    'Kaolinite_mol_kgw':        '#B5A800',
    'Muscovite_mol_kgw':        '#00838F',
}
CLAY_LABELS = {
    'Saponite-Mg_mol_kgw':      'Saponite-Mg',
    'Clinochlore-14A_mol_kgw':  'Clinochlore-14A',
    'Kaolinite_mol_kgw':        'Kaolinite',
    'Muscovite_mol_kgw':        'Muscovite',
}

COMP_COLORS = {'CRBG': '#C62828', 'MHOW': '#1565C0', 'FIBG': '#E65C00'}
PH_COLOR    = '#7B0000'

# ── Marker style ────────────────────────────────────────────────────────────────
DISS_MARKERS = {
    'Diopside_dissolved':    'o',
    'Anorthite_dissolved':   's',
    'Albite_dissolved':      '^',
    'Magnetite_dissolved':   'D',
    'Ilmenite_dissolved':    'v',
    'BasaltGlass_dissolved': 'P',
}
CARB_MARKERS = {
    'Calcite_mol_kgw':   'o',
    'Siderite_mol_kgw':  's',
    'Magnesite_mol_kgw': '^',
    'Dolomite_mol_kgw':  'D',
    'Ankerite_mol_kgw':  'v',
}
CLAY_MARKERS = {
    'Saponite-Mg_mol_kgw':      'o',
    'Clinochlore-14A_mol_kgw':  's',
    'Kaolinite_mol_kgw':        '^',
    'Muscovite_mol_kgw':        'D',
}

MARKER_SIZE  = 6.0
MARKER_EVERY = 14
LINE_STYLE   = '--'

PRESENT_THRESHOLD_DISS = 1e-10
PRESENT_THRESHOLD_PREC = 1e-14

COLLAB_SCENARIO_ORDER = ['single', 'continuous', 'pulsed']
SCENARIOS = ['single', 'continuous', 'pulsed']
REGIONS   = ['MHOW', 'CRBG']


# ── Helper: load data ──────────────────────────────────────────────────────────
def load(region, scenario, p_bar=_P_BAR):
    """
    Load the Excel file produced by run_rtm.py v29 save_excel().

    File layout (matches CONFIG["output_dirs"] in run_rtm.py):
      _RTM_ROOT / RTM_{region}_{scenario.capitalize()} /
          RTM_{region}_{scenario}_{p_bar}bar_v29.xlsx

    Sheets read:
      Run_Params            -- one-row metadata row
      {p_bar}bar_TimeSeries -- time-series DataFrame
    """
    subfolder = os.path.join(_RTM_ROOT, f"RTM_{region}_{scenario.capitalize()}")
    filename  = f"RTM_{region}_{scenario}_{p_bar}bar_v29.xlsx"
    path      = os.path.join(subfolder, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Run run_rtm.py for region={region}, mode={scenario} first."
        )

    df = pd.read_excel(path, sheet_name=f'{p_bar}bar_TimeSeries')
    rp = pd.read_excel(path, sheet_name='Run_Params').iloc[0]
    return df, rp


# ── Helper: injection shading ──────────────────────────────────────────────────
def add_injection_shading(ax, df, rp):
    """
    Shade injection/monitoring windows.  Uses v29 Run_Params column names:
    Mode, T_on_yr, T_off_yr, N_cycles, T_cont_yr, T_inj_days, Injection_on
    """
    mode   = rp['Mode']
    T_on   = float(rp['T_on_yr'])
    T_off  = float(rp['T_off_yr'])
    N      = int(rp['N_cycles'])
    T_cont = float(rp['T_cont_yr'])
    T_total = df['Time_years'].max()

    if mode == 'single':
        inj_rows  = df[df['Injection_on'] == 1]
        t_end_inj = (inj_rows['Time_years'].max() if len(inj_rows) > 0
                     else float(rp['T_inj_days']) / 365.25)
        ax.axvspan(0, t_end_inj, alpha=0.18, color='#FF6B6B', zorder=0, label='Burst')
    elif mode == 'continuous':
        ax.axvspan(0, T_cont, alpha=0.18, color='#FF6B6B', zorder=0, label='Continuous')
    elif mode == 'pulsed':
        t, first = 0.0, True
        for _ in range(N):
            ax.axvspan(t, t + T_on, alpha=0.18, color='#FF6B6B', zorder=0,
                       label='Burst' if first else '_nolegend_')
            first = False
            t += T_on + T_off

    if mode == 'single':
        inj_rows  = df[df['Injection_on'] == 1]
        t_inj_end = (inj_rows['Time_years'].max() if len(inj_rows) > 0
                     else float(rp['T_inj_days']) / 365.25)
    else:
        t_inj_end = T_cont

    if t_inj_end < T_total:
        ax.axvspan(t_inj_end, T_total, alpha=0.07, color='#90CAF9', zorder=0,
                   label='Monitoring')

    if mode == 'pulsed':
        ax.axvline(T_on,      color='#E53935', linewidth=0.9, linestyle=':', zorder=1)
    elif mode == 'single':
        inj_rows = df[df['Injection_on'] == 1]
        t_e = (inj_rows['Time_years'].max() if len(inj_rows) > 0
               else float(rp['T_inj_days']) / 365.25)
        ax.axvline(t_e,       color='#E53935', linewidth=0.9, linestyle=':', zorder=1)
    else:
        ax.axvline(T_cont,    color='#E53935', linewidth=0.9, linestyle=':', zorder=1)


def injection_label(rp):
    mode = rp['Mode']
    if mode == 'single':     return 'Single-burst'
    elif mode == 'continuous': return 'Continuous'
    else:                    return 'Pulsed WAG'


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    for attempt in range(5):
        try:
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f'  Saved: {name}')
            break
        except OSError as e:
            if attempt == 4:
                print(f'  [ERROR] Could not save {name}: {e}')
                raise
            time.sleep(0.5)
    plt.close(fig)


def _make_line_handle(color, marker, label, ls=LINE_STYLE):
    return Line2D([0], [0], color=color, linewidth=2.2, linestyle=ls,
                  marker=marker, markersize=MARKER_SIZE,
                  markerfacecolor=color, markeredgecolor='white',
                  markeredgewidth=0.6, label=label)


# ════════════════════════════════════════════════════════════════════════════════
# FIG 1 – pH Evolution
# ════════════════════════════════════════════════════════════════════════════════
def plot_pH_pCO2(df, rp, ax, title, collab_mode=False, collab_data=None):
    if not collab_mode:
        time = df['Time_years'];  ph = df['pH']
        add_injection_shading(ax, df, rp)
        n = len(time);  every = max(1, n // MARKER_EVERY)
        ax.plot(time, ph, color=PH_COLOR, linewidth=2.5, linestyle=LINE_STYLE,
                marker='o', markevery=every, markersize=MARKER_SIZE,
                markerfacecolor=PH_COLOR, markeredgecolor='white',
                markeredgewidth=0.6, zorder=5)
        ax.axhline(7.5, color='#1565C0', linewidth=1.4, linestyle=':', alpha=0.8)
        ax.set_ylabel('pH', fontsize=15, fontweight='bold')
        ax.set_xlim(0, time.max())
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ph_line   = _make_line_handle(PH_COLOR, 'o',
                        f'pH (min {ph.min():.2f}, final {ph.iloc[-1]:.2f})')
        init_line = Line2D([0],[0], color='#1565C0', linewidth=1.4, linestyle=':',
                           label='Initial pH 7.5')
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.legend(handles=[ph_line, init_line, inj_patch, mon_patch],
                  loc='lower right', framealpha=0.92, fontsize=13)
    else:
        lines = []
        for region, (dfc, rpc) in collab_data.items():
            time = dfc['Time_years'];  ph = dfc['pH']
            c    = COMP_COLORS[region]
            n    = len(time);  every = max(1, n // MARKER_EVERY)
            if region == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            ax.plot(time, ph, color=c, linewidth=2.5, linestyle=LINE_STYLE,
                    marker='o', markevery=every, markersize=MARKER_SIZE,
                    markerfacecolor=c, markeredgecolor='white', markeredgewidth=0.6, zorder=5)
            lines.append(_make_line_handle(c, 'o', region))
        ax.axhline(7.5, color='#1565C0', linewidth=1.4, linestyle=':', alpha=0.8)
        ax.set_ylabel('pH', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        init_line = Line2D([0],[0], color='#1565C0', linewidth=1.4, linestyle=':',
                           label='Initial pH 7.5')
        ax.legend(handles=lines + [init_line, inj_patch, mon_patch],
                  loc='lower right', framealpha=0.92, fontsize=13)


# ════════════════════════════════════════════════════════════════════════════════
# FIG 2 – Mineral Dissolution
# ════════════════════════════════════════════════════════════════════════════════
def plot_dissolution(df, rp, ax, title, collab_mode=False, collab_data=None):
    if not collab_mode:
        time = df['Time_years']
        add_injection_shading(ax, df, rp)
        legend_handles = []
        n = len(time);  every = max(1, n // MARKER_EVERY)
        for col, color in DISS_COLORS.items():
            if col not in df.columns:
                continue
            vals = df[col];  is_present = vals.max() > PRESENT_THRESHOLD_DISS
            mk   = DISS_MARKERS.get(col, 'o')
            if not is_present:
                ax.semilogy(time, np.maximum(vals, 1e-12), color=color,
                            linewidth=1.0, linestyle=':', alpha=0.35, zorder=2)
            else:
                ax.semilogy(time, np.maximum(vals, 1e-12), color=color,
                            linewidth=2.3, linestyle=LINE_STYLE, zorder=4,
                            marker=mk, markevery=every, markersize=MARKER_SIZE,
                            markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.6)
                legend_handles.append(_make_line_handle(color, mk, DISS_LABELS[col]))
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        legend_handles += [inj_patch, mon_patch]
        ax.set_ylabel('Cumulative Dissolved (mol)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.set_xlim(0, time.max())
        ax.legend(handles=legend_handles, loc='lower right', framealpha=0.92,
                  fontsize=12.5, ncol=2, columnspacing=0.8, handletextpad=0.4)
    else:
        handles = []
        for region, (dfc, rpc) in collab_data.items():
            time  = dfc['Time_years'];  c = COMP_COLORS[region]
            n     = len(time);  every = max(1, n // MARKER_EVERY)
            if region == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            cols_p = [c2 for c2 in DISS_COLORS
                      if c2 in dfc.columns and dfc[c2].max() > PRESENT_THRESHOLD_DISS]
            total  = (sum(dfc[c2] for c2 in cols_p)
                      if cols_p else pd.Series(np.zeros(len(time))))
            ax.semilogy(time, np.maximum(total, 1e-12), color=c, linewidth=2.5,
                        linestyle=LINE_STYLE, marker='o', markevery=every,
                        markersize=MARKER_SIZE, markerfacecolor=c,
                        markeredgecolor='white', markeredgewidth=0.6)
            handles.append(_make_line_handle(c, 'o', region))
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.set_ylabel('Total Dissolved (mol)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.legend(handles=handles+[inj_patch, mon_patch], framealpha=0.92, fontsize=13,
                  loc='lower right')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 3 – Carbonate Precipitation
# ════════════════════════════════════════════════════════════════════════════════
def plot_carbonates(df, rp, ax, title, collab_mode=False, collab_data=None):
    if not collab_mode:
        time = df['Time_years']
        add_injection_shading(ax, df, rp)
        legend_handles = []
        n = len(time);  every = max(1, n // MARKER_EVERY)
        for col, color in CARB_COLORS.items():
            if col not in df.columns:
                continue
            vals = df[col];  is_present = vals.max() > PRESENT_THRESHOLD_PREC
            mk   = CARB_MARKERS.get(col, 'o')
            if not is_present:
                ax.semilogy(time, np.maximum(vals, 1e-13), color=color,
                            linewidth=1.0, linestyle=':', alpha=0.35, zorder=2)
            else:
                ax.semilogy(time, np.maximum(vals, 1e-13), color=color,
                            linewidth=2.3, linestyle=LINE_STYLE, zorder=4,
                            marker=mk, markevery=every, markersize=MARKER_SIZE,
                            markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.6)
                legend_handles.append(_make_line_handle(color, mk, CARB_LABELS[col]))
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        legend_handles += [inj_patch, mon_patch]
        ax.set_ylabel('Carbonate Inventory (mol/kgw)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.set_xlim(0, time.max())
        ax.legend(handles=legend_handles, loc='lower right', framealpha=0.92,
                  fontsize=12.5, ncol=2, columnspacing=0.8, handletextpad=0.4)
    else:
        handles = []
        for region, (dfc, rpc) in collab_data.items():
            time  = dfc['Time_years'];  c = COMP_COLORS[region]
            n     = len(time);  every = max(1, n // MARKER_EVERY)
            if region == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            cols_p = [c2 for c2 in CARB_COLORS
                      if c2 in dfc.columns and dfc[c2].max() > PRESENT_THRESHOLD_PREC]
            total  = (sum(dfc[c2] for c2 in cols_p)
                      if cols_p else pd.Series(np.zeros(len(time))))
            ax.semilogy(time, np.maximum(total, 1e-14), color=c, linewidth=2.5,
                        linestyle=LINE_STYLE, marker='o', markevery=every,
                        markersize=MARKER_SIZE, markerfacecolor=c,
                        markeredgecolor='white', markeredgewidth=0.6)
            handles.append(_make_line_handle(c, 'o', region))
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.set_ylabel('Carbonate Inventory (mol/kgw)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.legend(handles=handles+[inj_patch, mon_patch], framealpha=0.92, fontsize=13,
                  loc='lower right')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 4 – Clay / Silicate Precipitation
# ════════════════════════════════════════════════════════════════════════════════
def plot_clays(df, rp, ax, title, collab_mode=False, collab_data=None):
    if not collab_mode:
        time = df['Time_years']
        add_injection_shading(ax, df, rp)
        legend_handles = []
        n = len(time);  every = max(1, n // MARKER_EVERY)
        for col, color in CLAY_COLORS.items():
            if col not in df.columns:
                continue
            vals = df[col];  is_present = vals.max() > PRESENT_THRESHOLD_PREC
            mk   = CLAY_MARKERS.get(col, 'o')
            if not is_present:
                ax.semilogy(time, np.maximum(vals, 1e-13), color=color,
                            linewidth=1.0, linestyle=':', alpha=0.35, zorder=2)
            else:
                ax.semilogy(time, np.maximum(vals, 1e-13), color=color,
                            linewidth=2.3, linestyle=LINE_STYLE, zorder=4,
                            marker=mk, markevery=every, markersize=MARKER_SIZE,
                            markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.6)
                legend_handles.append(_make_line_handle(color, mk, CLAY_LABELS[col]))
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        legend_handles += [inj_patch, mon_patch]
        ax.set_ylabel('EQ-phase Inventory (mol/kgw)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.set_xlim(0, time.max())
        ax.legend(handles=legend_handles, loc='lower right', framealpha=0.92,
                  fontsize=12.5, ncol=2, columnspacing=0.8, handletextpad=0.4)
    else:
        handles = []
        for region, (dfc, rpc) in collab_data.items():
            time  = dfc['Time_years'];  c = COMP_COLORS[region]
            n     = len(time);  every = max(1, n // MARKER_EVERY)
            if region == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            cols_p = [c2 for c2 in CLAY_COLORS
                      if c2 in dfc.columns and dfc[c2].max() > PRESENT_THRESHOLD_PREC]
            total  = (sum(dfc[c2] for c2 in cols_p)
                      if cols_p else pd.Series(np.zeros(len(time))))
            ax.semilogy(time, np.maximum(total, 1e-14), color=c, linewidth=2.5,
                        linestyle=LINE_STYLE, marker='o', markevery=every,
                        markersize=MARKER_SIZE, markerfacecolor=c,
                        markeredgecolor='white', markeredgewidth=0.6)
            handles.append(_make_line_handle(c, 'o', region))
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.set_ylabel('EQ-phase Inventory (mol/kgw)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.legend(handles=handles+[inj_patch, mon_patch], framealpha=0.92, fontsize=13,
                  loc='lower right')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 5 – Porosity  (tight y-axis around data)
# ════════════════════════════════════════════════════════════════════════════════
def plot_porosity(df, rp, ax, title, collab_mode=False, collab_data=None):
    region_key = rp['Region']
    meta  = REGION_META.get(region_key, {})
    phi0  = meta.get('phi0', 10.0)

    if not collab_mode:
        time = df['Time_years'];  phi = df['Porosity_%']
        add_injection_shading(ax, df, rp)
        n = len(time);  every = max(1, n // MARKER_EVERY)
        ax.plot(time, phi, color=PH_COLOR, linewidth=2.5, linestyle=LINE_STYLE,
                marker='o', markevery=every, markersize=MARKER_SIZE,
                markerfacecolor=PH_COLOR, markeredgecolor='white',
                markeredgewidth=0.6, zorder=5)
        ax.axhline(phi0, color='#1565C0', linewidth=1.4, linestyle=':', alpha=0.8)
        ymin_d = phi.min();  ymax_d = phi.max()
        yr     = ymax_d - ymin_d if ymax_d != ymin_d else 0.5
        ax.set_ylim(ymin_d - 0.12 * yr, ymax_d + 0.18 * yr)
        ax.set_ylabel('Porosity (%)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.set_xlim(0, time.max())
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.legend(loc='lower right', framealpha=0.92, fontsize=12.5, handles=[
            _make_line_handle(PH_COLOR, 'o', f'Porosity ({phi0:.1f}%\u2192{phi.max():.2f}%)'),
            Line2D([0],[0], color='#1565C0', lw=1.4, ls=':', label=f'Initial \u03c6={phi0:.1f}%'),
            inj_patch, mon_patch
        ])
    else:
        handles = [];  all_phi = []
        for rk, (dfc, _) in collab_data.items():
            all_phi.extend(dfc['Porosity_%'].tolist())
        for rk, (dfc, rpc) in collab_data.items():
            time  = dfc['Time_years'];  phi = dfc['Porosity_%']
            c     = COMP_COLORS[rk]
            n     = len(time);  every = max(1, n // MARKER_EVERY)
            if rk == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            ax.plot(time, phi, color=c, linewidth=2.5, linestyle=LINE_STYLE,
                    marker='o', markevery=every, markersize=MARKER_SIZE,
                    markerfacecolor=c, markeredgecolor='white', markeredgewidth=0.6)
            handles.append(_make_line_handle(c, 'o', rk))
        arr = np.array(all_phi)
        yr  = arr.max() - arr.min() if arr.max() != arr.min() else 0.5
        ax.set_ylim(arr.min() - 0.12 * yr, arr.max() + 0.18 * yr)
        ax.axhline(4.0, color='#E65C00', linewidth=1.2, linestyle=':', alpha=0.7)
        ax.set_ylabel('Porosity (%)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.legend(handles=handles+[inj_patch, mon_patch], framealpha=0.92, fontsize=13,
                  loc='lower right')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 6 – CO2 Sequestration Efficiency  (NEW in v3)
# Column written by v29 save_excel(): Efficiency_%
# ════════════════════════════════════════════════════════════════════════════════
def plot_efficiency(df, rp, ax, title, collab_mode=False, collab_data=None):
    EFF_COLOR = '#2E7D32'

    if not collab_mode:
        time = df['Time_years'];  eff = df['Efficiency_%']
        add_injection_shading(ax, df, rp)
        n = len(time);  every = max(1, n // MARKER_EVERY)
        ax.plot(time, eff, color=EFF_COLOR, linewidth=2.5, linestyle=LINE_STYLE,
                marker='s', markevery=every, markersize=MARKER_SIZE,
                markerfacecolor=EFF_COLOR, markeredgecolor='white',
                markeredgewidth=0.6, zorder=5)
        final_eff = eff.iloc[-1]
        ax.set_ylabel('Sequestration Efficiency (%)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        ax.set_xlim(0, time.max())
        ax.set_ylim(0, max(eff.max() * 1.15, 5))
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        eff_line  = _make_line_handle(EFF_COLOR, 's', f'Efficiency (final {final_eff:.1f}%)')
        ax.legend(handles=[eff_line, inj_patch, mon_patch],
                  loc='lower right', framealpha=0.92, fontsize=13)
    else:
        handles = []
        for rk, (dfc, rpc) in collab_data.items():
            time  = dfc['Time_years'];  eff = dfc['Efficiency_%']
            c     = COMP_COLORS[rk]
            n     = len(time);  every = max(1, n // MARKER_EVERY)
            if rk == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            ax.plot(time, eff, color=c, linewidth=2.5, linestyle=LINE_STYLE,
                    marker='s', markevery=every, markersize=MARKER_SIZE,
                    markerfacecolor=c, markeredgecolor='white', markeredgewidth=0.6)
            handles.append(_make_line_handle(c, 's', rk))
        ax.set_ylabel('Sequestration Efficiency (%)', fontsize=15, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=15, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=13)
        if title:
            ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.legend(handles=handles+[inj_patch, mon_patch], framealpha=0.92, fontsize=13,
                  loc='lower right')


# ── Plot registry ──────────────────────────────────────────────────────────────
PLOT_FNS = {
    'pH':          plot_pH_pCO2,
    'dissolution': plot_dissolution,
    'carbonate':   plot_carbonates,
    'clay':        plot_clays,
    'porosity':    plot_porosity,
    'efficiency':  plot_efficiency,
}

FIG_TITLES = {
    'pH':          'pH Evolution',
    'dissolution': 'Primary Mineral Dissolution',
    'carbonate':   'Carbonate Precipitation',
    'clay':        'Clay / Silicate Precipitation',
    'porosity':    'Porosity',
    'efficiency':  r'$\mathrm{CO_2}$ Sequestration Efficiency',
}


PANEL_NAMES = {
    'single':     'Single-burst',
    'continuous': 'Continuous',
    'pulsed':     'Pulsed WAG',
}


# ════════════════════════════════════════════════════════════════════════════════
# GENERATE COLLAB PLOTS — Single | Continuous | Pulsed  (per region)
# ════════════════════════════════════════════════════════════════════════════════
print("=== Generating Collab Plots (Single | Continuous | Pulsed) ===")

for fig_type, plot_fn in PLOT_FNS.items():
    for region in REGIONS:
        # Skip entire region if no files exist at all
        any_found = any(
            os.path.exists(os.path.join(
                _RTM_ROOT, f"RTM_{region}_{s.capitalize()}",
                f"RTM_{region}_{s}_{_P_BAR}bar_v29.xlsx"))
            for s in COLLAB_SCENARIO_ORDER
        )
        if not any_found:
            print(f"  [SKIP] No {region} output files found — skipping {fig_type}")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
        fig.patch.set_facecolor('white')

        for col_idx, scenario in enumerate(COLLAB_SCENARIO_ORDER):
            ax = axes[col_idx]
            try:
                df, rp = load(region, scenario)
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'File not found', transform=ax.transAxes,
                        ha='center', va='center', color='red', fontsize=12)
                ax.set_title(f'({chr(97 + col_idx)}) {PANEL_NAMES[scenario]} (missing)',
                             fontsize=15, fontweight='bold', pad=10)
                continue

            plot_fn(df, rp, ax, '', collab_mode=False)
            if col_idx > 0:
                ax.set_ylabel('')

            # Top-left panel badge (a, b, c)
            panel_letter = chr(97 + col_idx)
            ax.text(0.018, 0.975, panel_letter, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', color='white',
                    ha='left', va='top', zorder=25,
                    bbox=dict(boxstyle='square,pad=0.22', facecolor='black', edgecolor='black', alpha=1.0))

        fig.tight_layout(pad=1.5)
        save_fig(fig, f'collab_{fig_type}_{region}.png')


# ════════════════════════════════════════════════════════════════════════════════
# GENERATE COMPARATIVE PLOTS — all regions overlaid per scenario
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== Generating Comparative Plots (CRBG red vs MHOW blue) ===")

for fig_type, plot_fn in PLOT_FNS.items():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    fig.patch.set_facecolor('white')

    for col_idx, scenario in enumerate(SCENARIOS):
        ax = axes[col_idx]
        collab_data = {}
        rp0 = None
        # Comparative plots: CRBG and MHOW only (FIBG has its own collab panel)
        for region in ['CRBG', 'MHOW']:
            try:
                df, rp = load(region, scenario)
                collab_data[region] = (df, rp)
                if rp0 is None:
                    rp0 = rp
            except FileNotFoundError:
                print(f"  [SKIP comparative] {region}/{scenario} not found")

        if not collab_data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='red', fontsize=12)
            continue

        first_region = list(collab_data.keys())[0]
        plot_fn(collab_data[first_region][0], rp0, ax, '',
                collab_mode=True, collab_data=collab_data)
        if col_idx > 0:
            ax.set_ylabel('')

        # Top-left panel badge (a, b, c)
        panel_letter = chr(97 + col_idx)
        ax.text(0.018, 0.975, panel_letter, transform=ax.transAxes,
                fontsize=14, fontweight='bold', color='white',
                ha='left', va='top', zorder=25,
                bbox=dict(boxstyle='square,pad=0.22', facecolor='black', edgecolor='black', alpha=1.0))

    fig.tight_layout(pad=1.5)
    save_fig(fig, f'comparative_{fig_type}_all_scenarios.png')


print(f'\nAll plots saved to {OUT_DIR}')