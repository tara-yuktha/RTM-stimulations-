# -*- coding: utf-8 -*-
"""
Publication-quality RTM plots — v2
Changes from v1:
  - No gridlines anywhere
  - Porosity y-axis: tight auto-range around data
  - All curves: dashed lines with small markers
  - No grey used for any mineral or curve
  - More vivid, appealing color palette
"""

import os
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
DATA_DIR = r"C:\Users\TOREL SAMYUKTHA S\RTM_outputs_claude_29\RTM_Uploads"
OUT_DIR  = r"C:\Users\TOREL SAMYUKTHA S\RTM_outputs_claude_29\RTM_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':           'Arial',
    'font.size':             12,
    'axes.titlesize':        12,
    'axes.labelsize':        14,
    'xtick.labelsize':       10,
    'ytick.labelsize':       10,
    'legend.fontsize':       14,
    'legend.title_fontsize': 14,
    'figure.dpi':            150,
    'savefig.dpi':           300,
    'axes.facecolor':        '#F7FAFA',
    'figure.facecolor':      'white',
    'axes.grid':             False,       # ← NO gridlines
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    'axes.spines.left':      True,
    'axes.spines.bottom':    True,
})

# ── Region metadata ────────────────────────────────────────────────────────────
REGION_META = {
    'MHOW': dict(long='Deccan Trap Basalt (Mhow)', T_C=45, q0=20, M=600, phi0=10.0),
    'CRBG': dict(long='Columbia River Basalt (CRBG)', T_C=50, q0=27, M=810, phi0=12.0),
}

SCENARIO_META = {
    'continuous': 'Continuous',
    'pulsed':     'Pulsed',
    'single':     'Single',
}

# ── Color scheme — NO grey anywhere ───────────────────────────────────────────
DISS_COLORS = {
    'Diopside_dissolved':    '#C0392B',   # crimson red
    'Anorthite_dissolved':   '#1565C0',   # deep blue
    'Albite_dissolved':      '#2E7D32',   # forest green
    'Magnetite_dissolved':   '#6A1E9E',   # violet
    'Ilmenite_dissolved':    '#E65C00',   # burnt orange
    'BasaltGlass_dissolved': '#00838F',   # teal
}
DISS_LABELS = {
    'Diopside_dissolved':    'Diopside',
    'Anorthite_dissolved':   'Anorthite',
    'Albite_dissolved':      'Albite',
    'Magnetite_dissolved':   'Magnetite',
    'Ilmenite_dissolved':    'Ilmenite',
    'BasaltGlass_dissolved': 'BasaltGlass ★',
}

CARB_COLORS = {
    'Calcite_mol_kgw':   '#E6A817',   # amber
    'Siderite_mol_kgw':  '#C62828',   # deep red
    'Magnesite_mol_kgw': '#1565C0',   # deep blue  (was grey)
    'Dolomite_mol_kgw':  '#6A1E9E',   # violet
    'Ankerite_mol_kgw':  '#2E7D32',   # forest green
}
CARB_LABELS = {
    'Calcite_mol_kgw':   'Calcite',
    'Siderite_mol_kgw':  'Siderite',
    'Magnesite_mol_kgw': 'Magnesite',
    'Dolomite_mol_kgw':  'Dolomite',
    'Ankerite_mol_kgw':  'Ankerite',
}

CLAY_COLORS = {
    'Saponite-Mg_mol_kgw':      '#E65C00',   # burnt orange  (was grey)
    'Clinochlore-14A_mol_kgw':  '#1565C0',   # deep blue     (was dark grey)
    'Kaolinite_mol_kgw':        '#B5A800',   # gold
    'Muscovite_mol_kgw':        '#00838F',   # teal
}
CLAY_LABELS = {
    'Saponite-Mg_mol_kgw':      'Saponite-Mg',
    'Clinochlore-14A_mol_kgw':  'Clinochlore-14A',
    'Kaolinite_mol_kgw':        'Kaolinite',
    'Muscovite_mol_kgw':        'Muscovite',
}

COMP_COLORS = {'CRBG': '#C62828', 'MHOW': '#1565C0'}
PH_COLOR    = '#7B0000'   # very dark red

# ── Marker style — dashed lines + small markers ────────────────────────────────
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

MARKER_SIZE  = 4      # small markers
MARKER_EVERY = 14     # marker every N data points
LINE_STYLE   = '--'   # dashed for ALL active curves

# Threshold for "present" minerals
PRESENT_THRESHOLD_DISS = 1e-10
PRESENT_THRESHOLD_PREC = 1e-14

# ── Collab / scenario order ────────────────────────────────────────────────────
COLLAB_SCENARIO_ORDER = ['single', 'continuous', 'pulsed']
SCENARIOS = ['continuous', 'pulsed', 'single']
REGIONS   = ['MHOW', 'CRBG']


# ── Helper: load data ──────────────────────────────────────────────────────────
def load(region, scenario):
    path = os.path.join(DATA_DIR, f'RTM_{region}_{scenario}_8bar_v29.xlsx')
    df   = pd.read_excel(path, sheet_name='8bar_TimeSeries')
    rp   = pd.read_excel(path, sheet_name='Run_Params').iloc[0]
    return df, rp


# ── Helper: injection shading ──────────────────────────────────────────────────
def add_injection_shading(ax, df, rp):
    mode    = rp['Mode']
    T_on    = rp['T_on_yr']
    T_off   = rp['T_off_yr']
    N       = int(rp['N_cycles'])
    T_cont  = rp['T_cont_yr']
    T_total = df['Time_years'].max()

    if mode == 'single':
        t_end_inj = df[df['Injection_on'] == 1]['Time_years'].max()
        ax.axvspan(0, t_end_inj, alpha=0.18, color='#FF6B6B', zorder=0, label='Burst')
    elif mode == 'continuous':
        ax.axvspan(0, T_cont, alpha=0.18, color='#FF6B6B', zorder=0, label='Burst')
    elif mode == 'pulsed':
        t = 0.0
        first = True
        for _ in range(N):
            ax.axvspan(t, t + T_on, alpha=0.18, color='#FF6B6B', zorder=0,
                       label='Burst' if first else '_nolegend_')
            first = False
            t += T_on + T_off

    t_inj_end = T_cont if mode in ('continuous', 'pulsed') else \
        df[df['Injection_on'] == 1]['Time_years'].max()
    if t_inj_end < T_total:
        ax.axvspan(t_inj_end, T_total, alpha=0.07, color='#90CAF9', zorder=0, label='Monitoring')

    if mode == 'pulsed':
        ax.axvline(T_on,   color='#E53935', linewidth=0.9, linestyle=':', zorder=1)
    elif mode == 'single':
        t_e = df[df['Injection_on'] == 1]['Time_years'].max()
        ax.axvline(t_e,    color='#E53935', linewidth=0.9, linestyle=':', zorder=1)
    else:
        ax.axvline(T_cont, color='#E53935', linewidth=0.9, linestyle=':', zorder=1)


def injection_label(rp):
    mode = rp['Mode']
    if mode == 'single':
        return 'Burst'
    elif mode == 'continuous':
        return 'Continuous'
    else:
        return 'Pulsed'


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {name}')


def _make_line_handle(color, marker, label, ls=LINE_STYLE):
    return Line2D([0], [0], color=color, linewidth=1.8, linestyle=ls,
                  marker=marker, markersize=MARKER_SIZE,
                  markerfacecolor=color, markeredgecolor='white',
                  markeredgewidth=0.4, label=label)


# ════════════════════════════════════════════════════════════════════════════════
# FIG 1 – pH Evolution
# ════════════════════════════════════════════════════════════════════════════════
def plot_pH_pCO2(df, rp, ax, title, collab_mode=False, collab_data=None):
    if not collab_mode:
        time = df['Time_years']
        ph   = df['pH']
        add_injection_shading(ax, df, rp)

        n     = len(time)
        every = max(1, n // MARKER_EVERY)
        ax.plot(time, ph, color=PH_COLOR, linewidth=2.0, linestyle=LINE_STYLE,
                marker='o', markevery=every, markersize=MARKER_SIZE,
                markerfacecolor=PH_COLOR, markeredgecolor='white',
                markeredgewidth=0.4, zorder=5)
        ax.axhline(7.5, color='#1565C0', linewidth=1.0, linestyle=':', alpha=0.8)

        ax.set_ylabel('pH', fontsize=10)
        ax.set_xlim(0, time.max())
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)

        ph_line   = _make_line_handle(PH_COLOR, 'o',
                        f'pH | nadir={ph.min():.2f} final={ph.iloc[-1]:.2f}')
        init_line = Line2D([0],[0], color='#1565C0', linewidth=1, linestyle=':',
                           label='Initial pH 7.5')
        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.legend(handles=[ph_line, init_line, inj_patch, mon_patch],
                  loc='lower right', framealpha=0.88, fontsize=8, ncol=2, columnspacing=1)

    else:
        lines = []
        for region, (dfc, rpc) in collab_data.items():
            time = dfc['Time_years']
            ph   = dfc['pH']
            c    = COMP_COLORS[region]
            n    = len(time)
            every = max(1, n // MARKER_EVERY)
            if region == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            ax.plot(time, ph, color=c, linewidth=2.0, linestyle=LINE_STYLE,
                    marker='o', markevery=every, markersize=MARKER_SIZE,
                    markerfacecolor=c, markeredgecolor='white', markeredgewidth=0.4, zorder=5)
            lines.append(_make_line_handle(c, 'o', region))

        ax.axhline(7.5, color='#1565C0', linewidth=1.0, linestyle=':', alpha=0.8)
        ax.set_ylabel('pH', fontsize=10)
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)

        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        init_line = Line2D([0],[0], color='#1565C0', linewidth=1, linestyle=':',
                           label='Initial pH 7.5')
        ax.legend(handles=lines + [init_line, inj_patch, mon_patch],
                  loc='lower right', framealpha=0.88, fontsize=8, ncol=2)


# ════════════════════════════════════════════════════════════════════════════════
# FIG 2 – Mineral Dissolution
# ════════════════════════════════════════════════════════════════════════════════
def plot_dissolution(df, rp, ax, title, collab_mode=False, collab_data=None):
    if not collab_mode:
        time = df['Time_years']
        add_injection_shading(ax, df, rp)

        legend_handles = []
        n     = len(time)
        every = max(1, n // MARKER_EVERY)

        for col, color in DISS_COLORS.items():
            vals = df[col]
            is_present = vals.max() > PRESENT_THRESHOLD_DISS
            mk  = DISS_MARKERS.get(col, 'o')

            if not is_present:
                ax.semilogy(time, np.maximum(vals, 1e-12), color=color,
                            linewidth=1.0, linestyle=':', alpha=0.35, zorder=2)
            else:
                ax.semilogy(time, np.maximum(vals, 1e-12), color=color,
                            linewidth=1.8, linestyle=LINE_STYLE, zorder=4,
                            marker=mk, markevery=every,
                            markersize=MARKER_SIZE, markerfacecolor=color,
                            markeredgecolor='white', markeredgewidth=0.4)
                legend_handles.append(_make_line_handle(color, mk, DISS_LABELS[col]))

        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        legend_handles += [inj_patch, mon_patch]

        ax.set_ylabel('Cumulative Dissolved (mol)', fontsize=10)
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlim(0, time.max())
        ax.legend(handles=legend_handles, loc='lower right', framealpha=0.88,
                  fontsize=8, ncol=1)

    else:
        handles = []
        for region, (dfc, rpc) in collab_data.items():
            time  = dfc['Time_years']
            c     = COMP_COLORS[region]
            n     = len(time)
            every = max(1, n // MARKER_EVERY)
            if region == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            total = sum(dfc[col] for col in DISS_COLORS
                        if dfc[col].max() > PRESENT_THRESHOLD_DISS)
            ax.semilogy(time, np.maximum(total, 1e-12), color=c, linewidth=2.0,
                        linestyle=LINE_STYLE, marker='o', markevery=every,
                        markersize=MARKER_SIZE, markerfacecolor=c,
                        markeredgecolor='white', markeredgewidth=0.4)
            handles.append(_make_line_handle(c, 'o', region))

        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.set_ylabel('Total Dissolved (mol)', fontsize=10)
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.legend(handles=handles+[inj_patch, mon_patch], framealpha=0.88, fontsize=8,
                  loc='lower right')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 3 – Carbonate Precipitation
# ════════════════════════════════════════════════════════════════════════════════
def plot_carbonates(df, rp, ax, title, collab_mode=False, collab_data=None):
    if not collab_mode:
        time = df['Time_years']
        add_injection_shading(ax, df, rp)

        legend_handles = []
        n     = len(time)
        every = max(1, n // MARKER_EVERY)

        for col, color in CARB_COLORS.items():
            vals = df[col]
            is_present = vals.max() > PRESENT_THRESHOLD_PREC
            mk = CARB_MARKERS.get(col, 'o')

            if not is_present:
                ax.semilogy(time, np.maximum(vals, 1e-13), color=color,
                            linewidth=1.0, linestyle=':', alpha=0.35, zorder=2)
            else:
                ax.semilogy(time, np.maximum(vals, 1e-13), color=color,
                            linewidth=1.8, linestyle=LINE_STYLE, zorder=4,
                            marker=mk, markevery=every,
                            markersize=MARKER_SIZE, markerfacecolor=color,
                            markeredgecolor='white', markeredgewidth=0.4)
                legend_handles.append(_make_line_handle(color, mk, CARB_LABELS[col]))

        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        legend_handles += [inj_patch, mon_patch]

        ax.set_ylabel('Carbonate Inventory (mol/kgw)', fontsize=10)
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlim(0, time.max())
        ax.legend(handles=legend_handles, loc='lower right', framealpha=0.88, fontsize=8)

    else:
        handles = []
        for region, (dfc, rpc) in collab_data.items():
            time  = dfc['Time_years']
            c     = COMP_COLORS[region]
            n     = len(time)
            every = max(1, n // MARKER_EVERY)
            if region == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            total = sum(dfc[col] for col in CARB_COLORS
                        if dfc[col].max() > PRESENT_THRESHOLD_PREC)
            ax.semilogy(time, np.maximum(total, 1e-14), color=c, linewidth=2.0,
                        linestyle=LINE_STYLE, marker='o', markevery=every,
                        markersize=MARKER_SIZE, markerfacecolor=c,
                        markeredgecolor='white', markeredgewidth=0.4)
            handles.append(_make_line_handle(c, 'o', region))

        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.set_ylabel('Carbonate Inventory (mol/kgw)', fontsize=10)
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.legend(handles=handles+[inj_patch, mon_patch], framealpha=0.88, fontsize=8,
                  loc='lower right')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 4 – Clay / Silicate Precipitation
# ════════════════════════════════════════════════════════════════════════════════
def plot_clays(df, rp, ax, title, collab_mode=False, collab_data=None):
    if not collab_mode:
        time = df['Time_years']
        add_injection_shading(ax, df, rp)

        legend_handles = []
        n     = len(time)
        every = max(1, n // MARKER_EVERY)

        for col, color in CLAY_COLORS.items():
            vals = df[col]
            is_present = vals.max() > PRESENT_THRESHOLD_PREC
            mk = CLAY_MARKERS.get(col, 'o')

            if not is_present:
                ax.semilogy(time, np.maximum(vals, 1e-13), color=color,
                            linewidth=1.0, linestyle=':', alpha=0.35, zorder=2)
            else:
                ax.semilogy(time, np.maximum(vals, 1e-13), color=color,
                            linewidth=1.8, linestyle=LINE_STYLE, zorder=4,
                            marker=mk, markevery=every,
                            markersize=MARKER_SIZE, markerfacecolor=color,
                            markeredgecolor='white', markeredgewidth=0.4)
                legend_handles.append(_make_line_handle(color, mk, CLAY_LABELS[col]))

        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        legend_handles += [inj_patch, mon_patch]

        ax.set_ylabel('EQ-phase inventory (mol/kgw)', fontsize=10)
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlim(0, time.max())
        ax.legend(handles=legend_handles, loc='lower right', framealpha=0.88, fontsize=8)

    else:
        handles = []
        for region, (dfc, rpc) in collab_data.items():
            time  = dfc['Time_years']
            c     = COMP_COLORS[region]
            n     = len(time)
            every = max(1, n // MARKER_EVERY)
            if region == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            total = sum(dfc[col] for col in CLAY_COLORS
                        if dfc[col].max() > PRESENT_THRESHOLD_PREC)
            ax.semilogy(time, np.maximum(total, 1e-14), color=c, linewidth=2.0,
                        linestyle=LINE_STYLE, marker='o', markevery=every,
                        markersize=MARKER_SIZE, markerfacecolor=c,
                        markeredgecolor='white', markeredgewidth=0.4)
            handles.append(_make_line_handle(c, 'o', region))

        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.set_ylabel('EQ-phase inventory (mol/kgw)', fontsize=10)
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.legend(handles=handles+[inj_patch, mon_patch], framealpha=0.88, fontsize=8,
                  loc='lower right')


# ════════════════════════════════════════════════════════════════════════════════
# FIG 5 – Porosity  (tight y-axis around data)
# ════════════════════════════════════════════════════════════════════════════════
def plot_porosity(df, rp, ax, title, collab_mode=False, collab_data=None):
    meta = REGION_META[rp['Region']]

    if not collab_mode:
        time = df['Time_years']
        phi  = df['Porosity_%']
        phi0 = meta['phi0']
        add_injection_shading(ax, df, rp)

        n     = len(time)
        every = max(1, n // MARKER_EVERY)

        ax.plot(time, phi, color=PH_COLOR, linewidth=2.0, linestyle=LINE_STYLE,
                marker='o', markevery=every, markersize=MARKER_SIZE,
                markerfacecolor=PH_COLOR, markeredgecolor='white',
                markeredgewidth=0.4, zorder=5)
        ax.axhline(phi0, color='#1565C0', linewidth=1.0, linestyle=':', alpha=0.8)

        # ── Tight y-axis: pad 10% above/below data range ──────────────────────
        ymin_data = phi.min()
        ymax_data = phi.max()
        yrange    = ymax_data - ymin_data if ymax_data != ymin_data else 0.5
        ax.set_ylim(ymin_data - 0.12 * yrange, ymax_data + 0.18 * yrange)

        delta = phi.max() - phi0
        peak  = phi.max()
        ax.annotate(f'Peak: {peak:.2f}%\nΔφ = {delta:+.3f}%',
                    xy=(0.03, 0.97), xycoords='axes fraction',
                    fontsize=8, color='#1A1A2E',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#FFF9E3',
                              ec='#CCBBAA', alpha=0.9))

        ax.set_ylabel('Porosity (%)', fontsize=10)
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlim(0, time.max())

        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label=injection_label(rp))
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.legend(loc='lower right', framealpha=0.88, fontsize=8,
                  handles=[
                      _make_line_handle(PH_COLOR, 'o',
                          f'Porosity | {phi0:.2f}%→{phi.max():.2f}%'),
                      Line2D([0],[0], color='#1565C0', lw=1, ls=':',
                             label=f'Initial φ={phi0:.1f}%'),
                      inj_patch, mon_patch
                  ])

    else:
        handles = []
        all_phi = []
        for region, (dfc, rpc) in collab_data.items():
            all_phi.extend(dfc['Porosity_%'].tolist())

        for region, (dfc, rpc) in collab_data.items():
            time  = dfc['Time_years']
            phi   = dfc['Porosity_%']
            c     = COMP_COLORS[region]
            n     = len(time)
            every = max(1, n // MARKER_EVERY)
            if region == list(collab_data.keys())[0]:
                add_injection_shading(ax, dfc, rpc)
            ax.plot(time, phi, color=c, linewidth=2.0, linestyle=LINE_STYLE,
                    marker='o', markevery=every, markersize=MARKER_SIZE,
                    markerfacecolor=c, markeredgecolor='white', markeredgewidth=0.4)
            handles.append(_make_line_handle(c, 'o', region))

        # tight y-axis for collab mode too
        arr   = np.array(all_phi)
        yrange = arr.max() - arr.min() if arr.max() != arr.min() else 0.5
        ax.set_ylim(arr.min() - 0.12 * yrange, arr.max() + 0.18 * yrange)

        ax.axhline(4.0, color='#E65C00', linewidth=0.9, linestyle=':', alpha=0.7)
        ax.set_ylabel('Porosity (%)', fontsize=10)
        ax.set_xlabel('Time (years)', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)

        inj_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Injection')
        mon_patch = mpatches.Patch(color='#90CAF9', alpha=0.4, label='Monitoring')
        ax.legend(handles=handles+[inj_patch, mon_patch], framealpha=0.88, fontsize=8,
                  loc='lower right')


PLOT_FNS = {
    'pH':          plot_pH_pCO2,
    'dissolution': plot_dissolution,
    'carbonate':   plot_carbonates,
    'clay':        plot_clays,
    'porosity':    plot_porosity,
}

FIG_TITLES = {
    'pH':          'pH Evolution',
    'dissolution': 'Primary Mineral Dissolution',
    'carbonate':   'Carbonate Precipitation',
    'clay':        'Clay / Silicate Precipitation',
    'porosity':    'Porosity',
}


# ════════════════════════════════════════════════════════════════════════════════
# GENERATE COLLAB PLOTS — Single | Continuous | Pulsed
# ════════════════════════════════════════════════════════════════════════════════
print("=== Generating Collab Plots (Single | Continuous | Pulsed) ===")

for fig_type, plot_fn in PLOT_FNS.items():
    for region in REGIONS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
        fig.patch.set_facecolor('white')

        meta = REGION_META[region]
        fig.suptitle(
            f'{FIG_TITLES[fig_type]} — {meta["long"]}   |   8 bar'
            f'   |   Single / Continuous / Pulsed',
            fontsize=14, fontweight='bold', y=1.01, fontfamily='Arial'
        )

        for col_idx, scenario in enumerate(COLLAB_SCENARIO_ORDER):
            df, rp = load(region, scenario)
            ax = axes[col_idx]

            extra_line = (f'q={meta["q0"]}t/d×{int(rp["T_inj_days"])}d'
                          if fig_type == 'pH' else f'T={meta["T_C"]}°C')
            ax.set_title(f'{SCENARIO_META[scenario]}  |  {extra_line}',
                         fontsize=10, fontweight='bold')

            plot_fn(df, rp, ax, '', collab_mode=False)

            if col_idx > 0:
                ax.set_ylabel('')

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_fig(fig, f'collab_{fig_type}_{region}.png')


# ════════════════════════════════════════════════════════════════════════════════
# GENERATE COMPARATIVE PLOTS — CRBG (red) vs MHOW (blue)
# ════════════════════════════════════════════════════════════════════════════════
print("\n=== Generating Comparative Plots (CRBG red vs MHOW blue) ===")

for fig_type, plot_fn in PLOT_FNS.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'{FIG_TITLES[fig_type]} — CRBG (red) vs MHOW (blue)   |   8 bar',
        fontsize=14, fontweight='bold', y=1.01, fontfamily='Arial'
    )

    for col_idx, scenario in enumerate(SCENARIOS):
        ax = axes[col_idx]
        collab_data = {}
        rp0 = None
        for region in REGIONS:
            df, rp = load(region, scenario)
            collab_data[region] = (df, rp)
            if rp0 is None:
                rp0 = rp

        ax.set_title(SCENARIO_META[scenario], fontsize=10, fontweight='bold')
        plot_fn(collab_data['MHOW'][0], rp0, ax, '',
                collab_mode=True, collab_data=collab_data)

        if col_idx > 0:
            ax.set_ylabel('')

    leg_handles = [
        mpatches.Patch(color=COMP_COLORS['CRBG'], label='CRBG'),
        mpatches.Patch(color=COMP_COLORS['MHOW'], label='MHOW'),
    ]
    fig.legend(handles=leg_handles, loc='upper right',
               bbox_to_anchor=(1.06, 0.95), fontsize=9, framealpha=0.88)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, f'comparative_{fig_type}_all_scenarios.png')


print(f'\nAll plots saved to {OUT_DIR}')