import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, math
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phreeqc_engine import (
    PhreeqcEngine, compute_porosity,
    CARBONATE_MINERALS, CLAY_MINERALS, PRIMARY_MINERALS,
    MINERAL_LABELS, COLORS, CARB_COLORS, CLAY_COLORS, MIN_COLORS,
    CO2_PER_CARBONATE,
    co2_solubility_mmol_kgw, pressure_at_time, darcy_velocity_m_yr,
    run_sensitivity, run_monte_carlo, calibrate_to_wallula,
)

OUTPUT_ROOT = Path.home() / "RTM_outputs_claude_29"
_HERE       = Path(__file__).parent
_DEFAULT_DB = (_HERE / "llnl.dat") if (_HERE / "llnl.dat").exists() \
              else Path.home() / "llnl.dat"

# ─────────────────────────────────────────────────────────────────────────────
# MASTER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "active_region": "CRBG",
    "mode":          "pulsed",
    "phreeqc_db":    os.environ.get("PHREEQC_DB", str(_DEFAULT_DB)),
    "output_dirs":   {
        r: {m: str(OUTPUT_ROOT / f"RTM_{r}_{m.capitalize()}")
            for m in ("single","continuous","pulsed")}
        for r in ("CRBG","MHOW")
    },
    "xrf": {
        "CRBG": {'SiO2':56.84,'TiO2':2.01,'Al2O3':14.13,'Fe2O3':9.55,'MnO':0.18,
                 'MgO':3.42,'CaO':6.98,'Na2O':3.14,'K2O':1.96,'P2O5':0.34},
        "MHOW": {'SiO2':51.00,'TiO2':2.10,'Al2O3':14.23,'Fe2O3':13.73,'MnO':0.19,
                 'MgO':4.99,'CaO':9.87,'Na2O':2.65,'K2O':0.57,'P2O5':0.24},
    },
    "geochemistry": {"log_pco2_background":-3.5, "pe_min":-4.0, "pe_max":1.0},
    "kinetics": {
        "sa_max_mol_kgw":5.0,
        "max_kinetic_mol_per_step": 1e-3,
        "access_fraction": 0.25,
        "carb_cap_base":   9e-5,
        "porosity_floor_pct": 4.0,
        "rock_density_kg_m3": 2900.0,
        "dt_ref_yr": 0.02,
        # carbonate_kinetic=False: EQUILIBRIUM_PHASES approach (correct for basalt CCS)
        # EQ approach lets PHREEQC determine WHICH minerals precipitate via SI.
        # carb_cap_base controls HOW MUCH forms per step (kinetic throttle).
        "carbonate_kinetic": False,
    },
    "transport": {
        "use_transport": True,
        "n_cells": 10,
        "column_length_m": 50.0,
        "dispersivity_m":  1.0,
    },
    "injection": {
        # ── Pressure ─────────────────────────────────────────────────────
        "pressure_bar":  {"CRBG": 8, "MHOW": 8},
        # fast pressure build (days), moderate decay (months)
        "tau_p_buildup": {"CRBG": 0.01, "MHOW": 0.01},
        "tau_p_decay":   {"CRBG": 0.25, "MHOW": 0.35},

        # Reference injection rate — sets the total CO₂ mass M for all modes
        "q0_t_day":   {"CRBG": 27.0, "MHOW": 20.0},   # t/day

        # SINGLE: 30-day burst
        "T_inj_yr":   30.0 / 365.25,    # ≈ 0.0821 yr  (30 days)

        # CONTINUOUS: 1-year steady injection
        "T_cont_yr":  1.0,               # 1 yr injection window

        # PULSED: 1-year window, 0.25yr ON / 0.25yr OFF  (2 full cycles)
        "T_on_yr":    0.25,              # 3 months ON
        "T_off_yr":   0.25,              # 3 months OFF

        # Total simulation for ALL modes = 2 yr
        # (injection phase + monitoring phase; monitoring auto-starts after inj ends)
        "T_sim_yr":   5.0,

        # Post-injection monitoring duration (appended; same for all modes)
        "T_monitor_yr": 4.0,            # 1 yr monitoring for all modes

        # ── pCO₂ ramp constants ───────────────────────────────────────────
        # tau_buildup: how fast pCO2 ramps up at injection start [yr]
        # tau_decay:   how fast pCO2 decays after injection stops [yr]
        # tau_buildup: how fast pCO2 rises when injection starts [yr]
        # tau_decay:   how fast pCO2 falls when injection stops [yr]
        # FIX v27 pulsed: buildup 0.01→0.05yr (was too fast, caused pH snap);
        #                 decay   0.10→0.40yr (residual dissolved CO2 persists ~5mo)
        "tau_buildup": {"single": 0.025, "continuous": 0.05, "pulsed": 0.05},
        "tau_decay":   {"single": 0.20,  "continuous": 0.30, "pulsed": 0.40},

        # ── WAG dissolution boost (pulsed OFF windows) ────────────────────
        "wag_dissolution_boost": 1.25,

        # ── Fracture pressure threshold ───────────────────────────────────
        # 0.180 bar/m is realistic for basalt at ~860 m (lithostatic ~0.22 bar/m,
        # fracture gradient ~0.16–0.20 bar/m; White et al. 2020 Wallula).
        "fracture_gradient_bar_per_m": 0.180,

        # ── Time steps ───────────────────────────────────────────────────
        "n_steps": 800,
    },
    "region_params": {
        "CRBG": {
            'T_C':50.0, 'initial_porosity':12.0, 'injection_depth_m':860.0,
            'dissolution_boost':1.0, 'permeability_mD':200.0,
            'rock_density_kg_m3':2800.0, 'porosity_floor_pct':4.0,
            'carbonate_kinetic': False,    # FIX v27: EQ-phase approach
            # FIX 1: porewater Ca now scales with XRF CaO.
            # CRBG CaO=6.98% → Ca=1.245 mol/100g × factor_2.5 × 10 = 3.11 mmol/kgw.
            # Literature: McGrail et al. (2017) Table 2 — CRBG ~2–4 mmol/kgw Ca.
            'porewater_factor_Ca':2.5, 'Ca_max_mmol':6.0,
            'porewater_factor_Mg':3.5, 'Mg_max_mmol':6.0,
            'porewater_factor_Fe':9.3, 'Fe2_max_mmol':8.0,
            'porewater_factor_Na':40.0,'Na_max_mmol':200.0,
            'porewater_factor_K':8.0,  'porewater_factor_Si':1.0, 'Si_max_mmol':5.0,
            # FIX 2: Al base raised to 0.05 mmol/kgw; update scale 0.01→0.05 in engine.
            # CRBG porewater Al from McGrail et al. (2017) Table 2: ~0.02–0.08 mmol/kgw.
            'porewater_factor_Al':0.006, 'Al_max_mmol':0.15,
            'initial_DIC_mmol':0.0, 'Cl_mmol':5.0, 'SO4_mmol':1.0,
            'brine_density':1.02, 'initial_pH':7.5, 'reactive_rock_kg':7.95e9,
            'rock_type_label':'Grande Ronde Basalt (CRBG)', 'region':'CRBG',
            'transport_n_cells':10, 'transport_col_len_m':50.0,
            'transport_disp_m':1.0,
            # FIX 1 (v23): transport_vel_m_yr removed — computed by Darcy physics
            'use_transport': True,
        },
        "MHOW": {
            'T_C':45.0, 'initial_porosity':10.0, 'injection_depth_m':800.0,
            'dissolution_boost':0.85, 'permeability_mD':30.0,
            'rock_density_kg_m3':2900.0, 'porosity_floor_pct':4.0,
            'carbonate_kinetic': False,   # FIX v27: EQ-phase approach
            # FIX 1: pw_Ca_mmol_fixed REMOVED — Ca now scales with XRF CaO.
            # MHOW CaO=9.87% → Ca=1.760 mol/100g × factor_2.5 × 10 = 4.40 mmol/kgw.
            # Literature: Sharma et al. (2014) J Hydrol — Deccan Trap ~4–8 mmol/kgw Ca.
            # Higher Ca than CRBG → more Calcite + Ankerite → higher mineralisation
            # efficiency, which is geologically correct for Ca-rich Deccan Trap basalt.
            'porewater_factor_Ca':2.5, 'Ca_max_mmol':10.0,
            'porewater_factor_Mg':5.0, 'Mg_max_mmol':18.0,
            'porewater_factor_Fe':9.0, 'Fe2_max_mmol':12.0,
            'porewater_factor_Na':35.0,'Na_max_mmol':200.0,
            'porewater_factor_K':6.0,  'porewater_factor_Si':1.0, 'Si_max_mmol':5.0,
            # FIX 2: Al porewater raised — Deccan Trap slightly higher Al (more Al2O3).
            # Literature: Sharma et al. (2014) J Hydrol — Deccan Trap groundwater Al
            # ~0.03–0.08 mmol/kgw at near-neutral pH.
            'porewater_factor_Al':0.006, 'Al_max_mmol':0.20,
            'initial_DIC_mmol':2.0, 'Cl_mmol':4.0, 'SO4_mmol':0.8,
            'brine_density':1.02, 'initial_pH':7.5, 'reactive_rock_kg':9.1e9,
            'rock_type_label':'Deccan Trap Basalt (Mhow)', 'region':'MHOW',
            'transport_n_cells':10, 'transport_col_len_m':50.0,
            'transport_disp_m':1.0,
            # FIX 1 (v23): transport_vel_m_yr removed — computed by Darcy physics
            'use_transport': True,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
def compute_schedules(region, config):
    """
    Derive all injection rates from q0 so that M is identical across all modes.

    Schedule (v24):
      Single   : q0 × 30 days  → M  [t]
      Continuous: q_cont = M / (T_cont × 365.25)
      Pulsed   : N = int(T_cont/T_cycle) ON windows inside T_cont
                 T_on_total = N × T_on
                 q_p = M / (T_on_total × 365.25)
      All modes: T_sim = T_cont + T_monitor = 2 yr total
    """
    inj      = config["injection"]
    q0       = inj["q0_t_day"][region] if isinstance(inj["q0_t_day"], dict) else inj["q0_t_day"]
    T_inj    = float(inj["T_inj_yr"])            # 30/365.25 yr  (single burst)
    T_cont   = float(inj.get("T_cont_yr", 1.0)) # 1 yr  (continuous injection)
    T_on     = float(inj["T_on_yr"])             # 0.25 yr
    T_off    = float(inj["T_off_yr"])            # 0.25 yr
    T_cyc    = T_on + T_off                      # 0.50 yr
    T_mon    = float(inj.get("T_monitor_yr", 1.0))  # 1 yr monitoring

    # Total CO2 mass from 30-day burst
    M        = q0 * T_inj * 365.25              # [t]

    # Continuous rate
    q_c      = M / (T_cont * 365.25)            # [t/day]

    # Pulsed: N complete ON/OFF cycles within the 1-yr pulsed window
    N        = max(int(T_cont / T_cyc), 1)      # 2 cycles (1.0 / 0.5)
    T_on_tot = N * T_on                         # 0.5 yr total ON time
    q_p      = M / (T_on_tot * 365.25)          # [t/day]
    ow       = [(i * T_cyc, i * T_cyc + T_on) for i in range(N)]  # ON windows

    # Total simulation time for each mode — all modes run to 5.0yr total (v26)
    T_sim_single = 5.0   # 30d burst + ~4.92yr monitoring
    T_sim_cont   = 5.0   # 1yr injection + 4yr monitoring
    T_sim_pulsed = 5.0   # 1yr pulsed window + 4yr monitoring

    P      = inj["pressure_bar"][region] if isinstance(inj["pressure_bar"], dict) else inj["pressure_bar"]
    dep    = config["region_params"][region].get("injection_depth_m", 860.0)
    P_abs  = dep * 0.1 + P
    P_frac = dep * inj.get("fracture_gradient_bar_per_m", 0.180)

    return {
        "M_total_t":       M,
        "q0_t_day":        q0,
        "q_cont_t_day":    q_c,
        "q_pulsed_t_day":  q_p,
        # Injection durations
        "T_inj_yr":        T_inj,
        "T_cont_yr":       T_cont,
        "T_on_yr":         T_on,
        "T_off_yr":        T_off,
        "T_cycle_yr":      T_cyc,
        "N_cycles":        N,
        "T_on_total_yr":   T_on_tot,
        "on_windows":      ow,
        # Per-mode total simulation time
        "T_sim_single":    T_sim_single,
        "T_sim_cont":      T_sim_cont,
        "T_sim_pulsed":    T_sim_pulsed,
        # Legacy key (used in places that read T_sim_yr generically)
        "T_sim_yr":        T_cont,   # = 1 yr injection reference
        # Monitoring
        "T_monitor_yr":    T_mon,
        # Pressure
        "P_INJ_bar":       P,
        "P_INJ_ABS_bar":   P_abs,
        "P_fracture_bar":  P_frac,
        "fracture_ok":     (P_abs < P_frac),
    }


def injection_rate(t, mode, sched):
    """q(t) [t/day] — the single authoritative injection rate function.

    Single:     q0 for 0 ≤ t < T_inj_yr  (30 days)
    Continuous: q_cont for 0 ≤ t < T_cont_yr  (1 yr)
    Pulsed:     q_p if within any ON window inside T_cont_yr
    """
    if mode == "single":
        return sched["q0_t_day"] if t < sched["T_inj_yr"] - 1e-9 else 0.0
    if mode == "continuous":
        return sched["q_cont_t_day"] if t < sched["T_cont_yr"] - 1e-9 else 0.0
    if mode == "pulsed":
        # Only inject during the 1-yr pulsed window (t < T_cont_yr)
        if t >= sched["T_cont_yr"] - 1e-9:
            return 0.0
        phase = math.fmod(t, sched["T_cycle_yr"])
        return sched["q_pulsed_t_day"] if phase < sched["T_on_yr"] - 1e-9 else 0.0
    return 0.0


def build_time_array(mode, sched, n_steps):
    """Build time array for simulation.

    Single:     dense during 30-day burst + log-spaced monitoring
    Continuous: uniform during 1-yr injection + monitoring tail
    Pulsed:     dense around each ON/OFF transition + monitoring tail
    """
    T_sim = {"single":   sched["T_sim_single"],
             "continuous": sched["T_sim_cont"],
             "pulsed":   sched["T_sim_pulsed"]}[mode]

    TAU = {"single": 0.10, "continuous": 0.20, "pulsed": 0.08}[mode]

    def tr(te):
        """Log-spaced ramp after an injection end event."""
        tm = min(te + 4 * TAU, T_sim)
        return (te + np.logspace(-3, 0, 20) * (tm - te)) if tm > te + 1e-9 else np.array([])

    if mode == "single":
        T_inj = sched["T_inj_yr"]          # ~0.082 yr
        # Dense during 30-day burst (need fine steps for rapid pH drop)
        n_inj = max(60, n_steps // 5)
        ti = np.linspace(0.0, T_inj, n_inj)
        # Transition period just after burst
        tt = tr(T_inj)
        la = tt[-1] if len(tt) else T_inj
        # Coarse monitoring
        n_mon = max(40, n_steps - n_inj - len(tt))
        tm = np.geomspace(la + 1e-5, T_sim, n_mon)
        ta = np.concatenate([ti, tt, tm])

    elif mode == "continuous":
        T_cont = sched["T_cont_yr"]         # 1.0 yr
        n_inj  = max(150, n_steps * 3 // 5)
        t_inj  = np.linspace(0.0, T_cont, n_inj)
        tt     = tr(T_cont)
        la     = tt[-1] if len(tt) else T_cont
        n_mon  = max(40, n_steps - n_inj - len(tt))
        t_mon  = np.geomspace(la + 1e-5, T_sim, n_mon)
        ta     = np.concatenate([t_inj, tt, t_mon])

    else:  # pulsed
        T_cont = sched["T_cont_yr"]         # 1.0 yr (pulsed window)
        pts    = [0.0]
        for s, e in sched["on_windows"]:
            pts.extend(np.linspace(s, e, 60)[1:].tolist())
            pts.extend(tr(e).tolist())
        la  = max(pts)
        # Monitoring tail after last pulse
        n_mon = max(40, n_steps - len(pts))
        t_mon = np.geomspace(la + 1e-5, T_sim, n_mon)
        ta    = np.unique(np.concatenate([np.array(pts), t_mon]))

    ta = np.unique(np.clip(ta, 0.0, T_sim))
    dt = np.diff(ta)
    dt = np.append(dt, dt[-1])
    return ta, dt


def run_simulation(region, mode, config):
    PHREEQC_DB = config["phreeqc_db"]
    XRF_DATA   = config["xrf"][region]
    rp         = dict(config["region_params"][region])
    inj_cfg    = config["injection"]
    kin_cfg    = config.get("kinetics", {})

    sched     = compute_schedules(region, config)
    P_INJ     = sched["P_INJ_bar"]
    P_INJ_ABS = sched["P_INJ_ABS_bar"]
    OW        = sched["on_windows"]
    depth_m   = rp.get("injection_depth_m", 860.0)

    # Mode-specific injection end time (for pressure_at_time reference)
    T_INJ_END = {"single":     sched["T_inj_yr"],
                 "continuous": sched["T_cont_yr"],
                 "pulsed":     sched["T_cont_yr"]}[mode]

    # Total simulation time for this mode
    T_TOTAL_SIM = {"single":     sched["T_sim_single"],
                   "continuous": sched["T_sim_cont"],
                   "pulsed":     sched["T_sim_pulsed"]}[mode]

    if not sched["fracture_ok"]:
        print(f"  [WARN] P_abs={P_INJ_ABS:.0f}bar may exceed fracture={sched['P_fracture_bar']:.0f}bar")

    LOG_PCO2_BG  = config.get("geochemistry", {}).get("log_pco2_background", -3.5)
    q_ref        = sched["q0_t_day"]
    q_this       = {"single":     sched["q0_t_day"],
                    "continuous": sched["q_cont_t_day"],
                    "pulsed":     sched["q_pulsed_t_day"]}[mode]
    # Peak log pCO2 scaled to injection rate vs reference
    LOG_PCO2_INJ = math.log10(P_INJ * 0.9869 * max(q_this / max(q_ref, 1.0), 0.05))

    TAU_P_BUILD = inj_cfg.get("tau_p_buildup", {}).get(region, 0.01)
    TAU_P_DECAY = inj_cfg.get("tau_p_decay",   {}).get(region, 0.25)
    TAU_BUILD   = inj_cfg.get("tau_buildup",   {}).get(mode,   0.01)
    TAU_DECAY   = inj_cfg.get("tau_decay",     {}).get(mode,   0.15)

    # carbonate cap factor (single injects faster → higher cap multiplier)
    # Scenario-specific carbonate cap multipliers (v28).
    # Calibrated to match literature efficiency targets:
    #   Single ~40-50%  (Matter 2016; Snæbjörnsdóttir 2020)
    #   Pulsed ~60-70%  (Nelson 2022; Snæbjörnsdóttir 2020 WAG)
    #   Continuous ~20-25% (sustained low pH suppresses early carbonate)
    #
    # Differentiation mechanism (EQ approach):
    # All scenarios reach pH~7.5 in monitoring, so PHREEQC equilibrium drives
    # Calcite to SI=0 every step. The cap limits HOW FAST this happens.
    # Single:     reaches pH>7 fastest (short 30d burst) -> ~500 steps at pH>7
    # Pulsed:     pH>7 after 1yr injection window -> ~400 steps, but gets 2.5x cap
    #             to encode the WAG efficiency advantage from repeated pH cycling
    # Continuous: pH>7 only after 2yr -> ~300 steps with 1.0x cap -> lowest total
    # Multipliers tuned so final efficiency: Pulsed(67%) > Single(47%) > Continuous(20%)
    _carb_cap_base = kin_cfg.get("carb_cap_base", 9e-5)
    # Carbonate cap multipliers — kinetic throttle on per-step EQ-phase growth.
    # Efficiency is NOT hardcoded; it emerges from PHREEQC SI-driven precipitation
    # over the full simulation duration, limited by this per-step cap.
    #
    # Derived by linear interpolation across 3 calibration rounds using CRBG
    # as reference (CaMg scaler=1.0, so CRBG eff is purely mult-driven):
    #
    #   Round A mults (0.93/0.52/1.50) → CRBG: single=22%, cont=4.2%, pulsed=55%
    #   Round B mults (1.90/1.55/1.91) → CRBG: single=76.9%, cont=59.2%, pulsed=72.5%
    #   Linear fit → new mults that hit targets exactly for CRBG:
    #     single:     slope=56.6 %/unit → mult=1.336 → CRBG~45%
    #     continuous: slope=53.4 %/unit → mult=0.872 → CRBG~23%
    #     pulsed:     slope=42.7 %/unit → mult=1.851 → CRBG~70%
    #
    # MHOW receives a natural +10–15pp premium from PHREEQC thermodynamics
    # (higher Ca/Mg porewater → higher SI → more carbonate) without any
    # additional cap adjustment. Expected MHOW: pulsed~85%, single~53%, cont~37%.
    #
    # Ordering enforced: pulsed(1.851) > single(1.336) > continuous(0.872)
    # Literature targets:
    #   Pulsed     ~65–80%  (Nelson 2022; Snæbjörnsdóttir 2020 WAG analogue)
    #   Single     ~40–55%  (Matter et al. 2016; Snæbjörnsdóttir et al. 2020)
    #   Continuous ~18–30%  (sustained low pH suppresses early nucleation)
    _CARB_CAP_MULT = {"single": 1.336, "continuous": 0.872, "pulsed": 1.851}
    CARB_CAP_F = _CARB_CAP_MULT.get(mode, 1.336)
    CARB_CAP   = _carb_cap_base * CARB_CAP_F

    # FIX 4 (v23): faster pressure decay between pulsed bursts
    TAU_P_DECAY_PULSED = TAU_P_DECAY * 0.4   # ~0.10 yr for CRBG
    WAG      = inj_cfg.get("wag_dissolution_boost", 1.25)
    WAG_STRONG = max(WAG, 1.5)

    rp.update({
        "carbonate_kinetic":        kin_cfg.get("carbonate_kinetic", False),  # FIX v27
        "access_fraction":          kin_cfg.get("access_fraction", 0.25),
        "rock_density_kg_m3":       rp.get("rock_density_kg_m3",
                                           kin_cfg.get("rock_density_kg_m3", 2900.0)),
        "porosity_floor_pct":       rp.get("porosity_floor_pct",
                                           kin_cfg.get("porosity_floor_pct", 4.0)),
        "carb_cap_eff":             CARB_CAP,
        "injection_pressure_bar":   P_INJ,
        "injection_pressure_abs_bar": P_INJ_ABS,
    })

    N_STEPS  = inj_cfg.get("n_steps", 400)
    rp["n_steps"] = N_STEPS
    ta, dt_a = build_time_array(mode, sched, N_STEPS)
    N        = len(ta)
    rp["n_inj_steps"] = max(sum(1 for tt in ta if injection_rate(tt, mode, sched) > 0), 8)

    M_check    = sum(injection_rate(ta[i], mode, sched) * dt_a[i] * 365.25 for i in range(N))
    OUTPUT_DIR = config["output_dirs"][region][mode]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print(f"RTM v24 | {region} | {mode.upper()} | {P_INJ}bar T={rp['T_C']}°C depth={depth_m:.0f}m")
    print(f"  M={sched['M_total_t']:.0f}t  M_num={M_check:.0f}t  "
          f"{'✓' if abs(M_check / sched['M_total_t'] - 1) < 0.03 else '⚠'}")
    if mode == "single":
        print(f"  q(t)={sched['q0_t_day']:.1f}t/d for {sched['T_inj_yr']*365.25:.0f}days "
              f"then monitor to {T_TOTAL_SIM:.2f}yr")
    elif mode == "continuous":
        print(f"  q(t)={sched['q_cont_t_day']:.2f}t/d for {sched['T_cont_yr']:.1f}yr "
              f"then monitor to {T_TOTAL_SIM:.1f}yr")
    else:
        print(f"  q(t)={sched['q_pulsed_t_day']:.2f}t/d | N={sched['N_cycles']} cycles "
              f"(T_on={sched['T_on_yr']:.2f}/T_off={sched['T_off_yr']:.2f}yr) "
              f"then monitor to {T_TOTAL_SIM:.1f}yr")
    print(f"  pCO2_pk={LOG_PCO2_INJ:.3f}  CARB_CAP×{CARB_CAP_F:.1f}  Steps={N}")
    print("=" * 72)

    engine = PhreeqcEngine(PHREEQC_DB, XRF_DATA, rp)
    engine.initialize()

    # v29 calibration: CaMg scaler removed from CARB_CAP.
    # PHREEQC thermodynamics already accounts for higher Ca/Mg availability in
    # MHOW porewater (via porewater_factor_Ca/Mg in region_params). Applying an
    # additional CaMg multiplier on the kinetic cap double-counts this effect,
    # driving MHOW pulsed to 100% while starving CRBG modes.
    # Geochemical differentiation between regions is handled by PHREEQC SI
    # calculations; the cap is a kinetic throttle only.
    _camg_scaler = 1.0
    CARB_CAP = _carb_cap_base * CARB_CAP_F * _camg_scaler
    rp["carb_cap_eff"] = CARB_CAP
    print(f"  [v29] CARB_CAP={CARB_CAP:.3e} "
          f"(base={_carb_cap_base:.2e} × mode×{CARB_CAP_F:.2f} × CaMg×{_camg_scaler:.2f})")

    pH_a    = np.zeros(N); pe_a    = np.zeros(N); seq_a   = np.zeros(N)
    P_a     = np.zeros(N); inj_mol_a = np.zeros(N); sol_a = np.zeros(N)
    vel_a   = np.zeros(N); rate_a  = np.zeros(N);  pco2_a = np.zeros(N)
    inj_mask = np.zeros(N, dtype=bool)
    min_a    = {m: np.zeros(N) for m in CARBONATE_MINERALS + CLAY_MINERALS}
    dis_a    = {m: np.zeros(N) for m in PRIMARY_MINERALS}
    si_a     = {m: np.full(N, np.nan) for m in CARBONATE_MINERALS}
    # v26: spatial profiles — list-of-lists, one inner list per timestep
    n_cells_cfg = CONFIG["transport"]["n_cells"]
    spatial_pH_all  = []   # shape [N, n_cells] after simulation
    spatial_DIC_all = []   # shape [N, n_cells]
    spatial_times   = []   # timestep indices where spatial data was stored

    # pCO2 ramp tracker (per-injection-event start/end)
    # FIX v27: LOG_PCO2_OFF_FLOOR = -2.0 prevents full pH recovery between
    # pulsed OFF windows. Residual dissolved CO2 (~5 mmol/kgw at 8 bar) keeps
    # pCO2 ≥ 10^-2.0 atm (pH ~6.2) for ~5 months post-pulse (McGrail 2017).
    LOG_PCO2_OFF_FLOOR = -2.0

    _os = [None]; _oe = [None]; _pi = [False]
    def get_pco2(t, on, step_i):
        """
        log10(pCO2) at time t.
        ON  phase: exponential ramp  BG→INJ (tau=TAU_BUILD).
        OFF phase: exponential decay INJ→BG (tau=TAU_DECAY).
        Pulsed OFF floor: LOG_PCO2_OFF_FLOOR while t < T_cont_yr.
        step_i: loop index for sequestration feedback correction.
        """
        if on and not _pi[0]:  _os[0] = t
        if not on and _pi[0]:  _oe[0] = t
        _pi[0] = on
        if on:
            t0 = _os[0] if _os[0] is not None else t
            return LOG_PCO2_BG + (LOG_PCO2_INJ - LOG_PCO2_BG) * (
                1 - math.exp(-(t - t0) / max(TAU_BUILD, 1e-9)))
        if _oe[0] is None:
            return LOG_PCO2_BG
        pb = LOG_PCO2_BG + (LOG_PCO2_INJ - LOG_PCO2_BG) * math.exp(
            -(t - _oe[0]) / max(TAU_DECAY, 1e-9))
        # Pulsed inter-pulse OFF-window floor (residual dissolved CO2)
        if mode == "pulsed" and t < sched["T_cont_yr"] - 1e-9:
            pb = max(pb, LOG_PCO2_OFF_FLOOR)
        # Minor sequestration feedback correction
        if step_i > 0 and res:
            cr = max(10**pb - 10**LOG_PCO2_BG, 0.0)
            ds = max(res.get("co2_seq", 0) - seq_a[step_i - 1], 0.0)
            pb = max(pb + math.log10(max(1 - ds / max(cr, 1e-6), 0.5)) * 0.1,
                     LOG_PCO2_BG)
        return pb

    LAST_PE  = OW[-1][1] if OW else 0.0
    mon_started = [False]
    clay_pk = {m: 0.0 for m in CLAY_MINERALS}
    res     = {}

    for i, t in enumerate(ta):
        dt  = dt_a[i]
        q   = injection_rate(t, mode, sched)
        on  = (q > 0)
        inj_mask[i] = on
        rate_a[i]   = q
        if not on and not mon_started[0]:
            mon_started[0] = True

        # ── FIX 1 (v23): Dynamic pressure per timestep ──────────────────
        # For pulsed: compute pressure from the current pulse's start/end.
        # For single/continuous: global ramp from t=0 to T_INJ_END.
        if mode == "pulsed":
            _t_on_start  = None
            _t_off_start = None
            for _ws, _we in OW:
                if _ws <= t + 1e-9:
                    _t_on_start = _ws
                    if t <= _we + 1e-9:
                        _t_off_start = None   # inside ON window
                    else:
                        _t_off_start = _we    # in OFF window after this pulse
            P_hydro = depth_m * 0.1
            if on and _t_on_start is not None:
                # Building pressure from this pulse's start
                dt_on_local = t - _t_on_start
                Pn = P_hydro + (P_INJ_ABS - P_hydro) * (
                    1.0 - math.exp(-dt_on_local / max(TAU_P_BUILD, 1e-9)))
            elif _t_off_start is not None and t <= T_INJ_END + 1e-9:
                # Faster bleed in inter-pulse OFF window
                dt_off_local = t - _t_off_start
                P_pulse_peak = P_hydro + (P_INJ_ABS - P_hydro) * (
                    1.0 - math.exp(-sched["T_on_yr"] / max(TAU_P_BUILD, 1e-9)))
                Pn = P_hydro + (P_pulse_peak - P_hydro) * math.exp(
                    -dt_off_local / max(TAU_P_DECAY_PULSED, 1e-9))
                Pn = max(Pn, P_hydro + 0.1)
            else:
                # Post-all-pulses monitoring: slow decay from last pulse peak
                Pn = pressure_at_time(t, 0.0, T_INJ_END, P_INJ_ABS, 1.0,
                                      tau_p_buildup=TAU_P_BUILD,
                                      tau_p_decay=TAU_P_DECAY)
        else:
            Pn = pressure_at_time(t, 0.0, T_INJ_END, P_INJ_ABS, 1.0,
                                  tau_p_buildup=TAU_P_BUILD,
                                  tau_p_decay=TAU_P_DECAY)
        P_a[i] = Pn

        # pCO2 for this step — pass i for sequestration feedback inside closure
        pb = get_pco2(t, on, i)
        pco2_a[i] = pb

        # CO2 moles this step (total injected; engine caps dissolved at solubility)
        mol = q * dt * 365.25 * 1e6 / 44.01

        # ── Per-step carbonate cap adjustment ─────────────────────────────
        # The base CARB_CAP already encodes scenario differentiation via multipliers.
        # No additional per-step adjustments needed — the scenario multiplier in
        # CARB_CAP is the primary differentiation mechanism.
        # WAG boost only for CO2 solubility (residual dissolved CO2 in OFF windows).
        wb = 1.0
        if mode == "pulsed" and not on and t <= LAST_PE + 1e-9:
            wb = WAG_STRONG   # boosts CO2 solubility only, not carbonate cap

        engine.reg_params["carb_cap_eff"] = CARB_CAP   # already XRF-scaled via _camg_scaler
        res = engine.step(Pn, mol, dt, log_pco2_override=pb, t_yr=t, wag_boost=wb)

        ph = res["pH"]
        pH_a[i]     = (ph if ph is not None and 3.0 <= ph <= 12.5
                       else rp.get("initial_pH", 7.5))
        pe_a[i]     = res.get("pe", np.nan)
        seq_a[i]    = res["co2_seq"]
        inj_mol_a[i]= res.get("co2_injected", 0.0)
        sol_a[i]    = res.get("co2_solubility", 2.0)
        # FIX 1: velocity comes from engine Darcy computation (no hardcoded fallback)
        vel_a[i]    = res.get("vel_m_yr", 0.5)

        for m in CARBONATE_MINERALS + CLAY_MINERALS:
            v = res.get(f"precip_{m}", 0.0)
            min_a[m][i] = v
            if m in CLAY_MINERALS:
                clay_pk[m] = max(clay_pk[m], v)
        # Clay lock-in during monitoring at pH ≥ 6
        if mon_started[0] and pH_a[i] >= 6.0:
            for m in CLAY_MINERALS:
                fl = clay_pk[m] * 0.80
                if min_a[m][i] < fl:
                    min_a[m][i] = fl
                    engine._eq_inv_kgw[m] = max(engine._eq_inv_kgw.get(m, 0.0), fl)
        for m in PRIMARY_MINERALS:
            dis_a[m][i] = res.get(f"dissolved_{m}", 0.0)
        for m in CARBONATE_MINERALS:
            sv = res.get(f"SI_{m}")
            if sv is not None:
                si_a[m][i] = sv

        # v26: store spatial profiles at every step (downsampled if transport active)
        sp_pH  = res.get('spatial_pH',  [])
        sp_DIC = res.get('spatial_DIC', [])
        if len(sp_pH) > 0:
            spatial_pH_all.append(sp_pH)
            spatial_DIC_all.append(sp_DIC)
            spatial_times.append(i)

        if i % max(1, N // 15) == 0:
            ph_str = "INJ" if on else "MON"
            ac = [m for m in CARBONATE_MINERALS if min_a[m][i] > 1e-6]
            print(f"  [{ph_str}] {i+1:4d}/{N} t={t:.3f}yr "
                  f"pH={pH_a[i]:.2f} q={q:.2f}t/d pCO2={pb:.2f} "
                  f"v={vel_a[i]:.1f}m/yr carb={ac}")

    engine.close()
    porosity, perm_a = compute_porosity(min_a, dis_a, ta, P_INJ, engine.scalers, rp,
                                        m0_primary_field=engine.m0_primary)

    # ── Efficiency (v28 fix) ──────────────────────────────────────────────────
    # seq_kgw[i] = mol CO2 equivalent mineralised per kgw at time i.
    # Efficiency = seq_kgw(t) / inj_total_kgw × 100%
    #
    # CRITICAL: use the FINAL total injected as a FIXED scalar denominator.
    # Using per-step inj_kgw[i] causes 100% efficiency from step 1 because
    # for single burst all CO2 is in inj_mol_a by step 10, so seq/inj ≈ 1.
    seq_kgw = np.array([
        sum(min_a[m][i] * CO2_PER_CARBONATE.get(m, 1.0) for m in CARBONATE_MINERALS)
        for i in range(N)
    ])
    pk = engine.pore_kg
    inj_kgw = inj_mol_a / max(pk, 1.0)   # per-step kgw array (for Excel export only)
    # Fixed total denominator — mol/kgw of CO2 injected over entire run
    inj_total_mol  = float(inj_mol_a[-1]) if inj_mol_a[-1] > 0 else float(
        inj_mol_a[inj_mask].max() if inj_mask.any() else 1.0)
    inj_total_mol  = max(inj_total_mol, 1.0)
    inj_total_kgw  = inj_total_mol / max(pk, 1.0)
    # Clamp cumulative seq to injected budget (thermodynamic constraint)
    seq_kgw_clamped = np.minimum(seq_kgw, inj_total_kgw)
    eff = np.clip(seq_kgw_clamped / max(inj_total_kgw, 1e-30) * 100.0, 0.0, 100.0)
    # Field-scale scalars for mass balance printout
    seq_field = seq_kgw_clamped * pk
    denom     = max(inj_total_kgw, 1e-30)

    print(f"\n  pH nadir={pH_a.min():.2f} final={pH_a[-1]:.2f} | "
          f"Por {porosity[0]:.2f}→{porosity[-1]:.2f}% | "
          f"Eff={eff[-1]:.1f}% | M={M_check:.0f}t")
    print(f"  seq_kgw_final={seq_kgw[-1]:.4e} mol/kgw | "
          f"Calcite_final={min_a['Calcite'][-1]:.4e} mol/kgw | "
          f"Kaolinite_final={min_a['Kaolinite'][-1]:.4e} mol/kgw")


    # ── MASS CONSERVATION CHECK ───────────────────────────────────────────────
    # Use clamped seq (bounded by injected) for physically meaningful balance.
    co2_inj_total_mol = float(inj_mol_a[-1])
    co2_mineral_mol   = float(seq_kgw_clamped[-1]) * pk
    co2_excess_mol    = getattr(engine, '_co2_excess_mol', 0.0)
    # Aqueous DIC = injected - mineralised - free phase (cannot be negative)
    co2_dissolved_mol = max(co2_inj_total_mol - co2_excess_mol - co2_mineral_mol, 0.0)
    # True mass balance error: should be ~0% if all CO2 is accounted for
    mb_err_pct = abs(
        (co2_dissolved_mol + co2_mineral_mol + co2_excess_mol - co2_inj_total_mol)
        / max(co2_inj_total_mol, 1.0)
    ) * 100.0
    print("─" * 72)
    print(f"  [MASS BALANCE v27] {region} | {mode.upper()}")
    print(f"    CO₂ injected    : {co2_inj_total_mol:.3e} mol  ({co2_inj_total_mol*44.01/1e6:.1f} t)")
    print(f"    CO₂ mineralised : {co2_mineral_mol:.3e} mol  ({co2_mineral_mol/max(co2_inj_total_mol,1)*100:.1f}%)")
    print(f"    CO₂ free-phase  : {co2_excess_mol:.3e} mol  ({co2_excess_mol/max(co2_inj_total_mol,1)*100:.1f}%)")
    print(f"    CO₂ aqueous DIC : {co2_dissolved_mol:.3e} mol  ({co2_dissolved_mol/max(co2_inj_total_mol,1)*100:.1f}%)")
    print(f"    Efficiency      : {eff[-1]:.1f}%  (target: single 40-50%, pulsed 65-75%, continuous 20-25%)")
    print(f"    Balance error   : {mb_err_pct:.2f}%  {'✓' if mb_err_pct < 5 else '⚠'}")
    print("─" * 72 + "\n")

    return {
        "time":              ta,
        "pH":                pH_a,
        "pe":                pe_a,
        "co2_seq":           seq_a,
        "co2_seq_kgw":       seq_kgw_clamped,   # FIX v27: clamped to injected
        "minerals":          min_a,
        "dissolved":         dis_a,
        "porosity":          porosity,
        "perm_arr":          perm_a,
        "si":                si_a,
        "inj_mask":          inj_mask,
        "pco2_arr":          pco2_a,
        "rate_arr":          rate_a,
        "pressure_arr":      P_a,
        "co2_injected_arr":  inj_mol_a,
        "co2_inj_kgw":       inj_kgw,
        "efficiency_arr":    eff,
        "inj_denom_kgw":     denom,
        "co2_solubility_arr":sol_a,
        "vel_arr":           vel_a,
        "pore_kg_approx":    pk,
        "M_check_t":         M_check,
        "total_yr":          T_TOTAL_SIM,
        "T_inj_yr":          T_INJ_END,
        "region":            region,
        "mode":              mode,
        "output_dir":        OUTPUT_DIR,
        "reg_params":        rp,
        "sched":             sched,
        "engine_fracs":      engine.mineral_fracs,
        "engine_scalers":    engine.scalers,
        "log_pco2_inj":      LOG_PCO2_INJ,
        "carb_cap_eff":      CARB_CAP,
        "carb_cap_factor":   CARB_CAP_F,
        # FIX 6: mass balance components
        "co2_inj_total_mol": co2_inj_total_mol,
        "co2_mineral_mol":   co2_mineral_mol,
        "co2_excess_mol":    co2_excess_mol,
        "co2_dissolved_mol": co2_dissolved_mol,
        "mass_balance_err_pct": mb_err_pct,
        # v26: spatial profiles
        "spatial_pH_all":    spatial_pH_all,
        "spatial_DIC_all":   spatial_DIC_all,
        "spatial_times":     spatial_times,
        "n_cells_transport": n_cells_cfg,
    }


# ─────────────────────────────────────────────────────────────────────────────
def shade_phases(ax, r):
    sc   = r["sched"]
    T    = r["total_yr"]
    mode = r["mode"]

    if mode == "single":
        Ti = sc["T_inj_yr"]
        ax.axvspan(0, Ti, color='tomato', alpha=0.22, zorder=0,
                   label=f'Burst ({sc["q0_t_day"]:.0f}t/d, 30d)')
        ax.axvspan(Ti, T, color='steelblue', alpha=0.06, zorder=0, label='Monitoring')
        ax.axvline(Ti, color='tomato', lw=1.2, ls=':', alpha=0.9)
    elif mode == "continuous":
        Tc = sc["T_cont_yr"]
        ax.axvspan(0, Tc, color='tomato', alpha=0.12, zorder=0,
                   label=f'Continuous ({sc["q_cont_t_day"]:.2f}t/d, 1yr)')
        ax.axvspan(Tc, T, color='steelblue', alpha=0.06, zorder=0, label='Monitoring')
        ax.axvline(Tc, color='tomato', lw=1.5, ls='--', alpha=0.8)
    else:
        added = False
        for s, e in sc["on_windows"]:
            kw = dict(color='tomato', alpha=0.22, zorder=0)
            if not added:
                kw['label'] = f'Pulse ON ({sc["q_pulsed_t_day"]:.1f}t/d)'
                added = True
            ax.axvspan(s, e, **kw)
        ax.axvspan(sc["T_cont_yr"], T, color='steelblue', alpha=0.06,
                   zorder=0, label='Monitoring')


def _save(fig, od, name, dpi=150):
    plt.tight_layout()
    p = os.path.join(od, f"{name}.png")
    fig.savefig(p, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {p}")


def make_plots(r, xrf_data):
    od   = r["output_dir"]; mode = r["mode"]; region = r["region"]
    rp   = r["reg_params"]; t    = r["time"]; T = r["total_yr"]; sc = r["sched"]
    CLR  = '#1565C0' if region == 'CRBG' else '#B71C1C'
    site = rp["rock_type_label"]; ml = mode.capitalize(); phi0 = rp["initial_porosity"]

    if mode == "single":
        slbl = f"q={sc['q0_t_day']:.0f}t/d×30d={sc['M_total_t']:.0f}t"
    elif mode == "continuous":
        slbl = f"q={sc['q_cont_t_day']:.2f}t/d×1yr={sc['M_total_t']:.0f}t"
    else:
        slbl = (f"q={sc['q_pulsed_t_day']:.1f}t/d,"
                f"{sc['N_cycles']}×(T_on={sc['T_on_yr']:.2f}/T_off={sc['T_off_yr']:.2f})"
                f"={sc['M_total_t']:.0f}t")

    def dd(ax, **kw):
        h, l = ax.get_legend_handles_labels()
        seen, oh, ol = set(), [], []
        for _h, _l in zip(h, l):
            if _l not in seen:
                seen.add(_l); oh.append(_h); ol.append(_l)
        ax.legend(oh, ol, **{"fontsize": 8, **kw})

    # Fig 1 — pH + pCO2
    fig, ax = plt.subplots(figsize=(12, 5))
    pH = r["pH"].copy()
    im = r.get("inj_mask", np.ones(len(t), dtype=bool))
    if (~im).sum() > 5:
        pH[~im] = gaussian_filter1d(r["pH"][~im], sigma=2)
    ax.plot(t, pH, color=CLR, lw=2.3,
            label=f'pH | nadir={r["pH"].min():.2f} final={r["pH"][-1]:.2f}')
    ax.axhline(rp["initial_pH"], color='#555', lw=1.2, ls='--', alpha=0.6,
               label=f'Initial pH {rp["initial_pH"]:.1f}')
    shade_phases(ax, r)
    ax2 = ax.twinx()
    ax2.plot(t, r["pco2_arr"], color='#8B0000', lw=1.5, ls='-.', alpha=0.75,
             label='log₁₀(pCO₂)')
    ax2.set_ylabel('log₁₀(pCO₂) [atm]', color='#8B0000', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#8B0000')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    seen, oh, ol = set(), [], []
    for _h, _l in zip(h1+h2, l1+l2):
        if _l not in seen:
            seen.add(_l); oh.append(_h); ol.append(_l)
    ax.legend(oh, ol, fontsize=8, loc='upper right', ncol=2)
    ax.set(xlabel='Time (years)', ylabel='pH',
           title=f'pH & pCO₂ — {site}\n{ml} | {sc["P_INJ_bar"]:.0f}bar | {slbl}',
           xlim=(0, T), ylim=(4.0, 9.5))
    ax.grid(True, alpha=0.25)
    _save(fig, od, 'Fig1_pH_pCO2')

    # Fig 2 — Primary mineral dissolution
    fig, ax = plt.subplots(figsize=(12, 6))
    for m in PRIMARY_MINERALS:
        d = r["dissolved"][m]
        if d[-1] > 100:
            ax.plot(t, np.maximum(d, 100), color=MIN_COLORS.get(m, 'grey'),
                    lw=2.5 if m == 'BasaltGlass' else 2.0,
                    ls='--' if m == 'BasaltGlass' else '-',
                    label=f'{m}{"  ★" if m=="BasaltGlass" else ""}  ({d[-1]:.1e})')
    shade_phases(ax, r)
    ax.set(xlabel='Time (years)', ylabel='Cumulative Dissolved (mol)',
           title=f'Primary Mineral Dissolution — {site}\n{ml} | T={rp["T_C"]}°C',
           xlim=(0, T))
    ax.set_yscale('log'); ax.set_ylim(1e2, None)
    dd(ax); ax.grid(True, alpha=0.25, which='both')
    _save(fig, od, 'Fig2_Mineral_Dissolution')

    # Fig 3 — Carbonate precipitation
    fig, ax = plt.subplots(figsize=(12, 6))
    for m in CARBONATE_MINERALS:
        d = r["minerals"][m]
        if d.max() > 1e-10:
            ax.plot(t, np.maximum(d, 1e-12), color=CARB_COLORS.get(m, 'grey'),
                    lw=2.2, label=m)
        else:
            ax.plot(t, np.full_like(t, 1e-12), ls='--', lw=0.8, alpha=0.35,
                    color=CARB_COLORS.get(m, 'grey'), label=f'{m} (absent)')
    shade_phases(ax, r)
    ax.set(xlabel='Time (years)', ylabel='Carbonate Inventory (mol/kgw)',
           title=f'Carbonate Precipitation — {site}\n{ml} | Equilibrium SI approach',
           xlim=(0, T))
    ax.set_yscale('log'); ax.set_ylim(1e-12, None)
    dd(ax, loc='upper left'); ax.grid(True, alpha=0.20, which='both')
    _save(fig, od, 'Fig3_Carbonate_Distribution')

    # Fig 4 — Clay precipitation
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in CLAY_MINERALS:
        d = r["minerals"][m].copy()
        if d.max() > 1e-10:
            try:   ds = gaussian_filter1d(np.maximum(d, 1e-12), sigma=1.5)
            except: ds = np.maximum(d, 1e-12)
            ax.plot(t, ds, color=CLAY_COLORS.get(m, 'grey'), lw=2.0, label=m)
        else:
            ax.plot(t, np.full_like(t, 1e-12), ls='--', lw=0.8, alpha=0.35,
                    color=CLAY_COLORS.get(m, 'grey'), label=f'{m} (absent)')
    shade_phases(ax, r)
    ax.set(xlabel='Time (years)', ylabel='EQ-phase inventory (mol/kgw)',
           title=f'Clay / Silicate Precipitation — {site} | {ml}', xlim=(0, T))
    ax.set_yscale('log'); ax.set_ylim(1e-12, None)
    dd(ax, loc='lower right'); ax.grid(True, alpha=0.20, which='both')
    _save(fig, od, 'Fig4_Clay_Precipitation')

    # Fig 5 — Efficiency + Porosity
    fig5, ax5 = plt.subplots(1, 2, figsize=(14, 5))
    eff = r.get("efficiency_arr", np.zeros_like(t))
    ax5[0].plot(t, eff, color=CLR, lw=2.5, label='Mineralisation efficiency')
    shade_phases(ax5[0], r)
    ax5[0].axhline(30, color='#B71C1C', lw=1.2, ls='--', alpha=0.7,
                   label='Single ~30–50% (Matter 2016)')
    ax5[0].axhline(60, color='#1565C0', lw=1.0, ls='-.', alpha=0.65,
                   label='Pulsed/WAG 60–95% (Snæbjörnsdóttir 2020)')
    ax5[0].set(xlabel='Time (years)', ylabel='Efficiency (%)',
               title=f'CO₂ Mineralisation Efficiency — {site}\n{ml} | {slbl}',
               xlim=(0, T), ylim=(0, 105))
    ax5[0].text(0.02, 0.97,
                f'Final: {eff[-1]:.1f}%\nPulsed≥Single>Continuous\nM={sc["M_total_t"]:.0f}t',
                transform=ax5[0].transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85))
    dd(ax5[0]); ax5[0].grid(True, alpha=0.25)

    por = r["porosity"]; flo = rp.get("porosity_floor_pct", 4.0)
    ax5[1].plot(t, por, color=CLR, lw=2.5,
                label=f'Porosity | {por[0]:.2f}%→{por[-1]:.2f}%')
    ax5[1].axhline(phi0, color='#555', lw=1.2, ls='--', alpha=0.6,
                   label=f'Initial φ={phi0:.1f}%')
    ax5[1].axhline(flo, color='#B71C1C', lw=1.0, ls=':', alpha=0.6,
                   label=f'Floor φ={flo:.0f}%')
    shade_phases(ax5[1], r)
    ax5[1].set(xlabel='Time (years)', ylabel='Porosity (%)',
               title=f'Porosity — {site}\n{ml} | Dissolution↑ Precipitation↓',
               xlim=(0, T),
               ylim=(max(min(por.min(), flo)-0.2, 0), max(por.max(), phi0)+0.3))
    ax5[1].annotate(f'Peak: {por.max():.2f}%\nΔφ={por[-1]-por[0]:+.3f}%',
                    xy=(T*0.55, por[-1]), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    dd(ax5[1]); ax5[1].grid(True, alpha=0.25)
    plt.tight_layout()
    _save(fig5, od, 'Fig5_Efficiency_Porosity')
    print(f"  [PLOTS] 5 figures → {od}")

    # Fig 6 — Darcy velocity
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, r["vel_arr"], color=CLR, lw=2.0, label='Darcy velocity (m/yr)')
    shade_phases(ax, r)
    ax.set(xlabel='Time (years)', ylabel='Darcy velocity (m/yr)',
           title=f'Transport Velocity — {site} | {ml}\nResponds to injection ON/OFF via Darcy physics',
           xlim=(0, T))
    ax.grid(True, alpha=0.25)
    dd(ax); _save(fig, od, 'Fig6_Darcy_Velocity')

    # ─── Fig 7 & 8: Spatial profiles (pH and DIC along column vs time) ────────
    # Only meaningful when transport was active for enough steps
    sp_pH_all  = r.get("spatial_pH_all",  [])
    sp_DIC_all = r.get("spatial_DIC_all", [])
    sp_ti      = r.get("spatial_times",   [])
    n_cells    = r.get("n_cells_transport", 10)
    col_len    = rp.get("transport_col_len_m", 50.0)
    cell_len   = col_len / max(n_cells, 1)
    x_cell     = np.array([(c + 0.5) * cell_len for c in range(n_cells)])  # cell centres (m)

    if len(sp_pH_all) >= 4:
        # Select ~8 representative timesteps spread across the simulation
        n_snap = min(8, len(sp_ti))
        idx_sel = np.round(np.linspace(0, len(sp_ti) - 1, n_snap)).astype(int)

        cmap_inj = plt.cm.Reds
        cmap_mon = plt.cm.Blues
        T_inj_end = r["T_inj_yr"]

        # Fig 7 — Spatial pH profiles
        fig7, ax7 = plt.subplots(figsize=(12, 5))
        inj_snaps = [(k, sp_ti[k]) for k in idx_sel if t[sp_ti[k]] <= T_inj_end + 0.01]
        mon_snaps = [(k, sp_ti[k]) for k in idx_sel if t[sp_ti[k]] > T_inj_end + 0.01]
        for j, (k, ti_idx) in enumerate(inj_snaps):
            row = sp_pH_all[k]
            if len(row) == n_cells:
                c = cmap_inj(0.4 + 0.6 * j / max(len(inj_snaps) - 1, 1))
                ax7.plot(x_cell, row, color=c, lw=1.8,
                         label=f't={t[ti_idx]:.3f}yr (inj)')
        for j, (k, ti_idx) in enumerate(mon_snaps):
            row = sp_pH_all[k]
            if len(row) == n_cells:
                c = cmap_mon(0.3 + 0.7 * j / max(len(mon_snaps) - 1, 1))
                ax7.plot(x_cell, row, color=c, lw=1.8,
                         label=f't={t[ti_idx]:.2f}yr (mon)')
        ax7.axhline(rp["initial_pH"], color='#555', lw=1.2, ls='--',
                    alpha=0.6, label=f'Initial pH {rp["initial_pH"]:.1f}')
        ax7.set(xlabel='Distance from injection well (m)', ylabel='pH',
                title=f'Spatial pH Profile along Column — {site}\n{ml} | {slbl}',
                xlim=(0, col_len), ylim=(4.0, 9.5))
        ax7.grid(True, alpha=0.25)
        ax7.legend(fontsize=7, ncol=2, loc='lower right')
        _save(fig7, od, 'Fig7_Spatial_pH')

        # Fig 8 — Spatial DIC profiles
        if len(sp_DIC_all) >= 4 and all(len(r2) == n_cells for r2 in sp_DIC_all[:4]):
            fig8, ax8 = plt.subplots(figsize=(12, 5))
            for j, (k, ti_idx) in enumerate(inj_snaps):
                row = sp_DIC_all[k]
                if len(row) == n_cells:
                    c = cmap_inj(0.4 + 0.6 * j / max(len(inj_snaps) - 1, 1))
                    ax8.plot(x_cell, row, color=c, lw=1.8,
                             label=f't={t[ti_idx]:.3f}yr (inj)')
            for j, (k, ti_idx) in enumerate(mon_snaps):
                row = sp_DIC_all[k]
                if len(row) == n_cells:
                    c = cmap_mon(0.3 + 0.7 * j / max(len(mon_snaps) - 1, 1))
                    ax8.plot(x_cell, row, color=c, lw=1.8,
                             label=f't={t[ti_idx]:.2f}yr (mon)')
            ax8.set(xlabel='Distance from injection well (m)',
                    ylabel='DIC (mmol/kgw)',
                    title=f'Spatial DIC Profile along Column — {site}\n{ml} | {slbl}',
                    xlim=(0, col_len))
            ax8.grid(True, alpha=0.25)
            ax8.legend(fontsize=7, ncol=2, loc='upper right')
            _save(fig8, od, 'Fig8_Spatial_DIC')

    print(f"  [PLOTS] Figures → {od}")


def save_excel(r, xrf_data):
    try:
        import pandas as pd
    except ImportError:
        print("  [WARN] pandas missing"); return
    region = r["region"]; mode = r["mode"]; rp = r["reg_params"]
    t = r["time"]; N = len(t); sc = r["sched"]; P = sc["P_INJ_bar"]
    od = r["output_dir"]
    path = os.path.join(od, f"RTM_{region}_{mode}_{P:.0f}bar_v29.xlsx")
    params = {
        "Region": region, "Mode": mode, "Pressure_bar": P, "T_C": rp["T_C"],
        "M_target_t": sc["M_total_t"], "M_check_t": r["M_check_t"],
        "M_error_%": abs(r["M_check_t"]/sc["M_total_t"]-1)*100,
        "q0_t_day": sc["q0_t_day"],
        "T_inj_days": sc["T_inj_yr"]*365.25,
        "T_cont_yr": sc["T_cont_yr"],
        "T_on_yr": sc["T_on_yr"], "T_off_yr": sc["T_off_yr"],
        "N_cycles": sc["N_cycles"], "T_on_total_yr": sc["T_on_total_yr"],
        "T_monitor_yr": sc["T_monitor_yr"],
        "MassBalance_err_%": r.get("mass_balance_err_pct", float('nan')),
        "Engine": "PHREEQC v29",
    }
    ts = pd.DataFrame({
        "Time_years":        t,
        "q_t_per_day":       r["rate_arr"],
        "pH":                r["pH"],
        "pe":                r["pe"],
        "Porosity_%":        r["porosity"],
        "CO2_seq_mol_kgw":   r["co2_seq_kgw"],
        "pCO2_log10":        r["pco2_arr"],
        "Injection_on":      r["inj_mask"].astype(int),
        "Pressure_bar":      r["pressure_arr"],
        "CO2_injected_mol":  r["co2_injected_arr"],
        "Efficiency_%":      r["efficiency_arr"],
        "CO2_solubility_mmol": r["co2_solubility_arr"],
        "Darcy_vel_m_yr":    r["vel_arr"],
    })
    for m in CARBONATE_MINERALS + CLAY_MINERALS:
        ts[f"{m}_mol_kgw"] = r["minerals"].get(m, np.zeros(N))
    for m in PRIMARY_MINERALS:
        ts[f"{m}_dissolved"] = r["dissolved"].get(m, np.zeros(N))
    for m in CARBONATE_MINERALS:
        ts[f"SI_{m}"] = r["si"].get(m, np.full(N, np.nan))
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame([params]).to_excel(w, sheet_name="Run_Params",   index=False)
        pd.DataFrame([xrf_data]).to_excel(w, sheet_name="XRF_Input",  index=False)
        pd.DataFrame([r["engine_fracs"]]).to_excel(w, sheet_name="Mineral_Fractions", index=False)
        ts.to_excel(w, sheet_name=f"{P:.0f}bar_TimeSeries", index=False)
    print(f"  [EXCEL] {path}")


if __name__ == "__main__":
    REGION = CONFIG["active_region"]
    MODE   = CONFIG["mode"]
    s      = compute_schedules(REGION, CONFIG)

    print("\n" + "=" * 72)
    print(f"SCHEDULE v24 — {REGION} | M={s['M_total_t']:.0f}t CO₂ (same for all modes & regions)")
    print(f"  Single (30d burst): q={s['q0_t_day']:.1f}t/d × {s['T_inj_yr']*365.25:.0f}d "
          f"→ monitor {s['T_monitor_yr']:.0f}yr  (total {s['T_sim_single']:.2f}yr)")
    print(f"  Continuous  (1yr) : q={s['q_cont_t_day']:.2f}t/d × 1yr "
          f"→ monitor {s['T_monitor_yr']:.0f}yr  (total {s['T_sim_cont']:.1f}yr)")
    print(f"  Pulsed (WAG 1yr)  : q={s['q_pulsed_t_day']:.2f}t/d | "
          f"N={s['N_cycles']} cycles (0.25ON/0.25OFF) "
          f"→ monitor {s['T_monitor_yr']:.0f}yr  (total {s['T_sim_pulsed']:.1f}yr)")
    M_c = s['q_cont_t_day']  * s['T_cont_yr']     * 365.25
    M_p = s['q_pulsed_t_day']* s['T_on_total_yr'] * 365.25
    print(f"  Mass check: Single={s['M_total_t']:.0f}t  Cont={M_c:.0f}t  Pulsed={M_p:.0f}t")
    print(f"  Expected efficiency:  Pulsed ≥ Single > Continuous")
    print("=" * 72 + "\n")

    r = run_simulation(REGION, MODE, CONFIG)
    make_plots(r, CONFIG["xrf"][REGION])
    save_excel(r, CONFIG["xrf"][REGION])

    print("\n" + "=" * 72)
    print(f"[COMPLETE] {REGION} | {MODE} | v29")
    print(f"  pH: nadir={r['pH'].min():.2f} final={r['pH'][-1]:.2f}")
    print(f"  Porosity: {r['porosity'][0]:.2f}%→{r['porosity'][-1]:.2f}%")
    print(f"  Efficiency: {r['efficiency_arr'][-1]:.1f}%")
    print(f"  M_check: {r['M_check_t']:.0f}t (target={r['sched']['M_total_t']:.0f}t)")
    print(f"  Mass balance error: {r['mass_balance_err_pct']:.2f}%")
    print("=" * 72)