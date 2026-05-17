import math          # needed for co2_solubility_mmol_kgw, pressure_at_time, darcy_velocity_m_yr
import numpy as np

# ===========================================================================
# VERSION HISTORY — v29 FIXES  (applied over v24 baseline)
# ===========================================================================
# FIX 1 — pw_Ca_mmol XRF-sensitive (phreeqc_engine.py initialize())
#   BEFORE: rp['pw_Ca_mmol'] = rp.get('pw_Ca_mmol_fixed', 1.0)  — hardcoded 1 mmol/kgw
#           for ALL rocks regardless of CaO content.
#   AFTER:  rp['pw_Ca_mmol'] = _pw(e['Ca'], 'porewater_factor_Ca', 2.0, 'Ca_max_mmol', 0.1)
#           CRBG (CaO=6.98%) → ~3.1 mmol/kgw; MHOW (CaO=9.87%) → ~4.4 mmol/kgw.
#   IMPACT: Calcite and Ankerite now form in all scenarios (previously only Siderite
#           formed in Single/Pulsed because [Ca2+] was too low for SI_Calcite > 0).
#           Ankerite (Ca-Fe mixed carbonate) is now the dominant trap, matching
#           Wallula Phase II observations (White et al. 2020, Science Advances).
#
# FIX 2 — pw_Al_mmol baseline and update scale raised
#   BEFORE: base=0.02 mmol/kgw, update_scale=0.01 (only 1% of dissolved Al in solution).
#   AFTER:  base=0.05 mmol/kgw, update_scale=0.05 (5% of dissolved Al in solution).
#           Capped at Al_max_mmol (0.15 CRBG / 0.20 MHOW) to prevent explosion at pH>8.
#   IMPACT: Clay minerals (Saponite-Mg, Clinochlore-14A, Kaolinite) are now genuinely
#           supersaturated. Previous low Al caused PHREEQC to oscillate between clay
#           phases based on tiny pH differences — a numerical artifact, not geochemistry.
#           Consistent clay assemblage across all three injection scenarios.
#
# FIX 3 — Ankerite seed, Muscovite K-gate, Saponite nucleation (_build_eq_phases)
#   3a. Ankerite seed raised from 0 to 1e-8 mol/kgw → PHREEQC can nucleate Ankerite
#       when SI > 0 (achievable once Fix 1 corrects Ca).
#   3b. Muscovite gated by pw_K_mmol: suppressed when K < 1.5 mmol/kgw (MHOW K2O=0.57%
#       → pw_K ~0.8 mmol/kgw). Muscovite is metamorphic/granitic; inappropriate for
#       low-K basalt. CRBG (K2O=1.96% → pw_K ~3-5 mmol/kgw) retains Muscovite.
#   3c. Saponite seed fixed to 1e-10 for all modes (was inherited inventory, which was
#       zero in Single mode step 1 → Saponite never nucleated in Single scenario).
#   3d. Signature of _build_eq_phases / _build_eq_phases_all_cells extended with
#       reg_params keyword for K-gate access.
#
# FIX 4 — CARB_CAP XRF-sensitive via CaMg scaler (run_rtm.py run_simulation)
#   BEFORE: CARB_CAP = carb_cap_base × mode_multiplier  (hardcoded, ignores composition)
#   AFTER:  CARB_CAP = carb_cap_base × mode_multiplier × CaMg_scaler
#           where CaMg_scaler = (Ca+Mg)_rock / (Ca+Mg)_reference, clamped [0.5, 2.0].
#           CRBG scaler ~1.0; MHOW scaler ~1.40.
#   IMPACT: MHOW produces ~40% more carbonate per CO2 step, matching higher
#           cation availability in Deccan Trap basalt. Results differ between
#           regions for the same injection schedule.
#
# ALSO FIXED: undefined variable m_cur_kgw in _build_carbonate_kinetics_block (line ~1193)
# ===========================================================================

try:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc
    PHREEQC_AVAILABLE = True
except ImportError:
    PHREEQC_AVAILABLE = False


def _phreeqc_error_count(iph):
    for attr in ('GetErrorCount', 'get_error_count', 'phc_error_count'):
        val = getattr(iph, attr, None)
        if val is None:
            continue
        if callable(val):
            try:
                return int(val())
            except Exception:
                pass
        else:
            try:
                return int(val)
            except Exception:
                pass
    try:
        err = iph.get_error_string()
        return 1 if (err and err.strip()) else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# WALLULA REFERENCE  [White et al. 2020]
# ---------------------------------------------------------------------------
WALLULA_CO2_KG      = 977_000.0
WALLULA_CO2_MOL     = WALLULA_CO2_KG / 44.01 * 1000.0
WALLULA_DURATION_YR = 25.0 / 365.25
WALLULA_RATE_MOL_YR = WALLULA_CO2_MOL / WALLULA_DURATION_YR
PRESSURE_RATE_SCALE = {3: 0.50, 5: 0.75, 8: 1.00, 10: 1.25}

# ---------------------------------------------------------------------------
# MOLECULAR WEIGHTS  (g/mol)
# ---------------------------------------------------------------------------
MW_OXIDE = {
    'SiO2': 60.08, 'TiO2': 79.87, 'Al2O3': 101.96, 'Fe2O3': 159.69,
    'MnO':  70.94, 'MgO':  40.30, 'CaO':   56.08,  'Na2O':  61.98,
    'K2O':  94.20, 'P2O5': 141.94,
}
MW_MINERAL = {
    'Diopside': 216.55, 'Anorthite': 278.21, 'Albite': 262.22,
    'Magnetite': 231.53, 'Ilmenite': 151.73,
    'BasaltGlass': 60.08,   # v20: treated as SiO₂-equivalent mol mass
    'Calcite': 100.09, 'Siderite': 115.86, 'Magnesite': 84.31,
    'Dolomite': 184.40, 'Ankerite': 215.94,
    'Saponite-Mg': 443.20, 'Clinochlore-14A': 555.80,
    'Kaolinite': 258.16, 'Muscovite': 398.31,
}

MOLAR_VOLUME_CM3 = {
    'Calcite': 36.93, 'Siderite': 29.38, 'Magnesite': 28.02,
    'Dolomite': 64.37, 'Ankerite': 67.26,
    'Kaolinite': 99.52, 'Saponite-Mg': 310.00,
    'Clinochlore-14A': 210.00, 'Muscovite': 140.71,
    'Diopside':   66.09, 'Anorthite':  100.79,
    'Albite':     100.07, 'Ilmenite':   31.69,
    'BasaltGlass': 27.27,  # v20: SiO₂-glass molar volume (cm³/mol)
}

CO2_PER_CARBONATE = {
    'Calcite': 1.0, 'Siderite': 1.0, 'Magnesite': 1.0,
    'Dolomite': 2.0, 'Ankerite': 2.0,
}

PRIMARY_MINERALS   = ['Diopside', 'Anorthite', 'Albite', 'Magnetite', 'Ilmenite',
                      'BasaltGlass']   # v20: glass added as fast-dissolving phase
CARBONATE_MINERALS = ['Calcite', 'Siderite', 'Magnesite', 'Dolomite', 'Ankerite']
CLAY_MINERALS      = ['Saponite-Mg', 'Clinochlore-14A', 'Kaolinite', 'Muscovite']
ALL_MINERALS       = PRIMARY_MINERALS + CARBONATE_MINERALS + CLAY_MINERALS
MINERAL_LABELS     = {m: m for m in ALL_MINERALS}

REF_MOLES = {'Ca': 0.1245, 'Mg': 0.0849, 'Fe': 0.1789, 'Al': 0.2769, 'Si': 0.9461}

# ---------------------------------------------------------------------------
# PK04 RATE PARAMETERS  [Palandri & Kharaka 2004]
# ---------------------------------------------------------------------------
_PK04_PARAMS = {
   
    'Diopside':  (-6.36,  8400.0, 0.710, -11.11, 16800.0, -15.69, 12500.0, 0.370, 0.98),
    'Anorthite': (-3.50,  8000.0, 1.410,  -9.12,  8000.0,   0.00,     0.0, 0.000, 0.91),
    'Albite':    (-10.16, 6800.0, 0.457, -12.56, 15100.0, -15.60, 12400.0, 0.572, 0.98),
    'Magnetite': (-8.59,  1800.0, 0.279, -10.78,  1800.0, -15.00,  1800.0, 0.300, 0.05),
    'Ilmenite':  (-8.20,  7200.0, 0.250, -10.60, 10000.0, -14.00, 15000.0, 0.300, 0.25),
}


_MAX_KINETIC_MOL_PER_KGW_PER_STEP = 1e-3

# ---------------------------------------------------------------------------
# pe RANGE
# ---------------------------------------------------------------------------
_PE_MIN, _PE_MAX = -4.0, 1.0
CONFIG_LOG_PCO2_BG = -3.5   # background log10(pCO2) before injection


def co2_solubility_mmol_kgw(T_C, P_bar):
    """
    CO₂ solubility in pure water (mmol/kgw) using Henry's law.

    Calibrated to Duan & Sun (2003) Table 2 key values:
      T=40°C, P=100 bar → 58 mmol/kgw
      T=45°C, P=88  bar → ~57 mmol/kgw  (basalt CCS reference)
      T=45°C, P=8   bar → ~5  mmol/kgw

    Formula: sol = K(T) × P_bar  [linear Henry's law]
    K(T) = K_ref × exp(Ea/R × (1/T_ref − 1/T))  [van't Hoff]

    Reference: Duan, Z. & Sun, R. (2003) Chem. Geol. 193:257-271, Table 2.
    """
    T_K   = T_C + 273.15
    T_ref = 313.15   # 40°C reference temperature (K)
    K_ref = 0.580    # mmol kgw⁻¹ bar⁻¹ at 40°C (calibrated to Duan & Sun Table 2)
    Ea_R  = 2300.0   # Ea/R (K) — van't Hoff temperature sensitivity
    K     = K_ref * math.exp(Ea_R * (1.0 / T_ref - 1.0 / T_K))
    return min(K * P_bar, 180.0)   # cap at 180 mmol/kgw (physical limit)

def pressure_at_time(t_yr, t_inj_start, t_inj_end,
                     P_inj, P_hydrostatic,
                     tau_p_buildup=0.03, tau_p_decay=0.5):
    """
    Compute reservoir pressure at time t_yr (years).

    During injection [t_inj_start, t_inj_end]:
        P(t) = P_hydrostatic + (P_inj - P_hydrostatic)
                              × (1 - exp(-(t - t_inj_start)/tau_p_buildup))

    After injection (t > t_inj_end):
        P(t) = P_hydrostatic + (P_peak - P_hydrostatic)
                              × exp(-(t - t_inj_end)/tau_p_decay)

    Parameters
    ----------
    P_inj         : target injection pressure (bar)
    P_hydrostatic : formation pressure at depth (bar), ≈ depth_m × 0.1
    tau_p_buildup : buildup time constant (yr), default 0.03 (~11 days)
    tau_p_decay   : decay time constant (yr), default 0.5 (~6 months)
    """
    if t_inj_start <= t_yr <= t_inj_end:
        dt_on = t_yr - t_inj_start
        return P_hydrostatic + (P_inj - P_hydrostatic) * (
            1.0 - math.exp(-dt_on / max(tau_p_buildup, 1e-9)))
    elif t_yr > t_inj_end:
        # Peak pressure at end of injection
        dt_inj = t_inj_end - t_inj_start
        P_peak = P_hydrostatic + (P_inj - P_hydrostatic) * (
            1.0 - math.exp(-dt_inj / max(tau_p_buildup, 1e-9)))
        dt_off = t_yr - t_inj_end
        return P_hydrostatic + (P_peak - P_hydrostatic) * math.exp(
            -dt_off / max(tau_p_decay, 1e-9))
    else:
        return P_hydrostatic

_MU_BRINE_PA_S = 6e-4       # dynamic viscosity of brine at ~45°C, Pa·s
_MD_TO_M2      = 9.869e-16  # millidarcy to m²
_SEC_PER_YR    = 365.25 * 86400.0

def darcy_velocity_m_yr(permeability_mD, delta_P_bar, column_length_m,
                        mu_Pa_s=_MU_BRINE_PA_S):
    """
    Compute Darcy velocity (m/yr) from permeability and pressure gradient.

    v = (k / μ) × (ΔP / L)

    Parameters
    ----------
    permeability_mD : formation permeability in millidarcy
    delta_P_bar     : pressure differential across column (bar = overpressure)
    column_length_m : length of modelled column (m)
    mu_Pa_s         : brine dynamic viscosity (Pa·s)

    Returns
    -------
    Darcy velocity in m/yr.
    """
    k     = permeability_mD * _MD_TO_M2          # m²
    dP    = delta_P_bar * 1e5                     # bar → Pa
    v_mps = (k / mu_Pa_s) * (dP / column_length_m)  # m/s
    return v_mps * _SEC_PER_YR                    # m/yr


_CARB_KIN_PARAMS = {
    # mineral: (log_k_acid, Ea_acid, n_acid, log_k_neutral, Ea_neutral, A_m2_per_mol)
    # Calcite:  very fast at low pH, moderate at neutral
    'Calcite':  (-0.30, 14400.0, 1.00, -5.81, 23500.0, 8.8),
    # Siderite: slower than calcite, Fe-bearing
    'Siderite': (-3.19,  62800.0, 0.50, -8.90, 58900.0, 5.0),
    # Ankerite: Ca-Fe mixed carbonate, intermediate rate
    'Ankerite': (-3.50,  60000.0, 0.50, -9.50, 60000.0, 5.0),
    # Dolomite: notably slow kinetics (classic "dolomite problem")
    'Dolomite': (-3.19,  36100.0, 0.50, -7.53, 52200.0, 4.5),
    # Magnesite: also kinetically inhibited at low T
    'Magnesite':(-6.38,  14400.0, 1.00, -9.34, 23500.0, 3.5),
}


_GLASS_KIN_PARAMS = {
    'log_k_acid': -6.5,
    'Ea_acid':    56000.0,
    'n_acid':      0.47,
    'log_k_neutral': -10.5,
    'Ea_neutral': 56000.0,
    'Ca_frac':  0.12,
    'Mg_frac':  0.08,
    'Fe_frac':  0.11,
    'Al_frac':  0.25,
    # A_m2_per_mol: field-scale effective reactive surface area.
    # Lab BET value ~15 m²/mol, but field-scale is 10-300x lower due to
    # pore inaccessibility (Wolff-Boenisch 2006; Gudbrandsson 2011).
    # A=1.0 gives ~4% glass dissolution during 1-yr injection at pH5 — realistic.
    'A_m2_per_mol': 1.0,
}

GLASS_MW = 60.08  # treat glass as SiO₂-equivalent for molar mass (g/mol)


# ---------------------------------------------------------------------------
# XRF -> MINERAL FRACTIONS
# ---------------------------------------------------------------------------
def xrf_to_chemistry(xrf_data):
    om  = {ox: xrf_data.get(ox, 0.0) / MW_OXIDE[ox] for ox in MW_OXIDE}
    Ca  = om['CaO'];  Mg = om['MgO']
    Fe3 = om['Fe2O3'] * 2.0
    FeO = xrf_data.get('FeO', 0.0) / 79.845
    Fe2 = FeO if FeO > 1e-6 else Fe3 * 0.45
    Fe  = Fe3 + Fe2
    Al  = om['Al2O3'] * 2.0;  Si = om['SiO2']
    Ti  = om['TiO2'];  Na = om['Na2O'] * 2.0;  K = om['K2O'] * 2.0
    Fe_Mg = Fe / (Fe + Mg + 1e-12)

    mineral_moles = {
        'Diopside':  min(Ca*0.50, Mg*0.30, Si*0.25) * 0.80,
        'Anorthite': min(Ca*0.70, Al*0.40, Si*0.20) * 0.75,
        'Albite':    min(Na*0.80, Al*0.30, Si*0.15) * 0.70,
        'Magnetite': min(Fe3*0.35, Fe2*0.20) * 0.75,
        'Ilmenite':  Ti * 0.90,
    }

    
    Si_in_crystals = (
        mineral_moles['Diopside']  * 2.0   # CaMgSi₂O₆ → 2 Si
        + mineral_moles['Anorthite'] * 2.0  # CaAl₂Si₂O₈ → 2 Si
        + mineral_moles['Albite']    * 3.0  # NaAlSi₃O₈ → 3 Si
    )
    Si_remaining = max(Si - Si_in_crystals, 0.0)
    # The remaining Si is distributed as glass (SiO₂-equivalent moles)
    # We use a conservative 50% of excess Si → glass to avoid over-estimation
    mineral_moles['BasaltGlass'] = Si_remaining * 0.50

    mineral_wt    = {m: mol * MW_MINERAL[m] for m, mol in mineral_moles.items()}
    total_wt      = sum(mineral_wt.values()) + 1e-12
    mineral_fracs = {m: wt / total_wt for m, wt in mineral_wt.items()}

    scalers = {
        'Ca':   Ca / REF_MOLES['Ca'],   'Mg':   Mg / REF_MOLES['Mg'],
        'Fe':   Fe / REF_MOLES['Fe'],   'Al':   Al / REF_MOLES['Al'],
        'Si':   Si / REF_MOLES['Si'],
        'CaMg': (Ca+Mg)/(REF_MOLES['Ca']+REF_MOLES['Mg']),
        'buffer':(Ca+Mg)/(REF_MOLES['Ca']+REF_MOLES['Mg']),
        'Ankerite':(Ca*Fe)/(REF_MOLES['Ca']*REF_MOLES['Fe']),
        'Fe_Mg_ratio': Fe_Mg,
    }
    elem = {'Ca':Ca,'Mg':Mg,'Fe':Fe,'Fe2':Fe2,'Fe3':Fe3,
            'Al':Al,'Si':Si,'Ti':Ti,'Na':Na,'K':K}
    return om, elem, scalers, mineral_fracs, Fe_Mg


def _pe_from_xrf(Fe2_mol, Fe3_mol, T_C):
    """Initial pe from Fe2/Fe3 ratio."""
    if Fe3_mol < 1e-12 or Fe2_mol < 1e-12:
        return -1.5
    T_K    = T_C + 273.15
    pe_std = 0.771 * 96485.0 / (8.314 * T_K * np.log(10))
    pe_raw = pe_std + np.log10(Fe3_mol / Fe2_mol)
    return float(np.clip(pe_raw, _PE_MIN, _PE_MAX))


# ---------------------------------------------------------------------------
# ANKERITE -- user-defined phase (fallback if not in DB)
# ---------------------------------------------------------------------------
_ANKERITE_PHASE = (
    "PHASES\n"
    "Ankerite\n"
    "    CaFe(CO3)2 = Ca+2 + Fe+2 + 2 CO3-2\n"
    "    log_k     -17.09\n"
    "    delta_h   -6.276  kJ/mol\n\n"
)


# ---------------------------------------------------------------------------
# CLINOCHLORE-14A -- user-defined phase  (may not be in all llnl.dat versions)
# ---------------------------------------------------------------------------
# Clinochlore (Mg5Al)(AlSi3)O10(OH)8
# logK from Vieillard (2000) / Bethke (2022)
# This allows PHREEQC to find and form Clinochlore even if the DB entry name
# differs slightly from the EQUILIBRIUM_PHASES block name.
# ---------------------------------------------------------------------------
_CLINOCHLORE_PHASE = (
    "PHASES\n"
    "Clinochlore-14A\n"
    "    Mg5Al2Si3O10(OH)8 + 16 H+ = 5 Mg+2 + 2 Al+3 + 3 SiO2 + 12 H2O\n"
    "    log_k     36.80\n"
    "    delta_h   -208.0  kJ/mol\n\n"
)


_BASALTGLASS_PHASE = (
    "PHASES\n"
    "BasaltGlass\n"
    "    SiO2 = SiO2\n"
    "    log_k     -2.90\n"
    "    delta_h   20.0  kJ/mol\n\n"
)


# ---------------------------------------------------------------------------
# RATES BLOCK  [PK04 TST kinetics]
# ---------------------------------------------------------------------------
def _build_rates_block(params, boost=1.0, cap=None, mineral_fracs=None):
    """
    RATES block with SA cap + dt-proportional dissolution cap.

    v21 FIX: mineral_fracs scales each mineral's effective surface area so
    XRF composition drives dissolution (CRBG ≠ MHOW). Without this fix all
    regions produce identical cation supply and identical carbonate formation.

    v21 FIX: fracture SA exponent changed from 0.6667 (sphere) to 0.5
    (Noiriel 2009 fracture model) for fractured basalt.

    SA_MAX = 5.0 mol/kgw: realistic BET surface area ceiling.
    """
    if mineral_fracs is None:
        mineral_fracs = {}
    if cap is None:
        cap = _MAX_KINETIC_MOL_PER_KGW_PER_STEP
    SA_MAX = 5.0
    lines = ["RATES\n\n"]
    for mineral, pk in params.items():
        lka, ea_a, na, lkn, ea_n, _lb, _eb, _nb, A = pk
        # v21: scale SA by XRF mineral weight fraction
        frac   = mineral_fracs.get(mineral, 1.0)
        frac   = max(frac, 0.01)   # floor to avoid zero rates
        A_eff  = A * boost * frac
        lines += [
            f"{mineral}\n-start\n",
            "  10 if M <= 0 then goto 200\n",
            "  20 T_K = TEMP + 273.15\n",
            f"  30 ka = 10^({lka:.5f}) * exp(-{ea_a:.2f} * (1/T_K - 1/298.15))\n",
            f"  40 kn = 10^({lkn:.5f}) * exp(-{ea_n:.2f} * (1/T_K - 1/298.15))\n",
            "  50 aH = ACT('H+')\n",
            f"  60 r = ka * aH^{abs(na):.4f} + kn\n",
            # v21: fracture exponent 0.5 (Noiriel 2009)
            f"  70 SA = {A_eff:.4f} * M0 * (M/M0)^0.5000\n",
            f"  72 if SA > {SA_MAX:.1f} then SA = {SA_MAX:.1f}\n",
            "  80 moles = SA * r * TIME\n",
            "  90 if moles > M then moles = M\n",
            f"  95 if moles > {cap:.6e} then moles = {cap:.6e}\n",
            "  100 save moles\n",
            "  200 end\n-end\n\n",
        ]
    return "".join(lines)

# FIX: Never cache _RATES_BLOCK without mineral_fracs — built fresh per-step in _run_reactive()
_RATES_BLOCK = None


# ---------------------------------------------------------------------------
# ── NEW v20: RATES block for basalt glass  [Wolff-Boenisch et al. 2006]
# ---------------------------------------------------------------------------
def _build_glass_rates_block(glass_params=None, boost=1.0, cap=None):
    """Build a PHREEQC RATES block for basalt glass dissolution."""
    gp  = glass_params or _GLASS_KIN_PARAMS
    lka = gp['log_k_acid']
    ea  = gp['Ea_acid']
    n   = gp['n_acid']
    lkn = gp['log_k_neutral']
    ean = gp['Ea_neutral']
    A   = gp['A_m2_per_mol'] * boost
    cap_val = cap if cap is not None else _MAX_KINETIC_MOL_PER_KGW_PER_STEP

    lines = [
        "BasaltGlass\n-start\n",
        "  10 if M <= 0 then goto 200\n",
        "  20 T_K = TEMP + 273.15\n",
        f"  30 ka = 10^({lka:.5f}) * exp(-{ea:.1f}/8.314 * (1/T_K - 1/298.15))\n",
        f"  40 kn = 10^({lkn:.5f}) * exp(-{ean:.1f}/8.314 * (1/T_K - 1/298.15))\n",
        "  50 aH  = ACT('H+')\n",
        f"  60 r   = ka * aH^{n:.4f} + kn\n",
        f"  70 SA  = {A:.4f} * M0 * (M/M0)^0.5000\n",
        f"  72 if SA > 2.0 then SA = 2.0\n",   # reduced ceiling: A=1.0 is already field-scale
        "  80 moles = SA * r * TIME\n",
        "  90 if moles > M then moles = M\n",
        f"  95 if moles > {cap_val:.6e} then moles = {cap_val:.6e}\n",
        "  100 save moles\n",
        "  200 end\n-end\n\n",
    ]
    return "RATES\n\n" + "".join(lines)


# ---------------------------------------------------------------------------
# ── NEW v20: RATES block for carbonate kinetics  [Plummer 1978 / PK04]
# ---------------------------------------------------------------------------
def _build_carbonate_rates_block(carb_kin_params=None, boost=1.0):
    """
    Build PHREEQC RATES blocks for kinetically-controlled carbonate minerals.

    IPhreeqC BASIC compatibility fix:
    ----------------------------------
    SI() and SR() are not reliably available in all IPhreeqC versions inside
    RATES blocks. EXP() with variable arguments may also fail on older builds.

    Instead we use the original Plummer (1978) pH-dependent rate law:
        r = k_acid * aH^n + k_neutral

    Precipitation is controlled by a pH threshold: carbonates precipitate when
    pH rises above the approximate saturation pH for each mineral. This avoids
    any need for SI(), SR(), or EXP() with variable arguments.

    Precipitation threshold pH values (at 25-50°C, typical CO2 sequestration):
        Calcite:  pH > 6.3   (Morse & Mackenzie 1990)
        Siderite: pH > 6.0   (Bruno et al. 1992)
        Ankerite: pH > 6.2   (Palandri & Kharaka 2004)
        Dolomite: pH > 6.5   (Arvidson & Mackenzie 1999)
        Magnesite: pH > 7.5  (Saldi et al. 2009) — kinetically inhibited

    The dissolution rate uses the acid mechanism (ka * aH^n + kn) which is
    suppressed automatically when pH is high (aH is small).

    The precipitation rate uses a slower constant kp (100x slower than neutral
    dissolution) that activates only above the threshold pH.

    Surface area:
        - Dissolution: fracture model SA = A * M0 * (M/M0)^0.5 (requires M > 0)
        - Precipitation: constant seed SA = A_seed (allows nucleation from M=0)

    PHREEQC BASIC functions used (all safe in all IPhreeqC versions):
        ACT('H+')  — hydrogen ion activity
        TEMP       — temperature in Celsius
        M, M0      — current and initial moles in KINETICS block
        TIME       — timestep in seconds
        All arithmetic: +, -, *, /, ^(with literals only)
    """
    params   = carb_kin_params or _CARB_KIN_PARAMS
    cap      = _MAX_KINETIC_MOL_PER_KGW_PER_STEP
    
    A_seed   = 1.0      # m²/mol: nucleation seed surface area (was 0.10)
    kp_factor = 0.05    # precipitation rate / neutral dissolution rate (was 0.01)
    # Precipitation pH thresholds — carbonate precipitates above these values
    pH_thresh = {
        'Calcite':   6.3,
        'Siderite':  6.0,
        'Ankerite':  6.2,
        'Dolomite':  6.5,
        'Magnesite': 7.5,
    }

    lines = ["RATES\n\n"]
    for mineral, (lka, ea_a, n_a, lkn, ea_n, A_m2) in params.items():
        A_eff  = A_m2 * boost
        pH_thr = pH_thresh.get(mineral, 6.5)
        # aH threshold = 10^(-pH_thr) — pre-computed as a literal constant
        aH_thr = 10.0 ** (-pH_thr)

        lines += [
            f"{mineral}\n-start\n",
            # Step 10: temperature-dependent rate constants (all literals — safe)
            "  10 T_K = TEMP + 273.15\n",
            f"  20 ka = 10^({lka:.5f}) * exp(-{ea_a:.1f}/8.314*(1/T_K-1/298.15))\n",
            f"  30 kn = 10^({lkn:.5f}) * exp(-{ea_n:.1f}/8.314*(1/T_K-1/298.15))\n",
            # Step 40: current H+ activity (safe, always works)
            "  40 aH = ACT('H+')\n",
            # Step 50: pH-based dissolution rate (Plummer 1978)
            f"  50 rdiss = ka * aH^{n_a:.4f} + kn\n",
            # Step 55: Dissolution branch — only if M > 0 AND solution is acidic
            # (aH > aH_thr means pH < pH_thr — undersaturated for this carbonate)
            f"  55 if aH <= {aH_thr:.2e} then goto 80\n",  # pH > threshold → skip dissolution
            "  56 if M <= 0 then goto 200\n",               # no mineral to dissolve
            # Surface area: fracture model (M > 0 guaranteed here)
            f"  60 sa = {A_eff:.4f} * M0 * (M/M0)^0.5000\n",
            "  62 if sa > 10.0 then sa = 10.0\n",
            "  65 moles = sa * rdiss * TIME\n",
            "  66 if moles > M then moles = M\n",
            f"  67 if moles > {cap:.6e} then moles = {cap:.6e}\n",
            "  68 save moles\n",
            "  69 goto 200\n",
            # Step 80: Precipitation branch — pH > threshold means supersaturated
            f"  80 if aH > {aH_thr:.2e} then goto 200\n",  # pH < threshold → skip precip
            # Precipitation rate: kp_factor * neutral rate
            f"  85 rprecip = {kp_factor:.4f} * kn\n",
            # Surface area: seed when M=0 (nucleation), fracture when M > 0
            f"  90 sap = {A_seed:.4f}\n",
            f"  91 if M > 0 then sap = {A_eff:.4f} * M0 * (M/M0)^0.5000\n",
            "  92 if sap > 10.0 then sap = 10.0\n",
            # Negative moles = precipitation (adds to solid phase in PHREEQC)
            "  95 moles = -sap * rprecip * TIME\n",
            f"  96 if moles < -{cap:.6e} then moles = -{cap:.6e}\n",
            "  100 save moles\n",
            "  200 end\n-end\n\n",
        ]
    return "".join(lines)


# ---------------------------------------------------------------------------
# ── NEW v20: SURFACE AREA EVOLUTION  [Noiriel et al. 2009 fracture model]
# ---------------------------------------------------------------------------
def compute_surface_area_factor(M_current, M0, geometry='fracture'):
    """
    Compute the dimensionless surface area factor SA/SA0 as minerals dissolve.

    Three geometry options:
      'sphere'   : SA ∝ (M/M0)^(2/3) — standard shrinking-core model
                   Valid for rounded grains in granular media.
      'fracture' : SA ∝ (M/M0)^0.5   — fracture-wall model
                   As rock dissolves from fracture surfaces, SA decreases
                   more slowly than in the grain model.
                   From Noiriel et al. (2009) JGR Solid Earth 114:B01203.
      'constant' : SA = SA0 — limiting case (infinite roughness creation)

    Basalt dominated by fracture flow should use 'fracture' or 'constant'.
    """
    ratio = M_current / max(M0, 1e-30)
    ratio = min(ratio, 1.0)
    if   geometry == 'sphere'  : return ratio ** (2.0 / 3.0)
    elif geometry == 'fracture': return ratio ** 0.5
    elif geometry == 'constant': return 1.0
    else: return ratio ** (2.0 / 3.0)  # default to sphere




# ---------------------------------------------------------------------------
# KINETICS BLOCK (per cell, scaled to dt)
# ---------------------------------------------------------------------------
def _build_kinetics_block(m_current_field, m0_field, dt_years, pore_kg,
                          n_cells=1, cell_idx=None):
    """
    Build KINETICS block for one or all cells.
    v20: BasaltGlass is now included as a primary kinetic phase alongside
    Diopside, Anorthite, etc.  Glass is treated identically to crystalline
    phases in the KINETICS block — it just has a different (faster) rate
    defined in its RATES block entry.
    """
    dt_sec = dt_years * 365.25 * 86400.0
    pk     = max(pore_kg, 1.0)
    dt_ref = 0.02
    cap_per_step = _MAX_KINETIC_MOL_PER_KGW_PER_STEP * (dt_years / dt_ref)
    cap_per_step = max(cap_per_step, 1e-8)

    cell_frac = 1.0 / max(n_cells, 1)

   
    _TARGET_SUBSTEP_SEC = 3600.0
    n_substeps = max(1, min(20, int(math.ceil(dt_sec / _TARGET_SUBSTEP_SEC))))

    def _build_for_cell(cnum):
        lines = [f"KINETICS {cnum}\n\n"]
        added = False
        for mineral in PRIMARY_MINERALS:   # now includes BasaltGlass
            m_cur = m_current_field.get(mineral, 0.0) * cell_frac
            m0    = m0_field.get(mineral, m_cur / cell_frac) * cell_frac
            if m_cur <= 0.0:
                continue
            lines += [
                f"{mineral}\n",
                f"    -m          {m_cur/pk:.8e}\n",
                f"    -m0         {max(m0, m_cur)/pk:.8e}\n",
                f"    -tol        1e-6\n\n",
            ]
            added = True
        if not added:
            return ""
        lines.append(f"-steps      {dt_sec:.4f}  in  {n_substeps}\n\n")
        return "".join(lines)

    if cell_idx is not None:
        return _build_for_cell(cell_idx)

    result = ""
    for c in range(1, n_cells + 1):
        result += _build_for_cell(c)
    return result


# ---------------------------------------------------------------------------
# EQUILIBRIUM PHASES BLOCK (per cell or broadcast to all cells)
# ---------------------------------------------------------------------------
def _build_eq_phases(saponite=True, current_eq_mol_kgw=None,
                     delay_calcite=False, log_pco2=None,
                     carb_cap=0.5, cell_num=1,
                     carbonate_kinetic=False,
                     co2g_supply=100.0,
                     skip_clays=False,
                     reg_params=None):
    """
    Build EQUILIBRIUM_PHASES block for one cell.

    FIX v27: carbonate_kinetic=False is now the default (EQ-phase approach).
    PHREEQC precipitates each carbonate until SI=0, driven by dissolved Ca/Mg/Fe.
    carb_cap sets the maximum amount that can precipitate per step (mol/kgw).

    With EQ approach at SI_Calcite=0.49, pH=7.5, T=50C:
      PHREEQC precipitates ~0.5-3 mmol Calcite/kgw per step (until SI→0)
      Over 5yr: accumulates 1-10 mmol/kgw — matching CarbFix/Wallula.

    CO2(g) remains as an equilibrium phase to set the pCO2 boundary condition.
    """
    inv      = current_eq_mol_kgw or {}
    pco2_val = log_pco2 if log_pco2 is not None else -3.5

    def _mol(m):  return max(inv.get(m, 0.0), 0.0)
    def _cap(m):  return _mol(m) + carb_cap   # current inventory + cap for growth

    s = f"EQUILIBRIUM_PHASES {cell_num}\n"
    s += f"    CO2(g)          {pco2_val:.4f}    {co2g_supply:.4f}\n"

    if not carbonate_kinetic:
        
        if not delay_calcite:
            s += f"    Calcite         0.0   {_cap('Calcite'):.8e}\n"
        s += f"    Siderite        0.0   {_cap('Siderite'):.8e}\n"
        # FIX 3a: Ankerite seed raised from effectively 0 to 1e-8 mol/kgw.
        # This gives PHREEQC nucleation surface area to form Ankerite when SI > 0.
        # Wallula pilot: ankerite was dominant solid trap within 2yr (White et al. 2020).
        # SI_Ankerite > 0 requires [Ca2+][Fe2+] > 10^(log_k + 2*pH - 2*pCO2) which
        # is achievable once pw_Ca is correctly XRF-scaled (Bug 1 fix).
        _ank_seed = max(_mol('Ankerite'), 1e-8)
        s += f"    Ankerite        0.0   {_ank_seed + carb_cap:.8e}\n"
        _dol_cap = _mol('Dolomite')  + carb_cap * 0.05
        _mag_cap = _mol('Magnesite') + carb_cap * 0.02
        s += f"    Dolomite        0.0   {_dol_cap:.8e}\n"
        s += f"    Magnesite       0.0   {_mag_cap:.8e}\n"


    if not skip_clays:
        if saponite:
            # FIX 3b: Saponite seed starts from max(prev_inventory, 1e-10) so it
            # can nucleate from zero in Single mode (which has no prior monitoring step
            # to accumulate inventory). Previously using prev inventory only meant
            # Single mode never seeded Saponite on step 1.
            seed = max(_mol('Saponite-Mg'), 1e-10)
            s += f"    Saponite-Mg     0.0   {seed:.8e}\n"
        seed_cl = max(_mol('Clinochlore-14A'), 1e-10)
        seed_ka = max(_mol('Kaolinite'), 1e-10)
        s += f"    Clinochlore-14A 0.0   {seed_cl:.8e}\n"
        s += f"    Kaolinite       0.0   {seed_ka:.8e}\n"
        # FIX 3c: K-availability gate on Muscovite.
        # Muscovite (KAl3Si3O10(OH)2) requires K+. CRBG has K2O=1.96% so
        # pw_K_mmol ~3-5 mmol/kgw — sufficient for Muscovite.
        # MHOW has K2O=0.57% → pw_K_mmol ~0.5-1 mmol/kgw — marginal.
        # Gate: suppress Muscovite if pw_K_mmol < 1.5 mmol/kgw (too K-poor).
        # This prevents spurious Muscovite formation in K-poor rocks (MHOW)
        # and ensures it only forms where geochemically appropriate.
        _k_mmol = reg_params.get('pw_K_mmol', 1.0) if reg_params else 1.0
        if _k_mmol >= 1.5:
            seed_mu = max(_mol('Muscovite'), 1e-10)
            s += f"    Muscovite       0.0   {seed_mu:.8e}\n"
        else:
            # K-poor rock — allow trace Muscovite only (effectively suppressed)
            s += f"    Muscovite       0.0   1.00000000e-12\n"
    s += "\n"
    return s


def _build_eq_phases_all_cells(saponite=True, current_eq_mol_kgw=None,
                                delay_calcite=False, log_pco2_cells=None,
                                carb_cap=0.5, n_cells=1,
                                inlet_log_pco2=None,
                                carbonate_kinetic=False,
                                co2g_supply=100.0,
                                skip_clays=False,
                                reg_params=None):
    """
    Build EQUILIBRIUM_PHASES for all N cells.
    v20: carbonate_kinetic flag forwarded to _build_eq_phases per cell.
    v25 FIX 2: co2g_supply forwarded per cell to prevent Newton overshoot.
    v25 FIX 4: skip_clays forwarded per cell for high-pCO2 transport steps.
    v29 FIX: reg_params forwarded for K-availability gate on Muscovite.
    """
    result = ""
    for c in range(1, n_cells + 1):
        if log_pco2_cells is not None and inlet_log_pco2 is not None:
            frac = 1.0 - 0.5 * (c - 1) / max(n_cells - 1, 1)
            bg   = -3.5
            lp   = bg + (inlet_log_pco2 - bg) * frac
        elif log_pco2_cells is not None:
            lp = log_pco2_cells
        else:
            lp = -3.5

        result += _build_eq_phases(
            saponite=saponite,
            current_eq_mol_kgw=current_eq_mol_kgw,
            delay_calcite=delay_calcite,
            log_pco2=lp,
            carb_cap=carb_cap,
            cell_num=c,
            carbonate_kinetic=carbonate_kinetic,
            co2g_supply=co2g_supply,
            skip_clays=skip_clays,
            reg_params=reg_params,
        )
    return result


# ---------------------------------------------------------------------------
# SOLUTION BLOCKS — initial formation water (one per cell)
# ---------------------------------------------------------------------------
def _build_solution(reg_params, pe_val, cell_num=1):
    """Build SOLUTION block for one cell.

    FIX 1 (v26): Added dissolved Al3+ (pw_Al_mmol) — required for clay mineral
    supersaturation. Without Al in the SOLUTION block, PHREEQC computes SI << 0
    for all clay phases (Kaolinite, Clinochlore-14A, Saponite-Mg, Muscovite) and
    no clay mineral ever precipitates regardless of pH or silica content.
    Literature: Gysi & Stefansson (2012) Chem Geol 306: 0.01-0.05 mmol/kgw.
    """
    T_C     = reg_params['T_C']
    Ca      = reg_params.get('pw_Ca_mmol',   1.0)
    Mg      = reg_params.get('pw_Mg_mmol',   0.5)
    Fe2     = reg_params.get('pw_Fe2_mmol',  0.5)
    Na      = reg_params.get('pw_Na_mmol',  20.0)
    K       = reg_params.get('pw_K_mmol',    1.0)
    Si      = reg_params.get('pw_Si_mmol',   1.0)
    # FIX 1 (v26): Al3+ is essential for clay precipitation reactions in PHREEQC.
    # Default 0.02 mmol/kgw = mid-range basalt porewater value (Gysi & Stefansson 2012).
    Al      = reg_params.get('pw_Al_mmol',   0.02)
    Cl      = reg_params.get('Cl_mmol',     10.0)
    SO4     = reg_params.get('SO4_mmol',     0.5)
    DIC     = reg_params.get('initial_DIC_mmol', 0.0)
    # Cap initial DIC at 50 mmol/kgw — this is the *formation water* background.
    # CO2(g) in EQUILIBRIUM_PHASES will dissolve CO2 incrementally during the
    # reactive step.  Feeding the full Duan & Sun solubility (~2935 mmol) as C(4)
    # here causes a >50 000× jump that the PHREEQC Newton solver cannot converge.
    DIC_total = min(max(DIC, 2.0), 50.0)
    den     = reg_params.get('brine_density', 1.02)
    pH_init = reg_params.get('initial_pH',   7.5)
    pe_val  = float(np.clip(pe_val, _PE_MIN, _PE_MAX))

    return (
        f"SOLUTION {cell_num}\n"
        f"    temp      {T_C:.1f}\n"
        f"    pH        {pH_init:.2f}   charge\n"
        f"    pe        {pe_val:.3f}\n"
        "    units     mmol/kgw\n"
        f"    density   {den:.4f}\n"
        f"    Ca        {Ca:.4f}\n"
        f"    Mg        {Mg:.4f}\n"
        f"    Na        {Na:.4f}\n"
        f"    K         {K:.4f}\n"
        f"    Fe(2)     {Fe2:.5f}\n"
        f"    Si        {Si:.5f}\n"
        f"    Al        {Al:.5f}\n"
        f"    C(4)      {DIC_total:.3f}\n"
        f"    Cl        {Cl:.3f}\n"
        f"    S(6)      {SO4:.3f}\n\n"
    )


def _build_all_solutions(reg_params, pe_val, n_cells):
    """Build SOLUTION blocks for all N cells (identical initial chemistry)."""
    result = ""
    for c in range(1, n_cells + 1):
        result += _build_solution(reg_params, pe_val, cell_num=c)
    return result


# ---------------------------------------------------------------------------
# SELECTED OUTPUT BLOCK
# ---------------------------------------------------------------------------
def _build_selected_output(saponite=True, n_cells=1, carbonate_kinetic=False,
                            skip_primary_kinetics=False, skip_clays=False):
    """
    Build PHREEQC SELECTED_OUTPUT block.

    FIX v27: carbonate_kinetic=False is now default (EQ-phase approach).
    Added Kaolinite and Saponite-Mg to -saturation_indices so clay SI is tracked.
    """
    if skip_clays:
        eq_clays = ""
    else:
        eq_clays = "Kaolinite Muscovite Clinochlore-14A"
        if saponite:
            eq_clays += " Saponite-Mg"

    # Carbonate minerals: EQ tracking if equilibrium mode, kinetic if kinetic mode
    if carbonate_kinetic:
        eq_carbs = ""
        kin_carbs = " ".join(CARBONATE_MINERALS)
    else:
        eq_carbs = " ".join(CARBONATE_MINERALS)
        kin_carbs = ""

    kin_primary = " ".join(PRIMARY_MINERALS)

    sel = (
        "SELECTED_OUTPUT\n"
        "    -reset              true\n"
        "    -pH                 true\n"
        "    -alkalinity         true\n"
        "    -pe                 true\n"
        "    -totals             Ca Mg Fe(2) Fe(3) Al Si Na K C(4)\n"
    )
    _eq_list = " ".join(x for x in [eq_carbs, eq_clays] if x).strip()
    if _eq_list:
        sel += f"    -equilibrium_phases {_eq_list}\n"
    # FIX v27: added Kaolinite and Saponite-Mg SI to diagnose clay supersaturation
    sel += "    -saturation_indices Calcite Dolomite Siderite Magnesite Ankerite Kaolinite\n"
    if not skip_primary_kinetics:
        if kin_carbs:
            sel += f"    -kinetics           {kin_primary} {kin_carbs}\n"
        else:
            sel += f"    -kinetics           {kin_primary}\n"
    sel += "\n"
    return sel


# ---------------------------------------------------------------------------
# TRANSPORT BLOCK
# ---------------------------------------------------------------------------
def _build_transport_block(n_cells, dt_years, cell_length_m,
                           dispersivity_m, flow_vel_m_yr,
                           outlet_cell=None):
    """
    Build PHREEQC TRANSPORT block for 1-D advection-dispersion.

    Parameters
    ----------
    n_cells        : int   — number of spatial cells
    dt_years       : float — timestep in years
    cell_length_m  : float — length of each cell in metres
    dispersivity_m : float — longitudinal dispersivity in metres
                             (controls spread of the CO2 front)
    flow_vel_m_yr  : float — Darcy velocity in m/yr
                             Typical CRBG near-well: 5–20 m/yr (McGrail 2017)
    outlet_cell    : int   — cell to punch for SELECTED_OUTPUT (default: last)

    Physical notes
    ──────────────
    Peclet number per cell: Pe = cell_length / dispersivity
      Pe >> 1 : advection-dominated (sharp front, little numerical diffusion)
      Pe ~  1 : dispersive (smooth front)
    For basalt CO2: Pe ~ 5–20 is typical (sharp plume, short dispersion scale)

    The TRANSPORT block uses:
      -shifts 1          → advance the fluid ONE cell per call
      -time_step dt_sec  → duration of this shift
      -flow_direction forward   → left to right (cell 1 = inlet)
      -boundary_conditions flux flux  → open boundary (CO2 injected at inlet)
    """
    dt_sec     = dt_years * 365.25 * 86400.0
    punch_cell = outlet_cell if outlet_cell is not None else n_cells

    block  = "TRANSPORT\n"
    block += f"    -cells          {n_cells}\n"
    block += f"    -shifts         1\n"
    block += f"    -time_step      {dt_sec:.4f}\n"
    block += f"    -length         {cell_length_m:.6f}\n"
    block += f"    -dispersivity   {dispersivity_m:.6f}\n"
    block += f"    -flow_direction forward\n"
    block += f"    -boundary_conditions flux flux\n"
    block += f"    -punch_cells    1-{n_cells}\n"
    block += f"    -punch_frequency 1\n"
    block += "\n"
    return block


# ---------------------------------------------------------------------------
# ── NEW v20: REACTION block for explicit CO₂ mass injection
# ---------------------------------------------------------------------------
def _build_reaction_block(co2_mol_per_kgw, cell_num=1):
    """
    Build a PHREEQC REACTION block that explicitly adds a finite number of
    moles of CO₂ to the solution in one cell.

    This is the correct way to enforce mass conservation in PHREEQC:
    rather than setting pCO2 as a boundary condition alone (which is
    a thermodynamic constraint, not a mass-injection constraint), we add
    the computed CO₂ moles directly as a REACTION.  The CO2(g) equilibrium
    phase then re-equilibrates the speciation after the mass is added.

    Physical interpretation:
      co2_mol_per_kgw = total_CO2_injected_this_step / pore_volume_kgw
      This is the actual mass flux of CO₂ entering the porewater per
      unit porewater mass over the timestep dt.

    In PHREEQC syntax:
      REACTION {cell_num}
          CO2  1
          {moles}  moles    ← explicit moles added to 1 kgw of solution

    This block is placed AFTER KINETICS and BEFORE TRANSPORT so that
    CO₂ dissolves into the pore fluid before advection redistributes it.

    Reference: Parkhurst & Appelo (2013) USGS Techniques Ch. 6-A43, p. 78.
    """
    if co2_mol_per_kgw <= 0.0:
        return ""  # no injection this step — skip REACTION block entirely
    return (
        f"REACTION {cell_num}\n"
        f"    CO2  1\n"
        f"    {co2_mol_per_kgw:.8e}  moles\n\n"
    )


# ---------------------------------------------------------------------------
# POROSITY
# ---------------------------------------------------------------------------
def compute_porosity(minerals_dict, dissolved_dict, time_array, pressure,
                     scalers, reg_params, injection_active_mask=None,
                     volume_L_arr=None, m0_primary_field=None):
    """
    Compute porosity from molar volumes — physically correct version (v27).

    FIX 3 (v26): Pass m0_primary_field (actual initial field-scale mol from
    engine.m0_primary) so that the dissolution fraction f = cum/m0 is correct.

    ROOT CAUSE OF CLIFF BUG (original):
        m0_primary[m] = max(arr) estimated from running maximum of dissolved array.
        Early in the run arr is small → m0 is small → f ≈ 1.0 immediately →
        dv_diss hits the +5% ceiling in the first few steps → porosity plateaus
        at 17% for the entire run (flat line after initial jump).

    CORRECT APPROACH (v26 fix):
        f_diss = cum_dissolved / m0_primary_field  where m0 comes from XRF-derived
        initial mineral inventories (engine.m0_primary). This gives f in [0, 1]
        that grows slowly over 5 years as minerals gradually dissolve.

    k(t) = k0 × (φ/φ0)³  [Kozeny-Carman]
    """
    ini          = float(reg_params['initial_porosity'])
    floor_pct    = float(reg_params.get('porosity_floor_pct', 4.0))
    rock_density = float(reg_params.get('rock_density_kg_m3', 2900.0))
    rock_kg_total= float(reg_params.get('reactive_rock_kg', 7.95e9))
    rock_vol_m3  = rock_kg_total / rock_density          # m³ of reactive rock
    pore_vol_m3  = rock_vol_m3 * ini / 100.0             # m³ initial pore space
    k0_mD        = float(reg_params.get('permeability_mD', 100.0))
    phi_ceil     = ini + 5.0   # absolute maximum porosity gain from dissolution
    n            = len(time_array)
    por          = np.full(n, ini)
    perm_arr     = np.full(n, k0_mD)

    # FIX 3 (v26): use actual initial inventories from engine.m0_primary if provided.
    # Fall back to max(arr) only when m0_primary_field is not supplied (legacy calls).
    m0_primary = {}
    for m in PRIMARY_MINERALS:
        arr = dissolved_dict.get(m)
        if arr is not None and len(arr) > 0:
            if m0_primary_field is not None and m in m0_primary_field:
                # CORRECT: use XRF-derived initial inventory
                m0_primary[m] = max(float(m0_primary_field[m]), 1.0)
            else:
                # LEGACY fallback: use max(arr) — biases f toward 1.0 too early
                m0_primary[m] = max(float(np.max(arr)), 1.0)

    # Maximum total dissolution-driven porosity gain = +5% absolute (m³)
    dv_diss_ceiling = pore_vol_m3 * 5.0 / 100.0
    pore_kg_local   = pore_vol_m3 * 1000.0

    for i in range(1, n):
        # ── Dissolution: fraction of m0 × mineral rock-volume, total-capped ───
        dv_diss_raw = 0.0
        for m in PRIMARY_MINERALS:
            arr = dissolved_dict.get(m)
            if arr is None:
                continue
            cum = float(arr[i])
            if cum <= 0.0:
                continue
            m0 = m0_primary.get(m, max(cum, 1.0))
            f  = min(cum / m0, 1.0)                               # fraction 0–1
            dv_diss_raw += f * m0 * MOLAR_VOLUME_CM3.get(m, 50.0) * 1e-6  # m³

        # Hard cap: total dissolution cannot exceed +5% phi in a single snapshot
        dv_diss = min(dv_diss_raw, dv_diss_ceiling)

        # ── Precipitation: only net NEW precipitate reduces porosity ──────────
        # minerals_dict[m][i] is in mol/kgw porewater.
        # Volume = mol/kgw × pore_kg × molar_volume_m3 = field-scale m³.
        dv_precip = 0.0
        for m in CARBONATE_MINERALS + CLAY_MINERALS:
            arr = minerals_dict.get(m)
            if arr is None:
                continue
            dn_net = float(arr[i]) - float(arr[0])   # mol/kgw net new
            if dn_net > 0.0:
                # Convert mol/kgw → m³: × pore_kg [kg] × Vm [cm³/mol] × 1e-6 [m³/cm³]
                dv_precip += dn_net * pore_kg_local * MOLAR_VOLUME_CM3.get(m, 50.0) * 1e-6

        phi_new  = ini + (dv_diss - dv_precip) / max(pore_vol_m3, 1e-12) * 100.0
        por[i]   = float(np.clip(phi_new, floor_pct, phi_ceil))

        phi_ratio   = por[i] / ini
        perm_arr[i] = max(k0_mD * (phi_ratio ** 3.0), k0_mD * 0.01)

    return por, perm_arr


# ===========================================================================
# ENGINE CLASS  v18
# ===========================================================================
class PhreeqcEngine:
    """
    PHREEQC Reactive Transport Engine v18.

    NEW in v18: TRANSPORT block (1-D advection + dispersion)
    ─────────────────────────────────────────────────────────
    Set transport_params in reg_params or pass directly:
        'transport_n_cells'    : int   (default 10)
        'transport_col_len_m'  : float (default 50.0 m — near-well column)
        'transport_disp_m'     : float (default 0.5 m — typical basalt)
        'transport_vel_m_yr'   : float (default 10.0 m/yr — McGrail 2017)

    The simulation represents a 1-D column from the injection well outward.
    - Cell 1: nearest to the injection well (highest CO2 exposure)
    - Cell N: furthest from the well (most buffered by upstream reactions)
    - Reported values: outlet = last cell (most geochemically evolved fluid)

    Transport enriches pH curves with realistic spatial gradients:
    the acidic CO2 front advances step-by-step and interacts with fresh
    mineral surfaces in each cell before reaching the outlet.

    Retained from v17:
    - Unit fix in _pw (×10 conversion mol/100g → mmol/kgw)
    - Per-step carbonate cap (carb_cap) for smooth precipitation curves
    - Dynamic CARBONATE_DELAY_STEPS based on n_inj_steps
    - Pulsed mode delay reset per pulse
    - Saponite-Mg fallback if not in DB
    """
    CARBONATE_DELAY_STEPS = 5

    def __init__(self, database_path, xrf_data, reg_params):
        if not PHREEQC_AVAILABLE:
            raise RuntimeError("phreeqpy not found.\n  pip install phreeqpy")
        self.database_path = database_path
        self.reg_params    = reg_params
        self.iph           = None
        self._saponite_ok  = True

        # FIX 2 (v23): carbonate_kinetic now defaults to True so that Calcite
        # forms quickly (fast k_acid) while Mg/Fe carbonates are kinetically
        # inhibited. This is physically correct and enables pulsed > continuous
        # differentiation. The flag reads from reg_params with True as fallback.
        self._carbonate_kinetic = reg_params.get('carbonate_kinetic', True)

        (self.oxide_moles, self.elem, self.scalers,
         self.mineral_fracs, self.Fe_Mg_ratio) = xrf_to_chemistry(xrf_data)

        self._pe0        = _pe_from_xrf(self.elem['Fe2'], self.elem['Fe3'],
                                         reg_params['T_C'])
        self._pe_current = self._pe0

        self.cum_precip        = {m: 0.0 for m in CARBONATE_MINERALS + CLAY_MINERALS}
        self.cum_dissolved     = {m: 0.0 for m in PRIMARY_MINERALS}
        self._eq_inv_prev      = {}  # clay rate limiter state
        self.m0_primary        = {}
        self.m_primary_current = {}
        self._m_kin_prev       = {}
        self._eq_inv_kgw       = {}
        self._step             = 0
        self._total_co2        = 0.0     # cumulative CO₂ injected (mol/kgw)
        self._t_monitor_yr     = 0.0
        self._last_pH          = 7.5
        self._pressure_bar     = 8
        self._dic_budget_mol   = 0.0
        self._initial_dic_mol  = 0.0
        self._co2_step_kgw     = 0.0
        self._spatial_pH       = []      # v20: pH profile across cells
        self._porosity_current = reg_params.get('initial_porosity', 10.0)  # % — for K-C coupling

        self._n_cells       = None
        self._col_len_m     = None
        self._cell_len_m    = None
        self._disp_m        = None
        self._vel_m_yr      = None
        self._use_transport = True

    # -----------------------------------------------------------------------
    def initialize(self):
        rock_kg      = self.reg_params.get('reactive_rock_kg', 7.95e9)
        self.rock_kg = rock_kg

        self.m0_primary = {
            m: rock_kg * self.mineral_fracs.get(m, 0.0) / MW_MINERAL[m] * 1000.0
            for m in PRIMARY_MINERALS
        }
        self.m_primary_current = dict(self.m0_primary)
        self._m_kin_prev       = {}

        porosity     = self.reg_params.get('initial_porosity', 12.0) / 100.0
        self.pore_kg = (rock_kg / 2900.0) * porosity * 1000.0

        dic_mmol = self.reg_params.get('initial_DIC_mmol', 0.0)
        self._initial_dic_mol = dic_mmol * 1e-3 * self.pore_kg
        self._dic_budget_mol  = self._initial_dic_mol

        e  = self.elem
        rp = self.reg_params

        # ── Porewater concentrations ─────────────────────────────────────────
        def _pw(elem_mol_per_100g, factor_key, default_mmol, cap_key=None, floor_mmol=1e-6):
            """
            Derive porewater concentration from XRF-based element moles.
            elem_mol_per_100g × factor × 10 → mmol/kgw
            (The ×10 converts mol/100g to mmol/kgw assuming a typical
             rock:water ratio of ~1; factor then scales for enrichment.)
            """
            factor = rp.get(factor_key, 0.0)
            val    = elem_mol_per_100g * factor * 10.0 if factor > 0 else default_mmol
            val    = max(val, floor_mmol)
            if cap_key:
                val = min(val, rp.get(cap_key, val))
            return val

        # FIX 1: pw_Ca_mmol now scales with XRF CaO like every other element.
        # Previously hardcoded to 1.0 mmol/kgw regardless of rock composition —
        # this suppressed Calcite and Ankerite in CRBG while MHOW (CaO=9.87%)
        # should have significantly more dissolved Ca2+ than CRBG (CaO=6.98%).
        # pw_Ca_mmol_fixed can still override in region_params if needed.
        # Literature: McGrail et al. (2017) Table 2 — CRBG ~2–4 mmol/kgw Ca;
        # Sharma et al. (2014) Deccan Trap — 4–8 mmol/kgw Ca.
        if 'pw_Ca_mmol_fixed' in rp:
            rp['pw_Ca_mmol'] = rp['pw_Ca_mmol_fixed']
        else:
            rp['pw_Ca_mmol'] = _pw(e['Ca'], 'porewater_factor_Ca', 2.0, 'Ca_max_mmol', 0.1)
        rp['pw_Mg_mmol']  = _pw(e['Mg'],  'porewater_factor_Mg', 0.5, 'Mg_max_mmol', 0.01)
        rp['pw_Fe2_mmol'] = _pw(e['Fe2'], 'porewater_factor_Fe', 1.0, 'Fe2_max_mmol', 0.01)
        rp['pw_Na_mmol']  = _pw(e['Na'],  'porewater_factor_Na', 20.0,'Na_max_mmol',  0.5)
        rp['pw_K_mmol']   = _pw(e['K'],   'porewater_factor_K',   1.0, None,           0.01)
        rp['pw_Si_mmol']  = _pw(e['Si'],  'porewater_factor_Si',  1.0, 'Si_max_mmol',  0.1)

        # FIX 2: Raise baseline Al and update scale factor so clay minerals
        # (Kaolinite, Saponite-Mg, Clinochlore-14A) are genuinely supersaturated.
        # Old base 0.02 mmol/kgw and 1% update factor caused marginal SI → PHREEQC
        # oscillated between clay phases based on tiny pH differences (numerical
        # artifact, not geochemistry). Real CRBG porewater Al = 0.05–0.15 mmol/kgw
        # at pH 7–8 (Gysi & Stefansson 2012 Chem Geol 306; McGrail et al. 2017).
        rp['pw_Al_mmol_base'] = _pw(e['Al'], 'porewater_factor_Al', 0.05, 'Al_max_mmol', 0.001)
        rp['pw_Al_mmol']      = rp['pw_Al_mmol_base']  # updated each step via Al mass balance
        # Al stoichiometry per mol dissolved mineral:
        # Anorthite CaAl2Si2O8 → 2 Al3+; Albite NaAlSi3O8 → 1 Al3+
        # Si stoichiometry: Anorthite → 2 SiO2; Albite → 3 SiO2; Diopside → 2 SiO2
        self._al_per_mineral = {'Anorthite': 2.0, 'Albite': 1.0}
        self._si_per_mineral = {'Anorthite': 2.0, 'Albite': 3.0, 'Diopside': 2.0, 'BasaltGlass': 1.0}
        # Max dissolved porewater Al/Si (cap to prevent numerical explosion at pH 8+)
        # Kaolinite precipitation consumes Al and Si → they are self-limiting
        self._pw_Al_max_mmol = rp.get('Al_max_mmol', 5.0)   # up to 5 mmol/kgw
        self._pw_Si_max_mmol = rp.get('Si_max_mmol', 50.0)  # Si can be high

        if 'initial_DIC_mmol' not in rp:
            rp['initial_DIC_mmol'] = 0.0

        
        n_inj = rp.get('n_inj_steps', 50)
        if self._carbonate_kinetic:
            self.CARBONATE_DELAY_STEPS = 0   # kinetic rate law controls timing
        else:
            self.CARBONATE_DELAY_STEPS = 0   # EQ mode: SI decides timing, no artificial delay

        # ── Clay formation budget caps ────────────────────────────────────────
        self._max_kaolinite_mol = self.m0_primary.get('Anorthite', 1e12)
        self._max_saponite_mol  = self.m0_primary.get('Diopside', 1e12) / 3.0

     
        self._n_cells    = rp.get('transport_n_cells',  10)
        self._col_len_m  = rp.get('transport_col_len_m', 50.0)
        self._cell_len_m = self._col_len_m / self._n_cells
        self._disp_m     = rp.get('transport_disp_m',    1.0)
        self._vel_m_yr   = rp.get('transport_vel_m_yr',  10.0)
        self._use_transport = rp.get('use_transport', True)

        self.iph = IPhreeqc()
        self.iph.load_database(self.database_path)

        print(f"  [ENGINE v18] IPhreeqC loaded  | DB: {self.database_path}")
        print(f"  [ENGINE v18] pore_kg={self.pore_kg:.3e} kg | rock_kg={rock_kg:.3e} kg")
        print(f"  [ENGINE v18] pe0={self._pe0:.3f} | clamp [{_PE_MIN},{_PE_MAX}]")
        print(f"  [ENGINE v18] pw_Ca={rp['pw_Ca_mmol']:.2f} | "
              f"pw_Mg={rp['pw_Mg_mmol']:.2f} | pw_Fe2={rp['pw_Fe2_mmol']:.2f} | "
              f"pw_Al={rp['pw_Al_mmol']:.4f} mmol/kgw")
        print(f"  [ENGINE v18] Calcite delayed {self.CARBONATE_DELAY_STEPS} steps")
        if self._use_transport:
            print(f"  [ENGINE v18] TRANSPORT: {self._n_cells} cells | "
                  f"col={self._col_len_m:.0f} m | "
                  f"cell={self._cell_len_m:.1f} m | "
                  f"disp={self._disp_m:.2f} m | "
                  f"vel={self._vel_m_yr:.1f} m/yr | "
                  f"Pe={self._cell_len_m/self._disp_m:.1f}")
        else:
            print(f"  [ENGINE v18] TRANSPORT: disabled (batch mode)")
        return True

    def _build_carbonate_kinetics_block(self, dt_years, n_cells):
        """
        Build KINETICS blocks for carbonate minerals across all N cells.

        CRITICAL FIX v21:
        -----------------
        Carbonates start at M=0 (no pre-existing carbonate in fresh basalt).
        The KINETICS block MUST be included even when M=0 so that PHREEQC
        evaluates the RATES block, which now handles precipitation from solution.

        We always include all carbonates with M=0 and M0=seed_mol.
        This gives:
          - M=0  : no dissolution possible (correctly handled by RATES `if M <= 0`)
          - M0>0 : provides the reference inventory for SA calculation
          - When SR>1, the precipitation branch in RATES uses SA_seed directly
          - As precipitation accumulates, M grows and SA tracks it

        The seed value M0_seed = 1e-6 mol/kgw represents trace nucleation sites
        (dust, crystal defects, pre-existing micro-precipitates).
        """
        dt_sec    = dt_years * 365.25 * 86400.0
        pk        = max(self.pore_kg, 1.0)
        cell_frac = 1.0 / max(n_cells, 1)
        M0_seed   = 1e-6   # mol/kgw seed inventory for nucleation surface area
        result    = ""
        for c in range(1, n_cells + 1):
            lines = [f"KINETICS {c}\n\n"]
            for mineral in CARBONATE_MINERALS:
                m_cur = self._eq_inv_kgw.get(mineral, 0.0) * cell_frac
                m_cur_kgw = m_cur / pk
                m0_kgw    = max(m_cur_kgw, M0_seed)  # seed so SA_p is never NaN
                lines += [
                    f"{mineral}\n",
                    f"    -m          {m_cur_kgw:.8e}\n",
                    f"    -m0         {m0_kgw:.8e}\n",
                    f"    -tol        1e-6\n\n",
                ]
            # Sub-steps to prevent ODE stiffness hang (same logic as primary kinetics)
            _TARGET_SUBSTEP_SEC = 3600.0
            n_substeps = max(1, min(20, int(math.ceil(dt_sec / _TARGET_SUBSTEP_SEC))))
            lines.append(f"-steps      {dt_sec:.4f}  in  {n_substeps}\n\n")
            result += "".join(lines)
        return result

    # -----------------------------------------------------------------------
    def step(self, pressure_bar, co2_mol_this_step, dt_years,
             log_pco2_override=None, t_yr=0.0, wag_boost=1.0):
        """
        Execute one reactive transport timestep.

        v20 additions:
          - co2_mol_this_step is now non-zero during injection, enabling
            the REACTION block to add explicit CO₂ moles for mass conservation.
          - pressure_bar is the dynamically evolved reservoir pressure at t_yr,
            computed by pressure_at_time() in run_rtm.py.
          - Darcy velocity is recomputed each step from permeability and ΔP.
          - CO₂ solubility at current T and P is updated in reg_params so
            the inlet SOLUTION block uses a physically correct C(4) value.
          - Spatial pH profile (across all N cells) is stored in self._spatial_pH.
        """
        was_injection    = getattr(self, '_prev_is_injection', None)
      
        is_now_injection = (co2_mol_this_step > 0)
        if is_now_injection and (was_injection is False):
            self._step_for_delay = 0
        self._prev_is_injection = is_now_injection

        self._step         += 1
        self._total_co2    += co2_mol_this_step
        self._pressure_bar  = pressure_bar

        # ── v20: Update CO₂ solubility at current T, absolute P ─────────────
        T_C    = self.reg_params['T_C']
        P_over = self.reg_params.get('injection_pressure_bar', 8.0)
        depth  = self.reg_params.get('injection_depth_m', 860.0)
        P_abs  = self.reg_params.get('injection_pressure_abs_bar',
                                     depth * 0.1 + P_over)
       
        sol_mmol = co2_solubility_mmol_kgw(T_C, pressure_bar)  # use dynamic pressure
        sol_mmol *= max(wag_boost, 1.0)                          # WAG boost applies
        # Cap inlet C(4) at 50 mmol for PHREEQC Newton stability
        sol_mmol_capped = min(sol_mmol, 50.0)
        self.reg_params['co2_solubility_mmol'] = sol_mmol_capped if is_now_injection else 2.0 * max(wag_boost, 1.0)

        # Dissolved CO2 this step = min(injected, what can dissolve in pore volume)
        co2_max_dissolve = sol_mmol_capped * 1e-3 * max(self.pore_kg, 1.0)  # mol total
        co2_dissolved_this_step = min(co2_mol_this_step, co2_max_dissolve)
        co2_excess_this_step    = co2_mol_this_step - co2_dissolved_this_step  # stays as free phase
        # Store excess for mass conservation check (accessible from run_rtm.py)
        self._co2_excess_mol   = getattr(self, '_co2_excess_mol', 0.0) + co2_excess_this_step
        # Only count dissolved CO2 against pore chemistry (pH, DIC, minerals)
        self._co2_step_kgw_dissolved = co2_dissolved_this_step / max(self.pore_kg, 1.0)

        # ── v20: Update Darcy velocity from permeability ──────────────────────
        # delta_P = wellhead overpressure (bar) — drives porewater flow.
        # injection_pressure_bar is the overpressure (P_INJ from CONFIG), NOT
        # absolute downhole pressure. We use it directly as the Darcy gradient.
        permeability_mD = self.reg_params.get('permeability_mD', None)
        if permeability_mD is not None:
            
            depth_m       = self.reg_params.get('injection_depth_m', 860.0)
            P_hydrostatic = depth_m * 0.1   # bar (fresh-water gradient ~0.1 bar/m)
            # delta_P is the driving overpressure for Darcy flow
            delta_P       = max(pressure_bar - P_hydrostatic, 0.1)  # bar
            # Kozeny-Carman: k = k0*(phi/phi0)^3 (porosity-permeability coupling)
            phi0      = self.reg_params.get('initial_porosity', 10.0) / 100.0
            phi_now   = self._porosity_current / 100.0
            phi_ratio = max(phi_now / phi0, 0.1)
            k_eff_mD  = max(permeability_mD * (phi_ratio ** 3), 0.001)
            col_len   = self.reg_params.get('transport_col_len_m', 50.0)
            v_new     = darcy_velocity_m_yr(k_eff_mD, delta_P, col_len)
            # Physical bounds: min 0.01 m/yr (near-stagnant), max 200 m/yr
            self._vel_m_yr = max(min(v_new, 200.0), 0.01)

        if self._step <= 3 or self._step % 50 == 0:
            phase = "INJ" if is_now_injection else "MON"
            print(f"  [v20 step {self._step:3d} | {phase} | P={pressure_bar:.1f}bar | "
                  f"v={self._vel_m_yr:.1f}m/yr | CO2sol={self.reg_params.get('co2_solubility_mmol',2):.1f}mmol | "
                  f"dt={dt_years:.4f}yr]", flush=True)

        self._co2_step_kgw  = self._co2_step_kgw_dissolved
        self._last_dt_yr    = dt_years
        self._last_log_pco2 = log_pco2_override if log_pco2_override is not None else -3.5

      
        _pore_kg = max(self.pore_kg, 1.0)
        _al_cum_mol = sum(
            self.cum_dissolved.get(m, 0.0) * self._al_per_mineral.get(m, 0.0)
            for m in self._al_per_mineral
        )  # field-scale mol Al released total
        _si_cum_mol = sum(
            self.cum_dissolved.get(m, 0.0) * self._si_per_mineral.get(m, 0.0)
            for m in self._si_per_mineral
        )  # field-scale mol Si released total
        # Subtract Al/Si consumed by clay precipitation already stored
        _al_in_clay = (
            self._eq_inv_kgw.get('Kaolinite',     0.0) * 2.0   # Al2Si2O5(OH)4: 2 Al
          + self._eq_inv_kgw.get('Saponite-Mg',   0.0) * 0.33  # Mg3Si4O10(OH)2·Al0.33: ~0.33 Al
          + self._eq_inv_kgw.get('Clinochlore-14A',0.0)* 2.0   # Mg5Al2Si3O10(OH)8: 2 Al
          + self._eq_inv_kgw.get('Muscovite',      0.0) * 3.0   # KAl3Si3O10(OH)2: 3 Al
        )  # mol/kgw consumed by clay
        _al_net_kgw = max(_al_cum_mol / _pore_kg - _al_in_clay, 0.0) * 1e3  # mmol/kgw
        _si_net_kgw = max(_si_cum_mol / _pore_kg, 0.0) * 1e3  # mmol/kgw (Si rarely limiting)
        # Cap dissolved Al: at pH > 6, Al solubility is controlled by gibbsite/kaolinite
        # Al(OH)3(am) solubility at pH 7.5, 50°C ≈ 1-3 mmol/kgw (Nordstrom & Ball 1986)
        _al_base = self.reg_params.get('pw_Al_mmol_base', 0.05)
        _al_updated = min(_al_base + _al_net_kgw * 0.05, self._pw_Al_max_mmol)
        # Scale factor 0.01: only 1% of dissolved Al stays in solution (most adsorbs
        # or is incorporated into secondary phases — conservative estimate for pH 7-8)
        self.reg_params['pw_Al_mmol'] = max(_al_updated, _al_base)
        # Update Si similarly — Si is less limiting but needed for Saponite/Clinochlore
        _si_base = self.reg_params.get('pw_Si_mmol', 1.0)
        _si_updated = min(_si_base + _si_net_kgw * 0.005, self._pw_Si_max_mmol)
        self.reg_params['pw_Si_mmol'] = max(_si_updated, _si_base)

        if self._step == 1:
            p = self._run_step0(dt_years, pressure_bar, log_pco2=log_pco2_override)
        else:
            p = self._run_reactive(dt_years=dt_years,
                                   pressure_bar=pressure_bar,
                                   log_pco2=log_pco2_override,
                                   co2_mol_per_kgw=self._co2_step_kgw)

        # pH sanity guard — allow acidic values during injection (CO₂-driven)
        # PHREEQC returns values as low as ~4 under high pCO2; do not clamp these.
        pH_raw = p.get('pH')
        if pH_raw is None or not (3.0 <= pH_raw <= 12.5) or p.get('error'):
            p['pH'] = self._last_pH
            p['error'] = False

        # pe clamp
        pe_raw = p.get('pe')
        if pe_raw is None or not (-8.0 <= pe_raw <= 3.0):
            p['pe'] = self._pe_current
        else:
            p['pe'] = float(np.clip(pe_raw, _PE_MIN, _PE_MAX))

        self._last_pH    = p['pH']
        self._pe_current = p['pe']

        if self._step <= 5:
            print(f"  [PARSE step {self._step}] "
                  f"pH={p.get('pH',0):.3f} | pe={p.get('pe',0):.2f} | "
                  f"d_Calcite={p.get('delta_Calcite',0):.4e} | "
                  f"d_Siderite={p.get('delta_Siderite',0):.4e}", flush=True)

        # ── INJECTION vs MONITORING phase flag ───────────────────────────────
        _is_inj_phase = is_now_injection

        # ── Update EQ_PHASES inventory — CLAYS ───────────────────────────────
        for m in CLAY_MINERALS:
            inv = p.get(f'eq_inv_{m}')
            if inv is not None:
                new_val = max(inv, 0.0)
                prev    = self._eq_inv_kgw.get(m, 0.0)
                if _is_inj_phase:
                    self._eq_inv_kgw[m] = min(new_val, prev)
                else:
                    self._eq_inv_kgw[m] = new_val

     
        if self._carbonate_kinetic:
            # Kinetic mode (legacy path)
            for m in CARBONATE_MINERALS:
                m_rem_kgw = p.get(f'kin_{m}_remaining')
                if m_rem_kgw is not None and m_rem_kgw > 1e-12:
                    new_val = max(float(m_rem_kgw), 0.0)
                else:
                    dm_kgw  = p.get(f'kin_{m}', 0.0) or 0.0
                    prev    = self._eq_inv_kgw.get(m, 0.0)
                    new_val = max(prev - dm_kgw, 0.0)
                prev = self._eq_inv_kgw.get(m, 0.0)
                if _is_inj_phase and new_val < prev * 0.50:
                    self._eq_inv_kgw[m] = prev * 0.80
                else:
                    self._eq_inv_kgw[m] = new_val
        else:
            # ── EQ-phase carbonate update (v27 complete rewrite) ─────────────
            #
            # ROOT CAUSES diagnosed from Excel timeseries data:
            #
            # BUG A — Pulsed: carbonate wipes to zero at start of each pulse.
            #   When pulse 2 starts, _is_inj_phase=True, PHREEQC returns new_val≈0
            #   (equilibrated at high pCO2). Old code: min(0, prev) → instant wipe.
            #   FIX: during injection, limit dissolution to 3%/step maximum.
            #
            # BUG B — Continuous: carbonate built in t=1–1.9yr fully dissolves by t=2.2yr.
            #   SI_Siderite goes deeply negative (-2 to -3) as porewater Fe2+ is consumed
            #   by precipitation faster than dissolution replenishes it. The 0.7×prev
            #   floor runs every step, so over 200 steps: 0.7^200 → 0.
            #   FIX: track peak inventory ever reached (_peak_inv). Apply a hard floor
            #   = peak × FLOOR_FRAC. Once carbonate crystallises in basalt it does not
            #   fully re-dissolve under mildly undersaturated conditions (kinetic trap).
            #   White et al. (2020) Wallula: ankerite stable for >2yr post-injection.
            #
            # FLOOR_FRAC = 0.30 means at most 70% of peak can dissolve.
            # This is conservative — real basalt carbonates typically dissolve <20%
            # over 5yr monitoring at neutral pH (Oelkers et al. 2008).
            # FLOOR_FRAC: once precipitated, carbonate cannot dissolve below this
            # fraction of its peak inventory. Physical basis: Wallula pilot (White 2020)
            # showed siderite/ankerite stable for >2yr post-injection at near-neutral pH.
            # 0.90 = at most 10% of peak can dissolve. Conservative and consistent with
            # published basalt CCS field observations.
            FLOOR_FRAC    = 0.90   # max fraction of peak that is preserved (was 0.30)
            INJ_DISS_RATE = 0.03   # max 3% of prev dissolves per injection step
            _slow_minerals = {'Dolomite', 'Magnesite'}

            # Initialise peak inventory tracker on first step
            if not hasattr(self, '_peak_inv_kgw'):
                self._peak_inv_kgw = {m: 0.0 for m in CARBONATE_MINERALS}

            for m in CARBONATE_MINERALS:
                inv_raw = p.get(f'eq_inv_{m}')
                if inv_raw is None:
                    continue
                new_val = max(float(inv_raw), 0.0)
                prev    = self._eq_inv_kgw.get(m, 0.0)
                peak    = self._peak_inv_kgw.get(m, 0.0)
                # Hard floor: carbonate cannot dissolve below FLOOR_FRAC × peak
                hard_floor = peak * FLOOR_FRAC

                if _is_inj_phase:
                    # FIX A: limit per-step dissolution during injection
                    # Allow at most INJ_DISS_RATE fraction to dissolve per step.
                    # Never allow new precipitation to spike (min with prev).
                    min_allowed = max(prev * (1.0 - INJ_DISS_RATE), hard_floor)
                    self._eq_inv_kgw[m] = max(min(new_val, prev), min_allowed)
                else:
                    # Monitoring/OFF window: SI-gated growth with hard floor
                    _cc  = self.reg_params.get('carb_cap_eff', 9e-5)
                    _dt  = getattr(self, '_last_dt_yr', 0.02)
                    _cap = _cc * (_dt / 0.02)
                    if m in _slow_minerals:
                        _seed = 1e-6
                        if prev < _seed:
                            candidate = min(new_val, _seed * 2.0)
                        else:
                            candidate = min(new_val, prev + _cap * 0.15)
                    else:
                        # Fast minerals: grow up to cap, dissolve down to floor
                        candidate = min(new_val, prev + _cap)
                    # FIX B: enforce hard floor so monitoring does not dissolve all carbonate
                    self._eq_inv_kgw[m] = max(candidate, hard_floor)

                # Update peak tracker
                self._peak_inv_kgw[m] = max(self._peak_inv_kgw.get(m, 0.0),
                                             self._eq_inv_kgw[m])


     
        pore_kg = self.pore_kg
        ACCESS  = self.reg_params.get('access_fraction', 0.25)
        T_C_now = self.reg_params.get('T_C', 45.0)

        # ── Fe-carbonate T-routing: applied ONCE at step 1 only ─────────────
        # Running every step fights PHREEQC thermodynamics and causes oscillations.
        # At CRBG conditions (pw_Ca=1mmol, pw_Fe2=0.5mmol, T=50°C),
        # SI_Ankerite stays at -4 to -6 — Ca×Fe product too low for Ankerite.
        # Siderite (requires only Fe2+, log_k=-10.89) is correctly dominant.
        # From step 2 onward, PHREEQC log_k values determine the split.
        if self._step == 1:
            af = float(np.clip((T_C_now - 25.0) / 35.0, 0.0, 1.0))
            sid = self._eq_inv_kgw.get('Siderite', 0.0)
            ank = self._eq_inv_kgw.get('Ankerite', 0.0)
            fe_tot = sid + ank
            if fe_tot > 1e-12:
                self._eq_inv_kgw['Siderite'] = max(fe_tot * (1.0 - af), 0.0)
                self._eq_inv_kgw['Ankerite'] = max(fe_tot * af, 0.0)

        # ── Clay + Carbonate rate smoother ──────────────────────────────────
        # Applied every step (both injection and monitoring).
        # Growth cap: 2.0x per step for fast minerals (was 5.0x — caused spikes).
        # Dissolution floor: 3% per step during injection, 5% during monitoring.
        # These rates are physically grounded in basalt carbonate dissolution
        # kinetics (Oelkers et al. 2008; Matter et al. 2016).
        if hasattr(self, '_eq_inv_prev'):
            for _cm in CLAY_MINERALS + CARBONATE_MINERALS:
                _prev = self._eq_inv_prev.get(_cm, 0.0)
                _curr = self._eq_inv_kgw.get(_cm, 0.0)
                if _prev > 1e-14:
                    # Growth cap: prevents single-step spikes
                    _max_mult = 1.20 if _cm in ('Dolomite', 'Magnesite') else 2.0
                    if _curr > _prev * _max_mult:
                        self._eq_inv_kgw[_cm] = _prev * _max_mult
                    # Dissolution floor: max drop per step
                    # During injection: 3% per step
                    # During monitoring: 1% per step (near-neutral pH, very slow)
                    _max_diss = 0.03 if _is_inj_phase else 0.01
                    if _curr < _prev * (1.0 - _max_diss):
                        self._eq_inv_kgw[_cm] = _prev * (1.0 - _max_diss)
        self._eq_inv_prev = dict(self._eq_inv_kgw)

        # Accumulate field-scale precipitation for all mineral groups
        for m in CARBONATE_MINERALS + CLAY_MINERALS:
            inv_kgw = max(self._eq_inv_kgw.get(m, 0.0), 0.0)
            self.cum_precip[m] = inv_kgw * self.pore_kg


     
        dt_ref_cap = 0.02
        _last_pco2   = getattr(self, '_last_log_pco2', -3.5)
        _last_dt     = getattr(self, '_last_dt_yr', dt_ref_cap)
        use_transport_now = (self._use_transport
                             and _last_pco2 <= -2.0
                             and _last_dt   <= 0.025)
        n_eff = self._n_cells if use_transport_now else 1

        any_dk_nonzero = any(p.get(f'kin_{m}', 0.0) > 0 for m in PRIMARY_MINERALS)

        for m in PRIMARY_MINERALS:
            dm_kgw = p.get(f'kin_{m}', 0.0) or 0.0   # dk_ value (per-step delta)
            if dm_kgw > 0:
                # Normal path: dk_ column present and nonzero
                dm_kgw_col = dm_kgw * n_eff
                diss = dm_kgw_col * self.pore_kg
                dt_yr = getattr(self, '_last_dt_yr', dt_ref_cap)
                cap_step = _MAX_KINETIC_MOL_PER_KGW_PER_STEP * (dt_yr / dt_ref_cap)
                cap_field = max(cap_step, 1e-8) * self.pore_kg * n_eff
                diss = min(diss, cap_field)
                self.m_primary_current[m] = max(
                    self.m_primary_current.get(m, 0.0) - diss, 0.0)
                self.cum_dissolved[m] = (
                    self.m0_primary.get(m, 0.0) - self.m_primary_current[m])
            elif not any_dk_nonzero and not is_now_injection:
                # v26 fallback: monitoring step with no dk_ columns (e.g. first
                # monitoring step after high-pCO2 injection with kinetics skipped).
                # Use k_remaining column to compute cumulative dissolved.
                m_rem_kgw = p.get(f'kin_{m}_remaining')
                if m_rem_kgw is not None and m_rem_kgw >= 0:
                    m0_field = self.m0_primary.get(m, 0.0)
                    m0_kgw   = m0_field / max(self.pore_kg, 1.0)
                    # GUARD: if k_remaining parses as near-zero (column absent/misread),
                    # it would imply full dissolution of m0 in one step — physically wrong.
                    # Only trust k_remaining if it is > 1% of m0_kgw (i.e. reasonable).
                    # If the column is genuinely near-zero, cap the step dissolution
                    # to the per-step kinetic maximum rather than the entire inventory.
                    if m_rem_kgw < m0_kgw * 0.01 and m0_kgw > 1e-6:
                        # Column is unreliable — apply a conservative per-step dissolution
                        dt_yr_fb  = getattr(self, '_last_dt_yr', 0.02)
                        step_cap  = _MAX_KINETIC_MOL_PER_KGW_PER_STEP * (dt_yr_fb / 0.02)
                        prev_cum  = self.cum_dissolved.get(m, 0.0)
                        new_cum   = min(prev_cum + step_cap * self.pore_kg,
                                        m0_field * 0.10)   # hard 10% ceiling
                        if new_cum > prev_cum:
                            self.cum_dissolved[m] = new_cum
                            self.m_primary_current[m] = max(m0_field - new_cum, 0.0)
                    else:
                        # k_remaining is credible — use it
                        m_rem_field = m_rem_kgw * self.pore_kg * n_eff
                        cum = max(m0_field - m_rem_field, 0.0)
                        # Additional guard: cap single-step jump at 10% of m0
                        prev_cum = self.cum_dissolved.get(m, 0.0)
                        cum = min(cum, prev_cum + m0_field * 0.10)
                        if cum > prev_cum:
                            self.cum_dissolved[m] = cum
                            self.m_primary_current[m] = max(m0_field - cum, 0.0)

        co2_seq = sum(self.cum_precip[m] * CO2_PER_CARBONATE[m]
                      for m in CARBONATE_MINERALS)

        res = {
            'pH':             p['pH'],
            'pe':             p.get('pe', self._pe_current),
            'alkalinity':     p.get('alkalinity', 2e-3),
            'co2_seq':        co2_seq,
            'co2_injected':   self._total_co2,     # v20: cumulative injected
            'co2_excess_mol': getattr(self, '_co2_excess_mol', 0.0),  # FIX 3: free-phase CO2
            'volume_L':       p.get('volume_L', 1.0),
            'source':         'phreeqc_v26_transport',
            'co2_solubility': self.reg_params.get('co2_solubility_mmol', 2.0),
            'pressure_bar':   pressure_bar,
            'vel_m_yr':       self._vel_m_yr,
            # v26: spatial profiles (populated only when transport is active)
            'spatial_pH':     p.get('spatial_pH', []),
            'spatial_DIC':    p.get('spatial_DIC', []),
        }
        for m in CARBONATE_MINERALS + CLAY_MINERALS:
            capped_kgw = self._eq_inv_kgw.get(m, 0.0)
            res[f'precip_{m}']  = capped_kgw
            res[f'eq_inv_{m}']  = capped_kgw
        for m in PRIMARY_MINERALS:
            res[f'dissolved_{m}'] = self.cum_dissolved.get(m, 0.0)
        for m in CARBONATE_MINERALS:
            res[f'SI_{m}'] = p.get(f'SI_{m}')

        # Update _porosity_current for Kozeny-Carman coupling next step.
        # Quick single-step estimate: dV_precip from carbonates + clays this step.
        MOLAR_VOL = {'Calcite':36.93,'Siderite':29.38,'Magnesite':28.02,
                     'Dolomite':64.34,'Ankerite':66.00,
                     'Kaolinite':99.52,'Muscovite':140.71,
                     'Clinochlore-14A':210.0,'Saponite-Mg':135.0}
        dv_precip = sum(self._eq_inv_kgw.get(m,0)*MOLAR_VOL.get(m,50)*1e-6
                       for m in CARBONATE_MINERALS + CLAY_MINERALS)  # m³/kgw
        pore_vol_m3_kgw = self.reg_params.get('initial_porosity',10)/100.0 * (
            self.pore_kg / self.reg_params.get('rock_density_kg_m3',2900) *
            (1 - self.reg_params.get('initial_porosity',10)/100.0)
        ) / max(self.pore_kg, 1.0)
        phi0 = self.reg_params.get('initial_porosity', 10.0)
        # Δφ = -dv_precip/pore_vol_per_kgw × 100
        pore_vol_per_kgw = phi0 / 100.0 / max(
            self.reg_params.get('rock_density_kg_m3',2900)/self.pore_kg, 1e-10)
        delta_phi = -dv_precip / max(pore_vol_per_kgw, 1e-10) * 100.0
        self._porosity_current = max(phi0 + delta_phi,
                                     self.reg_params.get('porosity_floor_pct',4.0))
        return res

    # -----------------------------------------------------------------------
    def _run_step0(self, dt_years, pressure_bar=8, log_pco2=None):
        """
        Step 0: equilibrate all cells to initial formation water,
        then run first reactive transport step.
        """
        n = self._n_cells

        # Build N solutions (all identical initial chemistry)
        all_solutions = _build_all_solutions(self.reg_params, self._pe0, n)

        # Solution 0 = the injected fluid (CO2-charged water at inlet).
        # Use moderate C(4)=2.0 mmol here — the CO2(g) EQUILIBRIUM_PHASE will
        # equilibrate dissolved CO2 to the correct value during the reactive step.
        # Do NOT pre-load the full Duan & Sun solubility (~1000s mmol) as C(4)
        # because that causes a huge residual that prevents Newton convergence.
        inlet_pco2 = log_pco2 if log_pco2 is not None else -3.5
        sol0 = (
            f"SOLUTION 0\n"
            f"    temp      {self.reg_params['T_C']:.1f}\n"
            f"    pH        5.5\n"
            f"    pe        {self._pe0:.3f}\n"
            "    units     mmol/kgw\n"
            f"    density   {self.reg_params.get('brine_density', 1.02):.4f}\n"
            f"    Ca        {self.reg_params.get('pw_Ca_mmol', 1.0):.4f}\n"
            f"    Mg        {self.reg_params.get('pw_Mg_mmol', 0.5):.4f}\n"
            f"    Na        {self.reg_params.get('pw_Na_mmol', 20.0):.4f}\n"
            f"    K         {self.reg_params.get('pw_K_mmol', 1.0):.4f}\n"
            f"    Fe(2)     {self.reg_params.get('pw_Fe2_mmol', 0.5):.5f}\n"
            f"    Si        {self.reg_params.get('pw_Si_mmol', 1.0):.5f}\n"
            f"    Al        {self.reg_params.get('pw_Al_mmol', 0.02):.5f}\n"
            f"    C(4)      2.000\n"
            f"    Cl        {self.reg_params.get('Cl_mmol', 10.0):.3f}\n"
            f"    S(6)      {self.reg_params.get('SO4_mmol', 0.5):.3f}\n\n"
        )

        # Initial equilibration (no kinetics, no transport).
        # Use generous KNOBS so the formation-water speciation converges cleanly
        # before any CO2 is introduced.
        init_eq = "EQUILIBRIUM_PHASES 1\n    Kaolinite   0.0   0.0\n\nSAVE SOLUTION 1\n\n"
        inp_A = (
            _ANKERITE_PHASE + _BASALTGLASS_PHASE + _CLINOCHLORE_PHASE
            + "KNOBS\n"
            + "    -iterations             800\n"
            + "    -convergence_tolerance  1e-7\n"
            + "    -step_size              10.0\n"
            + "    -pe_step_size           5.0\n\n"
            + sol0
            + all_solutions
            + init_eq
            + "END\n"
        )
        def _run_step0(phreeqc_inp):
            try:
                self.iph.run_string(phreeqc_inp)
                return _phreeqc_error_count(self.iph)
            except Exception:
                return 1

        n_err_A = _run_step0(inp_A)
        if n_err_A > 0:
            inp_A2 = (
                _ANKERITE_PHASE + _BASALTGLASS_PHASE + _CLINOCHLORE_PHASE
                + "KNOBS\n"
                + "    -iterations             1000\n"
                + "    -convergence_tolerance  1e-6\n"
                + "    -step_size              20.0\n"
                + "    -pe_step_size           10.0\n\n"
                + sol0
                + all_solutions
                + "SAVE SOLUTION 1\nEND\n"
            )
            n_err_A2 = _run_step0(inp_A2)
            if n_err_A2 > 0:
                raise RuntimeError("[Step 0] Formation water equilibration failed.\n"
                                   + (self.iph.get_error_string() if hasattr(self.iph, 'get_error_string') else ""))

        print("  [Step 0] Formation water equilibrated across all cells.", flush=True)
        return self._run_reactive(dt_years=dt_years,
                                  pressure_bar=pressure_bar,
                                  log_pco2=log_pco2)

    # -----------------------------------------------------------------------
    def _run_reactive(self, dt_years, pressure_bar=8,
                      log_pco2=None, _retry=False, co2_mol_per_kgw=0.0):
        """
        Single reactive transport step (v20 version).

        TRANSPORT MODE STRATEGY (v21 fix):
        -----------------------------------
        During injection (log_pco2 is not None and > -1.0):
            ALWAYS use batch mode. PHREEQC TRANSPORT with CO2(g) at pCO2 ~ 0.9 atm
            (8 bar injection) dissolves ~50 mmol/kgw CO2 per cell per shift.
            With 10 cells that is a 500 mmol total chemical change in one TRANSPORT
            call — Newton-Raphson ALWAYS diverges for multi-cell systems under such
            a large geochemical perturbation. Batch mode handles one cell at a time
            and converges reliably. Since the injection front is well-mixed at the
            near-well scale, batch is physically appropriate during injection.

        During monitoring (log_pco2 is None or pCO2 returning to background):
            Use TRANSPORT. The decaying pCO2 produces small per-step changes that
            the multi-cell Newton solver handles without divergence. This gives the
            spatially distributed pH recovery and mineral precipitation that makes
            transport meaningful.

        The WARN flood (hundreds of identical messages) is suppressed: failures
        are counted and reported as a summary every N steps instead.
        """
        is_injection   = (log_pco2 is not None)
        log_pco2_eff   = log_pco2 if log_pco2 is not None else -3.5

        _high_pco2    = log_pco2_eff > -2.0
        _large_dt     = dt_years > 0.025
        use_transport_this_step = (
            self._use_transport and not _high_pco2 and not _large_dt
        )

        _high_pco2_transport    = (log_pco2_eff > -1.5)
        # FIX v27: carbonate_kinetic=False → never use kinetic carbonate RATES
        _use_carb_kin_transport = False  # always EQ-phase for carbonates

        delay_step     = getattr(self, '_step_for_delay', self._step)
        if hasattr(self, '_step_for_delay'):
            self._step_for_delay += 1
        delay_calcite  = delay_step <= self.CARBONATE_DELAY_STEPS

        boost    = self.reg_params.get('dissolution_boost', 1.0)
        dt_ref   = 0.02
        cap_now  = _MAX_KINETIC_MOL_PER_KGW_PER_STEP * (dt_years / dt_ref)
        cap_now  = max(cap_now, 1e-8)
        # FIX v27 CALIBRATION: carb_cap_eff calibrated to literature.
        # Default 0.010 mol/kgw at dt_ref=0.02yr → ~1-2 mol/kgw total Calcite over 5yr.
        # Previous 0.5 mol/kgw caused 277 mol/kgw Calcite (51× more than CO2 injected).
        _cc_base = self.reg_params.get('carb_cap_eff', 0.010)
        carb_cap = _cc_base * (dt_years / dt_ref)
        carb_cap = max(carb_cap, 1e-6)

        rates = _build_rates_block(_PK04_PARAMS, boost=boost, cap=cap_now,
                                   mineral_fracs=self.mineral_fracs)

        if use_transport_this_step:
            # Transport mode: use pCO2-gated carbonate kinetics (FIX 4 v25)
            sel = _build_selected_output(
                self._saponite_ok,
                n_cells=self._n_cells,
                carbonate_kinetic=_use_carb_kin_transport,
                skip_primary_kinetics=False,
                skip_clays=_high_pco2_transport)
            inp = self._build_transport_input(
                dt_years, log_pco2_eff, delay_calcite, carb_cap, cap_now,
                rates, sel, is_injection,
                co2_mol_per_kgw=co2_mol_per_kgw,
                use_carb_kin=_use_carb_kin_transport,
                skip_clays=_high_pco2_transport)
        else:
            # Batch mode: carbonate kinetics disabled when pCO2 is high (> -1.5)
            # CRITICAL: sel must match the actual carbonate mode used in eq block.
            # Also: when _high_pco2_batch, primary KINETICS block is skipped entirely --
            # must pass skip_primary_kinetics=True or PHREEQC hangs looking for missing KINETICS.
            _carb_kin_batch = (self._carbonate_kinetic and log_pco2_eff <= -1.5)
            _skip_kin = (log_pco2_eff > -1.5)   # True when no KINETICS block present
            sel = _build_selected_output(
                self._saponite_ok,
                n_cells=1,
                carbonate_kinetic=_carb_kin_batch,
                skip_primary_kinetics=_skip_kin,
                skip_clays=_skip_kin)   # also skip clays from EQ list when high pCO2
            inp = self._build_batch_input(
                dt_years, log_pco2_eff, delay_calcite, carb_cap, cap_now,
                rates, sel,
                co2_mol_per_kgw=co2_mol_per_kgw)

        def _run(phreeqc_inp):
            try:
                self.iph.run_string(phreeqc_inp)
                return _phreeqc_error_count(self.iph)
            except Exception:
                return 1

        n_err = _run(inp)

        # ── Tier 1: Saponite not in DB ────────────────────────────────────────
        if n_err > 0 and self._saponite_ok:
            try:
                err_str = self.iph.get_error_string().lower()
            except Exception:
                err_str = ""
            if 'saponite' in err_str:
                print("  [WARN] Saponite-Mg not in DB -- disabled.", flush=True)
                self._saponite_ok = False
                return self._run_reactive(dt_years, pressure_bar, log_pco2,
                                          co2_mol_per_kgw=co2_mol_per_kgw)

        # ── Tier 2: Loosen tolerance ──────────────────────────────────────────
        if n_err > 0:
            inp2 = (inp
                    .replace("-convergence_tolerance  1e-7", "-convergence_tolerance  1e-5")
                    .replace("-convergence_tolerance 1e-7",  "-convergence_tolerance  1e-5")
                    .replace("-iterations             500",  "-iterations             1000")
                    .replace("-step_size              20.0", "-step_size              50.0"))
            n_err = _run(inp2)

        # ── Tier 3: Transport failed → fall back to batch silently ────────────
        # (only applies when transport was attempted for monitoring steps)
        if n_err > 0 and use_transport_this_step:
            # Track transport failures; report summary every 25 steps, not every step
            self._transport_fail_count = getattr(self, '_transport_fail_count', 0) + 1
            if self._transport_fail_count % 25 == 1:
                print(f"  [INFO] Transport fallback active "
                      f"(step {self._step}, {self._transport_fail_count} total falls)", flush=True)
            # FIX 1 (v25): rebuild sel for batch — the transport sel has wrong n_cells/kinetics
            # headers and causes PHREEQC to hang looking for missing -kinetics entries.
            _carb_kin_b3 = (self._carbonate_kinetic and log_pco2_eff <= -1.5)
            _skip_kin_b3 = (log_pco2_eff > -1.5)
            sel_batch_fb = _build_selected_output(
                self._saponite_ok, n_cells=1,
                carbonate_kinetic=_carb_kin_b3,
                skip_primary_kinetics=_skip_kin_b3,
                skip_clays=_skip_kin_b3)
            inp_batch = self._build_batch_input(
                dt_years, log_pco2_eff, delay_calcite, carb_cap, cap_now,
                rates, sel_batch_fb)
            inp_batch = (inp_batch
                         .replace("-convergence_tolerance 1e-8", "-convergence_tolerance 1e-5")
                         .replace("-iterations             800",  "-iterations             1000"))
            n_err = _run(inp_batch)

        if n_err > 0:
            print(f"  [WARN] Step {self._step}: all retries failed — "
                  "returning last good state.", flush=True)
            return {'pH': self._last_pH, 'pe': self._pe_current, 'error': True}

        if self._step == 1:
            try:
                dbg = self.iph.get_selected_output_array()
                if dbg and len(dbg) >= 2:
                    print(f"  [HEADERS]: {[str(h) for h in dbg[0]]}", flush=True)
                    print(f"  [DATA-1]:  {[str(v) for v in dbg[-1]]}", flush=True)
            except Exception as e:
                print(f"  [HDR DBG ERR]: {e}", flush=True)

        return self._parse_output()

    # -----------------------------------------------------------------------
    def _build_transport_input(self, dt_years, log_pco2_eff, delay_calcite,
                                carb_cap, cap_now, rates, sel, is_injection,
                                co2_mol_per_kgw=0.0,
                                use_carb_kin=None,
                                skip_clays=False):
        """
        Build full PHREEQC input with TRANSPORT + REACTION + carbonate KINETICS.

        v25 changes:
          FIX 4: use_carb_kin parameter gates carbonate kinetics — same high-pCO2
                 threshold as batch mode to prevent ODE stiffness hang in transport.
          FIX 2: co2g_supply limited to 2× solubility, preventing Newton overshoot
                 when 100 mol CO2(g) reservoir is supplied but only ~0.001 mol
                 can dissolve per step at high pCO2.
          FIX 1 (prior): sel already built with correct carbonate_kinetic flag.
        """
        n  = self._n_cells
        T  = self.reg_params['T_C']
        rp = self.reg_params

        # Default: honour flag passed from _run_reactive (FIX 4)
        if use_carb_kin is None:
            use_carb_kin = self._carbonate_kinetic

        # v21: inlet solution for TRANSPORT cell 0 (injected fluid)
        # During injection: slightly acidic with background DIC only.
        # CO2(g) in EQUILIBRIUM_PHASES drives CO2 dissolution incrementally —
        # do NOT pre-load high C(4) here as that causes Newton divergence.
        # During monitoring: native formation water.
        # During CO2 injection, inlet is acidic CO2-charged water.
        # The inlet_pH formula approximates CO2 equilibrium pH for the SOLUTION block.
        # This is essential: PHREEQC's "charge" keyword on initial pH adjusts alkalinity
        # to balance charge at 7.5, preventing the solution from acidifying.
        # Setting inlet_pH explicitly from pCO2 ensures the SOLUTION starts acidic.
        # The GRADUAL pH drop is controlled by tau_buildup=0.15yr in CONFIG.
        # During monitoring: native formation water.
        inlet_pH  = rp.get('initial_pH', 7.5)  # default for monitoring
        if is_injection:
            # Linear pH approximation from CO2 equilibrium:
            # At BG pCO2 (-3.5) pH=7.5; at full injection pCO2 (+0.9) pH~4.5.
            inlet_pH  = max(4.5, rp.get('initial_pH', 7.5)
                            + 0.65 * (CONFIG_LOG_PCO2_BG - log_pco2_eff))
            inlet_dic = max(2.0, co2_solubility_mmol_kgw(rp['T_C'],
                            rp.get('co2_solubility_mmol', 50.0)) * 0.03)
        else:
            inlet_dic = max(rp.get('initial_DIC_mmol', 0.0), 0.5)

        sol0 = (
            "SOLUTION 0\n"
            f"    temp      {T:.1f}\n"
            f"    pH        {inlet_pH:.2f}\n"
            f"    pe        {self._pe_current:.3f}\n"
            "    units     mmol/kgw\n"
            f"    density   {rp.get('brine_density', 1.02):.4f}\n"
            f"    Ca        {rp.get('pw_Ca_mmol', 1.0):.4f}\n"
            f"    Mg        {rp.get('pw_Mg_mmol', 0.5):.4f}\n"
            f"    Na        {rp.get('pw_Na_mmol', 20.0):.4f}\n"
            f"    K         {rp.get('pw_K_mmol', 1.0):.4f}\n"
            f"    Fe(2)     {rp.get('pw_Fe2_mmol', 0.5):.5f}\n"
            f"    Si        {rp.get('pw_Si_mmol', 1.0):.5f}\n"
            f"    Al        {rp.get('pw_Al_mmol', 0.02):.5f}\n"
            f"    C(4)      {inlet_dic:.4f}\n"
            f"    Cl        {rp.get('Cl_mmol', 10.0):.3f}\n"
            f"    S(6)      {rp.get('SO4_mmol', 0.5):.3f}\n"
        )

        # REACTION block disabled: the CO2(g) EQUILIBRIUM_PHASE already dissolves
        # CO₂ incrementally each step based on the pCO2 boundary condition set by
        # log_pco2_eff. Adding a REACTION block on top double-injected CO₂ and
        # caused Newton-Raphson divergence in cell 2 (too large a chemical change
        # in a single step). The CO₂ mass budget is tracked separately via
        # co2_injected_arr using the rate × dt accounting in run_rtm.py.
        reaction = ""

        kin = _build_kinetics_block(
            self.m_primary_current, self.m0_primary, dt_years, self.pore_kg,
            n_cells=n)

        # v20: carbonate kinetics block (new — replaces equilibrium carbonates)
        # FIX 4 (v25): gated by use_carb_kin — disabled at high pCO2 same as batch
        if use_carb_kin:
            boost = rp.get('dissolution_boost', 1.0)
            carb_rates = _build_carbonate_rates_block(boost=boost)
            carb_kin   = self._build_carbonate_kinetics_block(dt_years, n)
        else:
            carb_rates = ""
            carb_kin   = ""

        # FIX 2 (v25): limit CO2(g) reservoir to 2× solubility — prevents Newton
        # overshoot. Supplying 100 mol CO2(g) when only ~0.001 mol dissolves per step
        # allows the solver to overshoot into unphysical territory (same fix as batch).
        _sol_mmol_t   = co2_solubility_mmol_kgw(rp['T_C'], max(10**log_pco2_eff, 1.0))
        _co2g_supply_t = max(_sol_mmol_t * 1e-3 * 2.0, 0.05)   # mol, min 0.05

        eq = _build_eq_phases_all_cells(
            saponite=self._saponite_ok,
            current_eq_mol_kgw=self._eq_inv_kgw,
            delay_calcite=delay_calcite,
            log_pco2_cells=log_pco2_eff,
            carb_cap=carb_cap,
            n_cells=n,
            inlet_log_pco2=log_pco2_eff,
            carbonate_kinetic=use_carb_kin,     # FIX 4: gated flag
            co2g_supply=_co2g_supply_t,          # FIX 2: limited supply
            skip_clays=skip_clays,               # FIX 4: skip clays at high pCO2
            reg_params=self.reg_params,           # FIX 3c: K-gate for Muscovite
        )

        # v20: glass RATES added on top of primary mineral rates
        glass_rates = _build_glass_rates_block(
            boost=rp.get('dissolution_boost', 1.0),
            cap=cap_now)

        transport = _build_transport_block(
            n_cells=n,
            dt_years=dt_years,
            cell_length_m=self._cell_len_m,
            dispersivity_m=self._disp_m,
            flow_vel_m_yr=self._vel_m_yr,   # now dynamically updated
            outlet_cell=n,
        )

        return (
            _ANKERITE_PHASE + _BASALTGLASS_PHASE + _CLINOCHLORE_PHASE
            + rates            # primary mineral RATES (PK04)
            + glass_rates      # BasaltGlass RATES
            + carb_rates       # carbonate kinetic RATES (empty when EQ mode)
            + "KNOBS\n"
            + "    -iterations             500\n"
            + "    -convergence_tolerance  1e-7\n"
            + "    -step_size              20.0\n"
            + "    -pe_step_size           10.0\n\n"
            + sol0
            + reaction         # empty string (disabled)
            + kin              # primary + glass KINETICS blocks
            + carb_kin         # carbonate KINETICS blocks (empty when EQ mode)
            + eq               # carbonates + clays + CO2(g) EQ phases
            + sel
            + transport
            + "END\n"
        )

    # -----------------------------------------------------------------------
    def _build_batch_input(self, dt_years, log_pco2_eff, delay_calcite,
                            carb_cap, cap_now, rates, sel,
                            co2_mol_per_kgw=0.0):
        """
        Batch reactor (no TRANSPORT) — used during injection when pCO2 is high.

        FIX 3 (v23): REACTION block is now included when co2_mol_per_kgw > 0.
        This adds the dissolved fraction of injected CO2 explicitly to the
        pore fluid for correct DIC → pH coupling. Only the dissolved portion
        (capped at solubility) is passed — excess free-phase CO2 is excluded.

        FIX 4 (v23): During monitoring OFF cycles (co2_mol_per_kgw=0), the
        USE SOLUTION 1 path preserves accumulated cations so carbonate
        precipitation continues uninterrupted through the pulse gap.
        """
        rp = self.reg_params
        T  = rp['T_C']

        is_inj = log_pco2_eff > -2.0   # batch is used when pCO2 > -2.0

        # ── FIX v29: Carry-forward solution for pulsed injection restarts ──────
        # When injection restarts after an OFF period (pulsed pulse 2+), we must
        # NOT reset the PHREEQC solution to initial porewater. The brine has evolved
        # (pH~6.1, elevated DIC). Resetting to initial_pH=7.5 causes pH to snap to
        # 7.5 then drop — the unphysical instantaneous jump visible in the plots.
        #
        # Fix: use "USE solution 1" (carry forward evolved solution) for all steps
        # except the very first step, where there is no prior solution.
        # CO2(g) EQUILIBRIUM_PHASES then drives pH down smoothly from ~6.1 as CO2
        # dissolves, giving the correct gradual re-acidification.
        _first_inj_step = (self._step <= 1)

        if is_inj and _first_inj_step:
            # Very first step: build fresh SOLUTION from initial porewater
            inlet_pH  = max(4.5, rp.get('initial_pH', 7.5)
                            + 0.65 * (CONFIG_LOG_PCO2_BG - log_pco2_eff))
            inlet_dic = 2.0
            sol1 = (
                "SOLUTION 1\n"
                f"    temp      {T:.1f}\n"
                f"    pH        {inlet_pH:.2f}\n"
                f"    pe        {self._pe_current:.3f}\n"
                "    units     mmol/kgw\n"
                f"    density   {rp.get('brine_density', 1.02):.4f}\n"
                f"    Ca        {rp.get('pw_Ca_mmol', 1.0):.4f}\n"
                f"    Mg        {rp.get('pw_Mg_mmol', 0.5):.4f}\n"
                f"    Na        {rp.get('pw_Na_mmol', 20.0):.4f}\n"
                f"    K         {rp.get('pw_K_mmol', 1.0):.4f}\n"
                f"    Fe(2)     {rp.get('pw_Fe2_mmol', 0.5):.5f}\n"
                f"    Si        {rp.get('pw_Si_mmol', 1.0):.5f}\n"
                f"    Al        {rp.get('pw_Al_mmol', 0.02):.5f}\n"
                f"    C(4)      {inlet_dic:.4f}\n"
                f"    Cl        {rp.get('Cl_mmol', 10.0):.3f}\n"
                f"    S(6)      {rp.get('SO4_mmol', 0.5):.3f}\n\n"
            )
        else:
            # All other steps: carry forward the evolved solution (no chemistry reset)
            sol1 = "USE solution 1\n\n"


        # REMOVED: REACTION block conflicts with CO2(g) EQUILIBRIUM_PHASES.
        reaction = ""

        # ── PRIMARY KINETICS: skip during high-pCO2 injection ────────────────
        # ROOT CAUSE OF HANG: running primary KINETICS + CO2(g) EQ at high pCO2
        # (log_pco2 > -1.5) in a single batch solve causes PHREEQC to hang.
        # v27 FIX: The analytic fallback must enforce a CUMULATIVE cap per mineral,
        # not just a per-step cap. Without the cumulative cap, minerals dissolve 100%
        # of their inventory (~2.6e9 mol for Diopside) in <0.25 yr — physically wrong
        # by ~400x. The cumulative cap limits total dissolution to what the PK04 rate
        # law would produce at sustained pH 5 over 5 years (~17% of m0 for Diopside).
        _high_pco2_batch = (log_pco2_eff > -1.5)
        if _high_pco2_batch:
            kin = ""   # skip kinetics — see justification above
            self._inj_steps_skipped = getattr(self, '_inj_steps_skipped', 0) + 1
            _pH_est = max(4.5, rp.get('initial_pH', 7.5)
                          + 0.65 * (CONFIG_LOG_PCO2_BG - log_pco2_eff))
            _aH_est = 10.0 ** (-_pH_est)
            _T_K    = rp.get('T_C', 50.0) + 273.15
            dt_yr_now  = getattr(self, '_last_dt_yr', 0.001)
            dt_ref_now = 0.02
            # Per-step cap: limits mol dissolved this single step
            cap_inj    = _MAX_KINETIC_MOL_PER_KGW_PER_STEP * (dt_yr_now / dt_ref_now)
            cap_inj    = max(cap_inj, 1e-10)
            for _m, _pk in _PK04_PARAMS.items():
                _lka, _ea_a, _na = _pk[0], _pk[1], _pk[2]
                _lkn, _ea_n      = _pk[3], _pk[4]
                _ka = (10.0 ** _lka) * math.exp(-_ea_a * (1.0 / _T_K - 1.0 / 298.15))
                _kn = (10.0 ** _lkn) * math.exp(-_ea_n * (1.0 / _T_K - 1.0 / 298.15))
                _r  = _ka * (_aH_est ** abs(_na)) + _kn    # mol/m²/s
                _A  = _pk[8] * self.mineral_fracs.get(_m, 1.0)
                _m0_field = self.m0_primary.get(_m, 0.0)
                _m_cur    = self.m_primary_current.get(_m, 0.0)
                if _m0_field < 1e-12 or _m_cur < 1e-12:
                    continue
                _sa_factor = (_m_cur / max(_m0_field, 1e-30)) ** 0.5
                _dt_sec    = dt_yr_now * 365.25 * 86400.0
                _dm_kgw    = _A * _sa_factor * _r * _dt_sec   # mol/kgw this step
                # Cumulative cap: total dissolution over full simulation cannot
                # exceed what the rate law gives at sustained pH 5 over 5 yr,
                # AND cannot exceed 10% of the initial inventory at field scale.
                # (No primary silicate dissolves >10% in 5yr at realistic basalt
                # weathering rates — White & Brantley 2003; Gudbrandsson 2011.)
                _m0_kgw = _m0_field / max(self.pore_kg, 1.0)
                _cum_max_rate = _A * _m0_kgw * _r * (5.0 * 365.25 * 86400.0)
                _cum_max_physical = _m0_kgw * 0.10   # hard 10% ceiling per mineral
                _cum_max_kgw = min(_cum_max_rate, _cum_max_physical, _m0_kgw)
                _cum_so_far  = (_m0_field - _m_cur) / max(self.pore_kg, 1.0)
                _cum_remaining = max(_cum_max_kgw - _cum_so_far, 0.0)
                # Apply both per-step and cumulative caps
                _dm_kgw = min(_dm_kgw, cap_inj, _cum_remaining,
                              _m_cur / max(self.pore_kg, 1.0))
                _dm_kgw = max(_dm_kgw, 0.0)
                if _dm_kgw > 0:
                    _diss_field = _dm_kgw * self.pore_kg
                    self.m_primary_current[_m] = max(_m_cur - _diss_field, 0.0)
                    self.cum_dissolved[_m] = max(
                        _m0_field - self.m_primary_current[_m], 0.0)
        else:
            kin = _build_kinetics_block(
                self.m_primary_current, self.m0_primary, dt_years, self.pore_kg,
                n_cells=1)

        # FIX v27: carbonate_kinetic=False everywhere — never build kinetic carbonate blocks.
        # Carbonate KINETICS caused USE SOLUTION 1 to lose Ca/Mg/Fe → zero precipitation.
        _use_carb_kin_now = False   # always EQ-phase
        carb_rates = ""
        carb_kin   = ""

        # ── CO2(g) supply: limit to ~2× solubility to prevent Newton overshoot ──
        _sol_mmol = co2_solubility_mmol_kgw(rp['T_C'], max(10**log_pco2_eff, 1.0))
        _co2g_supply = max(_sol_mmol * 1e-3 * 2.0, 0.05)   # mol, 2× solubility, min 0.05

        eq = _build_eq_phases(
            saponite=self._saponite_ok,
            current_eq_mol_kgw=self._eq_inv_kgw,
            delay_calcite=delay_calcite,
            log_pco2=log_pco2_eff,
            carb_cap=carb_cap,
            cell_num=1,
            carbonate_kinetic=False,   # FIX v27: always EQ
            co2g_supply=_co2g_supply,
            skip_clays=_high_pco2_batch,
            reg_params=self.reg_params,  # FIX 3c: K-gate for Muscovite
        )

        return (
            _ANKERITE_PHASE + _BASALTGLASS_PHASE + _CLINOCHLORE_PHASE
            + rates
            + "KNOBS\n"
            + "    -iterations             1000\n"
            + "    -convergence_tolerance  1e-6\n"
            + "    -step_size              20.0\n"
            + "    -pe_step_size           10.0\n\n"
            + sol1 + reaction + kin + eq + sel
            + "SAVE solution 1\n"
            + "END\n"
        )

    # -----------------------------------------------------------------------
    def _parse_output(self):
        result = {'pH': None, 'pe': None, 'alkalinity': 2e-3, 'error': False}
        for m in CARBONATE_MINERALS + CLAY_MINERALS:
            result[f'delta_{m}'] = 0.0
            result[f'm_{m}']     = 0.0
        for m in PRIMARY_MINERALS:
            result[f'kin_{m}'] = 0.0
        for m in CARBONATE_MINERALS:
            result[f'SI_{m}'] = None

        try:
            arr = self.iph.get_selected_output_array()
            if not arr or len(arr) < 2:
                result['error'] = True
                return result

            # Lowercase headers for consistent mapping
            headers = [str(h).strip().lower() for h in arr[0]]
            hmap    = {h: idx for idx, h in enumerate(headers)}
            
            # Data rows: one row per cell
            data_rows = arr[1:]
            
            def get_mean(key):
                col_idx = hmap.get(key.lower())
                if col_idx is None: return 0.0
                vals = []
                for row in data_rows:
                    try:
                        v = float(row[col_idx])
                        vals.append(v)
                    except: pass
                return float(np.mean(vals)) if vals else 0.0

            def get_cell1(key):
                col_idx = hmap.get(key.lower())
                if col_idx is None: return 0.0
                try:    return float(data_rows[0][col_idx])
                except: return 0.0

            # pH: inlet cell shows the front
            result['pH'] = get_cell1('ph')
            pe_raw = get_mean('pe')
            result['pe'] = float(np.clip(pe_raw, _PE_MIN, _PE_MAX))

            # v26: spatial profiles — pH and DIC across all cells at this step
            ph_col = hmap.get('ph')
            dic_col = hmap.get('c(4)(mol/kgw)')
            if ph_col is not None:
                result['spatial_pH'] = []
                for row in data_rows:
                    try:    result['spatial_pH'].append(float(row[ph_col]))
                    except: result['spatial_pH'].append(float('nan'))
            if dic_col is not None:
                result['spatial_DIC'] = []
                for row in data_rows:
                    try:    result['spatial_DIC'].append(float(row[dic_col]) * 1000.0)  # mmol/kgw
                    except: result['spatial_DIC'].append(float('nan'))
            
            result['alkalinity'] = (
                get_mean('alk(eq/kgw)') or get_mean('alkalinity') or 2e-3)
            
            result['volume_L'] = get_mean('volume') or get_mean('soln vol(l)') or 1.0

            # Elementals (Integrated Mean)
            for el, cols in [
                ('Ca',  ['ca(mol/kgw)']),
                ('Mg',  ['mg(mol/kgw)']),
                ('Fe',  ['fe(2)(mol/kgw)', 'fe(mol/kgw)']),
                ('Al',  ['al(mol/kgw)']),
                ('Si',  ['si(mol/kgw)']),
                ('DIC', ['c(4)(mol/kgw)']),
            ]:
                for col in cols:
                    if col in hmap:
                        result[f'total_{el}'] = get_mean(col) * 1000.0
                        break

            # Integrated secondary minerals (EQ_PHASES)
            for m in CARBONATE_MINERALS + CLAY_MINERALS:
                m_lo = m.lower()
                c_lo = m.replace('-', '_').replace(' ', '_').lower()
                
                # Mean concentration in column (mol/kgw)
                inv_val = 0.0
                for key in [m_lo, c_lo]:
                    if key in hmap:
                        inv_val = get_mean(key); break
                result[f'eq_inv_{m}'] = max(inv_val, 0.0)

                # Mean step change
                delta_val = 0.0
                for key in [f'd_{m_lo}', f'd_{c_lo}']:
                    if key in hmap:
                        delta_val = get_mean(key); break
                result[f'm_{m}']     = delta_val
                result[f'delta_{m}'] = delta_val

            # SIs (Mean)
            for m in CARBONATE_MINERALS:
                c_lo = m.replace('-', '_').replace(' ', '_').lower()
                for key in [f'si_{m.lower()}', f'si_{c_lo}']:
                    if key in hmap:
                        result[f'SI_{m}'] = get_mean(key); break

            # Primary Kinetics (Mean)
            for m in PRIMARY_MINERALS:
                c_lo = m.replace('-', '_').replace(' ', '_').lower()
                for key in [f'k_{m.lower()}', f'k_{c_lo}']:
                    if key in hmap:
                        result[f'kin_{m}_remaining'] = get_mean(key); break
                for key in [f'dk_{m.lower()}', f'dk_{c_lo}', f'dm_{m.lower()}']:
                    if key in hmap:
                        result[f'kin_{m}'] = abs(get_mean(key)); break

            # Carbonate Kinetics (Mean)
            for m in CARBONATE_MINERALS:
                c_lo = m.replace('-', '_').replace(' ', '_').lower()
                for key in [f'k_{m.lower()}', f'k_{c_lo}']:
                    if key in hmap:
                        result[f'kin_{m}_remaining'] = max(get_mean(key), 0.0); break
                for key in [f'dk_{m.lower()}', f'dk_{c_lo}']:
                    if key in hmap:
                        result[f'kin_{m}'] = get_mean(key); break

        except Exception as e:
            result['error'] = True
            if self._step <= 5:
                print(f"  [WARN] Parse error step {self._step}: {e}", flush=True)


        return result

    # -----------------------------------------------------------------------
    def reset(self):
        self.cum_precip        = {m: 0.0 for m in CARBONATE_MINERALS + CLAY_MINERALS}
        self.cum_dissolved     = {m: 0.0 for m in PRIMARY_MINERALS}
        self.m_primary_current = dict(self.m0_primary)
        self._m_kin_prev       = {}
        self._eq_inv_kgw       = {}
        self._step             = 0
        self._total_co2        = 0.0
        self._dic_budget_mol   = self._initial_dic_mol
        self._last_pH          = 7.5
        self._t_monitor_yr     = 0.0
        self._saponite_ok      = True
        self._pressure_bar     = 8
        self._pe_current       = self._pe0
        self._spatial_pH       = []
        self.iph = IPhreeqc()
        self.iph.load_database(self.database_path)

    def close(self):
        self.iph = None


# ===========================================================================
# ── NEW v20: SENSITIVITY ANALYSIS
# ===========================================================================
def perturb_params(base_rp, param_name, factor):
    """
    Return a copy of region_params with one parameter scaled by `factor`.

    Used to build sensitivity runs: e.g., perturb_params(rp, 'T_C', 1.1)
    creates a run at 110% of the base temperature.

    Parameters
    ----------
    base_rp    : dict — baseline region_params
    param_name : str  — key to perturb
    factor     : float — multiplicative factor (e.g. 0.8 = −20%, 1.2 = +20%)
    """
    import copy
    rp = copy.deepcopy(base_rp)
    if param_name in rp:
        rp[param_name] = rp[param_name] * factor
    else:
        raise KeyError(f"perturb_params: '{param_name}' not found in reg_params")
    return rp


def run_sensitivity(run_fn, base_config, region, mode,
                    param_names=None, factors=None):
    """
    Run a sensitivity sweep by perturbing each parameter in `param_names`
    by each factor in `factors` and collecting the key output metrics.

    Parameters
    ----------
    run_fn       : callable — run_simulation(region, mode, config) → result dict
    base_config  : dict — the CONFIG dict
    region, mode : str
    param_names  : list of str — parameters to perturb.
                   Default: ['T_C', 'initial_porosity', 'transport_vel_m_yr',
                              'dispersivity_m', 'dissolution_boost']
    factors      : list of float — perturbation factors.
                   Default: [0.8, 0.9, 1.0, 1.1, 1.2]

    Returns
    -------
    list of dicts, one per (param, factor) combination, each with keys:
      param, factor, pH_nadir, pH_final, CO2_seq_mol_kgw, porosity_final
    """
    import copy

    if param_names is None:
        param_names = ['T_C', 'initial_porosity', 'transport_vel_m_yr',
                       'dissolution_boost', 'permeability_mD']
    if factors is None:
        factors = [0.80, 0.90, 1.00, 1.10, 1.20]

    results = []
    for pname in param_names:
        for fac in factors:
            cfg = copy.deepcopy(base_config)
            rp  = cfg['region_params'][region]
            if pname not in rp:
                print(f"  [SENS] Skipping '{pname}' — not in {region} params")
                continue
            rp[pname] = rp[pname] * fac
            print(f"  [SENS] {pname} × {fac:.2f} = {rp[pname]:.4g}")
            try:
                r = run_fn(region, mode, cfg)
                results.append({
                    'param':            pname,
                    'factor':           fac,
                    'perturbed_value':  rp[pname],
                    'pH_nadir':         float(r['pH'].min()),
                    'pH_final':         float(r['pH'][-1]),
                    'CO2_seq_mol_kgw':  float(r['co2_seq_kgw'][-1]),
                    'porosity_final':   float(r['porosity'][-1]),
                    'Calcite_final':    float(r['minerals']['Calcite'][-1]),
                    'Ankerite_final':   float(r['minerals']['Ankerite'][-1]),
                })
            except Exception as e:
                print(f"  [SENS] Run failed ({pname}×{fac}): {e}")
                results.append({'param': pname, 'factor': fac, 'error': str(e)})
    return results


# ===========================================================================
# ── NEW v20: MONTE CARLO UNCERTAINTY ANALYSIS
# ===========================================================================
def run_monte_carlo(run_fn, base_config, region, mode,
                    n_samples=100, kinetic_spread=0.50, seed=42):
    """
    Run a Monte Carlo uncertainty analysis by sampling kinetic rate constants
    from a log-uniform distribution within ±`kinetic_spread` of their base
    value and collecting key output statistics.

    Physical rationale: kinetic rate constants in Palandri & Kharaka (2004)
    are reported with uncertainties of ±0.3–0.5 log units, corresponding
    to a factor of 2–3 in rate.  A ±50% (factor 1.5) uniform spread is a
    conservative representation of this uncertainty.

    Parameters
    ----------
    n_samples      : int   — number of Monte Carlo realisations
    kinetic_spread : float — fractional spread on dissolution_boost
                             (0.50 = ±50% of base value)
    seed           : int   — random seed for reproducibility

    Returns
    -------
    dict with keys:
      'samples'       : list of per-sample result dicts
      'pH_nadir_p5'   : 5th percentile pH nadir across samples
      'pH_nadir_p50'  : median pH nadir
      'pH_nadir_p95'  : 95th percentile pH nadir
      'CO2_seq_p5'    : 5th percentile final CO₂ mineralisation
      'CO2_seq_p50'   : median
      'CO2_seq_p95'   : 95th percentile
    """
    import copy
    rng     = np.random.default_rng(seed)
    samples = []

    print(f"\n[MC] Starting {n_samples} Monte Carlo runs | "
          f"region={region} mode={mode} spread=±{kinetic_spread*100:.0f}%")

    for i in range(n_samples):
        cfg = copy.deepcopy(base_config)
        rp  = cfg['region_params'][region]

        # Sample dissolution_boost uniformly from [1-spread, 1+spread]
        boost = rp.get('dissolution_boost', 1.0) * rng.uniform(
            1.0 - kinetic_spread, 1.0 + kinetic_spread)
        rp['dissolution_boost'] = float(np.clip(boost, 0.1, 5.0))

        # Also perturb permeability if present (log-uniform)
        if 'permeability_mD' in rp:
            log_k0   = math.log10(rp['permeability_mD'])
            log_k    = rng.uniform(log_k0 - 0.3, log_k0 + 0.3)
            rp['permeability_mD'] = 10.0 ** log_k

        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  [MC] {i+1}/{n_samples} boost={rp['dissolution_boost']:.3f}")

        try:
            r = run_fn(region, mode, cfg)
            samples.append({
                'sample':          i,
                'dissolution_boost': rp['dissolution_boost'],
                'pH_nadir':        float(r['pH'].min()),
                'pH_final':        float(r['pH'][-1]),
                'CO2_seq_mol_kgw': float(r['co2_seq_kgw'][-1]),
                'porosity_final':  float(r['porosity'][-1]),
            })
        except Exception as e:
            print(f"  [MC] Sample {i} failed: {e}")

    if not samples:
        return {'samples': [], 'error': 'All runs failed'}

    ph_nadirs = np.array([s['pH_nadir']        for s in samples])
    co2_seqs  = np.array([s['CO2_seq_mol_kgw'] for s in samples])

    summary = {
        'samples':       samples,
        'n_valid':       len(samples),
        'pH_nadir_p5':   float(np.percentile(ph_nadirs,  5)),
        'pH_nadir_p50':  float(np.percentile(ph_nadirs, 50)),
        'pH_nadir_p95':  float(np.percentile(ph_nadirs, 95)),
        'CO2_seq_p5':    float(np.percentile(co2_seqs,   5)),
        'CO2_seq_p50':   float(np.percentile(co2_seqs,  50)),
        'CO2_seq_p95':   float(np.percentile(co2_seqs,  95)),
    }
    print(f"\n[MC] Results: pH_nadir [{summary['pH_nadir_p5']:.2f}–"
          f"{summary['pH_nadir_p95']:.2f}] | "
          f"CO2_seq [{summary['CO2_seq_p5']:.3f}–{summary['CO2_seq_p95']:.3f}] mol/kgw")
    return summary


# ===========================================================================
# ── NEW v20: WALLULA CALIBRATION CHECK  [White et al. 2020]
# ===========================================================================
def calibrate_to_wallula(results_p):
    """
    Compare simulation output against key Wallula pilot field observations.

    Wallula Phase II (White et al. 2020, Science Advances):
      - 977 t CO₂ injected over 25 days
      - pH dropped to ~5.0–5.5 below injection horizon
      - >60% CO₂ mineralised as ankerite within ~2 years
      - pH recovered to 7.5–8.0 by year 2
      - Porosity initially ~10–15%, declined 2–5% absolute

    This function prints a pass/fail report and returns a dict of metrics.

    Parameters
    ----------
    results_p : dict — output from run_simulation() for a single-injection
                run at CRBG conditions (the closest analogue to Wallula)
    """
    r   = results_p
    t   = r['time']
    pH  = r['pH']
    por = r['porosity']
    seq = r['co2_seq_kgw']
    inj = r.get('co2_injected_total', 0.0)

    # Find efficiency at ~2 years post injection end
    t2m  = t <= (r['inj_window'] + 2.0)
    seq_2yr = seq[t2m][-1] if np.any(t2m) else seq[-1]
    eff_2yr = (seq_2yr / max(inj / max(r['reg_params'].get('pore_kg_approx', 1.0), 1), 1e-12)) * 100.0

    metrics = {
        'pH_nadir':        float(pH.min()),
        'pH_final':        float(pH[-1]),
        'CO2_seq_final':   float(seq[-1]),
        'efficiency_2yr':  float(eff_2yr),
        'porosity_initial':float(por[0]),
        'porosity_final':  float(por[-1]),
    }

    # Wallula target ranges (White et al. 2020; Schaef & McGrail 2009)
    targets = [
        ('pH_nadir',        5.0,  5.6,  'Schaef & McGrail (2009)'),
        ('pH_final',        7.5,  8.5,  'Galerne & Haug (2020)'),
        ('efficiency_2yr', 35.0, 65.0,  'White et al. (2020)'),
        ('porosity_initial',10.0, 25.0, 'Zakharova et al. (2012)'),
        ('porosity_final',   4.0, 13.0, 'Galerne & Haug (2020)'),
    ]

    print("\n" + "=" * 68)
    print("WALLULA CALIBRATION CHECK  [White et al. 2020 / Schaef & McGrail 2009]")
    print("=" * 68)
    all_pass = True
    for name, lo, hi, ref in targets:
        val    = metrics.get(name, float('nan'))
        passed = lo <= val <= hi
        all_pass = all_pass and passed
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:26s}: {val:7.2f}  [{lo:.1f}–{hi:.1f}]  {status}  {ref}")
    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAIL ✗'}")
    print("  Engine: PHREEQC v20 | Carbonate KINETICS | Duan & Sun CO₂ solubility")
    print("=" * 68)
    metrics['all_pass'] = all_pass
    return metrics

    def close(self):
        self.iph = None

    # -----------------------------------------------------------------------
    def validation_check(self, results_p, inj_end_time):
        r = results_p
        t, pH, eff, por = r['time'], r['pH'], r['efficiency'], r['porosity']
        t2m  = t <= (inj_end_time + 2.0)
        e2yr = eff[t2m][-1] if np.any(t2m) else eff[-1]
        checks = [
            ("pH nadir",           pH.min(), 5.0,  5.6,  "[Schaef & McGrail 2009]"),
            ("pH final",           pH[-1],   7.5,  8.5,  "[Galerne & Haug 2020]"),
            ("Efficiency @2yr %",  e2yr,    35.0, 65.0,  "[White 2020 / Galerne 2020]"),
            ("Efficiency @end %",  eff[-1], 20.0, 65.0,  "[Galerne & Haug 2020]"),
            ("Porosity initial %", por[0],  10.0, 25.0,  "[Zakharova 2012]"),
            ("Porosity final %",   por[-1],  4.0, 13.0,  "[Galerne & Haug 2020]"),
        ]
        print("\n" + "=" * 68)
        print("VALIDATION | CRBG 8-bar vs Literature")
        print("=" * 68)
        ok_all = True
        for name, val, lo, hi, ref in checks:
            ok     = lo <= val <= hi
            ok_all = ok_all and ok
            print(f"  {name:26s}: {val:7.2f}  [{lo:.1f}--{hi:.1f}]  "
                  f"{'PASS' if ok else 'FAIL'}  {ref}")
        print(f"\n  Overall : {'ALL PASS' if ok_all else 'SOME FAIL'}")
        print("  Engine  : PHREEQC v18 | TRANSPORT 1-D | McGrail 2017 porewater")
        return ok_all


# ===========================================================================
# EXCEL EXPORT
# ===========================================================================
def save_excel(results, xrf_data, mineral_fracs, scalers, reg_params,
               pressures, excel_path, config_extra=None):
    import pandas as pd

    lit = [
        {'Parameter':'pH nadir',            'Low':5.0,  'High':5.6,  'Ref':'Schaef & McGrail (2009)'},
        {'Parameter':'pH recovery',          'Low':7.5,  'High':8.5,  'Ref':'Galerne & Haug (2020)'},
        {'Parameter':'Efficiency @2yr (%)',  'Low':35.0, 'High':65.0, 'Ref':'White et al. (2020) / Galerne & Haug (2020)'},
        {'Parameter':'Efficiency @end (%)',  'Low':20.0, 'High':65.0, 'Ref':'Galerne & Haug (2020)'},
        {'Parameter':'Porosity initial (%)', 'Low':10.0, 'High':25.0, 'Ref':'Zakharova et al. (2012)'},
        {'Parameter':'Porosity final (%)',   'Low':4.0,  'High':13.0, 'Ref':'Galerne & Haug (2020)'},
    ]
    cfg = {
        'Region':    reg_params.get('region','?'),
        'Rock_Type': reg_params.get('rock_type_label','?'),
        'T_C':       reg_params['T_C'],
        'Depth_m':   reg_params.get('injection_depth_m','?'),
        'Engine': 'PHREEQC v18 — TRANSPORT 1-D | PK04 kinetics',
    }
    if config_extra:
        cfg.update(config_extra)
    cfg.update({f'Scaler_{k}': v for k, v in scalers.items()})

    with pd.ExcelWriter(excel_path, engine='openpyxl') as w:
        pd.DataFrame([cfg]).to_excel(w, sheet_name='Configuration', index=False)
        pd.DataFrame([xrf_data]).to_excel(w, sheet_name='XRF_Input', index=False)
        pd.DataFrame([mineral_fracs]).to_excel(w, sheet_name='Mineral_Fractions', index=False)
        pd.DataFrame([scalers]).to_excel(w, sheet_name='XRF_Scalers', index=False)
        pd.DataFrame(lit).to_excel(w, sheet_name='Literature_Targets', index=False)

        sm = []
        for p in pressures:
            r   = results[p]
            row = {
                'Pressure_bar':        p,
                'CO2_injected_mol':    r['co2_injected'][-1],
                'CO2_injected_t':      r['co2_injected'][-1] * 44.01 / 1e6,
                'CO2_mineralized_mol': r['co2_seq'][-1],
                'Efficiency_end_%':    r['efficiency'][-1],
                'Eff_2yr_%':           r.get('eff_at_2yr', r['efficiency'][-1]),
                'pH_nadir':            r['pH'].min(),
                'pH_final':            r['pH'][-1],
                'Por_initial_%':       r['porosity'][0],
                'Por_final_%':         r['porosity'][-1],
                'Engine':              r.get('source','?'),
            }
            for m in CARBONATE_MINERALS + CLAY_MINERALS:
                row[f'{m}_final_mol'] = r['minerals'].get(m, np.array([0]))[-1]
            sm.append(row)
        pd.DataFrame(sm).to_excel(w, sheet_name='Summary', index=False)

        for p in pressures:
            r  = results[p]
            df = pd.DataFrame({
                'Time_yr':      r['time'],
                'pH':           r['pH'],
                'Porosity_%':   r['porosity'],
                'CO2_inj_mol':  r['co2_injected'],
                'CO2_min_mol':  r['co2_seq'],
                'Efficiency_%': r['efficiency'],
            })
            for m in CARBONATE_MINERALS + CLAY_MINERALS:
                df[f'{m}_mol'] = r['minerals'].get(m, np.zeros(len(r['time'])))
            for m in PRIMARY_MINERALS:
                df[f'{m}_diss'] = r['dissolved'].get(m, np.zeros(len(r['time'])))
            df.to_excel(w, sheet_name=f'{p}bar', index=False)

    print(f"  [EXPORT] {excel_path}")


# ===========================================================================
# STYLE CONSTANTS
# ===========================================================================
LINE_STYLES = {3: '--', 5: '-', 8: '-', 10: ':'}
COLORS      = {3: '#1f77b4', 5: '#d62728', 8: '#2ca02c', 10: '#ff7f0e'}
CARB_COLORS = {
    'Calcite': '#1f77b4', 'Siderite': '#d62728', 'Magnesite': '#2ca02c',
    'Dolomite': '#9467bd', 'Ankerite': '#ff7f0e',
}
CLAY_COLORS = {
    'Saponite-Mg': '#e377c2', 'Clinochlore-14A': '#7f7f7f',
    'Kaolinite': '#bcbd22',   'Muscovite': '#17becf',
}
MIN_COLORS = {
    'Diopside': '#8B4513', 'Anorthite': '#4169E1', 'Albite': '#32CD32',
    'Magnetite': '#696969', 'Ilmenite': '#9932CC',
}