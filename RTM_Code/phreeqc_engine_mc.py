"""
phreeqc_engine_mc.py
====================
Mass-conservative reactive-transport engine (revision R2, corrected version).

WHY THIS MODULE EXISTS
----------------------
The engine used for the submitted revision (``PhreeqcEngine`` in
phreeqc_engine.py) was reviewed and found to have structural problems that
cannot be fixed by changing parameter values:

  [FIX-03] CO2 entered through a CO2(g) equilibrium phase placed in EVERY cell,
           with log pCO2 prescribed by a Python ramp function (plus pCO2
           "floors" of -2.5 / -3.5 during OFF and monitoring periods).  That
           is an unlimited, open-system carbon source: mineralised carbon was
           not limited by the CO2 actually injected (a run with NO injection
           gave almost the same "efficiency").  The REACTION block that would
           have added the injected mass was disabled.
  [FIX-05] Python post-processing overrode PHREEQC's mineral amounts after
           every step (no growth during injection, 90 %-of-peak floor, 2x
           growth cap, 1-3 % dissolution cap, clay lock-in, per-step carbonate
           cap CARB_CAP with mode-specific multipliers).  These overrides
           break mass conservation and set the result.
  [FIX-06] TRANSPORT always used "-shifts 1 -time_step dt" with 5-m cells, so
           the effective advection velocity was cell_length/dt, unrelated to
           the Darcy velocity reported in the manuscript.
  [FIX-07] Whenever log pCO2 > -1.5 the primary-mineral KINETICS were skipped
           and dissolution was computed in Python from an assumed pH, with the
           released cations never added to the solution; batch steps updated
           only cell 1.

This engine keeps the manuscript's conceptual set-up (1-D column of N cells,
same XRF-derived rock, same injection schedules, same Darcy velocity from
the 8-bar overpressure) but lets PHREEQC carry the full state:

  * Solutions, equilibrium-phase assemblages and kinetic reactants are defined
    ONCE, pre-equilibrated at t = 0, and then persist inside IPhreeqc between
    calls.  No Python code ever overwrites a mineral amount.
  * Injected CO2 is added BY MASS to cell 1 each step (q * dt), limited by the
    Spycher et al. (2003) solubility at the actual bottom-hole pressure
    [FIX-11]; any undissolved remainder is carried as a free-phase inventory
    and dissolved in later steps.  There is no CO2(g) phase anywhere, so the
    only carbon sources are (i) the injected CO2, (ii) the initial pore water
    and (iii) native formation water entering at the inlet.  All three are
    tracked.
  * TRANSPORT is advanced with an integer number of shifts per step obtained
    by accumulating v_pore * dt / dx, with time_step = dt / n_shifts, so the
    advective velocity equals the Darcy velocity / porosity [FIX-06].  When
    less than one cell volume has moved, the step is a reaction-only
    (diffusion_only) step and the fractional shift is carried forward.
  * Primary minerals dissolve by Palandri & Kharaka (2004) TST kinetics with
    the affinity term (1 - SR) [FIX-13], evaluated by PHREEQC in every cell on
    every step (no skipping at high pCO2) [FIX-07].
  * Basaltic glass dissolves with the full stoichiometry of the site's bulk
    composition (Si, Al, Fe, Mg, Ca, Na, K) instead of SiO2 only [FIX-09].
    Its rate law (acid + neutral terms) is the submitted one, multiplied by
    (1 - a_SiO2/K_SiO2(am)) unless CONFIG["mc_engine"]["glass_affinity"] is
    "none"; this affinity term is a modelling choice added in the correction.
  * Secondary minerals are unconstrained equilibrium phases (initial amount
    zero) with LLNL.dat thermodynamics; the user-defined Clinochlore-14A
    (log K 36.80) is NOT loaded [FIX-10].
  * Abiotic reduction of carbonate to CH4, of sulfate to sulfide and of water
    to H2 is switched off: none of these proceeds at 45-50 C on these time
    scales, and at the low pe of Fe(II)-rich basalt they would otherwise remove
    carbon from the C(4) budget and oxidise Fe(II) to Fe(OH)3.
  * A full carbon ledger is computed from PHREEQC totals: initial + injected +
    inflow = aqueous + mineral + free phase + exported.  The closure error is
    at machine precision; it is reported so that reviewers can verify it.

The engine exposes the same ``step()`` return dictionary as the legacy engine
so that run_rtm.run_simulation() needs only small changes.

References
----------
Palandri, J.L. & Kharaka, Y.K. (2004) USGS Open-File Report 2004-1068.
Spycher, N., Pruess, K. & Ennis-King, J. (2003) GCA 67, 3015-3031.
Parkhurst, D.L. & Appelo, C.A.J. (2013) USGS TM 6-A43 (PHREEQC 3 manual).
"""

from __future__ import annotations

import math
import numpy as np

from phreeqc_engine import (
    PhreeqcEngine, IPhreeqc, PHREEQC_AVAILABLE,
    xrf_to_chemistry, _pe_from_xrf, _PE_MIN, _PE_MAX,
    PRIMARY_MINERALS, CARBONATE_MINERALS, CLAY_MINERALS, CO2_PER_CARBONATE,
    MOLAR_VOLUME_CM3, MW_MINERAL, MW_OXIDE,
    _GLASS_KIN_PARAMS, _PK04_PARAMS, _PK04_PARAMS_PK2004,
    darcy_velocity_m_yr, _ANKERITE_PHASE,
)
from co2_solubility import co2_solubility_mol_kgw, co2_density_kg_m3

ENGINE_NAME = "PHREEQC mass-conservative engine (R2-corrected, phreeqc_engine_mc.py)"

# Secondary phases that are not in the legacy CARBONATE/CLAY lists
OTHER_SECONDARY = ['SiO2(am)', 'Fe(OH)3']

# Default equilibrium assemblage for 45-50 C basalt-CO2 systems.
#   Carbonates : calcite, siderite.  Ordered dolomite, magnesite and ankerite are
#                kinetically inhibited below ~60-80 C and are excluded by default;
#                the user-defined "Ankerite" copies dolomite's log K and has no
#                database basis.
#   Clays      : kaolinite (acidic) and saponite (neutral-alkaline) are the
#                low-temperature Al/Mg sinks.  Clinochlore (chlorite) and
#                muscovite are high-temperature phases and are excluded by
#                default (set "secondary_phases" in CONFIG["mc_engine"] to test).
#   Others     : amorphous silica (Si sink) and Fe(OH)3 (Fe(III) sink / redox
#                buffer from magnetite dissolution).
DEFAULT_SECONDARY_EQ = ['Calcite', 'Siderite', 'Kaolinite', 'Saponite-Mg',
                        'SiO2(am)', 'Fe(OH)3']

# Molar volumes (cm3/mol) used for porosity; extends the legacy table
MOLAR_VOLUME_MC = dict(MOLAR_VOLUME_CM3)
MOLAR_VOLUME_MC.update({
    'Magnetite': 44.52,        # was missing (legacy default 50)
    'SiO2(am)':  29.00,
    'Fe(OH)3':   34.36,
    'Clinochlore-14A': 207.11,
    'Ankerite':  64.39,
})

_SEC_PER_YR = 365.25 * 86400.0

# Charges used to build a charge-balanced glass formula
_GLASS_CATIONS = (  # (element, oxide, cations per oxide, valence, atomic mass)
    ('Al', 'Al2O3', 2.0, 3, 26.982),
    ('Fe', 'Fe2O3', 2.0, 2, 55.845),   # released as Fe(II); redox set by PHREEQC
    ('Mg', 'MgO',   1.0, 2, 24.305),
    ('Ca', 'CaO',   1.0, 2, 40.078),
    ('Na', 'Na2O',  2.0, 1, 22.990),
    ('K',  'K2O',   2.0, 1, 39.098),
)

# Suppress abiotic CO2 -> CH4, SO4 -> HS- and H2O -> H2 reduction.  llnl.dat
# links the C(-4), C(-3), C(-2), C(+2), S(-2)/S(+4) and H(0) species to C(4)/
# S(6)/H2O through O2.  At the very low pe produced by Fe(II)-rich basalt
# dissolution at high pH, full redox equilibrium would (i) convert part of the
# dissolved carbon to methane and (ii) oxidise dissolved Fe(II) to Fe(OH)3 by
# reducing water to H2, diverting Fe from siderite.  Neither reaction proceeds
# abiotically at 45-50 C on these time scales, so both are switched off.
_REDOX_DECOUPLE = """SOLUTION_SPECIES
 H+ + HCO3- + H2O = CH4 + 2 O2
    log_k -300
    -analytic -300 0 0 0 0
 2 H+ + 2 HCO3- + H2O = C2H6 + 3.5 O2
    log_k -300
    -analytic -300 0 0 0 0
 2 H+ + 2 HCO3- = C2H4 + 3 O2
    log_k -300
    -analytic -300 0 0 0 0
 HCO3- + H+ = CO + H2O + 0.5 O2
    log_k -300
    -analytic -300 0 0 0 0
 SO4-2 + H+ = HS- + 2 O2
    log_k -300
    -analytic -300 0 0 0 0
 SO4-2 = SO3-2 + 0.5 O2
    log_k -300
    -analytic -300 0 0 0 0
 H2O = H2 + 0.5 O2
    log_k -300
    -analytic -300 0 0 0 0

"""


def glass_formula_from_xrf(xrf_data):
    """
    [FIX-09] Charge-balanced basaltic-glass formula normalised to one Si,
    using the site's bulk XRF composition as the glass composition.

    Returns (formula_string, molar_mass_g_per_mol, stoichiometry_dict).
    """
    si = xrf_data.get('SiO2', 0.0) / MW_OXIDE['SiO2']
    if si <= 0:
        raise ValueError("SiO2 missing from XRF data")
    stoich = {}
    charge = 4.0  # Si(IV)
    mass = 28.086
    for el, ox, n_cat, val, amass in _GLASS_CATIONS:
        mol = xrf_data.get(ox, 0.0) / MW_OXIDE[ox] * n_cat / si
        stoich[el] = mol
        charge += mol * val
        mass += mol * amass
    o = charge / 2.0
    mass += o * 15.999
    stoich['O'] = o
    formula = "Si" + "".join(f"{el}{stoich[el]:.4f}" for el, *_ in _GLASS_CATIONS) + f"O{o:.4f}"
    return formula, mass, stoich


class PhreeqcEngineMC(PhreeqcEngine):
    """
    Mass-conservative engine.  Same constructor and step() signature as
    PhreeqcEngine; see module docstring for what is different.
    """

    def __init__(self, database_path, xrf_data, reg_params):
        super().__init__(database_path, xrf_data, reg_params)
        self.fixes = reg_params.get('revision_fixes', {}) or {}
        self.mc = reg_params.get('mc_engine', {}) or {}
        self.xrf_data = dict(xrf_data)
        # [FIX-12] Fe counted once: recompute the XRF-derived chemistry with the
        # corrected Fe split (legacy counted Fe2O3(t) + an extra 0.45*Fe).
        if self.fixes.get('fe_single_count', True):
            (self.oxide_moles, self.elem, self.scalers,
             self.mineral_fracs, self.Fe_Mg_ratio) = xrf_to_chemistry(
                 xrf_data, fe_single_count=True,
                 fe3_fraction=self.mc.get('fe3_fraction_of_total_fe', 0.15))
            self._pe0 = _pe_from_xrf(self.elem['Fe2'], self.elem['Fe3'],
                                     reg_params['T_C'])
            self._pe_current = self._pe0
        # [FIX-19] sensitivity hook: scale selected mineral weight fractions and
        # renormalise so that the modes still sum to 100 wt% (R1-3, R1-4)
        _mf = self.mc.get('modal_factors') or {}
        if _mf:
            fr = {m: v * float(_mf.get(m, 1.0)) for m, v in self.mineral_fracs.items()}
            tot = sum(fr.values())
            self.mineral_fracs = {m: v / tot for m, v in fr.items()}
        self.flow_model = self.mc.get('flow_model', 'overpressure_darcy')
        self.hyd = {'V_co2_m3': 0.0, 'V_water_m3': 0.0, 'm_water_t': 0.0,
                    'm_co2_t': 0.0, 't_inj_yr': 0.0, 'vpore_dt_inj': 0.0}
        self.secondary = list(self.mc.get('secondary_phases', DEFAULT_SECONDARY_EQ))
        self.all_secondary = list(dict.fromkeys(
            CARBONATE_MINERALS + CLAY_MINERALS + OTHER_SECONDARY + self.secondary))
        self.batch_census = {'n_steps': 0, 'n_transport': 0,
                             'n_reaction_only': 0, 'n_shifts_total': 0,
                             'n_batch_high_pco2': 0, 'n_batch_large_dt': 0,
                             'n_batch_fallback': 0, 'n_batch_disabled': 0}

    # ------------------------------------------------------------------
    def _rate_params(self):
        """[FIX-13] Palandri & Kharaka (2004) Ea values (legacy table kept for toggling)."""
        if self.fixes.get('pk2004_rate_parameters', True):
            return _PK04_PARAMS_PK2004
        return _PK04_PARAMS

    def _build_rates(self):
        access = float(self.reg_params.get('access_fraction', 0.25))
        boost = float(self.reg_params.get('dissolution_boost', 1.0))
        p_sa = float(self.mc.get('sa_exponent', 0.5))
        single_sa = self.fixes.get('surface_area_single_scaling', True)
        rf = self.mc.get('rate_factors') or {}          # [FIX-19] sensitivity hooks
        dlk = float(self.mc.get('logk_shift', 0.0))
        lines = ["RATES\n"]
        for mineral, pk in self._rate_params().items():
            lka, ea_a, na, lkn, ea_n, lkb, ea_b, nb, A = pk
            lka, lkn = lka + dlk, lkn + dlk
            if lkb != 0.0:
                lkb = lkb + dlk
            # Legacy multiplied A by the weight fraction AND by M0 (which already
            # scales with the fraction), i.e. SA ~ fraction^2.  [FIX-13b]
            frac = 1.0 if single_sa else max(self.mineral_fracs.get(mineral, 1.0), 0.01)
            A_eff = A * frac * access * boost * float(rf.get(mineral, 1.0))
            base = ""
            # base mechanism only with the P&K table (legacy table had no base term
            # in its RATES and positive n_base values)
            if lkb != 0.0 and self.fixes.get('pk2004_rate_parameters', True):
                base = (f" + 10^({lkb:.4f})*EXP(-{ea_b:.2f}*(1/TK-1/298.15))"
                        f"*ACT(\"H+\")^({nb:.4f})")
            lines += [
                f"{mineral}\n-start\n",
                "10 if M <= 0 then goto 200\n",
                f"20 k = 10^({lka:.4f})*EXP(-{ea_a:.2f}*(1/TK-1/298.15))*ACT(\"H+\")^{abs(na):.4f}"
                f" + 10^({lkn:.4f})*EXP(-{ea_n:.2f}*(1/TK-1/298.15)){base}\n",
                f"30 omg = SR(\"{mineral}\")\n",
                "40 if omg > 1 then omg = 1\n",          # primary phases dissolve only
                f"50 area = {A_eff:.6e}*M0*(M/M0)^{p_sa:.4f}\n",
                "60 moles = area*k*(1-omg)*TIME\n",
                "70 if moles > M then moles = M\n",
                "100 save moles\n",
                "200 end\n-end\n",
            ]
        gp = _GLASS_KIN_PARAMS
        # Keep the legacy specific surface per gram of glass: legacy A was per
        # mol of SiO2-equivalent (60.08 g); the new formula unit is heavier.
        A_g = (gp['A_m2_per_mol'] * (self._glass_mw / 60.08) * access * boost
               * float(rf.get('BasaltGlass', 1.0)))
        ea_a = gp['Ea_acid'] / 8.314
        ea_n = gp['Ea_neutral'] / 8.314
        lines += [
            "BasaltGlass\n-start\n",
            "10 if M <= 0 then goto 200\n",
            f"20 k = 10^({gp['log_k_acid']:.4f})*EXP(-{ea_a:.2f}*(1/TK-1/298.15))*ACT(\"H+\")^{gp['n_acid']:.4f}"
            f" + 10^({gp['log_k_neutral']:.4f})*EXP(-{ea_n:.2f}*(1/TK-1/298.15))\n",
            # affinity: "silica" -> (1 - a_SiO2/K_SiO2(am)) (Grambow 1985; Daux et al.
            # 1997): glass dissolution slows as the pore water approaches amorphous-
            # silica saturation.  "none" -> far-from-equilibrium law as submitted.
            ("30 omg = SR(\"SiO2(am)\")\n40 if omg > 1 then omg = 1\n"
             if self.mc.get('glass_affinity', 'silica') == 'silica' else "30 omg = 0\n"),
            f"50 area = {A_g:.6e}*M0*(M/M0)^{p_sa:.4f}\n",
            "60 moles = area*k*(1-omg)*TIME\n",
            "70 if moles > M then moles = M\n",
            "100 save moles\n",
            "200 end\n-end\n\n",
        ]
        return "".join(lines)

    # ------------------------------------------------------------------
    def _porewater_block(self, num):
        rp = self.reg_params
        fe2 = float(self.mc.get('fe2_porewater_mmol', 0.01))   # [FIX-04]
        pH = float(rp.get('initial_pH', 7.5))
        dic = float(rp.get('initial_DIC_mmol', 2.0))
        ion = self.mc.get('charge_balance_ion', 'Cl')
        cl = rp.get('Cl_mmol', 10.0)
        cl_line = f"    Cl        {cl:.4f}" + ("  charge\n" if ion == 'Cl' else "\n")
        ph_line = f"    pH        {pH:.3f}" + ("  charge\n" if ion == 'pH' else "\n")
        return (
            f"SOLUTION {num}  formation water (pre-equilibrated before t = 0)\n"
            f"    temp      {rp['T_C']:.2f}\n"
            + ph_line +
            f"    pe        {self._pe0:.3f}\n"
            "    units     mmol/kgw\n"
            f"    density   {rp.get('brine_density', 1.02):.4f}\n"
            f"    Ca        {rp.get('pw_Ca_mmol', 1.0):.5f}\n"
            f"    Mg        {rp.get('pw_Mg_mmol', 0.5):.5f}\n"
            f"    Na        {rp.get('pw_Na_mmol', 20.0):.5f}\n"
            f"    K         {rp.get('pw_K_mmol', 1.0):.5f}\n"
            f"    Fe(2)     {fe2:.6f}\n"
            f"    Si        {rp.get('pw_Si_mmol', 1.0):.5f}\n"
            f"    Al        {rp.get('pw_Al_mmol_base', rp.get('pw_Al_mmol', 0.02)):.6f}\n"
            f"    C(4)      {dic:.5f}\n"
            f"    S(6)      {rp.get('SO4_mmol', 0.5):.5f}\n"
            + cl_line + "\n"
        )

    def _punch_block(self):
        h = ["Ctot", "CO2aq", "H2O", "Ca", "Mg", "Fe2", "Al", "Si", "Cl", "alk", "SICO2g"]
        body = ['10 PUNCH TOT("C")*TOT("water"), MOL("CO2"), TOT("water"), TOT("Ca"), '
                'TOT("Mg"), TOT("Fe(2)"), TOT("Al"), TOT("Si"), TOT("Cl"), ALK, SI("CO2(g)")']
        for j, m in enumerate(self.all_secondary):
            h.append(f"eq_{m}")
            if m in self.secondary:
                body.append(f'{20 + j} PUNCH EQUI("{m}")')
            else:
                body.append(f'{20 + j} PUNCH 0')
        for j, m in enumerate(PRIMARY_MINERALS):
            h.append(f"kin_{m}")
            body.append(f'{60 + j} PUNCH KIN("{m}")')
        for j, m in enumerate(CARBONATE_MINERALS):
            h.append(f"si_{m}")
            if m == 'Ankerite':
                body.append(f'{80 + j} PUNCH -99')
            else:
                body.append(f'{80 + j} PUNCH SI("{m}")')
        self._punch_headings = h
        return ("SELECTED_OUTPUT 1\n    -reset false\n    -state true\n"
                "    -solution true\n    -step true\n    -pH true\n    -pe true\n"
                "USER_PUNCH 1\n    -headings " + " ".join(h) + "\n    "
                + "\n    ".join(body) + "\n\n")

    # ------------------------------------------------------------------
    def initialize(self):
        # legacy initialise: pore mass, porewater factors, rock inventory
        super().initialize()
        rp = self.reg_params
        N = int(rp.get('transport_n_cells', 10))
        self._n_cells = N
        self._cell_len_m = self._col_len_m / N
        self.F = self.pore_kg / N                     # field kg water per PHREEQC cell

        # [FIX-09] glass: stoichiometric formula and molar mass
        self._glass_formula, self._glass_mw, self._glass_stoich = \
            glass_formula_from_xrf(self.xrf_data)
        # legacy glass "moles" were SiO2-equivalents (60.08 g); keep the same glass
        # MASS fraction and convert to formula units of the new composition
        glass_mass_kg = self.rock_kg * self.mineral_fracs.get('BasaltGlass', 0.0)
        self.m0_primary['BasaltGlass'] = glass_mass_kg * 1000.0 / self._glass_mw
        self.m_primary_current = dict(self.m0_primary)
        self._vm_primary = {m: MOLAR_VOLUME_MC.get(m, 50.0) for m in PRIMARY_MINERALS}
        self._vm_primary['BasaltGlass'] = self._glass_mw / 2.75  # glass density 2.75 g/cm3
        self.reg_params['glass_molar_volume_cm3'] = self._vm_primary['BasaltGlass']
        self.m0_kgw = {m: self.m0_primary[m] / self.pore_kg for m in PRIMARY_MINERALS}

        # [FIX-17] modelled domain.  By default the whole column; with
        # CONFIG["mc_engine"]["domain_length_m"] only its first metres are modelled
        # (same cross-section, same rock per kg of pore water), which lets the
        # injection-driven runs resolve a plume that advances < 1 m in 5 yr.
        L_dom = float(self.mc.get('domain_length_m') or self._col_len_m)
        self.domain_fraction = min(L_dom / self._col_len_m, 1.0)
        self.pore_kg_domain = self.pore_kg * self.domain_fraction
        self._cell_len_m = L_dom / N
        self.F = self.pore_kg_domain / N
        self.m0_primary = {m: self.m0_kgw[m] * self.pore_kg_domain for m in PRIMARY_MINERALS}
        self.m_primary_current = dict(self.m0_primary)

        # fresh IPhreeqc instance: nothing from the legacy engine is reused
        self.iph = IPhreeqc()
        self.iph.load_database(self.database_path)

        kin = f"KINETICS 1\n"
        for m in PRIMARY_MINERALS:
            mk = self.m0_kgw[m]
            if mk <= 0:
                continue
            kin += f"{m}\n"
            if m == 'BasaltGlass':
                kin += f"    -formula {self._glass_formula} 1\n"
            kin += f"    -m  {mk:.8e}\n    -m0 {mk:.8e}\n    -tol 1e-8\n"
        eq = "EQUILIBRIUM_PHASES 1\n" + "".join(f"    {m:16s} 0.0  0.0\n" for m in self.secondary)
        # user-defined ankerite (log K copied from dolomite; no database basis) is
        # loaded only if it is explicitly requested in CONFIG["mc_engine"]
        extra_phases = _ANKERITE_PHASE if 'Ankerite' in self.secondary else ""
        init = (
            _REDOX_DECOUPLE
            + extra_phases
            + self._build_rates()
            + "KNOBS\n    -iterations 300\n    -step_size 10\n    -pe_step_size 5\n\n"
            + self._porewater_block(1)
            + eq + "\nSAVE solution 1\nSAVE equilibrium_phases 1\nEND\n"
            + f"COPY solution 1 0\n"
            + (f"COPY solution 1 2-{N}\nCOPY equilibrium_phases 1 2-{N}\n" if N > 1 else "")
            + "END\n"
            + kin + "END\n"
            + (f"COPY kinetics 1 2-{N}\nEND\n" if N > 1 else "")
        )
        self._run(init, "initial equilibration")
        self._run(self._punch_block() + "END\n", "selected output definition")

        # carbon in one cell volume of inflowing formation water (solution 0)
        rows = self._run_rows("USE solution 0\nREACTION 901\n    CO2 1\n    0 moles\nEND\n",
                              "solution 0 query")
        r0 = rows[-1]
        self._C_inflow_per_cell = r0['Ctot']
        self._inflow_pH = r0['pH']

        # initial column state: a zero-length reaction-only step punches all cells
        rows = self._transport(1, 1.0e-6, forward=False)
        st = self._final_state(rows)
        self._C_aq0 = sum(r['Ctot'] for r in st) * self.F
        self._C_min0 = self._mineral_carbon(st)
        self._eq0 = {m: np.mean([r[f'eq_{m}'] for r in st]) for m in self.all_secondary}
        self._state = st
        self._cellN_C = st[-1]['Ctot']

        # ledger
        self._C_injected = 0.0
        self._C_dissolved = 0.0
        self._C_free = 0.0
        self._C_inflow = 0.0
        self._C_export = 0.0
        self._shift_acc = 0.0
        self._phi_frac = rp.get('initial_porosity', 10.0) / 100.0
        self._porosity_current = rp.get('initial_porosity', 10.0)
        self.carbon_log = []
        self.solver_log = []

        # salinity for the Spycher/Duan-Sun salting-out term: Cl after charge balance
        self.reg_params['Cl_mmol_eff'] = float(np.mean([r['Cl'] for r in st])) * 1000.0
        print(f"  [ENGINE MC] {ENGINE_NAME}")
        print(f"  [ENGINE MC] cells={N} | dx={self._cell_len_m:.2f} m | F={self.F:.3e} kg/cell "
              f"| glass={self._glass_formula} (MW {self._glass_mw:.1f})")
        print(f"  [ENGINE MC] secondary EQ phases: {self.secondary}")
        print(f"  [ENGINE MC] t=0 pH={st[0]['pH']:.3f} | C_aq0={self._C_aq0:.3e} mol "
              f"| C_min0={self._C_min0:.3e} mol | inflow C={self._C_inflow_per_cell*1e3:.3f} mmol/cell")
        return True

    # ------------------------------------------------------------------
    def _run(self, text, what):
        try:
            self.iph.run_string(text)
        except Exception as exc:  # fail loudly: no silent fall-back to stale state
            raise RuntimeError(f"[ENGINE MC] PHREEQC failed during {what} "
                               f"(step {self._step}):\n{exc}") from None

    def _run_rows(self, text, what):
        self._run(text, what)
        arr = self.iph.get_selected_output_array()
        if not arr or len(arr) < 2:
            return []
        hdr = [str(h) for h in arr[0]]
        return [dict(zip(hdr, r)) for r in arr[1:]]

    def _transport(self, n_shifts, dt_shift_s, forward=True):
        N = self._n_cells
        block = (
            "TRANSPORT\n"
            f"    -cells {N}\n"
            f"    -shifts {int(n_shifts)}\n"
            f"    -time_step {dt_shift_s:.6e}\n"
            f"    -lengths {self._cell_len_m:.6f}\n"
            f"    -dispersivities {self._disp_m:.6f}\n"
            f"    -flow_direction {'forward' if forward else 'diffusion_only'}\n"
            "    -boundary_conditions flux flux\n"
            f"    -punch_cells 1-{N}\n"
            "    -punch_frequency 1\n"
            "    -print_frequency 1000000\n"
            "    -warnings false\n"
            "END\n")
        return self._run_rows(block, "TRANSPORT")

    @staticmethod
    def _rows_at(rows, step):
        return sorted((r for r in rows if str(r['state']).strip() == 'transp'
                       and int(r['step']) == step), key=lambda r: int(r['soln']))

    def _final_state(self, rows):
        steps = sorted({int(r['step']) for r in rows if str(r['state']).strip() == 'transp'})
        return self._rows_at(rows, steps[-1])

    def _mineral_carbon(self, state):
        c = 0.0
        for r in state:
            for m in CARBONATE_MINERALS:
                if m in self.secondary:
                    c += r[f'eq_{m}'] * CO2_PER_CARBONATE.get(m, 1.0)
        return c * self.F

    # ------------------------------------------------------------------
    def _velocity(self, pressure_bar):
        """Darcy velocity from the overpressure (legacy formula) with K-C permeability."""
        rp = self.reg_params
        k0 = rp.get('permeability_mD', None)
        if k0 is None:
            return self._vel_m_yr
        P_hydro = rp.get('injection_depth_m', 860.0) * 0.1
        dP = max(pressure_bar - P_hydro, 0.1)
        phi0 = rp.get('initial_porosity', 10.0) / 100.0
        ratio = max(self._phi_frac / phi0, 0.1)
        k = max(k0 * ratio ** 3, 0.001)
        v = darcy_velocity_m_yr(k, dP, rp.get('transport_col_len_m', 50.0))
        return max(min(v, 200.0), 0.01)

    def _velocity_injection(self, pressure_bar, fluid_rates):
        """
        [FIX-17] Darcy flux (m/yr) set by the injected fluid volume:
            q = (m_CO2 / rho_CO2(P, T) + m_water / rho_w) / A,   A = V_bulk / L
        plus an optional ambient (regional) flux.  The column cross-section A
        follows from the reactive rock volume and the column length.
        """
        q_co2, q_w = fluid_rates if fluid_rates is not None else (0.0, 0.0)
        rho_c = co2_density_kg_m3(self.reg_params['T_C'], pressure_bar)
        rho_w = float(self.reg_params.get('brine_density', 1.02)) * 1000.0
        V_dot = (q_co2 * 1000.0 / rho_c + q_w * 1000.0 / rho_w) * 365.25   # m3/yr
        phi0 = self.reg_params.get('initial_porosity', 10.0) / 100.0
        V_bulk = self.pore_kg / (1000.0 * phi0)
        A = V_bulk / self._col_len_m
        return V_dot / A + float(self.mc.get('ambient_darcy_m_yr', 0.0))

    def hydraulics_summary(self, sched=None):
        """[FIX-20] Per-scenario hydraulic quantities (Reviewer 1, comment 9)."""
        h = dict(self.hyd)
        N = self._n_cells
        cell_pv_m3 = self.F / 1000.0                      # pore water per cell, m3
        h['pore_volumes_displaced'] = self.batch_census['n_shifts_total'] / N
        h['native_water_through_m3'] = self.batch_census['n_shifts_total'] * cell_pv_m3
        h['column_pore_volume_m3'] = N * cell_pv_m3
        h['V_total_injected_m3'] = h['V_co2_m3'] + h['V_water_m3']
        h['co2_to_water_volume'] = (h['V_co2_m3'] / h['V_water_m3']) if h['V_water_m3'] > 0 else float('inf')
        vbar = h['vpore_dt_inj'] / h['t_inj_yr'] if h['t_inj_yr'] > 0 else float('nan')
        h['mean_pore_velocity_during_injection_m_yr'] = vbar
        h['residence_time_column_d'] = (self._col_len_m / vbar * 365.25) if vbar and vbar > 0 else float('inf')
        h['flow_model'] = self.flow_model
        return h

    def _solubility_mol_kgw(self, pressure_bar):
        """[FIX-11] Spycher et al. (2003) + Duan & Sun salting-out at the actual pressure."""
        T = self.reg_params['T_C']
        P = max(pressure_bar, 1.0)
        try:
            return co2_solubility_mol_kgw(T, P, m_NaCl=self._m_nacl_mol, model="spycher")
        except Exception:
            return co2_solubility_mol_kgw(T, P, m_NaCl=0.0, model="henry")

    # ------------------------------------------------------------------
    def step(self, pressure_bar, co2_mol_this_step, dt_years,
             log_pco2_override=None, t_yr=0.0, wag_boost=1.0, fluid_rates=None):
        """
        One coupled step.  ``log_pco2_override`` and ``wag_boost`` are accepted for
        signature compatibility and IGNORED: CO2 enters only as injected mass.
        """
        self._step += 1
        self._pressure_bar = pressure_bar
        N = self._n_cells
        dt_s = dt_years * _SEC_PER_YR

        # 1) velocity and shifts ----------------------------------------------
        if self.flow_model == 'injection_driven':
            self._vel_m_yr = self._velocity_injection(pressure_bar, fluid_rates)
        else:
            self._vel_m_yr = self._velocity(pressure_bar)
        # [FIX-20] hydraulic bookkeeping
        if fluid_rates is not None:
            q_co2, q_w = fluid_rates
            rho_c = co2_density_kg_m3(self.reg_params['T_C'], pressure_bar)
            rho_w = float(self.reg_params.get('brine_density', 1.02)) * 1000.0
            self.hyd['m_co2_t'] += q_co2 * 365.25 * dt_years
            self.hyd['m_water_t'] += q_w * 365.25 * dt_years
            self.hyd['V_co2_m3'] += q_co2 * 1000.0 / rho_c * 365.25 * dt_years
            self.hyd['V_water_m3'] += q_w * 1000.0 / rho_w * 365.25 * dt_years
            if q_co2 > 0 or q_w > 0:
                self.hyd['t_inj_yr'] += dt_years
                self.hyd['vpore_dt_inj'] += self._vel_m_yr / max(self._phi_frac, 1e-6) * dt_years
        v_pore = self._vel_m_yr / max(self._phi_frac, 1e-6)
        self._shift_acc += v_pore * dt_years / self._cell_len_m
        n_shift = int(math.floor(self._shift_acc + 1e-12))
        self._shift_acc -= n_shift

        # 2) CO2 injection by mass into cell 1, limited by solubility -------------
        self._C_injected += co2_mol_this_step
        self._C_free += co2_mol_this_step
        S = self._solubility_mol_kgw(pressure_bar)
        co2aq_1 = self._state[0]['CO2aq'] if self._state else 0.0
        capacity = max(S - co2aq_1, 0.0) * self.F
        add = min(self._C_free, capacity)
        if add > 0.0:
            self._run(f"USE solution 1\nREACTION 900\n    CO2 1\n    {add / self.F:.10e} moles\n"
                      "SAVE solution 1\nEND\n", "CO2 addition")
            self._C_free -= add
            self._C_dissolved += add

        # 3) transport + reaction ------------------------------------------------
        self.batch_census['n_steps'] += 1
        if n_shift >= 1:
            rows = self._transport(n_shift, dt_s / n_shift, forward=True)
            self.batch_census['n_transport'] += 1
            self.batch_census['n_shifts_total'] += n_shift
            mode = 'transport'
            # outflow: cell N content before each shift k = state after shift k-1
            out = self._cellN_C
            for k in range(1, n_shift):
                out += self._rows_at(rows, k)[-1]['Ctot']
            self._C_export += out * self.F
            self._C_inflow += n_shift * self._C_inflow_per_cell * self.F
        else:
            rows = self._transport(1, dt_s, forward=False)
            self.batch_census['n_reaction_only'] += 1
            mode = 'reaction_only'
        self._solver_reason_this_step = mode
        st = self._final_state(rows)
        self._state = st
        self._cellN_C = st[-1]['Ctot']
        self.solver_log.append({'step': self._step, 'dt_yr': dt_years, 'mode': mode,
                                'n_shifts': n_shift, 'v_darcy_m_yr': self._vel_m_yr})

        # 4) carbon ledger ---------------------------------------------------------
        C_aq = sum(r['Ctot'] for r in st) * self.F
        C_min = self._mineral_carbon(st)
        C_in = self._C_aq0 + self._C_min0 + self._C_injected + self._C_inflow
        C_out = C_aq + C_min + self._C_free + self._C_export
        resid = C_in - C_out
        closure = abs(resid) / max(C_in, 1.0) * 100.0
        C_min_net = C_min - self._C_min0
        self.carbon_log.append({
            'step': self._step, 'dt_yr': dt_years, 't_yr': t_yr,
            'C_injected': self._C_injected, 'C_initial': self._C_aq0 + self._C_min0,
            'C_inflow': self._C_inflow, 'C_aqueous': C_aq, 'C_mineral': C_min,
            'C_mineral_net': C_min_net, 'C_free': self._C_free,
            'C_export': self._C_export, 'C_residual': resid,
            'closure_err_pct': closure, 'solver_mode': mode, 'n_shifts': n_shift})

        # 5) mineral inventories, porosity (bulk-volume basis) ---------------------
        mean = lambda key: float(np.mean([r[key] for r in st]))
        kin_mean = {m: mean(f'kin_{m}') for m in PRIMARY_MINERALS}
        for m in PRIMARY_MINERALS:
            self.m_primary_current[m] = kin_mean[m] * self.pore_kg_domain
            self.cum_dissolved[m] = max(self.m0_primary[m] - self.m_primary_current[m], 0.0)
        eq_mean = {m: mean(f'eq_{m}') for m in self.all_secondary}
        dV_cm3_kgw = (sum((self.m0_kgw[m] - kin_mean[m]) * self._vm_primary[m]
                          for m in PRIMARY_MINERALS)
                      - sum((eq_mean[m] - self._eq0[m]) * MOLAR_VOLUME_MC.get(m, 50.0)
                            for m in self.all_secondary))
        phi0 = self.reg_params.get('initial_porosity', 10.0) / 100.0
        # 1 kgw occupies phi0 * V_bulk; dV (m3/kgw) / V_bulk-per-kgw
        self._phi_frac = max(phi0 + dV_cm3_kgw * 1e-6 * 1000.0 * phi0, 1e-4)
        self._porosity_current = self._phi_frac * 100.0

        # 6) result dictionary (legacy keys + additions) ----------------------------
        c1 = st[0]
        pH1 = float(c1['pH'])
        self._last_pH = pH1
        self._pe_current = float(np.clip(mean('pe'), -15, 20))
        res = {
            'pH': pH1,
            'pH_mean': mean('pH'),
            'pH_min_cells': float(min(r['pH'] for r in st)),
            'pH_outlet': float(st[-1]['pH']),
            'pe': self._pe_current,
            'alkalinity': mean('alk'),
            'co2_seq': C_min_net,
            'co2_injected': self._C_injected,
            'co2_excess_mol': self._C_free,
            'volume_L': 1.0,
            'source': 'phreeqc_mc_R2',
            'co2_solubility': S * 1000.0,
            'pressure_bar': pressure_bar,
            'vel_m_yr': self._vel_m_yr,
            'pore_vel_m_yr': v_pore,
            'n_shifts': n_shift,
            'log_pco2_cell1': float(c1['SICO2g']),
            'spatial_pH': [float(r['pH']) for r in st],
            'spatial_DIC': [float(r['Ctot']) / max(float(r['H2O']), 1e-9) * 1000.0 for r in st],
            'C_injected_mol': self._C_injected,
            'C_initial_mol': self._C_aq0 + self._C_min0,
            'C_inflow_mol': self._C_inflow,
            'C_aqueous_mol': C_aq,
            'C_mineral_mol': C_min,
            'C_mineral_net_mol': C_min_net,
            'C_free_mol': self._C_free,
            'C_export_mol': self._C_export,
            'C_residual_mol': resid,
            'closure_err_pct': closure,
            'total_DIC_mmol': C_aq / self.pore_kg_domain * 1000.0,
            'porosity_engine_pct': self._porosity_current,
        }
        for m in self.all_secondary:
            v = eq_mean.get(m, 0.0)
            res[f'precip_{m}'] = v
            res[f'eq_inv_{m}'] = v
        for m in PRIMARY_MINERALS:
            res[f'dissolved_{m}'] = self.cum_dissolved[m]
        for m in CARBONATE_MINERALS:
            vals = [r[f'si_{m}'] for r in st if r[f'si_{m}'] > -98]
            res[f'SI_{m}'] = float(np.mean(vals)) if vals else None
        if self._step <= 3 or self._step % 100 == 0:
            print(f"  [MC step {self._step:4d}] t={t_yr:.3f}yr v={self._vel_m_yr:.2f}m/yr "
                  f"shifts={n_shift} pH1={pH1:.2f} add={add:.3e} mol "
                  f"Cmin_net={C_min_net:.3e} closure={closure:.1e}%", flush=True)
        return res

    @property
    def _m_nacl_mol(self):
        return max(self.reg_params.get('Cl_mmol_eff', self.reg_params.get('Cl_mmol', 10.0)), 0.0) / 1000.0

    def close(self):
        self.iph = None
