"""
co2_solubility.py
=================
Thermodynamically consistent CO2 solubility in water and NaCl brine.

WHY THIS MODULE EXISTS
----------------------
The originally submitted model used a linear Henry's-law expression

    sol_mmol_kgw = K(T) * P_bar ,   K(40 degC) = 0.580 mmol kgw^-1 bar^-1

which returns ~5 mmol kgw^-1 at 8 bar / 45 degC. The correct Henry coefficient
for CO2 in pure water near 40 degC is ~23 mmol kgw^-1 bar^-1, i.e. the original
constant is low by a factor of ~40, and the docstring anchor quoted from
Duan & Sun (2003) ("100 bar, 40 degC -> 58 mmol/kgw") is low by a factor of ~20
(the tabulated value is ~1.1-1.2 mol kgw^-1).

This module replaces that expression with the Spycher, Pruess & Ennis-King
(2003) non-iterative CO2-H2O equilibrium model, extended with the Duan & Sun
(2003) salting-out activity coefficient for NaCl brines. Both are standard,
citable formulations valid over the pressure-temperature range relevant to
basalt-hosted CO2 storage (12-100 degC, 1-600 bar).

REFERENCES
----------
Spycher, N., Pruess, K. & Ennis-King, J. (2003) CO2-H2O mixtures in the
    geological sequestration of CO2. I. Assessment and calculation of mutual
    solubilities from 12 to 100 degC and up to 600 bar.
    Geochim. Cosmochim. Acta 67, 3015-3031.
Spycher, N. & Pruess, K. (2005) CO2-H2O mixtures in the geological
    sequestration of CO2. II. Partitioning in chloride brines at 12-100 degC
    and up to 600 bar. Geochim. Cosmochim. Acta 69, 3309-3320.
Duan, Z. & Sun, R. (2003) An improved model calculating CO2 solubility in pure
    water and aqueous NaCl solutions from 273 to 533 K and from 0 to 2000 bar.
    Chem. Geol. 193, 257-271.

USAGE
-----
    from co2_solubility import co2_solubility_mol_kgw, co2_solubility_mmol_kgw

    m = co2_solubility_mol_kgw(T_C=45.0, P_bar=88.0, m_NaCl=0.05)

A drop-in replacement for the legacy engine call is provided as
``co2_solubility_mmol_kgw(T_C, P_bar, m_NaCl=0.0, model="spycher")`` and the
legacy behaviour is retained under ``model="legacy_henry"`` so that the
submitted results can be reproduced exactly for comparison.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_BAR_CM3 = 83.1447          # cm^3 bar K^-1 mol^-1
M_H2O_INV = 55.508           # mol H2O per kg H2O

# Spycher et al. (2003) Redlich-Kwong parameters
_A_CO2_0 = 7.54e7            # bar cm^6 K^0.5 mol^-2
_A_CO2_T = -4.13e4           # bar cm^6 K^-0.5 mol^-2
_A_H2O_CO2 = 7.89e7          # bar cm^6 K^0.5 mol^-2  (mixture term)
_B_CO2 = 27.80               # cm^3 mol^-1
_B_H2O = 18.18               # cm^3 mol^-1

# Average partial molar volumes (cm^3 mol^-1), Spycher et al. (2003) Table 1
_V_H2O = 18.1
_V_CO2 = 32.6

# Legacy (submitted-version) Henry constant, retained for reproducibility
_LEGACY_K_REF = 0.580        # mmol kgw^-1 bar^-1 at 40 degC  -- KNOWN TO BE WRONG
_LEGACY_T_REF = 313.15       # K
_LEGACY_EA_R = 2300.0        # K


# ---------------------------------------------------------------------------
# Redlich-Kwong equation of state for the CO2-rich phase
# ---------------------------------------------------------------------------
def _rk_molar_volume(T_K: float, P_bar: float, a_mix: float, b_mix: float) -> float:
    """
    Largest real root of the Redlich-Kwong cubic in molar volume V (cm^3/mol):

        V^3 - (RT/P) V^2 - (RTb/P - a/(P sqrt(T)) + b^2) V - a b /(P sqrt(T)) = 0

    Solved analytically (Cardano); the gas-like (largest) root is returned.
    """
    RT_P = R_BAR_CM3 * T_K / P_bar
    a_sq = a_mix / (P_bar * math.sqrt(T_K))

    a2 = -RT_P
    a1 = -(RT_P * b_mix - a_sq + b_mix * b_mix)
    a0 = -a_sq * b_mix

    # Depressed cubic
    p = a1 - a2 * a2 / 3.0
    q = 2.0 * a2 ** 3 / 27.0 - a2 * a1 / 3.0 + a0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3

    if disc >= 0.0:
        sq = math.sqrt(disc)
        u = math.copysign(abs(-q / 2.0 + sq) ** (1.0 / 3.0), -q / 2.0 + sq)
        v = math.copysign(abs(-q / 2.0 - sq) ** (1.0 / 3.0), -q / 2.0 - sq)
        roots = [u + v - a2 / 3.0]
    else:
        r = math.sqrt(-(p ** 3) / 27.0)
        phi = math.acos(max(-1.0, min(1.0, -q / (2.0 * r))))
        m = 2.0 * math.sqrt(-p / 3.0)
        roots = [m * math.cos((phi + 2.0 * math.pi * k) / 3.0) - a2 / 3.0
                 for k in range(3)]

    valid = [x for x in roots if x > b_mix + 1e-9]
    if not valid:
        # Fall back to ideal-gas volume; only reachable at pathological inputs
        return RT_P
    return max(valid)


def _ln_phi(component: str, T_K: float, P_bar: float, V: float,
            a_mix: float, b_mix: float) -> float:
    """
    Natural log of the fugacity coefficient of ``component`` ("CO2" or "H2O")
    in the CO2-rich phase, Spycher et al. (2003) Eq. (7).
    """
    b_k = _B_CO2 if component == "CO2" else _B_H2O
    a_k = (2.0 * (_A_CO2_0 + _A_CO2_T * T_K) if component == "CO2"
           else 2.0 * _A_H2O_CO2)

    t15 = T_K ** 1.5
    ln_ratio = math.log((V + b_mix) / V)

    term1 = math.log(V / (V - b_mix))
    term2 = b_k / (V - b_mix)
    term3 = -a_k / (R_BAR_CM3 * t15 * b_mix) * ln_ratio
    term4 = (a_mix * b_k / (R_BAR_CM3 * t15 * b_mix * b_mix)) * (
        ln_ratio - b_mix / (V + b_mix))
    term5 = -math.log(P_bar * V / (R_BAR_CM3 * T_K))
    return term1 + term2 + term3 + term4 + term5


# ---------------------------------------------------------------------------
# Temperature-dependent equilibrium constants (Spycher et al. 2003, Table 2)
# T in degrees Celsius; reference pressure 1 bar
# ---------------------------------------------------------------------------
def _log_k0_h2o(T_C: float) -> float:
    return (-2.209 + 3.097e-2 * T_C - 1.098e-4 * T_C ** 2 + 2.048e-7 * T_C ** 3)


def _log_k0_co2_gas(T_C: float) -> float:
    return (1.189 + 1.304e-2 * T_C - 5.446e-5 * T_C ** 2)


def _log_k0_co2_liq(T_C: float) -> float:
    return (1.169 + 1.368e-2 * T_C - 5.380e-5 * T_C ** 2)


# ---------------------------------------------------------------------------
# Duan & Sun (2003) salting-out coefficients for NaCl
# ---------------------------------------------------------------------------
_LAMBDA_C = (-0.411370585, 6.07632013e-4, 97.5347708, 0.0, 0.0, 0.0, 0.0,
             -0.0237622469, 0.0170656236, 0.0, 1.41335834e-5)
_ZETA_C = (3.36389723e-4, -1.98298980e-5, 0.0, 0.0, 0.0, 0.0, 0.0,
           2.12220830e-3, -5.24873303e-3, 0.0, 0.0)


def _duan_par(c, T_K: float, P_bar: float) -> float:
    """Duan & Sun (2003) Eq. (7) generic parameter expansion."""
    return (c[0]
            + c[1] * T_K
            + c[2] / T_K
            + c[3] * T_K ** 2
            + c[4] / (630.0 - T_K)
            + c[5] * P_bar
            + c[6] * P_bar * math.log(T_K)
            + c[7] * P_bar / T_K
            + c[8] * P_bar / (630.0 - T_K)
            + c[9] * P_bar ** 2 / (630.0 - T_K) ** 2
            + c[10] * T_K * math.log(P_bar))


def nacl_activity_correction(T_C: float, P_bar: float, m_NaCl: float) -> float:
    """
    Salting-out factor gamma such that  m_CO2(brine) = m_CO2(pure) / gamma.

    Uses the Duan & Sun (2003) Pitzer-type interaction terms for Na-CO2 (lambda)
    and Na-Cl-CO2 (zeta). Returns 1.0 for zero ionic strength.
    """
    if m_NaCl <= 0.0:
        return 1.0
    T_K = T_C + 273.15
    lam = _duan_par(_LAMBDA_C, T_K, max(P_bar, 1.0))
    zeta = _duan_par(_ZETA_C, T_K, max(P_bar, 1.0))
    ln_gamma = 2.0 * lam * m_NaCl + zeta * m_NaCl * m_NaCl
    return math.exp(ln_gamma)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def co2_solubility_mol_kgw(T_C: float, P_bar: float, m_NaCl: float = 0.0,
                           model: str = "spycher") -> float:
    """
    CO2 solubility in the aqueous phase, mol per kg of water.

    Parameters
    ----------
    T_C     : temperature, degrees Celsius (valid 12-100 for "spycher")
    P_bar   : total pressure, bar (valid 1-600 for "spycher")
    m_NaCl  : NaCl molality, mol kgw^-1 (0 = pure water)
    model   : "spycher"       -- Spycher et al. (2003) + Duan & Sun salting-out
              "henry"         -- simple Henry's law with a CORRECT coefficient,
                                 useful only below ~10 bar
              "legacy_henry"  -- the expression used in the submitted manuscript,
                                 retained solely to reproduce those results

    Returns
    -------
    CO2 molality, mol kgw^-1.
    """
    if model == "legacy_henry":
        K = _LEGACY_K_REF * math.exp(_LEGACY_EA_R * (1.0 / _LEGACY_T_REF
                                                     - 1.0 / (T_C + 273.15)))
        return min(K * P_bar, 180.0) / 1000.0

    if model == "henry":
        # Henry constant for CO2 in pure water, Sander (2015) compilation:
        # H_cp(298.15 K) = 3.3e-4 mol m^-3 Pa^-1 = 0.0334 mol kgw^-1 bar^-1
        # d ln H / d(1/T) = 2400 K
        K25 = 0.0334
        K = K25 * math.exp(2400.0 * (1.0 / (T_C + 273.15) - 1.0 / 298.15))
        m_pure = K * P_bar
        return m_pure / nacl_activity_correction(T_C, P_bar, m_NaCl)

    if model != "spycher":
        raise ValueError(f"unknown model {model!r}")

    T_K = T_C + 273.15
    P = max(P_bar, 1.0)

    a_mix = _A_CO2_0 + _A_CO2_T * T_K
    b_mix = _B_CO2
    V = _rk_molar_volume(T_K, P, a_mix, b_mix)

    ln_phi_co2 = _ln_phi("CO2", T_K, P, V, a_mix, b_mix)
    ln_phi_h2o = _ln_phi("H2O", T_K, P, V, a_mix, b_mix)
    phi_co2 = math.exp(ln_phi_co2)
    phi_h2o = math.exp(ln_phi_h2o)

    # CO2 reference state: gas below its saturation pressure, liquid above
    p_sat_co2 = _co2_saturation_pressure(T_C)
    use_liquid = (T_C < 31.05) and (P > p_sat_co2)
    log_k0_co2 = _log_k0_co2_liq(T_C) if use_liquid else _log_k0_co2_gas(T_C)

    k0_h2o = 10.0 ** _log_k0_h2o(T_C)
    k0_co2 = 10.0 ** log_k0_co2

    # Spycher et al. (2003) Eqs. (10)-(13)
    A = (k0_h2o / (phi_h2o * P)) * math.exp((P - 1.0) * _V_H2O
                                            / (R_BAR_CM3 * T_K))
    B = (phi_co2 * P / (M_H2O_INV * k0_co2)) * math.exp(-(P - 1.0) * _V_CO2
                                                        / (R_BAR_CM3 * T_K))

    y_h2o = (1.0 - B) / (1.0 / A - B)
    x_co2 = B * (1.0 - y_h2o)
    x_co2 = min(max(x_co2, 0.0), 0.5)

    m_pure = M_H2O_INV * x_co2 / max(1.0 - x_co2, 1e-12)
    return m_pure / nacl_activity_correction(T_C, P, m_NaCl)


def _co2_saturation_pressure(T_C: float) -> float:
    """CO2 vapour pressure (bar) below the critical point, Span & Wagner (1996)."""
    Tc, Pc = 304.1282, 73.773
    T_K = T_C + 273.15
    if T_K >= Tc:
        return Pc
    tau = 1.0 - T_K / Tc
    a = (-7.0602087, 1.9391218, -1.6463597, -3.2995634)
    t = (1.0, 1.5, 2.0, 4.0)
    s = sum(ai * tau ** ti for ai, ti in zip(a, t))
    return Pc * math.exp(Tc / T_K * s)


def co2_solubility_mmol_kgw(T_C: float, P_bar: float, m_NaCl: float = 0.0,
                            model: str = "spycher", cap_mmol: float = None) -> float:
    """
    Drop-in replacement for the legacy engine function, in mmol kgw^-1.

    ``cap_mmol`` applies an optional numerical ceiling; pass None for none.
    The legacy engine capped at 180 mmol kgw^-1, which is BELOW the true
    solubility at bottom-hole pressure and must not be used with the
    "spycher" model.
    """
    m = co2_solubility_mol_kgw(T_C, P_bar, m_NaCl, model) * 1000.0
    return m if cap_mmol is None else min(m, cap_mmol)


# ---------------------------------------------------------------------------
# Salinity helper
# ---------------------------------------------------------------------------
def nacl_molality_from_cl_mmol(cl_mmol_kgw: float) -> float:
    """
    Convert the model's Cl- concentration (mmol kgw^-1) to an equivalent NaCl
    molality (mol kgw^-1) for the salting-out correction.
    """
    return max(cl_mmol_kgw, 0.0) / 1000.0


# ---------------------------------------------------------------------------
# Self-test / validation table
# ---------------------------------------------------------------------------
_VALIDATION = [
    # (T_C, P_bar, m_NaCl, expected mol/kgw, tolerance, source)
    (25.0, 1.0, 0.0, 0.0334, 0.004, "Sander (2015) Henry compilation"),
    (40.0, 100.0, 0.0, 1.18, 0.12, "Duan & Sun (2003) Table / model"),
    (50.0, 100.0, 0.0, 1.09, 0.11, "Duan & Sun (2003) Table / model"),
    (60.0, 200.0, 0.0, 1.32, 0.16, "Duan & Sun (2003) Table / model"),
    (50.0, 100.0, 1.0, 0.85, 0.12, "Duan & Sun (2003), 1 m NaCl"),
]


def self_test(verbose: bool = True) -> bool:
    ok = True
    if verbose:
        print(f"{'T(C)':>6} {'P(bar)':>8} {'mNaCl':>7} "
              f"{'model':>9} {'expected':>9} {'diff%':>7}  source")
        print("-" * 78)
    for T, P, ms, exp, tol, src in _VALIDATION:
        got = co2_solubility_mol_kgw(T, P, ms)
        d = (got - exp) / exp * 100.0
        good = abs(got - exp) <= tol
        ok &= good
        if verbose:
            print(f"{T:6.1f} {P:8.1f} {ms:7.2f} {got:9.4f} {exp:9.4f} "
                  f"{d:+7.1f}  {src}{'' if good else '   <-- FAIL'}")
    return ok


def comparison_table():
    """Legacy vs corrected solubility at the manuscript's conditions."""
    rows = []
    for label, T, depth in (("Mhow", 45.0, 800.0), ("GRB", 50.0, 860.0)):
        # Hydrostatic gradient from brine density 1020 kg m^-3:
        #   dP/dz = rho g = 1020 * 9.81 = 10007 Pa m^-1 = 0.1001 bar m^-1
        P_hydro = depth * 0.1001          # bar
        P_bh = P_hydro + 8.0              # bottom-hole = hydrostatic + 8 bar
        for tag, P in (("overpressure only (as submitted)", 8.0),
                       ("hydrostatic formation", P_hydro),
                       ("bottom-hole injection", P_bh)):
            rows.append(dict(
                site=label, T_C=T, tag=tag, P_bar=P,
                legacy_mmol=co2_solubility_mmol_kgw(T, P, model="legacy_henry"),
                corrected_mmol=co2_solubility_mmol_kgw(T, P, model="spycher"),
            ))
    return rows


if __name__ == "__main__":
    print("CO2 solubility model validation (Spycher et al. 2003 + Duan & Sun 2003)")
    passed = self_test()
    print(f"\nself-test: {'PASS' if passed else 'FAIL'}\n")

    print("Legacy vs corrected solubility at manuscript conditions")
    print(f"{'site':>6} {'T(C)':>6} {'P(bar)':>8} {'legacy':>10} "
          f"{'corrected':>11} {'ratio':>7}  pressure definition")
    print("-" * 88)
    for r in comparison_table():
        ratio = r["corrected_mmol"] / max(r["legacy_mmol"], 1e-9)
        print(f"{r['site']:>6} {r['T_C']:6.1f} {r['P_bar']:8.1f} "
              f"{r['legacy_mmol']:10.2f} {r['corrected_mmol']:11.2f} "
              f"{ratio:7.1f}  {r['tag']}")
    print("\nUnits: mmol CO2 per kg of water.")
