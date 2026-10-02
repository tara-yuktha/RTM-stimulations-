"""
verify_darcy.py
===============
Explicit, unit-by-unit evaluation of the Darcy velocity for both study sites
(Reviewer 2, Comment 5) and a linear-vs-radial comparison (Reviewer 1, C8).

Run:  python verify_darcy.py
"""

from phreeqc_engine import darcy_velocity_m_yr, _MU_BRINE_PA_S

SITES = [
    # label, k (mD), depth (m), overpressure (bar), column length (m), porosity
    ("Mhow (Deccan Trap)",     30.0,  800.0, 8.0, 50.0, 0.10),
    ("GRB (Columbia River)",  200.0,  860.0, 8.0, 50.0, 0.12),
]

MANUSCRIPT_REPORTED = {"Mhow (Deccan Trap)": 2.076, "GRB (Columbia River)": 13.84}


def worked_example(label, k_mD, depth_m, dP_bar, L_m, phi):
    v, s = darcy_velocity_m_yr(k_mD, dP_bar, L_m, trace=True,
                               porosity_frac=phi)
    print(f"\n{label}")
    print("-" * len(label))
    print(f"  Step 1  permeability          k  = {k_mD:g} mD")
    print(f"                                   = {k_mD:g} x 9.869e-16 m^2 mD^-1")
    print(f"                                   = {s['k_m2']:.4e} m^2")
    print(f"  Step 2  pressure differential dP = {dP_bar:g} bar")
    print(f"                                   = {dP_bar:g} x 1e5 Pa bar^-1")
    print(f"                                   = {s['delta_P_Pa']:.4e} Pa")
    print(f"  Step 3  column length         L  = {L_m:g} m")
    print(f"  Step 4  pressure gradient  dP/L  = {s['delta_P_Pa']:.4e} Pa / {L_m:g} m")
    print(f"                                   = {s['pressure_gradient_Pa_per_m']:.4e} Pa m^-1")
    print(f"  Step 5  brine viscosity       mu = {s['mu_Pa_s']:.2e} Pa s")
    print(f"  Step 6  mobility            k/mu = {s['mobility_m2_per_Pa_s']:.4e} m^2 Pa^-1 s^-1")
    print(f"  Step 7  Darcy velocity      v_D  = (k/mu)(dP/L)")
    print(f"                                   = {s['mobility_m2_per_Pa_s']:.4e} "
          f"x {s['pressure_gradient_Pa_per_m']:.4e}")
    print(f"                                   = {s['v_m_per_s']:.4e} m s^-1")
    print(f"  Step 8  unit conversion          = {s['v_m_per_s']:.4e} m s^-1 "
          f"x {s['seconds_per_year']:.5e} s yr^-1")
    print(f"                             v_D   = {s['v_m_per_yr']:.3f} m yr^-1")
    print(f"  Step 9  interstitial velocity    = v_D / phi = "
          f"{s['v_m_per_yr']:.3f} / {phi:g}")
    print(f"                             v_p   = {s['pore_velocity_m_per_yr']:.2f} m yr^-1")
    print(f"  Step 10 residence time over L    = L / v_p = "
          f"{L_m / s['pore_velocity_m_per_yr']:.4f} yr "
          f"= {L_m / s['pore_velocity_m_per_yr'] * 365.25:.1f} d")

    rep = MANUSCRIPT_REPORTED.get(label)
    if rep:
        print(f"\n  Value reported in the submitted manuscript : {rep:.3f} m yr^-1")
        print(f"  Value from the expression as written       : {v:.3f} m yr^-1")
        print(f"  Ratio                                      : {v / rep:.2f}")
        print(f"  The ratio is {v/rep:.1f}, i.e. the reported figure corresponds to")
        print(f"  the same expression evaluated with seconds-per-month")
        print(f"  (2.6297e6 s) instead of seconds-per-year (3.1557e7 s).")

    v_rad = darcy_velocity_m_yr(k_mD, dP_bar, L_m, geometry="radial",
                                r_well_m=0.108)
    print(f"\n  Radial geometry (r_well = 0.108 m, r_out = {L_m:g} m),")
    print(f"  local specific discharge at the geometric-mean radius")
    print(f"  r* = sqrt(r_well * r_out) = {(0.108 * L_m) ** 0.5:.2f} m :")
    print(f"                             v_D(r*) = {v_rad:.2f} m yr^-1  "
          f"({v_rad / v:.2f} x the linear value)")
    return v, v_rad


if __name__ == "__main__":
    print("=" * 78)
    print("DARCY VELOCITY - EXPLICIT CALCULATION WITH UNITS AT EACH STEP")
    print("=" * 78)
    print(f"Brine dynamic viscosity used throughout: mu = {_MU_BRINE_PA_S:.1e} Pa s")
    print("Conversion: 1 mD = 9.869e-16 m^2 ; 1 bar = 1e5 Pa ; "
          "1 yr = 365.25 x 86400 s = 3.15576e7 s")
    for row in SITES:
        worked_example(*row)
    print("\n" + "=" * 78)
