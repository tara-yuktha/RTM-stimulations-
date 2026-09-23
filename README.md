# Reactive Transport Modelling of CO₂ Mineralisation in Basalt Reservoirs

This repository contains the Python-IPhreeqc Reactive Transport Model (RTM) accompanying the manuscript:  
**"Comparative Evaluation of Injection Strategies for CO₂ Mineralisation in Basalt Formations: Deccan Trap (Mhow) vs. Columbia River Basalt Group (Wallula)"** (*Energy & Fuels*).

---

## 1. Repository Structure & Core Files

To replicate the study, the repository contains the following core files:

| File | Description |
| :--- | :--- |
| `run_rtm.py` | Main reactive transport simulation driver, scheduling logic, and parameter configuration (`CONFIG`). |
| `phreeqc_engine.py` | Core geochemical engine interfacing with IPhreeqc. Solves kinetic dissolution, equilibrium phase precipitation, dynamic Darcy velocity, and audited carbon mass balance. |
| `co2_solubility.py` | Thermodynamic mutual solubility module implementing Spycher et al. (2003) and Duan & Sun (2003) across temperature, pressure, and salinity. |
| `run_batch.py` | Production batch script running all six primary scenarios (Mhow and CRBG across Single-burst, Continuous, and Pulsed WAG modes). Generates time-series Excel workbooks and scenario figures. |
| `run_grid_convergence.py` | Spatial discretisation (10, 20, 50 cells) and temporal discretisation (400, 800, 1,600 steps) convergence study across all site/mode configurations. |
| `run_sensitivity_sweep.py` | Multi-core parallel sensitivity sweep evaluating basalt dissolution capacity (`carb_cap_base`) and residual porewater $p\text{CO}_2$ floor. |
| `verify_darcy.py` | Unit-by-unit Darcy velocity derivation, analytical verification, and dimensional conversion script. |
| `RTM_plots.py` | Publication figure generation script producing all cross-comparative multi-panel plots. |
| `llnl.dat` | Thermodynamic database file from Lawrence Livermore National Laboratory used by PHREEQC. |
| `requirements.txt` | Python dependencies. |

---

## 2. Prerequisites & Installation

### Python Environment
Recommended Python version: **3.9 – 3.11**.

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### IPhreeqc Shared Library
The geochemical engine relies on `IPhreeqc` (via `phreeqpy`).  
- **Windows**: Install the official USGS IPhreeqc COM or DLL, or ensure `IPhreeqc.dll` is on your system `PATH`.
- **Linux / macOS**: Install `libiphreeqc.so` / `libiphreeqc.dylib` via your package manager or compile from USGS source.

---

## 3. Usage & Reproducing Manuscript Results

### A. Production Batch Runs (Tables 5–7 and Figures 1–8)
To run the primary six scenarios (Mhow & CRBG across Single-burst, Continuous, and Pulsed WAG modes) and export all detailed time-series Excel sheets and scenario figures:
```bash
python run_batch.py
```
*Outputs are saved to `RTM_Output/RTM_<Region>_<Mode>/`.*

### B. Cross-Comparative Publication Figures (Figures 9–14)
To generate the multi-panel comparison plots comparing Mhow and CRBG across all injection strategies:
```bash
python RTM_plots.py
```
*Figures are saved to `RTM_Output/RTM_plots/`.*

### C. Grid and Time-Step Independence Study
To run the spatial (10, 20, 50 cells) and temporal (400, 800, 1,600 steps) discretisation study:
```bash
python run_grid_convergence.py --site ALL --workers 6
```
*Outputs are saved to `RTM_Output/results_grid_convergence.csv`.*

### D. Multi-Parameter Sensitivity Analysis
To run the parallel sensitivity sweeps across reactive capacity factors and residual $p\text{CO}_2$ floors:
```bash
python run_sensitivity_sweep.py --workers 6
```
*Generates summary workbooks and `sensitivity_ranking_heatmap.png` in `RTM_Output/sensitivity_sweep/`.*

### E. Darcy Velocity Verification
To verify the unit-by-unit Darcy velocity derivations and conversion factors:
```bash
python verify_darcy.py
```

---

## 4. Key Modeling Assumptions
- **Operator Splitting**: The model implements a sequential non-iterative approach (SNIA) within IPhreeqc, first-order accurate in time ($\mathcal{O}(\Delta t)$).
- **Near-Wellbore Reaction Mode**: High-$p\text{CO}_2$ injection steps ($\log p\text{CO}_2 > -2.0$) are solved via a well-mixed batch reactor to avoid Newton–Raphson divergence from steep gas-phase boundary gradients; the full 1D multi-cell `TRANSPORT` module is active during all monitoring and flush periods.
- **Thermodynamics & Kinetics**: Primary mineral dissolution follows Palandri & Kharaka (2004); natural basalt glass dissolution follows Wolff-Boenisch et al. (2006); secondary carbonate precipitation is constrained by saturation state and bulk cation availability.
