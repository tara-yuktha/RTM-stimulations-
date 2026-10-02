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


