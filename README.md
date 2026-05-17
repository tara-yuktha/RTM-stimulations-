# 🌋 Reactive Transport Model (RTM) for CO₂ Sequestration in Basalt

A Python-based **Reactive Transport Model (RTM)** that simulates CO₂ mineral trapping in basaltic rock formations using the **PHREEQC geochemical engine**. Designed for studying carbon capture and storage (CCS) in deep basalt aquifers — with support for two geological regions: **CRBG** (Columbia River Basalt Group, USA) and **MHOW** (Deccan Trap Basalt, India).

---

## 📌 Features

- Full geochemical simulation using the PHREEQC engine via `phreeqpy`
- Three CO₂ injection modes: **Single**, **Continuous**, and **Pulsed (WAG)**
- Reactive transport with Darcy flow, dispersion, and multi-cell column transport
- Carbonate and clay mineral precipitation tracking (Calcite, Ankerite, Siderite, Kaolinite, Saponite, etc.)
- Porosity evolution under mineral dissolution and precipitation
- Sensitivity analysis and Monte Carlo uncertainty quantification
- Calibration against Wallula Basalt Pilot Project (White et al., 2020)
- Publication-quality plots with matplotlib

---

## 🗂️ Project Structure

```
├── run_rtm.py          # Main simulation script — configure and run RTM scenarios
├── phreeqc_engine.py   # Core geochemical engine — PHREEQC interface, kinetics, transport
├── RTM_plots.py        # Plotting and visualization — generates publication-quality figures
├── llnl.dat            # PHREEQC thermodynamic database (required, not included)
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- [phreeqpy](https://pypi.org/project/phreeqpy/) — Python bindings for PHREEQC
- numpy
- scipy
- matplotlib
- pandas

Install dependencies:
```bash
pip install phreeqpy numpy scipy matplotlib pandas
```

> **Note:** You also need the PHREEQC thermodynamic database file `llnl.dat`. Download it from the [USGS PHREEQC website](https://www.usgs.gov/software/phreeqc-version-3) and place it in the project folder or set the environment variable:
> ```bash
> export PHREEQC_DB=/path/to/llnl.dat
> ```

---

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RTM-Simulation.git
   cd RTM-Simulation
   ```

2. Install dependencies:
   ```bash
   pip install phreeqpy numpy scipy matplotlib pandas
   ```

3. Place `llnl.dat` in the project directory.

4. Run the simulation:
   ```bash
   python run_rtm.py
   ```

Output files (CSV data + plots) will be saved to `~/RTM_outputs/` by default.

---

## 🔧 Configuration

All simulation parameters are controlled via the `CONFIG` dictionary in `run_rtm.py`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `active_region` | Geological region: `"CRBG"` or `"MHOW"` | `"MHOW"` |
| `mode` | Injection mode: `"single"`, `"continuous"`, `"pulsed"` | `"pulsed"` |
| `T_sim_yr` | Total simulation duration (years) | `5.0` |
| `pressure_bar` | CO₂ injection pressure (bar) | `8` |
| `n_cells` | Number of transport cells in 1D column | `10` |

---

## 🧪 Geological Regions

| Region | Formation | Location | Temperature | Porosity |
|--------|-----------|----------|-------------|---------|
| **CRBG** | Grande Ronde Basalt | Washington State, USA | 50°C | 12% |
| **MHOW** | Deccan Trap Basalt | Madhya Pradesh, India | 45°C | 10% |

---

## 📊 Outputs

The model produces:
- Time-series CSV files for pH, DIC, mineral volumes, porosity, CO₂ saturation
- Spatial profiles along the 1D transport column
- CO₂ trapping efficiency curves
- Mineral assemblage bar charts
- Sensitivity and Monte Carlo analysis plots

---

## 📚 References

- White, S.K., Spane, F.A., Schaef, H.T., Miller, Q.R.S., White, M.D., Horner, J.A., & McGrail, B.P. (2020). Quantification of CO₂ Mineralization at the Wallula Basalt Pilot Project. *Environmental Science & Technology*, 54(22), 14609–14616. https://doi.org/10.1021/acs.est.0c05142

- McGrail, B.P., Schaef, H.T., Spane, F.A., Horner, J.A., Owen, A.T., Cliff, J.B., Qafoku, O., Thompson, C.J., & Sullivan, E.C. (2017). Wallula Basalt Pilot Demonstration Project: Post-injection Results and Conclusions. *Energy Procedia*, 114, 5783–5790. https://doi.org/10.1016/j.egypro.2017.03.1716

- Matter, J.M., Stute, M., Snæbjörnsdóttir, S.Ó., Oelkers, E.H., Gislason, S.R., et al. (2016). Rapid carbon mineralization for permanent disposal of anthropogenic carbon dioxide emissions. *Science*, 352(6291), 1312–1314. https://doi.org/10.1126/science.aad8132

- Nelson, C., Goldberg, D., White, M., & Slagle, A. (2022). Optimizing Injection Strategies for CO₂ Storage and Mineralization in Basalt Through Multiphase Subsurface Reservoir Simulations. *Proceedings of GHGT-16*, 23–24 Oct 2022. https://doi.org/10.2139/ssrn.4280798

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

Developed as part of research on geochemical CO₂ sequestration in Indian and American basalt formations.

Feel free to open an [issue](../../issues) or submit a [pull request](../../pulls) for any bugs or improvements!
