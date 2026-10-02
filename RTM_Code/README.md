# Reactive transport model and results — manuscript ef-2026-03834z (revised)

**Injection strategy and CO₂ mineral trapping in continental flood basalts: a reactive transport modelling study**
T. Samyuktha S, S. Adak, J. Mallik — Energy & Fuels (revision, 2026)

This archive contains the corrected one-dimensional PHREEQC reactive transport model, every result file used in the revised manuscript and Supporting Information (SI), and the scripts that produce the revised tables and figures.

## Requirements
- Python ≥ 3.10
- `pip install -r requirements.txt` (PhreeqPy 0.6.0, which bundles IPhreeqc 3.7.3; numpy, scipy, pandas, matplotlib, openpyxl)
- Thermodynamic database: `llnl.dat` (included)

## Structure
| Path | Content |
|---|---|
| `run_rtm.py` | Configuration (`CONFIG`), site parameters, injection schedules, pressure history, time stepping |
| `phreeqc_engine_mc.py` | Corrected mass-conservative PHREEQC engine: persistent state, CO₂ added as mass, carbon ledger, equilibrium secondary phases, two flow closures |
| `phreeqc_engine.py` | Earlier engine; kept for the XRF-to-assemblage conversion, rate parameters and ablation. Not used for production results |
| `co2_solubility.py` | Spycher et al. (2003) CO₂ solubility with salting-out; CO₂ density |
| `verify_darcy.py` | Worked Darcy calculation with units (SI Text S4) |
| `run_sensitivity_matrix.py` | Definition and execution of every sensitivity configuration (A0, A1, R_, S_ families) |
| `round2_results/analysis/` | Harness, 20-yr runs, pressure-decay runs, time-series regeneration, table and figure scripts |
| `round2_results/*.jsonl, *.json, *.csv` | Result files (see manifest); copy `matrix_runs.jsonl` and `matrix_summary.csv` here after step 1 |
| `round2_results/figures/` | Figures as used in the revised manuscript |

The `revision_fixes` dictionary in `run_rtm.py` (FIX-00 … FIX-26) documents each correction relative to the model of the first submission and allows each one to be switched off for comparison.

## Reproducing the results
Runtime is about 2 minutes per simulation on one core (each case is run together with its no-injection control).

```bash
# 1. Sensitivity matrix: production + control runs     ->  RTM_Output/sensitivity_matrix/matrix_runs.jsonl, matrix_summary.csv
python run_sensitivity_matrix.py --workers 4          # --list shows the configurations; --only A1 A0 runs a subset
# 2. 20-yr runs (uniform time grid)                    ->  ledger_long.json
python round2_results/analysis/run_long.py
# 3. Post-injection pressure-decay sensitivity         ->  round2_results/tau_results.jsonl
python round2_results/analysis/gen_tau.py
# 4. Time series for Figures 4-15 (cached, ~1.5 h)     ->  round2_results/cache/*.pkl
python round2_results/analysis/gen_ms_series.py        # injection-driven runs
python round2_results/analysis/gen_ms_series3.py       # Mhow rate-factor (0.85) sensitivity
python round2_results/analysis/gen_ms_series4.py       # overpressure-driven runs
# 5. Summary numbers, tables and figures
python round2_results/analysis/compile2.py             # -> compiled2.json (regime and summary values)
python round2_results/analysis/compile_numbers.py      # -> base_numbers.json (base-case ledgers and inventories)
python round2_results/analysis/make_figs.py            # Figures 4-13
python round2_results/analysis/make_fig14.py           # Figures 14 and 15
python round2_results/analysis/plot_regime_big.py      # Figure 16
python round2_results/analysis/fig_concept.py          # Figure 3
python verify_darcy.py                                  # SI Text S4
```
Verification of this archive: a pulsed-WAG Mhow run (injection-driven closure) executed from the archive reproduced the stored gross efficiency (38.7337 %) exactly, with a carbon-ledger closure error of 3.6 × 10⁻⁶ %; `compile_numbers.py` regenerated `base_numbers.json` byte for byte.

## Definitions
- **Flow closures.** Overpressure-driven (`A0`): 50 m column, 10 cells, Darcy flux from the 8 bar overpressure. Injection-driven (`A1`): first 5 m of the column, 50 cells, Darcy flux from the injected fluid volume.
- **Net efficiency** = carbon in carbonate minerals (run minus its no-injection control) / scheduled CO₂ × 100, at a fixed denominator and without clipping.
- **Carbon ledger:** C_initial + C_injected + C_inflow = C_aqueous + C_mineral + C_free + C_exported + residual, each term from model state.

## Assumptions to note
The access fraction (0.25) and a rate-constant factor of 0.85 at Mhow are inherited from the earlier calibration; the mineral assemblage is allocated from literature XRF data with fixed empirical coefficients; secondary minerals are equilibrium phases; absolute efficiencies are grid dependent and are not validated against field data (see manuscript Sections 5.7 and 7).

## Licence
MIT (see `LICENSE`).
