# Manifest: manuscript and SI items → data and scripts

| Item | Data file | Script |
|---|---|---|
| Table 6 (carbon ledger), Table 8, SI Tables S3–S8, S15 | `base_numbers.json` (from `ledger_round2.json`) | `compile_numbers.py` |
| Table 7 (hydraulic comparability) | `matrix_runs.jsonl` (`hydraulics`), `base_numbers.json` | `compile_numbers.py` |
| Figure 3 (conceptual model) | — | `fig_concept.py` |
| Figures 4–13 (pH, dissolution, carbonates, clays, porosity) | `cache/A1_*.pkl` | `gen_ms_series.py`, `make_figs.py` |
| Figure 14 (net efficiency vs time) | `cache/A1_*.pkl`, `cache/A0_*.pkl` | `gen_ms_series.py`, `gen_ms_series4.py`, `make_fig14.py` |
| Figure 15 (undissolved CO₂) | `cache/A1_*.pkl` | `make_fig14.py` |
| Figure 16 (regional flux) | `compiled2.json` | `plot_regime_big.py` |
| Graphical abstract | `figures/toc_original.jpeg` + numbers from `base_numbers.json` | `toc.py` |
| Section 5.6, SI Tables S11, S11b, S13, S16 | `matrix_summary.csv`, `tau_results.jsonl`, `cache/A1boost1_*.pkl` | `run_sensitivity_matrix.py`, `gen_tau.py`, `gen_ms_series3.py` |
| Section 5.7, SI Tables S12, S14 | `matrix_summary.csv`, `ledger_long.json` | `run_sensitivity_matrix.py`, `run_long.py` |
| SI Text S4 | — | `verify_darcy.py` |
| First-submission model results (for comparison only) | `round1_results.jsonl` | — |

Cache files (`round2_results/cache/*.pkl`, about 1.2 MB each) are not included; steps 4 of the README regenerate them.
