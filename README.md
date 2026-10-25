# Network-metadata detection of protocol-deviating FL clients at the O-RAN RIC

Simulation code and results for the paper *Network-Metadata-Based Detection of Protocol-Deviating Federated Learning Clients via the O-RAN Near-RT RIC* (submitted to IEEE WCNC 2026).

All results in the paper come from one synthetic generator and one run (30 fixed seeds), so a rerun gives identical numbers. Tested with Python 3.13 and the package versions in `requirements.txt`.

## Files
| file | role |
|---|---|
| `core.py` | synthetic generator, D1 (sample / variance floor / Ledoit-Wolf), D2 autoencoder |
| `unified.py` | attacks A1-A8, baselines (D1-W, max-\|z\|, lag-1, timeout), cohort-median normalization, k-out-of-n decisions |
| `run_unified.py` | runs every experiment of the paper |
| `analyze_unified.py` | Figs. 3 and 5 (`fig_benign`, `fig_contention`) and the summary tables |
| `run_extra.py` | matched 1% false-isolation comparison (offline diagnostic), generator sensitivity, Byzantine fraction with cohort normalization, contaminated warm-up |
| `analyze_extra.py` | Fig. 4 (`fig_detectors`) and the summary tables of `run_extra.py` |
| `print_numbers.py` | prints every number quoted in the paper from the summary tables |
| `cond_numbers.py` | condition numbers of the covariance estimates (Sec. VII-B) |
| `check_d2_contention.py` | explains the ordering of D2 under contention (Sec. VII-B) |
| `fig_timeline.py` | Fig. 1 (schematic) |
| `results/` | summary tables (`*.csv`), `numbers.txt`, raw results `unified.csv.gz`; `results/extra/` raw and `results/out_extra/` summaries of `run_extra.py` |

## Check the numbers without rerunning (seconds)
```bash
pip install -r requirements.txt
python3 print_numbers.py results results/out_extra   # same output as results/numbers.txt
gunzip -k results/unified.csv.gz
python3 analyze_unified.py results/unified.csv results/out   # Figs. 3 and 5
python3 analyze_extra.py results/extra results/out_extra     # Fig. 4
```

## Rerun everything (about 35 min on 2 cores)
```bash
python3 run_unified.py 30 results/unified.csv
python3 analyze_unified.py results/unified.csv results/out
python3 run_extra.py results/extra
python3 analyze_extra.py results/extra results/out_extra
python3 print_numbers.py results/out results/out_extra
python3 cond_numbers.py
python3 fig_timeline.py
```
