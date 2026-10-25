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
| `run_more.py` | larger cohorts (N = 100, 200) and the D2 training-data control |
| `cond_numbers.py` | condition numbers of the covariance estimates (Sec. VII-B) |
| `check_d2_contention.py` | explains the ordering of D2 under contention (Sec. VII-B) |
| `fig_timeline.py` | Fig. 1 (schematic) |
| `results/` | summary tables (`*.csv`), `numbers.txt`, raw results `unified.csv.gz`; `results/extra/` raw and `results/out_extra/` summaries of `run_extra.py` |

## Simulation parameters
| parameter | value |
|---|---|
| UEs per cell N | 50 (100, 200 in `run_more.py`) |
| Byzantine fraction | 20% (A6: f = 0.1-0.4; cohort test: up to 45%) |
| update size | 8000 kbit, fixed |
| uplink rate | per-UE log-normal, median 2000 kbit/s, log-std 0.5 |
| start delay | per-UE log-normal, median 6 s, log-std 0.4; coupling to channel 0.5; independent noise sigma_tau = 0.15 |
| reporting interval g | 1 s (tau and d rounded up) |
| volume overhead | at most 0.4% |
| log channel gain | std 0.25; i.i.d., AR(1) 0.8, drift 0.001/round, or t3 |
| rounds | m in {30, 50, 100, 200} warm-up, 100 calibration, 200 evaluation |
| detector | alpha = 1%, s_min = 0.02, 3-out-of-5, L = 5, autoencoder 15-8-3-8-15 |
| seeds | 30 (separate benign seeds 100-129 for the matched-false-isolation thresholds) |

Values are assumptions, not measurements. For reference, commercial 5G uplink peaks at about 30-60 Mbit/s on smartphones (Narayanan et al., WWW 2020), and FL on smartphones needs under 30 s of communication and 1-5 min of training per round for a model of about 50M parameters (Yang et al., WWW 2021); our update (1 MB) and delays (seconds) correspond to a smaller model and a loaded cell.

## Check the numbers without rerunning (seconds)
```bash
pip install -r requirements.txt
python3 print_numbers.py results results/out_extra results/more   # same output as results/numbers.txt
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
python3 run_more.py results/more
python3 print_numbers.py results/out results/out_extra results/more
python3 cond_numbers.py
python3 fig_timeline.py
```
