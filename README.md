# Network-metadata detection of protocol-deviating FL clients at the O-RAN RIC

Simulation code for the paper *Network-Metadata-Based Detection of Protocol-Deviating Federated Learning Clients via the O-RAN Near-RT RIC* (WCNC 2026 submission).

All results in the paper come from one generator and one run (30 seeds, fixed), so a rerun gives identical numbers.

## Files
| file | role |
|---|---|
| `core.py` | synthetic generator, D1 (sample / variance floor / Ledoit-Wolf), D2 autoencoder |
| `unified.py` | attacks A1-A8, baselines (D1-W, max-\|z\|, lag-1, timeout), cohort-median normalization, k-out-of-n decisions |
| `run_unified.py` | runs every experiment |
| `analyze_unified.py` | figures and summary tables |
| `print_numbers.py` | prints every number quoted in the paper |
| `fig_timeline.py` | schematic timeline figure |
| `check_d2_contention.py` | explains the D2 ordering under contention |
| `results/` | summary tables, `numbers.txt`, and the raw results (`unified.csv.gz`) |

## Reproduce
```bash
pip install -r requirements.txt
python3 run_unified.py 30 results/unified.csv      # about 30 min on 2 cores
python3 analyze_unified.py results/unified.csv results/out
python3 print_numbers.py results/out
```
The condition numbers of the covariance estimates are in `results/cond_from_v1_run.csv`.
