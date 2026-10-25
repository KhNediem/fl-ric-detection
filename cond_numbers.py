"""Condition numbers of the per-UE covariance estimates of D1 (sample, variance floor, Ledoit-Wolf).
Median over UEs per run, then mean over 30 seeds x 4 scenarios, for each warm-up size m.
usage: python3 cond_numbers.py [NSEEDS]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import core
from core import CFG, SCENARIOS, generate, features, Detectors

core.Detectors._fit_ae = lambda self, *a, **k: None   # the autoencoder is not needed here
nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for m in (30, 50, 100, 200):
    acc = {"sample": [], "floor": [], "ledoit-wolf": []}
    for seed in range(nseeds):
        for scn in SCENARIOS:
            rng = np.random.default_rng([seed, SCENARIOS.index(scn), m])
            par = generate(scn, m + CFG["K_CAL"] + CFG["K_EVAL"], rng)
            det = Detectors(features(par), m, seed)
            acc["sample"].append(np.median(det.cond_raw))
            acc["floor"].append(np.median(det.cond_floor))
            acc["ledoit-wolf"].append(np.median(det.cond_lw))
    print(f"m={m:3d}  " + "  ".join(f"{k}: {np.mean(v):8.1f}" for k, v in acc.items()))
