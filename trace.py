"""Trace-driven channel: replaces the synthetic log channel gain of core.generate by real 5G channel traces.

For one run, the traces of a mobility type (static or driving) are concatenated in a random order. Each UE
starts at a random offset and follows the concatenated series (circularly) with one FL round every ROUND_S
seconds; its per-round channel is the mean spectral efficiency over the round's upload window (the first
UP_S seconds), and logc = log(SE) - median of log(SE) over the pool. Everything else (per-UE rate and delay,
noise, overhead, quantization, attacks) is exactly as in core.py.
"""
import os
import numpy as np
from core import CFG

ROUND_S, UP_S = 10, 4
_here = os.path.dirname(os.path.abspath(__file__))
_Z = np.load(os.path.join(_here, "traces", "se.npz"))
POOLS = {m: [_Z[k] for k in sorted(_Z.files, key=lambda s: int(s.split("_")[1])) if k.startswith(m)]
         for m in ("static", "driving")}
_MED = {m: np.median(np.log(np.concatenate(v))) for m, v in POOLS.items()}


def generate_trace(mob, T, rng, c=CFG):
    N = c["N"]
    logR = rng.normal(c["LOGR_MU"], c["LOGR_SD"], N)
    tau0 = np.exp(rng.normal(c["TAU_MU"], c["TAU_SD"], N))
    eta = rng.standard_normal((N, T))
    epsb = np.abs(rng.standard_normal((N, T))) * c["EPS_B"]
    pool = POOLS[mob]
    order = rng.permutation(len(pool))
    cat = np.concatenate([pool[i] for i in order])
    L = len(cat)
    starts = rng.integers(0, L, N)
    idx = (starts[:, None, None] + ROUND_S * np.arange(T)[None, :, None] + np.arange(UP_S)[None, None, :]) % L
    se = cat[idx].mean(-1)                                  # N x T
    logc = np.log(se) - _MED[mob]
    return dict(logR=logR, tau0=tau0, logc=logc, eta=eta, epsb=epsb)


def generate_trace_single(mob, T, rng, c=CFG):
    """variant without trace boundaries: each UE follows ONE trace, with its T rounds spread evenly over it
    (spacing = trace length // T seconds, at least 1 s; window = min(UP_S, spacing)), from a random
    circular offset."""
    N = c["N"]
    logR = rng.normal(c["LOGR_MU"], c["LOGR_SD"], N)
    tau0 = np.exp(rng.normal(c["TAU_MU"], c["TAU_SD"], N))
    eta = rng.standard_normal((N, T))
    epsb = np.abs(rng.standard_normal((N, T))) * c["EPS_B"]
    pool = POOLS[mob]
    pick = rng.integers(0, len(pool), N)
    off = rng.random(N)                      # random circular offset, so that UEs sharing a segment are not aligned
    logc = np.empty((N, T))
    for i, j in enumerate(pick):
        tr = pool[j]; sp = max(1, len(tr) // T); w = min(UP_S, sp)
        o = int(off[i] * len(tr))
        idx = (o + sp * np.arange(T)[:, None] + np.arange(w)[None, :]) % len(tr)
        logc[i] = np.log(tr[idx].mean(-1)) - _MED[mob]
    return dict(logR=logR, tau0=tau0, logc=logc, eta=eta, epsb=epsb)
