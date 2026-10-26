"""Two extensions of D1, evaluated on synthetic and trace-driven channels.

CA  channel-aware features: the RIC also observes a per-UE channel-quality report each round (modeled as the
    true log channel gain plus Gaussian measurement noise of std SIG_OBS). Per UE, log tau and log d are
    regressed on the reported channel over the warm-up, and D1 scores the residuals (b unchanged).
OR  online recalibration: every K rounds, each UE's baseline is re-estimated on its last W unflagged rounds
    (with a lag of LAG rounds, so that a new deviation does not enter its own baseline at once), the pooled threshold stays the one
    calibrated on clean data, so that attackers cannot raise it.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from core import CFG
from unified import thr, flags, decide, iso_stats

SMIN = CFG["S_MIN"]
SIG_OBS, K, W, LAG = 0.15, 10, 100, 5


def ca_features(X, logc_obs, m):
    f = np.log1p(X).astype(float)
    out = f.copy()
    for j in (1, 2):
        for i in range(X.shape[0]):
            A = np.vstack([np.ones(m), logc_obs[i, :m]]).T
            coef, *_ = np.linalg.lstsq(A, f[i, :m, j], rcond=None)
            out[i, :, j] = f[i, :, j] - (coef[0] + coef[1] * logc_obs[i])
    return out


def _fit(F):
    mu = F.mean(0); sd = np.maximum(F.std(0, ddof=1), SMIN)
    lw = LedoitWolf().fit((F - mu) / sd)
    return mu, sd, lw.location_, np.linalg.inv(lw.covariance_)


def _score(F, p):
    mu, sd, loc, P = p
    z = (F - mu) / sd - loc
    return np.einsum("tp,pq,tq->t", z, P, z)


def d1_static(F, m):
    S = np.full(F.shape[:2], np.nan)
    for i in range(F.shape[0]):
        S[i, m:] = _score(F[i, m:], _fit(F[i, :m]))
    return S


def d1_online(F, m, ev0, alpha=0.01, k=3, n=5):
    """returns flags (N x T) with online baselines and threshold; calibration rounds use the static fit."""
    N_, T, _ = F.shape
    params = [_fit(F[i, :m]) for i in range(N_)]
    S = np.full((N_, T), np.nan)
    for i in range(N_):
        S[i, m:ev0] = _score(F[i, m:ev0], params[i])
    th = np.quantile(S[:, m:ev0].ravel(), 1 - alpha)
    fl = np.zeros((N_, T), bool); fl[:, m:ev0] = S[:, m:ev0] > th
    for t0 in range(ev0, T, K):
        t1 = min(T, t0 + K)
        if t0 > ev0:
            hi = t0 - LAG
            for i in range(N_):
                idx = np.arange(max(m, hi - 3 * W), hi)
                idx = idx[~fl[i, idx]][-W:]
                if len(idx) >= 30:
                    params[i] = _fit(F[i, idx])
        for i in range(N_):
            S[i, t0:t1] = _score(F[i, t0:t1], params[i])
        fl[:, t0:t1] = S[:, t0:t1] > th
    return fl, S
