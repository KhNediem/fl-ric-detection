"""Unified simulation for the WCNC paper.

All experiments (original A1-A6, D1/D2, plus the additional baselines, A7 duty-cycle,
A8 free-rider, cohort-median normalization and k-out-of-n rules) run on ONE generator:
the original one in core.py (unchanged). Differences from the original run.py:
  * attacked start delays and durations are quantized to g like benign ones
    (ceil for attacks defined as physical time shifts: A1, A3, A7, A8; nearest
    integer for attacks defined directly in feature space: A4, A5);
  * decisions are taken on a rolling window that spans the calibration/evaluation
    boundary (as in the original core.decisions), for every k-out-of-n rule.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
import core
from core import CFG, generate, Detectors, threshold

N = CFG["N"]
L = CFG["L"]
SMIN = CFG["S_MIN"]


# ------------------------------------------------------------------ generator
def cont_features(par, logc=None, c=CFG):
    """continuous (b, tau, d) before quantization; identical formulas to core.features"""
    logc = par["logc"] if logc is None else logc
    b = c["S"] * (1 + par["epsb"])
    tau_c = par["tau0"][:, None] * np.exp(c["SIG_TAU"] * par["eta"] - c["KAPPA"] * logc)
    d_c = b / (np.exp(par["logR"])[:, None] * np.exp(logc))
    return b, tau_c, d_c


def quantize(b, tau_c, d_c, g=CFG["G"]):
    tau = np.maximum(1.0, np.ceil(tau_c / g) * g)
    d = np.maximum(1.0, np.ceil(d_c / g) * g)
    return np.stack([b, tau, d], axis=-1)


# ------------------------------------------------------------------ attacks
def apply_attack(name, mag, par, raw, det, byz, ev0, duty=None):
    """returns attacked quantized features. `det` gives warm-up statistics (mu, sd_f, cov_floor)."""
    m = det.m
    T = raw.shape[1]
    idx = np.where(byz)[0]
    b, tau_c, d_c = cont_features(par)
    sd_raw = raw[:, :m, :].std(1, ddof=1)
    r = np.arange(T - ev0)
    if name in ("A1", "A3", "A7", "A8"):
        tau_a, d_a = tau_c.copy(), d_c.copy()
        if name == "A1":
            tau_a[idx, ev0:] += mag * sd_raw[idx, 1][:, None]
        elif name == "A3":
            d_a[idx, ev0:] += mag * sd_raw[idx, 2][:, None]
        elif name == "A7":                       # duty-cycle: active in `duty`=(a, n) rounds of every n
            a, n = duty
            act = (r % n) < a
            tau_a[idx, ev0:] += mag * sd_raw[idx, 1][:, None] * act[None, :]
        elif name == "A8":                       # free-rider: skips fraction mag of training (80% of tau)
            tau_a[idx, ev0:] *= (1 - 0.8 * mag)
        return quantize(b, tau_a, d_a)
    out = raw.copy()
    if name == "A2":
        out[idx, ev0:, 0] *= (1 + mag)
        return out
    f = np.log1p(raw)
    if name == "A4":
        for i in idx:
            cz = det.cov_floor[i] / np.outer(det.sd_f[i], det.sd_f[i])
            w, V = np.linalg.eigh(cz)
            v = V[:, -1] * (1 if V[:, -1].sum() >= 0 else -1)
            f[i, ev0:, :] += det.sd_f[i] * mag * np.sqrt(w[-1]) * v
    elif name == "A5":
        s = (-1.0) ** np.arange(T)
        for i in idx:
            for j in (1, 2):
                f[i, ev0:, j] = det.mu[i, j] + mag * s[ev0:] * det.sd_f[i, j]
    out[idx, ev0:, :] = np.expm1(f[idx, ev0:, :])
    out[idx, ev0:, 1:] = np.maximum(1.0, np.round(out[idx, ev0:, 1:]))
    return out


# ------------------------------------------------------------------ light detectors
def cohort(f, m):
    """subtract, per round and feature, the median over all FL UEs of the deviation from the warm-up mean"""
    mu0 = f[:, :m, :].mean(1, keepdims=True)
    return f - np.median(f - mu0, axis=0, keepdims=True)


def light_scores(raw, m, coh=False):
    """D1 (Ledoit-Wolf), windowed D1, max-|z|, lag-1 and timeout; identical D1 to core.Detectors 'D1lw'."""
    f = np.log1p(raw)
    if coh:
        f = cohort(f, m)
    mu = f[:, :m].mean(1)
    sd = np.maximum(f[:, :m].std(1, ddof=1), SMIN)
    z = (f - mu[:, None, :]) / sd[:, None, :]
    P, loc = [], []
    for i in range(z.shape[0]):
        lw = LedoitWolf().fit(z[i, :m])
        P.append(np.linalg.inv(lw.covariance_)); loc.append(lw.location_)
    P, loc = np.stack(P), np.stack(loc)
    d2 = z - loc[:, None, :]
    T = raw.shape[1]
    out = {}
    s = np.einsum("ntp,npq,ntq->nt", d2, P, d2); s[:, :m] = np.nan
    out["D1lw"] = s
    cs = np.cumsum(np.concatenate([np.zeros((N, 1, 3)), d2], 1), 1)
    mean = (cs[:, L:] - cs[:, :-L]) / L                      # window ending at t = L-1..T-1
    w = np.full((N, T), np.nan)
    w[:, L - 1:] = L * np.einsum("ntp,npq,ntq->nt", mean, P, mean)
    w[:, :m + L - 1] = np.nan                                # windows fully after warm-up
    out["D1W"] = w
    zm = np.abs(z).max(-1); zm[:, :m] = np.nan
    out["Zmax"] = zm
    prod = -(z[:, 1:, 1] * z[:, :-1, 1] + z[:, 1:, 2] * z[:, :-1, 2]) / 2.0   # pair (t-1, t) at index t-1
    cs1 = np.cumsum(np.concatenate([np.zeros((N, 1)), prod], 1), 1)
    a = np.full((N, T), np.nan)
    a[:, L - 1:] = (cs1[:, L - 1:] - cs1[:, :-(L - 1)]) / (L - 1)   # window of L rounds ending at t
    a[:, :m + L - 1] = np.nan
    out["Lag1"] = a
    to = raw[:, :, 1] + raw[:, :, 2]
    to = to.astype(float); to[:, :m] = np.nan
    out["Timeout"] = to
    return out


# ------------------------------------------------------------------ decisions
def thr(S, m, alpha):
    cal = S[:, m:m + CFG["K_CAL"]]
    cal = cal[~np.isnan(cal)]
    return np.quantile(cal, 1 - alpha)


def flags(S, th):
    return np.nan_to_num(S, nan=-np.inf) > th


def decide(fl, k, n):
    """rolling k-of-n over the whole timeline (windows may span calibration and evaluation)"""
    cs = np.cumsum(fl.astype(int), axis=1)
    roll = cs.copy()
    roll[:, n:] = cs[:, n:] - cs[:, :-n]
    return roll >= k


def iso_stats(dec, ev0, rows):
    d = dec[rows][:, ev0:]
    det = d.any(1)
    delay = float(np.median(np.argmax(d[det], axis=1) + 1)) if det.any() else np.nan
    return float(det.mean()), delay
