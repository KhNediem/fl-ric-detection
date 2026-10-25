"""Synthetic simulation for D1 (Hotelling) and D2 (autoencoder) on per-UE FL round features.
All generator parameters are simulation assumptions (see CFG), not literature values."""
import copy
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neural_network import MLPRegressor

CFG = dict(
    N=50, K_CAL=100, K_EVAL=200,
    S=8000.0,          # update size [kbit], fixed
    G=1.0,             # reporting interval g [s]
    LOGR_MU=np.log(2000.0), LOGR_SD=0.5,   # per-UE uplink rate [kbit/s]
    TAU_MU=np.log(6.0), TAU_SD=0.4,        # per-UE download+compute time [s]
    SIG_C=0.25,        # channel log-std
    SIG_TAU=0.15,      # tau noise log-std
    KAPPA=0.5,         # coupling of tau to channel
    EPS_B=0.004,       # retransmission overhead scale
    PHI_AR=0.8, DRIFT=0.001,
    S_MIN=0.02, L=5, ALPHA=0.01, KOUT=3, WIN=5,
)
SCENARIOS = ["gauss", "ar", "drift", "heavy"]
METHODS = ["D1raw", "D1floor", "D1lw", "D2"]


def generate(scn, T, rng, c=CFG):
    N = c["N"]
    logR = rng.normal(c["LOGR_MU"], c["LOGR_SD"], N)
    tau0 = np.exp(rng.normal(c["TAU_MU"], c["TAU_SD"], N))
    if scn == "heavy":
        draw = lambda size: rng.standard_t(3, size) / np.sqrt(3.0)
    else:
        draw = lambda size: rng.standard_normal(size)
    e = draw((N, T))
    eta = draw((N, T))
    epsb = np.abs(rng.standard_normal((N, T))) * c["EPS_B"]
    phi = c["PHI_AR"] if scn == "ar" else 0.0
    logc = np.zeros((N, T))
    logc[:, 0] = c["SIG_C"] * e[:, 0]
    for t in range(1, T):
        logc[:, t] = phi * logc[:, t - 1] + np.sqrt(1 - phi ** 2) * c["SIG_C"] * e[:, t]
    if scn == "drift":
        logc -= c["DRIFT"] * np.arange(T)[None, :]
    return dict(logR=logR, tau0=tau0, logc=logc, eta=eta, epsb=epsb)


def features(par, logc=None, c=CFG):
    logc = par["logc"] if logc is None else logc
    b = c["S"] * (1 + par["epsb"])
    tau_c = par["tau0"][:, None] * np.exp(c["SIG_TAU"] * par["eta"] - c["KAPPA"] * logc)
    d_c = b / (np.exp(par["logR"])[:, None] * np.exp(logc))
    tau = np.maximum(1.0, np.ceil(tau_c / c["G"]))
    d = np.maximum(1.0, np.ceil(d_c / c["G"]))
    return np.stack([b, tau, d], axis=-1)


class Detectors:
    """Per-UE baselines fitted on the first m (clean) rounds; shared autoencoder for D2."""

    def __init__(self, raw, m, seed, c=CFG, ae_epochs=300, patience=20):
        self.c, self.m = c, m
        N, T, p = raw.shape
        f = np.log1p(raw[:, :m, :])
        self.mu = f.mean(1)
        sd = f.std(1, ddof=1)
        self.sd_f = np.maximum(sd, c["S_MIN"])
        cov = np.stack([np.cov(f[i].T) for i in range(N)])
        self.P_raw, self.cond_raw = [], []
        self.P_floor, self.cond_floor, self.cov_floor = [], [], []
        for i in range(N):
            self.P_raw.append(np.linalg.pinv(cov[i], rcond=1e-12))
            self.cond_raw.append(np.linalg.cond(cov[i]))
            fl = cov[i] + np.diag(np.maximum(0.0, c["S_MIN"] ** 2 - np.diag(cov[i])))
            self.cov_floor.append(fl)
            self.P_floor.append(np.linalg.inv(fl))
            self.cond_floor.append(np.linalg.cond(fl))
        self.P_raw, self.P_floor = np.stack(self.P_raw), np.stack(self.P_floor)
        self.cov_floor = np.stack(self.cov_floor)
        z = (f - self.mu[:, None, :]) / self.sd_f[:, None, :]
        self.lw_loc, self.P_lw, self.cond_lw = [], [], []
        for i in range(N):
            lw = LedoitWolf().fit(z[i])
            self.lw_loc.append(lw.location_)
            self.P_lw.append(np.linalg.inv(lw.covariance_))
            self.cond_lw.append(np.linalg.cond(lw.covariance_))
        self.lw_loc, self.P_lw = np.stack(self.lw_loc), np.stack(self.P_lw)
        self._fit_ae(z, seed, ae_epochs, patience)

    def _windows(self, z, ends):
        L = self.c["L"]
        return np.stack([z[:, t - L + 1:t + 1, :].reshape(z.shape[0], -1) for t in ends], axis=1)  # N x W x 15

    def _fit_ae(self, z, seed, epochs, patience):
        L, m = self.c["L"], self.m
        r_tr = int(0.7 * m) - 1
        tr_ends = list(range(L - 1, r_tr + 1))
        va_ends = list(range(r_tr + L, m))          # windows start after r_tr: no shared rounds
        Xtr = self._windows(z[:, :m, :], tr_ends).reshape(-1, 3 * L)
        Xva = self._windows(z[:, :m, :], va_ends).reshape(-1, 3 * L)
        ae = MLPRegressor(hidden_layer_sizes=(8, 3, 8), activation="tanh", solver="adam",
                          learning_rate_init=0.01, batch_size=64, random_state=seed, shuffle=True)
        ae.partial_fit(Xtr, Xtr)
        best = (np.mean((ae.predict(Xva) - Xva) ** 2), copy.deepcopy(ae.coefs_), copy.deepcopy(ae.intercepts_))
        bad = 0
        for _ in range(epochs):
            ae.partial_fit(Xtr, Xtr)
            v = np.mean((ae.predict(Xva) - Xva) ** 2)
            if v < best[0] - 1e-6:
                best = (v, copy.deepcopy(ae.coefs_), copy.deepcopy(ae.intercepts_)); bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        ae.coefs_, ae.intercepts_ = best[1], best[2]
        self.ae = ae
        self.n_train_windows = len(Xtr); self.n_val_windows = len(Xva); self.val_mse = best[0]

    def scores(self, raw):
        N, T, _ = raw.shape
        m, L = self.m, self.c["L"]
        f = np.log1p(raw)
        diff = f - self.mu[:, None, :]
        out = {}
        for name, P in (("D1raw", self.P_raw), ("D1floor", self.P_floor)):
            s = np.einsum("ntp,npq,ntq->nt", diff, P, diff)
            s[:, :m] = np.nan
            out[name] = s
        z = diff / self.sd_f[:, None, :]
        d2 = z - self.lw_loc[:, None, :]
        s = np.einsum("ntp,npq,ntq->nt", d2, self.P_lw, d2)
        s[:, :m] = np.nan
        out["D1lw"] = s
        ends = list(range(m, T))
        W = self._windows(z, ends)
        rec = self.ae.predict(W.reshape(-1, 3 * L)).reshape(W.shape)
        s = np.full((N, T), np.nan)
        s[:, m:] = ((rec - W) ** 2).mean(-1)
        out["D2"] = s
        return out


def decisions(flags, c=CFG):
    w, k = c["WIN"], c["KOUT"]
    cs = np.cumsum(flags.astype(int), axis=1)
    roll = cs.copy()
    roll[:, w:] = cs[:, w:] - cs[:, :-w]
    return roll >= k


def threshold(S, m, c=CFG):
    cal = S[:, m:m + c["K_CAL"]]
    return np.quantile(cal.ravel(), 1 - c["ALPHA"])


def benign_metrics(S, m, c=CFG):
    th = threshold(S, m)
    flags = np.nan_to_num(S, nan=-1) > th
    dec = decisions(flags)
    ev = slice(m + c["K_CAL"], S.shape[1])
    return dict(fpr=flags[:, ev].mean(), dec_rate=dec[:, ev].mean(), ever_iso=dec[:, ev].any(1).mean(), theta=th)


def attack_metrics(S, m, byz, c=CFG):
    th = threshold(S, m)
    ev0 = m + c["K_CAL"]
    ev = slice(ev0, S.shape[1])
    flags = np.nan_to_num(S, nan=-1) > th
    dec = decisions(flags)
    th_match = np.quantile(S[~byz][:, ev].ravel(), 1 - c["ALPHA"])
    power = (S[byz][:, ev] > th_match).mean()
    d = dec[byz][:, ev]
    det = d.any(1)
    delay = np.nan
    if det.any():
        delay = float(np.median(np.argmax(d[det], axis=1) + 1))
    return dict(power=power, det_rate=det.mean(), delay=delay, flag_rate=flags[byz][:, ev].mean(),
                ben_fpr=flags[~byz][:, ev].mean(), ben_ever_iso=dec[~byz][:, ev].any(1).mean())
