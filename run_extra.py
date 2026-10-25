"""Additional experiments (same generator and code as run_unified.py), m = 100:

A  power at matched 1% FALSE ISOLATION. For each detector, the per-round threshold level alpha*
   is chosen on 30 separate benign seeds (100-129) as the largest alpha whose pooled calibration
   quantile gives at most 1% of benign UEs isolated (3-out-of-5). It is then applied, unchanged,
   to the test seeds 0-29. This is an offline diagnostic (oracle on separate benign runs), not a
   deployable calibration.
B  sensitivity to the independent per-round noise of the generator (start delay; and duration).
C  cohort-median normalization against a growing Byzantine fraction.
D  contaminated warm-up: Byzantine UEs already active during a fraction of the warm-up.

usage: python3 run_extra.py OUTDIR
"""
import sys, os, time, itertools, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
import core
from core import CFG, SCENARIOS, generate, features, Detectors
from unified import *

M = 100
EV0 = M + CFG["K_CAL"]
T = M + CFG["K_CAL"] + CFG["K_EVAL"]
DETS = ["D1lw", "D2", "D1W", "Zmax", "Lag1", "Timeout"]
ALPHA_GRID = np.concatenate([[0.0], np.logspace(-4, np.log10(0.02), 60)])  # alpha=0: max of calibration
ATT_A = {"A1": [0.5, 1, 2, 3, 5], "A5": [1, 1.5, 2, 3], "A8": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0], "A7": [5]}


def setup(seed, scn, m=M):
    rng = np.random.default_rng([seed, SCENARIOS.index(scn), m])
    par = generate(scn, m + CFG["K_CAL"] + CFG["K_EVAL"], rng)
    raw = features(par)
    perm = rng.permutation(N)
    return par, raw, perm


def scores_all(det, X):
    S = det.scores(X)
    S.update({k: v for k, v in light_scores(X, M).items() if k != "D1lw"})
    return {k: S[k] for k in DETS}


def thr_q(S, alpha):
    cal = S[:, M:M + CFG["K_CAL"]]; cal = cal[~np.isnan(cal)]
    return cal.max() * (1 + 1e-9) if alpha == 0 else np.quantile(cal, 1 - alpha)


# ------------------------------------------------------------------ A: calibration on separate seeds
def task_cal(args):
    seed, scn = args
    par, raw, perm = setup(seed, scn)
    det = Detectors(raw, M, seed)
    S = scores_all(det, raw)
    allu = np.ones(N, bool)
    out = []
    for k in DETS:
        for a in ALPHA_GRID:
            fl = flags(S[k], thr_q(S[k], a))
            out.append(dict(seed=seed, scn=scn, method=k, alpha=a,
                            false_iso=iso_stats(decide(fl, 3, 5), EV0, allu)[0], fpr=fl[:, EV0:].mean()))
    return out


def task_test(args):
    seed, scn, alphas = args
    par, raw, perm = setup(seed, scn)
    det = Detectors(raw, M, seed)
    byz = np.zeros(N, bool); byz[perm[:int(0.2 * N)]] = True
    allu = np.ones(N, bool)
    out = []
    S0 = scores_all(det, raw)
    for k in DETS:
        fl = flags(S0[k], thr_q(S0[k], alphas[scn][k]))
        out.append(dict(seed=seed, scn=scn, exp="A_benign", method=k, fpr=fl[:, EV0:].mean(),
                        false_iso=iso_stats(decide(fl, 3, 5), EV0, allu)[0]))
    for name, mags in ATT_A.items():
        for mag in mags:
            X = apply_attack(name, mag, par, raw, det, byz, EV0, duty=(2, 5))
            S = scores_all(det, X)
            for k in DETS:
                for cal, a in (("fpr1", 0.01), ("fi1", alphas[scn][k])):
                    fl = flags(S[k], thr_q(S[k], a)); dec = decide(fl, 3, 5)
                    dr, dl = iso_stats(dec, EV0, byz)
                    out.append(dict(seed=seed, scn=scn, exp="A", cal=cal, method=k, attack=name, mag=mag,
                                    flag_rate=fl[byz][:, EV0:].mean(), det_rate=dr, delay=dl,
                                    ben_false_iso=iso_stats(dec, EV0, ~byz)[0]))
    return out


# ------------------------------------------------------------------ B, C, D (D1 only, no autoencoder)
class NoAE(Detectors):
    def _fit_ae(self, *a, **k):
        pass


def feats_noise(par, seed, sig_tau, sig_d, logc=None):
    """generator with modified independent noise; sig_tau = 0.15, sig_d = 0 reproduces core.features"""
    c = dict(CFG); c["SIG_TAU"] = sig_tau
    b, tau_c, d_c = cont_features(par, logc, c)
    if sig_d > 0:
        e = np.random.default_rng([seed, 777]).standard_normal(d_c.shape)
        d_c = d_c * np.exp(sig_d * e)
    return quantize(b, tau_c, d_c), (b, tau_c, d_c)


def iso_frac(S, rows, alpha=0.01, k=3, n=5):
    fl = flags(S, thr(S, M, alpha)); dec = decide(fl, k, n)
    return iso_stats(dec, EV0, rows), fl


def task_bcd(args):
    seed, scn = args
    out = []
    par, raw, perm = setup(seed, scn)
    allu = np.ones(N, bool)
    # ---- B: generator sensitivity
    for sig_tau, sig_d in [(0.05, 0), (0.10, 0), (0.15, 0), (0.25, 0), (0.35, 0), (0.15, 0.10), (0.15, 0.20)]:
        X0, (b, tc, dc) = feats_noise(par, seed, sig_tau, sig_d)
        det = NoAE(X0, M, seed)
        byz = np.zeros(N, bool); byz[perm[:10]] = True
        (fi0, _), _ = iso_frac(light_scores(X0, M)["D1lw"], allu)
        # A1 3 sigma on this generator
        sd = X0[:, :M, :].std(1, ddof=1)
        ta = tc.copy(); ta[byz, EV0:] += 3 * sd[byz, 1][:, None]
        XA = quantize(b, ta, dc)
        SA = light_scores(XA, M)["D1lw"]
        th_m = np.quantile(SA[~byz][:, EV0:].ravel(), 0.99)
        pw = (SA[byz][:, EV0:] > th_m).mean()
        (dr, dl), flA = iso_frac(SA, byz)
        a1_flag = flA[byz][:, EV0:].mean()
        # A6, Byzantine (loaders) fraction 0.3, beta 1
        b6 = np.zeros(N, bool); b6[perm[:15]] = True
        lc = par["logc"].copy(); lc[:, EV0:] -= np.log(1.3)
        X6, _ = feats_noise(par, seed, sig_tau, sig_d, lc)
        (fi6, _), fl6 = iso_frac(light_scores(X6, M)["D1lw"], ~b6)
        (fi6c, _), _ = iso_frac(light_scores(X6, M, coh=True)["D1lw"], ~b6)
        out.append(dict(seed=seed, scn=scn, exp="B", sig_tau=sig_tau, sig_d=sig_d, benign_fi=fi0, A1_power=pw, A1_flag=a1_flag,
                        A1_det=dr, A1_delay=dl, A6_fi=fi6, A6_fi_coh=fi6c, A6_flag=fl6[~b6][:, EV0:].mean()))
    # ---- C: cohort median vs Byzantine fraction
    det = NoAE(raw, M, seed)
    for fb in (0.1, 0.2, 0.3, 0.4, 0.45):
        byz = np.zeros(N, bool); byz[perm[:int(round(fb * N))]] = True
        for name, mag in (("A1", 3), ("A1", 5)):
            X = apply_attack(name, mag, par, raw, det, byz, EV0)
            for coh in (False, True):
                S = light_scores(X, M, coh=coh)["D1lw"]
                (dr, dl), fl = iso_frac(S, byz)
                out.append(dict(seed=seed, scn=scn, exp="C", byz_frac=fb, attack=name, mag=mag, coh=coh,
                                byz_iso=dr, byz_delay=dl, ben_fpr=fl[~byz][:, EV0:].mean(),
                                ben_fi=iso_stats(decide(fl, 3, 5), EV0, ~byz)[0]))
    # ---- D: contaminated warm-up (A1 3 sigma from round s on)
    byz = np.zeros(N, bool); byz[perm[:10]] = True
    b, tc, dc = cont_features(par)
    sd_clean = raw[:, :M, :].std(1, ddof=1)
    for frac in (0.0, 0.25, 0.5, 1.0):
        s = EV0 if frac == 0 else int(round(M * (1 - frac)))
        ta = tc.copy(); ta[byz, s:] += 3 * sd_clean[byz, 1][:, None]
        X = quantize(b, ta, dc)
        S = light_scores(X, M)["D1lw"]
        (dr, dl), fl = iso_frac(S, byz)
        th_m = np.quantile(S[~byz][:, EV0:].ravel(), 0.99)
        out.append(dict(seed=seed, scn=scn, exp="D", contam=frac, byz_iso=dr, byz_delay=dl,
                        power=(S[byz][:, EV0:] > th_m).mean(), flag_rate=fl[byz][:, EV0:].mean(),
                        ben_fi=iso_stats(decide(fl, 3, 5), EV0, ~byz)[0]))
    return out


if __name__ == "__main__":
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    with Pool(2) as pool:
        cal = sum(pool.map(task_cal, list(itertools.product(range(100, 130), ["gauss", "ar"]))), [])
        cal = pd.DataFrame(cal); cal.to_csv(f"{outdir}/A_calibration.csv", index=False)
        g = cal.groupby(["scn", "method", "alpha"]).false_iso.mean().reset_index()
        alphas = {}
        for (scn, k), d in g.groupby(["scn", "method"]):
            ok = d[d.false_iso <= 0.01]
            alphas.setdefault(scn, {})[k] = float(ok.alpha.max()) if len(ok) else 0.0
        json.dump(alphas, open(f"{outdir}/A_alphas.json", "w"), indent=1)
        print("calibration", time.time() - t0, alphas, flush=True)
        test = sum(pool.map(task_test, [(s, scn, alphas) for s in range(30) for scn in ["gauss", "ar"]]), [])
        pd.DataFrame(test).to_csv(f"{outdir}/A_test.csv", index=False)
        print("A test", time.time() - t0, flush=True)
        bcd = sum(pool.map(task_bcd, list(itertools.product(range(30), ["gauss", "ar"]))), [])
        pd.DataFrame(bcd).to_csv(f"{outdir}/BCD.csv", index=False)
    print("done", time.time() - t0, flush=True)
