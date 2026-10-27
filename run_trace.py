"""Trace-driven validation: benign behavior from real 5G channel traces (trace.py), attacks injected as in the
synthetic runs. Scenarios: static and driving traces; m in {30, 100}; 30 runs (different trace offsets and
UE parameters). Matched 1% false isolation at m = 100 with alphas chosen on 30 separate benign runs (100-129).
usage: python3 run_trace.py OUTDIR
"""
import sys, os, itertools, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from core import CFG, features, Detectors
from unified import *
from trace import generate_trace

MOBS = ["static", "driving"]
DETS = ["D1raw", "D1floor", "D1lw", "D2", "D1W", "Zmax", "Lag1", "Timeout"]
ATT = {"A1": [2, 3, 5], "A4": [3], "A5": [1.5, 3], "A8": [0.3, 0.5, 1.0]}
ALPHA_GRID = np.concatenate([[0.0], np.logspace(-4, np.log10(0.02), 60)])


def setup(seed, mob, m):
    rng = np.random.default_rng([seed, 10 + MOBS.index(mob), m])
    T = m + CFG["K_CAL"] + CFG["K_EVAL"]
    par = generate_trace(mob, T, rng)
    raw = features(par)
    perm = rng.permutation(N)
    return par, raw, perm, T, m + CFG["K_CAL"]


def scores(det, X, m):
    S = det.scores(X)
    S.update({k: v for k, v in light_scores(X, m).items() if k != "D1lw"})
    return S


def thr_q(S, m, a):
    cal = S[:, m:m + CFG["K_CAL"]]; cal = cal[~np.isnan(cal)]
    return cal.max() * (1 + 1e-9) if a == 0 else np.quantile(cal, 1 - a)


def task_cal(args):
    seed, mob = args; m = 100
    par, raw, perm, T, ev0 = setup(seed, mob, m)
    det = Detectors(raw, m, seed); S = scores(det, raw, m); allu = np.ones(N, bool)
    return [dict(seed=seed, mob=mob, method=k, alpha=a,
                 false_iso=iso_stats(decide(flags(S[k], thr_q(S[k], m, a)), 3, 5), ev0, allu)[0])
            for k in DETS for a in ALPHA_GRID]


def task(args):
    seed, mob, m, alphas = args
    par, raw, perm, T, ev0 = setup(seed, mob, m)
    det = Detectors(raw, m, seed)
    allu = np.ones(N, bool)
    out = []
    S0 = scores(det, raw, m)
    for k in DETS:
        fl = flags(S0[k], thr(S0[k], m, 0.01))
        out.append(dict(seed=seed, mob=mob, m=m, exp="benign", method=k, fpr=fl[:, ev0:].mean(),
                        false_iso=iso_stats(decide(fl, 3, 5), ev0, allu)[0]))
    Sc = light_scores(raw, m, coh=True)["D1lw"]; fl = flags(Sc, thr(Sc, m, 0.01))
    out.append(dict(seed=seed, mob=mob, m=m, exp="benign", method="D1lw_coh", fpr=fl[:, ev0:].mean(),
                    false_iso=iso_stats(decide(fl, 3, 5), ev0, allu)[0]))
    if m == 100:
        for k in DETS:
            fl = flags(S0[k], thr_q(S0[k], m, alphas[mob][k]))
            out.append(dict(seed=seed, mob=mob, m=m, exp="benign_fi1", method=k, fpr=fl[:, ev0:].mean(),
                            false_iso=iso_stats(decide(fl, 3, 5), ev0, allu)[0]))
    byz = np.zeros(N, bool); byz[perm[:10]] = True
    for name, mags in ATT.items():
        for mag in mags:
            X = apply_attack(name, mag, par, raw, det, byz, ev0)
            S = scores(det, X, m)
            for k in DETS:
                cals = [("fpr1", thr(S[k], m, 0.01))]
                if m == 100:
                    cals.append(("fi1", thr_q(S[k], m, alphas[mob][k])))
                for cal, th in cals:
                    fl = flags(S[k], th); dr, dl = iso_stats(decide(fl, 3, 5), ev0, byz)
                    out.append(dict(seed=seed, mob=mob, m=m, exp="attack", cal=cal, method=k, attack=name,
                                    mag=mag, flag_rate=fl[byz][:, ev0:].mean(), det_rate=dr, delay=dl))
    b6 = np.zeros(N, bool); b6[perm[:15]] = True
    lc = par["logc"].copy(); lc[:, ev0:] -= np.log(1.3); X6 = features(par, lc)
    S6 = det.scores(X6); S6["D1lw_coh"] = light_scores(X6, m, coh=True)["D1lw"]
    for k in ("D1lw", "D1lw_coh", "D2"):
        fl = flags(S6[k], thr(S6[k], m, 0.01))
        out.append(dict(seed=seed, mob=mob, m=m, exp="A6", method=k,
                        false_iso=iso_stats(decide(fl, 3, 5), ev0, ~b6)[0], flag=fl[~b6][:, ev0:].mean()))
    return out


if __name__ == "__main__":
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True); t0 = time.time()
    with Pool(2) as pool:
        cal = pd.DataFrame(sum(pool.map(task_cal, list(itertools.product(range(100, 130), MOBS))), []))
        cal.to_csv(f"{outdir}/T_calibration.csv", index=False)
        g = cal.groupby(["mob", "method", "alpha"]).false_iso.mean().reset_index()
        alphas = {}
        for (mob, k), d in g.groupby(["mob", "method"]):
            ok = d[d.false_iso <= 0.01]
            alphas.setdefault(mob, {})[k] = float(ok.alpha.max()) if len(ok) else 0.0
        json.dump(alphas, open(f"{outdir}/T_alphas.json", "w"), indent=1)
        print("cal", time.time() - t0, alphas, flush=True)
        res = sum(pool.map(task, [(s, mob, m, alphas) for s in range(30) for mob in MOBS for m in (30, 100)]), [])
        pd.DataFrame(res).to_csv(f"{outdir}/T_results.csv", index=False)
    print("done", time.time() - t0, flush=True)
