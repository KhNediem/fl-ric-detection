"""Two further checks (same generator and code), m = 100, 30 seeds:

E  larger cohorts: N in {50, 100, 200} UEs, 20% Byzantine. D1 and D2: benign per-round FPR and false
   isolation, A1 (3 sigma) isolation and delay; A6 (Byzantine fraction 0.3, beta 1) false isolation of
   D1 with and without cohort normalization, and of D2.
F  control for D2 under contention: in the autocorrelated scenario, the autoencoder (and the per-UE
   normalization) is fitted on an i.i.d. warm-up of the same UEs (same rates and delays, i.i.d. channel
   of the same std), and calibrated and evaluated on the autocorrelated rounds, without and with A6.

usage: python3 run_more.py OUTDIR      (N is set per process through CFG before the modules use it)
"""
import sys, os, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool

M = 100


def task_E(args):
    seed, scn, n = args
    import core, unified
    core.CFG["N"] = n; unified.N = n
    from core import CFG, SCENARIOS, generate, features, Detectors
    from unified import light_scores, thr, flags, decide, iso_stats, apply_attack
    rng = np.random.default_rng([seed, SCENARIOS.index(scn), M, n])
    T = M + CFG["K_CAL"] + CFG["K_EVAL"]; ev0 = M + CFG["K_CAL"]
    par = generate(scn, T, rng); raw = features(par); perm = rng.permutation(n)
    det = Detectors(raw, M, seed)
    allu = np.ones(n, bool)
    out = []
    S0 = det.scores(raw)
    for k in ("D1lw", "D2"):
        fl = flags(S0[k], thr(S0[k], M, 0.01))
        out.append(dict(seed=seed, scn=scn, N=n, exp="E_benign", method=k, fpr=fl[:, ev0:].mean(),
                        false_iso=iso_stats(decide(fl, 3, 5), ev0, allu)[0]))
    byz = np.zeros(n, bool); byz[perm[:int(0.2 * n)]] = True
    S = det.scores(apply_attack("A1", 3, par, raw, det, byz, ev0))
    for k in ("D1lw", "D2"):
        fl = flags(S[k], thr(S[k], M, 0.01)); dr, dl = iso_stats(decide(fl, 3, 5), ev0, byz)
        out.append(dict(seed=seed, scn=scn, N=n, exp="E_A1", method=k, det_rate=dr, delay=dl))
    b6 = np.zeros(n, bool); b6[perm[:int(0.3 * n)]] = True
    lc = par["logc"].copy(); lc[:, ev0:] -= np.log(1.3); X6 = features(par, lc)
    S6 = det.scores(X6); S6["D1lw_coh"] = light_scores(X6, M, coh=True)["D1lw"]
    for k in ("D1lw", "D1lw_coh", "D2"):
        fl = flags(S6[k], thr(S6[k], M, 0.01))
        out.append(dict(seed=seed, scn=scn, N=n, exp="E_A6", method=k,
                        false_iso=iso_stats(decide(fl, 3, 5), ev0, ~b6)[0]))
    return out


def task_F(seed):
    from core import CFG, SCENARIOS, generate, features, Detectors
    from unified import thr, flags, decide, iso_stats, N
    rng = np.random.default_rng([seed, SCENARIOS.index("ar"), M])
    T = M + CFG["K_CAL"] + CFG["K_EVAL"]; ev0 = M + CFG["K_CAL"]
    par = generate("ar", T, rng); perm = rng.permutation(N)
    b6 = np.zeros(N, bool); b6[perm[:15]] = True
    iid = dict(par); iid["logc"] = par["logc"].copy()
    iid["logc"][:, :M] = CFG["SIG_C"] * np.random.default_rng([seed, 991]).standard_normal((N, M))
    out = []
    for train in ("ar", "iid"):
        for load in (False, True):
            lc = par["logc"].copy()
            if load: lc[:, ev0:] -= np.log(1.3)
            X = features(par, lc)
            Xw = features(iid, iid["logc"]) if train == "iid" else X
            Xfit = X.copy(); Xfit[:, :M] = Xw[:, :M]
            det = Detectors(Xfit, M, seed)
            S = det.scores(X)["D2"]
            fl = flags(S, thr(S, M, 0.01))
            rows = ~b6 if load else np.ones(N, bool)
            out.append(dict(seed=seed, exp="F", train=train, load=load,
                            false_iso=iso_stats(decide(fl, 3, 5), ev0, rows)[0],
                            flag=fl[rows][:, ev0:].mean()))
    return out


if __name__ == "__main__":
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True); t0 = time.time()
    with Pool(2) as pool:
        F = sum(pool.map(task_F, range(30)), [])
        pd.DataFrame(F).to_csv(f"{outdir}/F_d2_control.csv", index=False); print("F", time.time() - t0, flush=True)
        E = sum(pool.map(task_E, list(itertools.product(range(30), ["gauss", "ar"], [50, 100, 200]))), [])
        pd.DataFrame(E).to_csv(f"{outdir}/E_cohort_size.csv", index=False)
    print("done", time.time() - t0, flush=True)
