"""Runs every experiment of the paper on the unified setup and writes one long CSV.
usage: python3 run_unified.py NSEEDS OUT.csv
"""
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from core import CFG, SCENARIOS, generate, features, Detectors
from unified import *

ATTACKS = {
    "A1": [0.5, 1, 2, 3, 5],
    "A2": [0.01, 0.02, 0.05, 0.1, 0.2],
    "A3": [0.5, 1, 2, 3, 5],
    "A4": [1, 2, 3, 4, 5],
    "A5": [1, 1.5, 2, 3],
    "A8": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    "A7": [5],
}
F_BYZ = 0.2
ALPHA = CFG["ALPHA"]
METHODS = ["D1raw", "D1floor", "D1lw", "D2", "D1W", "Zmax", "Lag1", "Timeout"]
RULES = [(2, 3), (3, 5), (4, 5), (5, 10), (6, 20)]
ALPHAS = [0.005, 0.01, 0.02]


def all_scores(det, X, m):
    S = det.scores(X)
    Lt = light_scores(X, m)
    assert np.allclose(np.nan_to_num(S["D1lw"]), np.nan_to_num(Lt["D1lw"]))
    S.update({k: v for k, v in Lt.items() if k != "D1lw"})
    return S


def task(args):
    seed, scn, m = args
    rows = []
    rng = np.random.default_rng([seed, SCENARIOS.index(scn), m])
    T = m + CFG["K_CAL"] + CFG["K_EVAL"]
    ev0 = m + CFG["K_CAL"]
    par = generate(scn, T, rng)
    raw = features(par)
    det = Detectors(raw, m, seed)
    S0 = all_scores(det, raw, m)
    base = dict(seed=seed, scn=scn, m=m)
    allu = np.ones(N, bool)
    th0 = {}
    # ---------------- benign
    for k in METHODS:
        th = thr(S0[k], m, ALPHA); th0[k] = th
        fl = flags(S0[k], th)
        dec = decide(fl, 3, 5)
        fi, _ = iso_stats(dec, ev0, allu)
        row = {**base, "exp": "benign", "method": k, "fpr": fl[:, ev0:].mean(), "false_iso": fi}
        if k == "D1lw":
            q = fl[:, ev0:].mean(1)
            p = 10 * q**3 * (1 - q)**2 + 5 * q**4 * (1 - q) + q**5
            W = CFG["K_EVAL"] - 4
            row.update(q_min=q.min(), q_med=np.median(q), q_p90=np.percentile(q, 90), q_max=q.max(),
                       pred_pooled=1 - (1 - (10 * q.mean()**3)) ** W,
                       pred_perue=np.mean(1 - (1 - p) ** W))
        rows.append(row)
    Sc = light_scores(raw, m, coh=True)["D1lw"]
    thc = thr(Sc, m, ALPHA); flc = flags(Sc, thc)
    rows.append({**base, "exp": "benign", "method": "D1lw_coh", "fpr": flc[:, ev0:].mean(),
                 "false_iso": iso_stats(decide(flc, 3, 5), ev0, allu)[0]})
    perm = rng.permutation(N)
    byz = np.zeros(N, bool); byz[perm[:int(F_BYZ * N)]] = True
    # ---------------- attacks
    for name, mags in ATTACKS.items():
        for mag in mags:
            X = apply_attack(name, mag, par, raw, det, byz, ev0, duty=(2, 5))
            S = all_scores(det, X, m)
            for k in METHODS:
                th = thr(S[k], m, ALPHA)
                fl = flags(S[k], th)
                dec = decide(fl, 3, 5)
                th_m = np.quantile(S[k][~byz][:, ev0:].ravel(), 1 - ALPHA)
                dr, dl = iso_stats(dec, ev0, byz)
                rows.append({**base, "exp": "attack", "method": k, "attack": name, "mag": mag,
                             "power": (S[k][byz][:, ev0:] > th_m).mean(),
                             "flag_rate": fl[byz][:, ev0:].mean(), "det_rate": dr, "delay": dl,
                             "ben_fpr": fl[~byz][:, ev0:].mean(),
                             "ben_false_iso": iso_stats(dec, ev0, ~byz)[0]})
            if (name == "A1" and mag in (2, 3, 5)) or (name == "A8" and mag in (0.5, 1.0)):
                Sc = light_scores(X, m, coh=True)["D1lw"]
                th = thr(Sc, m, ALPHA); fl = flags(Sc, th); dec = decide(fl, 3, 5)
                dr, dl = iso_stats(dec, ev0, byz)
                rows.append({**base, "exp": "attack", "method": "D1lw_coh", "attack": name, "mag": mag,
                             "flag_rate": fl[byz][:, ev0:].mean(), "det_rate": dr, "delay": dl,
                             "ben_fpr": fl[~byz][:, ev0:].mean(),
                             "ben_false_iso": iso_stats(dec, ev0, ~byz)[0]})
    # ---------------- A6 contention
    for f in (0.1, 0.2, 0.3, 0.4):
        for beta in (1.0, 2.0):
            b6 = np.zeros(N, bool); b6[perm[:int(round(f * N))]] = True
            lc = par["logc"].copy(); lc[:, ev0:] -= np.log(1 + beta * f)
            X = features(par, lc)
            S = all_scores(det, X, m)
            S["D1lw_coh"] = light_scores(X, m, coh=True)["D1lw"]
            for k in METHODS + ["D1lw_coh"]:
                th = thr(S[k], m, ALPHA); fl = flags(S[k], th); dec = decide(fl, 3, 5)
                rows.append({**base, "exp": "A6", "method": k, "f": f, "beta": beta,
                             "ben_fpr": fl[~b6][:, ev0:].mean(),
                             "ben_false_iso": iso_stats(dec, ev0, ~b6)[0]})
    # ---------------- decision rules (D1lw), m = 100 only
    if m == 100:
        XA1 = apply_attack("A1", 3, par, raw, det, byz, ev0)
        XA7 = apply_attack("A7", 5, par, raw, det, byz, ev0, duty=(2, 5))
        sets = {"benign": S0["D1lw"], "A1": light_scores(XA1, m)["D1lw"], "A7": light_scores(XA7, m)["D1lw"]}
        for (k, n) in RULES:
            Xad = apply_attack("A7", 5, par, raw, det, byz, ev0, duty=(k - 1, n))
            sets[f"A7aware{k}{n}"] = light_scores(Xad, m)["D1lw"]
        for a in ALPHAS:
            ths = {key: thr(s, m, a) for key, s in sets.items()}
            for (k, n) in RULES:
                fi = iso_stats(decide(flags(sets["benign"], ths["benign"]), k, n), ev0, allu)[0]
                d1, l1 = iso_stats(decide(flags(sets["A1"], ths["A1"]), k, n), ev0, byz)
                d7, l7 = iso_stats(decide(flags(sets["A7"], ths["A7"]), k, n), ev0, byz)
                key = f"A7aware{k}{n}"
                d7a, l7a = iso_stats(decide(flags(sets[key], ths[key]), k, n), ev0, byz)
                rows.append({**base, "exp": "rules", "method": "D1lw", "alpha": a, "k": k, "n": n,
                             "false_iso": fi, "A1_det": d1, "A1_delay": l1, "A7_det": d7, "A7_delay": l7,
                             "A7aware_det": d7a, "A7aware_delay": l7a})
    return rows


if __name__ == "__main__":
    nseeds = int(sys.argv[1]); out = sys.argv[2]
    jobs = list(itertools.product(range(nseeds), SCENARIOS, (30, 50, 100, 200)))
    t0 = time.time(); allrows = []
    with Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(task, jobs)):
            allrows += r
            if i % 8 == 0:
                print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(allrows).to_csv(out, index=False)
    pd.DataFrame(allrows).to_csv(out, index=False)
    print("done", time.time() - t0, flush=True)
