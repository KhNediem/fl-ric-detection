"""Figures and summary tables from the unified run.
usage: python3 analyze_unified.py results.csv OUTDIR
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
df = pd.read_csv(src)
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
                     "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
SC = {"gauss": "stationary Gaussian", "ar": "autocorrelated", "drift": "slow drift", "heavy": "heavy-tailed"}
STY = {"D1raw": ("#999999", "o", "-", "D1 sample cov."),
       "D1floor": ("#0072B2", "s", "-", "D1 + variance floor"),
       "D1lw": ("#009E73", "^", "-", "D1 + Ledoit-Wolf"),
       "D2": ("#D55E00", "D", "-", "D2 autoencoder"),
       "D1W": ("#56B4E9", "v", "--", "D1 on 5-round window"),
       "Zmax": ("#E69F00", "<", "--", "max-$|z|$"),
       "Lag1": ("#CC79A7", ">", "--", "lag-1"),
       "Timeout": ("#000000", "x", ":", "pooled timeout")}


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan)
    return (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


def agg(d, by, val):
    g = d.groupby(by)[val]
    r = g.apply(lambda s: pd.Series(ci(s), index=["mean", "ci"])).unstack()
    r["n"] = g.count()
    return r.reset_index()


summ = {}
# ------------------------------------------------------------ benign tables
ben = df[df.exp == "benign"]
tb_fpr = agg(ben, ["scn", "method", "m"], "fpr"); tb_fpr.to_csv(f"{out}/benign_fpr.csv", index=False)
tb_fi = agg(ben, ["scn", "method", "m"], "false_iso"); tb_fi.to_csv(f"{out}/benign_false_iso.csv", index=False)
q = ben[ben.method == "D1lw"].groupby(["scn", "m"])[["q_min", "q_med", "q_p90", "q_max", "pred_pooled", "pred_perue", "false_iso"]].mean()
q.to_csv(f"{out}/perue_rates_D1lw.csv")

# ------------------------------------------------------------ attack tables
att = df[df.exp == "attack"]
for v in ["power", "flag_rate", "det_rate", "delay", "ben_false_iso", "ben_fpr"]:
    agg(att, ["scn", "m", "attack", "mag", "method"], v).to_csv(f"{out}/attack_{v}.csv", index=False)
a6 = df[df.exp == "A6"]
agg(a6, ["scn", "m", "f", "beta", "method"], "ben_false_iso").to_csv(f"{out}/a6_false_iso.csv", index=False)
agg(a6, ["scn", "m", "f", "beta", "method"], "ben_fpr").to_csv(f"{out}/a6_fpr.csv", index=False)
ru = df[df.exp == "rules"]
ru.groupby(["scn", "alpha", "k", "n"])[["false_iso", "A1_det", "A1_delay", "A7_det", "A7_delay", "A7aware_det", "A7aware_delay"]].mean().to_csv(f"{out}/rules.csv")

# ------------------------------------------------------------ Fig. benign
meth_b = ["D1raw", "D1floor", "D1lw", "D2", "D1W", "Zmax", "Lag1", "Timeout"]
fig, ax = plt.subplots(2, 4, figsize=(7.16, 2.55), sharex=True)
for j, s in enumerate(["gauss", "ar", "drift", "heavy"]):
    for k in meth_b:
        col, mk, ls, lab = STY[k]
        for i, (tb, sc) in enumerate([(tb_fpr, 100), (tb_fi, 100)]):
            d = tb[(tb.scn == s) & (tb.method == k)].sort_values("m")
            ax[i, j].errorbar(d.m, d["mean"] * sc, yerr=d["ci"] * sc, color=col, marker=mk, ls=ls, ms=2.8, lw=0.9,
                              capsize=1.2, label=lab)
    ax[0, j].axhline(1.0, color="k", ls=":", lw=0.8)
    ax[0, j].set_title(SC[s]); ax[1, j].set_xlabel("warm-up size $m$")
    ax[1, j].set_xscale("log"); ax[1, j].set_xticks([30, 50, 100, 200]); ax[1, j].set_xticklabels(["30", "50", "100", "200"])
    ax[1, j].minorticks_off()
ax[0, 0].set_ylabel("per-round FPR (%)"); ax[1, 0].set_ylabel("benign UEs\nisolated (%)")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.05), fontsize=6.0)
fig.tight_layout(h_pad=0.4, w_pad=0.6)
fig.savefig(f"{out}/fig_benign.pdf", bbox_inches="tight"); fig.savefig(f"{out}/fig_benign.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Fig. detectors: see analyze_extra.py (matched per-round FPR vs matched false isolation)

# ------------------------------------------------------------ Fig. contention (column)
fa = agg(a6[(a6.m == 100) & (a6.beta == 1.0)], ["scn", "f", "method"], "ben_false_iso")
f0 = tb_fi[tb_fi.m == 100].assign(f=0.0)[["scn", "f", "method", "mean", "ci"]]
fa = pd.concat([f0, fa[["scn", "f", "method", "mean", "ci"]]])
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.45), sharey=True)
for j, s in enumerate(["gauss", "ar"]):
    for k, lab, col, mk in [("D1lw", "D1 (Ledoit-Wolf)", "#009E73", "^"),
                            ("D1lw_coh", "D1 + cohort median", "#0072B2", "s"),
                            ("D2", "D2", "#D55E00", "D")]:
        d = fa[(fa.scn == s) & (fa.method == k)].sort_values("f")
        ax[j].errorbar(d.f, d["mean"] * 100, yerr=d["ci"] * 100, color=col, marker=mk, ms=2.8, lw=0.9,
                       capsize=1.2, label=lab)
    ax[j].set_title(SC[s]); ax[j].set_xlabel("Byzantine fraction $f$")
    ax[j].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax[0].set_ylabel("benign UEs\nisolated (%)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.1), fontsize=6)
fig.tight_layout(w_pad=0.4)
fig.savefig(f"{out}/fig_contention.pdf", bbox_inches="tight"); fig.savefig(f"{out}/fig_contention.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("ok")
