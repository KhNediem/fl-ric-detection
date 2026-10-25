"""Fig. detectors (per-round detection at a pooled 1% per-round FPR vs at a matched 1% false isolation)
and summary tables of the extra experiments.  usage: python3 analyze_extra.py results/extra OUTDIR"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
                     "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3})
STY = {"D1lw": ("#009E73", "^", "-", "D1 (Ledoit-Wolf)"), "D2": ("#D55E00", "D", "-", "D2 autoencoder"),
       "D1W": ("#56B4E9", "v", "--", "D1 on 5-round window"), "Zmax": ("#E69F00", "<", "--", "max-$|z|$"),
       "Lag1": ("#CC79A7", ">", "--", "lag-1"), "Timeout": ("#000000", "x", ":", "pooled timeout")}
SC = {"gauss": "Gaussian", "ar": "autocorrelated"}
t = pd.read_csv(f"{src}/A_test.csv"); a = t[t.exp == "A"]
def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return pd.Series({"mean": x.mean() if len(x) else np.nan, "ci": 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0})
for v in ["flag_rate", "det_rate", "delay", "ben_false_iso"]:
    a.groupby(["scn", "cal", "attack", "mag", "method"])[v].apply(ci).unstack().reset_index().to_csv(f"{out}/A_{v}.csv", index=False)
b = t[t.exp == "A_benign"].groupby(["scn", "method"])[["fpr", "false_iso"]].mean(); b.to_csv(f"{out}/A_benign.csv")
fr = a.groupby(["scn", "cal", "attack", "mag", "method"]).flag_rate.apply(ci).unstack().reset_index()
cols = [(s, at) for s in ["gauss", "ar"] for at in ["A1", "A5", "A8"]]
XL = {"A1": "shift ($\\sigma$)", "A5": "amplitude ($\\sigma$)", "A8": "fraction skipped"}
TT = {"A1": "A1 delay", "A5": "A5 alternation", "A8": "A8 free-rider"}
fig, ax = plt.subplots(2, 6, figsize=(7.16, 2.6), sharey=True)
for i, cal in enumerate(["fpr1", "fi1"]):
    for j, (s, at) in enumerate(cols):
        for k, (col, mk, ls, lab) in STY.items():
            d = fr[(fr.scn == s) & (fr.cal == cal) & (fr.attack == at) & (fr.method == k)].sort_values("mag")
            ax[i, j].errorbar(d.mag, d["mean"], yerr=d["ci"], color=col, marker=mk, ls=ls, ms=2.4, lw=0.85, capsize=0.8, label=lab)
        ax[i, j].set_ylim(-0.03, 1.03)
        if i == 0: ax[i, j].set_title(f"{TT[at]}\n{SC[s]}", fontsize=6.3)
        if i == 1: ax[i, j].set_xlabel(XL[at], fontsize=6.5)
ax[0, 0].set_ylabel("per-round detection\n(1% per-round FPR)"); ax[1, 0].set_ylabel("per-round detection\n(1% false isolation)")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.06), fontsize=6.3)
fig.tight_layout(h_pad=0.3, w_pad=0.3)
fig.savefig(f"{out}/fig_detectors.pdf", bbox_inches="tight"); fig.savefig(f"{out}/fig_detectors.png", dpi=200, bbox_inches="tight")
d = pd.read_csv(f"{src}/BCD.csv")
d[d.exp == "B"].groupby(["scn", "sig_tau", "sig_d"])[["benign_fi", "A1_power", "A1_flag", "A1_det", "A1_delay", "A6_fi", "A6_fi_coh", "A6_flag"]].mean().to_csv(f"{out}/B_sensitivity.csv")
d[d.exp == "C"].groupby(["scn", "attack", "mag", "byz_frac", "coh"])[["ben_fpr", "ben_fi", "byz_iso", "byz_delay"]].mean().to_csv(f"{out}/C_cohort_byzfrac.csv")
d[d.exp == "D"].groupby(["scn", "contam"])[["power", "flag_rate", "byz_iso", "byz_delay", "ben_fi"]].mean().to_csv(f"{out}/D_contaminated.csv")
print("ok")
