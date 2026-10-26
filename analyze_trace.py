"""Fig. traces (synthetic vs real 5G channel traces) and summary tables of run_trace.py.
usage: python3 analyze_trace.py results/trace results/out OUTDIR   (results/out: summary tables of run_unified)"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src, syn, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3})
d = pd.read_csv(f"{src}/T_results.csv")
b = d[d.exp == "benign"].groupby(["mob", "m", "method"])[["fpr", "false_iso"]].mean(); b.to_csv(f"{out}/T_benign.csv")
a = d[d.exp == "attack"]
for v in ["flag_rate", "det_rate", "delay"]:
    a.groupby(["mob", "m", "cal", "attack", "mag", "method"])[v].mean().to_csv(f"{out}/T_{v}.csv")
d[d.exp == "A6"].groupby(["mob", "m", "method"])[["false_iso", "flag"]].mean().to_csv(f"{out}/T_A6.csv")
d[d.exp == "benign_fi1"].groupby(["mob", "method"])[["fpr", "false_iso"]].mean().to_csv(f"{out}/T_benign_fi1.csv")
# --- figure: benign false isolation per detector, synthetic vs real (m = 100)
fi = pd.read_csv(f"{syn}/benign_false_iso.csv"); fi = fi[fi.m == 100].set_index(["scn", "method"])["mean"]
meth = ["D1lw", "D2", "D1W", "Zmax", "Lag1", "Timeout"]
lab = ["D1", "D2", "D1-W", "max-|z|", "lag-1", "timeout"]
sets = [("Gaussian", lambda k: fi[("gauss", k)]), ("autocorr.", lambda k: fi[("ar", k)]),
        ("real static", lambda k: b.loc[("static", 100, k), "false_iso"]),
        ("real driving", lambda k: b.loc[("driving", 100, k), "false_iso"])]
cols = ["#bbbbbb", "#777777", "#0072B2", "#D55E00"]
fig, ax = plt.subplots(1, 1, figsize=(3.45, 1.45))
w = 0.2; x = np.arange(len(meth))
for j, (name, f) in enumerate(sets):
    ax.bar(x + (j - 1.5) * w, [100 * f(k) for k in meth], w, color=cols[j], label=name)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("benign UEs\nisolated (%)")
ax.legend(ncol=4, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
fig.savefig(f"{out}/fig_traces.pdf", bbox_inches="tight"); fig.savefig(f"{out}/fig_traces.png", dpi=200, bbox_inches="tight")
print("ok")
