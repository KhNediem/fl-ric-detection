"""One FL round as seen by the RIC (schematic, values consistent with the simulation: tau = 6 s, d = 4 s, g = 1 s)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42})
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.45, 1.95), sharex=True,
                              gridspec_kw={"height_ratios": [1.0, 1.05], "hspace": 0.12})
ph = [(0, 1.5, "DL", "#d9d9d9"), (1.5, 6, "local training", "#cfe2f3"),
      (6, 10, "UL upload", "#f9cb9c"), (10, 14, "idle", "#f3f3f3")]
for a, b, lab, c in ph:
    ax.add_patch(Rectangle((a, 0.3), b - a, 0.32, fc=c, ec="k", lw=0.5))
    ax.text((a + b) / 2, 0.46, lab, ha="center", va="center", fontsize=6.3)
ax.annotate("", xy=(6, 0.82), xytext=(0, 0.82), arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.text(3, 0.86, r"start delay $\tau_{i,r}$", ha="center", va="bottom", fontsize=6.5)
ax.annotate("", xy=(10, 0.1), xytext=(6, 0.1), arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.text(10.15, 0.1, r"duration $d_{i,r}$", ha="left", va="center", fontsize=6.5)
ax.text(0.95, 0.12, "free-rider: shorter $\\tau$", fontsize=5.8, color="#555555", va="center")
ax.axvline(14, color="k", lw=0.7, ls="--")
ax.text(13.85, 1.08, "window closes,\nxApp scores", fontsize=5.8, ha="right", va="top")
ax.set_ylim(-0.05, 1.12); ax.set_yticks([]); ax.set_xlim(0, 14)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(bottom=False)
t = np.arange(14); vol = np.zeros(14); vol[0:2] = [60, 40]; vol[6:10] = [1990, 2010, 2005, 1995]
ax2.bar(t + 0.5, vol, width=0.88, color=["#999999" if v < 150 else "#e69138" for v in vol], ec="k", lw=0.3)
ax2.axhline(150, color="r", lw=0.6, ls=":")
ax2.text(13.9, 260, "volume threshold", color="r", fontsize=5.8, ha="right")
ax2.text(0.1, 330, "ACKs (ignored)", fontsize=5.8, color="#555555")
ax2.text(8, 2250, r"$b_{i,r}=\sum$ intervals $\approx 8000$ kbit", ha="center", fontsize=6.3)
ax2.set_ylim(0, 2700); ax2.set_ylabel("UL RLC SDU\n(kbit per $g$)", fontsize=6.5)
ax2.set_xlabel("time in round (s); reporting interval $g=1$ s", fontsize=6.5)
ax2.set_xticks(range(0, 15, 2)); ax2.tick_params(labelsize=6)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
fig.savefig("fig_timeline.pdf", bbox_inches="tight"); fig.savefig("fig_timeline.png", dpi=220, bbox_inches="tight")
