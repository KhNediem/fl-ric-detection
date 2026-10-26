"""Builds per-second spectral-efficiency series from the 5G production dataset of Raca et al. (MMSys 2020,
https://github.com/uccmisl/5Gdataset, GPL-3.0). For each trace: rows are averaged per timestamp, missing SNR
gaps of at most 10 s are interpolated, the trace is split at longer gaps (segments shorter than 400 s, or stuck: fewer than 5 distinct SNR values or one value in more than half of the samples, are dropped), and the spectral efficiency is SE = log2(1 + 10^(SNR/10)) (Shannon),
with a floor of 0.05 bit/s/Hz. Traces are grouped by mobility (Static, Driving) and saved as one list each.
usage: python3 prep_traces.py DATASET_DIR traces/se.npz
"""
import sys, glob, numpy as np, pandas as pd
root, out = sys.argv[1], sys.argv[2]
MAXGAP, MINLEN, MINVALS, MAXMODE = 10, 400, 5, 0.5
series = {"static": [], "driving": []}
stats = []
for f in sorted(glob.glob(f"{root}/**/*.csv", recursive=True)):
    if "__MACOSX" in f:
        continue
    mob = "static" if "/Static" in f else "driving"
    d = pd.read_csv(f)
    d["t"] = pd.to_datetime(d.Timestamp, format="%Y.%m.%d_%H.%M.%S")
    d["snr"] = pd.to_numeric(d.SNR, errors="coerce")
    s = d.groupby("t").snr.mean().dropna()
    if len(s) < 2:
        continue
    # split the trace where the logger has no SNR sample for more than MAXGAP s (no long filling)
    gaps = np.diff(s.index.values).astype("timedelta64[s]").astype(int)
    cuts = np.where(gaps > MAXGAP)[0]
    for seg in np.split(np.arange(len(s)), cuts + 1):
        x = s.iloc[seg]
        x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="1s")).interpolate()  # gaps <= MAXGAP s
        if len(x) < MINLEN or x.round().nunique() < MINVALS or x.round().value_counts(normalize=True).iloc[0] > MAXMODE:  # short or stuck
            continue
        se = np.maximum(np.log2(1 + 10 ** (x.values / 10)), 0.05)
        series[mob].append(se.astype(np.float32))
        stats.append((mob, len(se), np.median(se)))
np.savez_compressed(out, **{f"{k}_{i}": v for k in series for i, v in enumerate(series[k])})
st = pd.DataFrame(stats, columns=["mob", "len", "med_se"])
print(st.groupby("mob").agg(n=("len", "size"), seconds=("len", "sum"), med_se=("med_se", "median")))
