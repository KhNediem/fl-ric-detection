"""Prints every number quoted in the Results section from results/out/*.csv."""
import sys, pandas as pd
o = sys.argv[1] if len(sys.argv) > 1 else "results/out"
P = lambda f: pd.read_csv(f"{o}/{f}")
f, fi = P("benign_fpr.csv"), P("benign_false_iso.csv")
st = f[f.scn.isin(["gauss", "ar", "heavy"])]
print("per-round FPR range (stationary, excl. timeout): %.2f-%.2f %%" % tuple(100 * st[st.method != "Timeout"]["mean"].agg(["min", "max"])))
print("timeout FPR range: %.2f-%.2f %%" % tuple(100 * st[st.method == "Timeout"]["mean"].agg(["min", "max"])))
print("\nfalse isolation (%), 3-of-5:"); print((fi.pivot_table(index=["scn", "m"], columns="method", values="mean") * 100).round(1).to_string())
print("\nper-UE rates (D1 LW):"); print(P("perue_rates_D1lw.csv").round(4).to_string())
pw, dr, dl, fr = P("attack_power.csv"), P("attack_det_rate.csv"), P("attack_delay.csv"), P("attack_flag_rate.csv")
def show(t, name, att, mags, m=None, meth=None):
    x = t[(t.attack == att) & t.mag.isin(mags) & t.scn.isin(["gauss", "ar"])]
    if m is not None: x = x[x.m == m]
    if meth: x = x[x.method.isin(meth)]
    print(f"\n{name} {att}"); print(x.pivot_table(index=["scn", "mag", "m"], columns="method", values="mean").round(2).to_string())
show(dr, "isolation rate", "A1", [3], meth=["D1floor", "D1lw"]); show(dl, "median delay", "A1", [3], meth=["D1floor", "D1lw"])
show(dr, "isolation rate", "A2", [0.05])
for a, mg in [("A1", [2, 3]), ("A4", [3]), ("A5", [1.5]), ("A8", [0.5])]: show(pw, "power@matched FPR", a, mg, m=100)
show(fr, "per-round detection (cohort cost)", "A1", [2, 3], m=100, meth=["D1lw", "D1lw_coh"])
show(dr, "isolation rate (cohort cost)", "A1", [2, 3], m=100, meth=["D1lw", "D1lw_coh"])
show(dl, "delay (cohort cost)", "A1", [3], m=100, meth=["D1lw", "D1lw_coh"])
a6 = P("a6_false_iso.csv"); print("\nA6 false isolation (%), m=100"); print((a6[a6.m == 100].pivot_table(index=["scn", "beta", "f"], columns="method", values="mean") * 100).round(1).to_string())
print("\nrules (alpha=1%)"); r = P("rules.csv"); print(r[r.alpha == 0.01].round(3).to_string())
