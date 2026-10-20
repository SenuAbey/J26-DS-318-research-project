"""
08_decision_impact.py - Does the uncertainty interval improve stocking decisions? (Component 1, WBS C1.12)

Simulation. Every week, for every product, a stock level Q is set when the forecast is made (h weeks before the demand y):
    overstock = max(Q - y, 0)  costs HOLD per unit      (holding / waste)
    shortage  = max(y - Q, 0)  costs SHORT per unit     (lost sale)
SHORT / HOLD is the cost ratio. A ratio of 4 means a missed sale hurts four times as much as an unsold unit.

Policies (the buffer size k is tuned on the CALIBRATION weeks only; results are judged on the TEST weeks):
  A  Point forecast               Q = yhat
  B  Point + constant buffer      Q = yhat + k * sigma_series          sigma = that model's mean absolute error on the calibration weeks
  C  Point + interval buffer      Q = yhat + k * (upper - yhat)        upper = conformal upper bound (CQR by horizon)
Policy B is what a planner does without an interval (a fixed safety stock per product). Policy C uses the interval, whose
width changes with the situation (promotion, season, horizon). If C beats B at the same tuning freedom, the interval adds value.
A moving-average forecast with policy A/B is included as the 'no machine learning' benchmark.

Run from the component1-demand-forecasting folder:  python src/08_decision_impact.py
Needs results/conformal_forecasts.parquet (06_conformal.py) and results/baseline_forecasts.parquet (02_baselines.py).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ---------------- SETTINGS ----------------
MODELS = ["LightGBM", "TFT"]              # machine-learning models with conformal intervals
INTERVAL_METHOD = "CQR by horizon"        # which conformal version supplies the upper bound
BENCH = "Moving average (4w)"             # simple benchmark (point forecast only)
HOLD = 1.0                                # cost of one unsold unit
RATIOS = [1, 2, 4, 9, 19]                 # SHORT / HOLD values to test (service level at the optimum = ratio / (1 + ratio))
MAIN_RATIO = 4
K_GRID = np.round(np.arange(0, 3.0001, 0.05), 2)
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# ---------------- LOAD ----------------
cf = pd.read_parquet(RES / "conformal_forecasts.parquet")
cf = cf[(cf.method == INTERVAL_METHOD) & cf.model.isin(MODELS)]
bf = pd.read_parquet(RES / "baseline_forecasts.parquet")
bf = bf[bf.model == BENCH]
cols = ["model", "series_id", "origin", "horizon", "target_idx", "split", "y", "yhat", "lo", "hi"]
d = pd.concat([cf[cols], bf[cols]], ignore_index=True)
key = ["series_id", "origin", "horizon"]
d = d[d.groupby(key).model.transform("nunique") == d.model.nunique()].copy()          # rows every model has
print(f"models: {list(d.model.unique())} | rows per model: {len(d) // d.model.nunique()} "
      f"(calib {int((d.split == 'calib').sum() / d.model.nunique())}, test {int((d.split == 'test').sum() / d.model.nunique())})")
d["abs_err"] = (d.yhat - d.y).abs()
sig = d[d.split == "calib"].groupby(["model", "series_id", "horizon"]).abs_err.mean().rename("sigma")
d = d.join(sig, on=["model", "series_id", "horizon"])
d["sigma"] = d.sigma.fillna(d.groupby(["model", "horizon"]).sigma.transform("mean"))
d["up"] = np.maximum(d.hi - d.yhat, 0)                                                # room between forecast and upper bound

def stock(df, policy, k):
    if policy == "A":
        return df.yhat.to_numpy()
    if policy == "B":
        return (df.yhat + k * df.sigma).to_numpy()
    return (df.yhat + k * df.up).to_numpy()

def cost(df, Q, ratio):
    y = df.y.to_numpy()
    over, short = np.maximum(Q - y, 0), np.maximum(y - Q, 0)
    return HOLD * over + ratio * HOLD * short, over, short

def tune(df_cal, policy, ratio):
    """Pick the buffer k with the lowest total cost on the calibration weeks."""
    best = min(K_GRID, key=lambda k: cost(df_cal, stock(df_cal, policy, k), ratio)[0].sum())
    return float(best)

POL_NAME = {"A": "A. Point forecast", "B": "B. Point + constant buffer", "C": "C. Point + interval buffer"}
rows, per_series = [], []
for ratio in RATIOS:
    for m in d.model.unique():
        g = d[d.model == m]
        cal, te = g[g.split == "calib"], g[g.split == "test"]
        for pol in ("A", "B", "C"):
            if pol == "C" and m == BENCH:
                continue                                  # the benchmark has no interval
            k = 0.0 if pol == "A" else tune(cal, pol, ratio)
            c, over, short = cost(te, stock(te, pol, k), ratio)
            rows.append(dict(ratio=ratio, model=m, policy=POL_NAME[pol], k=k, cost_per_unit_demand=c.sum() / te.y.sum(),
                             service_level_pct=(short == 0).mean() * 100, fill_rate_pct=(1 - short.sum() / te.y.sum()) * 100,
                             overstock_units_per_week=over.mean(), lost_units_per_week=short.mean()))
            if ratio == MAIN_RATIO:
                per_series.append(pd.DataFrame(dict(model=m, policy=POL_NAME[pol], series_id=te.series_id.to_numpy(), cost=c))
                                  .groupby(["model", "policy", "series_id"]).cost.mean().reset_index())
res = pd.DataFrame(rows)
res.round(4).to_csv(RES / "decision_impact_all_ratios.csv", index=False)
ps = pd.concat(per_series)

pd.set_option("display.width", 220)
main = res[res.ratio == MAIN_RATIO].drop(columns="ratio").sort_values("cost_per_unit_demand")
print(f"\n===== TEST WEEKS, SHORTAGE COST = {MAIN_RATIO} x HOLDING COST (lower cost = better) =====")
print("cost_per_unit_demand = total cost / total demand; k = buffer size chosen on the calibration weeks")
print(main.round(3).to_string(index=False))
main.round(4).to_csv(RES / "decision_impact_main.csv", index=False)

# does the interval beat the constant buffer? (paired over products)
print(f"\n===== INTERVAL BUFFER (C) vs CONSTANT BUFFER (B), ratio {MAIN_RATIO}: per-product cost, paired Wilcoxon test =====")
sig_rows = []
for m in MODELS:
    b = ps[(ps.model == m) & (ps.policy == POL_NAME["B"])].set_index("series_id").cost
    c_ = ps[(ps.model == m) & (ps.policy == POL_NAME["C"])].set_index("series_id").cost
    dd = (c_ - b).dropna()
    p = wilcoxon(dd).pvalue if (dd != 0).any() else np.nan
    sig_rows.append(dict(model=m, C_cheaper_in_pct_of_products=(dd < 0).mean() * 100,
                         mean_cost_B=b.mean(), mean_cost_C=c_.mean(), cost_change_pct=(c_.mean() / b.mean() - 1) * 100, p_value=p))
sg = pd.DataFrame(sig_rows)
sg.round(4).to_csv(RES / "decision_impact_interval_vs_constant.csv", index=False)
print(sg.round(4).to_string(index=False))

print("\n===== COST PER UNIT OF DEMAND BY COST RATIO (test weeks) =====")
piv = res.pivot_table(index=["model", "policy"], columns="ratio", values="cost_per_unit_demand").round(3)
print(piv.to_string())
sav = []
for m in MODELS:
    for r in RATIOS:
        x = res[(res.ratio == r) & (res.model == m)].set_index("policy").cost_per_unit_demand
        base_ma = res[(res.ratio == r) & (res.model == BENCH) & (res.policy == POL_NAME["B"])].cost_per_unit_demand.iloc[0]
        sav.append(dict(model=m, ratio=r, interval_vs_constant_buffer_pct=(x[POL_NAME["C"]] / x[POL_NAME["B"]] - 1) * 100,
                        interval_vs_point_pct=(x[POL_NAME["C"]] / x[POL_NAME["A"]] - 1) * 100,
                        interval_vs_moving_average_buffer_pct=(x[POL_NAME["C"]] / base_ma - 1) * 100))
sav = pd.DataFrame(sav)
sav.round(2).to_csv(RES / "decision_impact_savings.csv", index=False)
print("\n===== COST CHANGE OF THE INTERVAL POLICY (negative = cheaper) =====")
print(sav.round(1).to_string(index=False))

# ---------------- PLOTS ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
pp = main.pivot(index="policy", columns="model", values="cost_per_unit_demand")
pp.plot.barh(ax=ax[0]); ax[0].set(title=f"Cost per unit of demand (shortage = {MAIN_RATIO} x holding)", xlabel="lower is better")
ax[0].invert_yaxis(); ax[0].legend(fontsize=7)
for m in MODELS[:1]:
    for pol in ("A", "B", "C"):
        s_ = res[(res.model == m) & (res.policy == POL_NAME[pol])].sort_values("ratio")
        ax[1].plot(s_.ratio, s_.cost_per_unit_demand, marker="o", label=POL_NAME[pol])
ax[1].set(title=f"{MODELS[0]}: cost by shortage/holding ratio", xlabel="shortage cost / holding cost", ylabel="cost per unit of demand")
ax[1].set_xscale("log"); ax[1].legend(fontsize=7)
m0 = MODELS[0]
te0 = d[(d.model == m0) & (d.split == "test")]
for pol, colr in (("B", "#7f9fbf"), ("C", "#d62728")):
    pts = []
    for k in K_GRID:
        c, over, short = cost(te0, stock(te0, pol, k), 1)
        pts.append(((short == 0).mean() * 100, over.mean()))
    pts = np.array(pts)
    ax[2].plot(pts[:, 0], pts[:, 1], color=colr, label=POL_NAME[pol])
ax[2].set(title=f"{m0}: service level vs excess stock (every buffer size)", xlabel="weeks without a stockout (%)",
          ylabel="average overstock (units per product-week)"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(RES / "decision_impact.png", dpi=150); plt.close(fig)
print("\nSaved: results/decision_impact_main.csv, decision_impact_all_ratios.csv, decision_impact_interval_vs_constant.csv, "
      "decision_impact_savings.csv, decision_impact.png")