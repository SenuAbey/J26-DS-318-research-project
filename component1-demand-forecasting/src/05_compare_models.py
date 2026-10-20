"""
01_compare_models.py - fair comparison of TFT and all baselines (Component 1, WBS task C1.16)
Reads results/baseline_forecasts.parquet (01_baselines.py) and results/tft_forecasts.parquet (03_tft.py).
Only rows that EVERY model forecast are compared, so no model gets an easier job than another.
Run from the component1-demand-forecasting folder:  python src/04_compare_models.py
Needs:  pip install scipy matplotlib pyarrow
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

TRAIN_END = 193
FOCUS = "LightGBM"        # the model being tested against all others (selected model)
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# ---------------- LOAD AND ALIGN ----------------
files = ["baseline_forecasts.parquet", "tft_forecasts.parquet", "lightgbm_forecasts.parquet"]
f = pd.concat([pd.read_parquet(RES / n) for n in files if (RES / n).exists()], ignore_index=True)
models = list(f.model.unique())
key = ["series_id", "origin", "horizon"]
n_before = f.groupby("model").size().min()
f = f[f.groupby(key).model.transform("nunique") == len(models)].copy()
print(f"{len(models)} models: {models}")
print(f"rows per model: {len(f) // len(models)} (largest set was {n_before}; the rest dropped so all models share the same rows)")

meta = pd.read_parquet(ROOT / "data" / "processed" / "m5_weekly_ca1.parquet",
                       columns=["series_id", "cat_id", "time_idx", "sales"])
mu = meta[meta.time_idx <= TRAIN_END].groupby("series_id").sales.mean().clip(lower=1e-6)
cat = meta.drop_duplicates("series_id").set_index("series_id").cat_id
f["cat_id"] = f.series_id.map(cat)
f["mu"] = f.series_id.map(mu)
f["err"] = f.yhat - f.y

def metrics(g):
    e = g.err
    d = (g.y.abs() + g.yhat.abs()) / 2
    smape = (e.abs() / d.where(d > 0, 1)).where(d > 0, 0).mean() * 100
    nz = g.y > 0
    r = dict(n=len(g), MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()),
             WAPE=e.abs().sum() / g.y.abs().sum() * 100, sMAPE=smape,
             MAPE_nonzero=(e[nz].abs() / g.y[nz]).mean() * 100, rel_MAE=(e.abs() / g.mu).mean())
    if g.lo.notna().all():
        r["coverage_%"] = ((g.y >= g.lo) & (g.y <= g.hi)).mean() * 100
        r["width"] = (g.hi - g.lo).mean()
        r["rel_width"] = ((g.hi - g.lo) / g.mu).mean()
    return r

def table(by):
    rows = []
    for k, g in f.groupby(by):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(by, k)), **metrics(g)})
    return pd.DataFrame(rows)

main = table(["model", "split"])
main.round(3).to_csv(RES / "model_comparison.csv", index=False)
by_h = pd.DataFrame([{**dict(model=m, horizon=h), **metrics(g)}
                     for (m, h), g in f[f.split == "test"].groupby(["model", "horizon"])])
by_h.round(3).to_csv(RES / "model_comparison_by_horizon.csv", index=False)
by_cat = pd.DataFrame([{**dict(model=m, cat_id=c), **metrics(g)} for (m, c), g in f[f.split == "test"].groupby(["model", "cat_id"])])
by_cat.round(3).to_csv(RES / "model_comparison_by_category.csv", index=False)

pd.set_option("display.width", 220)
cols = ["model", "MAE", "RMSE", "WAPE", "sMAPE", "MAPE_nonzero", "rel_MAE", "coverage_%", "width"]
for split, title in (("test", "TEST SET (main result)"), ("calib", "CALIBRATION SET")):
    print(f"\n===== {title} =====")
    print(main[main.split == split][cols].sort_values("WAPE").round(2).to_string(index=False))

# ---------------- WHICH MODEL IS BEST? ----------------
t = main[main.split == "test"].set_index("model")
point_metrics = ["MAE", "RMSE", "WAPE", "sMAPE", "rel_MAE"]
ranks = t[point_metrics].rank()
print("\n===== RANK OF EACH MODEL ON THE TEST SET (1 = best) =====")
print(ranks.assign(mean_rank=ranks.mean(axis=1)).sort_values("mean_rank").round(1).to_string())
winners = t[point_metrics].idxmin()
print("\nbest model per metric:", winners.to_dict())
if FOCUS in t.index:
    n_best = int((winners == FOCUS).sum())
    print(f"{FOCUS} is best on {n_best} of {len(point_metrics)} point-accuracy metrics; its mean rank is "
          f"{ranks.loc[FOCUS].mean():.1f} out of {len(t)}")

iv = t[t["coverage_%"].notna()]
print("\n===== PREDICTION INTERVALS (target: 95% coverage, as narrow as possible) =====")
print(iv[["coverage_%", "width", "rel_width"]].assign(gap_to_95=lambda d: (d["coverage_%"] - 95).round(1))
      .sort_values("gap_to_95", key=abs).round(2).to_string())

# ---------------- IS THE DIFFERENCE REAL? (paired test over series) ----------------
if FOCUS in t.index:
    pm = f[f.split == "test"].assign(ae=lambda d: d.err.abs()).groupby(["series_id", "model"]).ae.mean().unstack()
    rows = []
    for m in pm.columns:
        if m == FOCUS:
            continue
        diff = pm[FOCUS] - pm[m]                       # negative = FOCUS has the smaller error
        p = wilcoxon(diff).pvalue if (diff != 0).any() else np.nan
        rows.append(dict(vs=m, FOCUS_better_in_pct_of_series=(diff < 0).mean() * 100,
                         mean_MAE_difference=diff.mean(), p_value=p, significant_5pct=bool(p < 0.05)))
    sig = pd.DataFrame(rows).sort_values("p_value")
    sig.round(4).to_csv(RES / f"{FOCUS}_vs_baselines_significance.csv", index=False)
    print(f"\n===== {FOCUS} vs EACH BASELINE: paired Wilcoxon test on per-series test MAE =====")
    print(f"(mean_MAE_difference < 0 means {FOCUS} is more accurate; significant = unlikely to be chance)")
    print(sig.round(4).to_string(index=False))

print("\n===== TEST WAPE % BY CATEGORY =====")
print(by_cat.pivot(index="model", columns="cat_id", values="WAPE").round(1).to_string())
if "coverage_%" in by_cat:
    print("\n===== TEST INTERVAL COVERAGE % BY CATEGORY =====")
    print(by_cat.pivot(index="model", columns="cat_id", values="coverage_%").dropna().round(1).to_string())

# ---------------- PLOTS ----------------
colors = ["#d62728" if m == FOCUS else "#7f9fbf" for m in t.sort_values("WAPE").index]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
ts = t.sort_values("WAPE")
axes[0].barh(ts.index, ts.WAPE, color=colors); axes[0].invert_yaxis()
axes[0].set(title="Test error: WAPE % (lower is better)", xlabel="WAPE %")
for name, g in by_h.groupby("model"):
    axes[1].plot(g.horizon, g.MAE, marker="o", lw=2.5 if name == FOCUS else 1, label=name)
axes[1].set(title="Test MAE by weeks ahead", xlabel="Weeks ahead", ylabel="MAE (units)")
axes[1].set_xticks(sorted(by_h.horizon.unique())); axes[1].legend(fontsize=6)
ivs = iv.sort_values("coverage_%")
axes[2].barh(ivs.index, ivs["coverage_%"], color=["#d62728" if m == FOCUS else "#e0a060" for m in ivs.index])
axes[2].axvline(95, color="grey", ls="--"); axes[2].set(title="Test interval coverage % (target 95)", xlim=(0, 100))
fig.tight_layout(); fig.savefig(RES / "model_comparison.png", dpi=150); plt.close(fig)

print("\nSaved: results/model_comparison.csv, model_comparison_by_horizon.csv, model_comparison_by_category.csv, "
      "LightGBM_vs_baselines_significance.csv, model_comparison.png")