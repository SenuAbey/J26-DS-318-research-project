"""
07_shap.py - What drives the FORECAST and what drives the UNCERTAINTY? (Component 1, WBS C1.10)
The selected LightGBM has one model for the 2.5% quantile, one for the median and one for the 97.5% quantile.
TreeSHAP is exact and additive, so the SHAP values of the 95% interval width are simply
        SHAP(width) = SHAP(97.5% model) - SHAP(2.5% model)
No surrogate model is needed. Features are grouped into the drivers named in the proposal:
seasonality, promotions/price, SNAP days (purchasing-power timing - the only 'economic' signal in M5), events, ...
Everything is computed on the calibration + test weeks (the weeks after training).
Run from the component1-demand-forecasting folder:  python src/07_shap.py
Needs: results/lightgbm_config.json and models/lightgbm/*.txt from 04_lightgbm.py. (Optional: pip install shap for a beeswarm plot)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

HORIZON = 4
TRAIN_END, CAL_END = 193, 235
QUANTILES = (0.025, 0.5, 0.975)
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RES, MODELS = ROOT / "results", ROOT / "models" / "lightgbm"
cfg = json.load(open(RES / "lightgbm_config.json"))
FEATURES = cfg["features"]

GROUPS = {
    "Recent demand (lags, averages)": [f for f in FEATURES if f.startswith("sales_lag") or f in ("mean_4w", "mean_26w")],
    "Seasonality (week, month, last year)": ["week_of_year", "month", "last_year_same_week", "last_year_3w_mean"],
    "Promotion and price": ["price_ratio", "promo_flag", "price_change_vs_now"],
    "SNAP days (purchasing-power timing)": ["snap_days"],
    "Calendar events": ["has_event"],
    "Product category and size": ["category", "log_series_level"],
    "Forecast horizon": ["weeks_ahead"],
}
F2G = {f: g for g, fs in GROUPS.items() for f in fs}
missing = [f for f in FEATURES if f not in F2G]
assert not missing, f"features without a group: {missing}"

# ---------------- DATA AND FEATURES (same as 04_lightgbm.py) ----------------
w = pd.read_parquet(ROOT / "data" / "processed" / "m5_weekly_ca1.parquet").sort_values(["series_id", "time_idx"])
series = np.sort(w.series_id.unique())
S, T = len(series), int(w.time_idx.max()) + 1

def mat(col):
    return w.pivot(index="series_id", columns="time_idx", values=col).loc[series].to_numpy(float)

Y, PR, PM, SN, EV = mat("sales"), mat("price_ratio"), mat("promo_flag"), mat("snap_days"), mat("has_event")
cat_name = w.groupby("series_id").cat_id.first().loc[series].to_numpy()
cats = np.sort(np.unique(cat_name))
cat_code = np.searchsorted(cats, cat_name)
t_vec = w.drop_duplicates("time_idx").sort_values("time_idx")
WOY = t_vec.week_of_year.to_numpy(float)
MONTH = pd.to_datetime(t_vec.week_start).dt.month.to_numpy(float)
mu = Y[:, :TRAIN_END + 1].mean(1)
mu[mu == 0] = 1.0

def take(M, idx):
    out = np.full((M.shape[0], len(idx)), np.nan)
    ok = (idx >= 0) & (idx < M.shape[1])
    out[:, ok] = M[:, idx[ok]]
    return out

LAGS = [0, 1, 2, 3, 4, 7, 12, 25, 51]

def make_features(origins, h):
    origins = np.asarray(origins)
    t = origins + h
    n = len(origins)
    level = np.nanmean(np.stack([take(Y, origins - j) for j in range(13)]), axis=0)
    level = np.maximum(level, 0.1 * mu[:, None])
    f = [take(Y, origins - k) / level for k in LAGS]
    for win in (4, 26):
        f.append(np.nanmean(np.stack([take(Y, origins - j) for j in range(win)]), axis=0) / level)
    f.append(take(Y, t - 52) / level)
    f.append(np.nanmean(np.stack([take(Y, t - 52 + j) for j in (-1, 0, 1)]), axis=0) / level)
    f += [take(PR, t), take(PM, t), take(SN, t), take(EV, t), take(PR, t) / take(PR, origins)]
    f.append(np.broadcast_to(np.where((t >= 0) & (t < T), WOY[np.clip(t, 0, T - 1)], np.nan), (S, n)))
    f.append(np.broadcast_to(np.where((t >= 0) & (t < T), MONTH[np.clip(t, 0, T - 1)], np.nan), (S, n)))
    f.append(np.full((S, n), float(h)))
    f.append(np.broadcast_to(cat_code[:, None].astype(float), (S, n)))
    f.append(np.broadcast_to(np.log(mu)[:, None], (S, n)))
    X = np.stack([a.reshape(-1) for a in f], axis=1)
    y = (take(Y, t) / level).reshape(-1)
    return X, y, level.reshape(-1)

O = np.arange(TRAIN_END, T - 1)
Xs, meta = [], []
for h in range(1, HORIZON + 1):
    X, y, lev = make_features(O, h)
    tgt = np.tile(O + h, S)
    ok = ~np.isnan(y)
    Xs.append(X[ok])
    meta.append(pd.DataFrame(dict(series_id=np.repeat(series, len(O)), origin=np.tile(O, S), horizon=h, target_idx=tgt,
                                  level=lev, y=(y * lev), cat_id=np.repeat(cat_name, len(O)))).loc[ok])
X = np.concatenate(Xs)
D = pd.concat(meta, ignore_index=True)
D["split"] = np.where(D.target_idx <= CAL_END, "calib", "test")
print(f"rows: {len(D)} | features: {len(FEATURES)} | series: {S}")

# ---------------- EXACT TREE-SHAP FROM LIGHTGBM ----------------
booster = {q: lgb.Booster(model_file=str(MODELS / f"lgbm_q{q}.txt")) for q in QUANTILES}
C, base, pred = {}, {}, {}
for q in QUANTILES:
    c = booster[q].predict(X, pred_contrib=True)          # last column = expected value
    C[q], base[q] = c[:, :-1], c[:, -1]
    pred[q] = booster[q].predict(X)
    err = np.abs(C[q].sum(1) + base[q] - pred[q]).max()
    assert err < 1e-6, f"SHAP additivity check failed for q={q}: {err}"
print("additivity check passed (SHAP values + base value = model prediction)")

W = C[0.975] - C[0.025]                                    # SHAP of the interval width (relative to recent level)
rel_width = pred[0.975] - pred[0.025]                      # predicted 95% width as a multiple of the recent level
lev = D.level.to_numpy()[:, None]

# ---------------- FEATURE AND GROUP IMPORTANCE ----------------
imp = pd.DataFrame(dict(feature=FEATURES,
                        forecast=np.abs(C[0.5]).mean(0),
                        width_relative=np.abs(W).mean(0),
                        width_units=np.abs(W * lev).mean(0)))
imp["group"] = imp.feature.map(F2G)
for col in ("forecast", "width_relative", "width_units"):
    imp[f"{col}_share_%"] = imp[col] / imp[col].sum() * 100
imp.sort_values("width_relative", ascending=False).round(4).to_csv(RES / "shap_feature_importance.csv", index=False)
share_cols = ["forecast_share_%", "width_relative_share_%", "width_units_share_%"]
grp = imp.groupby("group")[share_cols].sum().sort_values("width_relative_share_%", ascending=False)
grp.round(1).to_csv(RES / "shap_group_importance.csv")
pd.set_option("display.width", 200)
print("\n===== SHARE OF EACH DRIVER GROUP (%) =====")
print("forecast = what moves the point forecast | width_relative = what moves the 95% interval width, relative to demand level")
print("width_units = same, in units of product (large-volume products count more)")
print(grp.round(1).to_string())
print("\ntop features for the interval width:")
print(imp.sort_values("width_relative", ascending=False)[["feature", "group", "width_relative_share_%"]].head(8).round(1).to_string(index=False))

# by category
rows = []
for cname in cats:
    sel = (D.cat_id == cname).to_numpy()
    a = np.abs(W[sel]).mean(0)
    s = pd.Series(a, index=FEATURES).groupby(F2G).sum()
    rows.append((s / s.sum() * 100).rename(cname))
by_cat = pd.concat(rows, axis=1).round(1)
by_cat.to_csv(RES / "shap_width_by_category.csv")
print("\n===== INTERVAL-WIDTH DRIVERS BY PRODUCT CATEGORY (%) =====")
print(by_cat.to_string())

# ---------------- DOES UNCERTAINTY RISE IN PROMOTION / SNAP / EVENT WEEKS? ----------------
F = {f: X[:, i] for i, f in enumerate(FEATURES)}
flags = {"promotion week (promo_flag = 1)": F["promo_flag"] > 0, "SNAP days in the week (snap_days > 0)": F["snap_days"] > 0,
         "calendar event in the week": F["has_event"] > 0}
dirs = []
for name, m in flags.items():
    if m.sum() == 0 or (~m).sum() == 0:
        continue
    dirs.append(dict(condition=name, weeks_with=int(m.sum()), mean_relative_width_with=rel_width[m].mean(),
                     mean_relative_width_without=rel_width[~m].mean(),
                     change_pct=(rel_width[m].mean() / rel_width[~m].mean() - 1) * 100,
                     mean_forecast_ratio_with=pred[0.5][m].mean(), mean_forecast_ratio_without=pred[0.5][~m].mean()))
dirs = pd.DataFrame(dirs)
dirs.round(3).to_csv(RES / "shap_uncertainty_conditions.csv", index=False)
print("\n===== IS THE INTERVAL WIDER IN SPECIAL WEEKS? (relative width = interval width / recent demand level) =====")
print(dirs.round(2).to_string(index=False))

month_tab = pd.DataFrame(dict(month=F["month"].astype(int), width=rel_width, forecast=pred[0.5])).groupby("month").mean().round(3)
month_tab.to_csv(RES / "shap_by_month.csv")
print("\nrelative interval width by month of the forecast week:")
print(month_tab.T.to_string())

# ---------------- PLOTS ----------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
g1 = grp.sort_values("width_relative_share_%")
ax[0].barh(g1.index, g1["forecast_share_%"], color="#7f9fbf"); ax[0].set(title="What drives the FORECAST (% of SHAP importance)")
ax[1].barh(g1.index, g1["width_relative_share_%"], color="#d62728"); ax[1].set(title="What drives the UNCERTAINTY (interval width, %)")
fig.tight_layout(); fig.savefig(RES / "shap_groups.png", dpi=150); plt.close(fig)

top = imp.sort_values("width_relative", ascending=False).head(12)[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top.feature, top["width_relative_share_%"], color="#d62728")
ax.set(title="Top features driving the 95% interval width", xlabel="% of SHAP importance"); fig.tight_layout()
fig.savefig(RES / "shap_width_features.png", dpi=150); plt.close(fig)

rng = np.random.default_rng(SEED)
samp = rng.choice(len(X), size=min(4000, len(X)), replace=False)
try:
    import shap
    shap.summary_plot(W[samp], X[samp], feature_names=FEATURES, show=False, max_display=12)
    plt.title("SHAP values of the interval width (right = wider interval)")
    plt.tight_layout(); plt.savefig(RES / "shap_width_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
except Exception as e:                                      # the shap package is optional
    print("beeswarm skipped (pip install shap to enable):", repr(e)[:100])

top4 = imp.sort_values("width_relative", ascending=False).feature.head(4).tolist()
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for a, fname in zip(axes.ravel(), top4):
    i = FEATURES.index(fname)
    a.scatter(X[samp, i], W[samp, i], s=3, alpha=.35, color="#d62728")
    a.axhline(0, c="grey", lw=.8); a.set(title=fname, xlabel="feature value", ylabel="SHAP of interval width")
fig.tight_layout(); fig.savefig(RES / "shap_width_dependence.png", dpi=150); plt.close(fig)
print("\nSaved: results/shap_feature_importance.csv, shap_group_importance.csv, shap_width_by_category.csv, "
      "shap_uncertainty_conditions.csv, shap_by_month.csv, shap_groups.png, shap_width_features.png, shap_width_dependence.png")


# ---------------- ROW-LEVEL CONTRIBUTIONS BY DRIVER GROUP (used by the dashboard) ----------------
# For every forecast: how many units of the interval width does each driver group add or remove?
# width_units = width_base + sum of the group contributions (exact, because TreeSHAP is additive).
grp_names = list(GROUPS)
def by_group(M):
    return np.stack([M[:, [FEATURES.index(f) for f in GROUPS[g]]].sum(1) for g in grp_names], axis=1)
lev1 = D.level.to_numpy()
Wg, Fg = by_group(W) * lev, by_group(C[0.5]) * lev
rows_out = D[["series_id", "origin", "horizon", "target_idx", "split"]].copy()
rows_out["level"] = lev1
rows_out["width_base"] = (base[0.975] - base[0.025]) * lev1
rows_out["width_units"] = rel_width * lev1
rows_out["forecast_base"] = base[0.5] * lev1
rows_out["forecast_units"] = pred[0.5] * lev1
for i, g in enumerate(grp_names):
    rows_out[f"w::{g}"] = Wg[:, i]
    rows_out[f"f::{g}"] = Fg[:, i]
chk = np.abs(rows_out.width_base + rows_out[[f"w::{g}" for g in grp_names]].sum(axis=1) - rows_out.width_units).max()
assert chk < 1e-3, f"row-level width decomposition does not add up: {chk}"
rows_out.round(3).to_parquet(RES / "shap_rows.parquet", index=False)
print("saved results/shap_rows.parquet (row-level driver contributions; decomposition adds up exactly)")