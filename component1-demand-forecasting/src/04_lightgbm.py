"""
01_lightgbm.py - the selected model: LightGBM quantile regression (Component 1)
What it does
  1. builds the features (same as the baseline version in 02_baselines.py)
  2. optional random hyperparameter search (TUNE_TRIALS) - judged on the last 20 TRAINING weeks only
  3. trains three models (2.5%, 50%, 97.5% quantiles) on the training weeks
  4. expanding-window time-series cross-validation inside the training period (3 folds)
  5. forecasts the same rolling-origin weeks as every other model -> results/lightgbm_forecasts.parquet
  6. saves the models + feature importance (used later by the SHAP step)
The calibration weeks (<= 235) and test weeks (>= 236) are never used for training or tuning.
Run from the component1-demand-forecasting folder:  python src/05_lightgbm.py
Needs:  pip install lightgbm pyarrow matplotlib
"""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ---------------- SETTINGS ----------------
HORIZON = 4
TRAIN_END, CAL_END = 193, 235          # same split as 01_preprocess.py
VAL_WEEKS = 20                         # last 20 training weeks = validation (early stopping + tuning)
QUICK = False                          # True = 20 series, few trees, 3 trials (fast code check)
HPO_TRIALS = 12                           # configurations per method for random and Bayesian search; 0 = DEFAULT only
QUANTILES = (0.025, 0.5, 0.975)
CV_FOLDS = 3                           # expanding windows inside the training period
SEED = 42
DEFAULT = dict(num_leaves=31, learning_rate=0.05, min_child_samples=40, subsample=0.8,
               colsample_bytree=0.8, reg_lambda=0.0)
SPACE = dict(num_leaves=[7, 15, 31, 63], learning_rate=[0.02, 0.03, 0.05, 0.08],
             min_child_samples=[20, 40, 80, 150], subsample=[0.6, 0.8, 1.0],
             colsample_bytree=[0.5, 0.7, 0.9], reg_lambda=[0.0, 1.0, 10.0])

GRID = dict(num_leaves=[15, 31, 63], learning_rate=[0.03, 0.05], min_child_samples=[40, 80])   # 12 combinations
N_TREES = 600
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RES, MODELS = ROOT / "results", ROOT / "models" / "lightgbm"
RES.mkdir(exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# ---------------- DATA ----------------
w = pd.read_parquet(ROOT / "data" / "processed" / "m5_weekly_ca1.parquet")
series = np.sort(w.series_id.unique())
if QUICK:
    series = series[:20]
    HPO_TRIALS, N_TREES = min(HPO_TRIALS, 3), 60
w = w[w.series_id.isin(series)].sort_values(["series_id", "time_idx"])
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
mu = Y[:, :TRAIN_END + 1].mean(1)              # series level from TRAINING weeks only
mu[mu == 0] = 1.0
print(f"Series: {S} | weeks: {T} | train <= {TRAIN_END}, calib <= {CAL_END}, test after")

# ---------------- FEATURES ----------------
def take(M, idx):
    out = np.full((M.shape[0], len(idx)), np.nan)
    ok = (idx >= 0) & (idx < M.shape[1])
    out[:, ok] = M[:, idx[ok]]
    return out

LAGS = [0, 1, 2, 3, 4, 7, 12, 25, 51]
FEATURES = ([f"sales_lag_{k}" for k in LAGS] + ["mean_4w", "mean_26w", "last_year_same_week", "last_year_3w_mean",
            "price_ratio", "promo_flag", "snap_days", "has_event", "price_change_vs_now",
            "week_of_year", "month", "weeks_ahead", "category", "log_series_level"])

def make_features(origins, h):
    """Features for forecasting week o+h from information known at week o (rows are series-major).
    Sales are divided by the series' recent level (mean of last 13 weeks) so the model also works
    when overall demand drifts - a tree cannot extrapolate beyond what it saw."""
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
    return X, y, level

def build(lo, hi):
    """Rows whose TARGET week t satisfies lo < t <= hi, for all horizons (origin >= 12 so the lags exist)."""
    Xs, ys = [], []
    for h in range(1, HORIZON + 1):
        o = np.arange(max(12, lo - h + 1), hi - h + 1)
        if len(o) == 0:
            continue
        X, y, _ = make_features(o, h)
        Xs.append(X); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys)
    ok = ~np.isnan(y)
    return X[ok], y[ok]

def pinball(y, p, q):
    d = y - p
    return np.mean(np.maximum(q * d, (q - 1) * d))

def fit_quantiles(params, Xtr, ytr, Xva=None, yva=None, n_trees=None):
    """Train one model per quantile. With a validation set: early stopping. Otherwise fixed n_trees[q]."""
    models, best = {}, {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=N_TREES if n_trees is None else n_trees[q],
                              random_state=SEED, verbose=-1, subsample_freq=1, **params)
        if Xva is not None:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
            best[q] = m.best_iteration_ or N_TREES
        else:
            m.fit(Xtr, ytr)
        models[q] = m
    return models, best

# ---------------- TRAIN / VALIDATION (same protocol as 02_baselines.py) ----------------
Xtr, ytr = build(0, TRAIN_END - VAL_WEEKS)
Xva, yva = build(TRAIN_END - VAL_WEEKS, TRAIN_END)
print(f"training rows: {len(ytr)} | validation rows: {len(yva)}")

def val_loss(models):
    return float(np.mean([pinball(yva, models[q].predict(Xva), q) for q in QUANTILES]))

# ---------------- HYPERPARAMETER SEARCH ----------------
# ---------------- HYPERPARAMETER OPTIMIZATION (WBS C1.8): grid vs random vs Bayesian ----------------
# All three methods are judged the same way: mean pinball loss of the 3 quantile models on the validation weeks
# (the last 20 TRAINING weeks). The calibration and test weeks are never touched.
import itertools
trials_log = []                                    # one row per evaluated configuration
cache = {}

def evaluate(params, method, i):
    t0 = time.time()
    key = json.dumps(params, sort_keys=True)
    if key not in cache:
        ms, _ = fit_quantiles(params, Xtr, ytr, Xva, yva)
        cache[key] = val_loss(ms)
    loss = cache[key]
    trials_log.append(dict(method=method, trial=i, val_pinball=loss, seconds=time.time() - t0, **params))
    return loss

default_loss = evaluate(dict(DEFAULT), "default", 0)
print(f"default hyperparameters: validation pinball loss {default_loss:.4f}")

if HPO_TRIALS > 0:
    # 1. GRID SEARCH - every combination of a small grid (the other settings stay at DEFAULT)
    combos = [dict(DEFAULT, **dict(zip(GRID, vals))) for vals in itertools.product(*GRID.values())]
    combos = combos[:HPO_TRIALS] if QUICK else combos
    for i, p in enumerate(combos):
        evaluate(p, "grid", i)
    print(f"grid search: {len(combos)} configurations")

    # 2. RANDOM SEARCH - HPO_TRIALS random draws from the full search space
    rng = np.random.default_rng(SEED)
    for i in range(HPO_TRIALS):
        evaluate({k: v[rng.integers(len(v))] for k, v in SPACE.items()}, "random", i)
    print(f"random search: {HPO_TRIALS} configurations")

    # 3. BAYESIAN OPTIMIZATION (TPE, Optuna) - each new trial is chosen using the results of earlier trials
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            p = {k: trial.suggest_categorical(k, v) for k, v in SPACE.items()}
            return evaluate(p, "bayesian", trial.number)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=HPO_TRIALS)
        print(f"Bayesian optimization (TPE): {HPO_TRIALS} configurations")
    except ImportError:
        print("optuna not installed -> Bayesian optimization skipped (python -m pip install optuna)")

log = pd.DataFrame(trials_log)
log.round(5).to_csv(RES / "lightgbm_hpo_trials.csv", index=False)

# ---- compare the methods ----
cmp = log[log.method != "default"].groupby("method").agg(
    configurations=("val_pinball", "size"), best_val_pinball=("val_pinball", "min"),
    mean_val_pinball=("val_pinball", "mean"), total_seconds=("seconds", "sum")).reset_index()
cmp["improvement_vs_default_%"] = (default_loss - cmp.best_val_pinball) / default_loss * 100
cmp = cmp.sort_values("best_val_pinball")
cmp.round(4).to_csv(RES / "lightgbm_hpo_comparison.csv", index=False)
print("\n===== COMPARISON OF OPTIMIZATION METHODS (lower pinball = better) =====")
print(f"default: {default_loss:.4f}")
print(cmp.round(4).to_string(index=False))

# ---- select the best hyperparameters ----
best_row = log.sort_values("val_pinball").iloc[0]
best_params = {k: best_row[k].item() if hasattr(best_row[k], "item") else best_row[k] for k in DEFAULT}
best_params = {k: (int(v) if k in ("num_leaves", "min_child_samples") else float(v)) for k, v in best_params.items()}
best_loss, best_method = float(best_row.val_pinball), best_row.method
print(f"\nselected: {best_method} | {best_params} | validation loss {best_loss:.4f} (default {default_loss:.4f})")

# ---- plot: best-so-far loss by number of configurations tried ----
fig, ax = plt.subplots(figsize=(7, 4.2))
for m, g in log[log.method != "default"].groupby("method"):
    ax.plot(np.arange(1, len(g) + 1), g.sort_values("trial").val_pinball.cummin().values, marker="o", ms=3, label=m)
ax.axhline(default_loss, ls="--", c="grey", label="default")
ax.set(title="Hyperparameter optimization: best loss so far", xlabel="configurations tried", ylabel="validation pinball loss")
ax.legend(); fig.tight_layout(); fig.savefig(RES / "lightgbm_hpo.png", dpi=150); plt.close(fig)


# ---------------- FINAL MODELS ----------------
models, best_iter = fit_quantiles(best_params, Xtr, ytr, Xva, yva)
print("best iteration per quantile:", best_iter)
for q, m in models.items():
    m.booster_.save_model(str(MODELS / f"lgbm_q{q}.txt"))
json.dump(dict(hyperparameters={k: (float(v) if not isinstance(v, int) else v) for k, v in best_params.items()},
               best_iteration={str(q): int(v) for q, v in best_iter.items()}, features=FEATURES,
               categories=list(cats), val_pinball=best_loss, default_val_pinball=default_loss, selected_method=str(best_method)),
          open(RES / "lightgbm_config.json", "w"), indent=2)

# ---------------- TIME-SERIES CROSS-VALIDATION (expanding window, training period only) ----------------
print("\n--- Expanding-window time-series cross-validation (inside training weeks) ---")
cv_rows = []
for k in range(CV_FOLDS):
    hi = TRAIN_END - (CV_FOLDS - 1 - k) * VAL_WEEKS          # e.g. 153, 173, 193
    lo = hi - VAL_WEEKS
    Xa, ya = build(0, lo)                                     # train: targets <= lo
    Xb, yb = build(lo, hi)                                    # test : targets in (lo, hi]
    ms, _ = fit_quantiles(best_params, Xa, ya, n_trees=best_iter)
    p = {q: ms[q].predict(Xb) for q in QUANTILES}
    lv = []                                                   # each row's level, to convert back to units
    for h in range(1, HORIZON + 1):
        o = np.arange(max(12, lo - h + 1), hi - h + 1)
        if len(o) == 0:
            continue
        X_, y_, L_ = make_features(o, h)
        lv.append(L_.reshape(-1)[~np.isnan(y_)])
    lv = np.concatenate(lv)
    y_u, m_u = yb * lv, p[0.5] * lv
    lo_u, hi_u = np.minimum(p[0.025], p[0.5]) * lv, np.maximum(p[0.975], p[0.5]) * lv
    naive_u = Xb[:, 0] * lv                                   # naive reference: last observed week
    cv_rows.append(dict(fold=k + 1, train_up_to_week=lo, test_weeks=f"{lo + 1}-{hi}", n=len(yb),
                        MAE=np.abs(m_u - y_u).mean(), WAPE=np.abs(m_u - y_u).sum() / y_u.sum() * 100,
                        naive_WAPE=np.abs(naive_u - y_u).sum() / y_u.sum() * 100,
                        coverage=((y_u >= lo_u) & (y_u <= hi_u)).mean() * 100))
cv = pd.DataFrame(cv_rows)
cv.round(2).to_csv(RES / "lightgbm_timeseries_cv.csv", index=False)
print(cv.round(2).to_string(index=False))
print(f"mean WAPE over folds: {cv.WAPE.mean():.1f}% (std {cv.WAPE.std():.1f}) | naive {cv.naive_WAPE.mean():.1f}%")

# ---------------- FORECASTS FOR THE EVALUATION WEEKS (same rolling origins as all other models) ----------------
O = np.arange(TRAIN_END, T - 1)
NO = len(O)
TGT = O[:, None] + np.arange(1, HORIZON + 1)[None, :]
VALID = TGT <= T - 1
SPLIT = np.where(TGT <= CAL_END, "calib", "test")
cube = {q: np.full((S, NO, HORIZON), np.nan) for q in QUANTILES}
Yt = np.full((S, NO, HORIZON), np.nan)
for h in range(1, HORIZON + 1):
    X, _, lev = make_features(O, h)
    for q in QUANTILES:
        cube[q][:, :, h - 1] = models[q].predict(X).reshape(S, NO) * lev
    for i in range(NO):
        if VALID[i, h - 1]:
            Yt[:, i, h - 1] = Y[:, TGT[i, h - 1]]
P = np.clip(cube[0.5], 0, None)
lo = np.clip(np.minimum(cube[0.025], cube[0.5]), 0, None)        # avoid crossing quantiles
hi = np.clip(np.maximum(cube[0.975], cube[0.5]), 0, None)
s_i, o_i, h_i = np.where(VALID[None].repeat(S, 0) & ~np.isnan(P))
fc = pd.DataFrame(dict(model="LightGBM", series_id=series[s_i], origin=O[o_i], horizon=h_i + 1,
                       target_idx=TGT[o_i, h_i], split=SPLIT[o_i, h_i], y=Yt[s_i, o_i, h_i],
                       yhat=P[s_i, o_i, h_i], lo=lo[s_i, o_i, h_i], hi=hi[s_i, o_i, h_i]))
fc.to_parquet(RES / "lightgbm_forecasts.parquet", index=False)
for split, g in fc.groupby("split"):
    e = g.yhat - g.y
    print(f"LightGBM {split:5s} | MAE {np.abs(e).mean():.2f} | RMSE {np.sqrt((e ** 2).mean()):.2f} | "
          f"WAPE {np.abs(e).sum() / g.y.sum() * 100:.1f}% | raw 95% coverage {((g.y >= g.lo) & (g.y <= g.hi)).mean() * 100:.1f}% "
          f"| rows {len(g)}")

# ---------------- FEATURE IMPORTANCE ----------------
imp = pd.DataFrame({q: pd.Series(models[q].booster_.feature_importance("gain"), index=FEATURES) for q in QUANTILES})
imp = (imp / imp.sum() * 100).round(2).sort_values(0.5, ascending=False)
imp.to_csv(RES / "lightgbm_feature_importance.csv")
print("\nTop features (% of gain, median model):")
print(imp[0.5].head(10).to_string())
fig, ax = plt.subplots(figsize=(7, 5))
imp[0.5].head(15)[::-1].plot.barh(ax=ax, color="#d62728")
ax.set(title="LightGBM feature importance (median model)", xlabel="% of total gain")
fig.tight_layout(); fig.savefig(RES / "lightgbm_feature_importance.png", dpi=150); plt.close(fig)

print("\nSaved: results/lightgbm_forecasts.parquet, lightgbm_config.json, lightgbm_tuning.csv, "
      "lightgbm_timeseries_cv.csv, lightgbm_feature_importance.csv/.png, models/lightgbm/*.txt")
print("Next: python src/06_conformal.py")