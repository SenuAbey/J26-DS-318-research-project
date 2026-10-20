"""
01_conformal.py - Conformalized Quantile Regression (CQR) on the selected model (Component 1, WBS C1.9 / C1.11)
Takes the raw 95% interval [lo, hi] of the chosen model, measures how far the CALIBRATION weeks (target <= 235)
fall outside it, and widens (or narrows) the interval by that amount. The TEST weeks (target >= 236) are never
used to fit anything, so the coverage reported here is honest.
Run from the component1-demand-forecasting folder:  python src/01_conformal.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
MODEL = "LightGBM"                        # selected model (from 05_lightgbm.py -> lightgbm_forecasts.parquet)
ALSO = ["TFT"]                            # other models to conformalize for comparison ([] to skip)
ALPHA = 0.05                              # 1 - 0.05 = 95% target coverage
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# ---------------- LOAD ----------------
files = ["lightgbm_forecasts.parquet", "tft_forecasts.parquet"]          # TFT is optional (comparison only)
f = pd.concat([pd.read_parquet(RES / n) for n in files if (RES / n).exists()], ignore_index=True)
f = f[f.model.isin([MODEL] + ALSO)].copy()
meta = pd.read_parquet(ROOT / "data" / "processed" / "m5_weekly_ca1.parquet",
                       columns=["series_id", "cat_id", "time_idx", "sales"])
mu = meta[meta.time_idx <= 193].groupby("series_id").sales.mean().clip(lower=1e-6)
f["cat_id"] = f.series_id.map(meta.drop_duplicates("series_id").set_index("series_id").cat_id)
f["mu"] = f.series_id.map(mu)
f["score"] = np.maximum(f.lo - f.y, f.y - f.hi)           # CQR non-conformity score (>0: outside the interval)

def qhat(s):
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))/n empirical quantile of the scores."""
    n = len(s)
    return np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")

def conformalize(g, by=None):
    """Return (lo, hi) for all rows of one model. by=None: one global correction; by='horizon'/'cat_id': one per group."""
    cal = g[g.split == "calib"]
    if by is None:
        q = pd.Series(qhat(cal.score), index=g.index)
    else:
        qs = cal.groupby(by).score.apply(qhat)
        q = g[by].map(qs)
    return (g.lo - q).clip(lower=0), g.hi + q, q

def rolling_q(g, window=26, min_n=300):
    """Online CQR: for a forecast made at week `origin`, use only forecast errors that are already KNOWN then
    (target week <= origin) from the last `window` weeks. This follows a drifting demand level."""
    test = g[g.split == "test"]
    q = pd.Series(np.nan, index=test.index)
    for o, idx in test.groupby("origin").groups.items():
        known = g.score[(g.target_idx <= o) & (g.target_idx > o - window)]
        if len(known) >= min_n:
            q.loc[idx] = qhat(known.to_numpy())
    return q

METHODS = {"raw": None, "CQR global": None, "CQR by horizon": "horizon", "CQR by category": "cat_id",
           "CQR rolling 26w": "rolling"}
out = []
for m, g in f.groupby("model"):
    g = g.copy()
    for name, by in METHODS.items():
        h = g.copy()
        if by == "rolling":                                   # evaluated on test weeks only
            h = g[g.split == "test"].copy()
            h["q"] = rolling_q(g)
            h = h[h.q.notna()].copy()
            h["lo"], h["hi"] = (h.lo - h.q).clip(lower=0), h.hi + h.q
        elif name != "raw":
            h["lo"], h["hi"], h["q"] = conformalize(g, by)
        h["method"] = name
        out.append(h)
c = pd.concat(out, ignore_index=True)
c["covered"] = (c.y >= c.lo) & (c.y <= c.hi)
c["width"] = c.hi - c.lo
c["rel_width"] = c.width / c.mu

def summarize(g):
    return pd.Series(dict(n=len(g), coverage=g.covered.mean() * 100, width=g.width.mean(), rel_width=g.rel_width.mean()))

summ = c.groupby(["model", "method", "split"]).apply(summarize, include_groups=False).reset_index()
summ["gap_to_95"] = summ.coverage - 95
summ.round(2).to_csv(RES / "conformal_summary.csv", index=False)
pd.set_option("display.width", 200)
print("===== COVERAGE BEFORE / AFTER CONFORMAL (target 95%) =====")
print(summ[summ.split == "test"].drop(columns="split").round(2).to_string(index=False))
print("\ncalibration-set coverage (should be about 95% for the CQR rows by construction):")
print(summ[summ.split == "calib"].drop(columns="split").round(2).to_string(index=False))

t = c[(c.split == "test") & (c.model == MODEL)]
by_h = t.groupby(["method", "horizon"]).covered.mean().mul(100).unstack().round(1)
by_c = t.groupby(["method", "cat_id"]).covered.mean().mul(100).unstack().round(1)
print(f"\n===== {MODEL}: TEST COVERAGE % BY WEEKS AHEAD =====\n{by_h.to_string()}")
print(f"\n===== {MODEL}: TEST COVERAGE % BY CATEGORY =====\n{by_c.to_string()}")

# stability over time: test weeks in 4 equal blocks (does coverage drift as the demand level drifts?)
t = t.assign(block=pd.qcut(t.target_idx, 4, labels=False, duplicates="drop"))
by_t = t.groupby(["method", "block"]).covered.mean().mul(100).unstack().round(1)
print(f"\n(note: rolling rows are only the test weeks whose origin has enough observed errors)\n===== {MODEL}: TEST COVERAGE % OVER TIME (block 0 = first test weeks, 3 = last) =====\n{by_t.to_string()}")

# per-series coverage spread (marginal coverage can hide series that are badly covered)
for mth in ("CQR global", "CQR rolling 26w"):
    ps = t[t.method == mth].groupby("series_id").covered.mean() * 100
    print(f"per-series coverage ({mth}): median {ps.median():.0f}%, 10th pct {ps.quantile(.1):.0f}%, "
          f"series below 80%: {(ps < 80).sum()} of {len(ps)}")

by_h.to_csv(RES / "conformal_by_horizon.csv")
by_c.to_csv(RES / "conformal_by_category.csv")
by_t.to_csv(RES / "conformal_over_time.csv")
c.drop(columns=["score"]).to_parquet(RES / "conformal_forecasts.parquet", index=False)


# ---------------- INTERVAL SCORE ON SHARED ROWS (which model has the better intervals?) ----------------
# Interval score (Winkler): width + a penalty of 2/alpha times the miss distance whenever y falls outside.
# It rewards narrow intervals that still cover, so ONE number says which intervals are better (lower = better).
key = ["series_id", "origin", "horizon"]
tt = c[c.split == "test"].copy()
tt = tt[tt.groupby(["method"] + key).model.transform("nunique") == tt.model.nunique()]      # rows ALL models forecast
tt["miss"] = np.maximum(tt.lo - tt.y, 0) + np.maximum(tt.y - tt.hi, 0)
tt["interval_score"] = tt.width + (2 / ALPHA) * tt.miss
iscore = tt.groupby(["model", "method"]).agg(rows=("y", "size"), coverage=("covered", lambda x: x.mean() * 100),
                                             width=("width", "mean"), interval_score=("interval_score", "mean")).reset_index()
iscore.round(2).to_csv(RES / "conformal_interval_score.csv", index=False)
print("\n===== INTERVAL SCORE ON SHARED TEST ROWS (lower = better; same rows for every model) =====")
print(iscore.sort_values("interval_score").round(2).to_string(index=False))
ih = tt[tt.method == "CQR by horizon"].groupby(["model", "horizon"]).interval_score.mean().unstack().round(1)
print("\ninterval score by weeks ahead (CQR by horizon):\n" + ih.to_string())

# ---------------- PLOTS ----------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
s = summ[(summ.split == "test") & (summ.model == MODEL)].set_index("method").loc[list(METHODS)]
ax[0].bar(s.index, s.coverage, color=["#9aa5b1"] + ["#d62728"] * (len(METHODS) - 1))
ax[0].axhline(95, ls="--", c="grey"); ax[0].set(title=f"{MODEL}: test coverage %", ylim=(70, 100))
ax[0].tick_params(axis="x", rotation=20)
ax[1].bar(s.index, s.width, color=["#9aa5b1"] + ["#d62728"] * (len(METHODS) - 1))
ax[1].set(title="Mean interval width (units; narrower = sharper)"); ax[1].tick_params(axis="x", rotation=20)
for mth, row in by_h.iterrows():
    ax[2].plot(row.index, row.values, marker="o", label=mth)
ax[2].axhline(95, ls="--", c="grey"); ax[2].set(title="Test coverage % by weeks ahead", xlabel="Weeks ahead")
ax[2].set_xticks(list(by_h.columns)); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(RES / "conformal_coverage.png", dpi=150); plt.close(fig)

# example series: forecasts and conformal interval on test weeks
ex = t[t.method == "CQR global"].query("horizon == 1")
sid = ex.groupby("series_id").y.mean().sort_values().index[len(ex.series_id.unique()) // 2]
e = ex[ex.series_id == sid].sort_values("target_idx")
fig, a = plt.subplots(figsize=(10, 4))
a.fill_between(e.target_idx, e.lo, e.hi, alpha=.25, color="#d62728", label="95% conformal interval")
a.plot(e.target_idx, e.yhat, c="#d62728", label="forecast"); a.plot(e.target_idx, e.y, c="k", lw=1, label="actual")
a.set(title=f"{sid} - 1-week-ahead forecast, test weeks", xlabel="week index", ylabel="units"); a.legend()
fig.tight_layout(); fig.savefig(RES / "conformal_example.png", dpi=150); plt.close(fig)
print("\nSaved: results/conformal_summary.csv, conformal_by_horizon.csv, conformal_by_category.csv, "
      "conformal_over_time.csv, conformal_interval_score.csv, conformal_forecasts.parquet, conformal_coverage.png, conformal_example.png")