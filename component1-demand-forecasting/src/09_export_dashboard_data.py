"""
09_export_dashboard_data.py - bundle all Component 1 results into ONE file for the explainability dashboard (WBS C1.14)

Reads the CSV / parquet files in results/ and writes results/dashboard_data.json (about 1-2 MB):
  model comparison, hyperparameter search, conformal coverage, SHAP driver groups, decision impact, and for every product
  the test-week forecasts with intervals (1 and 4 weeks ahead, LightGBM and TFT) plus the row-level drivers.

Meaning of the per-product "drivers" block (all from the LightGBM quantile models, exact TreeSHAP):
  fbase / ftotal / fgroups : base value, final value and per-group contribution (units) of the FORECAST (median model)
  base  / total  / groups  : base value, final value and per-group contribution (units) of the INTERVAL WIDTH
                             (97.5% model minus 2.5% model), i.e. "why is the interval this wide"

By default it also builds dashboard/dashboard.html (template + this JSON) at the end.
Run from the component1-demand-forecasting folder:   python src/09_export_dashboard_data.py
Needs: 02-08 already run (results/*.csv, conformal_forecasts.parquet, shap_rows.parquet) and data/processed/m5_weekly_ca1.parquet.
"""
from pathlib import Path
import json
import subprocess
import sys
import datetime as dt
import numpy as np
import pandas as pd

# ---------------- SETTINGS ----------------
TRAIN_END, CAL_END = 193, 235
HIST_FROM = 168                         # show demand history from this week index (26 weeks before the evaluation period)
HORIZONS = [1, 4]                       # weeks ahead shown in the dashboard
MODELS = ["LightGBM", "TFT"]
INTERVAL = "CQR by horizon"             # conformal version shown as the main interval
BUILD_DASHBOARD = True                  # also run dashboard/refresh_dashboard.py when the JSON is saved
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DASH = ROOT / "dashboard"


def read(name, **kw):
    p = RES / name
    if not p.exists():
        print(f"  missing: {name} (skipped)")
        return None
    return pd.read_csv(p, **kw) if name.endswith(".csv") else pd.read_parquet(p, **kw)


def rec(df):
    return None if df is None else json.loads(df.round(4).to_json(orient="records"))


def lst(a, nd=2):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), nd) for v in a]


# ---------------- INPUT CHECK (the dashboard needs these) ----------------
need = ["model_comparison.csv", "conformal_summary.csv", "conformal_forecasts.parquet", "shap_group_importance.csv",
        "shap_rows.parquet", "decision_impact_main.csv", "decision_impact_interval_vs_constant.csv"]
absent = [n for n in need if not (RES / n).exists()]
if absent:
    print("WARNING: these results are missing, so the dashboard will be incomplete:", ", ".join(absent))
    print("         run the earlier scripts first (05_compare_models, 06_conformal, 07_shap, 08_decision_impact).")

out = {"generated": dt.date.today().isoformat(),
       "meta": {"dataset": "M5 Walmart, store CA_1, weekly", "train_end": TRAIN_END, "calib_end": CAL_END,
                "interval": INTERVAL, "horizons": HORIZONS, "models": MODELS}}

# ---------------- summary tables ----------------
mc = read("model_comparison.csv")
out["model_comparison"] = rec(mc[mc.split == "test"]) if mc is not None else None
out["significance"] = rec(read("LightGBM_vs_baselines_significance.csv"))
out["hpo"] = rec(read("lightgbm_hpo_comparison.csv"))
out["cv"] = rec(read("lightgbm_timeseries_cv.csv"))
cs = read("conformal_summary.csv")
out["conformal_summary"] = rec(cs[cs.split == "test"]) if cs is not None else None
for key, name in (("conformal_by_horizon", "conformal_by_horizon.csv"), ("conformal_by_category", "conformal_by_category.csv"),
                  ("conformal_over_time", "conformal_over_time.csv"), ("interval_score", "conformal_interval_score.csv"),
                  ("shap_groups", "shap_group_importance.csv"), ("shap_conditions", "shap_uncertainty_conditions.csv"),
                  ("shap_by_category", "shap_width_by_category.csv"), ("shap_by_month", "shap_by_month.csv"),
                  ("decision_main", "decision_impact_main.csv"), ("decision_savings", "decision_impact_savings.csv"),
                  ("decision_vs_constant", "decision_impact_interval_vs_constant.csv"),
                  ("decision_all_ratios", "decision_impact_all_ratios.csv")):
    out[key] = rec(read(name))
fi = read("shap_feature_importance.csv")
out["shap_features"] = rec(fi.head(15)) if fi is not None else None

# ---------------- per-product data ----------------
w = pd.read_parquet(ROOT / "data" / "processed" / "m5_weekly_ca1.parquet").sort_values(["series_id", "time_idx"])
T = int(w.time_idx.max()) + 1
wk = w.drop_duplicates("time_idx").set_index("time_idx").week_start
dates = {int(i): pd.Timestamp(d).strftime("%Y-%m-%d") for i, d in wk.items()}
hist_weeks = list(range(HIST_FROM, T))
test_weeks = list(range(CAL_END + 1, T))
out["weeks"] = {"history": hist_weeks, "history_dates": [dates[i] for i in hist_weeks],
                "test": test_weeks, "test_dates": [dates[i] for i in test_weeks]}
ev = w.drop_duplicates("time_idx").set_index("time_idx")
out["weeks"]["test_event"] = [str(ev.event_type.get(i, "none")) if "event_type" in ev else "none" for i in test_weeks]
out["weeks"]["test_week_of_year"] = [int(ev.week_of_year.get(i, 0)) for i in test_weeks]

cf = read("conformal_forecasts.parquet")
sh = read("shap_rows.parquet")
prods = {}
info = w.drop_duplicates("series_id").set_index("series_id")
for sid, g in w.groupby("series_id"):
    gi = g.set_index("time_idx")
    prods[sid] = dict(cat=info.loc[sid, "cat_id"], dept=info.loc[sid, "dept_id"], item=info.loc[sid, "item_id"],
                      sales=lst(gi.reindex(hist_weeks).sales.to_numpy(), 1),
                      promo=[int(v) if v == v else 0 for v in gi.reindex(hist_weeks).promo_flag.to_numpy()],
                      snap=[int(v) if v == v else 0 for v in gi.reindex(hist_weeks).snap_days.to_numpy()])
if cf is not None:
    cf = cf[(cf.split == "test") & cf.model.isin(MODELS) & cf.horizon.isin(HORIZONS) & cf.method.isin(["raw", INTERVAL])]
    for (sid, h, m), g in cf.groupby(["series_id", "horizon", "model"]):
        raw = g[g.method == "raw"].set_index("target_idx").reindex(test_weeks)
        con = g[g.method == INTERVAL].set_index("target_idx").reindex(test_weeks)
        p = prods[sid].setdefault(f"h{h}", {})
        p["y"] = prods[sid]["sales"][-len(test_weeks):]          # actual sales of the test weeks (same for every model, no gaps)
        p[m] = dict(yhat=lst(con.yhat.to_numpy(), 1), lo=lst(con.lo.to_numpy(), 1), hi=lst(con.hi.to_numpy(), 1),
                    raw_lo=lst(raw.lo.to_numpy(), 1), raw_hi=lst(raw.hi.to_numpy(), 1))
if sh is not None:
    sh = sh[(sh.split == "test") & sh.horizon.isin(HORIZONS)]
    gcols = [c for c in sh.columns if c.startswith("w::")]       # interval-width driver groups
    fcols = [c for c in sh.columns if c.startswith("f::")]       # forecast driver groups
    out["driver_groups"] = [c[3:] for c in gcols]
    for (sid, h), g in sh.groupby(["series_id", "horizon"]):
        g = g.set_index("target_idx").reindex(test_weeks)
        prods[sid].setdefault(f"h{int(h)}", {})["drivers"] = dict(
            base=lst(g.width_base.to_numpy(), 1), total=lst(g.width_units.to_numpy(), 1),
            groups={c[3:]: lst(g[c].to_numpy(), 1) for c in gcols},
            fbase=lst(g.forecast_base.to_numpy(), 1), ftotal=lst(g.forecast_units.to_numpy(), 1),
            fgroups={c[3:]: lst(g[c].to_numpy(), 1) for c in fcols})
out["products"] = prods

# ---------------- SAVE + CHECK ----------------
path = RES / "dashboard_data.json"
json.dump(out, open(path, "w"), separators=(",", ":"))
print(f"saved {path} | {len(prods)} products | {path.stat().st_size / 1e6:.2f} MB")
incomplete = [s for s, p in prods.items() if not all(k in p.get(f"h{h}", {}) for h in HORIZONS for k in (*MODELS, "drivers", "y"))]
print(f"products with complete forecasts and drivers: {len(prods) - len(incomplete)} of {len(prods)}"
      + (f"  (incomplete, e.g. {incomplete[:3]})" if incomplete else ""))

# ---------------- BUILD THE DASHBOARD PAGE ----------------
if BUILD_DASHBOARD:
    refresh = DASH / "refresh_dashboard.py"
    if refresh.exists():
        subprocess.run([sys.executable, str(refresh)], check=False)
    else:
        print(f"note: {refresh} not found, so dashboard.html was not rebuilt.")
