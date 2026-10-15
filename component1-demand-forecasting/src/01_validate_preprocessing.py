"""
validate_preprocessing.py - checks every step of preprocess.py
Run AFTER preprocess.py, from the component1-demand-forecasting folder:
    python src/validate_preprocessing.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ---- settings: must match preprocess.py ----
STORE = "CA_1"
SERIES_PER_CAT = 70
MAX_ZERO_SHARE = 0.30
TRAIN_END, CAL_END = 193, 235

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, RES = ROOT / "data", ROOT / "data" / "processed", ROOT / "results"
RES.mkdir(exist_ok=True)

results = []

def check(step, name, ok, detail="", soft=False):
    status = "PASS" if ok else ("WARN" if soft else "FAIL")
    results.append((step, name, status, str(detail)))
    print(f"[{status}] {step} | {name}" + (f"  ->  {detail}" if detail != "" else ""))

# ------------------------------------------------------------------
# STEP 1: file and columns
# ------------------------------------------------------------------
path = OUT / "m5_weekly_ca1.parquet"
check("1 File", "parquet file exists", path.exists(), path)
if not path.exists():
    sys.exit("Run preprocess.py first.")
w = pd.read_parquet(path)

expected_cols = ["series_id", "item_id", "dept_id", "cat_id", "store_id",
                 "week_start", "time_idx", "split", "sales",
                 "sell_price", "price_ratio", "promo_flag", "price_change",
                 "event_name", "event_type", "has_event", "snap_days",
                 "week_of_year", "month", "season"]
check("1 File", "columns match the expected list", list(w.columns) == expected_cols,
      "" if list(w.columns) == expected_cols else set(expected_cols) ^ set(w.columns))
check("1 File", "week_start is a date", pd.api.types.is_datetime64_any_dtype(w.week_start))
check("1 File", "sales and price columns are numeric",
      all(pd.api.types.is_numeric_dtype(w[c]) for c in ["sales", "sell_price", "price_ratio", "price_change"]))

# ------------------------------------------------------------------
# STEP 2: missing values, duplicates, valid ranges
# ------------------------------------------------------------------
na = w.isna().sum().sum()
check("2 Cleaning", "no missing values in any column", na == 0, f"{na} missing")
dups = w.duplicated(["series_id", "week_start"]).sum()
check("2 Cleaning", "no duplicate (series, week) rows", dups == 0, f"{dups} duplicates")
check("2 Cleaning", "no negative sales", (w.sales >= 0).all(), f"min={w.sales.min()}")
check("2 Cleaning", "all prices > 0", (w.sell_price > 0).all(), f"min={w.sell_price.min()}")
check("2 Cleaning", "no infinite values", np.isfinite(w[["price_ratio", "price_change"]]).all().all())
check("2 Cleaning", "column names are lowercase without spaces",
      all(c == c.lower() and " " not in c for c in w.columns))

# ------------------------------------------------------------------
# STEP 3: subset (store, categories, series ids)
# ------------------------------------------------------------------
check("3 Subset", f"only store {STORE}", set(w.store_id) == {STORE}, set(w.store_id))
check("3 Subset", "series_id = item_id + '_' + store_id",
      (w.series_id == w.item_id + "_" + w.store_id).all())
per_cat = w.groupby("cat_id").series_id.nunique()
check("3 Subset", "series per category <= setting", (per_cat <= SERIES_PER_CAT).all(), per_cat.to_dict())
check("3 Subset", "all 3 categories present (needed for decomposition)", len(per_cat) == 3,
      list(per_cat.index), soft=True)
check("3 Subset", "enough series to train TFT (>= 100)", w.series_id.nunique() >= 100,
      f"{w.series_id.nunique()} series", soft=True)

# ------------------------------------------------------------------
# STEP 4: weekly structure (equal length, no gaps, regular index)
# ------------------------------------------------------------------
lens = w.groupby("series_id").size()
check("4 Weekly", "every series has the same number of weeks", lens.nunique() == 1, f"{lens.iloc[0]} weeks")
check("4 Weekly", "weeks per series = 277 (expected for M5)", lens.iloc[0] == 277, lens.iloc[0], soft=True)
mm = w.groupby("series_id").time_idx.agg(["min", "max"])
check("4 Weekly", "time_idx starts at 0 for every series", (mm["min"] == 0).all())
check("4 Weekly", "time_idx ends at the same value for every series", mm["max"].nunique() == 1,
      f"max={mm['max'].iloc[0]}")
gap_ok = w.sort_values(["series_id", "week_start"]).groupby("series_id").week_start.apply(
    lambda s: (s.diff().dropna() == pd.Timedelta(days=7)).all()).all()
check("4 Weekly", "exactly 7 days between consecutive weeks (no gaps)", gap_ok)
idx_ok = w.sort_values(["series_id", "time_idx"]).groupby("series_id").time_idx.apply(
    lambda t: (t.diff().dropna() == 1).all()).all()
check("4 Weekly", "time_idx increases by exactly 1", idx_ok)
check("4 Weekly", "all series share the same week_start dates",
      w.groupby("week_start").series_id.nunique().nunique() == 1)

# ------------------------------------------------------------------
# STEP 5: time-ordered split (no leakage)
# ------------------------------------------------------------------
check("5 Split", "split labels are train / calib / test only", set(w.split) == {"train", "calib", "test"},
      set(w.split))
rule = np.select([w.time_idx <= TRAIN_END, w.time_idx <= CAL_END], ["train", "calib"], default="test")
check("5 Split", "split labels follow the time_idx rule", (w.split.values == rule).all())
t_max = w[w.split == "train"].time_idx.max()
c_min, c_max = w[w.split == "calib"].time_idx.min(), w[w.split == "calib"].time_idx.max()
te_min = w[w.split == "test"].time_idx.min()
check("5 Split", "train < calib < test in time (no overlap)", t_max < c_min and c_max < te_min,
      f"train<= {t_max}, calib {c_min}-{c_max}, test>= {te_min}")
share = w.split.value_counts(normalize=True)
check("5 Split", "split is about 70/15/15 (+-2%)",
      abs(share["train"] - 0.70) < 0.02 and abs(share["calib"] - 0.15) < 0.02 and abs(share["test"] - 0.15) < 0.02,
      share.round(3).to_dict())
check("5 Split", "every series appears in all three splits",
      (w.groupby("series_id").split.nunique() == 3).all())

# ------------------------------------------------------------------
# STEP 6: zero-heavy series filter
# ------------------------------------------------------------------
zs = w[w.split == "train"].groupby("series_id").sales.apply(lambda s: (s == 0).mean())
check("6 Zero filter", f"every series has < {MAX_ZERO_SHARE:.0%} zero weeks in training", (zs < MAX_ZERO_SHARE).all(),
      f"max share = {zs.max():.3f}")
check("6 Zero filter", "overall zero-sales share is small (< 15%)", (w.sales == 0).mean() < 0.15,
      f"{(w.sales == 0).mean():.3f}", soft=True)

# ------------------------------------------------------------------
# STEP 7: features (recomputed independently)
# ------------------------------------------------------------------
ref = w[w.split == "train"].groupby("series_id").sell_price.median()
ratio = w.sell_price / w.series_id.map(ref)
check("7 Features", "price_ratio uses the TRAINING median only (no leakage)",
      np.allclose(ratio, w.price_ratio))
check("7 Features", "promo_flag = (price_ratio < 0.95)", (w.promo_flag == (w.price_ratio < 0.95).astype(int)).all())
print(f"        promo weeks: {w.promo_flag.mean():.1%}")
pc = w.sort_values(["series_id", "time_idx"]).groupby("series_id").sell_price.pct_change() \
      .replace([np.inf, -np.inf], 0).fillna(0)
check("7 Features", "price_change matches a fresh recomputation",
      np.allclose(pc.sort_index(), w.loc[pc.index].price_change.sort_index()))
check("7 Features", "has_event = (event_name != 'none')", (w.has_event == (w.event_name != "none").astype(int)).all())
check("7 Features", "events exist and are not all 'none'", 0 < w.has_event.mean() < 1, f"{w.has_event.mean():.1%} event weeks")
check("7 Features", "'Christmas' appears in events", w.event_name.str.contains("Christmas").any(), soft=True)
check("7 Features", "week_of_year in 1-53", w.week_of_year.between(1, 53).all())
check("7 Features", "month is 1-12", w.month.astype(int).between(1, 12).all())
season_map = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
check("7 Features", "season matches month", (w.season == w.week_start.dt.month.map(season_map)).all())
check("7 Features", "snap_days between 0 and 7", w.snap_days.between(0, 7).all())

# ------------------------------------------------------------------
# STEP 8: compare against the RAW files (random sample of series)
# ------------------------------------------------------------------
print("\nComparing a random sample with the raw files (takes a moment)...")
raw_sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
cal = pd.read_csv(RAW / "calendar.csv")
cal["date"] = pd.to_datetime(cal["date"])
raw_prices = pd.read_csv(RAW / "sell_prices.csv")
week_map = cal.groupby("wm_yr_wk").date.min().rename("week_start").reset_index()
day_cols = [c for c in raw_sales.columns if c.startswith("d_")]
rng = np.random.default_rng(42)
sample = rng.choice(w.series_id.unique(), size=min(8, w.series_id.nunique()), replace=False)

sales_ok, price_ok, event_ok = [], [], []
for sid in sample:
    item = sid[: -(len(STORE) + 1)]
    row = raw_sales[(raw_sales.item_id == item) & (raw_sales.store_id == STORE)]
    s = row[day_cols].iloc[0].rename("sales").rename_axis("d").reset_index().merge(cal[["d", "wm_yr_wk"]], on="d")
    wk = s.groupby("wm_yr_wk").sales.sum().reset_index().merge(week_map, on="wm_yr_wk")
    mine = w[w.series_id == sid][["week_start", "sales", "sell_price"]]
    m = mine.merge(wk[["week_start", "sales"]], on="week_start", how="left", suffixes=("", "_raw"))
    sales_ok.append(m.sales_raw.notna().all() and (m.sales == m.sales_raw).all())
    p = raw_prices[(raw_prices.item_id == item) & (raw_prices.store_id == STORE)].merge(week_map, on="wm_yr_wk")
    m2 = mine.merge(p[["week_start", "sell_price"]], on="week_start", how="left", suffixes=("", "_raw"))
    has = m2.sell_price_raw.notna()
    price_ok.append(has.any() and np.allclose(m2.sell_price[has], m2.sell_price_raw[has]))
    ev = cal.groupby("wm_yr_wk").event_name_1.apply(lambda x: x.notna().any()).rename("raw_event").reset_index()
    e = w[w.series_id == sid][["week_start", "has_event"]].merge(week_map, on="week_start").merge(ev, on="wm_yr_wk")
    event_ok.append(((e.has_event == 1) == e.raw_event).all())

check("8 Raw match", f"weekly sales = sum of raw daily sales ({len(sample)} sampled series)", all(sales_ok),
      f"{sum(sales_ok)}/{len(sample)} match")
check("8 Raw match", "sell_price matches the raw price file", all(price_ok), f"{sum(price_ok)}/{len(sample)} match")
check("8 Raw match", "event weeks match the raw calendar", all(event_ok), f"{sum(event_ok)}/{len(sample)} match")

# ------------------------------------------------------------------
# STEP 9: quality log consistency
# ------------------------------------------------------------------
logp = RES / "data_quality_log.csv"
check("9 Log", "data_quality_log.csv exists", logp.exists())
if logp.exists():
    lg = pd.read_csv(logp).set_index("step").value
    check("9 Log", "logged 'final series' = series in the file",
          int(lg.get("final series", -1)) == w.series_id.nunique(), f"log={lg.get('final series')}, file={w.series_id.nunique()}")
    check("9 Log", "logged 'final rows' = rows in the file", int(lg.get("final rows", -1)) == len(w),
          f"log={lg.get('final rows')}, file={len(w)}")

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------
rep = pd.DataFrame(results, columns=["step", "check", "status", "detail"])
rep.to_csv(RES / "validation_report.csv", index=False)
n_fail, n_warn = (rep.status == "FAIL").sum(), (rep.status == "WARN").sum()
print(f"\n===== SUMMARY: {(rep.status == 'PASS').sum()} passed, {n_warn} warnings, {n_fail} failed =====")
if n_fail:
    print(rep[rep.status == "FAIL"][["step", "check", "detail"]].to_string(index=False))
    sys.exit(1)
print("Preprocessing validated. Report saved to results/validation_report.csv")