"""
01_preprocess.py - M5 daily data -> clean weekly dataset (Component 1)
  Part A: checks on the RAW data (before any change)
  Part B: preprocessing, with quick sanity checks after each step
Full validation of the result is in validate_preprocessing.py (run it after this file).
Run from the component1-demand-forecasting folder:  python src/01_preprocessing.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ---------------- SETTINGS (keep identical to validate_preprocessing.py) ----------------
STORE = "CA_1"
SERIES_PER_CAT = 70        # top sellers per category (about 210 series). Use 4 for a quick test
MAX_ZERO_SHARE = 0.30      # drop series with more than 30% zero-sales weeks (training weeks)
TRAIN_END, CAL_END = 193, 235   # 277 weeks -> about 70% / 15% / 15%
EXPECTED_DAYS, EXPECTED_WEEKS = 1941, 277

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, RES = ROOT / "data", ROOT / "data" / "processed", ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
RES.mkdir(exist_ok=True)

checks, log = [], []

def check(stage, name, ok, detail="", soft=False):
    ok = bool(ok)
    status = "PASS" if ok else ("WARN" if soft else "FAIL")
    checks.append((stage, name, status, str(detail)))
    print(f"[{status}] {stage} | {name}" + (f"  ->  {detail}" if detail != "" else ""))

def save_report():
    pd.DataFrame(checks, columns=["stage", "check", "status", "detail"]).to_csv(
        RES / "preprocessing_checks.csv", index=False)
    pd.DataFrame(log, columns=["step", "value"]).to_csv(RES / "data_quality_log.csv", index=False)

def gate(stage):
    """Stop the script if any check of this stage failed."""
    if any(c[0] == stage and c[2] == "FAIL" for c in checks):
        save_report()
        sys.exit(f"\nSTOPPED: fix the FAIL checks in '{stage}' first (see results/preprocessing_checks.csv).")

# =====================================================================
# PART A - CHECK THE RAW DATA (before any change)
# =====================================================================
A = "A raw data"
for f in ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"]:
    check(A, f"{f} exists", (RAW / f).exists())
gate(A)

calendar = pd.read_csv(RAW / "calendar.csv")
sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
prices = pd.read_csv(RAW / "sell_prices.csv")
print("Raw shapes  calendar / sales / prices:", calendar.shape, sales.shape, prices.shape)

# column names: lowercase, no spaces; text columns: strip spaces (structural errors)
for df in (calendar, sales, prices):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()

day_cols = [c for c in sales.columns if c.startswith("d_")]
check(A, "required columns exist",
      {"item_id", "dept_id", "cat_id", "store_id", "state_id"} <= set(sales.columns)
      and {"d", "date", "wm_yr_wk", "event_name_1", "event_type_1"} <= set(calendar.columns)
      and {"store_id", "item_id", "wm_yr_wk", "sell_price"} <= set(prices.columns))
check(A, f"sales has {EXPECTED_DAYS} day columns", len(day_cols) == EXPECTED_DAYS, len(day_cols), soft=True)

# missing values
check(A, "no missing values in raw sales", sales.isna().sum().sum() == 0, sales.isna().sum().sum())
check(A, "no missing values in raw prices", prices.isna().sum().sum() == 0, prices.isna().sum().sum())
cal_na = calendar.isna().sum()
event_cols = [c for c in calendar.columns if c.startswith("event_")]
other_na = cal_na.drop(event_cols).sum()
check(A, "calendar: missing values only in event columns", other_na == 0, f"{other_na} elsewhere")
print("        (missing event values mean 'no event':", cal_na[event_cols].to_dict(), ")")

# duplicates
check(A, "no duplicate sales rows", (~sales.duplicated(["item_id", "store_id"])).all())
check(A, "no duplicate days in calendar", calendar["d"].is_unique)
check(A, "no duplicate price rows", (~prices.duplicated(["store_id", "item_id", "wm_yr_wk"])).all())

# valid ranges
check(A, "no negative sales", sales[day_cols].min().min() >= 0, sales[day_cols].min().min())
check(A, "all prices > 0", (prices.sell_price > 0).all(), prices.sell_price.min())

# calendar coverage and continuity
calendar["date"] = pd.to_datetime(calendar["date"])
check(A, "calendar covers every sales day", set(day_cols) <= set(calendar.d))
check(A, "calendar dates are continuous (1 day apart)",
      (calendar.sort_values("date").date.diff().dropna() == pd.Timedelta(days=1)).all())
check(A, "price weeks exist in the calendar", set(prices.wm_yr_wk) <= set(calendar.wm_yr_wk))

# categories and stores (structural errors)
check(A, "chosen store is in the sales file", STORE in set(sales.store_id), STORE)
check(A, "3 categories (FOODS, HOBBIES, HOUSEHOLD)", set(sales.cat_id) == {"FOODS", "HOBBIES", "HOUSEHOLD"},
      sorted(sales.cat_id.unique()), soft=True)
check(A, "10 stores", sales.store_id.nunique() == 10, sales.store_id.nunique(), soft=True)
print(f"        raw zero-sales share (all series, all days): {(sales[day_cols].values == 0).mean():.1%}")
gate(A)

# =====================================================================
# PART B - PREPROCESSING (with quick sanity checks)
# =====================================================================
# ---- B1. Subset: one store, top sellers per category ----
B = "B1 subset"
sub = sales[sales.store_id == STORE].copy()
del sales                                            # free memory
sub["total"] = sub[day_cols].sum(axis=1)
sub = sub.sort_values("total", ascending=False).groupby("cat_id").head(SERIES_PER_CAT).copy()
log.append(("series after subset", len(sub)))
check(B, "only the chosen store, unique items", set(sub.store_id) == {STORE} and sub.item_id.is_unique,
      sub.groupby("cat_id").size().to_dict())
gate(B)

# ---- B2. Melt: wide (one column per day) -> long ----
B = "B2 melt"
id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
long = sub.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
check(B, "rows = series x days", len(long) == len(sub) * len(day_cols), len(long))
check(B, "total sales unchanged by melt", long.sales.sum() == sub[day_cols].sum().sum())
del sub
gate(B)

# ---- B3. Merge calendar (events: NaN -> 'none') ----
B = "B3 calendar"
calendar[["event_name_1", "event_type_1"]] = calendar[["event_name_1", "event_type_1"]].fillna("none")
snap_col = "snap_" + STORE[:2].lower()               # e.g. snap_ca
n0 = len(long)
long = long.merge(calendar[["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", snap_col]],
                  on="d", how="left")
check(B, "row count unchanged and every day matched a date", len(long) == n0 and long.date.notna().all())
check(B, "no missing event values left", long.event_name_1.notna().all() and long.event_type_1.notna().all())
gate(B)

# ---- B4. Aggregate to weekly ----
B = "B4 weekly"
def first_event(s):
    s = s[s != "none"]
    return s.iloc[0] if len(s) else "none"

weekly = (long.groupby(id_cols + ["wm_yr_wk"], as_index=False)
          .agg(sales=("sales", "sum"), week_start=("date", "min"), n_days=("d", "count"),
               event_name=("event_name_1", first_event), event_type=("event_type_1", first_event),
               snap_days=(snap_col, "sum")))
check(B, "total sales unchanged by aggregation", weekly.sales.sum() == long.sales.sum())
n0 = len(weekly)
weekly = weekly[weekly.n_days == 7].drop(columns="n_days")      # drop the partial last week
log.append(("incomplete-week rows dropped", n0 - len(weekly)))
wk_per_item = weekly.groupby("item_id").size()
check(B, "every item has the same number of full weeks", wk_per_item.nunique() == 1, wk_per_item.iloc[0])
check(B, f"{EXPECTED_WEEKS} full weeks per item", wk_per_item.iloc[0] == EXPECTED_WEEKS, soft=True)
del long
gate(B)

# ---- B5. Merge prices; a missing price = item not on sale yet ----
B = "B5 prices"
n0 = len(weekly)
weekly = weekly.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
check(B, "row count unchanged by the price merge", len(weekly) == n0)
weekly["series_id"] = weekly.item_id + "_" + weekly.store_id
weekly = weekly.sort_values(["series_id", "week_start"]).reset_index(drop=True)

log.append(("weeks with missing price (before cleaning)", int(weekly.sell_price.isna().sum())))
n0 = len(weekly)
launch = weekly[weekly.sell_price.notna()].groupby("series_id").week_start.min().rename("launch")
weekly = weekly.merge(launch, on="series_id")
weekly = weekly[weekly.week_start >= weekly.launch].drop(columns="launch")      # drop pre-launch weeks
log.append(("pre-launch rows dropped", n0 - len(weekly)))
gaps = int(weekly.sell_price.isna().sum())
weekly["sell_price"] = weekly.groupby("series_id").sell_price.ffill()           # gaps after launch
log.append(("price gaps forward-filled", gaps))
check(B, "no missing prices left", weekly.sell_price.notna().all(), f"{gaps} gap(s) filled")
gate(B)

# ---- B6. Duplicates ----
n0 = len(weekly)
weekly = weekly.drop_duplicates(subset=["series_id", "week_start"])
log.append(("duplicate rows dropped", n0 - len(weekly)))

# ---- B7. Keep only full-history series ----
B = "B7 full history"
counts = weekly.groupby("series_id").size()
full = counts[counts == counts.max()].index
log.append(("series dropped (incomplete history)", len(counts) - len(full)))
weekly = weekly[weekly.series_id.isin(full)].copy()
check(B, "at least 100 series left (needed for TFT)", weekly.series_id.nunique() >= 100,
      weekly.series_id.nunique(), soft=True)

# ---- B8. Time index and time-ordered split (no shuffling) ----
B = "B8 split"
weekly["time_idx"] = ((weekly.week_start - weekly.week_start.min()).dt.days // 7).astype(int)
weekly["split"] = np.select([weekly.time_idx <= TRAIN_END, weekly.time_idx <= CAL_END],
                            ["train", "calib"], default="test")
share = weekly.split.value_counts(normalize=True)
check(B, "split is about 70/15/15 (+-2%)",
      abs(share["train"] - .70) < .02 and abs(share["calib"] - .15) < .02 and abs(share["test"] - .15) < .02,
      share.round(3).to_dict())
gate(B)

# ---- B9. Drop mostly-zero series (judged on training weeks only) ----
B = "B9 zero filter"
zs = weekly[weekly.split == "train"].groupby("series_id").sales.apply(lambda s: (s == 0).mean())
keep = zs[zs < MAX_ZERO_SHARE].index
log.append(("series dropped (too many zero weeks)", len(zs) - len(keep)))
weekly = weekly[weekly.series_id.isin(keep)].copy()
check(B, "enough series left (>= 100)", weekly.series_id.nunique() >= 100, weekly.series_id.nunique(), soft=True)
check(B, "all 3 categories left (needed for decomposition)", weekly.cat_id.nunique() == 3,
      weekly.groupby("cat_id").series_id.nunique().to_dict(), soft=True)

# ---- B10. Outlier report (information only: event spikes are real demand, NOT removed) ----
g = weekly.groupby("series_id").sales
q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
extreme = weekly.sales > (q3 + 3 * (q3 - q1))
on_event = (weekly.event_name != "none")
log.append(("extreme-sales weeks, share (kept)", round(extreme.mean(), 4)))
print(f"        extreme weeks (> Q3 + 3*IQR): {extreme.mean():.2%} overall | "
      f"{extreme[on_event].mean():.2%} on event weeks | {extreme[~on_event].mean():.2%} on other weeks")

# ---- B11. Features (reference price uses training weeks only = no leakage) ----
B = "B11 features"
weekly = weekly.sort_values(["series_id", "week_start"]).reset_index(drop=True)
ref = weekly[weekly.split == "train"].groupby("series_id").sell_price.median().rename("ref_price")
weekly = weekly.merge(ref, on="series_id")
weekly["price_ratio"] = weekly.sell_price / weekly.ref_price
weekly["promo_flag"] = (weekly.price_ratio < 0.95).astype(int)
weekly["price_change"] = (weekly.groupby("series_id").sell_price.pct_change()
                          .replace([np.inf, -np.inf], 0).fillna(0))
weekly["week_of_year"] = weekly.week_start.dt.isocalendar().week.astype(int)
weekly["month"] = weekly.week_start.dt.month.astype(str)
weekly["has_event"] = (weekly.event_name != "none").astype(int)
season_map = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
weekly["season"] = weekly.week_start.dt.month.map(season_map)
check(B, "no infinite or missing values in the new features",
      np.isfinite(weekly[["price_ratio", "price_change"]]).all().all() and weekly[["season", "month"]].notna().all().all())
check(B, "promo weeks are a sensible share (1%-60%)", 0.01 < weekly.promo_flag.mean() < 0.60,
      f"{weekly.promo_flag.mean():.1%}", soft=True)
check(B, "event weeks are a sensible share (1%-50%)", 0.01 < weekly.has_event.mean() < 0.50,
      f"{weekly.has_event.mean():.1%}", soft=True)
gate(B)

# ---- B12. Keep only the needed columns and save ----
final_cols = ["series_id", "item_id", "dept_id", "cat_id", "store_id",
              "week_start", "time_idx", "split", "sales",
              "sell_price", "price_ratio", "promo_flag", "price_change",
              "event_name", "event_type", "has_event", "snap_days",
              "week_of_year", "month", "season"]
weekly = weekly[final_cols].reset_index(drop=True)

log.append(("final series", weekly.series_id.nunique()))
log.append(("weeks per series", weekly.groupby("series_id").size().iloc[0]))
log.append(("final rows", len(weekly)))
log.append(("zero-sales share overall", round((weekly.sales == 0).mean(), 3)))

weekly.to_parquet(OUT / "m5_weekly_ca1.parquet", index=False)   # main file for modelling
weekly.to_csv(OUT / "m5_weekly_ca1.csv", index=False)           # easy to open in Excel
save_report()

# ---------------- SUMMARY ----------------
rep = pd.DataFrame(checks, columns=["stage", "check", "status", "detail"])
print("\n=== DATA QUALITY LOG ===")
print(pd.DataFrame(log, columns=["step", "value"]).to_string(index=False))
print("\n=== SPLIT (rows) ===");          print(weekly.split.value_counts())
print("\n=== SERIES PER CATEGORY ===");   print(weekly.groupby("cat_id").series_id.nunique())
print(f"\n===== CHECKS: {(rep.status == 'PASS').sum()} passed, {(rep.status == 'WARN').sum()} warnings, "
      f"{(rep.status == 'FAIL').sum()} failed =====")
print("Saved:", OUT / "m5_weekly_ca1.parquet")
print("Next: python src/validate_preprocessing.py")