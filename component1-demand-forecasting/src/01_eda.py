"""
01_eda.py - exploratory data analysis (Component 1)
EDA asks “what patterns are in the sales?”
Its purpose is to find out what drives demand, so we know which factors to give the model.
Run from the component1-demand-forecasting folder:  python src/01_eda.py
Needs: pip install matplotlib seaborn
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_END, CAL_END = 193, 235              # same as 01_preprocessing.py
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

w = pd.read_parquet(ROOT / "data" / "processed" / "m5_weekly_ca1.parquet")#loads the cleaned weekly table

# scale each series by its TRAINING mean so series of different size are comparable
train_mean = w[w.split == "train"].groupby("series_id").sales.mean()
w["rel_sales"] = w.sales / w.series_id.map(train_mean)

# dates where the splits start (for shading)
d_cal = w.loc[w.time_idx == TRAIN_END + 1, "week_start"].iloc[0]
d_test = w.loc[w.time_idx == CAL_END + 1, "week_start"].iloc[0]
d_end = w.week_start.max()

def shade(ax):
    ax.axvspan(d_cal, d_test, color="orange", alpha=0.15, label="calibration")
    ax.axvspan(d_test, d_end, color="red", alpha=0.12, label="test")

# ---------- 1. total weekly sales over time ----------(Is demand growing, flat or wavy?)
total = w.groupby("week_start").sales.sum()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(total.index, total.values, lw=1.2)
shade(ax)
ax.set(title="Total weekly sales (all series) with data splits", xlabel="Week", ylabel="Units sold")
ax.legend()
fig.tight_layout(); fig.savefig(RES / "eda_01_total_sales.png", dpi=150); plt.close(fig)

# ---------- 2. event effect ----------(Do holidays and events change sales?)
ev = (w.groupby("event_type").rel_sales.agg(["mean", "count"]).sort_values("mean", ascending=False))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x=ev.index, y=ev["mean"], ax=axes[0], color="steelblue")
axes[0].axhline(1, color="grey", ls="--"); axes[0].set(title="Relative sales by event type", xlabel="", ylabel="Mean sales / series mean")
top = (w[w.event_name != "none"].groupby("event_name").rel_sales.mean().sort_values(ascending=False).head(8))
sns.barplot(x=top.values, y=top.index, ax=axes[1], color="darkorange")
axes[1].axvline(1, color="grey", ls="--"); axes[1].set(title="Top 8 events by relative sales", xlabel="Mean sales / series mean", ylabel="")
fig.tight_layout(); fig.savefig(RES / "eda_02_events.png", dpi=150); plt.close(fig)

# ---------- 3. promotion effect, by category ----------(Do promotions raise sales?_)
fig, ax = plt.subplots(figsize=(7, 4))
pe = w.groupby(["cat_id", "promo_flag"]).rel_sales.mean().reset_index()
pe["promo_flag"] = pe.promo_flag.map({0: "normal price", 1: "promo (price <95% of usual)"})
sns.barplot(data=pe, x="cat_id", y="rel_sales", hue="promo_flag", ax=ax)
ax.axhline(1, color="grey", ls="--"); ax.set(title="Promotion effect by category", xlabel="", ylabel="Mean sales / series mean")
ax.legend(title="")
fig.tight_layout(); fig.savefig(RES / "eda_03_promotion.png", dpi=150); plt.close(fig)

# ---------- 4. categories over time ----------(Which category sells most?)
cat = w.groupby(["week_start", "cat_id"]).sales.sum().reset_index()
fig, ax = plt.subplots(figsize=(11, 4))
sns.lineplot(data=cat, x="week_start", y="sales", hue="cat_id", ax=ax)
shade(ax)
ax.set(title="Weekly sales by category", xlabel="Week", ylabel="Units sold")
fig.tight_layout(); fig.savefig(RES / "eda_04_categories.png", dpi=150); plt.close(fig)

# ---------- 5. seasonality ----------(Are some weeks of the year busier?)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
woy = w[w.split == "train"].groupby("week_of_year").rel_sales.mean()
axes[0].plot(woy.index, woy.values); axes[0].axhline(1, color="grey", ls="--")
axes[0].set(title="Seasonality: mean relative sales by week of year (training)", xlabel="Week of year", ylabel="Mean sales / series mean")
sns.boxplot(data=w[w.split == "train"], x="season", y="rel_sales", order=["winter", "spring", "summer", "autumn"],
            showfliers=False, ax=axes[1], color="lightsteelblue")
axes[1].set(title="Relative sales by season (training)", xlabel="", ylabel="")
fig.tight_layout(); fig.savefig(RES / "eda_05_seasonality.png", dpi=150); plt.close(fig)

# ---------- 6. series level: zero share and three example series ----------(How “lumpy” is the data?)
zs = w[w.split == "train"].groupby("series_id").sales.apply(lambda s: (s == 0).mean())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(zs, bins=20, color="slategrey")
axes[0].set(title="Zero-sales weeks per series (training)", xlabel="Share of zero weeks", ylabel="Number of series")
best = w.groupby("series_id").sales.sum().sort_values(ascending=False)
examples = [best.index[0], best.index[len(best) // 2], best.index[-1]]
for sid in examples:
    s = w[w.series_id == sid]
    axes[1].plot(s.week_start, s.sales, lw=1, label=sid)
axes[1].axvline(d_cal, color="orange", ls="--"); axes[1].axvline(d_test, color="red", ls="--")
axes[1].set(title="Three example series (high, medium, low volume)", xlabel="Week", ylabel="Units sold")
axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(RES / "eda_06_series.png", dpi=150); plt.close(fig)

# ---------- 7. SNAP effect ----------(Does the benefit-payment timing matter?)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

snap_effect = w.groupby("snap_days").rel_sales.mean()
axes[0].bar(snap_effect.index, snap_effect.values, color="steelblue")
axes[0].axhline(1.0, color="grey", ls="--")
axes[0].set(title="SNAP effect on relative sales",
            xlabel="SNAP days in week",
            ylabel="Mean sales / series mean")

snap_cat = w.groupby(["cat_id", "snap_days"]).rel_sales.mean().reset_index()
for cat_id in snap_cat.cat_id.unique():
    sub = snap_cat[snap_cat.cat_id == cat_id]
    axes[1].plot(sub.snap_days, sub.rel_sales, marker="o", label=cat_id)
axes[1].axhline(1.0, color="grey", ls="--")
axes[1].set(title="SNAP effect by category",
            xlabel="SNAP days in week",
            ylabel="Mean sales / series mean")
axes[1].legend()

fig.tight_layout(); fig.savefig(RES / "eda_07_snap.png", dpi=150); plt.close(fig)

# ---------- 8. price effect ---------(Do discounts or price rises matter?)
labels = ["<0.85", "0.85-0.95", "0.95-1.05", "1.05-1.15", ">1.15"]
w["price_bin"] = pd.cut(w.price_ratio, bins=[0, 0.85, 0.95, 1.05, 1.15, 10], labels=labels)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
price_effect = w.groupby("price_bin", observed=False).rel_sales.mean()
share = w.price_bin.value_counts(normalize=True).reindex(labels)
axes[0].bar(labels, price_effect.values, color="darkorange")
axes[0].axhline(1.0, color="grey", ls="--")
for i, s in enumerate(share.values):
    axes[0].text(i, 0.02, f"{s:.0%} of weeks", ha="center", fontsize=8)
axes[0].set(title="Price effect on relative sales", xlabel="Price ratio to usual price",
            ylabel="Mean sales / series mean")

pivot = w.pivot_table(index="price_bin", columns="cat_id", values="rel_sales",
                      aggfunc="mean", observed=False).reindex(labels)
pivot.plot(marker="o", ax=axes[1])
axes[1].axhline(1.0, color="grey", ls="--")
axes[1].set(title="Price effect by category", xlabel="Price ratio to usual price",
            ylabel="Mean sales / series mean")
fig.tight_layout(); fig.savefig(RES / "eda_08_price.png", dpi=150); plt.close(fig)
print("\nShare of weeks per price bin:\n", share.round(3))

# ---------- summary numbers for the slides ----------
rows = [
    ("series", w.series_id.nunique()),
    ("weeks per series", w.groupby("series_id").size().iloc[0]),
    ("date range", f"{w.week_start.min().date()} to {w.week_start.max().date()}"),
    ("mean weekly sales per series", round(w.sales.mean(), 2)),
    ("median weekly sales per series", w.sales.median()),
    ("coefficient of variation (median over series)",
     round((w.groupby("series_id").sales.std() / w.groupby("series_id").sales.mean()).median(), 2)),
    ("event weeks relative sales", round(w[w.has_event == 1].rel_sales.mean(), 3)),
    ("non-event weeks relative sales", round(w[w.has_event == 0].rel_sales.mean(), 3)),
    ("promo weeks relative sales", round(w[w.promo_flag == 1].rel_sales.mean(), 3)),
    ("non-promo weeks relative sales", round(w[w.promo_flag == 0].rel_sales.mean(), 3)),
    ("share of promo weeks", round(w.promo_flag.mean(), 3)),
    ("share of event weeks", round(w.has_event.mean(), 3)),
    ("overall zero-sales share", round((w.sales == 0).mean(), 3)),
    ("SNAP days mean", round(w.snap_days.mean(), 2)),
    ("SNAP days = 0 relative sales", round(w[w.snap_days == 0].rel_sales.mean(), 3)),
    ("SNAP days > 0 relative sales", round(w[w.snap_days > 0].rel_sales.mean(), 3)),
    ("SNAP effect (difference)", round(
        w[w.snap_days > 0].rel_sales.mean() - w[w.snap_days == 0].rel_sales.mean(), 3
    )),
    ("promo weeks relative sales", round(w[w.promo_flag == 1].rel_sales.mean(), 3)),
    ("non-promo weeks relative sales", round(w[w.promo_flag == 0].rel_sales.mean(), 3)),
    ("price ratio mean", round(w.price_ratio.mean(), 3)),
    ("price ratio std", round(w.price_ratio.std(), 3)),
]
for cat_id, g in w.groupby("cat_id"):
    rows.append((f"series in {cat_id}", g.series_id.nunique()))
summary = pd.DataFrame(rows, columns=["metric", "value"])
summary.to_csv(RES / "eda_summary.csv", index=False)
print(summary.to_string(index=False))
print("\nRelative sales by event type:\n", ev.round(3))
print("\nPlots saved in", RES)