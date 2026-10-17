"""
Component 3 - Step 0: data cleaning and preprocessing for the DataCo dataset

Run:  python c3_clean.py --csv DataCoSupplyChainDataset.csv
Out:  dataco_clean.csv (UTF-8) and cleaning_report.txt (every decision with row counts)

Every step below was chosen after profiling the raw file (see the reasons in comments).
"""
import argparse
import pandas as pd

# Personal data (masked in DataCo but still must not be kept) + useless columns
PII = ["Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
       "Customer Street", "Customer Zipcode"]
USELESS = ["Product Description",   # 100% empty
           "Product Status",        # constant
           "Order Zipcode",         # ~86% missing
           "Product Image"]         # image URL, no analytic value
REDUNDANT = ["Benefit per order"]   # identical to 'Order Profit Per Order'
NOT_REAL_DELIVERIES = ["CANCELED", "SUSPECTED_FRAUD"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="dataco_clean.csv")
    a = ap.parse_args()
    log = []

    def say(msg):
        print(msg); log.append(msg)

    # 1. Load (raw file is NOT UTF-8)
    df = pd.read_csv(a.csv, encoding="latin-1")
    say(f"1. Loaded raw file: {df.shape[0]} rows x {df.shape[1]} columns")
    say(f"   duplicate rows: {df.duplicated().sum()} | duplicate Order Item Id: {df['Order Item Id'].duplicated().sum()}")

    # 2. Drop personal / empty / constant / redundant columns
    drop = [c for c in PII + USELESS + REDUNDANT if c in df.columns]
    df = df.drop(columns=drop)
    say(f"2. Dropped {len(drop)} columns (personal data, empty/constant, redundant): {drop}")
    say(f"   remaining missing values: {int(df.isna().sum().sum())}")

    # 3. Parse dates and sanity-check them
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["shipping date (DateOrders)"], errors="coerce")
    bad = df["order_date"].isna() | df["ship_date"].isna() | (df["ship_date"] < df["order_date"])
    say(f"3. Dates parsed: {df['order_date'].min().date()} to {df['order_date'].max().date()} | invalid/inconsistent rows removed: {int(bad.sum())}")
    df = df[~bad]

    # 4. Remove orders that were never really delivered.
    #    CANCELED and SUSPECTED_FRAUD rows are all 'Shipping canceled' with Late_delivery_risk = 0
    #    by construction, which would wrongly lower late rates and wrongly contradict their delays.
    before = len(df)
    df = df[~df["Order Status"].isin(NOT_REAL_DELIVERIES)].copy()
    say(f"4. Removed cancelled / suspected-fraud orders: {before - len(df)} rows ({100 * (before - len(df)) / before:.1f}%) -> {len(df)} rows")

    # 5. Recompute delay + late flag and verify they agree
    df["delay"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
    df["late"] = (df["delay"] > 0).astype(int)
    agree = (df["late"] == df["Late_delivery_risk"]).mean()
    say(f"5. late flag vs (real days > scheduled days) agreement: {agree:.4f} | overall late rate: {df['late'].mean():.4f}")

    # 6. Country/region check. A country in several regions is NOT an error: the USA is split into
    #    West / East / Center / South regions. In a graph this simply gives one Country -> Region edge
    #    per region, so we keep the data as it is and only log it.
    cr = df.groupby("Order Country")["Order Region"].nunique()
    multi = cr[cr > 1].index.tolist()
    say(f"6. Countries linked to more than one region: {multi} - kept (valid, gives multiple Country->Region edges)")

    # 7. Clean text columns: trim ends AND collapse repeated inner spaces (e.g. 'South of  USA')
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    say("7. Stripped whitespace and collapsed repeated spaces in text columns")

    # 8. Winsorise extreme profit values (1st-99th percentile) so averages are not dominated by outliers
    for c in ["Order Item Profit Ratio", "Order Profit Per Order"]:
        lo, hi = df[c].quantile(0.01), df[c].quantile(0.99)
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        df[c] = df[c].clip(lo, hi)
        say(f"8. {c}: clipped {n_out} values to [{lo:.3f}, {hi:.3f}]")

    # 9. Summary of what is safe to use as a model input
    say("9. Leakage warning: 'Delivery Status', 'Late_delivery_risk', 'late', 'delay' describe the outcome. "
        "Use them only to build the label (future window) or past-window features, never as inputs for the same period.")
    say("   'Days for shipment (scheduled)' is fully determined by 'Shipping Mode' (Same Day 0, First 1, Second 2, Standard 4) - keep one.")

    monthly = df["order_date"].dt.to_period("M").value_counts().sort_index()
    say(f"10. Monthly volume: ~{int(monthly.iloc[:33].mean())} orders/month until Sep 2017, ~{int(monthly.iloc[33:].mean())}/month after - keep in mind for time splits")
    say(f"FINAL: {df.shape[0]} rows x {df.shape[1]} columns | orders: {df['Order Id'].nunique()} | "
        f"categories: {df['Category Name'].nunique()} | countries: {df['Order Country'].nunique()} | regions: {df['Order Region'].nunique()}")

    df.to_csv(a.out, index=False, encoding="utf-8")
    with open("cleaning_report.txt", "w") as f:
        f.write("\n".join(log))
    print(f"\nsaved {a.out} and cleaning_report.txt")


if __name__ == "__main__":
    main()
