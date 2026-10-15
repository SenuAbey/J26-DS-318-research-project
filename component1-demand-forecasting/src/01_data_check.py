import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

# Load data
calendar = pd.read_csv(ROOT / "data" / "calendar.csv")
sales = pd.read_csv(ROOT / "data" / "sales_train_evaluation.csv")
prices = pd.read_csv(ROOT / "data" / "sell_prices.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nCalendar shape:", calendar.shape)
print("Sales shape:", sales.shape)
print("Prices shape:", prices.shape)

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

print("\nCalendar columns:", calendar.columns.tolist())
print("\nSales columns (first 10):", sales.columns.tolist()[:10])
print("\nSales columns (last 5):", sales.columns.tolist()[-5:])
print("\nPrices columns:", prices.columns.tolist())

print("\n" + "=" * 60)
print("DATA SUMMARY")
print("=" * 60)

print("\nDate range:", calendar['date'].min(), "to", calendar['date'].max())
print("Number of days:", len(calendar))
print("Stores:", sales['store_id'].unique())
print("States:", sales['state_id'].unique())
print("Categories:", sales['cat_id'].unique())
print("Departments:", sales['dept_id'].nunique())
print("Products:", sales['item_id'].nunique())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("\nCalendar missing values:\n", calendar.isnull().sum()[calendar.isnull().sum() > 0])
print("\nSales missing values:", sales.isnull().sum().sum())
print("Prices missing values:", prices.isnull().sum().sum())

print("\n" + "=" * 60)
print("SAMPLE DATA")
print("=" * 60)

print("\nCalendar sample:")
print(calendar.head(3))

print("\nSales sample (first 8 columns):")
print(sales.iloc[:3, :8])

print("\nPrices sample:")
print(prices.head(3))

print("\n" + "=" * 60)
print("EVENTS AND PROMOTIONS")
print("=" * 60)

print("\nEvent types:", calendar['event_type_1'].dropna().unique())
print("Number of event days:", calendar['event_name_1'].notna().sum())