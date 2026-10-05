import pandas as pd
import numpy as np
import os
from datetime import datetime

# Path to the dataset
file_path = r"C:\Users\KRITIKA\Downloads\Agriculture_price_dataset.csv"

print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

print("="*50)
print("TASK 1 — DATASET AUDIT")
print("="*50)
print(f"1. Number of rows: {len(df)}")
print(f"2. Number of columns: {len(df.columns)}")
print(f"3. Column names: {list(df.columns)}")
print(f"4. Data types:\n{df.dtypes}")
print(f"\n5. Missing values per column:\n{df.isnull().sum()}")
print(f"\n6. Missing-value percentage:\n{df.isnull().mean() * 100}")
print(f"\n7. Duplicate rows: {df.duplicated().sum()}")
print(f"8. Duplicate percentage: {(df.duplicated().sum() / len(df)) * 100:.2f}%")

print(f"\n9. Invalid/negative/zero price values:")
for col in ['Min_Price', 'Max_Price', 'Modal_Price']:
    if col in df.columns:
        invalid = df[(df[col] <= 0) | (df[col].isnull())]
        print(f"   {col} <= 0 or null: {len(invalid)}")

print(f"\n10. Unique states: {df['STATE'].nunique()}")
print(f"11. Unique districts: {df['District Name'].nunique() if 'District Name' in df.columns else df.get('District', pd.Series()).nunique()}")
print(f"12. Unique markets: {df['Market Name'].nunique() if 'Market Name' in df.columns else df.get('Market', pd.Series()).nunique()}")
print(f"13. Unique commodities: {df['Commodity'].nunique()}")
print(f"14. Unique varieties: {df['Variety'].nunique()}")
print(f"15. Unique grades: {df['Grade'].nunique()}")

if 'Price Date' in df.columns:
    df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce')
    print(f"16. Minimum Price Date: {df['Price Date'].min()}")
    print(f"    Maximum Price Date: {df['Price Date'].max()}")

print("\n" + "="*50)
print("TASK 2 — DATA TYPE AND DATE VALIDATION")
print("="*50)
invalid_dates = df['Price Date'].isnull().sum()
print(f"Invalid dates: {invalid_dates}")
print(f"Date range: {df['Price Date'].min()} to {df['Price Date'].max()}")
print(f"Number of unique dates: {df['Price Date'].nunique()}")
is_sorted = df['Price Date'].is_monotonic_increasing
print(f"Are dates sorted?: {is_sorted}")

print("\n" + "="*50)
print("TASK 3 — PRICE VALIDATION")
print("="*50)
print("Zero/Negative values:")
for col in ['Min_Price', 'Max_Price', 'Modal_Price']:
    if col in df.columns:
        print(f"  {col} == 0: {(df[col] == 0).sum()}")
        print(f"  {col} < 0: {(df[col] < 0).sum()}")

print("Min_Price > Max_Price cases:", (df['Min_Price'] > df['Max_Price']).sum())
print("Modal_Price outside Min/Max range:", ((df['Modal_Price'] < df['Min_Price']) | (df['Modal_Price'] > df['Max_Price'])).sum())

print("\n" + "="*50)
print("TASK 4 — DUPLICATE ANALYSIS")
print("="*50)
print(f"Completely duplicated rows: {df.duplicated().sum()}")
subset = ['STATE', 'District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'Price Date']
subset = [c for c in subset if c in df.columns]
print(f"Duplicate observations based on subset {subset}: {df.duplicated(subset=subset).sum()}")

print("\n" + "="*50)
print("TASK 5 — MARKET/COMMODITY TIME-SERIES ANALYSIS")
print("="*50)
# Create Series ID
df['Series_ID'] = df['STATE'] + "_" + df['District Name'] + "_" + df['Market Name'] + "_" + df['Commodity'] + "_" + df['Variety'] + "_" + df['Grade']
series_counts = df['Series_ID'].value_counts()
print("Series length distribution:")
print(f"  <30: {(series_counts < 30).sum()}")
print(f"  30-59: {((series_counts >= 30) & (series_counts <= 59)).sum()}")
print(f"  60-89: {((series_counts >= 60) & (series_counts <= 89)).sum()}")
print(f"  90-179: {((series_counts >= 90) & (series_counts <= 179)).sum()}")
print(f"  180-364: {((series_counts >= 180) & (series_counts <= 364)).sum()}")
print(f"  365-729: {((series_counts >= 365) & (series_counts <= 729)).sum()}")
print(f"  730+: {(series_counts >= 730).sum()}")

print("\n" + "="*50)
print("TASK 6 - Top 10 longest series (for consideration)")
print("="*50)
top_series = series_counts.head(10).index
for s_id in top_series:
    s_data = df[df['Series_ID'] == s_id].sort_values('Price Date')
    print(f"\nSeries ID: {s_id}")
    print(f"  Observations: {len(s_data)}")
    print(f"  Date range: {s_data['Price Date'].min()} to {s_data['Price Date'].max()}")
    print(f"  Unique dates: {s_data['Price Date'].nunique()}")
    
    # Gap analysis
    gaps = s_data['Price Date'].diff().dt.days
    print(f"  Typical gap: {gaps.median()} days")
    print(f"  Max gap: {gaps.max()} days")
    print(f"  Modal_Price stats: Mean={s_data['Modal_Price'].mean():.2f}, Min={s_data['Modal_Price'].min()}, Max={s_data['Modal_Price'].max()}, Std={s_data['Modal_Price'].std():.2f}")

