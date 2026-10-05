import pandas as pd
import numpy as np

file_path = r"C:\Users\KRITIKA\Downloads\Agriculture_price_dataset.csv"
df = pd.read_csv(file_path)
df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce')
df['Series_ID'] = df['STATE'] + "_" + df['District Name'] + "_" + df['Market Name'] + "_" + df['Commodity'] + "_" + df['Variety'] + "_" + df['Grade']

print("=== ISSUE 4: RANGE VIOLATIONS & INVALIDITY ===")
target_invalid = df[df['Modal_Price'] <= 0]
print(f"Target Invalid (Modal_Price <= 0): {len(target_invalid)} rows")

feature_invalid = df[(df['Min_Price'] <= 0) | (df['Max_Price'] <= 0)]
print(f"Feature Invalid (Min/Max <= 0): {len(feature_invalid)} rows")

range_violations = df[(df['Min_Price'] > df['Max_Price']) | (df['Modal_Price'] < df['Min_Price']) | (df['Modal_Price'] > df['Max_Price'])]
print(f"Range Violations: {len(range_violations)} rows ({(len(range_violations)/len(df))*100:.2f}%)")
print(range_violations[['Min_Price', 'Max_Price', 'Modal_Price']].head())

print("\n=== ISSUE 5: DUPLICATE AGGREGATION ===")
duplicate_groups = df.groupby(['Series_ID', 'Price Date']).size()
duplicates = duplicate_groups[duplicate_groups > 1]
print(f"Number of duplicate series-date groups: {len(duplicates)}")
if len(duplicates) > 0:
    print(f"Number of rows belonging to those groups: {duplicates.sum()}")
    print("Distribution of duplicate group sizes:")
    print(duplicates.value_counts())
    
    # Examples
    example_group = duplicates.index[0]
    print(f"Example group {example_group}:")
    print(df[(df['Series_ID'] == example_group[0]) & (df['Price Date'] == example_group[1])][['Min_Price', 'Max_Price', 'Modal_Price']])

print("\n=== ISSUE 9 & 10: SERIES COVERAGE ===")
series_to_check = [
    "Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ",
    "West Bengal_paschim bardhaman_Asansol_Potato_Jyoti_FAQ",
    "Uttar Pradesh_sambhal_Sambhal_Potato_Desi_FAQ"
]

for s_id in series_to_check:
    s_data = df[df['Series_ID'] == s_id].sort_values('Price Date')
    n_obs = len(s_data)
    first_date = s_data['Price Date'].min()
    last_date = s_data['Price Date'].max()
    expected_days = (last_date - first_date).days + 1
    coverage = (n_obs / expected_days) * 100
    
    gaps = s_data['Price Date'].diff().dt.days
    median_gap = gaps.median()
    max_gap = gaps.max()
    gaps_gt_1 = (gaps > 1).sum()
    missing_calendar_days = expected_days - n_obs
    
    print(f"\nSeries: {s_id}")
    print(f"Observations: {n_obs}")
    print(f"Date range: {first_date.date()} to {last_date.date()}")
    print(f"Expected calendar days: {expected_days}")
    print(f"Observed days: {n_obs}")
    print(f"Coverage percentage: {coverage:.2f}%")
    print(f"Median gap: {median_gap} days")
    print(f"Maximum gap: {max_gap} days")
    print(f"Number of missing calendar days: {missing_calendar_days}")
    print(f"Number of gaps > 1 day: {gaps_gt_1}")
