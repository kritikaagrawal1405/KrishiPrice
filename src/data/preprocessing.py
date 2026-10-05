import pandas as pd
import numpy as np
import os
from .validation import validate_schema, clean_invalid_records

def load_and_parse_data(filepath: str) -> pd.DataFrame:
    """Loads raw CSV and parses dates."""
    df = pd.read_csv(filepath)
    validate_schema(df)
    
    # Parse dates
    df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce')
    df = df.dropna(subset=['Price Date'])
    return df

def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for duplicate records for the same Market/Commodity/Date.
    Based on Phase 1 analysis, exact Series-Date duplicates are 0.
    """
    if 'Series_ID' not in df.columns:
        # Create Series_ID if not already present
        df['Series_ID'] = df['STATE'] + "_" + df['District Name'] + "_" + df['Market Name'] + "_" + df['Commodity'] + "_" + df['Variety'] + "_" + df['Grade']
    
    # Verify uniqueness
    duplicate_count = df.duplicated(subset=['Series_ID', 'Price Date']).sum()
    if duplicate_count > 0:
        print(f"Warning: Found {duplicate_count} duplicate Series_ID and Price Date pairs. Keeping the first occurrence.")
        df = df.drop_duplicates(subset=['Series_ID', 'Price Date'], keep='first')
        
    return df

def extract_selected_series(df: pd.DataFrame, series_ids: list) -> pd.DataFrame:
    """Filters dataset to only include selected series IDs and sorts chronologically."""
    df_filtered = df[df['Series_ID'].isin(series_ids)].copy()
    df_filtered = df_filtered.sort_values(by=['Series_ID', 'Price Date']).reset_index(drop=True)
    return df_filtered

def process_pipeline(input_path: str, output_path: str, selected_series: list):
    """Full preprocessing pipeline execution."""
    print("Loading data...")
    df = load_and_parse_data(input_path)
    
    print("Cleaning invalid records...")
    df = clean_invalid_records(df)
    
    print("Handling duplicates...")
    df = handle_duplicates(df)
    
    # Save the cleaned full dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_full_path = os.path.join(os.path.dirname(output_path), 'cleaned_mandi_prices.csv')
    df.to_csv(cleaned_full_path, index=False)
    print(f"Cleaned full dataset saved to {cleaned_full_path}")
    
    print("Extracting selected series...")
    df_selected = extract_selected_series(df, selected_series)
    
    df_selected.to_csv(output_path, index=False)
    print(f"Selected series dataset saved to {output_path}")
    return df_selected
