import pandas as pd
import numpy as np

def validate_schema(df: pd.DataFrame) -> bool:
    """Validates if the dataset has the required columns."""
    required_columns = [
        'STATE', 'District Name', 'Market Name', 'Commodity', 
        'Variety', 'Grade', 'Min_Price', 'Max_Price', 'Modal_Price', 'Price Date'
    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

def clean_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans invalid records based on Phase 1 corrected rules:
    - Removes records where Modal_Price <= 0 (Target Invalidity).
    - Flags Min_Price or Max_Price <= 0 (Feature Invalidity).
    - Flags records where Min_Price > Max_Price or Modal_Price outside [Min_Price, Max_Price] (Range Violations).
    """
    df_clean = df.copy()
    
    # 1. Target Invalidity: Remove rows where Modal_Price is <= 0 or missing
    valid_target = (df_clean['Modal_Price'] > 0) & (df_clean['Modal_Price'].notnull())
    df_clean = df_clean[valid_target]
    
    # 2. Feature Invalidity: Flag Min/Max <= 0
    df_clean['feature_invalid'] = (df_clean['Min_Price'] <= 0) | (df_clean['Max_Price'] <= 0)
    
    # 3. Range Violations: Flag logical range errors
    df_clean['range_violation'] = (
        (df_clean['Min_Price'] > df_clean['Max_Price']) | 
        (df_clean['Modal_Price'] < df_clean['Min_Price']) | 
        (df_clean['Modal_Price'] > df_clean['Max_Price'])
    )
    
    return df_clean

def check_data_leakage_features(features: list) -> bool:
    """Ensures no future features are present in the list."""
    # Future leakage features check
    leakage_keywords = ['future', 'next', 'target_lead']
    for feat in features:
        if any(leak in feat.lower() for leak in leakage_keywords):
            raise ValueError(f"Potential data leakage detected in feature: {feat}")
    return True
