import pandas as pd
import numpy as np

def create_targets(df: pd.DataFrame, target_col: str = 'Modal_Price', horizons: list = [1, 7]) -> pd.DataFrame:
    """Creates future targets. Assumes dataframe is chronologically sorted per series."""
    df_out = df.copy()
    for h in horizons:
        df_out[f'target_{h}'] = df_out.groupby('Series_ID')[target_col].shift(-h)
        df_out[f'target_date_{h}'] = df_out.groupby('Series_ID')['Price Date'].shift(-h)
    return df_out

def create_time_features(df: pd.DataFrame, date_col: str = 'Price Date') -> pd.DataFrame:
    """Extracts calendar and seasonality features."""
    df_feat = df.copy()
    df_feat[date_col] = pd.to_datetime(df_feat[date_col])
    df_feat['day_of_week'] = df_feat[date_col].dt.dayofweek
    df_feat['month'] = df_feat[date_col].dt.month
    df_feat['week_of_year'] = df_feat[date_col].dt.isocalendar().week.astype(int)
    
    # Cyclical encoding
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month']/12.0)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month']/12.0)
    df_feat['day_sin'] = np.sin(2 * np.pi * df_feat['day_of_week']/7.0)
    df_feat['day_cos'] = np.cos(2 * np.pi * df_feat['day_of_week']/7.0)
    return df_feat

def create_lag_features(df: pd.DataFrame, target_col: str = 'Modal_Price', lags: list = [1, 3, 7, 14, 30]) -> pd.DataFrame:
    """Creates lag features."""
    df_feat = df.copy()
    for lag in lags:
        df_feat[f'{target_col}_lag_{lag}'] = df_feat.groupby('Series_ID')[target_col].shift(lag)
    return df_feat

def create_rolling_features(df: pd.DataFrame, target_col: str = 'Modal_Price', windows: list = [7, 14, 30]) -> pd.DataFrame:
    """Creates rolling statistics. Shifted by 1 to prevent leakage."""
    df_feat = df.copy()
    for w in windows:
        df_feat[f'{target_col}_rolling_mean_{w}'] = df_feat.groupby('Series_ID')[target_col].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        df_feat[f'{target_col}_rolling_std_{w}'] = df_feat.groupby('Series_ID')[target_col].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).std())
        df_feat[f'{target_col}_rolling_min_{w}'] = df_feat.groupby('Series_ID')[target_col].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).min())
        df_feat[f'{target_col}_rolling_max_{w}'] = df_feat.groupby('Series_ID')[target_col].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).max())
    return df_feat

def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates derived price features. Min_Price and Max_Price of day t can be used to predict t+h (h >= 1).
    Valid only if Min/Max are valid.
    """
    df_feat = df.copy()
    # Mask invalid min/max based on Phase 1 rules (feature invalidity flag should have been generated, but we can do a quick sanity check)
    valid_min_max = (df_feat['Min_Price'] > 0) & (df_feat['Max_Price'] > 0) & (df_feat['Min_Price'] <= df_feat['Max_Price'])
    
    df_feat['price_spread'] = np.where(valid_min_max, df_feat['Max_Price'] - df_feat['Min_Price'], np.nan)
    df_feat['spread_ratio'] = np.where(valid_min_max, df_feat['price_spread'] / (df_feat['Modal_Price'] + 1e-8), np.nan)
    return df_feat

def apply_feature_engineering(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Runs target generation and feature engineering."""
    df = df.sort_values(by=['Series_ID', 'Price Date']).reset_index(drop=True)
    
    df = create_targets(df, horizons=config['horizons'])
    df = create_time_features(df)
    df = create_lag_features(df, lags=config['lag_features'])
    df = create_rolling_features(df, windows=config['rolling_windows'])
    df = create_price_features(df)
    
    return df
