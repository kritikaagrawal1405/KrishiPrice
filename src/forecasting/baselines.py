import pandas as pd
import numpy as np

class NaiveModel:
    """Predicts the most recently observed Modal_Price."""
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        # The most recent Modal_Price available at time t is simply Modal_Price at time t
        # Note: X must contain 'Modal_Price'
        return X['Modal_Price'].values

class MovingAverageModel:
    """Predicts using a moving average of recent observations."""
    def __init__(self, window=7):
        self.window = window
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        # We assume X contains 'Modal_Price_rolling_mean_7' or similar based on window
        col_name = f'Modal_Price_rolling_mean_{self.window}'
        if col_name in X.columns:
            return X[col_name].fillna(X['Modal_Price']).values
        else:
            raise ValueError(f"Required column {col_name} not found in features.")
