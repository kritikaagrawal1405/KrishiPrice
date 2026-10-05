import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import time

class ARIMAModel:
    """ARIMA Model wrapper for the pipeline."""
    def __init__(self, order=(7, 1, 1)):
        self.order = order
        self.model_fit = None
        
    def fit(self, X_train_series):
        # We need the series itself, not feature matrix
        start_time = time.time()
        model = ARIMA(X_train_series, order=self.order)
        self.model_fit = model.fit()
        self.train_time = time.time() - start_time
        return self
        
    def predict(self, history_series, steps=1):
        """
        In a rolling validation/test setting, ARIMA can use append/extend or simply predict steps ahead.
        For a true 1-step or 7-step ahead, we typically retrain or update the model with new observations.
        To avoid massive compute in Phase 2, we can fit once on train, and then do rolling forecasts
        or use `apply` to update the state with new test data.
        """
        # A simple approach for out-of-sample: 
        # Update model with history, then forecast steps.
        # This can be slow if done per observation, so we return a fitted object
        pass
    
    def rolling_forecast(self, train_series, test_series, step=1):
        """Rolling forecast for test series."""
        start_time = time.time()
        history = list(train_series)
        predictions = []
        
        attempted = len(test_series)
        successful = 0
        fallback = 0
        
        for t in range(attempted):
            history.append(test_series.iloc[t])
            try:
                model = ARIMA(history, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit()
                yhat = model_fit.forecast(steps=step)[-1]
                successful += 1
            except Exception as e:
                try:
                    # Fallback to (1,1,1) if original order fails
                    model = ARIMA(history, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
                    model_fit = model.fit()
                    yhat = model_fit.forecast(steps=step)[-1]
                    successful += 1
                except Exception as e2:
                    # Final fallback to naive (last value)
                    yhat = history[-1]
                    fallback += 1
            predictions.append(yhat)
            
        fallback_rate = (fallback / attempted) * 100 if attempted > 0 else 0
        print(f"ARIMA forecasts attempted: {attempted}")
        print(f"Successful ARIMA forecasts: {successful}")
        print(f"Fallback forecasts: {fallback}")
        print(f"Fallback rate: {fallback_rate:.2f}%")
        
        return predictions, time.time() - start_time
