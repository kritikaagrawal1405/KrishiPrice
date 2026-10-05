import pandas as pd
import numpy as np
import os
import logging
from src.api.schemas import DataContext, ForecastResult
from src.utils.config import DATA_DIR
from src.forecasting.baselines import NaiveModel
from src.data.feature_engineering import create_targets

logger = logging.getLogger(__name__)

class ForecastingAgent:
    def __init__(self):
        self.data_path = os.path.join(DATA_DIR, 'selected_series.csv')
        self.df = None
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            self.df['Price Date'] = pd.to_datetime(self.df['Price Date'])
            
    def process(self, context: DataContext, horizon: int = 1) -> ForecastResult:
        result = ForecastResult()
        
        if not context or context.error:
            result.error = "Invalid or missing DataContext."
            return result
            
        if not context.historical_data_available:
            result.error = "Insufficient historical data for forecasting."
            return result
            
        if horizon not in [1, 7]:
            result.error = f"Unsupported forecast horizon: {horizon}. Only 1 and 7 are supported."
            return result
            
        if self.df is None:
            result.error = "Data file not found for forecasting."
            return result
            
        logger.info(f"[FORECASTING_AGENT] Generating {horizon}-step forecast using Naive model for {context.series_id}")
        
        # Isolate the requested series
        series_data = self.df[self.df['Series_ID'] == context.series_id].copy()
        series_data = series_data.sort_values('Price Date').reset_index(drop=True)
        
        # Reuse Phase 2 logic to determine target dates
        series_data = create_targets(series_data, horizons=[horizon])
        
        # Locate the context's observation
        obs_df = series_data[series_data['Price Date'] == pd.to_datetime(context.latest_date)]
        
        if obs_df.empty:
            result.error = f"Observation for date {context.latest_date} not found."
            return result
            
        # Call Phase 2 forecasting functionality
        model = NaiveModel()
        predicted_price = float(model.predict(obs_df)[0])
        
        target_date_ts = obs_df.iloc[0][f'target_date_{horizon}']
        
        if pd.isna(target_date_ts):
            result.forecast_date = None
            result.forecast_date_status = "BEYOND_HISTORICAL_DATA"
        else:
            result.forecast_date = target_date_ts.strftime('%Y-%m-%d')
            result.forecast_date_status = "HISTORICAL_TARGET_DATE"
            
        result.series_id = context.series_id
        result.model = "Naive"
        result.horizon = horizon
        result.predicted_price = predicted_price
        result.latest_observed_price = context.latest_modal_price
        result.confidence = None
        
        return result

