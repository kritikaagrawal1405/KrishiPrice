import pandas as pd
import logging
import os
from src.api.schemas import DataContext, MarketAnalysisResult
from src.utils.config import DATA_DIR

logger = logging.getLogger(__name__)

class MarketAnalysisAgent:
    def __init__(self):
        self.data_path = os.path.join(DATA_DIR, 'selected_series.csv')
        self.df = None
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            self.df['Price Date'] = pd.to_datetime(self.df['Price Date'])
            
    def process(self, context: DataContext) -> MarketAnalysisResult:
        result = MarketAnalysisResult()
        
        if not context or context.error or self.df is None:
            result.error = "Invalid DataContext or missing data."
            return result
            
        logger.info(f"[MARKET_ANALYSIS_AGENT] Analyzing market trend for {context.series_id}")
        
        # Get historical data for the series
        series_data = self.df[self.df['Series_ID'] == context.series_id].copy()
        series_data = series_data.sort_values('Price Date')
        
        if len(series_data) < 2:
            result.error = "Not enough data points for market analysis."
            return result
            
        prices = series_data['Modal_Price'].values
        latest_price = float(prices[-1])
        previous_price = float(prices[-2])
        
        diff = latest_price - previous_price
        pct_change = (diff / previous_price) * 100 if previous_price > 0 else 0
        
        if pct_change > 5:
            trend = "sharply increasing"
        elif pct_change > 1:
            trend = "increasing"
        elif pct_change < -5:
            trend = "sharply decreasing"
        elif pct_change < -1:
            trend = "decreasing"
        else:
            trend = "stable"
            
        result.trend = trend
        result.latest_price = latest_price
        result.previous_price = previous_price
        result.percentage_change = round(pct_change, 2)
        result.significant_movement = abs(pct_change) >= 5.0
        result.market_comparison = None
        
        return result
