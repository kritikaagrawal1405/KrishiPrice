import pandas as pd
import os
from src.api.schemas import Entities, DataContext
from src.utils.config import DATA_DIR

class DataAgent:
    def __init__(self):
        self.data_path = os.path.join(DATA_DIR, 'selected_series.csv')
        self.df = None
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            self.df['Price Date'] = pd.to_datetime(self.df['Price Date'])
            
    def process(self, entities: Entities) -> DataContext:
        context = DataContext()
        
        if self.df is None:
            context.error = "Data file not found."
            return context
            
        # Match series based on provided entities
        mask = pd.Series(True, index=self.df.index)
        
        if entities.commodity:
            mask = mask & (self.df['Commodity'].str.lower() == entities.commodity.lower())
        if entities.market:
            mask = mask & (self.df['Market Name'].str.lower() == entities.market.lower())
        if entities.state:
            mask = mask & (self.df['STATE'].str.lower() == entities.state.lower())
            
        filtered = self.df[mask]
        
        if filtered.empty:
            context.error = f"No data found for commodity '{entities.commodity}' in market '{entities.market}'."
            return context
            
        # Resolve to the most common exact series_id matching the criteria
        series_id = filtered['Series_ID'].value_counts().idxmax()
        series_data = self.df[self.df['Series_ID'] == series_id].copy()
        series_data = series_data.sort_values('Price Date')
        
        latest_row = series_data.iloc[-1]
        
        context.series_id = series_id
        context.state = latest_row['STATE']
        context.district = latest_row['District Name']
        context.market = latest_row['Market Name']
        context.commodity = latest_row['Commodity']
        context.variety = latest_row['Variety']
        context.grade = latest_row['Grade']
        
        context.latest_date = latest_row['Price Date'].strftime('%Y-%m-%d')
        context.latest_modal_price = float(latest_row['Modal_Price'])
        context.observation_count = len(series_data)
        context.historical_data_available = context.observation_count > 30
        
        if not context.historical_data_available:
            context.error = "Insufficient historical data for forecasting."
            
        return context
