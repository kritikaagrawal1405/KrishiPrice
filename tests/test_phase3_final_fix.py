import os
import sys
import unittest
import pandas as pd
import warnings

# Suppress deprecation warnings for cleaner test output
warnings.filterwarnings("ignore", category=DeprecationWarning)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.forecasting_agent import ForecastingAgent
from src.krishimlm.interface import KrishiMLMInterface
from src.api.schemas import DataContext, ForecastResult, AgentState, Entities
from src.orchestration.graph import MultiAgentOrchestrator
from src.utils.config import DATA_DIR

class TestPhase3FinalFix(unittest.TestCase):
    def setUp(self):
        self.forecast_agent = ForecastingAgent()
        self.mlm_interface = KrishiMLMInterface()
        
        # We need the latest date for the series in the dataset to test BEYOND_HISTORICAL_DATA
        df = pd.read_csv(os.path.join(DATA_DIR, 'selected_series.csv'))
        df['Price Date'] = pd.to_datetime(df['Price Date'])
        series_id = "Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ"
        series_data = df[df['Series_ID'] == series_id]
        self.latest_date_str = series_data['Price Date'].max().strftime('%Y-%m-%d')
        self.latest_price = series_data.loc[series_data['Price Date'] == series_data['Price Date'].max(), 'Modal_Price'].iloc[0]

    def test_1_existing_future_target_date(self):
        """TEST 1: When requested forecast target exists in historical data."""
        # We use a date earlier than the max date
        context = DataContext(
            series_id="Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ",
            latest_date="2023-06-06",
            latest_modal_price=1040.0,
            historical_data_available=True
        )
        
        result = self.forecast_agent.process(context, horizon=1)
        
        self.assertIsNotNone(result.forecast_date)
        self.assertEqual(result.forecast_date_status, "HISTORICAL_TARGET_DATE")
        self.assertEqual(result.forecast_date, "2023-06-07")

    def test_2_latest_observation_1_step(self):
        """TEST 2: LATEST OBSERVATION, 1-STEP"""
        context = DataContext(
            series_id="Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ",
            latest_date=self.latest_date_str,
            latest_modal_price=self.latest_price,
            historical_data_available=True
        )
        
        result = self.forecast_agent.process(context, horizon=1)
        
        self.assertIsNotNone(result.predicted_price)
        self.assertIsNone(result.forecast_date)
        self.assertEqual(result.forecast_date_status, "BEYOND_HISTORICAL_DATA")
        self.assertNotEqual(result.forecast_date, "Unknown (Beyond available data)")

    def test_3_latest_observation_7_step(self):
        """TEST 3: LATEST OBSERVATION, 7-STEP"""
        context = DataContext(
            series_id="Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ",
            latest_date=self.latest_date_str,
            latest_modal_price=self.latest_price,
            historical_data_available=True
        )
        
        result = self.forecast_agent.process(context, horizon=7)
        
        self.assertIsNotNone(result.predicted_price)
        self.assertIsNone(result.forecast_date)
        self.assertEqual(result.forecast_date_status, "BEYOND_HISTORICAL_DATA")
        
        # Verify it does not just use +7 days
        import datetime
        latest_date_dt = datetime.datetime.strptime(self.latest_date_str, "%Y-%m-%d")
        plus_7 = (latest_date_dt + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        self.assertNotEqual(result.forecast_date, plus_7)

    def test_4_response_generation_null_date(self):
        """TEST 4: RESPONSE GENERATION for null date"""
        state = AgentState(
            query="What is the price of Potato in Tulsipur next 7 days?",
            entities=Entities(commodity="Potato", market="Tulsipur", horizon=7),
            data_context=DataContext(
                latest_date="2023-12-31",
                latest_modal_price=1200.0,
            ),
            forecast_result=ForecastResult(
                horizon=7,
                predicted_price=1250,
                forecast_date=None,
                forecast_date_status="BEYOND_HISTORICAL_DATA"
            )
        )
        
        response = self.mlm_interface.generate_response(state)
        
        self.assertIn("1,250", response)
        self.assertIn("7-step forecast", response)
        self.assertNotIn("Unknown", response)
        self.assertNotIn("Beyond available data", response)
        self.assertNotIn("7 days", response)  # make sure it doesn't say "in 7 days"

    def test_5_end_to_end_test(self):
        """TEST 5: EXISTING END-TO-END TEST handles latest-data case correctly"""
        # We simulate the orchestrator when DataAgent picks the *latest* available data
        # Actually, let's just mock DataAgent to return the latest date context to ensure we hit it.
        # But wait, the standard orchestrator process will hit whatever DataAgent does.
        # Let's just run it. If it gets the max date, it will hit our new logic. 
        # If it doesn't get max date, it will hit HISTORICAL_TARGET_DATE logic. Both are clean.
        
        orchestrator = MultiAgentOrchestrator()
        state = orchestrator.process_query("What is the price of Potato in Tulsipur market next 7 days?")
        
        self.assertIsNotNone(state.final_response)
        self.assertNotIn("Unknown", state.final_response)
        self.assertNotIn("Beyond available data", state.final_response)
        self.assertEqual(len(state.errors), 0)

if __name__ == '__main__':
    unittest.main()
