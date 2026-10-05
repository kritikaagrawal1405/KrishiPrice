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
from src.agents.alert_decision_agent import AlertDecisionAgent
from src.api.schemas import DataContext, ForecastResult, MarketAnalysisResult
from src.orchestration.graph import MultiAgentOrchestrator

class TestPhase3CriticalFixes(unittest.TestCase):
    def setUp(self):
        self.forecast_agent = ForecastingAgent()
        self.alert_agent = AlertDecisionAgent(forecast_threshold_percent=10.0)

    def test_1_forecasting_agent_uses_phase2(self):
        """TEST 1: Confirm Forecasting Agent calls existing Phase 2 functionality."""
        # Using a context we know exists in the dataset
        context = DataContext(
            series_id="Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ",
            latest_date="2023-06-06",
            latest_modal_price=1040.0,
            historical_data_available=True
        )
        
        result = self.forecast_agent.process(context, horizon=1)
        
        self.assertIsNone(result.error)
        self.assertEqual(result.model, "Naive")
        # Since NaiveModel predicts the same value, it should be 1040.0
        self.assertEqual(result.predicted_price, 1040.0)
        self.assertEqual(result.latest_observed_price, 1040.0)

    def test_2_one_step_date(self):
        """TEST 2: Verify 1-step forecast date corresponds to the next available observation."""
        context = DataContext(
            series_id="Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ",
            latest_date="2023-06-06",
            latest_modal_price=1040.0,
            historical_data_available=True
        )
        
        result = self.forecast_agent.process(context, horizon=1)
        
        # In the dataset, the observation after 2023-06-06 is 2023-06-07
        self.assertEqual(result.forecast_date, "2023-06-07")

    def test_3_seven_step_date(self):
        """TEST 3: Verify 7-step forecast date is the seventh future available observation."""
        context = DataContext(
            series_id="Uttar Pradesh_balrampur_Tulsipur_Potato_Desi_FAQ",
            latest_date="2023-06-06",
            latest_modal_price=1040.0,
            historical_data_available=True
        )
        
        result = self.forecast_agent.process(context, horizon=7)
        
        # 7 rows after 2023-06-06 in the dataset is 2023-06-13
        self.assertEqual(result.forecast_date, "2023-06-13")

    def test_4_forecast_based_alert(self):
        """TEST 4: Create a deterministic test for forecast-based surge."""
        forecast = ForecastResult(
            predicted_price=2250,
            latest_observed_price=2000
        )
        analysis = MarketAnalysisResult(
            significant_movement=False,
            latest_price=2000,
            trend="stable",
            percentage_change=0
        )
        
        result = self.alert_agent.process(forecast, analysis)
        
        self.assertTrue(result.alert)
        self.assertIn("FORECAST_PRICE_INCREASE", result.alert_type)
        self.assertEqual(result.severity, "MEDIUM")

    def test_5_forecast_based_drop(self):
        """TEST 5: Create a deterministic test for forecast-based drop."""
        forecast = ForecastResult(
            predicted_price=1750,
            latest_observed_price=2000
        )
        analysis = MarketAnalysisResult(
            significant_movement=False,
            latest_price=2000,
            trend="stable",
            percentage_change=0
        )
        
        result = self.alert_agent.process(forecast, analysis)
        
        self.assertTrue(result.alert)
        self.assertIn("FORECAST_PRICE_DECREASE", result.alert_type)
        self.assertEqual(result.severity, "MEDIUM")

    def test_6_no_forecast_alert(self):
        """TEST 6: Create a test where forecast change < 10%."""
        forecast = ForecastResult(
            predicted_price=2050,  # 2.5% change
            latest_observed_price=2000
        )
        analysis = MarketAnalysisResult(
            significant_movement=False,
            latest_price=2000,
            trend="stable",
            percentage_change=0
        )
        
        result = self.alert_agent.process(forecast, analysis)
        
        self.assertFalse(result.alert)

    def test_7_end_to_end(self):
        """TEST 7: Verify the full orchestration pipeline."""
        orchestrator = MultiAgentOrchestrator()
        state = orchestrator.process_query("What is the price of Potato in Tulsipur market tomorrow?")
        
        self.assertIsNotNone(state.data_context)
        self.assertIsNotNone(state.forecast_result)
        self.assertIsNotNone(state.market_analysis)
        self.assertIsNotNone(state.alert_decision)
        self.assertIsNotNone(state.final_response)
        
        self.assertEqual(len(state.errors), 0)
        self.assertEqual(state.forecast_result.horizon, 1)

if __name__ == '__main__':
    unittest.main()
