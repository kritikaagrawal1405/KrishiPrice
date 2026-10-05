import logging
from src.api.schemas import AgentState
from src.agents.data_agent import DataAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.market_analysis_agent import MarketAnalysisAgent
from src.agents.alert_decision_agent import AlertDecisionAgent
from src.krishimlm.interface import KrishiMLMInterface

logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    def __init__(self):
        self.mlm = KrishiMLMInterface()
        self.data_agent = DataAgent()
        self.forecasting_agent = ForecastingAgent()
        self.market_agent = MarketAnalysisAgent()
        self.alert_agent = AlertDecisionAgent()
        
    def process_query(self, query: str) -> AgentState:
        logger.info(f"Processing query: {query}")
        
        # 1. Parse intent and entities
        state = self.mlm.extract_intent_and_entities(query)
        
        if not state.entities or (not state.entities.commodity and not state.entities.market):
            state.errors.append("Could not extract required entities (market or commodity) from the query.")
            state.final_response = self.mlm.generate_response(state)
            return state
            
        # 2. Data Agent
        state.data_context = self.data_agent.process(state.entities)
        if state.data_context.error:
            state.errors.append(f"Data Agent Error: {state.data_context.error}")
            state.final_response = self.mlm.generate_response(state)
            return state
            
        # 3. Market Analysis Agent
        state.market_analysis = self.market_agent.process(state.data_context)
        if state.market_analysis and state.market_analysis.error:
            state.errors.append(f"Market Analysis Error: {state.market_analysis.error}")
            
        # 4. Forecasting Agent
        horizon = state.entities.horizon if state.entities.horizon else 1
        state.forecast_result = self.forecasting_agent.process(state.data_context, horizon=horizon)
        if state.forecast_result and state.forecast_result.error:
            state.errors.append(f"Forecasting Error: {state.forecast_result.error}")
            
        # 5. Alert & Decision Agent
        state.alert_decision = self.alert_agent.process(state.forecast_result, state.market_analysis)
        
        # 6. Generate final grounded response
        state.final_response = self.mlm.generate_response(state)
        
        return state
