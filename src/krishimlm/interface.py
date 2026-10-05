import re
from src.api.schemas import Entities, AgentState

class KrishiMLMInterface:
    def __init__(self):
        pass
        
    def extract_intent_and_entities(self, query: str) -> AgentState:
        # Simple rule-based extraction mimicking the MLM for now
        query_lower = query.lower()
        
        state = AgentState(query=query)
        entities = Entities()
        
        # Horizon extraction
        if "next 7 days" in query_lower or "week" in query_lower:
            entities.horizon = 7
        else:
            entities.horizon = 1
            
        # Basic keyword matching for intent
        if "alert" in query_lower or "trend" in query_lower:
            state.intent = "market_analysis"
        else:
            state.intent = "forecast"
            
        # Basic entity extraction (Mock logic - would be replaced by actual MLM)
        # Assuming format like "What is the price of Potato in Tulsipur market?"
        if "potato" in query_lower:
            entities.commodity = "Potato"
            
        if "tulsipur" in query_lower:
            entities.market = "Tulsipur"
            
        if "asansol" in query_lower:
            entities.market = "Asansol"
            
        if "sambhal" in query_lower:
            entities.market = "Sambhal"
            
        if "uttar pradesh" in query_lower:
            entities.state = "Uttar Pradesh"
            
        if "west bengal" in query_lower:
            entities.state = "West Bengal"

        state.entities = entities
        return state

    def generate_response(self, state: AgentState) -> str:
        if state.errors:
            return f"I encountered an error while processing your request: {'; '.join(state.errors)}"
            
        # Grounded response generation
        commodity = state.entities.commodity if state.entities and state.entities.commodity else "the commodity"
        market = state.entities.market if state.entities and state.entities.market else "the selected market"
        
        response_parts = []
        
        if state.data_context and not state.data_context.error:
            latest_price = state.data_context.latest_modal_price
            latest_date = state.data_context.latest_date
            response_parts.append(f"As of {latest_date}, the latest modal price of {commodity} in {market} is ₹{latest_price} per quintal.")
            
        if state.forecast_result and not state.forecast_result.error:
            pred_price = state.forecast_result.predicted_price
            f_date = state.forecast_result.forecast_date
            horizon = state.forecast_result.horizon
            
            f_status = getattr(state.forecast_result, 'forecast_date_status', None)
            
            if f_status == "BEYOND_HISTORICAL_DATA" or not f_date:
                response_parts.append(f"The {horizon}-step forecast for {commodity} in {market} is ₹{pred_price:,.0f} based on the latest available mandi data.")
            else:
                response_parts.append(f"The {horizon}-step forecast for {commodity} in {market} on {f_date} is ₹{pred_price:,.0f}.")
            
        if state.market_analysis and not state.market_analysis.error:
            trend = state.market_analysis.trend
            response_parts.append(f"The market is currently showing a {trend} trend.")
            
        if state.alert_decision and state.alert_decision.alert:
            alert_msg = state.alert_decision.message
            response_parts.append(f"ALERT: {alert_msg}")
            
        if not response_parts:
            return "I couldn't generate a specific response based on the available data."
            
        return " ".join(response_parts)
