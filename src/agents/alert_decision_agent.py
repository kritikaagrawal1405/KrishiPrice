import logging
from src.api.schemas import ForecastResult, MarketAnalysisResult, AlertDecisionResult

logger = logging.getLogger(__name__)

class AlertDecisionAgent:
    def __init__(self, forecast_threshold_percent: float = 10.0):
        self.forecast_threshold = forecast_threshold_percent
        
    def process(self, forecast: ForecastResult, analysis: MarketAnalysisResult) -> AlertDecisionResult:
        result = AlertDecisionResult()
        
        if analysis and analysis.error:
            result.error = analysis.error
            return result
            
        logger.info("[ALERT_DECISION_AGENT] Evaluating alerts based on market analysis and forecast")
        
        # Default state
        result.alert = False
        result.alert_type = "NORMAL"
        result.severity = "LOW"
        result.message = "Market is stable. No significant price movements detected."
        result.reason = "Price fluctuations are within normal range."
        
        reasons = []
        messages = []
        is_alert = False
        max_severity = "LOW"
        alert_types = []
        
        # 1. Historical/current movement
        if analysis and analysis.significant_movement:
            is_alert = True
            
            if analysis.percentage_change > 0:
                alert_types.append("PRICE_SURGE")
                max_severity = "HIGH" if analysis.percentage_change > 10 else "MEDIUM"
                messages.append(f"Significant price surge detected. Prices increased by {analysis.percentage_change}%.")
            else:
                alert_types.append("PRICE_DROP")
                max_severity = "HIGH" if analysis.percentage_change < -10 else "MEDIUM"
                messages.append(f"Significant price drop detected. Prices decreased by {abs(analysis.percentage_change)}%.")
                
            reasons.append("Market analysis indicates volatile recent price movements.")
            
        # 2. Forecast signal
        forecast_change_percent = 0.0
        if forecast and not forecast.error and forecast.predicted_price and forecast.latest_observed_price:
            forecast_change = forecast.predicted_price - forecast.latest_observed_price
            if forecast.latest_observed_price > 0:
                forecast_change_percent = (forecast_change / forecast.latest_observed_price) * 100
                
            if abs(forecast_change_percent) >= self.forecast_threshold:
                is_alert = True
                
                if forecast_change_percent > 0:
                    alert_types.append("FORECAST_PRICE_INCREASE")
                    if max_severity != "HIGH":
                        max_severity = "HIGH" if forecast_change_percent >= (self.forecast_threshold * 2) else "MEDIUM"
                    messages.append(f"Model estimates a {round(forecast_change_percent, 2)}% increase from the latest observed price.")
                    reasons.append(f"Forecast indicates a {round(forecast_change_percent, 2)}% increase from the latest observed price.")
                else:
                    alert_types.append("FORECAST_PRICE_DECREASE")
                    if max_severity != "HIGH":
                        max_severity = "HIGH" if abs(forecast_change_percent) >= (self.forecast_threshold * 2) else "MEDIUM"
                    messages.append(f"Model estimates a {round(abs(forecast_change_percent), 2)}% decrease from the latest observed price.")
                    reasons.append(f"Forecast indicates a {round(abs(forecast_change_percent), 2)}% decrease from the latest observed price.")
        
        if is_alert:
            result.alert = True
            result.alert_type = " | ".join(alert_types)
            result.severity = max_severity
            result.message = " ".join(messages)
            result.reason = " ".join(reasons)
            
        result.supporting_data = {
            "latest_price": analysis.latest_price if analysis else None,
            "trend": analysis.trend if analysis else None,
            "percentage_change": analysis.percentage_change if analysis else None,
            "forecast_price": forecast.predicted_price if forecast else None,
            "forecast_change_percent": round(forecast_change_percent, 2)
        }
        
        return result
