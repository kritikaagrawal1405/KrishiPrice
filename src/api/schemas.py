from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class QueryRequest(BaseModel):
    query: str

class Entities(BaseModel):
    commodity: Optional[str] = None
    market: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    variety: Optional[str] = None
    grade: Optional[str] = None
    horizon: Optional[int] = None

class DataContext(BaseModel):
    series_id: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    market: Optional[str] = None
    commodity: Optional[str] = None
    variety: Optional[str] = None
    grade: Optional[str] = None
    latest_date: Optional[str] = None
    latest_modal_price: Optional[float] = None
    historical_data_available: bool = False
    observation_count: int = 0
    error: Optional[str] = None

class ForecastResult(BaseModel):
    series_id: Optional[str] = None
    model: Optional[str] = None
    horizon: Optional[int] = None
    forecast_date: Optional[str] = None
    forecast_date_status: Optional[str] = None
    predicted_price: Optional[float] = None
    latest_observed_price: Optional[float] = None
    confidence: Optional[Any] = None
    error: Optional[str] = None

class MarketAnalysisResult(BaseModel):
    trend: Optional[str] = None
    latest_price: Optional[float] = None
    previous_price: Optional[float] = None
    percentage_change: Optional[float] = None
    significant_movement: bool = False
    market_comparison: Optional[Any] = None
    error: Optional[str] = None

class AlertDecisionResult(BaseModel):
    alert: bool = False
    alert_type: Optional[str] = None
    severity: str = "NONE"
    message: str = "No significant alert condition detected."
    reason: Optional[str] = None
    supporting_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AgentState(BaseModel):
    query: str
    intent: Optional[str] = None
    entities: Optional[Entities] = None
    data_context: Optional[DataContext] = None
    forecast_result: Optional[ForecastResult] = None
    market_analysis: Optional[MarketAnalysisResult] = None
    alert_decision: Optional[AlertDecisionResult] = None
    final_response: Optional[str] = None
    errors: List[str] = []

class QueryResponse(BaseModel):
    query: str
    intent: Optional[str] = None
    entities: Optional[Entities] = None
    latest_price: Optional[float] = None
    forecast: Optional[Dict[str, Any]] = None
    market_analysis: Optional[Dict[str, Any]] = None
    alert: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
