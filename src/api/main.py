import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import QueryRequest, QueryResponse
from src.orchestration.graph import MultiAgentOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KrishiPrice Multi-Agent API",
    description="Backend API for the KrishiPrice Multi-Agent System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator on startup
orchestrator = MultiAgentOrchestrator()

@app.get("/")
def read_root():
    return {"status": "KrishiPrice API is running"}

@app.post("/api/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    try:
        state = orchestrator.process_query(request.query)
        
        response = QueryResponse(
            query=state.query,
            intent=state.intent,
            entities=state.entities,
            latest_price=state.data_context.latest_modal_price if state.data_context else None,
            forecast=state.forecast_result.model_dump() if state.forecast_result else None,
            market_analysis=state.market_analysis.model_dump() if state.market_analysis else None,
            alert=state.alert_decision.model_dump() if state.alert_decision else None,
            response=state.final_response
        )
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
