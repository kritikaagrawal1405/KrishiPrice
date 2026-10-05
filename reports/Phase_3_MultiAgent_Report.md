# Phase 3 Report: Multi-Agent System & Backend

## Overview
Phase 3 of the **KrishiPrice** project successfully implements the modular Multi-Agent System (MAS) and the FastAPI backend. This phase builds upon the robust forecasting pipelines and data structures finalized in Phase 2. The core objective—creating a seamless, end-to-end processing workflow driven by natural language queries—has been achieved.

## Architecture

The system uses a highly cohesive, state-driven multi-agent architecture built on top of FastAPI and Pydantic:

1.  **FastAPI Backend (`src/api/main.py`)**: 
    - The main entrypoint for the system, offering REST endpoints for interacting with the MAS.
    - Features CORS support, strict Pydantic model validation, and structured JSON responses containing detailed intent, entity, agent insights, and grounded responses.

2.  **Shared Schemas (`src/api/schemas.py`)**:
    - Strictly typed Pydantic models (e.g., `AgentState`, `Entities`, `DataContext`, `ForecastResult`) orchestrate data flow between agents. This prevents schema drift and guarantees robust integrations.

3.  **KrishiMLM Interface (`src/krishimlm/interface.py`)**:
    - Serves as the natural language understanding bridge.
    - Simulates the eventual TensorFlow-based MLM by mapping user queries to structured `Intent` and `Entities` (e.g., extracting Commodity, Market, Horizon).
    - Generates grounded, natural-language responses aggregating insights from all downstream agents.

4.  **Multi-Agent Orchestrator (`src/orchestration/graph.py`)**:
    - The state-machine orchestrating agent execution.
    - Pipeline sequence: `Query -> KrishiMLM -> DataAgent -> MarketAnalysisAgent -> ForecastingAgent -> AlertDecisionAgent -> KrishiMLM (Response Generation)`.

## The Agents

### 1. Data Agent (`src/agents/data_agent.py`)
- Reads the clean `selected_series.csv` curated from Phase 1/2.
- Intelligently cross-references extracted entities to the exact `Series_ID` representation using dynamic pandas masking.
- Fetches the most recent `Modal_Price`, observation count, and historical availability to establish the data context.

### 2. Market Analysis Agent (`src/agents/market_analysis_agent.py`)
- Evaluates recent market trends by comparing the latest price with historical data points.
- Computes percentage changes and determines directional trends (`stable`, `increasing`, `sharply decreasing`).
- Flags significant movements based on volatility thresholds (>5%).

### 3. Forecasting Agent (`src/agents/forecasting_agent.py`)
- Configured to output future prices based on requested horizons (1-day or 7-day).
- Operates primarily using the `NaiveModel` (Last Observation Carried Forward - LOCF), aligning exactly with Phase 2's validation that declared the Naive approach as the best-performing primary baseline model.

### 4. Alert & Decision Agent (`src/agents/alert_decision_agent.py`)
- Evaluates inputs from the Market Analysis Agent and Forecasting Agent to derive business logic and alert decisions.
- Automatically generates severity classifications (`HIGH`, `MEDIUM`, `LOW`) and actionable text strings for the end user if volatility or drops/surges exceed tolerances.

## Testing & Validation
An end-to-end validation test script (`tests/test_end_to_end.py`) was successfully executed, confirming:
- Accurate entity extraction for commodities (e.g., Potato) and markets (e.g., Tulsipur).
- Flawless traversal through all four agents without exceptions or missing data links.
- Generation of detailed, actionable alert states (e.g., "PRICE_DROP") embedded naturally inside the final MLM response string.

## Conclusion & Next Steps
Phase 3 is now fully complete, fully integrated, and successfully leverages the artifact structures of Phase 2. The project is positioned securely for subsequent development phases, including a potential React-based frontend dashboard.
