# KrishiPrice

### Autonomous Multi-Agent System for Real-Time Mandi Price Prediction and Alerts

**Project:** Development of Micro Language Model for Ministry of
Agriculture & Farmers Welfare using TensorFlow Framework\
**System:** KrishiPrice\
**Project Type:** Final-Year Major Project

------------------------------------------------------------------------

## 1. Project Objective

KrishiPrice is an AI-based agricultural decision-support system for
mandi prices.

The system will:

1.  Process historical mandi price data.
2.  Forecast future prices for a selected commodity and market.
3.  Analyze market trends, anomalies, and price differences.
4.  Generate relevant price alerts and decision-support information.
5.  Allow farmers and ministry users to interact with the system using
    natural-language queries.
6.  Use a **real TensorFlow-based Micro Language Model (MLM)** to
    understand agricultural queries and communicate results.
7.  Use a small number of **real autonomous agents** to perform and
    coordinate the required tasks.

The system is **decision support**, not a guaranteed price or financial
prediction service.

------------------------------------------------------------------------

# 2. Dataset

The provided dataset is:

`Agriculture_price_dataset.csv`

Current dataset size:

-   **737,392 records**
-   **10 columns**

### Columns

  Column          Description
  --------------- ------------------------
  STATE           State
  District Name   District
  Market Name     Mandi/market
  Commodity       Agricultural commodity
  Variety         Commodity variety
  Grade           Grade
  Min_Price       Minimum observed price
  Max_Price       Maximum observed price
  Modal_Price     Modal/prevailing price
  Price Date      Date of observation

### Initial prediction target

**Modal_Price**

The initial forecasting problem is:

> Given the historical price series for a commodity-market combination,
> predict its future modal price.

------------------------------------------------------------------------

# 3. Final System Architecture

The system uses a **hybrid sequential + parallel agent workflow**.

``` text
                         USER / MINISTRY
                               |
                               v
                      +------------------+
                      |    MICRO MLM     |
                      |    TensorFlow    |
                      |    Transformer   |
                      +--------+---------+
                               |
                         Intent + Entities
                               |
                               v
                       +---------------+
                       | Agent Router  |
                       +-------+-------+
                               |
                               v
                        +-------------+
                        |  DATA AGENT |
                        +------+------+ 
                               |
                         Clean/Validated
                              Data
                               |
                    +----------+----------+
                    |                     |
                    |     PARALLEL       |
                    v                     v
           +----------------+    +--------------------+
           |  FORECASTING   |    | MARKET ANALYSIS    |
           |     AGENT      |    |      AGENT         |
           |                |    |                    |
           | TensorFlow ML  |    | Trend              |
           | Price forecast |    | Anomaly            |
           |                |    | Market comparison  |
           +-------+--------+    +---------+----------+
                   |                       |
                   +----------+------------+
                              |
                              v
                 +--------------------------+
                 | ALERT & DECISION AGENT   |
                 |                          |
                 | Alerts + decision        |
                 | support                  |
                 +------------+-------------+
                              |
                              v
                       Structured Result
                              |
                              v
                      +---------------+
                      |    MICRO MLM  |
                      | Natural-language
                      | response      |
                      +-------+-------+
                              |
                              v
                         USER / MINISTRY
```

### Execution model

The workflow is intentionally **not completely sequential**.

``` text
DATA AGENT
    |
    v
Clean / Valid Data
    |
    +-----------------------+
    |                       |
    v                       v
FORECASTING AGENT     MARKET ANALYSIS AGENT
    |                       |
    |                       |
    +-----------+-----------+
                |
                v
       ALERT & DECISION AGENT
                |
                v
             RESULT
```

### Why parallel?

The Forecasting Agent and Market Analysis Agent use the same cleaned
data but perform independent operations.

They can therefore execute concurrently.

The Alert & Decision Agent waits for their outputs because it needs both
the forecast and market analysis before generating the final
decision-support result.

------------------------------------------------------------------------

# 4. Agents --- Only Four

We will **not create unnecessary agents**.

## Agent 1 --- Data Agent

### Responsibility

Handles mandi data before it reaches the forecasting and analysis
components.

### Tasks

-   Validate incoming data
-   Handle missing/invalid values
-   Detect duplicate/problematic records
-   Prepare time-series data
-   Provide clean data to downstream agents

### Position in workflow

**First agent after the request/data trigger.**

``` text
Data Source
    ↓
Data Agent
    ↓
Clean Data
```

------------------------------------------------------------------------

## Agent 2 --- Forecasting Agent

### Responsibility

Produces future mandi price predictions.

### Tasks

-   Retrieve the required historical series
-   Prepare forecasting features
-   Run the selected forecasting model
-   Produce predictions for the requested horizon

### Experimental models

``` text
Naive baseline
ARIMA
LSTM
GRU
```

LSTM and GRU are **experimental alternatives**, not two required
production models.

The final deployed forecasting model is selected based on validation
performance.

### Output

Example:

``` json
{
  "market": "Lasalgaon",
  "commodity": "Onion",
  "current_price": 2300,
  "forecast": {
    "day_1": 2380,
    "day_3": 2450,
    "day_7": 2520
  }
}
```

------------------------------------------------------------------------

## Agent 3 --- Market Analysis Agent

### Responsibility

Analyzes the current and predicted market situation.

### Tasks

-   Determine price trend
-   Detect significant/anomalous price movements
-   Compare selected markets
-   Compare current price with historical and forecast values

These functions remain inside **one agent** rather than creating
separate trend, anomaly, and comparison agents.

### Output

Example:

``` json
{
  "trend": "increasing",
  "change_percent": 9.6,
  "anomaly": false,
  "market_comparison": {
    "Lasalgaon": 2300,
    "Pune": 2150,
    "Nashik": 2400
  }
}
```

------------------------------------------------------------------------

## Agent 4 --- Alert & Decision Agent

### Responsibility

Combines forecasting and market-analysis outputs and determines whether
an alert or decision-support message should be generated.

### Inputs

``` text
Forecasting Agent output
+
Market Analysis Agent output
```

### Tasks

-   Apply predefined alert rules
-   Detect significant expected price increases/decreases
-   Generate price alerts
-   Produce decision-support information

Example:

``` text
Current price: ₹2,300
Predicted 7-day price: ₹2,520
Expected change: +9.6%
Trend: Increasing

→ Generate price-increase alert
```

The system should communicate uncertainty and should not claim that a
predicted price is guaranteed.

------------------------------------------------------------------------

# 5. Real Micro Language Model

The MLM is a **real machine-learning component**, not an API call to
ChatGPT/Gemini and not merely a wrapper around an existing LLM.

## KrishiMLM

The project will develop a small domain-specific Transformer model
using:

**TensorFlow / Keras**

### Responsibilities

1.  Intent classification
2.  Entity extraction
3.  Query understanding
4.  Natural-language response generation from structured system results

### Example

User:

> Will onion prices increase in Lasalgaon in the next 7 days?

MLM output:

``` json
{
  "intent": "PRICE_FORECAST",
  "commodity": "onion",
  "market": "Lasalgaon",
  "horizon": 7
}
```

The Forecasting Agent performs the actual numerical prediction.

The MLM does **not** invent the numerical price.

If the forecasting system returns:

``` json
{
  "current_price": 2300,
  "predicted_7_day_price": 2520,
  "trend": "increasing"
}
```

the MLM can generate:

> The current modal onion price in Lasalgaon is ₹2,300. The forecasting
> model predicts approximately ₹2,520 over the next 7 days, indicating
> an expected upward trend.

The numerical prediction comes from the forecasting model.

------------------------------------------------------------------------

# 6. MLM Architecture

``` text
User Query
    |
    v
Tokenizer
    |
    v
Token Embedding
    |
    v
Positional Encoding
    |
    v
Transformer Encoder
    |
    +----------------------+
    |                      |
    v                      v
Intent Head           Entity Head
    |                      |
    v                      v
Intent                 Entities
```

The MLM should remain small and domain-specific.

It does not need to be a large general-purpose language model.

------------------------------------------------------------------------

# 7. How MLM and Agents Work Together

The MLM and agents have different responsibilities.

### MLM

``` text
Language understanding
Intent
Entities
Natural-language response
```

### Agents

``` text
Data processing
Forecasting
Market analysis
Alert/decision processing
```

### Example complete workflow

User:

> Will onion prices increase in Lasalgaon next week?

``` text
USER
  |
  v
MICRO MLM
  |
  | Intent = PRICE_FORECAST
  | Commodity = Onion
  | Market = Lasalgaon
  | Horizon = 7
  |
  v
AGENT ROUTER
  |
  v
DATA AGENT
  |
  v
Clean historical data
  |
  +-----------------------------+
  |                             |
  v                             v
FORECASTING AGENT          MARKET ANALYSIS AGENT
  |                             |
  v                             v
TensorFlow model             Trend/Anomaly/
  |                         Market comparison
  |                             |
  +-------------+---------------+
                |
                v
       ALERT & DECISION AGENT
                |
                v
         Structured result
                |
                v
             MICRO MLM
                |
                v
       Natural-language answer
                |
                v
              USER
```

------------------------------------------------------------------------

# 8. Project Phases

The entire project is divided into **5 phases**.

------------------------------------------------------------------------

## PHASE 1 --- Data Pipeline & Problem Definition

### Goal

Understand and prepare the provided dataset for forecasting.

### Tasks

-   Dataset audit
-   Missing-value analysis
-   Duplicate analysis
-   Date-range analysis
-   Commodity/market analysis
-   Identify usable commodity-market time series
-   Convert and sort dates
-   Clean invalid prices
-   Define forecasting target
-   Create chronological train/validation/test splits
-   Feature engineering

### Important features

-   Lag prices
-   Rolling mean
-   Rolling standard deviation
-   Price spread
-   Month/season features
-   Other useful historical features

### Deliverables

``` text
Clean dataset
EDA report
Feature pipeline
Selected time-series
Train/validation/test datasets
```

### Phase completion condition

We must know:

> Which commodity-market series will be used for the first complete
> prototype and why.

------------------------------------------------------------------------

# PHASE 2 --- Forecasting Engine

### Goal

Build and evaluate the actual price prediction system.

### Step 1 --- Baselines

Implement:

``` text
Naive
Moving Average
ARIMA
```

### Step 2 --- TensorFlow models

Implement and compare:

``` text
LSTM
GRU
```

Do not assume LSTM is automatically better.

### Evaluation

Use:

-   MAE
-   RMSE
-   sMAPE/WAPE where appropriate
-   Directional accuracy where useful

### Important rule

The data must be split chronologically.

Do **not** randomly mix future and past observations.

### Model selection

``` text
Naive
   |
ARIMA
   |
LSTM
   |
GRU
   |
Compare validation results
   |
Select appropriate final model
```

### Deliverables

``` text
Trained models
Evaluation results
Model comparison
Selected forecasting model
Forecasting inference pipeline
```

------------------------------------------------------------------------

# PHASE 3 --- KrishiMLM

### Goal

Build the real domain-specific Micro Language Model.

### Model

A small TensorFlow/Keras Transformer.

### Required intents

Keep the initial set focused:

``` text
CURRENT_PRICE
PRICE_FORECAST
MARKET_COMPARISON
PRICE_TREND
PRICE_ALERT
```

### Required entities

At minimum:

``` text
Commodity
Market
State
Forecast horizon
```

### Training data

Create a domain-specific dataset containing agricultural queries such
as:

``` text
"What is today's onion price?"
"Predict onion price for the next 7 days."
"Compare onion prices in Pune and Nashik."
"Is tomato price increasing?"
"Alert me when onion prices rise."
```

Each training example should contain the appropriate intent and entity
annotations.

### Deliverables

``` text
KrishiMLM
Tokenizer
Training dataset
Intent classifier
Entity extraction component
Evaluation results
Inference pipeline
```

------------------------------------------------------------------------

# PHASE 4 --- Multi-Agent System & Backend

### Goal

Connect the MLM, agents, forecasting engine, and database into one
working system.

### Agent workflow

``` text
User / New Data
      |
      v
  MICRO MLM
      |
      v
Intent + Entities
      |
      v
 Agent Router
      |
      v
 DATA AGENT
      |
      v
 Clean Data
      |
      +----------------------+
      |                      |
      v                      v
FORECASTING AGENT     MARKET ANALYSIS AGENT
      |                      |
      |      PARALLEL        |
      +----------+-----------+
                 |
                 v
       ALERT & DECISION AGENT
                 |
                 v
          Structured Result
                 |
                 v
             MICRO MLM
                 |
                 v
             Response
```

### Backend

Recommended:

``` text
FastAPI
```

### Agent orchestration

A stateful orchestration framework such as **LangGraph** can be used.

The framework is only the orchestration layer. Each agent must perform
an actual project responsibility.

### Parallel execution

The Forecasting Agent and Market Analysis Agent can run concurrently
because they operate independently on the cleaned data.

The Alert & Decision Agent waits for both outputs.

``` text
Data Agent
    |
    +-------------------+
    |                   |
    v                   v
Forecasting        Market Analysis
    |                   |
    +---------+---------+
              |
              v
       Alert & Decision
```

### Deliverables

``` text
Four working agents
Agent router/orchestrator
FastAPI backend
Database integration
API endpoints
End-to-end agent workflow
```

------------------------------------------------------------------------

# PHASE 5 --- Dashboard, Integration & Final Evaluation

### Goal

Create the final KrishiPrice application and evaluate the complete
system.

## Farmer interface

Should provide:

-   State/market/commodity selection
-   Current/latest price
-   Future price forecast
-   Price trend
-   Market comparison
-   Alerts
-   Natural-language query interface

## Ministry interface

Should provide:

-   Regional market overview
-   Commodity trends
-   Significant price movements
-   Forecasts
-   Alerts
-   Market comparison

### Final system test

``` text
Data
 ↓
Data Agent
 ↓
+-----------------------+
|                       |
v                       v
Forecasting Agent   Market Analysis Agent
|                       |
+-----------+-----------+
            |
            v
Alert & Decision Agent
            |
            v
      Final Result
            |
            v
         MicroMLM
            |
            v
        User/UI
```

### Evaluate three areas

#### Forecasting

``` text
MAE
RMSE
sMAPE/WAPE
Directional accuracy
```

#### MLM

``` text
Intent accuracy
Entity extraction performance
Response correctness
```

#### Complete system

Test:

``` text
Correct query routing
Correct agent execution
Correct numerical result
Correct alert generation
No hallucinated prices
```

### Deliverables

``` text
Final web application
Integrated AI system
Final evaluation
Graphs/charts
Documentation
Final project report
Demo
```

------------------------------------------------------------------------

# 9. Proposed Technology Stack

  Component                 Technology
  ------------------------- ------------------------------
  Programming               Python
  Data Processing           Pandas, NumPy
  ML                        Scikit-learn
  Statistical Forecasting   Statsmodels
  Deep Learning             TensorFlow / Keras
  MLM                       TensorFlow/Keras Transformer
  Backend                   FastAPI
  Agent Orchestration       LangGraph
  Database                  PostgreSQL
  Frontend                  React / Next.js
  Visualization             Plotly / Chart.js
  Version Control           Git + GitHub

------------------------------------------------------------------------

# 10. Repository Structure

``` text
krishi-price/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_lstm_gru_experiments.ipynb
│   └── 04_mlm_experiments.ipynb
│
├── src/
│   │
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   └── feature_engineering.py
│   │
│   ├── forecasting/
│   │   ├── baseline.py
│   │   ├── arima.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   ├── evaluation.py
│   │   └── inference.py
│   │
│   ├── mlm/
│   │   ├── tokenizer.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── dataset.py
│   │
│   ├── agents/
│   │   ├── data_agent.py
│   │   ├── forecasting_agent.py
│   │   ├── market_analysis_agent.py
│   │   └── alert_decision_agent.py
│   │
│   ├── orchestration/
│   │   └── graph.py
│   │
│   └── api/
│       └── main.py
│
├── models/
│   ├── forecasting/
│   └── krishimlm/
│
├── frontend/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

# 11. Team Division

Responsibilities should be divided by modules.

### Member 1 --- Data & Forecasting

-   Dataset analysis
-   Preprocessing
-   Feature engineering
-   Baseline models
-   LSTM/GRU experiments
-   Model evaluation
-   Final forecasting model

### Member 2 --- KrishiMLM

-   MLM dataset
-   Tokenization
-   Transformer architecture
-   Intent classification
-   Entity extraction
-   MLM evaluation

### Member 3 --- Agents & Backend

-   Four agents
-   Agent orchestration
-   FastAPI
-   Database integration
-   End-to-end backend workflow

### Member 4 --- Frontend & Integration

-   Farmer dashboard
-   Ministry dashboard
-   Forecast visualization
-   Alerts
-   MLM interface
-   Final integration/testing

All members should understand the complete architecture for the final
viva.

------------------------------------------------------------------------

# 12. What We Are NOT Building

To keep the project focused:

-   No unnecessary large language model/API dependency
-   No 7--8 artificial agents
-   No separate agent for every small function
-   No requirement to deploy both LSTM and GRU
-   No model that claims guaranteed future prices
-   No giant language model trained from scratch
-   No unnecessary blockchain/IoT components
-   No unrelated recommendation systems
-   No random train/test split for time-series forecasting

------------------------------------------------------------------------

# 13. Definition of Done

## Data

-   [ ] Dataset cleaned and validated
-   [ ] Usable time-series identified
-   [ ] Chronological train/validation/test split implemented

## Forecasting

-   [ ] Baselines implemented
-   [ ] ARIMA evaluated
-   [ ] LSTM and/or GRU experimentally evaluated
-   [ ] Final model selected using validation results
-   [ ] Forecasting metrics reported

## MLM

-   [ ] Domain-specific dataset created
-   [ ] TensorFlow Transformer trained
-   [ ] Intent classification works
-   [ ] Entity extraction works
-   [ ] MLM does not invent numerical predictions
-   [ ] MLM can convert structured results into user-readable responses

## Agents

-   [ ] Data Agent works
-   [ ] Forecasting Agent works
-   [ ] Market Analysis Agent works
-   [ ] Alert & Decision Agent works
-   [ ] Forecasting and Market Analysis can execute in parallel
-   [ ] Alert & Decision waits for their outputs
-   [ ] Agents communicate through the orchestration layer

## Application

-   [ ] Farmer dashboard works
-   [ ] Ministry dashboard works
-   [ ] Forecasts are displayed
-   [ ] Alerts are generated
-   [ ] Natural-language queries work
-   [ ] Complete end-to-end workflow works

------------------------------------------------------------------------

# 14. One-Line Project Definition

> **KrishiPrice is a TensorFlow-based agricultural decision-support
> system that combines a domain-specific Micro Language Model with four
> autonomous agents to process mandi data, forecast prices, analyze
> market conditions, and generate farmer/ministry alerts.**

------------------------------------------------------------------------

# 15. Development Order

The team should follow this order:

``` text
PHASE 1
Data Pipeline
     |
     v
PHASE 2
Forecasting Engine
     |
     v
PHASE 3
KrishiMLM
     |
     v
PHASE 4
Agents + Backend
     |
     v
PHASE 5
Dashboard + Integration + Evaluation
```

Within the final agent workflow:

``` text
Data Agent
     |
     +---------------------+
     |                     |
     v                     v
Forecasting Agent    Market Analysis Agent
     |                     |
     +----------+----------+
                |
                v
       Alert & Decision Agent
                |
                v
             Result
```

**Do not start agent integration until the basic forecasting pipeline is
working.**

The forecasting engine provides the numerical intelligence. The MLM
provides language intelligence. The four agents provide autonomous
orchestration and decision-support.
