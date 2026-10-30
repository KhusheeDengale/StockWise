# StockWise — Explainable Intelligent Inventory Forecasting & Reorder Optimization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Pytest-32%20Passed-green.svg)](https://pytest.org/)

**StockWise** is an end-to-end intelligent, explainable retail inventory management and replenishment optimization system. It bridges time-series statistical modeling, multi-step supervised machine learning, Mamdani fuzzy logic risk inference, and constrained genetic algorithm optimization to automate purchasing decisions under real-world operational constraints.

The system supports two complementary data ecosystems:
1. **UCI Online Retail II**: Real-world e-commerce transaction dataset (1,067,371 rows, 4,904 products) used for continuous daily demand modeling and backtesting benchmarks.
2. **StockWise Pune v2.0**: Synthetic multi-store stationery retail network (18 stores, 100 SKUs, 3,036,504 daily sales rows, ₹424.6M net revenue) featuring store-level inventory snapshots, supplier lead-time dynamics, and cold-start new store planning.

---

## 1. System Architecture & Computational Pipeline

The system executes a strictly sequential, explainable computational pipeline where every downstream module ingests the validated mathematical output of preceding stages:

```
                  ┌──────────────────────────────────────────────┐
                  │   Retail Transaction Ingestion & Cleaning    │
                  │  (UCI E-Commerce / Pune Multi-Store Network) │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Continuous Calendar Daily Aggregation     │
                  │     (Zero-Day Insertion for Inactive Days)   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │          Time Series Diagnostics             │
                  │ (ADF Stationarity, ACF/PACF, Decomposition)  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     Supervised Feature Engineering           │
                  │  (Strict shift(1) Lags, Rolling Stats, OHE)  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     Chronological 4-Model Tournament         │
                  │ (Seasonal Naive, 7d MA, Linear Reg, ARIMA)   │
                  │     70% Train, 15% Validation, 15% Test      │
                  └──────────────────────┬───────────────────────┘
                                         │ Validation Winner + Uncertainty Proxy
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     Mamdani Fuzzy Inference System           │
                  │  (Coverage, Lead Time Pressure, Uncertainty) │
                  │     27 Rules ──▶ Reorder Urgency [0 - 100]   │
                  └──────────────────────┬───────────────────────┘
                                         │ Urgency Weights + Future Daily Forecast
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   Multi-Product Constrained Optimization     │
                  │  (Daily Simulation, GA vs. Greedy Baseline)  │
                  │  Constraints: Budget, Capacity, MOQ, Multiple│
                  │       Invariant Guarantee: Z_GA <= Z_Greedy  │
                  └──────────────────────┬───────────────────────┘
                                         │ Actionable Purchase Orders
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     Streamlit 8-Page Interactive Dashboard   │
                  │  (Overview, Explorer, Forecasting, Risk,     │
                  │   Optimizer, AI/ML Lab, Reports, New Store)  │
                  └──────────────────────────────────────────────┘
```

---

## 2. Key Modules & Methodologies

### 2.1. Data Ingestion, Cleaning & Normalization (`core/preprocessing.py`, `core/data_loader.py`)
- Standardizes transactions to a canonical schema (`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`).
- Prunes duplicates, cancellations (`InvoiceNo` starting with `C`), non-positive quantities ($Quantity \le 0$), non-positive prices ($UnitPrice \le 0$), and non-merchandise fees (`POST`, `D`, `BANK CHARGES`, etc.).
- Continuous calendar aggregation inserts zero-demand days ($DailyUnits = 0$), preventing intermittent gaps from distorting rolling metrics.

### 2.2. Feature Engineering & Selection (`core/features.py`)
- **No Target Leakage**: All autoregressive lags ($y_{t-1}, y_{t-7}, y_{t-14}, y_{t-28}$) and trailing rolling statistics ($RollingMean_7, RollingMean_{14}, RollingStd_7$) are computed via `shift(1)`.
- **Calendar Signals**: `DayOfWeek` (0–6), `Month` (1–12) with one-hot encoding, and `IsWeekend` binary flags.
- **Redundancy Pruning**: Pairs of numeric predictors with inter-feature correlation $|r| > 0.95$ are pruned, retaining the feature with higher training target correlation.
- **Warm-Up Guard**: The first 28 observations are dropped to ensure complete lag initialization.

### 2.3. Chronological Forecasting & Auto-Model Tournament (`core/forecast_models.py`, `core/forecast_evaluation.py`)
- Evaluates 4 distinct model families across fixed-origin, non-overlapping $H$-day validation blocks:
  1. **Weekly Seasonal Naive**: Recursive 7-day cyclical baseline.
  2. **7-Day Trailing Moving Average**: Recursive rolling mean.
  3. **Supervised Linear Regression**: Multi-step recursive regression with transparent coefficient breakdown.
  4. **Univariate ARIMA**: Automatic order search over $p \in \{0,1,2\}, d \in \{0,1\}, q \in \{0,1,2\}$.
- **Selection Metric**: Validation WAPE ($100 \times \frac{\sum |y_t - \hat{y}_t|}{\sum y_t}$) with tie-breaking on MAE, RMSE, and model simplicity hierarchy.
- **Forecast Uncertainty Proxy**:
  $$\text{UncertaintyScore} = \min\left(1.0, \frac{\text{ValidationMAE}}{\max(\bar{y}_{train}, 0.1)}\right)$$

### 2.4. 27-Rule Mamdani Fuzzy Inference System (`core/fuzzy_engine.py`)
- **Three Crisp Inputs**:
  - $CoverageRatio = \frac{CurrentStock}{\max(\sum_{d=1}^L \hat{y}_d, 0.1)} \in [0, 3]$ (Trapezoidal/Triangular: Low, Medium, High)
  - $LeadTimePressure = \min(1.0, LeadTimeDays / H) \in [0, 1]$ (Low, Medium, High)
  - $ForecastUncertainty \in [0, 1]$ (Low, Medium, High)
- **Inference Engine**:
  - Full factorial grid of 27 linguistic rules ($3 \times 3 \times 3$).
  - Minimum conjunction (AND), minimum implication, maximum aggregation.
- **Defuzzification**: Numerical Centroid integration across $u \in [0, 100]$ yielding crisp Reorder Urgency $\in [0, 100]$ mapped to categories: Low ($<35$), Moderate ($35-60$), High ($60-80$), Critical ($\ge 80$).

### 2.5. Multi-Product Constrained Optimization (`core/optimization.py`, `core/greedy_baseline.py`)
- **Deterministic Day-by-Day Simulation**: Accrues orders at day $d = LeadTimeDays$, deducting daily demand and calculating unmet shortage.
- **Multi-Objective Function**:
  $$\min Z = 0.55 \cdot NS + 0.20 \cdot NE + 0.15 \cdot NC + 0.10 \cdot NI$$
  where $NS$ is urgency-weighted shortage, $NE$ is excess stock above 3 days demand, $NC$ is budget expenditure, and $NI$ is ending stock coverage imbalance.
- **5 Hard Physical Constraints**:
  $$q_i \ge 0, \quad q_i = 0 \lor q_i \ge MOQ_i, \quad q_i \pmod{OrderMultiple_i} = 0$$
  $$CurrentStock_i + q_i \le StorageCapacity_i, \quad \sum_{i=1}^N q_i \times SupplierCost_i \le Budget$$
- **Chromosome Repair Operator**: Enforces all 5 hard constraints deterministically; over-budget allocations are trimmed in order-multiple steps prioritized by lowest urgency-to-cost ratio.
- **Invariant Guarantee**: Initial population seeds both the zero-order and repaired greedy solution; with elitism ($k=2$), $Z_{GA} \le Z_{Greedy}$ is mathematically guaranteed.

### 2.6. Cold-Start New Store Planning (`ui/page_new_store.py`)
- Recommends opening assortment and initial inventory for new store locations using an 840-scenario reference table and comparable-store multi-attribute similarity matching (store type, locality footfall, square footage, and customer segment mix).

---

## 3. Repository Structure

```
stockwise/
├── app.py                         # Streamlit multi-page web application entrypoint
├── config.py                      # Global paths, thresholds, and GA hyperparameters
├── requirements.txt               # Exact frozen dependency versions
├── README.md                      # Comprehensive system documentation
├── MIGRATION_AUDIT.md             # Technical audit document for Pune v2.0 migration
├── core/                          # Pure Python computational engine
│   ├── __init__.py                # Public interface (13 service endpoints)
│   ├── schemas.py                 # Strongly typed dataclasses and domain exceptions
│   ├── data_loader.py             # Dual-sheet Excel ingestion & schema normalization
│   ├── preprocessing.py           # Cleaning sequence & zero-day calendar expansion
│   ├── statistics.py              # ADF test, ACF/PACF, seasonal decomposition
│   ├── features.py                # Supervised lag/rolling features & selection
│   ├── forecast_models.py         # Seasonal Naive, 7d MA, Linear Regression, ARIMA
│   ├── forecast_evaluation.py     # Chronological backtesting & block evaluation
│   ├── forecast_service.py        # Forecasting orchestration & refitting
│   ├── fuzzy_engine.py            # 27-rule Mamdani fuzzy inference system
│   ├── optimization.py            # Genetic Algorithm purchase optimizer
│   ├── greedy_baseline.py         # Marginal efficiency greedy benchmark & simulator
│   ├── recommendation_service.py  # End-to-end recommendation synthesis & stability
│   ├── inventory.py               # Inventory config validator & generator
│   ├── pune_data.py               # Repository data access layer for Pune dataset
│   ├── reporting.py               # CSV export & tabular reporting
│   └── explanations.py            # Explainability data structures
├── ui/                            # Streamlit user interface components
│   ├── __init__.py                # UI module exports
│   ├── styles.py                  # SaaS dark/light CSS design tokens
│   ├── components.py              # KPI cards, status badges, Plotly themes
│   ├── page_overview.py           # Page 1: Business Overview
│   ├── page_explorer.py           # Page 2: Data Quality & Statistical Profiler
│   ├── page_forecasting.py        # Page 3: Demand Forecasting Studio
│   ├── page_fuzzy.py              # Page 4: Stock Risk & Fuzzy Inference
│   ├── page_optimizer.py          # Page 5: Reorder Optimizer
│   ├── page_aiml_lab.py           # Page 6: Academic AI/ML Explainability Lab
│   ├── page_reports.py            # Page 7: Final Purchase Reports & Export
│   └── page_new_store.py          # Page 8: New Store Planner (Cold-Start Setup)
├── data/                          # Processed and configuration data
│   ├── config/sample_inventory.csv# Simulated realistic inventory configuration
│   ├── processed/                 # UCI cleaned Parquet tables and catalog CSVs
│   ├── pune_cache/                # High-performance Parquet cache for Pune daily sales
│   └── stockwise_pune/            # Master CSVs for 18 Pune stores, assortment, and suppliers
├── scripts/                       # Automation and diagnostic scripts
│   ├── run_checks.py              # Automated system health check & pytest runner
│   ├── train_demo.py              # End-to-end computational demonstration
│   ├── prepare_data.py            # Offline data processing pipeline
│   └── verify_pune_data.py        # Pune data integrity & aggregate validation
└── tests/                         # Pytest test suite (32 passing automated tests)
    ├── test_data.py               # Ingestion, cleaning & schema normalization tests
    ├── test_features.py           # Lag indexing, rolling stats & leakage tests
    ├── test_forecasting.py        # 4 models, chronological splits & metric tests
    ├── test_fuzzy.py              # Membership bounds, 27 rules & centroid tests
    ├── test_optimization.py       # Hard constraints, repair operator & GA invariants
    ├── test_integration.py        # End-to-end dataflow & parameter scaling tests
    └── test_pune_integration.py   # Multi-store Pune data access & closure tests
```

---

## 4. Quickstart & Installation

### 4.1. Prerequisites
- Python 3.10 or 3.11
- Git

### 4.2. Clone & Setup Environment
```bash
# Clone repository
git clone https://github.com/KhusheeDengale/StockWise.git
cd StockWise

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4.3. Run System Health Checks & Automated Tests
Execute the verification suite to ensure all dependencies, data caches, and test suites pass:
```bash
python scripts/run_checks.py
```
Or execute `pytest` directly:
```bash
python -m pytest -v
```
*(Expected: 32 passed tests in ~25 seconds)*

### 4.4. Run the End-to-End Terminal Demo
Run the complete forecasting, fuzzy inference, and GA optimization pipeline across 5 real retail products:
```bash
python scripts/train_demo.py
```

### 4.5. Launch the Streamlit Web Application
Launch the 8-page interactive dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Experimental Results & Benchmarks

Results generated by `scripts/train_demo.py` on 5 representative retail products over a 14-day planning horizon ($H=14$) with a budget of £1,500.00:

### 5.1. Chronological Model Tournament

| StockCode | Product Description | Selected Model | Val WAPE | Test WAPE | Test $R^2$ | Uncertainty Score | 14-Day Demand |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `84077` | WORLD WAR 2 GLIDERS | Moving Average (7d) | 101.67% | 175.28% | -0.5630 | 0.6534 | 2,562.8 units |
| `85099B` | JUMBO BAG RED RETROSPOT | Linear Regression | 63.92% | 78.16% | 0.1724 | 0.6189 | 1,074.8 units |
| `21212` | PACK OF 72 CAKE CASES | Moving Average (7d) | 106.02% | 75.05% | 0.0461 | 0.5714 | 668.4 units |
| `85123A` | WHITE HANGING HEART | Moving Average (7d) | 81.44% | 88.72% | -0.0972 | 0.4367 | 1,393.0 units |
| `22197` | SMALL POPCORN HOLDER | Moving Average (7d) | 94.57% | 98.43% | 0.0010 | 1.0000 | 12,147.8 units |

### 5.2. Replenishment Optimization Benchmark

| Algorithm | Objective ($Z$) | Total Spend | Shortage | Excess | Budget Util | Runtime | Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Order Baseline** | 0.712735 | £0.00 | 11,860.0 | 1,433.8 | 0.0% | 0.0001s | 100% |
| **Greedy Heuristic** | 0.673812 | £301.40 | 11,152.8 | 1,434.2 | 20.1% | 2.3767s | 100% |
| **Genetic Algorithm** | **0.673812** | **£301.40** | **11,152.8** | **1,434.2** | **20.1%** | **0.6920s** | **100%** |

- **GA Multi-Seed Stability**: Evaluated across 5 random seeds (`[11, 21, 31, 41, 51]`), standard deviation of the objective function is **0.000000** with 100% feasibility.
- **Physical Invariant**: $Z_{GA} \le Z_{Greedy}$ is verified and satisfied across all runs.


