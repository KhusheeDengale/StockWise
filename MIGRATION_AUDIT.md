# MIGRATION_AUDIT.md — StockWise UCI → Pune Migration

## Repository Audit (2026-10-20)

### Current Structure
```
stockwise/
├── app.py                          # Streamlit entry point (7 pages, UCI branding)
├── config.py                       # Paths, thresholds, GA hyperparameters
├── requirements.txt                # pandas, numpy, scipy, sklearn, statsmodels, skfuzzy, plotly, streamlit
├── README.md
├── core/
│   ├── __init__.py                 # Public API: 13 service functions + exports
│   ├── schemas.py                  # ForecastResult, FuzzyResult, OptimizationResult, RecommendationReport
│   ├── data_loader.py              # UCI Excel dual-sheet loader + column normalization
│   ├── preprocessing.py            # UCI cleaning (cancellations, neg qty, non-merchandise), daily aggregation
│   ├── statistics.py               # Descriptive stats, ADF, ACF/PACF, decomposition, correlations
│   ├── features.py                 # Lag/rolling features for Linear Regression
│   ├── forecast_models.py          # Seasonal Naive, MA(7), Linear Regression, ARIMA
│   ├── forecast_evaluation.py      # Chronological backtesting, block evaluation, model selection
│   ├── forecast_service.py         # Orchestrator: evaluate → select → refit → forecast
│   ├── inventory.py                # Inventory CSV loader, validator, demo generator
│   ├── fuzzy_engine.py             # 27-rule Mamdani FIS for reorder urgency
│   ├── optimization.py             # GA optimizer (population, tournament, crossover, mutation, repair)
│   ├── greedy_baseline.py          # Greedy optimizer + simulation + normalization
│   ├── recommendation_service.py   # End-to-end: forecast → fuzzy → GA → recommendations
│   ├── explanations.py             # Explainability dataclasses
│   └── reporting.py                # CSV exports, summary tables
├── ui/
│   ├── __init__.py                 # UI public exports
│   ├── styles.py                   # SaaS CSS design tokens
│   ├── components.py               # KPI cards, badges, info boxes, currency formatting
│   ├── page_overview.py            # Business overview (revenue, units, trends)
│   ├── page_explorer.py            # Data quality, statistics, correlations
│   ├── page_forecasting.py         # Demand forecasting studio
│   ├── page_fuzzy.py               # Stock risk assessment
│   ├── page_optimizer.py           # Reorder optimizer
│   ├── page_aiml_lab.py            # Academic explanations
│   └── page_reports.py             # Purchase recommendations & reports
├── data/
│   ├── raw/online_retail_II.xlsx   # UCI dataset (45 MB)
│   ├── processed/                  # Parquet/CSV derived from UCI
│   ├── config/sample_inventory.csv # Simulated inventory (30 SKUs, GBP costs)
│   └── stockwise_pune/             # NEW: 12 Pune CSVs extracted from ZIP
├── scripts/
│   ├── prepare_data.py             # UCI preprocessing pipeline
│   ├── train_demo.py               # End-to-end demo on 5 products
│   └── run_checks.py               # Diagnostic runner
├── tests/ (6 test files)
└── models/ (empty)
```

### Feature Reuse/Adapt/Add/Retire Matrix

| Component | Action | Notes |
|-----------|--------|-------|
| **forecast_models.py** | ✅ REUSE | Seasonal Naive, MA, LR, ARIMA — algorithm-agnostic, work on any daily series |
| **forecast_evaluation.py** | ✅ REUSE | Chronological backtesting — needs Date+DailyUnits columns |
| **forecast_service.py** | 🔄 ADAPT | Change cache key to include StoreID, adapt column names |
| **fuzzy_engine.py** | ✅ REUSE | Pure math: 27 rules, membership functions — input-agnostic |
| **optimization.py** | ✅ REUSE | GA engine — uses products_data dicts, budget param |
| **greedy_baseline.py** | ✅ REUSE | Greedy + simulation — same interface as GA |
| **recommendation_service.py** | 🔄 ADAPT | Change field names (StockCode→ProductID, GBP→INR) |
| **statistics.py** | ✅ REUSE | All stat functions work on any numeric series |
| **features.py** | ✅ REUSE | Lag/rolling features — series-agnostic |
| **schemas.py** | 🔄 ADAPT | Add store_id field, rename GBP→INR in display |
| **data_loader.py** | 🔀 RETIRE (legacy) | UCI-specific Excel loader — not needed for Pune |
| **preprocessing.py** | 🔀 RETIRE (legacy) | UCI cleaning (cancellations etc.) — Pune data is pre-cleaned |
| **inventory.py** | 🔄 ADAPT | Pune uses pune_current_inventory.csv with different schema |
| **explanations.py** | ✅ REUSE | Dataclasses for transparency |
| **reporting.py** | 🔄 ADAPT | Update labels GBP→INR |
| **config.py** | 🔄 ADAPT | Add Pune data paths, keep GA/forecast settings |
| **app.py** | 🔄 ADAPT | Add workflow switch, Pune sidebar, 8th page |
| **ui/styles.py** | ✅ REUSE | CSS design tokens unchanged |
| **ui/components.py** | 🔄 ADAPT | Already has ₹ INR formatting from prior changes |
| **ui/page_*.py** (7 pages) | 🔄 ADAPT | Replace UCI data calls with Pune adapter |
| **core/__init__.py** | 🔄 ADAPT | New Pune-mode initialization and service functions |
| **Pune data repository** | ➕ ADD | New `core/pune_data.py` — thin data access layer |
| **New Store Planner** | ➕ ADD | New `core/new_store_service.py` + `ui/page_new_store.py` |

### Key Interface Mappings (UCI → Pune)

| UCI Concept | Pune Equivalent |
|-------------|-----------------|
| StockCode | ProductID |
| Description | ProductName |
| DailyUnits | UnitsSold |
| DailyRevenueGBP | NetRevenueINR |
| AverageSellingPriceGBP | ActualSellingPriceINR |
| RevenueGBP | NetRevenueINR |
| SupplierCostGBP | SupplierUnitCostINR |
| Country | LocalityName / AreaID |
| InvoiceNo | (not applicable — daily aggregated) |
| InvoiceDate | Date |
| CurrentStock | CurrentStockUnits |
| StorageCapacity | StorageCapacityUnits |

### Data Architecture Change
- UCI: Invoice-level transactions → cleaned → daily aggregation → product-level
- Pune: Already daily (Date, StoreID, ProductID) grain — **NO cleaning pipeline needed**
- Pune adds: Store dimension, Area/Locality, Suppliers (effective-dated), Calendar, Events, POs

### Risks
1. 3M-row daily sales CSV needs efficient loading strategy (chunked/cached Parquet)
2. Old core/__init__.py initialize_project() assumes UCI pipeline — needs Pune path
3. Old GBP field names in schemas.py are semantic, not just display — adapter bridges needed
4. Supplier effective dates add complexity to optimizer inputs
5. Store closures (S02, S12) need explicit handling in all views

### Backup
Pre-migration backup at: `stockwise_backup_20261020/`
