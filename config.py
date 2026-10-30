"""StockWise Global Configuration and Constants.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIG_DATA_DIR = DATA_DIR / "config"
MODELS_DIR = BASE_DIR / "models"

# Pune Dataset Paths
PUNE_DATA_DIR = DATA_DIR / "stockwise_pune"
PUNE_CACHE_DIR = DATA_DIR / "pune_cache"
PUNE_SNAPSHOT_DATE = "2025-12-31"

# Legacy UCI Dataset Files (retained for backward compatibility)
RAW_RETAIL_EXCEL = RAW_DATA_DIR / "online_retail_II.xlsx"
CLEANED_TRANSACTIONS_PARQUET = PROCESSED_DATA_DIR / "cleaned_transactions.parquet"
EXCLUDED_TRANSACTIONS_PARQUET = PROCESSED_DATA_DIR / "excluded_transactions.parquet"
DAILY_DEMAND_PARQUET = PROCESSED_DATA_DIR / "daily_demand.parquet"
PRODUCT_CATALOG_CSV = PROCESSED_DATA_DIR / "product_catalog.csv"
DEMO_PRODUCTS_CSV = PROCESSED_DATA_DIR / "demo_products.csv"
CLEANING_AUDIT_JSON = PROCESSED_DATA_DIR / "cleaning_audit.json"
SAMPLE_INVENTORY_CSV = CONFIG_DATA_DIR / "sample_inventory.csv"

# Preprocessing Constants
COLUMN_MAPPING = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
}

CANONICAL_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

# Non-merchandise StockCodes and description patterns
NON_PRODUCT_STOCKCODES = {
    "POST",
    "D",
    "M",
    "PADS",
    "DOT",
    "CRUK",
    "BANK CHARGES",
    "TEST",
    "ADJUST",
    "ADJUST2",
    "AMAZONFEE",
    "S",
}

NON_PRODUCT_DESCRIPTIONS = [
    "POSTAGE",
    "DISCOUNT",
    "MANUAL",
    "ADJUST BAD DEBT",
    "BANK CHARGES",
    "DOTCOM POSTAGE",
    "CRUK COMMISSION",
    "AMAZON FEE",
    "SAMPLES",
]

# Demo Selection Criteria
DEMO_CRITERIA = {
    "min_calendar_days": 240,
    "min_positive_demand_days": 60,
    "min_recent_positive_demand_days": 30,
    "recent_window_days": 120,
    "max_days_before_dataset_end": 30,
    "target_demo_count": 30,
}

# Forecasting Settings
HORIZONS = [7, 14, 30]
DEFAULT_HORIZON = 14
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Optimization Settings
GA_CONFIG = {
    "population_size": 40,
    "generations": 80,
    "tournament_size": 3,
    "crossover_probability": 0.8,
    "mutation_probability": 0.1,
    "elite_count": 2,
    "random_seed": 42,
    "early_stopping_patience": 20,
}

OBJECTIVE_WEIGHTS = {
    "shortage": 0.55,
    "excess": 0.20,
    "cost": 0.15,
    "coverage": 0.10,
}

# New Store Planner — comparable store similarity weights
NEW_STORE_SIMILARITY_WEIGHTS = {
    "store_type_segments": 0.25,
    "proximity_profile": 0.25,
    "log_footfall": 0.20,
    "size": 0.15,
    "area_customer_mix": 0.15,
}

# New Store Planner — SKU ranking weights
NEW_STORE_SKU_RANKING_WEIGHTS = {
    "demand": 0.40,
    "relevance": 0.25,
    "margin": 0.20,
    "storage_efficiency": 0.15,
}
