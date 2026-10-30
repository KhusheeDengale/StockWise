"""Inventory configuration management, validation, and demo data generator.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from config import SAMPLE_INVENTORY_CSV
from core.schemas import InventoryItem, SchemaValidationError

INVENTORY_COLUMNS = [
    "StockCode",
    "Description",
    "CurrentStock",
    "SupplierCostGBP",
    "LeadTimeDays",
    "StorageCapacity",
    "MinimumOrderQuantity",
    "OrderMultiple",
]


def validate_inventory_config(data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> bool:
    """Validate inventory configuration records against strict business rules.
    
    Rules:
    - All required columns present
    - CurrentStock >= 0
    - SupplierCostGBP > 0
    - LeadTimeDays >= 0
    - StorageCapacity >= CurrentStock
    - MinimumOrderQuantity >= 0
    - OrderMultiple >= 1
    
    Args:
        data: DataFrame or list of dicts.
        
    Returns:
        True if all records pass validation.
        
    Raises:
        SchemaValidationError: If any rule is violated.
    """
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    missing = [c for c in INVENTORY_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing inventory columns: {missing}")

    errors = []
    for idx, row in df.iterrows():
        code = row["StockCode"]
        if row["CurrentStock"] < 0:
            errors.append(f"Row {idx} ({code}): CurrentStock must be >= 0 (got {row['CurrentStock']})")
        if row["SupplierCostGBP"] <= 0:
            errors.append(f"Row {idx} ({code}): SupplierCostGBP must be > 0 (got {row['SupplierCostGBP']})")
        if row["LeadTimeDays"] < 0:
            errors.append(f"Row {idx} ({code}): LeadTimeDays must be >= 0 (got {row['LeadTimeDays']})")
        if row["StorageCapacity"] < row["CurrentStock"]:
            errors.append(
                f"Row {idx} ({code}): StorageCapacity ({row['StorageCapacity']}) "
                f"cannot be less than CurrentStock ({row['CurrentStock']})"
            )
        if row["MinimumOrderQuantity"] < 0:
            errors.append(f"Row {idx} ({code}): MinimumOrderQuantity must be >= 0")
        if row["OrderMultiple"] < 1:
            errors.append(f"Row {idx} ({code}): OrderMultiple must be >= 1 (got {row['OrderMultiple']})")

    if errors:
        raise SchemaValidationError("Inventory validation failed:\n" + "\n".join(errors))
    return True


def load_inventory_config(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and validate the inventory configuration file.
    
    Args:
        filepath: Optional path to sample_inventory.csv.
        
    Returns:
        Validated DataFrame.
    """
    path = Path(filepath) if filepath else SAMPLE_INVENTORY_CSV
    if not path.exists():
        raise FileNotFoundError(f"Inventory configuration file not found at: {path}")

    df = pd.read_csv(path, dtype={"StockCode": str})
    validate_inventory_config(df)
    return df


def save_inventory_config(
    config_df: pd.DataFrame, filepath: Optional[Union[str, Path]] = None
) -> None:
    """Validate and persist the inventory configuration file.
    
    Args:
        config_df: Inventory dataframe.
        filepath: Destination file path.
    """
    validate_inventory_config(config_df)
    path = Path(filepath) if filepath else SAMPLE_INVENTORY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    config_df.to_csv(path, index=False)


def reset_demo_inventory(
    demo_products_df: pd.DataFrame,
    daily_demand_df: pd.DataFrame,
    filepath: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate simulated realistic inventory configuration for the demo products.
    
    Method:
    - seed = 42
    - For each product:
        - Calculate average daily demand over the last 60 days of its observed history
        - Generate simulated stock coverage duration between 3 and 21 days
        - CurrentStock = rounded(avg_demand * coverage_duration)
        - Historical median selling price defines scale
        - SupplierCostGBP between 40% and 75% of median selling price
        - LeadTimeDays between 2 and 14 integer days
        - StorageCapacity >= CurrentStock + ~30 days average demand + practical margin
        - MOQ = 0, OrderMultiple = 1 for initial demo
    
    Args:
        demo_products_df: Catalog subset of demo products.
        daily_demand_df: Complete daily demand dataframe.
        filepath: Path to save generated CSV.
        seed: Random seed for reproducibility.
        
    Returns:
        Generated and saved inventory DataFrame.
    """
    rng = np.random.RandomState(seed)
    records = []

    for _, row in demo_products_df.iterrows():
        code = str(row["StockCode"])
        desc = str(row["Description"])
        
        # Product daily demand over last 60 calendar days
        prod_daily = daily_demand_df[daily_demand_df["StockCode"] == code].sort_values("Date")
        last_60 = prod_daily.tail(60)
        avg_demand_60 = float(last_60["DailyUnits"].mean()) if not last_60.empty else 1.0
        avg_demand_60 = max(avg_demand_60, 0.2)

        # Coverage duration between 3 and 21 days
        coverage_days = rng.uniform(3.0, 21.0)
        current_stock = int(round(avg_demand_60 * coverage_days))
        current_stock = max(current_stock, 1)

        # Historical median price
        median_price = float(row.get("HistoricalMedianPriceGBP", 2.50))
        if median_price <= 0:
            median_price = 2.50
            
        # Supplier cost between 40% and 75% of median selling price
        cost_ratio = rng.uniform(0.40, 0.75)
        supplier_cost = round(max(median_price * cost_ratio, 0.20), 2)

        # Lead time between 2 and 14 days
        lead_time = int(rng.randint(2, 15))

        # Storage capacity: current stock + ~30 days demand + safety margin
        margin = int(max(round(10 * avg_demand_60), 25))
        storage_capacity = int(current_stock + round(30 * avg_demand_60) + margin)

        records.append({
            "StockCode": code,
            "Description": desc,
            "CurrentStock": current_stock,
            "SupplierCostGBP": supplier_cost,
            "LeadTimeDays": lead_time,
            "StorageCapacity": storage_capacity,
            "MinimumOrderQuantity": 0,
            "OrderMultiple": 1,
        })

    config_df = pd.DataFrame(records)[INVENTORY_COLUMNS]
    validate_inventory_config(config_df)
    save_inventory_config(config_df, filepath=filepath)
    return config_df
