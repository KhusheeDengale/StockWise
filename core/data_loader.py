"""Data loading and schema normalization for UCI Online Retail II dataset.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from config import (
    CANONICAL_COLUMNS,
    COLUMN_MAPPING,
    RAW_RETAIL_EXCEL,
)
from core.schemas import SchemaValidationError


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw dataframe column names to canonical schema.
    
    Args:
        df: Input DataFrame with raw column names.
        
    Returns:
        DataFrame with canonical column names.
        
    Raises:
        SchemaValidationError: If any canonical column cannot be found or mapped.
    """
    df = df.copy()
    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    
    # Rename mapped columns if present
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Validate canonical columns
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required canonical columns: {missing}. "
            f"Present columns: {list(df.columns)}"
        )
    return df


def load_raw_dataset(
    filepath: Optional[Union[str, Path]] = None,
    sheet_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load and concatenate both worksheets of the Online Retail II dataset.
    
    Args:
        filepath: Path to the online_retail_II.xlsx file. Defaults to config path.
        sheet_names: List of sheet names to load. Defaults to ['Year 2009-2010', 'Year 2010-2011'].
        
    Returns:
        Tuple of (concatenated raw DataFrame with SourceSheet column, dictionary of sheet row counts).
        
    Raises:
        FileNotFoundError: If Excel file does not exist.
        SchemaValidationError: If sheets cannot be normalized or required columns are missing.
    """
    target_path = Path(filepath) if filepath else RAW_RETAIL_EXCEL
    if not target_path.exists():
        raise FileNotFoundError(f"Raw retail dataset not found at: {target_path}")

    sheets_to_load = sheet_names or ["Year 2009-2010", "Year 2010-2011"]
    sheet_counts: Dict[str, int] = {}
    dfs = []

    for sheet in sheets_to_load:
        sheet_df = pd.read_excel(
            target_path,
            sheet_name=sheet,
            dtype={"Invoice": str, "StockCode": str, "InvoiceNo": str},
        )
        sheet_counts[sheet] = len(sheet_df)
        sheet_df = normalize_column_names(sheet_df)
        sheet_df["SourceSheet"] = sheet
        dfs.append(sheet_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df, sheet_counts
