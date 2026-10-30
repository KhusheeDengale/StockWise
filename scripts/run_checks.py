"""Automated system diagnostics and health checks script for StockWise backend.
"""
import sys
import subprocess
from pathlib import Path

# Add stockwise root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CLEANED_TRANSACTIONS_PARQUET,
    CLEANING_AUDIT_JSON,
    DAILY_DEMAND_PARQUET,
    DEMO_PRODUCTS_CSV,
    PRODUCT_CATALOG_CSV,
    SAMPLE_INVENTORY_CSV,
)
from core.inventory import load_inventory_config, validate_inventory_config


def main():
    print("=" * 70)
    print(" STOCKWISE: SYSTEM DIAGNOSTICS & VERIFICATION SUITE")
    print("=" * 70)

    # 1. Check Python and libraries
    print("\n[Check 1/4] Checking Python Environment & Libraries...")
    required_packages = [
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "statsmodels",
        "skfuzzy",
        "plotly",
        "matplotlib",
        "openpyxl",
        "pyarrow",
        "joblib",
        "pytest",
        "streamlit",
    ]
    all_pkgs_ok = True
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}")
        except ImportError as e:
            print(f"  [MISSING] {pkg}: {e}")
            all_pkgs_ok = False

    assert all_pkgs_ok, "Missing required libraries!"

    # 2. Check Data Caches
    print("\n[Check 2/4] Verifying Processed Data Caches...")
    files_to_check = [
        CLEANED_TRANSACTIONS_PARQUET,
        DAILY_DEMAND_PARQUET,
        PRODUCT_CATALOG_CSV,
        DEMO_PRODUCTS_CSV,
        CLEANING_AUDIT_JSON,
        SAMPLE_INVENTORY_CSV,
    ]
    for f in files_to_check:
        if f.exists():
            print(f"  [OK] {f.name} ({f.stat().st_size:,} bytes)")
        else:
            print(f"  [MISSING] {f.name}")
            raise FileNotFoundError(f"Missing required cached file: {f}")

    # 3. Validate Inventory Configuration
    print("\n[Check 3/4] Validating Inventory Configuration Schema...")
    inv_df = load_inventory_config()
    validate_inventory_config(inv_df)
    print(f"  [OK] Successfully validated {len(inv_df)} items in sample_inventory.csv.")

    # 4. Run Pytest Suite
    print("\n[Check 4/4] Executing Automated Pytest Suite...")
    res = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=str(Path(__file__).parent.parent))
    if res.returncode == 0:
        print("\n" + "=" * 70)
        print(" ALL CHECKS PASSED: STOCKWISE ENGINE IS 100% OPERATIONAL")
        print("=" * 70)
    else:
        print("\n[FAIL] Pytest tests encountered failures.")
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
