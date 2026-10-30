"""One-time offline data ingestion, cleaning, daily aggregation, and Parquet caching script.
"""
import sys
import time
from pathlib import Path

# Add stockwise to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.inventory import reset_demo_inventory
from core.preprocessing import prepare_dataset


def main():
    print("=" * 70)
    print(" STOCKWISE: OFFLINE DATA PREPARATION AND CACHING")
    print("=" * 70)
    
    start_time = time.time()
    print("\n1. Starting data loading and cleaning sequence...")
    cleaned_df, excluded_df, daily_df, catalog_df, audit = prepare_dataset()
    
    elapsed = time.time() - start_time
    print(f"Data cleaning and aggregation completed in {elapsed:.2f} seconds.")
    print("\n--- Cleaning Audit Summary ---")
    for k, v in audit.items():
        print(f"  {k}: {v}")

    print("\n2. Initializing simulated demo inventory configuration...")
    demo_df = catalog_df.head(30)
    inv_df = reset_demo_inventory(demo_df, daily_df)
    print(f"Sample inventory configuration saved with {len(inv_df)} items.")
    
    print("\nData preparation successful! Processed files are cached in 'data/processed/'.")


if __name__ == "__main__":
    main()
