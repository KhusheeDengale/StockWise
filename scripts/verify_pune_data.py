"""Verify Pune dataset integrity and compute aggregate totals."""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stockwise_pune")

# 1. Stores - closed stores
stores = pd.read_csv(os.path.join(DATA_DIR, "pune_stores.csv"))
closed = stores[stores["StoreClosingDate"].notna()]
print(f"Total stores: {len(stores)}")
print(f"Closed stores: {len(closed)}")
for _, r in closed.iterrows():
    print(f"  {r['StoreID']} ({r['StoreName']}) closed {r['StoreClosingDate']}")
print(f"Active at 2025-12-31: {len(stores) - len(closed)}")

# 2. Inventory snapshot
inv = pd.read_csv(os.path.join(DATA_DIR, "pune_current_inventory.csv"))
print(f"\nInventory snapshot rows: {len(inv)}")
print(f"Unique stores in inventory: {inv['StoreID'].nunique()}")
print(f"Snapshot dates: {inv['SnapshotDate'].unique()}")

# 3. Aggregate sales totals (chunked reading for memory)
print("\nComputing aggregate sales totals (chunked)...")
total_units = 0
total_revenue = 0.0
total_rows = 0
for chunk in pd.read_csv(os.path.join(DATA_DIR, "pune_daily_sales.csv"), chunksize=500000,
                          usecols=["UnitsSold", "NetRevenueINR"]):
    total_units += chunk["UnitsSold"].sum()
    total_revenue += chunk["NetRevenueINR"].sum()
    total_rows += len(chunk)

print(f"Total rows: {total_rows:,}")
print(f"Total UnitsSold: {total_units:,}")
print(f"Total NetRevenueINR: {total_revenue:,.2f}")
print(f"Expected UnitsSold: 8,591,644")
print(f"Expected NetRevenueINR: 424,655,085.69")
print(f"Units match: {total_units == 8591644}")
print(f"Revenue match (0.01 tol): {abs(total_revenue - 424655085.69) < 0.01}")

# 4. New store reference
ref = pd.read_csv(os.path.join(DATA_DIR, "pune_new_store_reference.csv"))
print(f"\nNew store reference rows: {len(ref)}")
print(f"Scenarios per area: {ref.groupby('AreaID').size().to_dict()}")
