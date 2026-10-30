"""Page 7: Reports — Consolidated purchase plans, store summaries, and export.
"""
import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.pune_data import (
    get_store_inventory,
    get_store_overview,
    is_store_active,
    list_stores,
    load_stores,
)
from config import PUNE_SNAPSHOT_DATE
from ui.components import (
    render_info_box,
    render_kpi_card,
    render_page_header,
)


def render_reports_page():
    """Render the Reports page for purchase plan summaries and exports."""
    render_page_header(
        title="📋 Reports & Purchase Plan",
        subtitle="View consolidated purchasing recommendations and download detailed reports.",
    )

    render_info_box(
        "What is this page?",
        "This page consolidates all your analysis into an <b>actionable purchase plan</b>. "
        "It combines forecast results, risk assessments, and optimization recommendations "
        "into a downloadable report you can share with your purchasing team."
    )

    store_id = st.session_state.get("selected_store_id")

    tab_store, tab_inv, tab_export = st.tabs([
        "📊 Store Summary",
        "📦 Inventory Report",
        "📥 Export",
    ])

    with tab_store:
        if not store_id:
            st.info("👈 Select a store from the sidebar to see the store summary report.")
            # All-stores summary
            st.markdown('<div class="section-title">🏪 All Stores Summary</div>', unsafe_allow_html=True)
            active_stores = list_stores(active_at=PUNE_SNAPSHOT_DATE)
            rows = []
            for _, sr in active_stores.iterrows():
                sid = sr["StoreID"]
                ov = get_store_overview(store_id=sid, start_date="2025-01-01", end_date="2025-12-31")
                inv = get_store_inventory(sid)
                rows.append({
                    "Store": f"{sid} — {sr['StoreName']}",
                    "Locality": sr["LocalityName"],
                    "Type": sr["StoreType"],
                    "2025 Revenue (₹)": f"{ov['net_revenue_inr']:,.0f}",
                    "Units Sold": f"{ov['units_sold']:,}",
                    "Margin %": f"{ov['gross_margin_pct']:.1f}%",
                    "SKUs in Inventory": len(inv),
                    "Stockout Events": ov['stockout_days'],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            stores_df = load_stores()
            s_row = stores_df[stores_df["StoreID"] == store_id]
            if not s_row.empty:
                s = s_row.iloc[0]
                st.markdown(f"### 🏪 {s['StoreName']} ({store_id})")
                st.markdown(f"📍 {s['LocalityName']} · **{s['StoreType']}** · "
                            f"Opened: {str(s['StoreOpeningDate'].date())}")

            # Current year KPIs
            ov = get_store_overview(store_id=store_id, start_date="2025-01-01", end_date="2025-12-31")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card("2025 REVENUE", f"₹{ov['net_revenue_inr']:,.0f}", "Full year", "💰", "accent")
            with k2:
                render_kpi_card("UNITS SOLD", f"{ov['units_sold']:,}", "Full year", "📦", "default")
            with k3:
                render_kpi_card("GROSS MARGIN", f"{ov['gross_margin_pct']:.1f}%", "Profit ratio", "💚", "success")
            with k4:
                render_kpi_card("STOCKOUT EVENTS", f"{ov['stockout_days']:,}", "Lost sales days", "⚠️", "warning")

            # Optimization results (if available)
            opt = st.session_state.get("opt_result")
            if opt:
                st.markdown("### 🛒 Latest Optimization Results")
                final_res = opt["final_res"]
                products_data = opt["products_data"]
                best_chrom = opt["best_chrom"]

                st.markdown(f"**Budget:** ₹{opt['budget']:,.0f} · "
                            f"**Spend:** ₹{final_res.metrics.TotalPurchaseCost:,.0f} · "
                            f"**Utilization:** {final_res.metrics.BudgetUtilizationPercent:.1f}%")

                rec_rows = []
                for i, p in enumerate(products_data):
                    qty = best_chrom[i]
                    if qty > 0:
                        rec_rows.append({
                            "Product": f"{p['StockCode']} — {p['Description'][:25]}",
                            "Order Qty": qty,
                            "Unit Cost (₹)": round(p["SupplierCostGBP"], 2),
                            "Total Cost (₹)": round(qty * p["SupplierCostGBP"], 2),
                            "Current Stock": p["CurrentStock"],
                            "Forecast Demand": round(sum(p["DailyForecast"]), 0),
                        })
                if rec_rows:
                    st.dataframe(pd.DataFrame(rec_rows), use_container_width=True)
                else:
                    st.info("No orders recommended for the current budget and risk level.")
            else:
                st.info("💡 Run the **Reorder Optimizer** first to see purchase recommendations here.")

    with tab_inv:
        if not store_id:
            st.info("👈 Select a store to see inventory details.")
            return

        inv_df = get_store_inventory(store_id)
        if inv_df.empty:
            st.warning(f"No inventory data for {store_id}.")
            return

        st.markdown(f"### 📦 Inventory Snapshot — {store_id} (as of {PUNE_SNAPSHOT_DATE})")
        st.markdown(f"**{len(inv_df)} products** in inventory")

        # Summary stats
        avg_coverage = inv_df["StockCoverageDays"].mean() if "StockCoverageDays" in inv_df.columns else 0
        below_rop = len(inv_df[inv_df["CurrentStockUnits"] <= inv_df["ReorderPointUnits"]]) if "ReorderPointUnits" in inv_df.columns else 0
        below_safety = len(inv_df[inv_df["CurrentStockUnits"] <= inv_df["SafetyStockUnits"]]) if "SafetyStockUnits" in inv_df.columns else 0

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Avg Coverage Days", f"{avg_coverage:.1f}")
        with k2:
            st.metric("Below Reorder Point", f"{below_rop}")
        with k3:
            st.metric("Below Safety Stock", f"{below_safety}")

        display_cols = ["ProductID", "ProductName", "Category", "CurrentStockUnits",
                        "PendingOrderUnits", "SafetyStockUnits", "ReorderPointUnits",
                        "StockCoverageDays", "SupplierUnitCostINR", "LeadTimeDays"]
        available = [c for c in display_cols if c in inv_df.columns]
        st.dataframe(inv_df[available], use_container_width=True)

    with tab_export:
        st.markdown("### 📥 Download Reports")

        if store_id:
            # Inventory CSV
            inv_df = get_store_inventory(store_id)
            if not inv_df.empty:
                st.download_button(
                    "📥 Download Inventory Snapshot (CSV)",
                    data=inv_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"inventory_{store_id}_{PUNE_SNAPSHOT_DATE}.csv",
                    mime="text/csv",
                )

            # Purchase plan
            opt = st.session_state.get("opt_result")
            if opt:
                rec_rows = []
                for i, p in enumerate(opt["products_data"]):
                    qty = opt["best_chrom"][i]
                    rec_rows.append({
                        "StoreID": store_id,
                        "ProductID": p["StockCode"],
                        "Description": p["Description"],
                        "CurrentStock": p["CurrentStock"],
                        "ForecastDemand": round(sum(p["DailyForecast"]), 2),
                        "OrderQuantity": qty,
                        "UnitCostINR": round(p["SupplierCostGBP"], 2),
                        "TotalCostINR": round(qty * p["SupplierCostGBP"], 2),
                        "LeadTimeDays": p["LeadTimeDays"],
                    })
                rec_df = pd.DataFrame(rec_rows)
                st.download_button(
                    "📥 Download Purchase Plan (CSV)",
                    data=rec_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"purchase_plan_{store_id}.csv",
                    mime="text/csv",
                )
            else:
                st.info("Run the Reorder Optimizer first to generate a purchase plan.")
        else:
            st.info("👈 Select a store to enable report downloads.")
