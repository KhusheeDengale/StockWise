"""Page 8: New Store Planner — Cold-start SKU and inventory recommendations for new Pune store locations.

Uses the 840-scenario new_store_reference table and comparable-store analysis to
generate opening inventory and assortment recommendations.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.pune_data import (
    get_new_store_reference,
    get_store_overview,
    get_category_revenue_breakdown,
    get_top_products,
    list_areas,
    list_stores,
    load_stores,
    load_products,
    load_current_inventory,
    is_store_active,
)
from config import PUNE_SNAPSHOT_DATE
from ui.components import (
    apply_plotly_theme,
    render_info_box,
    render_kpi_card,
    render_page_header,
)


def _find_comparable_stores(scenario: pd.Series) -> pd.DataFrame:
    """Find comparable existing stores based on scenario parameters."""
    stores = load_stores()
    active = stores[stores.apply(lambda r: is_store_active(r["StoreID"]), axis=1)].copy()

    if active.empty:
        return pd.DataFrame()

    # Score each store on similarity
    scores = []
    for _, s in active.iterrows():
        score = 0.0

        # Store type match
        if s["StoreType"] == scenario.get("ProposedStoreType", ""):
            score += 0.3

        # Customer segment match
        if s["PrimaryCustomerSegment"] == scenario.get("PrimaryCustomerSegment", ""):
            score += 0.2
        if s.get("SecondaryCustomerSegment") == scenario.get("SecondaryCustomerSegment", ""):
            score += 0.1

        # Footfall similarity (log scale)
        exp_footfall = float(scenario.get("ExpectedDailyFootfall", 100))
        store_footfall = float(s.get("EstimatedDailyFootfall", 100))
        if exp_footfall > 0 and store_footfall > 0:
            log_ratio = abs(np.log(exp_footfall) - np.log(store_footfall))
            score += max(0, 0.2 * (1 - log_ratio))

        # Size similarity
        prop_size = float(scenario.get("ProposedStoreSizeSqFt", 500))
        store_size = float(s.get("StoreSizeSqFt", 500))
        if prop_size > 0 and store_size > 0:
            size_ratio = min(prop_size, store_size) / max(prop_size, store_size)
            score += 0.2 * size_ratio

        scores.append({
            "StoreID": s["StoreID"],
            "StoreName": s["StoreName"],
            "StoreType": s["StoreType"],
            "LocalityName": s["LocalityName"],
            "SimilarityScore": round(score, 3),
        })

    result = pd.DataFrame(scores).sort_values("SimilarityScore", ascending=False)
    return result.head(5)  # Top 5 comparable stores


def _recommend_opening_inventory(
    comparable_stores: pd.DataFrame,
    budget_inr: float,
    storage_capacity: int,
    scenario: pd.Series,
) -> pd.DataFrame:
    """Generate opening inventory recommendations based on comparable stores."""
    products = load_products()
    all_inv = load_current_inventory()

    # Get average inventory across comparable stores
    comparable_ids = list(comparable_stores["StoreID"])
    comp_inv = all_inv[all_inv["StoreID"].isin(comparable_ids)]

    if comp_inv.empty:
        return pd.DataFrame()

    # Average across comparable stores
    avg_inv = comp_inv.groupby("ProductID").agg(
        AvgStock=("CurrentStockUnits", "mean"),
        AvgDailySales7=("AverageDailySalesLast7Days", "mean"),
        AvgDailySales30=("AverageDailySalesLast30Days", "mean"),
        AvgCost=("SupplierUnitCostINR", "mean"),
        AvgLeadTime=("LeadTimeDays", "mean"),
    ).reset_index()

    # Merge with product info
    avg_inv = avg_inv.merge(
        products[["ProductID", "ProductName", "Category", "Subcategory",
                   "MRP_INR", "BaseSellingPriceINR", "BasePurchaseCostINR",
                   "GrossMarginPercent", "StorageVolumeUnits"]],
        on="ProductID",
        how="left",
    )

    # Scoring: demand + margin + relevance
    demand_scale = float(scenario.get("EstimatedDemandScale", 1.0))
    avg_inv["EstimatedDailyDemand"] = avg_inv["AvgDailySales30"] * demand_scale
    avg_inv["CoverageDays"] = float(scenario.get("RecommendedCoverageDays", 30))

    # Recommended opening stock = estimated_daily × coverage_days
    avg_inv["RecommendedStock"] = (avg_inv["EstimatedDailyDemand"] * avg_inv["CoverageDays"]).clip(lower=1).round(0).astype(int)
    avg_inv["EstimatedCost"] = avg_inv["RecommendedStock"] * avg_inv["AvgCost"]
    avg_inv["EstimatedStorage"] = avg_inv["RecommendedStock"] * avg_inv.get("StorageVolumeUnits", 1).fillna(1)

    # Rank by demand × margin
    avg_inv["Score"] = avg_inv["EstimatedDailyDemand"] * avg_inv["GrossMarginPercent"].fillna(30) / 100

    # Sort by score descending
    avg_inv = avg_inv.sort_values("Score", ascending=False)

    # Budget and storage constrained selection
    cumulative_cost = 0.0
    cumulative_storage = 0
    selected = []

    for _, row in avg_inv.iterrows():
        cost = float(row["EstimatedCost"])
        storage = int(row.get("EstimatedStorage", row["RecommendedStock"]))

        if cumulative_cost + cost <= budget_inr and cumulative_storage + storage <= storage_capacity:
            cumulative_cost += cost
            cumulative_storage += storage
            selected.append(row)

    return pd.DataFrame(selected).reset_index(drop=True)


def render_new_store_page():
    """Render the New Store Planner page."""
    render_page_header(
        title="🏗️ New Store Planner",
        subtitle="Plan a new stationery store in Pune — cold-start inventory and assortment recommendations.",
    )

    render_info_box(
        "What is this page?",
        "Planning a new store? This tool helps you decide <b>what products to stock</b> and "
        "<b>how much to order</b> before you have any sales history. It uses the "
        "<b>840 pre-computed scenarios</b> from the reference table and compares your planned store "
        "to existing similar stores to generate smart recommendations."
    )

    # Scenario selection
    st.markdown('<div class="section-title">1️⃣ Select Your Store Scenario</div>', unsafe_allow_html=True)

    areas_df = list_areas()
    area_options = [f"{r['AreaID']} — {r['LocalityName']}" for _, r in areas_df.iterrows()]
    sel_area = st.selectbox("📍 Select Area", area_options, key="ns_area")
    area_id = sel_area.split(" — ")[0]

    ref_df = get_new_store_reference(area_id=area_id)
    if ref_df.empty:
        st.warning(f"No scenarios found for area {area_id}.")
        return

    # Filter by store type if available
    store_types = sorted(ref_df["ProposedStoreType"].unique())
    sel_type = st.selectbox("🏪 Store Type", store_types, key="ns_type")
    ref_filtered = ref_df[ref_df["ProposedStoreType"] == sel_type]

    if ref_filtered.empty:
        st.warning("No scenarios match these filters.")
        return

    # Select scenario
    scenario_ids = ref_filtered["ScenarioID"].tolist()
    sel_scenario = st.selectbox(
        "📋 Select Scenario",
        scenario_ids,
        format_func=lambda s: f"{s} — Budget ₹{ref_filtered[ref_filtered['ScenarioID']==s].iloc[0]['StartingInventoryBudgetINR']:,.0f}",
        key="ns_scenario",
    )
    scenario = ref_filtered[ref_filtered["ScenarioID"] == sel_scenario].iloc[0]

    # Scenario details
    st.markdown('<div class="section-title">2️⃣ Scenario Details</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        render_kpi_card("AREA", f"{scenario['LocalityName']}", f"AreaID: {scenario['AreaID']}", "📍", "default")
    with d2:
        render_kpi_card("STORE SIZE", f"{int(scenario['ProposedStoreSizeSqFt']):,} sq ft",
                        f"Storage: {int(scenario['ProposedStorageCapacityUnits']):,} units", "🏪", "default")
    with d3:
        render_kpi_card("INVENTORY BUDGET", f"₹{scenario['StartingInventoryBudgetINR']:,.0f}",
                        f"Monthly: ₹{scenario['MonthlyPurchaseBudgetINR']:,.0f}", "💰", "accent")
    with d4:
        render_kpi_card("EXP. FOOTFALL", f"{int(scenario['ExpectedDailyFootfall'])} / day",
                        f"Demand Scale: {scenario['EstimatedDemandScale']:.2f}", "👥", "default")

    st.markdown("**Customer Profile:**")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f"🎯 Primary: **{scenario['PrimaryCustomerSegment']}**")
        st.markdown(f"🎯 Secondary: **{scenario['SecondaryCustomerSegment']}**")
    with p2:
        st.markdown(f"📊 Price Sensitivity: **{scenario['PriceSensitivityIndex']:.2f}**")
        st.markdown(f"🏢 Competition: **{scenario['CompetitionIntensityIndex']:.2f}**")
    with p3:
        st.markdown(f"📦 Preferred Category: **{scenario.get('PreferredCategory', 'All')}**")
        st.markdown(f"📅 Coverage Target: **{scenario.get('RecommendedCoverageDays', 30)} days**")

    # Comparable stores
    st.markdown('<div class="section-title">3️⃣ Comparable Existing Stores</div>', unsafe_allow_html=True)

    comp_stores = _find_comparable_stores(scenario)
    if not comp_stores.empty:
        st.dataframe(comp_stores, use_container_width=True)

        # Performance of comparable stores
        st.markdown("**📊 Comparable Store Performance (2025):**")
        perf_rows = []
        for _, cs in comp_stores.iterrows():
            sid = cs["StoreID"]
            ov = get_store_overview(store_id=sid, start_date="2025-01-01", end_date="2025-12-31")
            perf_rows.append({
                "Store": f"{sid} — {cs['StoreName']}",
                "Revenue (₹)": f"{ov['net_revenue_inr']:,.0f}",
                "Units Sold": f"{ov['units_sold']:,}",
                "Margin %": f"{ov['gross_margin_pct']:.1f}%",
                "Similarity": f"{cs['SimilarityScore']:.2f}",
            })
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True)
    else:
        st.warning("No comparable stores found.")

    # Generate recommendations
    st.markdown('<div class="section-title">4️⃣ Opening Inventory Recommendations</div>', unsafe_allow_html=True)

    if st.button("🚀 Generate Recommendations", type="primary", key="ns_gen_btn"):
        with st.spinner("Analyzing comparable stores and generating recommendations..."):
            budget = float(scenario["StartingInventoryBudgetINR"])
            storage = int(scenario["ProposedStorageCapacityUnits"])

            rec_df = _recommend_opening_inventory(comp_stores, budget, storage, scenario)

            if rec_df.empty:
                st.warning("Could not generate recommendations. Try a different scenario.")
                return

            st.session_state["ns_recommendations"] = rec_df
            st.success(f"✅ Recommended {len(rec_df)} products for opening inventory!")

    rec_df = st.session_state.get("ns_recommendations")
    if rec_df is not None and not rec_df.empty:
        # Summary
        total_cost = rec_df["EstimatedCost"].sum()
        total_skus = len(rec_df)
        budget = float(scenario["StartingInventoryBudgetINR"])

        s1, s2, s3 = st.columns(3)
        with s1:
            render_kpi_card("TOTAL SKUs", str(total_skus), "Products to stock", "📦", "accent")
        with s2:
            render_kpi_card("TOTAL COST", f"₹{total_cost:,.0f}",
                            f"{total_cost/budget*100:.1f}% of budget", "💰", "success")
        with s3:
            render_kpi_card("BUDGET REMAINING", f"₹{budget - total_cost:,.0f}",
                            "Available for additional orders", "💚", "default")

        # Category breakdown
        if "Category" in rec_df.columns:
            cat_summary = rec_df.groupby("Category").agg(
                SKUs=("ProductID", "count"),
                TotalCost=("EstimatedCost", "sum"),
            ).reset_index()
            fig_cat = px.pie(cat_summary, values="TotalCost", names="Category", hole=0.4,
                              title="Investment by Category")
            fig_cat.update_layout(height=300)
            st.plotly_chart(fig_cat, use_container_width=True)

        # Recommendations table
        display_cols = ["ProductID", "ProductName", "Category", "RecommendedStock",
                        "EstimatedDailyDemand", "EstimatedCost", "AvgCost", "GrossMarginPercent"]
        available = [c for c in display_cols if c in rec_df.columns]
        st.dataframe(rec_df[available], use_container_width=True)

        # Download
        st.download_button(
            "📥 Download Opening Inventory Plan (CSV)",
            data=rec_df[available].to_csv(index=False).encode("utf-8"),
            file_name=f"opening_inventory_{area_id}_{sel_type}.csv",
            mime="text/csv",
        )
