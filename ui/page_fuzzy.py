"""Page 4: Intelligent Stock Risk — Fuzzy logic risk assessment for store inventory.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.pune_data import (
    get_store_inventory,
    get_product_daily_series,
    get_valid_suppliers,
    is_store_active,
    list_products_for_store,
    load_stores,
)
from core.fuzzy_engine import evaluate_fuzzy_urgency, MEMBERSHIP_PARAMS
from core.forecast_service import ForecastService
from core.schemas import FuzzyResult
from config import PUNE_SNAPSHOT_DATE
from ui.components import (
    apply_plotly_theme,
    render_info_box,
    render_kpi_card,
    render_page_header,
    render_risk_badge,
)

_FORECAST_SERVICE = ForecastService()


def _compute_risk_for_product(row, store_id, horizon):
    """Compute fuzzy risk for a single product row. Uses fallback (rolling avg) instead of full forecast for speed."""
    pid = row["ProductID"]
    current_stock = int(row["CurrentStockUnits"])
    lead_time = int(row["LeadTimeDays"])
    pending = int(row.get("PendingOrderUnits", 0))

    # Use cached forecast if available, otherwise use fast fallback
    fc_key = f"fc_{store_id}_{pid}_{horizon}"
    fc_result = st.session_state.get(fc_key)

    if fc_result is not None:
        daily_forecast = fc_result.daily_predictions
        if lead_time > len(daily_forecast):
            daily_forecast = fc_result.model_parameters.get("extended_lead_time_predictions", daily_forecast)
        uncertainty = fc_result.uncertainty_score
    else:
        # Fast fallback: use rolling average from inventory data
        avg_daily = float(row.get("AverageDailySalesLast30Days", 1.0))
        daily_forecast = [max(0.0, avg_daily)] * max(horizon, lead_time)
        uncertainty = 0.5

    # Compute fuzzy risk
    effective_stock = current_stock + pending
    fuzzy_res = evaluate_fuzzy_urgency(
        current_stock=effective_stock,
        lead_time_days=lead_time,
        planning_horizon_days=horizon,
        lead_time_forecast=daily_forecast,
        uncertainty_score=uncertainty,
    )

    return {
        "ProductID": pid,
        "ProductName": str(row.get("ProductName", "")),
        "Category": str(row.get("Category", "")),
        "CurrentStock": current_stock,
        "PendingOrders": pending,
        "EffectiveStock": effective_stock,
        "LeadTimeDays": lead_time,
        "CoverageRatio": fuzzy_res.coverage_ratio,
        "CoverageDays": round(fuzzy_res.coverage_ratio * lead_time, 1) if lead_time > 0 else 999,
        "RiskScore": fuzzy_res.centroid,
        "RiskCategory": fuzzy_res.category,
        "FuzzyResult": fuzzy_res,
    }


def render_fuzzy_page():
    """Render the Stock Risk page using Mamdani Fuzzy Inference."""
    render_page_header(
        title="⚠️ Stock Risk Assessment",
        subtitle="See which products are at risk of running out — get clear reorder urgency levels.",
    )

    render_info_box(
        "How does this work?",
        "We analyze three factors for each product: <b>how much stock you have left</b>, "
        "<b>how long your supplier takes to deliver</b>, and <b>how predictable sales are</b>. "
        "These are combined using smart rules to give each product a risk score from 0 (safe) to 100 (urgent). "
        "Products with higher scores need reordering first."
    )

    store_id = st.session_state.get("selected_store_id")

    if not store_id:
        st.warning("👈 Select a store from the sidebar to view stock risk.")
        return

    if not is_store_active(store_id):
        st.error(f"🔴 Store {store_id} is closed. Stock risk is only available for active stores.")
        return

    # Get inventory snapshot
    inv_df = get_store_inventory(store_id)
    if inv_df.empty:
        st.warning(f"No inventory data found for store {store_id}.")
        return

    horizon = st.session_state.get("selected_forecast_horizon", 14)

    stores_df = load_stores()
    s_row = stores_df[stores_df["StoreID"] == store_id]
    store_name = s_row.iloc[0]["StoreName"] if not s_row.empty else store_id
    st.markdown(f"🏪 **{store_name}** ({store_id}) · {len(inv_df)} products · Planning: {horizon} days ahead")

    # Category filter
    categories = sorted(inv_df["Category"].unique()) if "Category" in inv_df.columns else []
    sel_cat = st.selectbox("Filter by Category", ["All Categories"] + categories, key="risk_cat_filter")
    if sel_cat != "All Categories":
        inv_df = inv_df[inv_df["Category"] == sel_cat]

    # Compute risk for all products (fast — uses fallback, no full forecasting)
    risk_results = []
    progress = st.progress(0, text="Analyzing product risk...")
    total = len(inv_df)
    for idx, (_, row) in enumerate(inv_df.iterrows()):
        risk_results.append(_compute_risk_for_product(row, store_id, horizon))
        if (idx + 1) % max(1, total // 10) == 0 or idx == total - 1:
            progress.progress((idx + 1) / total, text=f"Analyzed {idx + 1}/{total} products...")
    progress.empty()

    risk_df = pd.DataFrame(risk_results).sort_values("RiskScore", ascending=False)

    # Summary KPIs
    n_critical = len(risk_df[risk_df["RiskCategory"] == "Critical"])
    n_high = len(risk_df[risk_df["RiskCategory"] == "High"])
    n_moderate = len(risk_df[risk_df["RiskCategory"] == "Moderate"])
    n_low = len(risk_df[risk_df["RiskCategory"] == "Low"])

    st.markdown('<div class="section-title">📊 Risk Summary</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("CRITICAL", str(n_critical), "Reorder immediately", "🔴", "critical" if n_critical > 0 else "default")
    with k2:
        render_kpi_card("HIGH RISK", str(n_high), "Order soon", "🟠", "warning" if n_high > 0 else "default")
    with k3:
        render_kpi_card("MODERATE", str(n_moderate), "Monitor closely", "🟡", "default")
    with k4:
        render_kpi_card("LOW RISK", str(n_low), "Stock is sufficient", "🟢", "success")

    # Risk table
    st.markdown('<div class="section-title">📋 Product Risk Details</div>', unsafe_allow_html=True)
    st.caption("Products sorted by risk — highest risk at the top. Scroll down to see all products.")
    display_df = risk_df[["ProductID", "ProductName", "Category", "CurrentStock", "PendingOrders",
                           "LeadTimeDays", "RiskScore", "RiskCategory"]].copy()
    display_df["RiskScore"] = display_df["RiskScore"].round(1)
    st.dataframe(display_df, use_container_width=True, height=400)

    # Detailed view for selected product
    st.markdown('<div class="section-title">🔎 Adjust & Recalculate</div>', unsafe_allow_html=True)
    st.caption("Change stock or lead time to see how risk changes for a specific product.")

    product_id = st.session_state.get("selected_product_id")
    if product_id and product_id in risk_df["ProductID"].values:
        sel_risk = risk_df[risk_df["ProductID"] == product_id].iloc[0]
    elif not risk_df.empty:
        sel_risk = risk_df.iloc[0]
        product_id = sel_risk["ProductID"]
    else:
        st.info("No products to analyze.")
        return

    st.markdown(f"**{sel_risk['ProductID']}** — {sel_risk['ProductName']} ({sel_risk['Category']})")

    # What-if sliders
    w1, w2 = st.columns(2)
    with w1:
        override_stock = st.number_input(
            "📦 Stock Level",
            min_value=0,
            value=int(sel_risk["CurrentStock"]),
            step=10,
            key="risk_stock_override",
            help="Adjust the stock level to see how risk changes",
        )
    with w2:
        override_lead = st.number_input(
            "🚚 Lead Time (days)",
            min_value=0,
            value=int(sel_risk["LeadTimeDays"]),
            step=1,
            key="risk_lead_override",
            help="Adjust supplier delivery time to see impact on risk",
        )

    recalc = st.button("🔄 Recalculate Risk", key="risk_recalc_btn")

    fuzzy_res = sel_risk["FuzzyResult"]
    if recalc:
        fc_key = f"fc_{store_id}_{product_id}_{horizon}"
        fc_result = st.session_state.get(fc_key)
        if fc_result:
            daily_fc = fc_result.daily_predictions
            if override_lead > len(daily_fc):
                daily_fc = fc_result.model_parameters.get("extended_lead_time_predictions", daily_fc)
            unc = fc_result.uncertainty_score
        else:
            inv_row = inv_df[inv_df["ProductID"] == product_id]
            avg_daily = float(inv_row.iloc[0].get("AverageDailySalesLast30Days", 1.0)) if not inv_row.empty else 1.0
            daily_fc = [max(0.0, avg_daily)] * max(horizon, override_lead)
            unc = 0.5

        fuzzy_res = evaluate_fuzzy_urgency(
            current_stock=override_stock,
            lead_time_days=override_lead,
            planning_horizon_days=horizon,
            lead_time_forecast=daily_fc,
            uncertainty_score=unc,
        )
        st.success(f"Recalculated: Risk Score = **{fuzzy_res.centroid:.1f}** ({fuzzy_res.category})")

    # Display fuzzy results
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Stock Coverage", f"{fuzzy_res.coverage_ratio:.2f}")
    with r2:
        st.metric("Delivery Pressure", f"{fuzzy_res.lead_time_pressure:.2f}")
    with r3:
        st.metric("Forecast Confidence", f"{1 - fuzzy_res.uncertainty:.0%}")

    # Fired rules (academic mode only)
    if st.session_state.get("display_mode") == "Academic" and fuzzy_res.fired_rules:
        st.markdown("**🔧 Fired Fuzzy Rules:**")
        rules_df = pd.DataFrame(fuzzy_res.fired_rules)
        st.dataframe(rules_df, use_container_width=True)

    # Risk badge
    badge_html = render_risk_badge(fuzzy_res.category, fuzzy_res.centroid)
    st.markdown(badge_html, unsafe_allow_html=True)
