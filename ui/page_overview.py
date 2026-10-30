"""Page 1: Business Overview — Store-level KPIs, trends, category breakdown, and top products.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.pune_data import (
    get_store_overview,
    get_category_revenue_breakdown,
    get_top_products,
    get_daily_trend,
    get_low_stock_summary,
    list_stores,
    load_stores,
    is_store_active,
)
from config import PUNE_SNAPSHOT_DATE
from ui.components import (
    apply_plotly_theme,
    render_info_box,
    render_kpi_card,
    render_page_header,
)


def render_overview_page():
    """Render the Business Overview page for Pune stationery data."""
    render_page_header(
        title="📊 Business Overview",
        subtitle="See how your store is performing — revenue, units, profit, and top products at a glance.",
    )

    render_info_box(
        "What is this page?",
        "This page gives you a <b>bird's-eye view of store sales performance</b>. "
        "You can see total revenue, units sold, gross profit, top-selling products, and how sales change over time. "
        "Use the sidebar filters to zoom into specific stores, areas, or date ranges."
    )

    # Get filter state from sidebar
    store_id = st.session_state.get("selected_store_id")
    area_id = st.session_state.get("selected_area_id")
    date_range = st.session_state.get("selected_date_range", ("2024-01-01", "2025-12-31"))
    start_date, end_date = date_range

    # Show scope banner
    scope_parts = []
    if store_id:
        stores_df = load_stores()
        s_row = stores_df[stores_df["StoreID"] == store_id]
        if not s_row.empty:
            s = s_row.iloc[0]
            scope_parts.append(f"🏪 **{s['StoreName']}** ({store_id}) · {s['LocalityName']}")
    elif area_id:
        scope_parts.append(f"📍 Area: **{area_id}**")
    else:
        scope_parts.append("🌐 **All Stores**")
    scope_parts.append(f"📅 {start_date} to {end_date}")
    st.markdown(" · ".join(scope_parts))

    # KPIs
    overview = get_store_overview(
        store_id=store_id,
        area_id=area_id,
        start_date=start_date,
        end_date=end_date,
    )

    st.markdown('<div class="section-title">📈 Key Numbers</div>', unsafe_allow_html=True)
    st.caption("These are the most important business metrics for the selected filters.")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card(
            title="NET REVENUE",
            value=f"₹{overview['net_revenue_inr']:,.2f}",
            caption="Total net sales value",
            icon="💰",
            color_type="accent",
        )
    with k2:
        render_kpi_card(
            title="UNITS SOLD",
            value=f"{overview['units_sold']:,}",
            caption="Total items sold to customers",
            icon="📦",
            color_type="default",
        )
    with k3:
        render_kpi_card(
            title="TRANSACTIONS",
            value=f"{overview['transaction_count']:,}",
            caption="Sum of daily transaction counts",
            icon="🧾",
            color_type="default",
        )
    with k4:
        render_kpi_card(
            title="GROSS PROFIT",
            value=f"₹{overview['gross_profit_inr']:,.2f}",
            caption=f"Gross Margin: {overview['gross_margin_pct']:.1f}%",
            icon="💚",
            color_type="success",
        )
    with k5:
        render_kpi_card(
            title="AVG TRANSACTION VALUE",
            value=f"₹{overview['avg_transaction_value']:,.2f}",
            caption="Net Revenue ÷ Transaction Count",
            icon="📊",
            color_type="teal",
        )

    # Additional operational KPIs
    k6, k7, k8 = st.columns(3)
    with k6:
        render_kpi_card(
            title="ACTIVE SKUs",
            value=f"{overview['active_skus']}",
            caption="Unique products sold in this period",
            icon="📋",
            color_type="default",
        )
    with k7:
        render_kpi_card(
            title="OPEN DAYS",
            value=f"{overview['open_days']}",
            caption=f"Out of {overview['total_days']} calendar days",
            icon="📅",
            color_type="default",
        )
    with k8:
        render_kpi_card(
            title="STOCKOUT EVENTS",
            value=f"{overview['stockout_days']:,}",
            caption="Store-product-days with zero stock",
            icon="⚠️",
            color_type="warning" if overview['stockout_days'] > 0 else "default",
        )

    # Charts row: Trend + Category Donut
    st.markdown('<div class="section-title">📈 Sales Trends & Category Breakdown</div>', unsafe_allow_html=True)

    # Trend frequency selector
    freq_options = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    freq_label = st.selectbox("Group by", list(freq_options.keys()), index=1, key="overview_freq")
    freq = freq_options[freq_label]

    c_trend, c_donut = st.columns([2, 1])

    with c_trend:
        trend_df = get_daily_trend(
            store_id=store_id,
            area_id=area_id,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
        )
        if not trend_df.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df["Date"],
                y=trend_df["NetRevenueINR"],
                mode="lines+markers",
                name="Net Revenue (₹)",
                line=dict(color="#2563EB", width=2),
                marker=dict(size=4),
            ))
            apply_plotly_theme(fig_trend, height=350, x_title="Date", y_title="Net Revenue (₹ INR)")
            fig_trend.update_layout(title="Revenue Trend")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No sales data for the selected period.")

    with c_donut:
        cat_df = get_category_revenue_breakdown(
            store_id=store_id,
            area_id=area_id,
            start_date=start_date,
            end_date=end_date,
        )
        if not cat_df.empty:
            fig_donut = px.pie(
                cat_df,
                values="NetRevenueINR",
                names="Category",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_donut.update_traces(textposition="inside", textinfo="percent+label")
            fig_donut.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                showlegend=False,
                title="Revenue by Category",
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No category data for the selected period.")

    # Top Products
    st.markdown('<div class="section-title">🏆 Top 10 Products</div>', unsafe_allow_html=True)
    top_df = get_top_products(
        store_id=store_id,
        area_id=area_id,
        start_date=start_date,
        end_date=end_date,
        top_n=10,
    )
    if not top_df.empty:
        fig_top = go.Figure(go.Bar(
            x=top_df["NetRevenueINR"],
            y=[f"{r['ProductID']} — {r['ProductName'][:25]}" for _, r in top_df.iterrows()],
            orientation="h",
            marker=dict(color="#2563EB"),
            text=[f"₹{v:,.0f}" for v in top_df["NetRevenueINR"]],
            textposition="auto",
        ))
        apply_plotly_theme(fig_top, height=350, x_title="Net Revenue (₹)", y_title="")
        fig_top.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No product data for the selected period.")

    # Low Stock Summary (only for specific active store)
    if store_id and is_store_active(store_id):
        st.markdown('<div class="section-title">🚨 Low Stock Alert (as of 31 Dec 2025)</div>', unsafe_allow_html=True)
        low_df = get_low_stock_summary(store_id)
        if not low_df.empty:
            st.caption(f"⚠️ {len(low_df)} products are below their reorder point at the snapshot date.")
            display_cols = ["ProductID", "ProductName", "Category", "CurrentStockUnits",
                            "ReorderPointUnits", "SafetyStockUnits", "StockCoverageDays"]
            available_cols = [c for c in display_cols if c in low_df.columns]
            st.dataframe(low_df[available_cols].head(15), use_container_width=True)
        else:
            st.success("✅ All products are above their reorder point at the snapshot date.")

    # Store comparison (when viewing all stores)
    if not store_id:
        st.markdown('<div class="section-title">🏪 Store Comparison</div>', unsafe_allow_html=True)
        st.caption("Revenue and units comparison across all stores for the selected period.")
        active_stores = list_stores(active_at=PUNE_SNAPSHOT_DATE)
        comparison_rows = []
        for _, sr in active_stores.iterrows():
            sid = sr["StoreID"]
            ov = get_store_overview(store_id=sid, start_date=start_date, end_date=end_date)
            comparison_rows.append({
                "StoreID": sid,
                "Store Name": sr["StoreName"],
                "Locality": sr["LocalityName"],
                "Type": sr["StoreType"],
                "Revenue (₹)": ov["net_revenue_inr"],
                "Units Sold": ov["units_sold"],
                "Gross Margin %": ov["gross_margin_pct"],
                "Active SKUs": ov["active_skus"],
            })
        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df.sort_values("Revenue (₹)", ascending=False), use_container_width=True)


