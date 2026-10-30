"""Page 2: Data Explorer — Data quality, statistics, correlation, and time series analysis.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.pune_data import (
    get_data_quality_summary,
    get_store_daily_sales,
    get_product_daily_series,
    load_purchase_orders,
    load_current_inventory,
    load_products,
    load_stores,
    load_assortment,
)
from core.statistics import (
    calculate_acf_pacf,
    calculate_correlations,
    calculate_rolling_statistics,
    decompose_time_series,
    describe_numeric,
    perform_adf_test,
)
from config import PUNE_SNAPSHOT_DATE
from ui.components import (
    apply_plotly_theme,
    render_info_box,
    render_page_header,
)


def render_explorer_page():
    """Render the Data Explorer page for Pune data."""
    render_page_header(
        title="🔍 Data Explorer",
        subtitle="Explore your sales data — quality checks, statistics, and trends.",
    )

    store_id = st.session_state.get("selected_store_id")
    product_id = st.session_state.get("selected_product_id")
    date_range = st.session_state.get("selected_date_range", ("2024-01-01", "2025-12-31"))

    tab_quality, tab_stats, tab_corr, tab_ts = st.tabs([
        "📋 Data Quality",
        "📊 Statistics",
        "🔗 Correlation",
        "📈 Time Series",
    ])

    # --- TAB 1: DATA QUALITY ---
    with tab_quality:
        render_info_box(
            "What is Data Quality?",
            "This tab shows the <b>integrity and completeness</b> of the Pune stationery dataset. "
            "We verify operational quality metrics like zero-demand days, stockout censoring, and data consistency."
        )

        dq = get_data_quality_summary(store_id=store_id)

        if dq.get("total_rows", 0) > 0:
            q1, q2, q3, q4 = st.columns(4)
            with q1:
                st.metric("Total Daily Records", f"{dq['total_rows']:,}")
            with q2:
                st.metric("Unique Dates", f"{dq['unique_dates']:,}")
            with q3:
                st.metric("Unique Products", f"{dq['unique_products']:,}")
            with q4:
                st.metric("Date Range", f"{dq['date_range'][0]} to {dq['date_range'][1]}")

            q5, q6, q7, q8 = st.columns(4)
            with q5:
                st.metric("Zero-Demand Days", f"{dq['zero_demand_share']:.1f}%",
                          help="Percentage of store-product-days with zero units sold")
            with q6:
                st.metric("Stockout Days", f"{dq['stockout_share']:.1f}%",
                          help="Days where stock ran out — sales are censored during stockouts")
            with q7:
                st.metric("Closed-Day Share", f"{dq['closed_day_share']:.1f}%",
                          help="Days the store was closed (weekly off or events)")
            with q8:
                st.metric("Duplicate Keys", f"{dq['duplicate_keys']:,}",
                          help="Rows with duplicate (Date, StoreID, ProductID)")

            render_info_box(
                "⚠️ About Stockout Days",
                "On stockout days, <b>observed sales are zero</b> even if customer demand existed. "
                "This means UnitsSold may underestimate true demand during stockouts. "
                "Our forecasting models work with observed sales data."
            )

            # PO fulfillment summary
            with st.expander("📦 Purchase Order Status", expanded=False):
                po_df = load_purchase_orders()
                if store_id:
                    po_df = po_df[po_df["StoreID"] == store_id]
                po_status = po_df["OrderStatus"].value_counts().to_dict()
                for status, count in po_status.items():
                    st.markdown(f"- **{status}**: {count:,}")

            # Sample data
            with st.expander("📝 Sample Data (first 20 rows)", expanded=False):
                sample_df = get_store_daily_sales(store_id or "S01", start_date=date_range[0], end_date=date_range[1])
                if not sample_df.empty:
                    display_cols = ["Date", "StoreID", "ProductID", "ProductName", "Category",
                                    "UnitsSold", "TransactionCount", "NetRevenueINR", "GrossProfitINR",
                                    "ClosingStockUnits", "StockoutFlag", "IsStoreClosed"]
                    available_cols = [c for c in display_cols if c in sample_df.columns]
                    st.dataframe(sample_df[available_cols].head(20), use_container_width=True)
        else:
            st.warning("No data found for the selected store.")

    # --- TAB 2: STATISTICS ---
    with tab_stats:
        render_info_box(
            "What are Descriptive Statistics?",
            "These numbers summarize your sales patterns — averages, variability, and distribution. "
            "They help you understand whether demand is consistent or highly variable."
        )

        if product_id and store_id:
            series_df = get_product_daily_series(store_id, product_id, date_range[0], date_range[1])
            if not series_df.empty and "DailyUnits" in series_df.columns:
                series = series_df["DailyUnits"]
                desc = describe_numeric(series)

                st.markdown(f"**📊 Statistics for {product_id} at {store_id}:**")

                # Display stats as formatted metrics (avoiding Arrow serialization issue with mixed types)
                stat_items = {k: v for k, v in desc.items() if k != "formulas" and not isinstance(v, dict)}
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Count", f"{stat_items.get('count', 0):,}")
                    st.metric("Mean", f"{stat_items.get('mean', 0):.2f}")
                    st.metric("Median", f"{stat_items.get('median', 0):.2f}")
                with c2:
                    st.metric("Std Dev", f"{stat_items.get('sample_std', 0):.2f}")
                    st.metric("Variance", f"{stat_items.get('sample_variance', 0):.2f}")
                    st.metric("Skewness", f"{stat_items.get('skewness', 0):.2f}")
                with c3:
                    st.metric("Min", f"{stat_items.get('minimum', 0):.2f}")
                    st.metric("Max", f"{stat_items.get('maximum', 0):.2f}")
                with c4:
                    st.metric("Q1 (25%)", f"{stat_items.get('q1', 0):.2f}")
                    st.metric("Q3 (75%)", f"{stat_items.get('q3', 0):.2f}")
                    st.metric("IQR", f"{stat_items.get('IQR', 0):.2f}")

                # Histogram
                fig_hist = px.histogram(series_df, x="DailyUnits", nbins=30,
                                        title=f"Daily Units Distribution — {product_id}",
                                        color_discrete_sequence=["#2563EB"])
                fig_hist.update_layout(height=300, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
                st.plotly_chart(fig_hist, use_container_width=True)

                # ADF Test (in expander for non-technical users)
                with st.expander("🧪 Stationarity Test (Advanced)", expanded=False):
                    adf = perform_adf_test(series)
                    if adf.get("p_value") is not None:
                        if adf.get("p_value", 1.0) < 0.05:
                            st.success(f"✅ The series is stationary (p = {adf['p_value']:.4f}). Good for forecasting!")
                        else:
                            st.warning(f"⚠️ The series may be non-stationary (p = {adf['p_value']:.4f}). ARIMA differencing may help.")
                        st.caption(f"ADF Statistic: {adf.get('statistic', 'N/A')} · Lags Used: {adf.get('used_lag', 'N/A')}")
                    else:
                        st.info(adf.get("reason", "Unable to run stationarity test."))
            else:
                st.info("No daily series data. Select a product from the sidebar.")
        else:
            st.info("👈 Select a specific store and product from the sidebar to see statistics.")

    # --- TAB 3: CORRELATION ---
    with tab_corr:
        render_info_box(
            "What is Correlation?",
            "Correlation measures how two things move together. <b>+1</b> means they always increase together, "
            "<b>-1</b> means they move in opposite directions, and <b>0</b> means no relationship."
        )

        if product_id and store_id:
            series_df = get_product_daily_series(store_id, product_id, date_range[0], date_range[1])
            if not series_df.empty:
                numeric_cols = ["DailyUnits", "DailyRevenueINR", "TransactionCount",
                                "ClosingStockUnits", "DiscountPercent"]
                available = [c for c in numeric_cols if c in series_df.columns]
                if len(available) >= 2:
                    corr = calculate_correlations(series_df, available)
                    if "pearson_matrix" in corr:
                        st.markdown("**Correlation Matrix:**")
                        corr_df = pd.DataFrame(corr["pearson_matrix"])
                        fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto",
                                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                              title="Pearson Correlation")
                        fig_corr.update_layout(height=350, paper_bgcolor="#FFFFFF")
                        st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Not enough numeric columns for correlation analysis.")
            else:
                st.info("No data for the selected product and store.")
        else:
            st.info("👈 Select a specific store and product to see correlations.")

    # --- TAB 4: TIME SERIES ---
    with tab_ts:
        render_info_box(
            "Time Series Analysis",
            "View how sales change over time, including <b>rolling averages</b> (smoothed trends) "
            "and <b>seasonal patterns</b> (weekly cycles in demand)."
        )

        if product_id and store_id:
            series_df = get_product_daily_series(store_id, product_id, date_range[0], date_range[1])
            if not series_df.empty and "DailyUnits" in series_df.columns:
                # Time series plot
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=series_df["Date"],
                    y=series_df["DailyUnits"],
                    mode="lines",
                    name="Daily Units Sold",
                    line=dict(color="#2563EB", width=1),
                ))
                apply_plotly_theme(fig_ts, height=300, x_title="Date", y_title="Units Sold")
                fig_ts.update_layout(title=f"Daily Sales — {product_id} at {store_id}")
                st.plotly_chart(fig_ts, use_container_width=True)

                series = series_df["DailyUnits"]

                # Rolling statistics
                if len(series) >= 14:
                    rolling = calculate_rolling_statistics(series)
                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(x=list(range(len(series))), y=series.values,
                                                   mode="lines", name="Daily", line=dict(width=0.5, color="#94A3B8")))
                    # Use correct column names from statistics.py: RollingMean7, RollingMean30
                    if "RollingMean7" in rolling.columns:
                        fig_roll.add_trace(go.Scatter(x=list(range(len(rolling))), y=rolling["RollingMean7"].values,
                                                       mode="lines", name="7-day Average", line=dict(color="#2563EB", width=2)))
                    if "RollingMean30" in rolling.columns:
                        fig_roll.add_trace(go.Scatter(x=list(range(len(rolling))), y=rolling["RollingMean30"].values,
                                                       mode="lines", name="30-day Average", line=dict(color="#16A34A", width=2)))
                    apply_plotly_theme(fig_roll, height=250, x_title="Day Index", y_title="Units")
                    fig_roll.update_layout(title="Rolling Averages")
                    st.plotly_chart(fig_roll, use_container_width=True)

                # Decomposition (in expander)
                if len(series) >= 28:
                    with st.expander("📉 Seasonal Decomposition (Advanced)", expanded=False):
                        try:
                            decomp = decompose_time_series(series)
                            if decomp and decomp.get("status") == "success" and "trend" in decomp:
                                for comp_name in ["trend", "seasonal", "residual"]:
                                    if comp_name in decomp and decomp[comp_name]:
                                        comp_data = decomp[comp_name]
                                        fig_comp = go.Figure(go.Scatter(
                                            y=comp_data, mode="lines",
                                            line=dict(width=1, color="#2563EB"),
                                        ))
                                        fig_comp.update_layout(height=150, title=comp_name.title(),
                                                                margin=dict(l=40, r=20, t=30, b=20),
                                                                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
                                        st.plotly_chart(fig_comp, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Decomposition not available: {e}")

                # ACF/PACF (in expander)
                if len(series) >= 30:
                    with st.expander("📊 Autocorrelation (ACF/PACF) — Advanced", expanded=False):
                        try:
                            acf_pacf = calculate_acf_pacf(series)
                            if acf_pacf and acf_pacf.get("status") == "success":
                                c1, c2 = st.columns(2)
                                with c1:
                                    if "acf" in acf_pacf and acf_pacf["acf"]:
                                        acf_vals = acf_pacf["acf"]
                                        fig_acf = go.Figure(go.Bar(y=acf_vals, marker_color="#2563EB"))
                                        fig_acf.update_layout(height=200, title="ACF",
                                                              margin=dict(l=40, r=20, t=30, b=20),
                                                              paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
                                        st.plotly_chart(fig_acf, use_container_width=True)
                                with c2:
                                    if "pacf" in acf_pacf and acf_pacf["pacf"]:
                                        pacf_vals = acf_pacf["pacf"]
                                        fig_pacf = go.Figure(go.Bar(y=pacf_vals, marker_color="#0D9488"))
                                        fig_pacf.update_layout(height=200, title="PACF",
                                                                margin=dict(l=40, r=20, t=30, b=20),
                                                                paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
                                        st.plotly_chart(fig_pacf, use_container_width=True)
                        except Exception as e:
                            st.warning(f"ACF/PACF computation not available: {e}")
            else:
                st.info("No daily data found for this product and store combination.")
        else:
            st.info("👈 Select a specific store and product to see time series analysis.")
