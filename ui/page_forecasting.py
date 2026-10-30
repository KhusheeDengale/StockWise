"""Page 3: Demand Forecasting Studio — store-product level forecasting with model comparison.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.pune_data import (
    get_product_daily_series,
    is_store_active,
    list_products_for_store,
    load_stores,
)
from core.forecast_evaluation import compare_and_select_models, calculate_metrics
from core.forecast_service import ForecastService
from core.schemas import ForecastResult, InsufficientHistoryError
from config import PUNE_SNAPSHOT_DATE, HORIZONS
from ui.components import (
    apply_plotly_theme,
    render_info_box,
    render_kpi_card,
    render_page_header,
)

# Shared forecast service with caching
_FORECAST_SERVICE = ForecastService()


def _get_forecast_cache_key(store_id: str, product_id: str, horizon: int) -> str:
    return f"fc_{store_id}_{product_id}_{horizon}"


def render_forecasting_page():
    """Render the Demand Forecasting Studio."""
    render_page_header(
        title="📈 Demand Forecasting",
        subtitle="Predict future product sales — our system tests 4 models and picks the best one automatically.",
    )

    render_info_box(
        "How does this work?",
        "We test <b>4 forecasting methods</b> on your past sales data: "
        "Seasonal Patterns, Moving Average, Linear Regression, and ARIMA. "
        "Each method is scored on accuracy using data it hasn't seen before. "
        "The best performer is automatically selected to predict future demand. "
        "Just select a product and click <b>Generate Forecast</b>."
    )

    store_id = st.session_state.get("selected_store_id")
    product_id = st.session_state.get("selected_product_id")

    if not store_id:
        st.warning("👈 Select a store from the sidebar to begin forecasting.")
        return

    if not product_id:
        st.info("👈 Select a specific product from the sidebar to forecast its demand.")
        return

    if not is_store_active(store_id):
        st.error(f"🔴 Store {store_id} is closed. Forecasting is only available for active stores.")
        return

    # Store and product info
    stores_df = load_stores()
    s_row = stores_df[stores_df["StoreID"] == store_id]
    if not s_row.empty:
        s = s_row.iloc[0]
        st.markdown(f"🏪 **{s['StoreName']}** ({store_id}) · Product: **{product_id}**")

    # Forecast controls — simplified
    c1, c2 = st.columns([1, 2])
    with c1:
        horizon = st.selectbox("📅 How many days ahead?", HORIZONS, index=1, key="fc_horizon",
                                help="Predict demand for the next 7, 14, or 30 days")
        st.session_state["selected_forecast_horizon"] = horizon
    with c2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_forecast = st.button("🚀 Generate Forecast", type="primary", key="fc_run_btn")

    forecast_origin = "2025-12-31"  # Fixed to snapshot date

    # Check for cached result
    cache_key = _get_forecast_cache_key(store_id, product_id, horizon)
    cached_result = st.session_state.get(cache_key)

    if run_forecast:
        with st.spinner(f"Analyzing sales patterns and generating forecast..."):
            try:
                # Get daily series for this store-product
                series_df = get_product_daily_series(store_id, product_id)
                if series_df.empty or "DailyUnits" not in series_df.columns:
                    st.error(f"No daily sales data found for {product_id} at {store_id}.")
                    return

                # Filter up to forecast origin
                origin_dt = pd.Timestamp(forecast_origin)
                series_df = series_df[series_df["Date"] <= origin_dt].copy()

                if len(series_df) < 60:
                    st.warning(f"Only {len(series_df)} days of history available. Results may be less accurate with limited data.")

                # Get product description
                prods = list_products_for_store(store_id, as_of=PUNE_SNAPSHOT_DATE)
                desc_row = prods[prods["ProductID"] == product_id]
                desc = str(desc_row["ProductName"].iloc[0]) if not desc_row.empty else product_id

                # Run forecast using existing service
                result = _FORECAST_SERVICE.forecast_product(
                    daily_df=series_df,
                    stock_code=product_id,
                    horizon=horizon,
                    product_description=desc,
                    use_cache=False,
                )

                # Store in session
                st.session_state[cache_key] = result
                st.session_state["latest_forecast_result"] = result
                cached_result = result
                st.success(f"✅ Forecast ready! Best model: **{result.selected_model}**")

            except InsufficientHistoryError as e:
                st.error(f"⚠️ Not enough sales history for this product. Need at least 60 days of data.")
                return
            except Exception as e:
                st.error(f"❌ Forecasting failed: {e}")
                return

    # Display results
    if cached_result:
        result = cached_result

        # KPI summary
        st.markdown('<div class="section-title">📊 Forecast Summary</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_kpi_card("BEST MODEL", result.selected_model, "Automatically selected", "🏆", "accent")
        with m2:
            wape_val = result.validation_metrics.get("WAPE", "N/A")
            render_kpi_card("ACCURACY", f"{100 - float(wape_val):.1f}%" if wape_val and wape_val != "N/A" else "N/A",
                            "Based on validation testing", "📊", "success")
        with m3:
            render_kpi_card("TOTAL PREDICTED", f"{result.total_predicted_demand:,.0f} units",
                            f"Next {horizon} days", "📦", "default")
        with m4:
            avg_daily = result.total_predicted_demand / max(horizon, 1)
            render_kpi_card("AVG DAILY", f"{avg_daily:,.1f} units/day",
                            "Average predicted daily demand", "📈", "teal")

        # Forecast chart
        st.markdown('<div class="section-title">📈 Sales Prediction Chart</div>', unsafe_allow_html=True)

        # Historical + Forecast plot
        series_df = get_product_daily_series(store_id, product_id)
        origin_dt = pd.Timestamp(forecast_origin)
        historical = series_df[series_df["Date"] <= origin_dt].tail(60)  # Last 60 days for context

        fig = go.Figure()

        # Historical
        if not historical.empty:
            fig.add_trace(go.Scatter(
                x=historical["Date"],
                y=historical["DailyUnits"],
                mode="lines",
                name="Past Sales",
                line=dict(color="#94A3B8", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(148, 163, 184, 0.1)",
            ))

        # Forecast
        fc_dates = pd.to_datetime(result.forecast_dates)
        fig.add_trace(go.Scatter(
            x=fc_dates,
            y=result.daily_predictions,
            mode="lines+markers",
            name=f"Predicted ({result.selected_model})",
            line=dict(color="#2563EB", width=2.5, dash="dash"),
            marker=dict(size=5, color="#2563EB"),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
        ))

        # Origin line
        fig.add_vline(x=origin_dt, line_dash="dot", line_color="#F59E0B",
                       annotation_text="Today")

        apply_plotly_theme(fig, height=380, x_title="Date", y_title="Units")
        fig.update_layout(title=f"Demand Forecast — {product_id}")
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.markdown('<div class="section-title">📋 Day-by-Day Predictions</div>', unsafe_allow_html=True)
        fc_table = pd.DataFrame({
            "Date": result.forecast_dates,
            "Predicted Units": [round(v, 1) for v in result.daily_predictions],
        })
        st.dataframe(fc_table, use_container_width=True)

        # Model comparison (collapsible for non-technical users)
        with st.expander("🔬 Model Comparison Details", expanded=False):
            render_info_box(
                "How was the best model chosen?",
                f"We tested 4 models and the one with the <b>lowest error rate</b> won. "
                f"Winner: <b>{result.selected_model}</b>."
            )
            if result.all_model_results:
                lb_df = pd.DataFrame(result.all_model_results)
                display_cols = ["Model", "ValidationWAPE", "ValidationMAE", "TestWAPE", "TestMAE", "Selected"]
                available = [c for c in display_cols if c in lb_df.columns]
                st.dataframe(lb_df[available], use_container_width=True)

        # Warnings
        if result.warnings:
            for w in result.warnings:
                st.warning(f"⚠️ {w}")

        # Download
        st.download_button(
            "📥 Download Forecast",
            data=fc_table.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{store_id}_{product_id}_H{horizon}.csv",
            mime="text/csv",
        )

    elif not run_forecast:
        st.info("👆 Click **Generate Forecast** to predict future demand for this product.")
