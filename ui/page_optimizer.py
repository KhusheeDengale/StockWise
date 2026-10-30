"""Page 5: Reorder Optimizer — GA + Greedy optimization for store-level purchasing.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

from core.pune_data import (
    get_product_daily_series,
    get_store_inventory,
    get_valid_suppliers,
    is_store_active,
    load_stores,
)
from core.forecast_service import ForecastService
from core.fuzzy_engine import evaluate_fuzzy_urgency
from core.greedy_baseline import (
    compute_normalization_scales,
    compute_objective_components,
    simulate_inventory_for_product,
    solve_greedy_baseline,
)
from core.optimization import GeneticAlgorithmOptimizer
from core.schemas import ForecastResult, OptimizationMetrics
from config import GA_CONFIG, OBJECTIVE_WEIGHTS, PUNE_SNAPSHOT_DATE
from ui.components import (
    apply_plotly_theme,
    render_info_box,
    render_kpi_card,
    render_page_header,
)

_FORECAST_SERVICE = ForecastService()


def render_optimizer_page():
    """Render the Reorder Optimizer page for Pune stores."""
    render_page_header(
        title="🛒 Reorder Optimizer",
        subtitle="Find the best quantities to order for each product — stay within budget and avoid stockouts.",
    )

    render_info_box(
        "How does this work?",
        "The optimizer finds the <b>smartest way to spend your purchasing budget</b>. "
        "It considers each product's forecasted demand, current stock, supplier lead times, "
        "and storage capacity. It uses two approaches — a quick rule-based method and an "
        "advanced evolutionary algorithm — and picks whichever gives you the best result."
    )

    store_id = st.session_state.get("selected_store_id")
    if not store_id:
        st.warning("👈 Select a store from the sidebar.")
        return
    if not is_store_active(store_id):
        st.error(f"🔴 Store {store_id} is closed. Optimization is only for active stores.")
        return

    # Get store info and inventory
    stores_df = load_stores()
    s_row = stores_df[stores_df["StoreID"] == store_id]
    store_name = s_row.iloc[0]["StoreName"] if not s_row.empty else store_id
    monthly_budget = float(s_row.iloc[0].get("MonthlyPurchaseBudgetINR", 50000)) if not s_row.empty else 50000.0

    inv_df = get_store_inventory(store_id)
    if inv_df.empty:
        st.warning(f"No inventory data for {store_id}.")
        return

    st.markdown(f"🏪 **{store_name}** ({store_id}) · {len(inv_df)} products in inventory")

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        horizon = st.selectbox("📅 Planning Horizon", [7, 14, 30], index=1, key="opt_horizon")
    with c2:
        budget = st.number_input(
            "💰 Budget (₹ INR)",
            min_value=1000.0,
            value=monthly_budget,
            step=5000.0,
            key="opt_budget",
            help="Total purchasing budget for this planning period",
        )
    with c3:
        # Category filter
        categories = sorted(inv_df["Category"].unique()) if "Category" in inv_df.columns else []
        sel_cat = st.selectbox("📦 Category Filter", ["All"] + categories, key="opt_cat")

    if sel_cat != "All" and "Category" in inv_df.columns:
        inv_df = inv_df[inv_df["Category"] == sel_cat]

    # Product limit
    max_products = st.slider("Max products to optimize", 5, min(50, len(inv_df)), min(20, len(inv_df)), key="opt_max_prods")

    run_opt = st.button("🚀 Run Optimization", type="primary", key="opt_run_btn")

    if run_opt:
        with st.spinner(f"Optimizing purchases for {store_id}..."):
            progress = st.progress(0, text="Forecasting products...")

            # Build products_data for optimizer
            products_data = []
            skipped = []

            for idx, (_, row) in enumerate(inv_df.head(max_products).iterrows()):
                pid = row["ProductID"]
                progress.progress((idx + 1) / min(max_products, len(inv_df)),
                                  text=f"Forecasting {pid}...")

                current_stock = int(row["CurrentStockUnits"])
                lead_time = int(row["LeadTimeDays"])
                cost = float(row["SupplierUnitCostINR"])
                storage_cap = int(row.get("StorageCapacityUnits", 500))
                moq = int(row.get("MinimumOrderQuantity", 1))
                multiple = int(row.get("OrderMultiple", 1))
                desc = str(row.get("ProductName", pid))

                # Get forecast
                try:
                    series_df = get_product_daily_series(store_id, pid)
                    series_df = series_df[series_df["Date"] <= pd.Timestamp(PUNE_SNAPSHOT_DATE)]

                    if len(series_df) >= 60:
                        fc = _FORECAST_SERVICE.forecast_product(
                            daily_df=series_df,
                            stock_code=pid,
                            horizon=max(horizon, lead_time),
                            product_description=desc,
                            include_arima=False,
                        )
                        daily_forecast = fc.daily_predictions[:horizon]
                        urgency = fc.uncertainty_score
                    else:
                        avg = float(row.get("AverageDailySalesLast30Days", 1.0))
                        daily_forecast = [max(0.0, avg)] * horizon
                        urgency = 0.5
                except Exception:
                    avg = float(row.get("AverageDailySalesLast30Days", 1.0))
                    daily_forecast = [max(0.0, avg)] * horizon
                    urgency = 0.5

                products_data.append({
                    "StockCode": pid,
                    "Description": desc,
                    "CurrentStock": current_stock,
                    "SupplierCostGBP": cost,  # Field name kept for compatibility with optimizer internals
                    "LeadTimeDays": lead_time,
                    "StorageCapacity": storage_cap,
                    "MinimumOrderQuantity": moq,
                    "OrderMultiple": multiple,
                    "UrgencyScore": urgency,
                    "DailyForecast": daily_forecast,
                })

            if not products_data:
                st.error("No products could be prepared for optimization.")
                return

            progress.progress(0.7, text="Running greedy optimization...")

            # Run optimization
            scales = compute_normalization_scales(products_data, budget)
            greedy_res = solve_greedy_baseline(products_data, budget, scales, OBJECTIVE_WEIGHTS)

            progress.progress(0.85, text="Running genetic algorithm...")
            ga_opt = GeneticAlgorithmOptimizer()
            ga_res = ga_opt.optimize(products_data, budget, greedy_baseline_result=greedy_res)

            # Select best
            final_res = ga_res if ga_res.metrics.ObjectiveValue <= greedy_res.metrics.ObjectiveValue else greedy_res
            best_chrom = final_res.chromosome

            progress.progress(1.0, text="Done!")

            # Store results
            st.session_state["opt_result"] = {
                "products_data": products_data,
                "greedy_res": greedy_res,
                "ga_res": ga_res,
                "final_res": final_res,
                "best_chrom": best_chrom,
                "budget": budget,
                "horizon": horizon,
            }

    # Display results
    opt = st.session_state.get("opt_result")
    if opt:
        final_res = opt["final_res"]
        greedy_res = opt["greedy_res"]
        ga_res = opt["ga_res"]
        products_data = opt["products_data"]
        best_chrom = opt["best_chrom"]
        budget = opt["budget"]
        horizon = opt["horizon"]

        st.markdown('<div class="section-title">📊 Optimization Results</div>', unsafe_allow_html=True)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("TOTAL SPEND", f"₹{final_res.metrics.TotalPurchaseCost:,.2f}",
                            f"Budget: ₹{budget:,.0f}", "💰", "accent")
        with k2:
            render_kpi_card("BUDGET USED", f"{final_res.metrics.BudgetUtilizationPercent:.1f}%",
                            f"Remaining: ₹{final_res.metrics.BudgetRemaining:,.0f}", "📊", "default")
        with k3:
            render_kpi_card("SHORTAGE REDUCTION",
                            f"{final_res.metrics.TotalExpectedShortage:.0f} units",
                            "Expected stockout units", "📉", "success")
        with k4:
            improvement = ((greedy_res.metrics.ObjectiveValue - ga_res.metrics.ObjectiveValue) /
                           max(greedy_res.metrics.ObjectiveValue, 1e-6) * 100)
            render_kpi_card("GA vs GREEDY", f"{improvement:.2f}%",
                            "Objective improvement", "🧬", "teal")

        # Comparison table
        st.markdown("**🔄 Algorithm Comparison:**")
        comp_data = {
            "Metric": ["Objective Value", "Total Cost (₹)", "Budget Used (%)",
                        "Expected Shortage", "Expected Excess", "Runtime (s)"],
            "Zero-Order": ["-", "₹0", "0%", "-", "-", "-"],
            "Greedy": [
                f"{greedy_res.metrics.ObjectiveValue:.4f}",
                f"₹{greedy_res.metrics.TotalPurchaseCost:,.2f}",
                f"{greedy_res.metrics.BudgetUtilizationPercent:.1f}%",
                f"{greedy_res.metrics.TotalExpectedShortage:.0f}",
                f"{greedy_res.metrics.TotalExpectedExcess:.0f}",
                f"{greedy_res.metrics.RuntimeSeconds:.3f}",
            ],
            "Genetic Algorithm": [
                f"{ga_res.metrics.ObjectiveValue:.4f}",
                f"₹{ga_res.metrics.TotalPurchaseCost:,.2f}",
                f"{ga_res.metrics.BudgetUtilizationPercent:.1f}%",
                f"{ga_res.metrics.TotalExpectedShortage:.0f}",
                f"{ga_res.metrics.TotalExpectedExcess:.0f}",
                f"{ga_res.metrics.RuntimeSeconds:.3f}",
            ],
        }
        st.table(pd.DataFrame(comp_data))

        # Purchase recommendations table
        st.markdown('<div class="section-title">📋 Purchase Recommendations</div>', unsafe_allow_html=True)
        rec_rows = []
        for i, p in enumerate(products_data):
            qty = best_chrom[i]
            cost_unit = p["SupplierCostGBP"]
            total_cost = qty * cost_unit
            rec_rows.append({
                "ProductID": p["StockCode"],
                "Product": p["Description"][:30],
                "Current Stock": p["CurrentStock"],
                "Forecast Demand": round(sum(p["DailyForecast"]), 0),
                "Order Qty": qty,
                "Unit Cost (₹)": f"{cost_unit:.2f}",
                "Total Cost (₹)": f"{total_cost:.2f}",
                "Lead Time": p["LeadTimeDays"],
            })
        rec_df = pd.DataFrame(rec_rows)
        st.dataframe(rec_df, use_container_width=True)

        # GA convergence chart (academic mode)
        if st.session_state.get("display_mode") == "Academic" and ga_res.generation_history:
            st.markdown('<div class="section-title">📈 GA Convergence</div>', unsafe_allow_html=True)
            gen_hist = ga_res.generation_history
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=[g["generation"] for g in gen_hist],
                y=[g["best_objective"] for g in gen_hist],
                mode="lines+markers",
                name="Best Objective",
                line=dict(color="#2563EB", width=2),
            ))
            fig_conv.add_trace(go.Scatter(
                x=[g["generation"] for g in gen_hist],
                y=[g["mean_objective"] for g in gen_hist],
                mode="lines",
                name="Mean Objective",
                line=dict(color="#94A3B8", width=1, dash="dot"),
            ))
            apply_plotly_theme(fig_conv, height=300, x_title="Generation", y_title="Objective Value")
            fig_conv.update_layout(title="GA Convergence")
            st.plotly_chart(fig_conv, use_container_width=True)

        # Download
        st.download_button(
            "📥 Download Purchase Plan CSV",
            data=rec_df.to_csv(index=False).encode("utf-8"),
            file_name=f"purchase_plan_{store_id}_H{horizon}.csv",
            mime="text/csv",
        )
