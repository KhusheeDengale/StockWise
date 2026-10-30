"""StockWise — Intelligent Inventory Forecasting & Reorder Optimization for Pune Stationery Retail.

Streamlit application with 8 pages:
  1. Business Overview         5. Reorder Optimizer
  2. Data Explorer             6. AI/ML Lab
  3. Demand Forecasting        7. Reports
  4. Stock Risk                8. New Store Planner
"""
from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure stockwise root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import PUNE_SNAPSHOT_DATE
from core.pune_data import (
    initialize_pune_data,
    list_areas,
    list_stores,
    list_products_for_store,
    is_store_active,
    load_stores,
)
from ui import (
    apply_custom_css,
    render_aiml_lab_page,
    render_explorer_page,
    render_forecasting_page,
    render_fuzzy_page,
    render_optimizer_page,
    render_overview_page,
    render_reports_page,
)

# 1. Page Configuration
st.set_page_config(
    page_title="StockWise Pune | Stationery Inventory Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject SaaS Design System CSS
apply_custom_css()

# 3. Global Session State Initialization
if "initialized" not in st.session_state:
    try:
        init_info = initialize_pune_data()
        st.session_state["initialized"] = True
        st.session_state["init_info"] = init_info
    except Exception as e:
        st.error(f"❌ Data Initialization Error: {e}")
        st.session_state["initialized"] = False

# Session state defaults
_DEFAULTS = {
    "current_page": "Overview",
    "workflow_mode": "Existing Store",
    "selected_area_id": None,
    "selected_store_id": None,
    "selected_product_id": None,
    "selected_date_range": ("2024-01-01", "2025-12-31"),
    "selected_forecast_horizon": 14,
    "display_mode": "Business",
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Navigation pages
NAV_PAGES_EXISTING = [
    "Overview",
    "Data Explorer",
    "Demand Forecasting",
    "Stock Risk",
    "Reorder Optimizer",
    "AI/ML Lab",
    "Reports",
]

NAV_PAGES_NEW_STORE = [
    "New Store Planner",
]

# Handle programmatic page navigation requests
if "pending_page" in st.session_state and st.session_state["pending_page"]:
    target = st.session_state.pop("pending_page")
    st.session_state["current_page"] = target
    st.session_state["main_navigation_radio"] = target

# 4. Sidebar
with st.sidebar:
    # Brand Header
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-title">
                <span>StockWise</span>
                <span class="brand-badge">Pune v2.0</span>
            </div>
            <div class="brand-tagline">📍 Pune Stationery Retail Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Workflow Mode
    st.markdown("<p style='font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 4px;'>Workflow</p>", unsafe_allow_html=True)
    wf_mode = st.radio(
        "Workflow",
        options=["Existing Store", "Plan a New Store"],
        index=0 if st.session_state["workflow_mode"] == "Existing Store" else 1,
        key="wf_mode_radio",
        label_visibility="collapsed",
    )
    st.session_state["workflow_mode"] = wf_mode

    # --- Existing Store Controls ---
    if wf_mode == "Existing Store":
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Area filter
        areas_df = list_areas()
        area_options = ["All Areas"] + [f"{r['AreaID']} — {r['LocalityName']}" for _, r in areas_df.iterrows()]
        sel_area = st.selectbox("📍 Locality / Area", area_options, key="area_selector")
        area_id = None if sel_area == "All Areas" else sel_area.split(" — ")[0]
        st.session_state["selected_area_id"] = area_id

        # Store filter
        stores_df = list_stores(area_id=area_id)
        store_options = []
        for _, r in stores_df.iterrows():
            sid = r["StoreID"]
            sname = r["StoreName"]
            active = is_store_active(sid)
            tag = "" if active else " [CLOSED]"
            store_options.append(f"{sid} — {sname}{tag}")

        if not store_options:
            store_options = ["No stores available"]

        sel_store_idx = 0
        # Preserve current selection
        if st.session_state.get("selected_store_id"):
            current_sid = st.session_state["selected_store_id"]
            for i, opt in enumerate(store_options):
                if opt.startswith(current_sid + " "):
                    sel_store_idx = i
                    break

        sel_store = st.selectbox("🏪 Store", store_options, index=sel_store_idx, key="store_selector")
        store_id = sel_store.split(" — ")[0] if " — " in sel_store else None
        st.session_state["selected_store_id"] = store_id
        store_active = is_store_active(store_id) if store_id else False

        # Show store info
        if store_id:
            s_row = stores_df[stores_df["StoreID"] == store_id]
            if not s_row.empty:
                s = s_row.iloc[0]
                status_icon = "🟢" if store_active else "🔴"
                st.markdown(
                    f"""<div style='font-size: 11px; color: #64748B; padding: 4px 0;'>
                    {status_icon} <b>{s['StoreType']}</b> · {s['LocalityName']}<br>
                    📅 Opened {str(s['StoreOpeningDate'].date())}{' · Closed ' + str(s['StoreClosingDate'].date()) if pd.notna(s.get('StoreClosingDate')) else ''}
                    </div>""",
                    unsafe_allow_html=True,
                )

        # Product filter (for forecasting/risk pages)
        if store_id:
            prods = list_products_for_store(store_id, as_of=PUNE_SNAPSHOT_DATE)
            prod_options = ["All Products"] + [f"{r['ProductID']} — {r['ProductName']}" for _, r in prods.iterrows()]
        else:
            prod_options = ["All Products"]

        sel_prod = st.selectbox("📦 Product", prod_options, key="product_selector")
        product_id = None if sel_prod == "All Products" else sel_prod.split(" — ")[0]
        st.session_state["selected_product_id"] = product_id

        # Date range
        dr = st.date_input(
            "📅 Historical Date Range",
            value=(date(2024, 1, 1), date(2025, 12, 31)),
            min_value=date(2019, 1, 1),
            max_value=date(2025, 12, 31),
            key="date_range_input",
        )
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            st.session_state["selected_date_range"] = (str(dr[0]), str(dr[1]))

        # Navigation
        nav_pages = NAV_PAGES_EXISTING
    else:
        # New Store mode
        nav_pages = NAV_PAGES_EXISTING + NAV_PAGES_NEW_STORE
        st.markdown(
            "<div style='padding: 8px; background: #1E3A5F; border-radius: 6px; margin: 8px 0;'>"
            "<p style='color: #60A5FA; font-size: 12px; margin: 0;'>🏗️ Planning a new stationery store in Pune. "
            "Use the <b>New Store Planner</b> page for cold-start recommendations.</p></div>",
            unsafe_allow_html=True,
        )
        nav_pages = NAV_PAGES_EXISTING + NAV_PAGES_NEW_STORE

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    nav_icons = {
        "Overview": "📊 Business Overview — Sales at a Glance",
        "Data Explorer": "🔍 Data Explorer — Understand Your Data",
        "Demand Forecasting": "📈 Demand Forecasting — Predict Future Sales",
        "Stock Risk": "⚠️ Stock Risk — Check Stockout Danger",
        "Reorder Optimizer": "🛒 Reorder Optimizer — Smart Purchasing",
        "AI/ML Lab": "🧠 AI/ML Lab — How It Works",
        "Reports": "📋 Reports — Final Purchase Plan",
        "New Store Planner": "🏗️ New Store Planner — Cold-Start Setup",
    }

    if "main_navigation_radio" not in st.session_state:
        st.session_state["main_navigation_radio"] = st.session_state["current_page"]

    def _on_nav_change():
        st.session_state["current_page"] = st.session_state["main_navigation_radio"]

    selected_nav = st.radio(
        "Navigation",
        options=nav_pages,
        format_func=lambda page: nav_icons.get(page, page),
        key="main_navigation_radio",
        on_change=_on_nav_change,
    )

    # Display Mode Toggle
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>Display Mode</p>", unsafe_allow_html=True)
    disp_mode = st.radio(
        "Mode",
        options=["Business Analytics", "Academic Lab"],
        index=0 if st.session_state["display_mode"] == "Business" else 1,
        key="display_mode_radio",
        label_visibility="collapsed",
    )
    st.session_state["display_mode"] = "Business" if disp_mode == "Business Analytics" else "Academic"

    # Dataset Info Card
    init_info = st.session_state.get("init_info", {})
    st.markdown(
        f"""
        <div class="sidebar-footer-card">
            <div class="sidebar-footer-title">📦 Data Summary</div>
            <div class="sidebar-footer-text">
                <b>Status:</b> {'Data Loaded ✅' if st.session_state.get('initialized') else 'Not Loaded ❌'}<br>
                <b>Stores:</b> {init_info.get('active_stores', 0)} active / {init_info.get('total_stores', 0)} total<br>
                <b>Products:</b> {init_info.get('total_products', 0)} stationery SKUs<br>
                <b>History:</b> Jan 2019 – Dec 2025<br>
                <b>Currency:</b> ₹ INR (Indian Rupee)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 5. Route to Active Page
active_page = selected_nav
st.session_state["current_page"] = selected_nav

if active_page == "Overview":
    render_overview_page()
elif active_page == "Data Explorer":
    render_explorer_page()
elif active_page == "Demand Forecasting":
    render_forecasting_page()
elif active_page == "Stock Risk":
    render_fuzzy_page()
elif active_page == "Reorder Optimizer":
    render_optimizer_page()
elif active_page == "AI/ML Lab":
    render_aiml_lab_page()
elif active_page == "Reports":
    render_reports_page()
elif active_page == "New Store Planner":
    try:
        from ui.page_new_store import render_new_store_page
        render_new_store_page()
    except ImportError:
        st.info("🏗️ New Store Planner page is under construction.")
else:
    st.error(f"Unknown page: {active_page}")
