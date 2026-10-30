"""StockWise UI module package initialization.
"""
from ui.page_overview import render_overview_page
from ui.page_explorer import render_explorer_page
from ui.page_forecasting import render_forecasting_page
from ui.page_fuzzy import render_fuzzy_page
from ui.page_optimizer import render_optimizer_page
from ui.page_aiml_lab import render_aiml_lab_page
from ui.page_reports import render_reports_page
from ui.styles import apply_custom_css

__all__ = [
    "render_overview_page",
    "render_explorer_page",
    "render_forecasting_page",
    "render_fuzzy_page",
    "render_optimizer_page",
    "render_aiml_lab_page",
    "render_reports_page",
    "apply_custom_css",
]
