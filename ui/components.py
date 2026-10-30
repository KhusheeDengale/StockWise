"""Reusable Streamlit visual components, metric cards, badges, and Plotly theme formatters.
"""
from typing import Any, Dict, List, Optional
import plotly.graph_objects as go
import streamlit as st


# ── Currency Display Helper ──────────────────────────────────────────────
# Backend data uses 'GBP' column names but we display everything as ₹ INR
CURRENCY_SYMBOL = "₹"
CURRENCY_NAME = "INR"


def format_currency(value: float, show_symbol: bool = True) -> str:
    """Format a number as Indian Rupee display string.
    
    Uses Indian numbering system (lakhs/crores) for large numbers.
    """
    if show_symbol:
        if abs(value) >= 10_00_000:  # 10 lakh+
            return f"₹{value/1_00_000:,.2f}L"
        return f"₹{value:,.2f}"
    return f"{value:,.2f}"


def render_page_header(title: str, subtitle: str):
    """Render standardized page header and subtitle."""
    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_box(title: str, message: str, icon: str = "💡"):
    """Render a user-friendly info/explanation box that helps non-technical users understand what they're seeing."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #EFF6FF, #F0F9FF); border: 1px solid #BFDBFE; border-radius: 12px; padding: 16px 20px; margin: 12px 0;">
            <div style="font-size: 14px; font-weight: 700; color: #1E40AF; margin-bottom: 6px;">{icon} {title}</div>
            <div style="font-size: 13px; color: #334155; line-height: 1.6;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_help_tooltip(label: str, explanation: str) -> str:
    """Return HTML for a label with a subtle help tooltip explanation."""
    return f'{label} <span style="font-size: 11px; color: #64748B; font-weight: 400;">— {explanation}</span>'


def render_kpi_card(
    title: str,
    value: str,
    caption: str,
    icon: str = "",
    color_type: str = "default",
):
    """Render a styled white SaaS metric card.
    
    Args:
        title: Uppercase metric label.
        value: Primary prominent number or text.
        caption: Descriptive supporting detail.
        icon: Optional icon or emoji.
        color_type: 'default', 'success', 'warning', 'critical', 'accent'.
    """
    color_map = {
        "default": "#172033",
        "success": "#16A34A",
        "warning": "#D97706",
        "critical": "#DC2626",
        "accent": "#2563EB",
        "teal": "#0D9488",
    }
    val_color = color_map.get(color_type, "#172033")

    st.markdown(
        f"""
        <div class="stockwise-card">
            <div class="stockwise-card-title">
                <span>{title}</span>
                <span>{icon}</span>
            </div>
            <div class="metric-value" style="color: {val_color};">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge(category: str, score: Optional[float] = None) -> str:
    """Return HTML string for colored risk category badge."""
    cat_lower = str(category).lower()
    badge_class = f"badge-{cat_lower}" if cat_lower in ["low", "moderate", "high", "critical"] else "badge-blue"
    score_txt = f" ({score:.1f})" if score is not None else ""
    return f'<span class="badge {badge_class}">{category}{score_txt}</span>'


def render_model_badge(model_name: str, selected: bool = False) -> str:
    """Return HTML string for model name badge."""
    badge_class = "badge-teal" if selected else "badge-blue"
    check = "[Selected] " if selected else ""
    return f'<span class="badge {badge_class}">{check}{model_name}</span>'


def render_status_badge(is_pass: bool, pass_text: str = "PASS", fail_text: str = "FAIL") -> str:
    """Return HTML string for a pass/fail status badge."""
    badge_class = "badge-low" if is_pass else "badge-critical"
    txt = pass_text if is_pass else fail_text
    return f'<span class="badge {badge_class}">{txt}</span>'


def render_empty_state(title: str, message: str, icon: str = ""):
    """Render informative empty state when user has not yet triggered an analysis."""
    icon_html = f'<div style="font-size: 24px; font-weight: 700; color: #2563EB; margin-bottom: 12px;">{icon}</div>' if icon else ""
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1.5px dashed #CBD5E1; border-radius: 12px; padding: 48px 24px; text-align: center; margin: 20px 0;">
            {icon_html}
            <div style="font-size: 18px; font-weight: 700; color: #1E293B; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 14px; color: #64748B; max-width: 520px; margin: 0 auto;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_theme(
    fig: go.Figure,
    title: str = "",
    height: int = 400,
    x_title: str = "",
    y_title: str = "",
    legend_orientation: str = "h",
) -> go.Figure:
    """Apply consistent StockWise SaaS theme styling to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=15, color="#172033", family="Inter, Segoe UI, sans-serif"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=40, r=20, t=45 if title else 20, b=40),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#172033", size=12),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, color="#64748B")),
            showgrid=True,
            gridcolor="#E9EEF5",
            zeroline=False,
            showline=True,
            linecolor="#E2E8F0",
            tickfont=dict(color="#64748B", size=11),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=12, color="#64748B")),
            showgrid=True,
            gridcolor="#E9EEF5",
            zeroline=False,
            showline=True,
            linecolor="#E2E8F0",
            tickfont=dict(color="#64748B", size=11),
        ),
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom" if legend_orientation == "h" else "top",
            y=1.02 if legend_orientation == "h" else 1.0,
            xanchor="right" if legend_orientation == "h" else "left",
            x=1.0 if legend_orientation == "h" else 1.02,
            font=dict(size=11, color="#64748B"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            font_size=12,
            font_family="Inter, Segoe UI, sans-serif",
            font_color="#FFFFFF",
        ),
    )
    return fig
