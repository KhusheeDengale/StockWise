"""Design tokens, typography, and custom CSS styling for StockWise Streamlit application.
"""

CUSTOM_CSS = """
<style>
/* Google Font Import */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* -------------------------------------------------------------
   STOCKWISE DESIGN SYSTEM TOKENS & BASE STYLING
   Primary Background: #F6F8FC
   Card Background: #FFFFFF
   Sidebar Background: #111827
   Primary Accent: #2563EB
   Secondary Accent: #0D9488
   Text: #172033 / #64748B
   Border: #E2E8F0
------------------------------------------------------------- */

/* Global App Container */
.stApp {
    background-color: #F6F8FC !important;
    color: #172033 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

/* FORCE ALL TEXT TO BE DARK ON LIGHT BACKGROUND */
.stApp p, .stApp span, .stApp label, .stApp div,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp li, .stApp td, .stApp th, .stApp caption,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stText"],
.stAlert p, .stAlert span,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #172033 !important;
}

/* Fix metric component text colors explicitly */
[data-testid="stMetricValue"] {
    color: #172033 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #64748B !important;
}
[data-testid="stMetricDelta"] svg {
    fill: currentColor !important;
}

/* Caption text */
[data-testid="stCaptionContainer"], .stCaption, small {
    color: #64748B !important;
}

/* Fix selectbox, input, radio label text colors */
.stSelectbox label, .stTextInput label, .stNumberInput label,
.stDateInput label, .stSlider label, .stRadio label,
.stMultiSelect label, .stCheckbox label,
[data-testid="stWidgetLabel"] p {
    color: #172033 !important;
    font-weight: 500 !important;
}

/* Fix widget text inside inputs */
.stSelectbox [data-baseweb="select"] span,
.stTextInput input,
.stNumberInput input,
[data-baseweb="input"] input {
    color: #172033 !important;
}

/* Fix selectbox dropdown text */
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
    color: #172033 !important;
}

/* Fix tab text */
.stTabs [data-baseweb="tab"] {
    color: #64748B !important;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
}

/* Fix download button text */
.stDownloadButton button {
    color: #1E293B !important;
}

/* Fix info/warning/error/success boxes text */
[data-testid="stAlert"] p {
    color: #172033 !important;
}

/* Fix expander text */
[data-testid="stExpander"] summary span {
    color: #172033 !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: #172033 !important;
}

/* Fix JSON viewer text */
.stJson {
    color: #172033 !important;
}

/* Fix table text */
[data-testid="stTable"] td, [data-testid="stTable"] th {
    color: #172033 !important;
}

/* Fix progress bar text */
.stProgress [data-testid="stMarkdownContainer"] p {
    color: #64748B !important;
}

/* Hide default streamlit header/footer decorations */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}
footer {
    display: none !important;
}
#MainMenu {
    visibility: hidden;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1F2937;
    width: 270px !important;
}

section[data-testid="stSidebar"] div.stRadio > label {
    display: none;
}

/* SIDEBAR TEXT MUST BE LIGHT (white/grey on dark bg) */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stDateInput label {
    color: #E2E8F0 !important;
}

/* Sidebar selectbox text */
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
section[data-testid="stSidebar"] [data-baseweb="input"] input {
    color: #E2E8F0 !important;
}

/* Radio Navigation Styling in Sidebar */
section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label {
    background-color: transparent;
    padding: 10px 14px;
    border-radius: 8px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
    cursor: pointer;
    border: 1px solid transparent;
}

section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label:hover {
    background-color: #1F2937;
    border-color: #374151;
}

section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
    background-color: rgba(37, 99, 235, 0.2) !important;
    border-color: #2563EB !important;
}

section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label[data-checked="true"] p {
    color: #60A5FA !important;
    font-weight: 600 !important;
}

/* Brand Banner in Sidebar */
.brand-container {
    padding: 16px 12px 24px 12px;
    border-bottom: 1px solid #1F2937;
    margin-bottom: 16px;
}
.brand-title {
    font-size: 24px;
    font-weight: 800;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.brand-tagline {
    font-size: 12px;
    color: #94A3B8 !important;
    font-weight: 500;
    margin-top: 2px;
}
.brand-badge {
    background: linear-gradient(135deg, #2563EB, #0D9488);
    color: white !important;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 6px;
    text-transform: uppercase;
}

/* Sidebar Footer Info Card */
.sidebar-footer-card {
    background-color: #1F2937;
    border: 1px solid #374151;
    border-radius: 10px;
    padding: 14px;
    margin-top: 24px;
}
.sidebar-footer-title {
    font-size: 11px;
    font-weight: 700;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.sidebar-footer-text {
    font-size: 12px;
    color: #CBD5E1 !important;
    line-height: 1.4;
}

/* Page Header & Subtitles */
.page-title {
    font-size: 30px;
    font-weight: 800;
    color: #172033 !important;
    letter-spacing: -0.6px;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 15px;
    color: #64748B !important;
    margin-bottom: 22px;
    font-weight: 400;
}

/* Section Headings */
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #172033 !important;
    letter-spacing: -0.3px;
    margin-top: 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* SaaS White Cards */
.stockwise-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
    margin-bottom: 16px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stockwise-card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
}

.stockwise-card-title {
    font-size: 14px;
    font-weight: 600;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Metric Display Inside Cards */
.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #172033 !important;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 4px;
}
.metric-caption {
    font-size: 12px;
    color: #64748B !important;
    font-weight: 400;
}

/* Badges for Risk, Status, and Models */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    line-height: 1.2;
}
.badge-low {
    background-color: #DCFCE7;
    color: #166534 !important;
    border: 1px solid #BBF7D0;
}
.badge-moderate {
    background-color: #FEF3C7;
    color: #92400E !important;
    border: 1px solid #FDE68A;
}
.badge-high {
    background-color: #FFEDD5;
    color: #9A3412 !important;
    border: 1px solid #FED7AA;
}
.badge-critical {
    background-color: #FEE2E2;
    color: #991B1B !important;
    border: 1px solid #FECACA;
}
.badge-blue {
    background-color: #EFF6FF;
    color: #1E40AF !important;
    border: 1px solid #DBEAFE;
}
.badge-teal {
    background-color: #CCFBF1;
    color: #115E59 !important;
    border: 1px solid #99F6E4;
}

/* Primary and Secondary Streamlit Buttons */
button[kind="primary"],
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2) !important;
    transition: all 0.15s ease !important;
}
button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3) !important;
}

button[kind="secondary"],
.stButton > button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    color: #1E293B !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
button[kind="secondary"]:hover,
.stButton > button[kind="secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
}

/* Tab Component Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    height: 42px;
    padding: 0 16px;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    color: #64748B !important;
    border: none;
    background-color: transparent;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #2563EB !important;
    background-color: rgba(37, 99, 235, 0.04) !important;
}

/* Dataframe/Table Container Customization */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
    background-color: #FFFFFF;
}

/* Math / Latex Box Styling */
.academic-box {
    background-color: #F8FAFC;
    border-left: 4px solid #2563EB;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
}
.academic-box-title {
    font-size: 13px;
    font-weight: 700;
    color: #2563EB !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

/* Process Diagram Stepper */
.stepper-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    overflow-x: auto;
}
.step-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    min-width: 110px;
}
.step-icon-circle {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background-color: #EFF6FF;
    color: #2563EB !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: bold;
    border: 2px solid #BFDBFE;
    margin-bottom: 8px;
}
.step-label {
    font-size: 13px;
    font-weight: 600;
    color: #1E293B !important;
}
.step-sublabel {
    font-size: 11px;
    color: #64748B !important;
}
.step-arrow {
    color: #94A3B8 !important;
    font-size: 20px;
    font-weight: bold;
    margin: 0 12px;
}

/* Spinner text color fix */
.stSpinner > div > div {
    color: #172033 !important;
}

/* Fix markdown links */
.stApp a {
    color: #2563EB !important;
}

/* Fix radio button text in main area */
.main .stRadio label p {
    color: #172033 !important;
}
</style>
"""


def apply_custom_css():
    """Inject custom SaaS CSS into the current Streamlit app session."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
