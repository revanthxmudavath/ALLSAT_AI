
# -----------------------------------------------------------------------------
# APP STATIC - DEMO
"""
ALLSAT AI - Environmental Warning & Intelligence System (EWIS)
Static Demo Version - Prineville, Oregon
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import sys

# Add demo folder to Python path so we can import demo_data
sys.path.insert(0, str(Path(__file__).parent / "demo"))

from demo_data import (
    get_location_info,
    get_vegetation_data,
    get_climate_data,
    get_risk_data,
    get_all_location_keys,
    get_location_display_name
)

# Page config
st.set_page_config(
    page_title="ALLSAT AI - EWIS Demo",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_theme_css(theme: str):
    """Generate theme-specific CSS based on current theme"""

    if theme == "light":
        return """
        <style>
        :root {
            --text-strong: #0f172a;
            --text-default: #111827;
            --text-muted: #4b5563;

            --card-bg: #ffffff;
            --card-border: #e5e7eb;

            --chip-bg: #f4f6f8;
            --chip-border: #e2e8f0;
            --chip-text: #334155;

            --risk-title: #0f172a;
            --risk-body: #111827;
            --risk-subtle: #374151;

            --alert-critical-bg: #ffebee;
            --alert-critical-text: #333333;
            --alert-critical-title: #000000;
            --alert-critical-border: #f44336;
            --alert-critical-subtle: #555555;

            --alert-high-bg: #fff3e0;
            --alert-high-text: #333333;
            --alert-high-title: #000000;
            --alert-high-border: #ff9800;
            --alert-high-subtle: #555555;

            --alert-medium-bg: #e3f2fd;
            --alert-medium-text: #333333;
            --alert-medium-title: #000000;
            --alert-medium-border: #2196f3;
            --alert-medium-subtle: #555555;

            --header-main: #1f77b4;
            --header-sub: #666666;
            --note-bg: #fffaf0;
            --note-border: #fed7aa;
            --note-text: #2d3748;
        }

        [data-testid="stAppViewContainer"] {
            background: #ffffff;
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
        }

        .stApp {
            color: #111827;
        }

        /* Ensure buttons have visible text - target ALL Streamlit buttons */
        button, .stButton > button, button[data-testid*="baseButton"] {
            color: #111827 !important;
            background-color: #f3f4f6 !important;
        }

        /* Ensure metric labels and values are visible */
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: #111827 !important;
        }

        /* Fix Streamlit warning/info/success boxes */
        [data-testid="stAlert"], .stAlert, [data-baseweb="notification"] {
            color: #111827 !important;
        }

        /* Fix markdown text */
        .stMarkdown, .stMarkdown p, .stMarkdown li {
            color: #111827 !important;
        }

        /* Fix selectbox/dropdown */
        [data-baseweb="select"], .stSelectbox > div > div, [role="combobox"] {
            background-color: #f3f4f6 !important;
            color: #111827 !important;
        }

        /* Fix selectbox dropdown menu */
        [data-baseweb="popover"] li, [role="option"] {
            background-color: #ffffff !important;
            color: #111827 !important;
        }

        /* Fix toggle widget - comprehensive styling */
        [data-testid="stCheckbox"], .stCheckbox {
            color: #111827 !important;
        }

        /* Toggle switch track */
        input[type="checkbox"][role="switch"] {
            background-color: #d1d5db !important;
        }

        /* Toggle switch when ON */
        input[type="checkbox"][role="switch"]:checked {
            background-color: #ef4444 !important;
        }

        /* Toggle label text - comprehensive selectors */
        label[data-testid*="stWidgetLabel"],
        .stCheckbox label,
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] span,
        [data-testid="stCheckbox"] p,
        .row-widget.stCheckbox label,
        .row-widget.stCheckbox span {
            color: #111827 !important;
            font-weight: 500 !important;
        }

        /* Target the toggle container text */
        div[data-testid="column"] > div > div > label,
        div[data-testid="column"] span,
        div[data-testid="column"] > div > div > div > label,
        div[data-testid="column"] > div > div > div > label > div,
        div[data-testid="column"] > div > div > div > label > div > p {
            color: #111827 !important;
        }

        /* Force all toggle-related text to be dark in light mode */
        .row-widget label > div,
        .row-widget label > div > p,
        label > div > p {
            color: #111827 !important;
        }

        /* Nuclear option: force ALL text within checkbox containers to be dark */
        [data-testid="stCheckbox"] *,
        .stCheckbox *,
        .row-widget.stCheckbox * {
            color: #111827 !important;
        }

        /* Fix expander headers */
        [data-testid="stExpander"] summary, .streamlit-expanderHeader, details summary {
            background-color: #f3f4f6 !important;
            color: #111827 !important;
        }

        /* Fix top header bar */
        [data-testid="stHeader"] {
            background-color: #f8fafc !important;
        }

        /* Fix sidebar headers and text */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
            color: #111827 !important;
        }

        /* Fix horizontal dividers */
        hr {
            border-color: #36454F !important;
            background-color: #36454F !important;
            opacity: 1 !important;
        }
        </style>
        """
    else:  # dark mode
        return """
        <style>
        :root {
            --text-strong: #ffffff;
            --text-default: #e5e7eb;
            --text-muted: #9ca3af;

            --card-bg: rgba(255,255,255,0.06);
            --card-border: rgba(255,255,255,0.12);

            --chip-bg: rgba(255,255,255,0.08);
            --chip-border: rgba(255,255,255,0.12);
            --chip-text: #e5e7eb;

            --risk-title: #ffffff;
            --risk-body: #ffffff;
            --risk-subtle: rgba(255,255,255,0.85);

            --alert-critical-bg: #4a1f1f;
            --alert-critical-text: #ffebee;
            --alert-critical-title: #ffffff;
            --alert-critical-border: #f44336;
            --alert-critical-subtle: #e5e5e5;

            --alert-high-bg: #4a3a1f;
            --alert-high-text: #fff3e0;
            --alert-high-title: #ffffff;
            --alert-high-border: #ff9800;
            --alert-high-subtle: #e5e5e5;

            --alert-medium-bg: #1f2f4a;
            --alert-medium-text: #e3f2fd;
            --alert-medium-title: #ffffff;
            --alert-medium-border: #2196f3;
            --alert-medium-subtle: #e5e5e5;

            --header-main: #60a5fa;
            --header-sub: #d1d5db;
            --note-bg: #3f2f1f;
            --note-border: #92400e;
            --note-text: #fde68a;
        }

        [data-testid="stAppViewContainer"] {
            background: #0e1117;
        }

        [data-testid="stSidebar"] {
            background: #262730;
        }

        .stApp {
            color: #e5e7eb;
        }

        /* Ensure buttons have visible text - target ALL Streamlit buttons */
        button, .stButton > button, button[data-testid*="baseButton"] {
            color: #ffffff !important;
            background-color: #374151 !important;
        }

        /* Ensure metric labels and values are visible */
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: #e5e7eb !important;
        }

        /* Fix Streamlit warning/info/success boxes */
        [data-testid="stAlert"], .stAlert, [data-baseweb="notification"] {
            color: #ffffff !important;
        }

        /* Fix markdown text */
        .stMarkdown, .stMarkdown p, .stMarkdown li {
            color: #e5e7eb !important;
        }

        /* Fix selectbox/dropdown */
        [data-baseweb="select"], .stSelectbox > div > div, [role="combobox"] {
            background-color: #374151 !important;
            color: #ffffff !important;
        }

        /* Fix selectbox dropdown menu */
        [data-baseweb="popover"] li, [role="option"] {
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
        }

        /* Fix toggle widget - comprehensive styling */
        [data-testid="stCheckbox"], .stCheckbox {
            color: #e5e7eb !important;
        }

        /* Toggle switch track */
        input[type="checkbox"][role="switch"] {
            background-color: #6b7280 !important;
        }

        /* Toggle switch when ON */
        input[type="checkbox"][role="switch"]:checked {
            background-color: #ef4444 !important;
        }

        /* Toggle label text - comprehensive selectors */
        label[data-testid*="stWidgetLabel"],
        .stCheckbox label,
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] span,
        [data-testid="stCheckbox"] p,
        .row-widget.stCheckbox label,
        .row-widget.stCheckbox span {
            color: #e5e7eb !important;
            font-weight: 500 !important;
        }

        /* Target the toggle container text */
        div[data-testid="column"] > div > div > label,
        div[data-testid="column"] span,
        div[data-testid="column"] > div > div > div > label,
        div[data-testid="column"] > div > div > div > label > div,
        div[data-testid="column"] > div > div > div > label > div > p {
            color: #e5e7eb !important;
        }

        /* Force all toggle-related text to be light in dark mode */
        .row-widget label > div,
        .row-widget label > div > p,
        label > div > p {
            color: #e5e7eb !important;
        }

        /* Nuclear option: force ALL text within checkbox containers to be light */
        [data-testid="stCheckbox"] *,
        .stCheckbox *,
        .row-widget.stCheckbox * {
            color: #e5e7eb !important;
        }


        /* Fix expander headers */
        [data-testid="stExpander"] summary, .streamlit-expanderHeader, details summary {
            background-color: #374151 !important;
            color: #ffffff !important;
        }

        /* Fix top header bar */
        [data-testid="stHeader"] {
            background-color: #1f2937 !important;
        }

        /* Fix sidebar headers and text */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
            color: #e5e7eb !important;
        }

        /* Fix horizontal dividers */
        hr {
            border-color: #374151 !important;
            background-color: #374151 !important;
            opacity: 1 !important;
        }
        </style>
        """

# Base CSS that doesn't change with theme
st.markdown("""
<style>


.main-header {
  font-size: 2.5rem;
  font-weight: bold;
  color: var(--header-main);
  text-align: center;
  margin-bottom: 1rem;
}

.sub-header {
  font-size: 1.2rem;
  text-align: center;
  color: var(--header-sub);
  margin-bottom: 2rem;
}

.ewis-note {
  margin: 0.75rem auto 1.25rem auto;
  padding: 0.75rem 1rem;
  text-align: center;
  font-size: 0.95rem;
  line-height: 1.35;
  color: var(--note-text);
  background: var(--note-bg);
  border: 1px solid var(--note-border);
  border-radius: 0.75rem;
}

.demo-tag-wrap {
  text-align: center;
  margin-bottom: 0.5rem;
}
.demo-tag {
  display: inline-block;
  font-size: 0.8rem;
  font-weight: 600;

  color: var(--chip-text);
  background: var(--chip-bg);
  border: 1px solid var(--chip-border);

  border-radius: 999px;
  padding: 0.2rem 0.7rem;
  letter-spacing: 0.02em;
}


.metric-card {
  background-color: var(--card-bg);
  border: 1px solid var(--card-border);
  padding: 1rem;
  border-radius: 0.5rem;
  margin: 0.5rem 0;
}

.alert-critical {
  background-color: var(--alert-critical-bg);
  color: var(--alert-critical-text);
  padding: 1rem;
  border-left: 4px solid var(--alert-critical-border);
  margin: 0.5rem 0;
}
.alert-high {
  background-color: var(--alert-high-bg);
  color: var(--alert-high-text);
  padding: 1rem;
  border-left: 4px solid var(--alert-high-border);
  margin: 0.5rem 0;
}
.alert-medium {
  background-color: var(--alert-medium-bg);
  color: var(--alert-medium-text);
  padding: 1rem;
  border-left: 4px solid var(--alert-medium-border);
  margin: 0.5rem 0;
}

/* Alert text element classes */
.alert-title {
  color: var(--alert-critical-title);
  margin: 0 0 0.5rem 0;
}
.alert-message {
  color: var(--alert-critical-text);
  margin: 0.25rem 0;
}
.alert-recommendation {
  color: var(--alert-critical-subtle);
  margin: 0.25rem 0;
}

/* Risk card text helpers */
.risk-title { color: var(--risk-title); margin: 0; }
.risk-level { color: var(--risk-title); margin: 0.5rem 0; }
.risk-body  { color: var(--risk-body);  margin: 1rem 0; }
.risk-subtle{ color: var(--risk-subtle); margin: 0.5rem 0; }

</style>
""", unsafe_allow_html=True)


def display_vegetation_analysis(location_key):
    """Display vegetation analysis results"""
    veg_data = get_vegetation_data(location_key)
    loc_info = get_location_info(location_key)
    
    if not veg_data or not loc_info:
        st.error(f"❌ No vegetation data available for {location_key}")
        return
    
    # Location Information
    st.markdown("## 📍 Location Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Address", loc_info['display_name'])
        st.metric("Latitude", f"{loc_info['coordinates']['lat']:.4f}°N")
    with col2:
        st.metric("Longitude", f"{loc_info['coordinates']['lon']:.4f}°W")
        st.metric("Coverage Area", "30 km radius")
    
    st.markdown("---")
    
    # Vegetation Indices
    st.markdown("## 🌱 Vegetation Indices")
    
    metrics = veg_data['metrics']
    
    # NDVI metrics
    st.markdown("### NDVI (Normalized Difference Vegetation Index)")
    st.caption("Measures vegetation health and density (-1.0 to +1.0)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{metrics['ndvi']['mean']:.3f}")
    with col2:
        st.metric("Minimum", f"{metrics['ndvi']['min']:.3f}")
    with col3:
        st.metric("Maximum", f"{metrics['ndvi']['max']:.3f}")
    with col4:
        st.metric("Std Dev", f"{metrics['ndvi']['std']:.3f}")
    
    # NDVI interpretation
    with st.expander("📖 Example NDVI Interpretation Guide"):
        st.markdown("""
        - **< 0.2**: Bare soil, rocks, water
        - **0.2 - 0.3**: Sparse vegetation
        - **0.3 - 0.6**: Moderate vegetation ✓
        - **0.6 - 0.8**: Dense vegetation ✓✓
        - **> 0.8**: Very dense vegetation
        """)
    
    st.markdown("---")
    
    # NDMI metrics
    st.markdown("### NDMI (Normalized Difference Moisture Index)")
    st.caption("Measures plant water content and stress (-1.0 to +1.0)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average", f"{metrics['ndmi']['mean']:.3f}")
    with col2:
        st.metric("Minimum", f"{metrics['ndmi']['min']:.3f}")
    with col3:
        st.metric("Maximum", f"{metrics['ndmi']['max']:.3f}")
    with col4:
        st.metric("Std Dev", f"{metrics['ndmi']['std']:.3f}")
    
    # NDMI interpretation
    with st.expander("📖 Example NDMI Interpretation Guide"):
        st.markdown("""
        - **< 0.0**: Severe drought stress ⚠️
        - **0.0 - 0.2**: Moderate stress
        - **0.2 - 0.4**: Adequate moisture ✓
        - **> 0.4**: High moisture content
        """)
    
    st.markdown("---")
    
    # Data Quality
    st.markdown("## 📊 Data Quality")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cloud Coverage", f"{veg_data['cloud_coverage']:.1f}%")
    with col2:
        valid_pixels = metrics['valid_pixels']
        total_pixels = metrics['total_pixels']
        coverage_pct = metrics['ndvi']['coverage_percent']
        st.metric("Valid Pixels", f"{valid_pixels:,} / {total_pixels:,} ({coverage_pct:.1f}%)")
    
    st.markdown("---")
    
    # Alerts
    st.markdown("## ⚠️ Pre-generated Demonstration Indicator Status")
    st.warning("**NOTE:** Severity labels shown are pre-generated examples for demonstration purposes only.")
    
    if veg_data['alerts']:
        pass
        
        for i, alert in enumerate(veg_data['alerts'], 1):
            severity_class = f"alert-{alert['severity'].lower()}"

            st.markdown(f"""
            <div class="{severity_class}">
                <h4 class="alert-title">[{i}] [{alert['severity']} (EXAMPLE)] {alert['title']}</h4>
                <p class="alert-message"><strong>Message:</strong> {alert['message']}</p>
                <p class="alert-recommendation">💡 <em>{alert['recommendation']}</em></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ **No Pre-generated Demonstration Indicators**")
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("## 🗺️ Example Satellite Imagery Analysis")
    
    # Display the sentinel image
    if 'sentinel' in veg_data['images']:
        # Use Path to handle both Windows and Unix separators
        image_path = Path("demo") / veg_data['images']['sentinel']
        
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            # Try alternate location at root level
            alt_path = Path(veg_data['images']['sentinel'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    else:
        st.warning("⚠️ No vegetation imagery available for this location")

def display_climate_analysis(location_key):
    """Display climate analysis results"""
    climate_data = get_climate_data(location_key)
    loc_info = get_location_info(location_key)
    
    if not climate_data or not loc_info:
        st.error(f"❌ No climate data available for {location_key}")
        return
    
    # Location Information
    st.markdown("## 📍 Location Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Address", loc_info['display_name'])
        st.metric("Latitude", f"{loc_info['coordinates']['lat']:.4f}°N")
    with col2:
        st.metric("Longitude", f"{loc_info['coordinates']['lon']:.4f}°W")
        st.metric("Data Distance (Grid Mismatch)", "0.1 km")
    
    st.markdown("---")
    
    # Climate Metrics
    st.markdown("## 🌡️ Climate Metrics")
    
    metrics = climate_data['metrics']
    
    # Temperature
    st.markdown("### Temperature (2m above ground)")
    col1, col2, col3 = st.columns(3)
    with col1:
        temp_c = metrics['temperature']['mean_c']
        temp_f = temp_c * 9/5 + 32
        st.metric("Average", f"{temp_c:.1f}°C ({temp_f:.1f}°F)")
    with col2:
        max_c = metrics['temperature']['max_c']
        max_f = max_c * 9/5 + 32
        st.metric("Maximum", f"{max_c:.1f}°C ({max_f:.1f}°F)")
    with col3:
        min_c = metrics['temperature']['min_c']
        min_f = min_c * 9/5 + 32
        st.metric("Minimum", f"{min_c:.1f}°C ({min_f:.1f}°F)")
    
    st.markdown("---")
    
    # Precipitation
    st.markdown("### Precipitation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", f"{metrics['precipitation']['total_mm']:.1f} mm")
    with col2:
        st.metric("Average/Month", f"{metrics['precipitation']['mean_daily_mm']:.1f} mm")
    with col3:
        st.metric("Max in Month", f"{metrics['precipitation']['max_daily_mm']:.1f} mm")
    
    st.markdown("---")
    
    # Soil Moisture
    st.markdown("### Soil Moisture (Top Layer)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average", f"{metrics['soil_moisture']['mean']:.3f} m³/m³")
    with col2:
        st.metric("Minimum", f"{metrics['soil_moisture']['min']:.3f} m³/m³")
    with col3:
        st.metric("Maximum", f"{metrics['soil_moisture']['max']:.3f} m³/m³")
    
    st.markdown("---")
    
    # Time Period
    st.markdown("## 📅 Time Period")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Start Date", climate_data['date_range']['start'])
    with col2:
        st.metric("End Date", climate_data['date_range']['end'])
    with col3:
        st.metric("Duration", "1 month")
    
    st.markdown("---")
    
    # Alerts
    st.markdown("## ⚠️ Pre-generated Demonstration Indicator Status")
    
    
    if climate_data['alerts']:
        st.warning("**NOTE:** Severity labels shown are pre-generated examples for demonstration purposes only.")
    
        
        for i, alert in enumerate(climate_data['alerts'], 1):
            severity_class = f"alert-{alert['severity'].lower()}"

            st.markdown(f"""
            <div class="{severity_class}">
                <h4 class="alert-title">[{i}] [{alert['severity']} (EXAMPLE)] {alert['title']}</h4>
                <p class="alert-message"><strong>Message:</strong> {alert['message']}</p>
                <p class="alert-recommendation">💡 <em>{alert['recommendation']}</em></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ **No Pre-generated Demonstration Indicators**")
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("## 📈 Example Climate Analysis Charts")
    
    # Display the combined analysis image
    if 'combined_analysis' in climate_data['images']:
        image_path = Path("demo") / climate_data['images']['combined_analysis']
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            alt_path = Path(climate_data['images']['combined_analysis'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    elif 'climate' in climate_data['images']:
        image_path = Path("demo") / climate_data['images']['climate']
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            alt_path = Path(climate_data['images']['climate'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    else:
        st.warning("⚠️ No climate visualizations available for this location")

def display_risk_assessment(location_key):
    """Display combined risk assessment"""
    risk_data = get_risk_data(location_key)
    loc_info = get_location_info(location_key)
    
    if not risk_data or not loc_info:
        st.error(f"❌ No risk data available for {location_key}")
        return
    
    # Overall Alert Banner
    alert = risk_data['overall_alert']
    st.markdown("⚠️ Pre-generated Demonstration Overall Risk Indicator")
    st.warning("**NOTE:** Severity labels shown are pre-generated examples for demonstration purposes only.")
    if alert['severity'] == 'CRITICAL':
        st.error(f"{alert['icon']} **{alert['severity']}**: {alert['message']} (EXAMPLE)")
    elif alert['severity'] == 'WARNING':
        st.warning(f"{alert['icon']} **{alert['severity']}**: {alert['message']} (EXAMPLE)")
    else:
        st.success(f"{alert['icon']} **{alert['severity']}**: {alert['message']} (EXAMPLE)")
    
    st.markdown("---")
    
    # Risk Scores Side by Side
    col1, col2 = st.columns(2)
    
    # Drought Risk
    with col1:
        drought = risk_data['drought_risk']
        st.markdown(f"## 🌾 Example Drought Risk")
        
        st.markdown(f"""
        <div style="background-color: {drought['risk_color']}22; padding: 2rem; border-radius: 0.5rem; border-left: 4px solid {drought['risk_color']};">
            <h1 class="risk-title">{drought['mean_score']:.2f}</h1>
            <h3 class="risk-level">{drought['risk_level']} Risk (SCENARIO)</h3>
            <p class="risk-body">{drought['recommendation']}</p>
            <hr style="border-color: {drought['risk_color']};">
            <p class="risk-subtle"><strong>Pre-generated demonstration scenario. No live analysis occurs.</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Distribution
        with st.expander("📊 Example Risk Distribution"):
            dist = drought['class_distribution']
            pixels = drought['class_pixels']
            st.markdown(f"""
            **Low (0.00-0.33)**  
            {dist['low']:.1f}%  
            ↑ {pixels['low']:,} pixels
            
            **Moderate (0.33-0.66)**  
            {dist['moderate']:.1f}%  
            ↑ {pixels['moderate']:,} pixels
            
            **High (0.66-1.00)**  
            {dist['high']:.1f}%  
            ↑ {pixels['high']:,} pixels
            """)
    
    # Wildfire Risk
    with col2:
        wildfire = risk_data['wildfire_risk']
        st.markdown(f"## 🔥 Example Wildfire Risk")
        
        st.markdown(f"""
        <div style="background-color: {wildfire['risk_color']}22; padding: 2rem; border-radius: 0.5rem; border-left: 4px solid {wildfire['risk_color']};">
            <h1 class="risk-title">{wildfire['mean_score']:.2f}</h1>
            <h3 class="risk-level">{wildfire['risk_level']} Risk (SCENARIO)</h3>
            <p class="risk-body">{wildfire['recommendation']}</p>
            <hr style="border-color: {wildfire['risk_color']};">
            <p class="risk-subtle"><strong>Pre-generated demonstration scenario. No live analysis occurs.</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Distribution
        with st.expander("📊 Example Risk Distribution"):
            dist = wildfire['class_distribution']
            pixels = wildfire['class_pixels']
            st.markdown(f"""
            **Low (0.00-0.33)**  
            {dist['low']:.1f}%  
            ↑ {pixels['low']:,} pixels
            
            **Moderate (0.33-0.66)**  
            {dist['moderate']:.1f}%  
            ↑ {pixels['moderate']:,} pixels
            
            **High (0.66-1.00)**  
            {dist['high']:.1f}%  
            ↑ {pixels['high']:,} pixels
            """)
    
    st.markdown("---")
    
    # Risk Maps Visualization
    st.markdown("## 🗺️ Example Risk Maps")
    
    if 'combined_visualization' in risk_data['images']:
        image_path = Path("demo") / risk_data['images']['combined_visualization']
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            alt_path = Path(risk_data['images']['combined_visualization'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    elif 'risk' in risk_data['images']:
        image_path = Path("demo") / risk_data['images']['risk']
        if image_path.exists():
            st.image(str(image_path), width='stretch')
        else:
            alt_path = Path(risk_data['images']['risk'])
            if alt_path.exists():
                st.image(str(alt_path), width='stretch')
            else:
                st.warning(f"⚠️ Visualization not available. Looking for: {image_path} or {alt_path}")
    else:
        st.warning("⚠️ No risk visualizations available for this location")

def toggle_theme_callback():
    """Callback to sync theme with toggle state"""
    # When toggle is ON (True) → Light mode
    # When toggle is OFF (False) → Dark mode
    st.session_state.theme = "light" if st.session_state.theme_toggle else "dark"

def main():
    # Header

    # Initialize theme state - defaults to light mode
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    # Initialize toggle state based on current theme (only on first run)
    if "theme_toggle" not in st.session_state:
        st.session_state.theme_toggle = (st.session_state.theme == "light")

    # Apply theme CSS dynamically based on current theme state
    st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

    top_left, top_right = st.columns([0.82, 0.18])

    with top_right:
        # Use key parameter with callback - no double binding
        st.toggle(
            "🌑 / ☀️ Theme",
            value=st.session_state.theme_toggle,
            key="theme_toggle",
            on_change=toggle_theme_callback
        )

    st.markdown(
        '<div class="demo-tag-wrap"><span class="demo-tag">Pre-generated Demonstration Prototype</span></div>',
        unsafe_allow_html=True
    )

    st.markdown("<p class='ewis-note'>EWIS outputs are provided for evaluation and demonstration purposes only and are not intended for operational, regulatory, or decision-making use.</p>", unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🛰️ ALLSAT AI - EWIS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Environmental Warning & Intelligence System - Demo</p>', unsafe_allow_html=True)
    
    # Sidebar - Analysis Type Selection
    st.sidebar.markdown("## 📊 Analysis Settings")
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["vegetation", "climate", "both"],
        format_func=lambda x: {
            "vegetation": "Vegetation (Sentinel-2)",
            "climate": "Climate (ERA5-Land)",
            "both": "Combined Risk Assessment"
        }[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This Demo")
    st.sidebar.info("""
    This demo showcases ALLSAT AI's environmental monitoring capabilities using simulated and publicly available data for 5 locations.
    
    **Data Sources:**
    - Sentinel-2 L2A (Last 30 days)
    - ERA5-Land (Oct-Nov 2025)
    """)
    
    # Main content
    st.markdown("## 🗺️ Select Location")
    
    # Location buttons (horizontal)
    locations = ['prineville', 'bend', 'portland', 'Dallas', 'nashville']
    location_names = ['Prineville', 'Bend', 'Portland', 'Dallas', 'Nashville']
    
    cols = st.columns(5)
    selected_location = None
    
    for i, (loc_key, loc_name) in enumerate(zip(locations, location_names)):
        with cols[i]:
            if st.button(loc_name, key=f"btn_{loc_key}", width='stretch'):
                selected_location = loc_key
    
    # Display results based on selection
    if selected_location:
        st.markdown("---")
        st.markdown(f"## 📊 Analysis Results - {get_location_display_name(selected_location)}")
        
        if analysis_type == "vegetation":
            display_vegetation_analysis(selected_location)
        elif analysis_type == "climate":
            display_climate_analysis(selected_location)
        else:  # both
            display_risk_assessment(selected_location)
    else:
        # Empty state
        st.info("👆 **Please select a location above to view analysis results**")
        
        st.markdown("---")
        st.markdown("### Available Locations")
        
        for loc_key, loc_name in zip(locations, location_names):
            loc_info = get_location_info(loc_key)
            if loc_info:
                st.markdown(f"**{loc_name}**: {loc_info['coordinates']['lat']:.2f}°N, {loc_info['coordinates']['lon']:.2f}°W")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-muted); padding: 1rem;">
        <p><strong>ALLSAT AI - Environmental Warning & Intelligence System</strong></p>
        <p>EWIS outputs are provided for evaluation and demonstration purposes only and are not intended for operational,
        regulatory, or decision-making use.</p>
        <p>Sentinel-2 Satellite Imagery + ERA5-Land Climate Data </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()