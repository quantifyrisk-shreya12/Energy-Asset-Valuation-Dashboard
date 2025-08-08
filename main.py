"""
Energy Asset Valuation Dashboard - Main Application
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page imports
from pages import market_overview, asset_analysis, valuation_models, scenarios

def main():
    """
    Main application function
    """
    st.set_page_config(
        page_title="Energy Asset Valuation Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">⚡ Energy Asset Valuation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Choose Analysis Module:",
        [
            "🏠 Home",
            "📊 Market Overview", 
            "🏭 Asset Analysis",
            "💰 Valuation Models",
            "🎯 Scenario Planning"
        ]
    )
    
    # Add some info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 About This Dashboard")
    st.sidebar.info("""
    This dashboard demonstrates comprehensive energy asset 
    strategy and valuation capabilities, including:
    
    • Real-time market data analysis
    • Asset performance monitoring  
    • DCF and valuation models
    • Portfolio optimization
    • Risk assessment
    • Scenario planning
    """)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Market Overview":
        market_overview.show_market_overview()
    elif page == "🏭 Asset Analysis":
        asset_analysis.show_asset_analysis()
    elif page == "💰 Valuation Models":
        valuation_models.show_valuation_models()
    elif page == "🎯 Scenario Planning":
        scenarios.show_scenarios()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Built for Uniper Interview Demonstration*")

def show_home_page():
    """
    Display home page with overview
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 🎯 Project Overview
        
        This dashboard demonstrates end-to-end implementation capabilities for 
        **Asset Strategy & Valuation** at Uniper, showcasing:
        """)
        
        # Key capabilities
        st.markdown("""
        ### 🔧 Core Capabilities Demonstrated:
        
        **📊 Market Data Integration**
        - Real-time electricity price data (ENTSO-E API)
        - Gas and carbon price monitoring
        - Cross-functional market analytics
        
        **🏭 Asset Performance Analysis**
        - Power plant technical data analysis
        - Efficiency and availability metrics
        - Merit order positioning
        
        **💰 Financial Modeling**
        - DCF valuation models
        - Monte Carlo simulations
        - Risk assessment (VaR, sensitivity)
        - Portfolio optimization
        
        **🎯 Strategic Planning**
        - Scenario analysis
        - Dispatch optimization
        - Hedging recommendations
        - Maintenance scheduling
        """)
        
        # Technical stack info
        with st.expander("🛠️ Technical Implementation"):
            st.markdown("""
            **Architecture:** Functional Python programming
            **Frontend:** Streamlit interactive dashboard
            **Data Sources:** ENTSO-E Transparency Platform, Financial APIs
            **Models:** NumPy, SciPy, Scikit-learn
            **Visualization:** Plotly interactive charts
            **APIs:** Real-time market data integration
            """)
        
        # Business impact
        with st.expander("📈 Business Value"):
            st.markdown("""
            **For Uniper's Asset Strategy & Valuation Team:**
            - Automated market data analysis supporting business cases
            - Interactive dashboards for cross-functional collaboration
            - Model-based analysis for investment decisions
            - Risk assessment tools for portfolio management
            - Scenario planning for strategic initiatives
            """)
    
    # Quick metrics overview
    st.markdown("---")
    st.subheader("📊 Quick Market Snapshot")
    
    # Import required modules for quick metrics
    from data.market_data import get_market_overview
    from data.sample_assets import get_sample_assets
    
    try:
        market_data, price_data = get_market_overview()
        assets_df = get_sample_assets()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Power Price",
                f"€{market_data['current_power_price']:.1f}/MWh",
                f"{market_data['current_power_price'] - market_data['avg_power_price_30d']:.1f}"
            )
        
        with col2:
            st.metric(
                "Gas Price", 
                f"€{market_data['gas_price']:.1f}/MWh",
                f"{market_data['gas_change_24h']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Carbon Price",
                f"€{market_data['carbon_price']:.1f}/t",
                f"{market_data['carbon_change_24h']:.1f}%"
            )
            
        with col4:
            st.metric(
                "Portfolio Capacity",
                f"{assets_df['capacity_mw'].sum():,.0f} MW",
                f"{len(assets_df)} Assets"
            )
    
    except Exception as e:
        st.warning("Market data temporarily unavailable. Please check Market Overview page.")
    
    # Getting started
    st.markdown("---")
    st.info("👈 **Start exploring:** Use the sidebar navigation to access different analysis modules!")

if __name__ == "__main__":
    main()