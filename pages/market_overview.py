"""
Market Overview Dashboard Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
from data.market_data import get_market_overview, calculate_spark_spread, analyze_price_trends, calculate_market_indicators
from data.sample_assets import get_sample_assets
from utils.visualization import create_price_chart, create_spark_spread_chart, create_price_heatmap, create_correlation_matrix
from utils.calculations import format_currency, format_percentage

def show_market_overview():
    """
    Display market overview dashboard
    """
    st.title("📊 Market Overview & Analysis")
    
    # Load data
    try:
        market_data, price_data = get_market_overview()
        market_indicators = calculate_market_indicators()
        
        # Market summary cards
        st.subheader("🎯 Current Market Conditions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Power Price",
                f"€{market_data['current_power_price']:.1f}/MWh",
                f"{market_data['current_power_price'] - market_data['avg_power_price_30d']:.1f}",
                delta_color="inverse"
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
            spark_spread = calculate_spark_spread(
                market_data['current_power_price'],
                market_data['gas_price']
            )
            st.metric(
                "Spark Spread", 
                f"€{spark_spread:.1f}/MWh",
                "Gas Plant Margin"
            )
        
        # Market regime indicators
        st.subheader("🌡️ Market Regime Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **Price Regime:** 
            <span style="color: {market_indicators['price_regime_color']}; font-weight: bold;">
            {market_indicators['price_regime']}
            </span>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            **Volatility:** {market_indicators['volatility_regime']}  
            (σ = {market_data['power_price_volatility']:.1f} EUR/MWh)
            """)
            
        with col3:
            market_heat = market_indicators['market_heat']
            heat_color = "red" if market_heat > 70 else "orange" if market_heat > 40 else "green"
            st.markdown(f"""
            **Market Heat:** 
            <span style="color: {heat_color}; font-weight: bold;">
            {market_heat:.0f}/100
            </span>
            """, unsafe_allow_html=True)
        
        # Price analysis section
        st.markdown("---")
        st.subheader("📈 Price Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Price chart
            price_chart = create_price_chart(price_data, "Electricity Price Trend (30 Days)")
            st.plotly_chart(price_chart, use_container_width=True)
            
        with col2:
            # Price statistics
            st.markdown("**📊 Price Statistics**")
            
            price_stats = {
                "Current": f"€{market_data['current_power_price']:.1f}",
                "30-Day Average": f"€{market_data['avg_power_price_30d']:.1f}",
                "Volatility": f"€{market_data['power_price_volatility']:.1f}",
                "Min (30d)": f"€{price_data['price_eur_mwh'].min():.1f}",
                "Max (30d)": f"€{price_data['price_eur_mwh'].max():.1f}"
            }
            
            for metric, value in price_stats.items():
                st.markdown(f"**{metric}:** {value}")
        
        # Spark spread analysis
        st.markdown("---")
        st.subheader("⚡ Spark Spread Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create synthetic gas price data for chart
            gas_prices = np.full(len(price_data), market_data['gas_price'])
            spark_chart = create_spark_spread_chart(
                price_data['price_eur_mwh'].values,
                gas_prices
            )
            st.plotly_chart(spark_chart, use_container_width=True)
            
        with col2:
            st.markdown("**⚡ Spark Spread Metrics**")
            
            # Calculate different efficiency scenarios
            efficiencies = [0.35, 0.50, 0.58, 0.65]  # Old to new plants
            
            for eff in efficiencies:
                spread = calculate_spark_spread(
                    market_data['current_power_price'],
                    market_data['gas_price'],
                    efficiency=eff
                )
                plant_type = "Old" if eff < 0.45 else "Average" if eff < 0.60 else "Efficient"
                color = "red" if spread < 0 else "green"
                
                st.markdown(f"""
                **{plant_type} Plant** (η={eff:.0%}):  
                <span style="color: {color};">€{spread:.1f}/MWh</span>
                """, unsafe_allow_html=True)
        
        # Price pattern analysis
        st.markdown("---")
        st.subheader("🔍 Price Pattern Analysis")
        
        # Analyze trends
        trends = analyze_price_trends(price_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern chart
            st.markdown("**Hourly Price Pattern**")
            hourly_df = pd.DataFrame({
                'Hour': trends['hourly_avg'].index,
                'Average Price': trends['hourly_avg'].values
            })
            st.bar_chart(hourly_df.set_index('Hour'))
            
        with col2:
            # Daily pattern
            st.markdown("**Weekly Price Pattern**")
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            weekly_df = pd.DataFrame({
                'Day': days,
                'Average Price': trends['weekly_pattern'].values
            })
            st.bar_chart(weekly_df.set_index('Day'))
        
        # Correlation analysis
        st.markdown("---")
        st.subheader("🔗 Price Correlation Analysis")
        
        # Create correlation matrix chart
        correlation_chart = create_correlation_matrix({})
        st.plotly_chart(correlation_chart, use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.subheader("💡 Key Market Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔥 Current Conditions:**")
            insights = []
            
            if market_data['current_power_price'] > market_data['avg_power_price_30d']:
                insights.append("• Power prices above 30-day average")
            else:
                insights.append("• Power prices below 30-day average")
                
            if spark_spread > 10:
                insights.append("• Positive spark spreads favor gas generation")
            elif spark_spread > 0:
                insights.append("• Marginal spark spreads for gas plants")
            else:
                insights.append("• Negative spark spreads challenging for gas plants")
                
            if market_data['power_price_volatility'] > 20:
                insights.append("• High price volatility indicates market stress")
            else:
                insights.append("• Moderate price volatility")
                
            for insight in insights:
                st.markdown(insight)
        
        with col2:
            st.markdown("**📈 Trading Opportunities:**")
            opportunities = []
            
            if market_indicators['market_heat'] > 70:
                opportunities.append("• Consider hedging high prices")
                opportunities.append("• Optimize flexible asset dispatch")
            elif market_indicators['market_heat'] < 30:
                opportunities.append("• Opportunity for strategic purchases")
                opportunities.append("• Consider maintenance scheduling")
            else:
                opportunities.append("• Balanced market conditions")
                opportunities.append("• Monitor for regime changes")
                
            for opp in opportunities:
                st.markdown(opp)
        
        # Data refresh info
        st.markdown("---")
        st.info(f"📊 **Data last updated:** {market_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | **Refresh interval:** 5 minutes")
        
    except Exception as e:
        st.error(f"❌ Error loading market data: {str(e)}")
        st.info("💡 **Tip:** Ensure API keys are configured in environment variables for live data access.")
        
        # Show sample data structure
        with st.expander("🔧 Sample Data Structure"):
            st.code("""
            # Expected data structure:
            market_data = {
                'current_power_price': 65.5,
                'avg_power_price_30d': 62.3,
                'gas_price': 45.2,
                'carbon_price': 85.7,
                'timestamp': datetime.now()
            }
            """)

if __name__ == "__main__":
    show_market_overview()