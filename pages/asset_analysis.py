"""
Asset Analysis Dashboard Page
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import our modules
from data.sample_assets import get_sample_assets, get_fuel_prices
from data.market_data import get_market_overview, calculate_merit_order_position, calculate_dispatch_economics
from utils.visualization import (create_asset_performance_dashboard, create_merit_order_chart, 
                               create_portfolio_pie_chart, create_dispatch_optimization_chart)
from utils.calculations import format_currency, calculate_capacity_factor


def show_asset_analysis():
    """
    Display asset analysis dashboard
    """
    st.title("🏭 Asset Performance Analysis")
    
    # Load data
    try:
        assets_df = get_sample_assets()
        market_data, price_data = get_market_overview()
        fuel_prices = get_fuel_prices()
        
        # Asset overview
        st.subheader("🏢 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_capacity = assets_df['capacity_mw'].sum()
            st.metric("Total Capacity", f"{total_capacity:,.0f} MW")
            
        with col2:
            gas_capacity = assets_df[assets_df['type'] == 'Gas']['capacity_mw'].sum()
            st.metric("Gas Assets", f"{gas_capacity:,.0f} MW")
            
        with col3:
            renewable_capacity = assets_df[assets_df['type'].isin(['Wind', 'Solar'])]['capacity_mw'].sum()
            st.metric("Renewables", f"{renewable_capacity:,.0f} MW")
            
        with col4:
            avg_age = 2024 - assets_df['construction_year'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        
        # Asset selection
        st.markdown("---")
        st.subheader("🎯 Asset Deep Dive")
        
        selected_asset = st.selectbox(
            "Select Asset for Detailed Analysis:",
            assets_df['name'].tolist(),
            index=0
        )
        
        asset_details = assets_df[assets_df['name'] == selected_asset].iloc[0]
        
        # Asset details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔧 Technical Specifications**")
            st.markdown(f"**Type:** {asset_details['type']}")
            st.markdown(f"**Capacity:** {asset_details['capacity_mw']:,.0f} MW")
            st.markdown(f"**Efficiency:** {asset_details['efficiency']:.1%}")
            st.markdown(f"**Fuel:** {asset_details['fuel_type']}")
            
        with col2:
            st.markdown("**💰 Economic Parameters**")
            st.markdown(f"**Fixed O&M:** €{asset_details['fixed_cost']:,.0f}/MW/year")
            st.markdown(f"**Variable O&M:** €{asset_details['variable_cost']:.1f}/MWh")
            st.markdown(f"**CO₂ Intensity:** {asset_details['co2_intensity']:.2f} tCO₂/MWh")
            st.markdown(f"**Year Built:** {asset_details['construction_year']}")
            
        with col3:
            st.markdown("**📊 Performance Metrics**")
            st.markdown(f"**Availability:** {asset_details['availability']:.1%}")
            
            annual_generation = (asset_details['capacity_mw'] * 8760 * 
                               asset_details['availability'])
            st.markdown(f"**Annual Generation:** {annual_generation:,.0f} MWh")
            
            load_factor = asset_details['availability']
            st.markdown(f"**Load Factor:** {load_factor:.1%}")
            
            age = 2024 - asset_details['construction_year']
            st.markdown(f"**Asset Age:** {age} years")
        
        # Portfolio composition charts
        st.markdown("---")
        st.subheader("📊 Portfolio Composition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Capacity by type
            capacity_chart = create_portfolio_pie_chart(
                assets_df, 'capacity_mw', "Capacity Distribution by Asset Type"
            )
            st.plotly_chart(capacity_chart, use_container_width=True)
            
        with col2:
            # Technology breakdown
            tech_summary = assets_df.groupby('type').agg({
                'capacity_mw': 'sum',
                'name': 'count'
            }).rename(columns={'name': 'count'})
            
            st.markdown("**Technology Summary**")
            for tech, data in tech_summary.iterrows():
                pct_capacity = data['capacity_mw'] / total_capacity * 100
                st.markdown(f"**{tech}:** {data['capacity_mw']:,.0f} MW ({pct_capacity:.1f}%) - {data['count']} assets")
        
        # Merit order analysis
        st.markdown("---")
        st.subheader("📈 Merit Order Analysis")
        
        # Calculate merit order
        merit_order = calculate_merit_order_position(assets_df, market_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            merit_chart = create_merit_order_chart(merit_order)
            st.plotly_chart(merit_chart, use_container_width=True)
            
        with col2:
            st.markdown("**Merit Order Position**")
            st.dataframe(
                merit_order[['name', 'type', 'marginal_cost']].round(1),
                hide_index=True
            )
        
        # Dispatch economics
        st.markdown("---")
        st.subheader("⚡ Current Dispatch Economics")
        
        dispatch_results = calculate_dispatch_economics(assets_df, market_data)

        print(dispatch_results)

        
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Dispatch profitability chart
            dispatch_chart = create_dispatch_optimization_chart(dispatch_results)
            st.plotly_chart(dispatch_chart, use_container_width=True)
        
        print("i am in line 153")

        with col2:
            st.markdown("**Current Dispatch Status**")
            
            total_dispatched = dispatch_results['should_dispatch'].sum()
            total_capacity_dispatched = dispatch_results[dispatch_results['should_dispatch']]['capacity_mw'].sum()
            
            st.metric("Assets Dispatched", f"{total_dispatched}/{len(dispatch_results)}")
            st.metric("Capacity Dispatched", f"{total_capacity_dispatched:,.0f} MW")
            
            hourly_profit = dispatch_results['hourly_profit'].sum()
            st.metric("Portfolio Hourly Profit", format_currency(hourly_profit))
        
        # Dispatch details table
        st.markdown("**📋 Detailed Dispatch Analysis**")
        
        display_cols = ['name', 'type', 'marginal_cost', 'profit_margin', 'should_dispatch', 'hourly_profit']
        dispatch_display = dispatch_results[display_cols].copy()
        dispatch_display['marginal_cost'] = dispatch_display['marginal_cost'].round(1)
        dispatch_display['profit_margin'] = dispatch_display['profit_margin'].round(1)
        dispatch_display['hourly_profit'] = dispatch_display['hourly_profit'].round(0)
        dispatch_display['should_dispatch'] = dispatch_display['should_dispatch'].map({True: '✅ Yes', False: '❌ No'})
        
        st.dataframe(dispatch_display, hide_index=True, use_container_width=True)
        
        # Performance analytics
        st.markdown("---")
        st.subheader("📈 Performance Analytics")
        
        # Create performance dashboard
        performance_metrics = {}  # Placeholder for additional metrics
        performance_chart = create_asset_performance_dashboard(assets_df, performance_metrics)
        st.plotly_chart(performance_chart, use_container_width=True)
        
        # Asset comparison table
        st.markdown("---")
        st.subheader("📊 Asset Comparison Matrix")
        
        # Create comprehensive comparison table
        comparison_data = assets_df.copy()
        comparison_data['annual_generation_gwh'] = (comparison_data['capacity_mw'] * 8760 * 
                                                   comparison_data['availability'] / 1000)
        comparison_data['co2_emissions_kt'] = (comparison_data['annual_generation_gwh'] * 1000 * 
                                             comparison_data['co2_intensity'] / 1000)
        
        # Calculate current economics
        current_power_price = market_data['current_power_price']
        
        comparison_data['annual_revenue_m'] = (comparison_data['annual_generation_gwh'] * 1000 * 
                                             current_power_price / 1e6)
        
        # Display comparison
        comparison_display = comparison_data[[
            'name', 'type', 'capacity_mw', 'efficiency', 'availability',
            'annual_generation_gwh', 'annual_revenue_m', 'co2_emissions_kt'
        ]].round(2)
        
        comparison_display.columns = [
            'Asset', 'Type', 'Capacity (MW)', 'Efficiency', 'Availability',
            'Generation (GWh/year)', 'Revenue (€M/year)', 'CO₂ (kt/year)'
        ]
        
        st.dataframe(comparison_display, hide_index=True, use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.subheader("💡 Key Asset Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Top Performers:**")
            
            # Most profitable assets
            top_profitable = dispatch_results.nlargest(3, 'hourly_profit')
            for _, asset in top_profitable.iterrows():
                if asset['hourly_profit'] > 0:
                    st.markdown(f"• **{asset['name']}**: €{asset['hourly_profit']:,.0f}/hour profit")
            
            # Highest efficiency
            most_efficient = assets_df.nlargest(2, 'efficiency')
            st.markdown("\n**Highest Efficiency:**")
            for _, asset in most_efficient.iterrows():
                st.markdown(f"• **{asset['name']}**: {asset['efficiency']:.1%} efficient")
                
        with col2:
            st.markdown("**⚠️ Areas for Attention:**")
            
            # Assets not dispatching
            not_dispatched = dispatch_results[~dispatch_results['should_dispatch']]
            if not not_dispatched.empty:
                st.markdown("**Assets Out of Merit:**")
                for _, asset in not_dispatched.iterrows():
                    margin = asset['profit_margin']
                    st.markdown(f"• **{asset['name']}**: €{margin:.1f}/MWh margin")
            
            # Older assets
            old_assets = assets_df[assets_df['construction_year'] < 2010]
            if not old_assets.empty:
                st.markdown("\n**Aging Assets (>14 years):**")
                for _, asset in old_assets.iterrows():
                    age = 2024 - asset['construction_year']
                    st.markdown(f"• **{asset['name']}**: {age} years old")
        
        # Recommendations
        st.markdown("---")
        st.subheader("🎯 Strategic Recommendations")
        
        recommendations = []
        
        # Check dispatch status
        if hourly_profit < 0:
            recommendations.append("🔸 **Market Conditions**: Current market conditions challenging for thermal generation")
        elif hourly_profit > 100000:
            recommendations.append("🔸 **High Margins**: Excellent market conditions - maximize generation")
            
        # Check renewable vs thermal mix
        renewable_share = renewable_capacity / total_capacity
        if renewable_share < 0.3:
            recommendations.append("🔸 **Portfolio Balance**: Consider increasing renewable capacity for better margin stability")
            
        # Check efficiency spread
        gas_assets = assets_df[assets_df['type'] == 'Gas']
        if not gas_assets.empty:
            efficiency_range = gas_assets['efficiency'].max() - gas_assets['efficiency'].min()
            if efficiency_range > 0.1:
                recommendations.append("🔸 **Efficiency Gap**: Significant efficiency differences - prioritize high-efficiency assets")
        
        # Age analysis
        avg_age = 2024 - assets_df['construction_year'].mean()
        if avg_age > 15:
            recommendations.append("🔸 **Asset Age**: Portfolio averaging >15 years - consider modernization strategies")
            
        for rec in recommendations[:4]:  # Show top 4 recommendations
            st.markdown(rec)
        
        # Data refresh info
        st.markdown("---")
        st.info(f"📊 **Analysis based on current market price:** €{current_power_price:.1f}/MWh | **Last updated:** {market_data['timestamp'].strftime('%H:%M:%S')}")
        
    except Exception as e:
        st.error(f"❌ Error loading asset data: {str(e)}")
        st.info("💡 **Tip:** Check data connections and try refreshing the page.")
        
        # Show sample data
        with st.expander("🔧 Sample Asset Data Structure"):
            st.code("""
            # Expected asset data structure:
            assets_df = pd.DataFrame([{
                'name': 'Power Plant Name',
                'type': 'Gas',
                'capacity_mw': 1000,
                'efficiency': 0.58,
                'availability': 0.92,
                'variable_cost': 3.5,
                'fixed_cost': 45000,
                'co2_intensity': 0.35,
                'construction_year': 2015,
                'country': 'Germany',
                'fuel_type': 'Natural Gas'
            }])
            """)

if __name__ == "__main__":
    show_asset_analysis()