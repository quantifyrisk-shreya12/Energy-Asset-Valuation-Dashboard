"""
Scenario Planning Dashboard Page
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import our modules
from data.sample_assets import get_sample_assets
from data.market_data import get_market_overview
from models.market_analysis import forecast_prices
from models.financial_models import calculate_asset_dcf
from models.optimization import scenario_optimization, optimize_dispatch_schedule, calculate_hedging_strategy
from utils.visualization import create_scenario_analysis_chart, create_price_chart
from utils.calculations import format_currency, format_percentage

def show_scenarios():
    """
    Display scenario planning dashboard
    """
    st.title("🎯 Scenario Planning & Strategy")
    
    # Load data
    try:
        assets_df = get_sample_assets()
        market_data, price_data = get_market_overview()
        
        # Scenario selection
        st.subheader("📋 Scenario Configuration")
        
        scenario_type = st.selectbox(
            "Select Analysis Type:",
            ["Market Scenarios", "Price Forecasting", "Portfolio Optimization", "Hedging Strategy"]
        )
        
        st.markdown("---")
        
        if scenario_type == "Market Scenarios":
            show_market_scenarios(assets_df, market_data)
            
        elif scenario_type == "Price Forecasting":
            show_price_forecasting(price_data, assets_df)
            
        elif scenario_type == "Portfolio Optimization":
            show_portfolio_optimization(assets_df, market_data)
            
        elif scenario_type == "Hedging Strategy":
            show_hedging_strategy(assets_df, market_data, price_data)
        
    except Exception as e:
        import traceback
        import sys
        
        # Get full traceback
        tb = traceback.format_exc()
        st.error("❌ Error in scenario analysis")
        
        # Show exact line and file
        st.error(f"**Error Type:** {type(e).__name__}")
        st.error(f"**Error Message:** {str(e)}")
        
        # Extract line number from traceback
        tb_lines = tb.split('\n')
        relevant_line = None
        for line in tb_lines:
            if 'scenarios.py' in line:
                relevant_line = line.strip()
                break
        
        if relevant_line:
            st.error(f"**Location:** {relevant_line}")
        
        # Show full traceback in expander
        with st.expander("🔍 View Full Error Details"):
            st.code(tb, language='python')
        
        st.info("💡 **Tip:** Check the error location above for specific debugging")

def show_market_scenarios(assets_df, market_data):
    """
    Show market scenario analysis
    """
    st.subheader("🌍 Market Scenario Analysis")
    
    # Define scenarios
    scenarios = {
        "Base Case": {
            'power_price': market_data['current_power_price'],
            'gas_price': market_data['gas_price'],
            'carbon_price': market_data['carbon_price'],
            'probability': 0.4
        },
        "High Price": {
            'power_price': market_data['current_power_price'] * 1.3,
            'gas_price': market_data['gas_price'] * 1.2,
            'carbon_price': market_data['carbon_price'] * 1.4,
            'probability': 0.2
        },
        "Low Price": {
            'power_price': market_data['current_power_price'] * 0.7,
            'gas_price': market_data['gas_price'] * 0.8,
            'carbon_price': market_data['carbon_price'] * 0.6,
            'probability': 0.25
        },
        "Green Transition": {
            'power_price': market_data['current_power_price'] * 0.9,
            'gas_price': market_data['gas_price'] * 1.1,
            'carbon_price': market_data['carbon_price'] * 2.0,
            'probability': 0.15
        }
    }
    
    # Scenario configuration
    st.markdown("**⚙️ Scenario Parameters**")
    
    scenario_df = pd.DataFrame(scenarios).T
    scenario_df['power_price'] = scenario_df['power_price'].round(1)
    scenario_df['gas_price'] = scenario_df['gas_price'].round(1)
    scenario_df['carbon_price'] = scenario_df['carbon_price'].round(1)
    scenario_df['probability'] = scenario_df['probability'].apply(lambda x: f"{x:.1%}")
    
    scenario_df.columns = ['Power Price (€/MWh)', 'Gas Price (€/MWh)', 'Carbon Price (€/t)', 'Probability']
    st.dataframe(scenario_df, use_container_width=True)
    
    # Calculate portfolio value under each scenario
    st.markdown("---")
    st.subheader("💰 Portfolio Valuation by Scenario")

    scenario_results = []

    for scenario_name, scenario_params in scenarios.items():
        try:
            portfolio_npv = 0
            asset_values = []
            
            # Calculate NPV for each asset under this scenario
            for _, asset in assets_df.iterrows():
                try:
                    dcf_result = calculate_asset_dcf(asset, scenario_params)
                    if dcf_result and 'npv' in dcf_result:
                        portfolio_npv += dcf_result['npv']
                        asset_values.append(dcf_result['npv'])
                    else:
                        st.warning(f"Missing NPV for {asset['name']} in {scenario_name}")
                        asset_values.append(0)
                except Exception as e:
                    st.error(f"Error calculating DCF for {asset['name']}: {str(e)}")
                    asset_values.append(0)
            
            # Ensure portfolio_value is always calculated
            scenario_results.append({
                'scenario_name': scenario_name,
                'portfolio_value': portfolio_npv / 1e6,  # Convert to millions
                'weight': scenario_params['probability'],
                'asset_values': asset_values
            })
            
        except Exception as e:
            st.error(f"Error processing scenario {scenario_name}: {str(e)}")
            scenario_results.append({
                'scenario_name': scenario_name,
                'portfolio_value': 0,
                'weight': scenario_params['probability'],
                'asset_values': [0] * len(assets_df)
            })

    # Verify scenario results structure
    if scenario_results:
        with st.expander("🔍 Debug: Scenario Results"):
            for i, result in enumerate(scenario_results):
                st.write(f"**{result['scenario_name']}**: €{result['portfolio_value']:.1f}M (weight: {result['weight']})")

    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scenario chart
        scenario_chart = create_scenario_analysis_chart(scenario_results)
        st.plotly_chart(scenario_chart, use_container_width=True)
    
    with col2:
        st.markdown("**📊 Scenario Summary**")
        
        for result in scenario_results:
            value = result['portfolio_value']
            prob = result['weight']
            color = "green" if value > 0 else "red"
            
            st.markdown(f"""
            **{result['scenario_name']}**  
            Value: <span style="color: {color};">€{value:.0f}M</span>  
            Prob: {prob:.1%}
            """, unsafe_allow_html=True)
    
    # Expected value calculation
    expected_value = sum(r['portfolio_value'] * r['weight'] for r in scenario_results)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expected Portfolio Value", f"€{expected_value:.0f}M")
    
    with col2:
        best_case = max(r['portfolio_value'] for r in scenario_results)
        st.metric("Best Case", f"€{best_case:.0f}M")
    
    with col3:
        worst_case = min(r['portfolio_value'] for r in scenario_results)
        st.metric("Worst Case", f"€{worst_case:.0f}M")
    
    # Strategic insights
    st.markdown("---")
    st.subheader("🎯 Strategic Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Key Observations:**")
        
        if expected_value > 0:
            st.markdown("• Portfolio shows positive expected value across scenarios")
        else:
            st.markdown("• Portfolio faces headwinds in current market outlook")

        # ✅ FIXED: Use scenario_results instead of scenarios
        try:
            green_scenario = next(r for r in scenario_results if r['scenario_name'] == 'Green Transition')
            base_scenario = next(r for r in scenario_results if r['scenario_name'] == 'Base Case')
            high_scenario = next(r for r in scenario_results if r['scenario_name'] == 'High Price')
            low_scenario = next(r for r in scenario_results if r['scenario_name'] == 'Low Price')

            high_carbon_impact = green_scenario['portfolio_value'] - base_scenario['portfolio_value']
            if abs(high_carbon_impact) > 100:
                st.markdown("• Significant exposure to carbon price risk")
            
            price_sensitivity = high_scenario['portfolio_value'] - low_scenario['portfolio_value']
            st.markdown(f"• Portfolio value range: €{price_sensitivity:.0f}M across price scenarios")
        except StopIteration:
            st.markdown("• Scenario data incomplete for strategic analysis")
    
    with col2:
        st.markdown("**📈 Recommendations:**")
        
        if worst_case < -500:
            st.markdown("• Consider downside protection strategies")
        
        try:
            green_scenario = next(r for r in scenario_results if r['scenario_name'] == 'Green Transition')
            base_scenario = next(r for r in scenario_results if r['scenario_name'] == 'Base Case')
            
            if green_scenario['portfolio_value'] < base_scenario['portfolio_value']:
                st.markdown("• Evaluate renewable energy investments")
        except StopIteration:
            pass
        
        st.markdown("• Monitor carbon price developments closely")
        st.markdown("• Consider scenario-based hedging strategies")

def show_price_forecasting(price_data, assets_df):
    """
    Show price forecasting and impact analysis
    """
    st.subheader("📈 Price Forecasting & Impact")
    
    # Forecast parameters
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    
    with col2:
        forecast_method = st.selectbox("Forecast Method", ["Linear Trend", "Moving Average", "Seasonal"])
    
    # Generate price forecast
    with st.spinner("Generating price forecast..."):
        price_forecast = forecast_prices(price_data, forecast_days)
    
    # Display forecast
    st.markdown("**📊 Price Forecast**")
    
    # Combine historical and forecast data
    historical = price_data.copy()
    historical['forecast'] = False
    
    combined_data = pd.concat([historical, price_forecast], ignore_index=True)
    
    forecast_chart = create_price_chart(combined_data, f"Electricity Price Forecast ({forecast_days} days)")
    st.plotly_chart(forecast_chart, use_container_width=True)
    
    # Forecast statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_forecast = price_forecast['price_eur_mwh'].mean()
        st.metric("Average Forecast", f"€{avg_forecast:.1f}/MWh")
    
    with col2:
        current_price = price_data['price_eur_mwh'].iloc[-1]
        price_change = avg_forecast - current_price
        st.metric("vs Current", f"€{price_change:+.1f}/MWh")
    
    with col3:
        forecast_volatility = price_forecast['price_eur_mwh'].std()
        st.metric("Forecast Volatility", f"€{forecast_volatility:.1f}/MWh")
    
    with col4:
        max_forecast = price_forecast['price_eur_mwh'].max()
        st.metric("Peak Forecast", f"€{max_forecast:.1f}/MWh")
    
    # Impact on dispatch
    st.markdown("---")
    st.subheader("⚡ Dispatch Impact Analysis")
    
    # Calculate dispatch schedule
    dispatch_schedule = optimize_dispatch_schedule(assets_df, price_forecast)
    
    if not dispatch_schedule.empty:
        # Dispatch summary
        total_dispatch_profit = dispatch_schedule['profit'].sum()
        avg_hourly_profit = dispatch_schedule.groupby('timestamp')['profit'].sum().mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Forecast Profit", format_currency(total_dispatch_profit))
            st.metric("Avg Hourly Profit", format_currency(avg_hourly_profit))
        
        with col2:
            # Asset utilization
            asset_utilization = dispatch_schedule.groupby('asset').size()
            most_utilized = asset_utilization.idxmax() if not asset_utilization.empty else "N/A"
            utilization_hours = asset_utilization.max() if not asset_utilization.empty else 0
            
            st.metric("Most Utilized Asset", most_utilized)
            st.metric("Utilization Hours", f"{utilization_hours}/{forecast_days * 24}")
    
    else:
        st.warning("No profitable dispatch opportunities identified in forecast period.")

def show_portfolio_optimization(assets_df, market_data):
    """
    Show portfolio optimization analysis
    """
    st.subheader("🎯 Portfolio Optimization")
    
    # Optimization objectives
    st.markdown("**🎯 Optimization Objectives**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_objective = st.selectbox(
            "Primary Objective:",
            ["Maximize NPV", "Maximize IRR", "Minimize Risk", "Maximize Sharpe Ratio"]
        )
    
    with col2:
        risk_tolerance = st.selectbox(
            "Risk Tolerance:",
            ["Conservative", "Moderate", "Aggressive"]
        )
    
    # Portfolio metrics
    st.markdown("---")
    st.subheader("📊 Current Portfolio Metrics")
    
    # Calculate current portfolio performance
    current_valuations = []
    total_capacity = assets_df['capacity_mw'].sum()
    
    for _, asset in assets_df.iterrows():
        dcf_result = calculate_asset_dcf(asset, market_data)
        current_valuations.append({
            'asset': asset['name'],
            'type': asset['type'],
            'capacity_mw': asset['capacity_mw'],
            'npv': dcf_result['npv'],
            'irr': dcf_result['irr'] if not np.isnan(dcf_result['irr']) else 0
        })
    
    # Portfolio composition
    type_composition = assets_df.groupby('type')['capacity_mw'].sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏭 Technology Mix**")
        for tech, capacity in type_composition.items():
            pct = capacity / total_capacity * 100
            st.markdown(f"• **{tech}:** {capacity:,.0f} MW ({pct:.1f}%)")
    
    with col2:
        st.markdown("**💰 Value Distribution**")
        total_npv = sum(v['npv'] for v in current_valuations)
        
        for val in sorted(current_valuations, key=lambda x: x['npv'], reverse=True)[:5]:
            pct = val['npv'] / total_npv * 100 if total_npv != 0 else 0
            st.markdown(f"• **{val['asset']}:** {pct:.1f}% of portfolio value")
    
    # Optimization recommendations
    st.markdown("---")
    st.subheader("🎯 Optimization Recommendations")
    
    # Analyze portfolio gaps and opportunities
    gas_share = type_composition.get('Gas', 0) / total_capacity
    renewable_share = (type_composition.get('Wind', 0) + type_composition.get('Solar', 0)) / total_capacity
    
    recommendations = []
    
    if gas_share > 0.7:
        recommendations.append("🔸 **Diversification**: High gas concentration increases commodity price risk")
    
    if renewable_share < 0.3:
        recommendations.append("🔸 **Renewables**: Consider increasing renewable capacity for stable margins")
    
    # Age analysis
    avg_age = 2024 - assets_df['construction_year'].mean()
    if avg_age > 15:
        recommendations.append("🔸 **Modernization**: Aging fleet may benefit from efficiency upgrades")
    
    # Efficiency analysis
    gas_assets = assets_df[assets_df['type'] == 'Gas']
    if not gas_assets.empty:
        efficiency_spread = gas_assets['efficiency'].max() - gas_assets['efficiency'].min()
        if efficiency_spread > 0.1:
            recommendations.append("🔸 **Efficiency Focus**: Prioritize high-efficiency assets in merit order")
    
    # Market conditions
    current_spark_spread = market_data['current_power_price'] - market_data['gas_price'] / 0.58
    if current_spark_spread < 5:
        recommendations.append("🔸 **Market Timing**: Consider maintenance during low spark spread periods")
    
    for rec in recommendations[:6]:
        st.markdown(rec)
    
    # Strategic scenarios
    st.markdown("---")
    st.subheader("🔮 Strategic Scenarios")
    
    strategic_scenarios = {
        "Current Portfolio": {"description": "Maintain current asset mix", "score": 7.0},
        "Renewable Focus": {"description": "Increase wind/solar to 50% of capacity", "score": 8.5},
        "Efficiency Upgrade": {"description": "Upgrade gas plants to >60% efficiency", "score": 7.8},
        "Flexible Portfolio": {"description": "Add battery storage and peaking assets", "score": 8.2}
    }
    
    scenario_df = pd.DataFrame.from_dict(strategic_scenarios, orient='index')
    scenario_df['Score'] = scenario_df['score'].apply(lambda x: f"{x:.1f}/10")
    scenario_df = scenario_df[['description', 'Score']]
    scenario_df.columns = ['Strategy Description', 'Strategic Score']
    
    st.dataframe(scenario_df, use_container_width=True)

def show_hedging_strategy(assets_df, market_data, price_data):
    """
    Show hedging strategy recommendations
    """
    st.subheader("🛡️ Hedging Strategy")
    
    # Hedging parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hedge_ratio = st.slider("Hedge Ratio", 0.0, 1.0, 0.5, 0.05)
    
    with col2:
        hedge_horizon = st.selectbox("Hedge Horizon", ["1 Month", "3 Months", "6 Months", "1 Year"])
    
    with col3:
        hedge_instruments = st.multiselect(
            "Hedge Instruments",
            ["Power Forwards", "Gas Forwards", "Carbon Forwards", "Options"],
            default=["Power Forwards"]
        )
    
    # Calculate hedging recommendations
    hedge_recommendations, portfolio_hedge = calculate_hedging_strategy(assets_df, price_data, hedge_ratio)
    
    # Hedging summary
    st.markdown("---")
    st.subheader("📋 Hedging Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Generation Forecast", f"{portfolio_hedge['total_generation_forecast']:,.0f} MWh")
    
    with col2:
        st.metric("Recommended Hedge Volume", f"{portfolio_hedge['total_recommended_hedge']:,.0f} MWh")
    
    with col3:
        st.metric("Estimated Hedge Value", format_currency(portfolio_hedge['total_hedge_value'], millions=True))
    
    # Asset-level hedging
    st.markdown("**🏭 Asset-Level Hedging Recommendations**")
    
    hedge_df = pd.DataFrame(hedge_recommendations)
    hedge_df['generation_forecast_mwh'] = hedge_df['generation_forecast_mwh'].round(0)
    hedge_df['recommended_hedge_volume_mwh'] = hedge_df['recommended_hedge_volume_mwh'].round(0)
    hedge_df['recommended_hedge_price'] = hedge_df['recommended_hedge_price'].round(1)
    hedge_df['hedge_value'] = hedge_df['hedge_value'].round(0)
    
    hedge_df.columns = ['Asset', 'Generation (MWh)', 'Hedge Volume (MWh)', 'Hedge Price (€/MWh)', 'Hedge Value (€)']
    st.dataframe(hedge_df, hide_index=True, use_container_width=True)
    
    # Risk analysis
    st.markdown("---")
    st.subheader("⚠️ Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Price Risk Exposure**")
        
        price_volatility = price_data['price_eur_mwh'].std()
        annual_generation = sum(h['generation_forecast_mwh'] for h in hedge_recommendations)
        unhedged_volume = annual_generation * (1 - hedge_ratio)
        
        # Value at Risk calculation (simplified)
        daily_var = price_volatility * np.sqrt(252) * unhedged_volume * 0.025  # 2.5% daily VaR
        
        st.markdown(f"""
        • **Price Volatility:** €{price_volatility:.1f}/MWh
        • **Unhedged Volume:** {unhedged_volume:,.0f} MWh
        • **Daily VaR (95%):** {format_currency(daily_var)}
        • **Annual VaR:** {format_currency(daily_var * 16)}
        """)
    
    with col2:
        st.markdown("**🎯 Hedging Benefits**")
        
        hedge_effectiveness = min(95, hedge_ratio * 100 + 20)
        revenue_stability = 60 + hedge_ratio * 30
        
        st.markdown(f"""
        • **Risk Reduction:** ~{hedge_effectiveness:.0f}%
        • **Revenue Stability:** {revenue_stability:.0f}%
        • **Cash Flow Predictability:** Improved
        • **Credit Profile:** Enhanced
        """)
    
    # Strategic recommendations
    st.markdown("---")
    st.subheader("💡 Strategic Hedging Recommendations")
    
    hedging_insights = []
    
    if hedge_ratio < 0.3:
        hedging_insights.append("🔸 **Low Hedge Ratio**: Consider increasing hedge coverage to reduce volatility")
    elif hedge_ratio > 0.8:
        hedging_insights.append("🔸 **High Hedge Ratio**: May limit upside participation in favorable markets")
    
    current_price = market_data['current_power_price']
    avg_historical = price_data['price_eur_mwh'].mean()
    
    if current_price > avg_historical * 1.2:
        hedging_insights.append("🔸 **High Prices**: Favorable environment for forward sales")
    elif current_price < avg_historical * 0.8:
        hedging_insights.append("🔸 **Low Prices**: Consider reducing hedge ratios or using options")
    
    hedging_insights.append("🔸 **Dynamic Hedging**: Adjust ratios based on market conditions and forecasts")
    hedging_insights.append("🔸 **Multi-Commodity**: Consider cross-commodity hedges (power vs gas)")
    
    for insight in hedging_insights:
        st.markdown(insight)

if __name__ == "__main__":
    show_scenarios()