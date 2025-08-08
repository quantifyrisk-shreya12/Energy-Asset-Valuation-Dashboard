"""
Valuation Models Dashboard Page
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import our modules
from data.sample_assets import get_sample_assets
from data.market_data import get_market_overview
from models.financial_models import calculate_asset_dcf, monte_carlo_valuation, sensitivity_analysis, calculate_lcoe
from utils.visualization import create_monte_carlo_chart, create_sensitivity_chart, create_valuation_waterfall
from utils.calculations import format_currency, format_percentage

def show_valuation_models():
    """
    Display valuation models dashboard
    """
    st.title("💰 Asset Valuation Models")
    
    # Load data
    try:
        assets_df = get_sample_assets()
        market_data, price_data = get_market_overview()
        
        # Model selection
        st.subheader("🎯 Valuation Model Selection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_asset = st.selectbox(
                "Select Asset for Valuation:",
                assets_df['name'].tolist(),
                index=0
            )
        
        with col2:
            valuation_method = st.selectbox(
                "Valuation Method:",
                ["DCF Analysis", "Monte Carlo", "Sensitivity Analysis", "LCOE Analysis"]
            )
        
        # Get selected asset data
        asset_data = assets_df[assets_df['name'] == selected_asset].iloc[0]
        
        # Market assumptions section
        st.markdown("---")
        st.subheader("📊 Market Assumptions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            power_price_assumption = st.number_input(
                "Power Price (EUR/MWh)",
                min_value=20.0,
                max_value=200.0,
                value=float(market_data['current_power_price']),
                step=1.0
            )
            
        with col2:
            # gas_price_assumption = st.number_input(
            #     "Gas Price (EUR/MWh)",
            #     min_value=10.0,
            #     max_value=100.0,
            #     value=float(market_data['gas_price']),
            #     step=1.0
            # )

            # ✅ NEW (fixed range):
            gas_price_assumption = st.number_input(
                "Gas Price (EUR/MWh)",
                min_value=10.0,
                max_value=500.0,      # 🔄 Increased to handle current market prices
                value=min(float(market_data['gas_price']), 500.0),  # 🔄 Ensure value within range
                step=1.0
            )
            
        with col3:
            carbon_price_assumption = st.number_input(
                "Carbon Price (EUR/t)",
                min_value=10.0,
                max_value=200.0,
                value=float(market_data['carbon_price']),
                step=1.0
            )
            
        with col4:
            discount_rate = st.number_input(
                "Discount Rate (%)",
                min_value=3.0,
                max_value=15.0,
                value=7.0,
                step=0.1
            ) / 100
        
        # Create market assumptions dictionary
        market_assumptions = {
            'power_price': power_price_assumption,
            'gas_price': gas_price_assumption,
            'carbon_price': carbon_price_assumption,
            'discount_rate': discount_rate
        }
        
        st.markdown("---")
        
        # Display different valuation methods
        if valuation_method == "DCF Analysis":
            show_dcf_analysis(asset_data, market_assumptions)
            
        elif valuation_method == "Monte Carlo":
            show_monte_carlo_analysis(asset_data, market_assumptions)
            
        elif valuation_method == "Sensitivity Analysis":
            show_sensitivity_analysis(asset_data, market_assumptions)
            
        elif valuation_method == "LCOE Analysis":
            show_lcoe_analysis(asset_data, market_assumptions)
        
        # Portfolio valuation summary
        st.markdown("---")
        show_portfolio_valuation(assets_df, market_assumptions)
        
    # except Exception as e:
    #     st.error(f"❌ Error in valuation models: {str(e)}")
    #     st.info("💡 **Tip:** Check that all required data is available and try refreshing.")




    except Exception as e:
        import traceback
        import sys
        
        # Get full traceback
        tb = traceback.format_exc()
        
        # Show exact line and file
        st.error(f"**Error Type:** {type(e).__name__}")
        st.error(f"**Error Message:** {str(e)}")
        
        # Extract line number from traceback
        tb_lines = tb.split('\n')
        relevant_line = None
        for line in tb_lines:
            if 'valuation_models.py' in line:
                relevant_line = line.strip()
                break
        
        if relevant_line:
            st.error(f"**Location:** {relevant_line}")
        
        # Show full traceback in expander
        with st.expander("🔍 View Full Error Details"):
            st.code(tb, language='python')
        
        st.info("💡 **Tip:** Check the error location above for specific debugging")

def show_dcf_analysis(asset_data, market_assumptions):
    """
    Show DCF analysis results
    """
    st.subheader("📈 Discounted Cash Flow Analysis")
    
    # Calculate DCF
    dcf_results = calculate_asset_dcf(asset_data, market_assumptions)
    
    # Results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        npv_color = "green" if dcf_results['npv'] > 0 else "red"
        st.markdown(f"""
        **Net Present Value**  
        <span style="color: {npv_color}; font-size: 1.2em; font-weight: bold;">
        {format_currency(dcf_results['npv'], millions=True)}
        </span>
        """, unsafe_allow_html=True)
    
    with col2:
        irr_color = "green" if dcf_results['irr'] > market_assumptions['discount_rate'] else "red"
        irr_display = f"{dcf_results['irr']:.1%}" if not np.isnan(dcf_results['irr']) else "N/A"
        st.markdown(f"""
        **Internal Rate of Return**  
        <span style="color: {irr_color}; font-size: 1.2em; font-weight: bold;">
        {irr_display}
        </span>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        **NPV per MW**  
        <span style="font-size: 1.2em; font-weight: bold;">
        {format_currency(dcf_results['npv_per_mw'])}
        </span>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        **Annual EBITDA**  
        <span style="font-size: 1.2em; font-weight: bold;">
        {format_currency(dcf_results['annual_ebitda'], millions=True)}
        </span>
        """, unsafe_allow_html=True)
    
    # Cash flow analysis
    st.markdown("**💰 Cash Flow Projection**")
    
    cash_flows = dcf_results['cash_flows'][1:11]  # First 10 years
    years = list(range(1, len(cash_flows) + 1))
    
    cf_df = pd.DataFrame({
        'Year': years,
        'Cash Flow (€M)': [cf/1e6 for cf in cash_flows]
    })
    
    st.bar_chart(cf_df.set_index('Year'))
    
    # Detailed metrics
    with st.expander("📊 Detailed Financial Metrics"):
        st.markdown(f"""
        **Asset Details:**
        - **Capacity:** {asset_data['capacity_mw']:,.0f} MW
        - **Annual Generation:** {dcf_results['annual_generation_mwh']:,.0f} MWh
        - **Capacity Factor:** {asset_data['availability']:.1%}
        - **Efficiency:** {asset_data['efficiency']:.1%}
        
        **Economic Assumptions:**
        - **Power Price:** €{market_assumptions['power_price']:.1f}/MWh
        - **Gas Price:** €{market_assumptions['gas_price']:.1f}/MWh
        - **Carbon Price:** €{market_assumptions['carbon_price']:.1f}/tonne
        - **Discount Rate:** {market_assumptions['discount_rate']:.1%}
        
        **Valuation Results:**
        - **NPV:** {format_currency(dcf_results['npv'])}
        - **IRR:** {dcf_results['irr']:.1%} (vs {market_assumptions['discount_rate']:.1%} hurdle)
        - **NPV/MW:** {format_currency(dcf_results['npv_per_mw'])}
        - **Payback Period:** ~{abs(dcf_results['npv']/dcf_results['annual_ebitda']):.1f} years
        """)

def show_monte_carlo_analysis(asset_data, market_assumptions):
    """
    Show Monte Carlo simulation results
    """
    st.subheader("🎲 Monte Carlo Simulation")
    
    # Run Monte Carlo simulation
    with st.spinner("Running Monte Carlo simulation..."):
        npv_stats, simulation_results = monte_carlo_valuation(asset_data, market_assumptions, n_simulations=1000)
    
    # Results display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected NPV", format_currency(npv_stats['mean'], millions=True))
    
    with col2:
        st.metric("P10 (Downside)", format_currency(npv_stats['p10'], millions=True))
    
    with col3:
        st.metric("P90 (Upside)", format_currency(npv_stats['p90'], millions=True))
    
    with col4:
        st.metric("Probability > 0", f"{npv_stats['prob_positive']:.1%}")
    
    # Monte Carlo chart
    st.markdown("**📊 NPV Distribution**")
    mc_chart = create_monte_carlo_chart(np.array(simulation_results)/1e6, "Monte Carlo NPV Results")
    st.plotly_chart(mc_chart, use_container_width=True)
    
    # Risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Statistical Summary**")
        st.markdown(f"""
        - **Mean:** {format_currency(npv_stats['mean'], millions=True)}
        - **Median:** {format_currency(npv_stats['p50'], millions=True)}
        - **Std Dev:** {format_currency(npv_stats['std'], millions=True)}
        - **Prob. Positive:** {npv_stats['prob_positive']:.1%}
        """)
    
    with col2:
        st.markdown("**⚠️ Risk Assessment**")
        
        if npv_stats['prob_positive'] > 0.8:
            risk_level = "🟢 Low Risk"
        elif npv_stats['prob_positive'] > 0.6:
            risk_level = "🟡 Medium Risk"
        else:
            risk_level = "🔴 High Risk"
            
        st.markdown(f"**Risk Level:** {risk_level}")
        
        value_at_risk = npv_stats['p10'] - npv_stats['mean']
        st.markdown(f"**Value at Risk (10%):** {format_currency(value_at_risk, millions=True)}")

def show_sensitivity_analysis(asset_data, market_assumptions):
    """
    Show sensitivity analysis results
    """
    st.subheader("📊 Sensitivity Analysis")
    
    # Define sensitivity ranges
    sensitivity_ranges = {
        'power_price': 0.3,  # ±30%
        'gas_price': 0.4,    # ±40%
        'carbon_price': 0.5  # ±50%
    }
    
    # Calculate sensitivity
    with st.spinner("Calculating sensitivities..."):
        sensitivity_results = sensitivity_analysis(asset_data, market_assumptions, sensitivity_ranges)
    
    # Create sensitivity chart
    sensitivity_chart = create_sensitivity_chart(sensitivity_results, asset_data['name'])
    st.plotly_chart(sensitivity_chart, use_container_width=True)
    
    # Sensitivity table
    st.markdown("**📋 Sensitivity Analysis Results**")
    
    sensitivity_df = pd.DataFrame()
    
    for variable, results in sensitivity_results.items():
        # Get impact of ±20% change
        low_impact = next(r['npv_change'] for r in results if abs(r['parameter_change'] + 0.2) < 0.01)
        high_impact = next(r['npv_change'] for r in results if abs(r['parameter_change'] - 0.2) < 0.01)
        
        sensitivity_df = pd.concat([sensitivity_df, pd.DataFrame({
            'Variable': variable.replace('_', ' ').title(),
            'Impact of -20%': f"{low_impact:.1%}",
            'Impact of +20%': f"{high_impact:.1%}",
            'Sensitivity': f"{(high_impact - low_impact)/0.4:.1f}"
        })], ignore_index=True)
    
    st.dataframe(sensitivity_df, hide_index=True, use_container_width=True)

def show_lcoe_analysis(asset_data, market_assumptions):
    """
    Show LCOE analysis
    """
    st.subheader("⚡ Levelized Cost of Energy (LCOE)")
    
    # LCOE calculation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        capex_mw = st.number_input(
            "CAPEX (EUR/MW)",
            min_value=100000,
            max_value=5000000,
            value=1500000 if asset_data['type'] == 'Gas' else 2500000,
            step=50000
        )
        
    with col2:
        project_life = st.number_input(
            "Project Life (years)",
            min_value=15,
            max_value=40,
            value=25,
            step=1
        )
    
    # Calculate LCOE
    total_capex = capex_mw * asset_data['capacity_mw']
    
    lcoe = calculate_lcoe(
        capex=total_capex,
        fixed_om=asset_data['fixed_cost'],
        variable_om=asset_data['variable_cost'],
        fuel_cost=market_assumptions['gas_price'] / asset_data['efficiency'] if asset_data['fuel_type'] == 'Natural Gas' else 0,
        capacity_factor=asset_data['availability'],
        capacity_mw=asset_data['capacity_mw'],
        project_life=project_life,
        discount_rate=market_assumptions['discount_rate']
    )
    
    # LCOE results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lcoe_color = "green" if lcoe < market_assumptions['power_price'] else "red"
        st.markdown(f"""
        **LCOE**  
        <span style="color: {lcoe_color}; font-size: 1.5em; font-weight: bold;">
        €{lcoe:.1f}/MWh
        </span>
        """, unsafe_allow_html=True)
    
    with col2:
        margin = market_assumptions['power_price'] - lcoe
        margin_color = "green" if margin > 0 else "red"
        st.markdown(f"""
        **Margin vs Power Price**  
        <span style="color: {margin_color}; font-size: 1.2em; font-weight: bold;">
        €{margin:.1f}/MWh
        </span>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        **Break-even Power Price**  
        <span style="font-size: 1.2em; font-weight: bold;">
        €{lcoe:.1f}/MWh
        </span>
        """, unsafe_allow_html=True)
    
    # LCOE breakdown
    st.markdown("**💰 LCOE Cost Breakdown**")
    
    # Calculate cost components
    annual_generation = asset_data['capacity_mw'] * 8760 * asset_data['availability']
    
    capex_component = (total_capex * market_assumptions['discount_rate'] * 
                      (1 + market_assumptions['discount_rate'])**project_life) / \
                     (((1 + market_assumptions['discount_rate'])**project_life - 1) * annual_generation)
    
    fixed_om_component = asset_data['fixed_cost'] * asset_data['capacity_mw'] / annual_generation
    variable_om_component = asset_data['variable_cost']
    fuel_component = market_assumptions['gas_price'] / asset_data['efficiency'] if asset_data['fuel_type'] == 'Natural Gas' else 0
    
    cost_breakdown = pd.DataFrame({
        'Component': ['CAPEX', 'Fixed O&M', 'Variable O&M', 'Fuel'],
        'Cost (EUR/MWh)': [capex_component, fixed_om_component, variable_om_component, fuel_component]
    })
    
    st.bar_chart(cost_breakdown.set_index('Component'))

def show_portfolio_valuation(assets_df, market_assumptions):
    """
    Show portfolio-level valuation summary
    """
    st.subheader("🏢 Portfolio Valuation Summary")
    
    # Calculate valuation for each asset
    portfolio_valuations = []
    total_npv = 0
    
    for _, asset in assets_df.iterrows():
        dcf_result = calculate_asset_dcf(asset, market_assumptions)
        portfolio_valuations.append({
            'Asset': asset['name'],
            'Type': asset['type'],
            'Capacity (MW)': asset['capacity_mw'],
            'NPV (€M)': dcf_result['npv'] / 1e6,
            'IRR': dcf_result['irr'] if not np.isnan(dcf_result['irr']) else 0,
            'NPV/MW (€k)': dcf_result['npv_per_mw'] / 1000
        })
        total_npv += dcf_result['npv']
    
    # Portfolio summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio NPV", format_currency(total_npv, millions=True))
    
    with col2:
        total_capacity = assets_df['capacity_mw'].sum()
        st.metric("Total Capacity", f"{total_capacity:,.0f} MW")
    
    with col3:
        avg_npv_per_mw = total_npv / total_capacity if total_capacity > 0 else 0
        st.metric("Avg NPV/MW", format_currency(avg_npv_per_mw))
    
    with col4:
        positive_npv_assets = sum(1 for v in portfolio_valuations if v['NPV (€M)'] > 0)
        st.metric("Positive NPV Assets", f"{positive_npv_assets}/{len(portfolio_valuations)}")
    
    # Portfolio table
    st.markdown("**📊 Portfolio Valuation Details**")
    
    portfolio_df = pd.DataFrame(portfolio_valuations)
    portfolio_df = portfolio_df.round(2)
    portfolio_df['IRR'] = portfolio_df['IRR'].apply(lambda x: f"{x:.1%}" if x != 0 else "N/A")
    
    st.dataframe(portfolio_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    show_valuation_models()