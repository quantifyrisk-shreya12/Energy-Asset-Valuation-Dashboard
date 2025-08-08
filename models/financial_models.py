"""
Financial models for asset valuation
"""

import numpy as np
import pandas as pd
from scipy import optimize
from config import DEFAULT_DISCOUNT_RATE, DEFAULT_PROJECT_LIFE, TAX_RATE, INFLATION_RATE

def calculate_npv(cash_flows, discount_rate=DEFAULT_DISCOUNT_RATE):
    """
    Calculate Net Present Value of cash flows
    """
    npv = 0
    for i, cf in enumerate(cash_flows):
        npv += cf / (1 + discount_rate) ** i
    return npv

def calculate_irr(cash_flows):
    """
    Calculate Internal Rate of Return
    """
    try:
        # Use scipy optimize to find IRR
        def npv_function(rate):
            return calculate_npv(cash_flows, rate)
        
        # Find rate where NPV = 0
        irr = optimize.brentq(npv_function, -0.99, 10.0)
        return irr
    except:
        return np.nan

def calculate_lcoe(capex, fixed_om, variable_om, fuel_cost, capacity_factor, 
                   capacity_mw, project_life=DEFAULT_PROJECT_LIFE, discount_rate=DEFAULT_DISCOUNT_RATE):
    """
    Calculate Levelized Cost of Energy (LCOE)
    """
    # Annual generation
    annual_generation_mwh = capacity_mw * 8760 * capacity_factor
    
    # Calculate present value of costs
    pv_capex = capex
    pv_fixed_om = sum([fixed_om * capacity_mw / (1 + discount_rate) ** year 
                       for year in range(1, project_life + 1)])
    
    pv_variable_costs = sum([variable_om * annual_generation_mwh / (1 + discount_rate) ** year
                            for year in range(1, project_life + 1)])
    
    pv_fuel_costs = sum([fuel_cost * annual_generation_mwh / (1 + discount_rate) ** year
                        for year in range(1, project_life + 1)])
    
    total_pv_costs = pv_capex + pv_fixed_om + pv_variable_costs + pv_fuel_costs
    
    # Calculate present value of generation
    pv_generation = sum([annual_generation_mwh / (1 + discount_rate) ** year
                        for year in range(1, project_life + 1)])
    
    lcoe = total_pv_costs / pv_generation
    return lcoe

def calculate_asset_dcf(asset, market_assumptions, project_life=DEFAULT_PROJECT_LIFE):
    """
    Calculate DCF valuation for a power plant asset
    """
    capacity_mw = asset['capacity_mw']
    efficiency = asset['efficiency']
    availability = asset['availability']
    fixed_cost = asset['fixed_cost']
    variable_cost = asset['variable_cost']
    co2_intensity = asset['co2_intensity']
    
    # Market assumptions
    power_price = market_assumptions.get('power_price', 65.0)
    gas_price = market_assumptions.get('gas_price', 45.0) 
    carbon_price = market_assumptions.get('carbon_price', 85.0)
    
    # Annual generation
    annual_generation_mwh = capacity_mw * 8760 * availability
    
    # Revenue calculation
    annual_revenue = annual_generation_mwh * power_price
    
    # Cost calculations
    annual_fixed_costs = fixed_cost * capacity_mw
    annual_variable_costs = variable_cost * annual_generation_mwh
    
    # Fuel costs (for thermal plants)
    if asset['fuel_type'] == 'Natural Gas':
        annual_fuel_costs = annual_generation_mwh * gas_price / efficiency
    else:
        annual_fuel_costs = 0
    
    # Carbon costs
    annual_carbon_costs = annual_generation_mwh * co2_intensity * carbon_price
    
    # EBITDA
    annual_ebitda = (annual_revenue - annual_variable_costs - 
                     annual_fuel_costs - annual_carbon_costs - annual_fixed_costs)
    
    # Generate cash flows with price escalation and cost inflation
    cash_flows = [0]  # Year 0 - no initial investment for existing assets
    
    for year in range(1, project_life + 1):
        # Apply inflation/escalation
        revenue = annual_revenue * (1 + INFLATION_RATE) ** year
        fixed_costs = annual_fixed_costs * (1 + INFLATION_RATE) ** year
        variable_costs = annual_variable_costs * (1 + INFLATION_RATE) ** year
        fuel_costs = annual_fuel_costs * (1 + INFLATION_RATE) ** year
        carbon_costs = annual_carbon_costs * (1 + INFLATION_RATE) ** year
        
        # EBITDA
        ebitda = revenue - variable_costs - fuel_costs - carbon_costs - fixed_costs
        
        # Apply tax
        net_cash_flow = ebitda * (1 - TAX_RATE)
        cash_flows.append(net_cash_flow)
    
    # Calculate valuation metrics
    npv = calculate_npv(cash_flows)
    irr = calculate_irr(cash_flows)
    
    # Per MW valuation
    npv_per_mw = npv / capacity_mw
    
    return {
        'npv': npv,
        'irr': irr,
        'npv_per_mw': npv_per_mw,
        'annual_ebitda': annual_ebitda,
        'annual_generation_mwh': annual_generation_mwh,
        'cash_flows': cash_flows
    }

def monte_carlo_valuation(asset, base_market_assumptions, n_simulations=1000):
    """
    Monte Carlo simulation for asset valuation
    """
    results = []
    
    for _ in range(n_simulations):
        # Create random market scenarios
        market_assumptions = {
            'power_price': np.random.normal(
                base_market_assumptions.get('power_price', 65), 15
            ),
            'gas_price': np.random.normal(
                base_market_assumptions.get('gas_price', 45), 8
            ),
            'carbon_price': np.random.normal(
                base_market_assumptions.get('carbon_price', 85), 15
            )
        }
        
        # Ensure positive prices
        for key in market_assumptions:
            market_assumptions[key] = max(5, market_assumptions[key])
        
        # Calculate DCF for this scenario
        dcf_result = calculate_asset_dcf(asset, market_assumptions)
        results.append(dcf_result['npv'])
    
    # Calculate statistics
    npv_stats = {
        'mean': np.mean(results),
        'std': np.std(results),
        'p10': np.percentile(results, 10),
        'p50': np.percentile(results, 50),
        'p90': np.percentile(results, 90),
        'prob_positive': sum(1 for x in results if x > 0) / len(results)
    }
    
    return npv_stats, results

def calculate_portfolio_metrics(assets_valuations):
    """
    Calculate portfolio-level metrics
    """
    total_capacity = sum(asset['capacity_mw'] for asset in assets_valuations)
    total_npv = sum(val['npv'] for val in assets_valuations)
    weighted_avg_irr = np.average(
        [val['irr'] for val in assets_valuations if not np.isnan(val['irr'])],
        weights=[asset['capacity_mw'] for asset, val in zip(assets_valuations, assets_valuations) 
                if not np.isnan(val['irr'])]
    )
    
    portfolio_metrics = {
        'total_capacity_mw': total_capacity,
        'total_portfolio_npv': total_npv,
        'portfolio_value_per_mw': total_npv / total_capacity if total_capacity > 0 else 0,
        'weighted_avg_irr': weighted_avg_irr
    }
    
    return portfolio_metrics

def sensitivity_analysis(asset, base_market_assumptions, sensitivity_ranges):
    """
    Perform sensitivity analysis on key variables
    """
    base_npv = calculate_asset_dcf(asset, base_market_assumptions)['npv']
    
    sensitivity_results = {}
    
    for variable, range_pct in sensitivity_ranges.items():
        base_value = base_market_assumptions.get(variable, 65.0)
        
        sensitivities = []
        changes = np.linspace(-range_pct, range_pct, 11)  # -50% to +50% in 10% steps
        
        for change in changes:
            modified_assumptions = base_market_assumptions.copy()
            modified_assumptions[variable] = base_value * (1 + change)
            
            new_npv = calculate_asset_dcf(asset, modified_assumptions)['npv']
            npv_change = (new_npv - base_npv) / base_npv if base_npv != 0 else 0
            
            sensitivities.append({
                'parameter_change': change,
                'npv_change': npv_change
            })
        
        sensitivity_results[variable] = sensitivities
    
    return sensitivity_results