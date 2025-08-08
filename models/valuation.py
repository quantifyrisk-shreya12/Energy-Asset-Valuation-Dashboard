"""
Asset valuation models and financial analysis functions
"""

import numpy as np
import pandas as pd
from scipy import optimize
import streamlit as st
from datetime import datetime, timedelta
from models.financial_models import (
    calculate_npv, calculate_irr, calculate_lcoe, calculate_asset_dcf,
    monte_carlo_valuation, sensitivity_analysis
)
from config import DEFAULT_DISCOUNT_RATE, DEFAULT_PROJECT_LIFE, TAX_RATE, INFLATION_RATE

class AssetValuation:
    """
    Comprehensive asset valuation class with multiple methodologies
    """
    
    def __init__(self, asset_data, market_assumptions, valuation_date=None):
        self.asset_data = asset_data
        self.market_assumptions = market_assumptions
        self.valuation_date = valuation_date or datetime.now()
        self.discount_rate = market_assumptions.get('discount_rate', DEFAULT_DISCOUNT_RATE)
        self.project_life = market_assumptions.get('project_life', DEFAULT_PROJECT_LIFE)
        
    def calculate_base_case_valuation(self):
        """
        Calculate base case DCF valuation
        """
        dcf_result = calculate_asset_dcf(self.asset_data, self.market_assumptions, self.project_life)
        
        valuation_summary = {
            'asset_name': self.asset_data['name'],
            'valuation_date': self.valuation_date,
            'methodology': 'DCF Base Case',
            'npv_eur': dcf_result['npv'],
            'irr': dcf_result['irr'],
            'npv_per_mw': dcf_result['npv_per_mw'],
            'annual_ebitda': dcf_result['annual_ebitda'],
            'annual_generation_mwh': dcf_result['annual_generation_mwh'],
            'market_assumptions': self.market_assumptions.copy(),
            'cash_flows': dcf_result['cash_flows']
        }
        
        return valuation_summary
    
    def calculate_risk_adjusted_valuation(self, confidence_level=0.95):
        """
        Calculate risk-adjusted valuation using Monte Carlo simulation
        """
        npv_stats, simulation_results = monte_carlo_valuation(
            self.asset_data, 
            self.market_assumptions, 
            n_simulations=1000
        )
        
        # Risk adjustment based on volatility
        risk_adjustment_factor = 1 - (npv_stats['std'] / abs(npv_stats['mean'])) * 0.1
        risk_adjusted_npv = npv_stats['mean'] * risk_adjustment_factor
        
        risk_valuation = {
            'asset_name': self.asset_data['name'],
            'methodology': 'Monte Carlo Risk-Adjusted',
            'expected_npv': npv_stats['mean'],
            'risk_adjusted_npv': risk_adjusted_npv,
            'npv_volatility': npv_stats['std'],
            'probability_positive': npv_stats['prob_positive'],
            'var_p10': npv_stats['p10'],
            'var_p90': npv_stats['p90'],
            'confidence_interval': {
                'lower': npv_stats['p10'],
                'upper': npv_stats['p90']
            },
            'simulation_results': simulation_results
        }
        
        return risk_valuation
    
    def calculate_real_options_valuation(self):
        """
        Calculate real options value for flexible assets
        """
        base_npv = self.calculate_base_case_valuation()['npv_eur']
        
        # Simplified real options calculation
        # In practice, this would use Black-Scholes or binomial models
        
        # Option to expand (if capacity can be increased)
        expansion_option_value = 0
        if self.asset_data['type'] in ['Wind', 'Solar']:
            expansion_volatility = 0.3  # Renewable project volatility
            time_to_expiry = 5  # 5 years option to expand
            expansion_option_value = max(0, base_npv * 0.1 * expansion_volatility * np.sqrt(time_to_expiry))
        
        # Option to abandon (salvage value)
        abandonment_option_value = 0
        asset_age = datetime.now().year - self.asset_data['construction_year']
        if asset_age > 15:  # Older assets have abandonment optionality
            salvage_rate = max(0.1, 1 - asset_age / 40)  # Declining salvage value
            abandonment_option_value = self.asset_data['capacity_mw'] * 50000 * salvage_rate
        
        # Fuel switching option (for multi-fuel plants)
        fuel_switching_value = 0
        if 'dual_fuel' in self.asset_data and self.asset_data.get('dual_fuel', False):
            fuel_switching_value = base_npv * 0.05  # 5% uplift for fuel flexibility
        
        total_option_value = expansion_option_value + abandonment_option_value + fuel_switching_value
        
        real_options_valuation = {
            'asset_name': self.asset_data['name'],
            'methodology': 'Real Options Enhanced DCF',
            'base_dcf_npv': base_npv,
            'expansion_option_value': expansion_option_value,
            'abandonment_option_value': abandonment_option_value,
            'fuel_switching_value': fuel_switching_value,
            'total_option_value': total_option_value,
            'enhanced_npv': base_npv + total_option_value,
            'option_premium': total_option_value / base_npv if base_npv != 0 else 0
        }
        
        return real_options_valuation
    
    def calculate_sum_of_parts_valuation(self):
        """
        Sum-of-parts valuation breaking down asset value components
        """
        base_case = self.calculate_base_case_valuation()
        
        # Decompose value into components
        capacity_value = self.asset_data['capacity_mw'] * 200000  # €200k/MW capacity value
        
        # Energy margin value (based on generation and spark spread)
        annual_generation = base_case['annual_generation_mwh']
        power_price = self.market_assumptions['power_price']
        marginal_cost = self._calculate_marginal_cost()
        energy_margin = max(0, power_price - marginal_cost)
        
        # Present value of energy margins
        energy_value = 0
        for year in range(1, self.project_life + 1):
            annual_margin_value = energy_margin * annual_generation
            discounted_value = annual_margin_value / (1 + self.discount_rate) ** year
            energy_value += discounted_value
        
        # Ancillary services value (simplified)
        ancillary_services_value = 0
        if self.asset_data['type'] == 'Gas':
            # Gas plants provide grid services
            ancillary_services_value = self.asset_data['capacity_mw'] * 20000  # €20k/MW/year NPV
        
        # Environmental value (carbon credits, etc.)
        environmental_value = 0
        if self.asset_data['type'] in ['Wind', 'Solar']:
            co2_avoided_annually = annual_generation * 0.4  # Assume 0.4 tCO2/MWh avoided
            carbon_credit_value = co2_avoided_annually * self.market_assumptions.get('carbon_price', 85)
            environmental_value = sum([carbon_credit_value / (1 + self.discount_rate) ** year 
                                     for year in range(1, min(self.project_life, 10) + 1)])
        
        # Fixed costs (negative value)
        annual_fixed_costs = self.asset_data['fixed_cost'] * self.asset_data['capacity_mw']
        fixed_cost_pv = sum([annual_fixed_costs / (1 + self.discount_rate) ** year 
                           for year in range(1, self.project_life + 1)])
        
        sum_of_parts = {
            'asset_name': self.asset_data['name'],
            'methodology': 'Sum of Parts',
            'capacity_value': capacity_value,
            'energy_value': energy_value,
            'ancillary_services_value': ancillary_services_value,
            'environmental_value': environmental_value,
            'fixed_cost_pv': -fixed_cost_pv,  # Negative because it's a cost
            'total_value': (capacity_value + energy_value + ancillary_services_value + 
                          environmental_value - fixed_cost_pv),
            'value_per_mw': ((capacity_value + energy_value + ancillary_services_value + 
                            environmental_value - fixed_cost_pv) / self.asset_data['capacity_mw'])
        }
        
        return sum_of_parts
    
    def _calculate_marginal_cost(self):
        """
        Calculate asset marginal cost
        """
        variable_cost = self.asset_data['variable_cost']
        
        # Fuel cost
        fuel_cost = 0
        if self.asset_data['fuel_type'] == 'Natural Gas':
            gas_price = self.market_assumptions['gas_price']
            efficiency = self.asset_data['efficiency']
            fuel_cost = gas_price / efficiency
        
        # Carbon cost
        carbon_price = self.market_assumptions['carbon_price']
        co2_intensity = self.asset_data['co2_intensity']
        carbon_cost = carbon_price * co2_intensity
        
        return variable_cost + fuel_cost + carbon_cost

def calculate_portfolio_valuation(assets_df, market_assumptions, methodology='dcf'):
    """
    Calculate valuation for entire asset portfolio
    """
    portfolio_results = []
    total_portfolio_value = 0
    
    for _, asset in assets_df.iterrows():
        # Create valuation instance
        asset_valuation = AssetValuation(asset, market_assumptions)
        
        if methodology == 'dcf':
            result = asset_valuation.calculate_base_case_valuation()
            portfolio_value = result['npv_eur']
        elif methodology == 'monte_carlo':
            result = asset_valuation.calculate_risk_adjusted_valuation()
            portfolio_value = result['risk_adjusted_npv']
        elif methodology == 'real_options':
            result = asset_valuation.calculate_real_options_valuation()
            portfolio_value = result['enhanced_npv']
        elif methodology == 'sum_of_parts':
            result = asset_valuation.calculate_sum_of_parts_valuation()
            portfolio_value = result['total_value']
        else:
            result = asset_valuation.calculate_base_case_valuation()
            portfolio_value = result['npv_eur']
        
        result['methodology'] = methodology
        portfolio_results.append(result)
        total_portfolio_value += portfolio_value
    
    # Portfolio-level metrics
    total_capacity = assets_df['capacity_mw'].sum()
    
    portfolio_summary = {
        'valuation_date': datetime.now(),
        'methodology': methodology,
        'total_portfolio_value': total_portfolio_value,
        'total_capacity_mw': total_capacity,
        'value_per_mw': total_portfolio_value / total_capacity if total_capacity > 0 else 0,
        'asset_count': len(assets_df),
        'individual_valuations': portfolio_results,
        'market_assumptions': market_assumptions
    }
    
    return portfolio_summary

def calculate_relative_valuation(asset_data, comparable_assets, market_data):
    """
    Calculate relative valuation using comparable asset multiples
    """
    # Key valuation multiples for power generation assets
    multiples = {}
    
    # Enterprise Value / MW multiple
    if comparable_assets:
        ev_mw_multiple = np.mean([comp.get('ev_per_mw', 800000) for comp in comparable_assets])
    else:
        # Default multiples by technology
        default_multiples = {
            'Gas': 650000,      # €650k/MW
            'Wind': 1200000,    # €1.2M/MW
            'Solar': 900000,    # €900k/MW
            'Coal': 400000,     # €400k/MW
            'Nuclear': 2000000  # €2M/MW
        }
        ev_mw_multiple = default_multiples.get(asset_data['type'], 800000)
    
    multiples['ev_per_mw'] = ev_mw_multiple
    
    # EBITDA multiple (simplified calculation)
    annual_generation = asset_data['capacity_mw'] * 8760 * asset_data['availability']
    current_power_price = market_data['current_power_price']
    estimated_ebitda = annual_generation * (current_power_price - 30)  # Rough EBITDA estimate
    
    if estimated_ebitda > 0:
        ebitda_multiple = 8.0  # Typical power generation EBITDA multiple
        multiples['ebitda_multiple'] = ebitda_multiple
        multiples['estimated_annual_ebitda'] = estimated_ebitda
    
    # Calculate relative valuations
    relative_valuations = {
        'asset_name': asset_data['name'],
        'methodology': 'Relative Valuation',
        'ev_mw_valuation': asset_data['capacity_mw'] * ev_mw_multiple,
        'ev_per_mw_multiple': ev_mw_multiple,
    }
    
    if estimated_ebitda > 0:
        relative_valuations['ebitda_valuation'] = estimated_ebitda * ebitda_multiple
        relative_valuations['ebitda_multiple'] = ebitda_multiple
        relative_valuations['estimated_ebitda'] = estimated_ebitda
    
    # Book value multiple (replacement cost approach)
    asset_age = datetime.now().year - asset_data['construction_year']
    depreciation_rate = 0.04  # 4% per year
    book_value_multiple = max(0.2, 1 - (asset_age * depreciation_rate))  # Min 20% of replacement cost
    
    replacement_cost_per_mw = {
        'Gas': 1000000,     # €1M/MW
        'Wind': 1500000,    # €1.5M/MW  
        'Solar': 800000,    # €800k/MW
        'Coal': 1200000,    # €1.2M/MW
        'Nuclear': 5000000  # €5M/MW
    }
    
    replacement_cost = (asset_data['capacity_mw'] * 
                       replacement_cost_per_mw.get(asset_data['type'], 1000000))
    book_value = replacement_cost * book_value_multiple
    
    relative_valuations['book_value'] = book_value
    relative_valuations['replacement_cost'] = replacement_cost
    relative_valuations['book_value_multiple'] = book_value_multiple
    
    return relative_valuations

def calculate_break_even_analysis(asset_data, market_assumptions):
    """
    Calculate break-even prices and volumes for asset
    """
    # Fixed costs
    annual_fixed_costs = asset_data['fixed_cost'] * asset_data['capacity_mw']
    
    # Variable costs
    variable_cost = asset_data['variable_cost']
    
    # Fuel costs
    fuel_cost_per_mwh = 0
    if asset_data['fuel_type'] == 'Natural Gas':
        fuel_cost_per_mwh = market_assumptions['gas_price'] / asset_data['efficiency']
    
    # Carbon costs
    carbon_cost_per_mwh = market_assumptions['carbon_price'] * asset_data['co2_intensity']
    
    # Total variable cost per MWh
    total_variable_cost = variable_cost + fuel_cost_per_mwh + carbon_cost_per_mwh
    
    # Break-even power price (to cover all costs)
    max_annual_generation = asset_data['capacity_mw'] * 8760
    break_even_price = total_variable_cost + (annual_fixed_costs / max_annual_generation)
    
    # Break-even capacity factor (at current power price)
    current_power_price = market_assumptions['power_price']
    if current_power_price > total_variable_cost:
        margin_per_mwh = current_power_price - total_variable_cost
        break_even_generation = annual_fixed_costs / margin_per_mwh
        break_even_capacity_factor = break_even_generation / max_annual_generation
    else:
        break_even_capacity_factor = float('inf')  # Cannot break even
    
    # Break-even fuel price (gas plants only)
    break_even_fuel_price = None
    if asset_data['fuel_type'] == 'Natural Gas':
        target_margin = current_power_price - variable_cost - carbon_cost_per_mwh
        if target_margin > 0:
            break_even_fuel_price = target_margin * asset_data['efficiency']
    
    break_even_analysis = {
        'asset_name': asset_data['name'],
        'break_even_power_price': break_even_price,
        'break_even_capacity_factor': min(1.0, break_even_capacity_factor),
        'break_even_fuel_price': break_even_fuel_price,
        'current_margin_per_mwh': current_power_price - total_variable_cost,
        'annual_fixed_costs': annual_fixed_costs,
        'variable_cost_per_mwh': total_variable_cost,
        'minimum_generation_for_positive_ebitda': break_even_generation if break_even_capacity_factor != float('inf') else 0
    }
    
    return break_even_analysis

def calculate_scenario_valuations(asset_data, base_market_assumptions, scenarios):
    """
    Calculate asset valuation under different market scenarios
    
    scenarios = {
        'Bull Case': {'power_price_factor': 1.3, 'gas_price_factor': 1.1, 'carbon_price_factor': 1.5},
        'Bear Case': {'power_price_factor': 0.7, 'gas_price_factor': 0.9, 'carbon_price_factor': 0.6},
        'Green Transition': {'power_price_factor': 0.9, 'gas_price_factor': 1.2, 'carbon_price_factor': 2.0}
    }
    """
    scenario_valuations = {}
    
    for scenario_name, scenario_factors in scenarios.items():
        # Adjust market assumptions
        scenario_assumptions = base_market_assumptions.copy()
        scenario_assumptions['power_price'] *= scenario_factors.get('power_price_factor', 1.0)
        scenario_assumptions['gas_price'] *= scenario_factors.get('gas_price_factor', 1.0) 
        scenario_assumptions['carbon_price'] *= scenario_factors.get('carbon_price_factor', 1.0)
        
        # Calculate valuation
        asset_valuation = AssetValuation(asset_data, scenario_assumptions)
        scenario_result = asset_valuation.calculate_base_case_valuation()
        
        scenario_valuations[scenario_name] = {
            'npv': scenario_result['npv_eur'],
            'irr': scenario_result['irr'],
            'annual_ebitda': scenario_result['annual_ebitda'],
            'market_assumptions': scenario_assumptions,
            'scenario_factors': scenario_factors
        }
    
    # Calculate scenario statistics
    npv_values = [result['npv'] for result in scenario_valuations.values()]
    
    scenario_summary = {
        'asset_name': asset_data['name'],
        'scenarios': scenario_valuations,
        'scenario_statistics': {
            'min_npv': min(npv_values),
            'max_npv': max(npv_values),
            'mean_npv': np.mean(npv_values),
            'npv_std': np.std(npv_values),
            'npv_range': max(npv_values) - min(npv_values)
        }
    }
    
    return scenario_summary

def calculate_replacement_cost_valuation(asset_data):
    """
    Calculate asset valuation based on replacement cost approach
    """
    # Current replacement costs by technology (EUR/MW)
    replacement_costs = {
        'Gas': {'ccgt': 900000, 'ocgt': 600000, 'default': 750000},
        'Wind': {'onshore': 1400000, 'offshore': 2200000, 'default': 1500000},
        'Solar': {'utility': 800000, 'rooftop': 1200000, 'default': 900000},
        'Coal': {'subcritical': 1800000, 'supercritical': 2200000, 'default': 2000000},
        'Nuclear': {'gen3': 6000000, 'gen2': 4000000, 'default': 5000000},
        'Hydro': {'run_of_river': 2000000, 'pumped_storage': 1500000, 'default': 1750000}
    }
    
    asset_type = asset_data['type']
    base_replacement_cost = replacement_costs.get(asset_type, {}).get('default', 1000000)
    
    # Adjust for technology improvements over time
    construction_year = asset_data['construction_year']
    current_year = datetime.now().year
    years_since_construction = current_year - construction_year
    
    # Technology improvement factor (costs generally decline)
    tech_improvement_rates = {
        'Gas': 0.01,     # 1% per year cost reduction
        'Wind': 0.03,    # 3% per year (significant improvements)
        'Solar': 0.08,   # 8% per year (dramatic cost reductions)
        'Coal': 0.005,   # 0.5% per year
        'Nuclear': -0.02, # 2% per year cost increase (more complex regulations)
        'Hydro': 0.01    # 1% per year
    }
    
    improvement_rate = tech_improvement_rates.get(asset_type, 0.01)
    technology_factor = (1 - improvement_rate) ** years_since_construction
    
    # Adjust replacement cost for current technology
    current_replacement_cost_per_mw = base_replacement_cost * technology_factor
    
    # Total replacement cost
    total_replacement_cost = current_replacement_cost_per_mw * asset_data['capacity_mw']
    
    # Depreciation adjustment
    typical_life = {'Gas': 30, 'Wind': 25, 'Solar': 25, 'Coal': 40, 'Nuclear': 60, 'Hydro': 80}
    asset_life = typical_life.get(asset_type, 30)
    
    # Straight-line depreciation
    depreciation_rate = years_since_construction / asset_life
    depreciated_replacement_cost = total_replacement_cost * (1 - min(0.8, depreciation_rate))
    
    # Obsolescence adjustment
    obsolescence_factor = 1.0
    if asset_type == 'Coal' and years_since_construction > 20:
        obsolescence_factor = 0.6  # Coal plants becoming obsolete
    elif asset_type == 'Gas' and asset_data['efficiency'] < 0.50:
        obsolescence_factor = 0.8  # Inefficient gas plants
    
    adjusted_replacement_cost = depreciated_replacement_cost * obsolescence_factor
    
    replacement_valuation = {
        'asset_name': asset_data['name'],
        'methodology': 'Replacement Cost',
        'base_replacement_cost_per_mw': base_replacement_cost,
        'current_replacement_cost_per_mw': current_replacement_cost_per_mw,
        'total_replacement_cost': total_replacement_cost,
        'depreciated_replacement_cost': depreciated_replacement_cost,
        'obsolescence_adjustment': obsolescence_factor,
        'final_replacement_value': adjusted_replacement_cost,
        'replacement_value_per_mw': adjusted_replacement_cost / asset_data['capacity_mw'],
        'technology_factor': technology_factor,
        'depreciation_factor': 1 - min(0.8, depreciation_rate)
    }
    
    return replacement_valuation

def calculate_liquidation_value(asset_data):
    """
    Calculate asset liquidation/salvage value
    """
    replacement_valuation = calculate_replacement_cost_valuation(asset_data)
    replacement_value = replacement_valuation['final_replacement_value']
    
    # Liquidation discount factors by asset type and age
    construction_year = asset_data['construction_year']
    asset_age = datetime.now().year - construction_year
    
    # Base liquidation rates (% of replacement value)
    base_liquidation_rates = {
        'Gas': 0.15,    # Gas turbines have decent resale market
        'Wind': 0.10,   # Wind turbines harder to relocate
        'Solar': 0.20,  # Solar panels more portable
        'Coal': 0.05,   # Coal plants have limited market
        'Nuclear': 0.02, # Nuclear plants very difficult to liquidate
        'Hydro': 0.25   # Hydro infrastructure valuable
    }
    
    base_rate = base_liquidation_rates.get(asset_data['type'], 0.10)
    
    # Age adjustment
    if asset_age < 5:
        age_factor = 1.2  # Young assets worth more
    elif asset_age < 15:
        age_factor = 1.0
    elif asset_age < 25:
        age_factor = 0.7
    else:
        age_factor = 0.4  # Very old assets
    
    # Market conditions adjustment
    market_adjustment = 1.0  # Could be adjusted based on current market conditions
    
    # Final liquidation value
    liquidation_rate = base_rate * age_factor * market_adjustment
    liquidation_value = replacement_value * liquidation_rate
    
    # Scrap value (minimum floor)
    scrap_value_per_mw = {'Gas': 10000, 'Wind': 15000, 'Solar': 5000, 
                         'Coal': 8000, 'Nuclear': 50000, 'Hydro': 20000}
    minimum_scrap_value = (asset_data['capacity_mw'] * 
                          scrap_value_per_mw.get(asset_data['type'], 10000))
    
    final_liquidation_value = max(liquidation_value, minimum_scrap_value)
    
    liquidation_valuation = {
        'asset_name': asset_data['name'],
        'methodology': 'Liquidation Value',
        'base_liquidation_rate': base_rate,
        'age_factor': age_factor,
        'adjusted_liquidation_rate': liquidation_rate,
        'liquidation_value': final_liquidation_value,
        'minimum_scrap_value': minimum_scrap_value,
        'liquidation_value_per_mw': final_liquidation_value / asset_data['capacity_mw'],
        'replacement_value_reference': replacement_value
    }
    
    return liquidation_valuation

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def calculate_comprehensive_valuation(asset_data, market_assumptions):
    """
    Calculate comprehensive valuation using multiple methodologies
    """
    asset_valuation = AssetValuation(asset_data, market_assumptions)
    
    # Calculate all valuation methods
    valuations = {
        'dcf_base_case': asset_valuation.calculate_base_case_valuation(),
        'monte_carlo': asset_valuation.calculate_risk_adjusted_valuation(),
        'real_options': asset_valuation.calculate_real_options_valuation(),
        'sum_of_parts': asset_valuation.calculate_sum_of_parts_valuation(),
        'replacement_cost': calculate_replacement_cost_valuation(asset_data),
        'liquidation_value': calculate_liquidation_value(asset_data),
        'break_even_analysis': calculate_break_even_analysis(asset_data, market_assumptions)
    }
    
    # Extract NPV/value from each method
    npv_values = {}
    for method, result in valuations.items():
        if method == 'dcf_base_case':
            npv_values[method] = result['npv_eur']
        elif method == 'monte_carlo':
            npv_values[method] = result['risk_adjusted_npv']
        elif method == 'real_options':
            npv_values[method] = result['enhanced_npv']
        elif method == 'sum_of_parts':
            npv_values[method] = result['total_value']
        elif method == 'replacement_cost':
            npv_values[method] = result['final_replacement_value']
        elif method == 'liquidation_value':
            npv_values[method] = result['liquidation_value']
    
    # Calculate valuation statistics
    valid_npvs = [v for v in npv_values.values() if not np.isnan(v) and v != float('inf')]
    
    valuation_summary = {
        'asset_name': asset_data['name'],
        'valuation_date': datetime.now(),
        'individual_valuations': valuations,
        'npv_by_method': npv_values,
        'valuation_statistics': {
            'min_value': min(valid_npvs) if valid_npvs else 0,
            'max_value': max(valid_npvs) if valid_npvs else 0,
            'mean_value': np.mean(valid_npvs) if valid_npvs else 0,
            'median_value': np.median(valid_npvs) if valid_npvs else 0,
            'std_deviation': np.std(valid_npvs) if valid_npvs else 0
        },
        'market_assumptions': market_assumptions
    }
    
    return valuation_summary