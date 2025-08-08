"""
Asset data management and processing functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from data.sample_assets import get_sample_assets, get_fuel_prices
from data.market_data import get_market_overview

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_asset_portfolio():
    """
    Load and return the complete asset portfolio data
    """
    try:
        assets_df = get_sample_assets()
        return assets_df
    except Exception as e:
        st.error(f"Error loading asset portfolio: {e}")
        return pd.DataFrame()

def validate_asset_data(asset_data):
    """
    Validate asset data for completeness and consistency
    """
    required_columns = [
        'name', 'type', 'capacity_mw', 'efficiency', 'variable_cost',
        'fixed_cost', 'co2_intensity', 'availability', 'construction_year',
        'country', 'fuel_type'
    ]
    
    validation_results = {
        'is_valid': True,
        'missing_columns': [],
        'data_issues': []
    }
    
    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in asset_data.columns]
    if missing_cols:
        validation_results['missing_columns'] = missing_cols
        validation_results['is_valid'] = False
    
    # Check for data quality issues
    if not asset_data.empty:
        # Check for negative capacities
        if (asset_data['capacity_mw'] <= 0).any():
            validation_results['data_issues'].append("Negative or zero capacity values found")
            validation_results['is_valid'] = False
        
        # Check efficiency bounds
        if (asset_data['efficiency'] <= 0).any() or (asset_data['efficiency'] > 1).any():
            validation_results['data_issues'].append("Efficiency values outside valid range (0-1)")
            validation_results['is_valid'] = False
        
        # Check availability factors
        if (asset_data['availability'] < 0).any() or (asset_data['availability'] > 1).any():
            validation_results['data_issues'].append("Availability factors outside valid range (0-1)")
            validation_results['is_valid'] = False
    
    return validation_results

def enrich_asset_data(asset_data, market_data=None):
    """
    Enrich asset data with calculated metrics and market-dependent values
    """
    enriched_data = asset_data.copy()
    
    # Calculate asset age
    current_year = datetime.now().year
    enriched_data['asset_age_years'] = current_year - enriched_data['construction_year']
    
    # Calculate annual generation capacity
    enriched_data['annual_generation_mwh'] = (
        enriched_data['capacity_mw'] * 8760 * enriched_data['availability']
    )
    
    # Calculate theoretical maximum generation
    enriched_data['max_annual_generation_mwh'] = enriched_data['capacity_mw'] * 8760
    
    # Calculate load factor (same as availability for these assets)
    enriched_data['load_factor'] = enriched_data['availability']
    
    # Calculate annual fixed costs
    enriched_data['annual_fixed_cost_eur'] = (
        enriched_data['fixed_cost'] * enriched_data['capacity_mw']
    )
    
    # Add technology classification
    enriched_data['technology_class'] = enriched_data['type'].map({
        'Gas': 'Thermal',
        'Coal': 'Thermal', 
        'Nuclear': 'Thermal',
        'Wind': 'Renewable',
        'Solar': 'Renewable',
        'Hydro': 'Renewable'
    })
    
    # Calculate capacity factor quintiles for benchmarking
    enriched_data['capacity_factor_quintile'] = pd.qcut(
        enriched_data['availability'], 
        q=5, 
        labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']
    )
    
    # If market data is available, calculate economic metrics
    if market_data is not None:
        enriched_data = add_market_dependent_metrics(enriched_data, market_data)
    
    return enriched_data

def add_market_dependent_metrics(asset_data, market_data):
    """
    Add market-dependent economic metrics to asset data
    """
    enriched_data = asset_data.copy()
    
    # Current market prices
    power_price = market_data['current_power_price']
    gas_price = market_data['gas_price']
    carbon_price = market_data['carbon_price']
    
    # Calculate marginal costs
    marginal_costs = []
    annual_fuel_costs = []
    annual_carbon_costs = []
    
    for _, asset in enriched_data.iterrows():
        # Fuel cost calculation
        if asset['fuel_type'] == 'Natural Gas':
            fuel_cost_per_mwh = gas_price / asset['efficiency']
            annual_fuel_cost = fuel_cost_per_mwh * asset['annual_generation_mwh']
        else:
            fuel_cost_per_mwh = 0
            annual_fuel_cost = 0
        
        # Carbon cost calculation  
        carbon_cost_per_mwh = carbon_price * asset['co2_intensity']
        annual_carbon_cost = carbon_cost_per_mwh * asset['annual_generation_mwh']
        
        # Total marginal cost
        marginal_cost = asset['variable_cost'] + fuel_cost_per_mwh + carbon_cost_per_mwh
        
        marginal_costs.append(marginal_cost)
        annual_fuel_costs.append(annual_fuel_cost)
        annual_carbon_costs.append(annual_carbon_cost)
    
    enriched_data['marginal_cost_eur_mwh'] = marginal_costs
    enriched_data['annual_fuel_cost_eur'] = annual_fuel_costs
    enriched_data['annual_carbon_cost_eur'] = annual_carbon_costs
    
    # Calculate gross margins
    enriched_data['gross_margin_eur_mwh'] = power_price - enriched_data['marginal_cost_eur_mwh']
    
    # Calculate annual gross profit
    enriched_data['annual_gross_profit_eur'] = (
        enriched_data['gross_margin_eur_mwh'] * enriched_data['annual_generation_mwh']
    )
    
    # Calculate annual EBITDA (Gross Profit - Fixed Costs)
    enriched_data['annual_ebitda_eur'] = (
        enriched_data['annual_gross_profit_eur'] - enriched_data['annual_fixed_cost_eur']
    )
    
    # Dispatch feasibility
    enriched_data['is_economic_dispatch'] = enriched_data['gross_margin_eur_mwh'] > 0
    
    # Merit order ranking
    enriched_data['merit_order_rank'] = enriched_data['marginal_cost_eur_mwh'].rank()
    
    return enriched_data

def calculate_asset_performance_metrics(asset_data, historical_generation_data=None):
    """
    Calculate comprehensive performance metrics for assets
    """
    performance_metrics = {}
    
    for _, asset in asset_data.iterrows():
        asset_name = asset['name']
        
        # Basic performance metrics
        metrics = {
            'capacity_mw': asset['capacity_mw'],
            'availability_factor': asset['availability'],
            'capacity_factor': asset['availability'],  # For these assets, same as availability
            'annual_generation_mwh': asset.get('annual_generation_mwh', 
                                             asset['capacity_mw'] * 8760 * asset['availability']),
            'load_factor': asset['availability'],
            'efficiency': asset['efficiency'],
            'asset_age_years': datetime.now().year - asset['construction_year']
        }
        
        # Calculate performance ratios
        theoretical_max = asset['capacity_mw'] * 8760
        metrics['generation_ratio'] = metrics['annual_generation_mwh'] / theoretical_max
        
        # Technology-specific metrics
        if asset['type'] in ['Wind', 'Solar']:
            metrics['renewable_capacity_factor'] = asset['availability']
            metrics['co2_avoidance_tonnes'] = 0  # Renewables avoid emissions
        else:
            metrics['thermal_efficiency'] = asset['efficiency']
            metrics['annual_co2_emissions_tonnes'] = (
                metrics['annual_generation_mwh'] * asset['co2_intensity']
            )
        
        # Economic performance indicators (if market data available)
        if 'annual_ebitda_eur' in asset:
            metrics['ebitda_per_mw'] = asset['annual_ebitda_eur'] / asset['capacity_mw']
            metrics['ebitda_margin'] = (
                asset['annual_ebitda_eur'] / 
                (asset.get('annual_generation_mwh', 1) * asset.get('gross_margin_eur_mwh', 0) + 1)
            )
        
        performance_metrics[asset_name] = metrics
    
    return performance_metrics

def benchmark_assets(asset_data, benchmark_type='efficiency'):
    """
    Benchmark assets against portfolio averages and industry standards
    """
    benchmarks = {}
    
    # Portfolio averages by technology
    tech_benchmarks = asset_data.groupby('type').agg({
        'efficiency': 'mean',
        'availability': 'mean',
        'capacity_mw': 'mean',
        'asset_age_years': 'mean'
    }).to_dict()
    
    for _, asset in asset_data.iterrows():
        asset_name = asset['name']
        asset_type = asset['type']
        
        benchmark_results = {
            'asset_name': asset_name,
            'asset_type': asset_type,
            'benchmarks': {}
        }
        
        # Efficiency benchmark
        if asset_type in tech_benchmarks['efficiency']:
            portfolio_avg_eff = tech_benchmarks['efficiency'][asset_type]
            efficiency_vs_avg = asset['efficiency'] / portfolio_avg_eff - 1
            benchmark_results['benchmarks']['efficiency_vs_portfolio'] = efficiency_vs_avg
        
        # Availability benchmark
        if asset_type in tech_benchmarks['availability']:
            portfolio_avg_avail = tech_benchmarks['availability'][asset_type]
            availability_vs_avg = asset['availability'] / portfolio_avg_avail - 1
            benchmark_results['benchmarks']['availability_vs_portfolio'] = availability_vs_avg
        
        # Age benchmark
        if asset_type in tech_benchmarks['asset_age_years']:
            portfolio_avg_age = tech_benchmarks['asset_age_years'][asset_type]
            age_vs_avg = asset['asset_age_years'] - portfolio_avg_age
            benchmark_results['benchmarks']['age_vs_portfolio_years'] = age_vs_avg
        
        # Industry benchmarks (approximate values)
        industry_benchmarks = {
            'Gas': {'efficiency': 0.60, 'availability': 0.90},
            'Wind': {'efficiency': 1.0, 'availability': 0.35},
            'Solar': {'efficiency': 1.0, 'availability': 0.20},
            'Coal': {'efficiency': 0.45, 'availability': 0.85},
            'Nuclear': {'efficiency': 0.35, 'availability': 0.90}
        }
        
        if asset_type in industry_benchmarks:
            industry_eff = industry_benchmarks[asset_type]['efficiency']
            industry_avail = industry_benchmarks[asset_type]['availability']
            
            benchmark_results['benchmarks']['efficiency_vs_industry'] = (
                asset['efficiency'] / industry_eff - 1
            )
            benchmark_results['benchmarks']['availability_vs_industry'] = (
                asset['availability'] / industry_avail - 1
            )
        
        benchmarks[asset_name] = benchmark_results
    
    return benchmarks

def filter_assets(asset_data, filters):
    """
    Filter assets based on specified criteria
    
    filters = {
        'technology': ['Gas', 'Wind'],
        'capacity_min': 100,
        'capacity_max': 2000,
        'efficiency_min': 0.5,
        'age_max': 20,
        'country': ['Germany', 'Netherlands']
    }
    """
    filtered_data = asset_data.copy()
    
    # Technology filter
    if 'technology' in filters and filters['technology']:
        filtered_data = filtered_data[filtered_data['type'].isin(filters['technology'])]
    
    # Capacity filters
    if 'capacity_min' in filters:
        filtered_data = filtered_data[filtered_data['capacity_mw'] >= filters['capacity_min']]
    
    if 'capacity_max' in filters:
        filtered_data = filtered_data[filtered_data['capacity_mw'] <= filters['capacity_max']]
    
    # Efficiency filter
    if 'efficiency_min' in filters:
        filtered_data = filtered_data[filtered_data['efficiency'] >= filters['efficiency_min']]
    
    # Age filter
    if 'age_max' in filters:
        current_year = datetime.now().year
        asset_ages = current_year - filtered_data['construction_year']
        filtered_data = filtered_data[asset_ages <= filters['age_max']]
    
    # Country filter
    if 'country' in filters and filters['country']:
        filtered_data = filtered_data[filtered_data['country'].isin(filters['country'])]
    
    return filtered_data

def calculate_portfolio_summary(asset_data):
    """
    Calculate high-level portfolio summary statistics
    """
    if asset_data.empty:
        return {}
    
    summary = {
        # Capacity metrics
        'total_capacity_mw': asset_data['capacity_mw'].sum(),
        'asset_count': len(asset_data),
        'average_asset_size_mw': asset_data['capacity_mw'].mean(),
        'largest_asset_mw': asset_data['capacity_mw'].max(),
        'smallest_asset_mw': asset_data['capacity_mw'].min(),
        
        # Technology mix
        'technology_mix': asset_data.groupby('type')['capacity_mw'].sum().to_dict(),
        'technology_count': asset_data['type'].nunique(),
        
        # Performance metrics
        'weighted_avg_efficiency': (
            (asset_data['efficiency'] * asset_data['capacity_mw']).sum() / 
            asset_data['capacity_mw'].sum()
        ),
        'weighted_avg_availability': (
            (asset_data['availability'] * asset_data['capacity_mw']).sum() / 
            asset_data['capacity_mw'].sum()
        ),
        
        # Age analysis
        'average_age_years': (datetime.now().year - asset_data['construction_year']).mean(),
        'oldest_asset_years': datetime.now().year - asset_data['construction_year'].min(),
        'newest_asset_years': datetime.now().year - asset_data['construction_year'].max(),
        
        # Generation potential
        'total_annual_generation_potential_mwh': (
            asset_data['capacity_mw'] * 8760 * asset_data['availability']
        ).sum(),
        
        # Geographic distribution
        'countries': asset_data['country'].nunique(),
        'country_distribution': asset_data.groupby('country')['capacity_mw'].sum().to_dict(),
        
        # Environmental metrics
        'total_annual_co2_potential_tonnes': (
            asset_data['capacity_mw'] * 8760 * asset_data['availability'] * 
            asset_data['co2_intensity']
        ).sum(),
        'renewable_share': (
            asset_data[asset_data['type'].isin(['Wind', 'Solar', 'Hydro'])]['capacity_mw'].sum() /
            asset_data['capacity_mw'].sum()
        )
    }
    
    return summary

def get_asset_by_name(asset_name):
    """
    Retrieve specific asset data by name
    """
    assets_df = load_asset_portfolio()
    if not assets_df.empty:
        asset_match = assets_df[assets_df['name'] == asset_name]
        if not asset_match.empty:
            return asset_match.iloc[0].to_dict()
    return None

def update_asset_data(asset_name, updates):
    """
    Update specific asset parameters
    Note: This is for demonstration - in production, would update database
    """
    assets_df = load_asset_portfolio()
    
    if not assets_df.empty:
        asset_index = assets_df[assets_df['name'] == asset_name].index
        if not asset_index.empty:
            for field, value in updates.items():
                if field in assets_df.columns:
                    assets_df.loc[asset_index[0], field] = value
            
            # Clear cache to force reload
            st.cache_data.clear()
            return True
    return False