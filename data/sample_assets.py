"""
Sample power plant data for demonstration
"""

import pandas as pd

def get_sample_assets():
    """
    Returns sample power plant data similar to Uniper's portfolio
    """
    assets_data = [
        {
            'name': 'Maasvlakte 3',
            'type': 'Gas',
            'capacity_mw': 1070,
            'efficiency': 0.59,
            'variable_cost': 3.5,  # EUR/MWh
            'fixed_cost': 45000,   # EUR/MW/year
            'co2_intensity': 0.35, # tCO2/MWh
            'availability': 0.92,
            'construction_year': 2013,
            'country': 'Netherlands',
            'fuel_type': 'Natural Gas'
        },
        {
            'name': 'Scholven B/C',
            'type': 'Gas',
            'capacity_mw': 760,
            'efficiency': 0.58,
            'variable_cost': 4.0,
            'fixed_cost': 48000,
            'co2_intensity': 0.37,
            'availability': 0.89,
            'construction_year': 2016,
            'country': 'Germany',
            'fuel_type': 'Natural Gas'
        },
        {
            'name': 'Grain Power',
            'type': 'Gas',
            'capacity_mw': 1275,
            'efficiency': 0.57,
            'variable_cost': 3.8,
            'fixed_cost': 46000,
            'co2_intensity': 0.36,
            'availability': 0.91,
            'construction_year': 2010,
            'country': 'UK',
            'fuel_type': 'Natural Gas'
        },
        {
            'name': 'Nord Stream Wind',
            'type': 'Wind',
            'capacity_mw': 385,
            'efficiency': 1.0,
            'variable_cost': 2.0,
            'fixed_cost': 95000,
            'co2_intensity': 0.0,
            'availability': 0.35,  # Capacity factor for wind
            'construction_year': 2020,
            'country': 'Germany',
            'fuel_type': 'Wind'
        },
        {
            'name': 'Provence Solar',
            'type': 'Solar',
            'capacity_mw': 150,
            'efficiency': 1.0,
            'variable_cost': 1.0,
            'fixed_cost': 75000,
            'co2_intensity': 0.0,
            'availability': 0.18,  # Capacity factor for solar
            'construction_year': 2019,
            'country': 'France',
            'fuel_type': 'Solar'
        }
    ]
    
    return pd.DataFrame(assets_data)

def get_fuel_prices():
    """
    Sample fuel price data (EUR/MWh thermal)
    """
    return {
        'Natural Gas': 45.0,
        'Coal': 12.0,
        'Uranium': 3.0,
        'Wind': 0.0,
        'Solar': 0.0
    }