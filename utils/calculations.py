"""
Utility calculation functions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def format_currency(value, currency='EUR', millions=False):
    """
    Format currency values for display
    """
    if millions:
        value = value / 1e6
        suffix = 'M'
    else:
        suffix = ''
    
    if abs(value) >= 1e9:
        return f"{currency} {value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{currency} {value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{currency} {value/1e3:.1f}k"
    else:
        return f"{currency} {value:.0f}{suffix}"

def format_percentage(value, decimal_places=1):
    """
    Format percentage values
    """
    return f"{value * 100:.{decimal_places}f}%"

def calculate_capacity_factor(actual_generation, capacity_mw, time_period_hours):
    """
    Calculate capacity factor
    """
    theoretical_max = capacity_mw * time_period_hours
    return actual_generation / theoretical_max if theoretical_max > 0 else 0

def calculate_availability(operational_hours, total_hours):
    """
    Calculate asset availability
    """
    return operational_hours / total_hours if total_hours > 0 else 0

def heat_rate_to_efficiency(heat_rate_btu_kwh):
    """
    Convert heat rate to efficiency
    Heat rate in BTU/kWh to efficiency percentage
    """
    return 3412 / heat_rate_btu_kwh if heat_rate_btu_kwh > 0 else 0

def efficiency_to_heat_rate(efficiency):
    """
    Convert efficiency to heat rate
    """
    return 3412 / efficiency if efficiency > 0 else 0

def calculate_emissions_intensity(fuel_type, efficiency=1.0):
    """
    Calculate CO2 emissions intensity by fuel type
    Returns tCO2/MWh
    """
    emission_factors = {
        'Natural Gas': 0.202,  # tCO2/MWh thermal
        'Coal': 0.341,
        'Oil': 0.279,
        'Nuclear': 0.0,
        'Wind': 0.0,
        'Solar': 0.0,
        'Hydro': 0.0
    }
    
    base_emission = emission_factors.get(fuel_type, 0.0)
    return base_emission / efficiency if efficiency > 0 else base_emission

def calculate_load_factor(actual_output, rated_capacity, time_period):
    """
    Calculate load factor (average load / peak load)
    """
    avg_load = actual_output / time_period if time_period > 0 else 0
    return avg_load / rated_capacity if rated_capacity > 0 else 0

def calculate_equivalent_availability_factor(available_hours, total_hours, forced_outage_hours):
    """
    Calculate Equivalent Availability Factor (EAF)
    """
    planned_hours = total_hours - available_hours - forced_outage_hours
    return (total_hours - forced_outage_hours) / total_hours if total_hours > 0 else 0

def interpolate_price_curve(price_points, demand_points, target_demand):
    """
    Interpolate price from demand curve
    """
    return np.interp(target_demand, demand_points, price_points)

def calculate_price_elasticity(price_change, demand_change):
    """
    Calculate price elasticity of demand
    """
    if price_change != 0:
        return demand_change / price_change
    return 0

def moving_average(data, window_size):
    """
    Calculate moving average
    """
    return pd.Series(data).rolling(window=window_size, center=True).mean()

def exponential_smoothing(data, alpha=0.3):
    """
    Apply exponential smoothing to time series data
    """
    smoothed = [data[0]]
    for i in range(1, len(data)):
        smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i-1])
    return smoothed

def calculate_statistical_summary(data):
    """
    Calculate comprehensive statistical summary
    """
    data_array = np.array(data)
    
    return {
        'count': len(data_array),
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std': np.std(data_array),
        'min': np.min(data_array),
        'max': np.max(data_array),
        'q25': np.percentile(data_array, 25),
        'q75': np.percentile(data_array, 75),
        'skewness': calculate_skewness(data_array),
        'kurtosis': calculate_kurtosis(data_array)
    }

def calculate_skewness(data):
    """
    Calculate skewness of data
    """
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data):
    """
    Calculate kurtosis of data
    """
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def convert_units(value, from_unit, to_unit):
    """
    Convert between common energy units
    """
    # Conversion factors to MWh
    conversion_factors = {
        'MWh': 1.0,
        'GWh': 1000.0,
        'kWh': 0.001,
        'TWh': 1000000.0
    }
    
    if from_unit in conversion_factors and to_unit in conversion_factors:
        # Convert to MWh first, then to target unit
        mwh_value = value * conversion_factors[from_unit]
        return mwh_value / conversion_factors[to_unit]
    
    return value  # Return original if conversion not available