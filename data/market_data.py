"""
Market data processing functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.data_fetcher import fetch_electricity_prices, fetch_gas_prices, fetch_carbon_prices

def get_market_overview():
    """
    Get comprehensive market overview data
    """
    electricity_data = fetch_electricity_prices(days_back=30)
    gas_data = fetch_gas_prices()
    carbon_data = fetch_carbon_prices()
    
    # Calculate market metrics
    current_power_price = electricity_data['price_eur_mwh'].iloc[-1]
    avg_power_price = electricity_data['price_eur_mwh'].mean()
    power_volatility = electricity_data['price_eur_mwh'].std()
    
    market_data = {
        'current_power_price': current_power_price,
        'avg_power_price_30d': avg_power_price,
        'power_price_volatility': power_volatility,
        'gas_price': gas_data['price_eur_mwh'],
        'carbon_price': carbon_data['price_eur_tonne'],
        'gas_change_24h': gas_data.get('change_24h', 0),
        'carbon_change_24h': carbon_data.get('change_24h', 0),
        'timestamp': datetime.now()
    }
    
    return market_data, electricity_data

def calculate_spark_spread(electricity_price, gas_price, efficiency=0.58, carbon_price=85):
    """
    Calculate spark spread for gas power plants
    Spark Spread = Electricity Price - (Gas Price / Efficiency + Carbon Cost)
    """
    carbon_cost = carbon_price * 0.35  # Assuming 0.35 tCO2/MWh
    fuel_cost = gas_price / efficiency
    
    spark_spread = electricity_price - fuel_cost - carbon_cost
    return spark_spread

def calculate_clean_spark_spread(electricity_price, gas_price, efficiency=0.58, carbon_price=85, co2_intensity=0.35):
    """
    Calculate clean spark spread including carbon costs
    """
    fuel_cost = gas_price / efficiency
    carbon_cost = carbon_price * co2_intensity
    
    clean_spark_spread = electricity_price - fuel_cost - carbon_cost
    return clean_spark_spread

def analyze_price_trends(price_data):
    """
    Analyze electricity price trends
    """
    df = price_data.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['date'] = df['timestamp'].dt.date
    
    # Calculate trend metrics
    trends = {
        'hourly_avg': df.groupby('hour')['price_eur_mwh'].mean(),
        'daily_avg': df.groupby('date')['price_eur_mwh'].mean(),
        'weekly_pattern': df.groupby('day_of_week')['price_eur_mwh'].mean(),
        'price_range': {
            'min': df['price_eur_mwh'].min(),
            'max': df['price_eur_mwh'].max(),
            'mean': df['price_eur_mwh'].mean(),
            'std': df['price_eur_mwh'].std()
        }
    }
    
    return trends

def calculate_merit_order_position(asset_data, market_data):
    """
    Calculate where assets sit in the merit order
    """
    gas_price = market_data['gas_price']
    carbon_price = market_data['carbon_price']
    
    positions = []
    for _, asset in asset_data.iterrows():
        if asset['type'] in ['Wind', 'Solar']:
            marginal_cost = asset['variable_cost']
        else:
            fuel_cost = gas_price / asset['efficiency'] if asset['fuel_type'] == 'Natural Gas' else 0
            carbon_cost = carbon_price * asset['co2_intensity']
            marginal_cost = asset['variable_cost'] + fuel_cost + carbon_cost
        
        positions.append({
            'name': asset['name'],
            'type': asset['type'],
            'marginal_cost': marginal_cost,
            'capacity_mw': asset['capacity_mw']
        })
    
    return pd.DataFrame(positions).sort_values('marginal_cost')

def calculate_market_indicators():
    """
    Calculate key market indicators
    """
    market_data, price_data = get_market_overview()
    
    # System marginal price indicators
    current_price = market_data['current_power_price']
    
    # Price level indicators
    if current_price < 30:
        price_regime = "Low"
        regime_color = "green"
    elif current_price < 80:
        price_regime = "Medium"
        regime_color = "orange"
    else:
        price_regime = "High"
        regime_color = "red"
    
    # Volatility indicators
    volatility = market_data['power_price_volatility']
    if volatility < 15:
        volatility_regime = "Low"
    elif volatility < 30:
        volatility_regime = "Medium"  
    else:
        volatility_regime = "High"
    
    indicators = {
        'price_regime': price_regime,
        'price_regime_color': regime_color,
        'volatility_regime': volatility_regime,
        'spark_spread': calculate_spark_spread(
            current_price, 
            market_data['gas_price']
        ),
        'market_heat': min(100, max(0, (current_price - 20) * 2))  # 0-100 scale
    }
    
    return indicators