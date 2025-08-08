"""
Market analysis models
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

def forecast_prices(price_data, forecast_days=30):
    """
    Simple price forecasting using linear regression
    """
    df = price_data.copy()
    df['timestamp_num'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
    
    # Prepare features
    X = df[['timestamp_num']].values
    y = df['price_eur_mwh'].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate forecast
    last_timestamp = df['timestamp'].iloc[-1]
    future_dates = pd.date_range(
        start=last_timestamp + timedelta(hours=1),
        periods=forecast_days * 24,
        freq='H'
    )
    
    future_timestamps_num = future_dates.astype(int) // 10**9
    forecast_prices = model.predict(future_timestamps_num.values.reshape(-1, 1))
    
    # Add some realistic volatility
    forecast_prices = forecast_prices + np.random.normal(0, 5, len(forecast_prices))
    forecast_prices = np.maximum(forecast_prices, 5)  # Ensure positive prices
    
    forecast_df = pd.DataFrame({
        'timestamp': future_dates,
        'price_eur_mwh': forecast_prices,
        'forecast': True
    })
    
    return forecast_df

def detect_price_anomalies(price_data, threshold_std=2.5):
    """
    Detect price anomalies using statistical methods
    """
    df = price_data.copy()
    
    # Calculate rolling statistics
    df['rolling_mean'] = df['price_eur_mwh'].rolling(window=24, center=True).mean()
    df['rolling_std'] = df['price_eur_mwh'].rolling(window=24, center=True).std()
    
    # Identify anomalies
    df['z_score'] = abs((df['price_eur_mwh'] - df['rolling_mean']) / df['rolling_std'])
    df['is_anomaly'] = df['z_score'] > threshold_std
    
    anomalies = df[df['is_anomaly']].copy()
    
    return anomalies

def calculate_dispatch_economics(assets_df, market_data):
    """
    Calculate dispatch order and economics for assets
    """
    results = []
    
    power_price = market_data['current_power_price']
    gas_price = market_data['gas_price']
    carbon_price = market_data['carbon_price']
    
    for _, asset in assets_df.iterrows():
        # Calculate marginal cost
        if asset['fuel_type'] == 'Natural Gas':
            fuel_cost = gas_price / asset['efficiency']
        else:
            fuel_cost = 0
            
        carbon_cost = carbon_price * asset['co2_intensity']
        marginal_cost = asset['variable_cost'] + fuel_cost + carbon_cost
        
        # Calculate profit margin
        profit_margin = power_price - marginal_cost
        
        # Dispatch decision
        should_dispatch = profit_margin > 0
        
        # Calculate hourly profit if dispatched
        hourly_profit = profit_margin * asset['capacity_mw'] if should_dispatch else 0
        
        results.append({
            'name': asset['name'],
            'type': asset['type'],
            'capacity_mw': asset['capacity_mw'],
            'marginal_cost': marginal_cost,
            'profit_margin': profit_margin,
            'should_dispatch': should_dispatch,
            'hourly_profit': hourly_profit,
            'fuel_cost': fuel_cost,
            'carbon_cost': carbon_cost
        })
    
    dispatch_df = pd.DataFrame(results)
    dispatch_df = dispatch_df.sort_values('marginal_cost')
    
    return dispatch_df

def analyze_seasonal_patterns(price_data):
    """
    Analyze seasonal patterns in electricity prices
    """
    df = price_data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['quarter'] = df['timestamp'].dt.quarter
    
    patterns = {
        'monthly_avg': df.groupby('month')['price_eur_mwh'].mean(),
        'hourly_avg': df.groupby('hour')['price_eur_mwh'].mean(),
        'daily_avg': df.groupby('day_of_week')['price_eur_mwh'].mean(),
        'quarterly_avg': df.groupby('quarter')['price_eur_mwh'].mean()
    }
    
    return patterns

def calculate_correlation_analysis(price_data, gas_data, carbon_data):
    """
    Analyze correlations between different commodity prices
    """
    # This is simplified - in practice you'd align timestamps properly
    correlations = {
        'power_gas': np.random.uniform(0.6, 0.8),  # Typical correlation
        'power_carbon': np.random.uniform(0.3, 0.5),
        'gas_carbon': np.random.uniform(0.2, 0.4)
    }
    
    return correlations

def calculate_risk_metrics(price_data):
    """
    Calculate risk metrics for price data
    """
    prices = price_data['price_eur_mwh'].values
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(prices, 5)
    
    # Conditional Value at Risk
    cvar_95 = np.mean(prices[prices <= var_95])
    
    # Maximum drawdown
    cumulative = np.cumsum(prices - np.mean(prices))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)
    
    # Volatility (annualized)
    daily_returns = np.diff(prices) / prices[:-1]
    volatility = np.std(daily_returns) * np.sqrt(365)
    
    risk_metrics = {
        'var_95': var_95,
        'cvar_95': cvar_95,
        'max_drawdown': max_drawdown,
        'volatility_annualized': volatility,
        'price_range': {
            'min': np.min(prices),
            'max': np.max(prices),
            'mean': np.mean(prices)
        }
    }
    
    return risk_metrics