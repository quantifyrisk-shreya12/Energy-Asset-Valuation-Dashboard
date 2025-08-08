"""
Real-time data fetching functions
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
from config import ENTSO_E_API_TOKEN, OPENWEATHER_API_KEY, ENTSO_E_BASE_URL

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_electricity_prices(country_code='10Y1001A1001A83F', days_back=7):
    """
    Fetch real-time electricity prices from ENTSO-E API
    If API token not available, return synthetic data
    """
    if not ENTSO_E_API_TOKEN:
        st.warning("⚠️ ENTSO-E API token not configured. Using synthetic data for demo.")
        return generate_synthetic_price_data(days_back)
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            'securityToken': ENTSO_E_API_TOKEN,
            'documentType': 'A44',  # Price document type
            'in_Domain': country_code,
            'out_Domain': country_code,
            'periodStart': start_date.strftime('%Y%m%d%H%M'),
            'periodEnd': end_date.strftime('%Y%m%d%H%M')
        }
        
        response = requests.get(ENTSO_E_BASE_URL, params=params)
        
        if response.status_code == 200:
            # Parse XML response (simplified for demo)
            return parse_entso_e_response(response.text)
        else:
            st.warning("API call failed. Using synthetic data.")
            return generate_synthetic_price_data(days_back)
            
    except Exception as e:
        st.warning(f"Error fetching data: {e}. Using synthetic data.")
        return generate_synthetic_price_data(days_back)

def generate_synthetic_price_data(days_back=7):
    """
    Generate realistic synthetic electricity price data
    """
    hours = days_back * 24
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=days_back), 
        periods=hours, 
        freq='h'
    )
    
    # Create realistic price pattern
    base_price = 65.0
    daily_pattern = np.sin(np.arange(hours) * 2 * np.pi / 24) * 20
    weekly_pattern = np.sin(np.arange(hours) * 2 * np.pi / (24 * 7)) * 10
    noise = np.random.normal(0, 5, hours)
    
    prices = base_price + daily_pattern + weekly_pattern + noise
    prices = np.maximum(prices, 5)  # Ensure positive prices
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'price_eur_mwh': prices,
        'country': 'Germany'
    })

@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_gas_prices():
    """
    Fetch natural gas prices (TTF) using yfinance
    """
    try:
        # TTF Natural Gas futures
        gas = yf.Ticker("TTF=F")
        hist = gas.history(period="1mo")
        
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1]
            # Convert from currency per unit to EUR/MWh (approximate conversion)
            gas_price_eur_mwh = latest_price * 9.77  # Rough conversion factor
            
            return {
                'price_eur_mwh': gas_price_eur_mwh,
                'timestamp': hist.index[-1],
                'change_24h': (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
            }
    except:
        pass
    
    # Fallback synthetic data
    return {
        'price_eur_mwh': 45.0 + np.random.normal(0, 2),
        'timestamp': datetime.now(),
        'change_24h': np.random.normal(0, 3)
    }

@st.cache_data(ttl=600)
def fetch_carbon_prices():
    """
    Fetch EU ETS carbon prices
    """
    try:
        # Using a carbon price approximation
        # In real implementation, you'd use specific carbon futures data
        carbon = yf.Ticker("^ICEEUA")  # This might not work, fallback to synthetic
        hist = carbon.history(period="1mo")
        
        if not hist.empty:
            return {
                'price_eur_tonne': hist['Close'].iloc[-1],
                'timestamp': hist.index[-1],
                'change_24h': (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
            }
    except:
        pass
    
    # Synthetic carbon price data
    base_carbon_price = 85.0
    return {
        'price_eur_tonne': base_carbon_price + np.random.normal(0, 3),
        'timestamp': datetime.now(),
        'change_24h': np.random.normal(0, 2)
    }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_weather_data(city="Berlin"):
    """
    Fetch weather data affecting renewable generation
    """
    if not OPENWEATHER_API_KEY:
        return generate_synthetic_weather_data()
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'wind_speed': data['wind']['speed'],
                'cloud_cover': data['clouds']['all'],
                'solar_irradiance': max(0, 1000 * (1 - data['clouds']['all'] / 100)),
                'timestamp': datetime.now()
            }
    except:
        pass
    
    return generate_synthetic_weather_data()

def generate_synthetic_weather_data():
    """
    Generate synthetic weather data
    """
    return {
        'temperature': np.random.normal(15, 8),
        'wind_speed': max(0, np.random.normal(8, 3)),
        'cloud_cover': max(0, min(100, np.random.normal(50, 25))),
        'solar_irradiance': max(0, np.random.normal(600, 200)),
        'timestamp': datetime.now()
    }

def parse_entso_e_response(xml_data):
    """
    Parse ENTSO-E XML response (simplified)
    In real implementation, you'd use xmltodict or similar
    """
    # This is a placeholder - real implementation would parse XML
    # For now, return synthetic data
    return generate_synthetic_price_data(7)

### got this code from kimi
@st.cache_data(ttl=300) 
def fetch_instant_european_data():
    """
    Fallback function for immediate European market data
    Uses synthetic data that looks realistic for demo purposes
    """
    print('I am in this function fetch_instant_european_data')
    return generate_synthetic_price_data(7)

@st.cache_data(ttl=600)
def fetch_nordpool_prices():
    """
    Fetch Nord Pool day-ahead prices for Germany (immediate source)
    """
    try:
        # Nord Pool API for Germany/Luxembourg bidding zone
        url = "https://www.nordpoolgroup.com/api/marketdata/page/10"
        params = {"currency": "EUR"}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'Rows' in data['data']:
                rows = data['data']['Rows']
                prices = []
                timestamps = []
                
                for row in rows:
                    if 'Columns' in row and len(row['Columns']) > 1:
                        try:
                            price = float(row['Columns'][1]['Value'].replace(',', '.'))
                            hour = int(row['Name'].split('-')[0])
                            timestamp = datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
                            prices.append(price)
                            timestamps.append(timestamp)
                        except:
                            continue
                
                if prices and timestamps:
                    return pd.DataFrame({
                        'timestamp': timestamps,
                        'price_eur_mwh': prices,
                        'country': 'Germany'
                    })
    except:
        pass
    print('I am in this function Fetch _northpool _prices')
    return generate_synthetic_price_data(7)

def generate_enhanced_synthetic_data(days=30):
    """
    Enhanced synthetic data with more realistic patterns
    """
    print('I am in this function Generic _innant _synthetic _data')
    hours = days * 24
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=days), 
        periods=hours, 
        freq='h'
    )
    
    # More sophisticated patterns
    base_price = 68.5
    daily_pattern = np.sin(np.arange(hours) * 2 * np.pi / 24) * 25
    weekly_pattern = np.sin(np.arange(hours) * 2 * np.pi / (24 * 7)) * 12
    seasonal_pattern = np.sin(np.arange(hours) * 2 * np.pi / (24 * 30)) * 8
    
    # Add realistic volatility
    volatility = np.random.normal(0, 6, hours)
    
    # Combine patterns
    prices = base_price + daily_pattern + weekly_pattern + seasonal_pattern + volatility
    prices = np.maximum(prices, 8)  # Ensure minimum price
    
    # Add correlated gas and carbon prices
    gas_prices = 42 + np.random.normal(0, 4, hours) + (prices - base_price) * 0.3
    carbon_prices = 82 + np.random.normal(0, 5, hours) + (prices - base_price) * 0.2
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'price_eur_mwh': prices,
        'country': 'Germany',
        'gas_price': np.maximum(gas_prices, 25),
        'carbon_price': np.maximum(carbon_prices, 50)
    })