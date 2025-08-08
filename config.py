"""
Configuration settings for Energy Asset Valuation Dashboard
"""

import os
from datetime import datetime, timedelta

# API Configuration
ENTSO_E_API_TOKEN = os.getenv('ENTSO_E_TOKEN', '')  # Get free token from transparency.entsoe.eu
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')  # Get free key from openweathermap.org

# Market Data URLs
ENTSO_E_BASE_URL = "https://transparency.entsoe.eu/api"

# Time configurations
DEFAULT_START_DATE = datetime.now() - timedelta(days=30)
DEFAULT_END_DATE = datetime.now()

# Market configurations
EUROPEAN_BIDDING_ZONES = {
    'Germany': '10Y1001A1001A83F',
    'France': '10Y1001A1001A92E', 
    'Netherlands': '10Y1001A1001A92E',
    'UK': '10Y1001A1001A92E'
}

# Asset configurations
ASSET_TYPES = ['Gas', 'Coal', 'Nuclear', 'Wind', 'Solar', 'Hydro']
DEFAULT_DISCOUNT_RATE = 0.07
DEFAULT_CARBON_PRICE = 85.0  # EUR/tonne CO2

# Financial model parameters
DEFAULT_PROJECT_LIFE = 25  # years
TAX_RATE = 0.25
INFLATION_RATE = 0.02

# Visualization settings
PLOT_THEME = 'plotly_white'
PRIMARY_COLOR = '#1f77b4'
SECONDARY_COLOR = '#ff7f0e'