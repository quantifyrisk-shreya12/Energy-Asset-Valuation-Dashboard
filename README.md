# Energy Asset Valuation Dashboard

A comprehensive energy asset strategy and valuation platform demonstrating end-to-end capabilities for power generation portfolio analysis.

## 🎯 Project Overview

This dashboard demonstrates key capabilities required for **Asset Strategy & Valuation** roles in energy companies like Uniper:

- **📊 Real-time Market Analysis**: Live electricity, gas, and carbon price monitoring
- **🏭 Asset Performance Analytics**: Technical and economic analysis of power plants
- **💰 Financial Modeling**: DCF valuation, Monte Carlo simulation, sensitivity analysis
- **🎯 Strategic Planning**: Scenario analysis, portfolio optimization, hedging strategies

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start

1. **Clone/Download Project**
```bash
# Create project directory
mkdir energy_asset_dashboard
cd energy_asset_dashboard
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys (optional)
# Without keys, synthetic data will be used
```

4. **Run Application**
```bash
streamlit run main.py
```

5. **Access Dashboard**
Open browser to: `http://localhost:8501`

## 📋 Features & Capabilities

### 🏠 Home Dashboard
- Project overview and capabilities
- Quick market snapshot
- Navigation to analysis modules

### 📊 Market Overview
- **Real-time Data**: ENTSO-E electricity prices, gas/carbon prices
- **Market Analytics**: Price trends, volatility analysis, correlation matrices
- **Spark Spread Analysis**: Gas plant economics and dispatch signals
- **Price Forecasting**: Linear trend and seasonal pattern analysis

### 🏭 Asset Analysis
- **Portfolio Overview**: Technology mix, capacity analysis
- **Asset Performance**: Efficiency, availability, generation metrics
- **Merit Order**: Cost stack positioning and dispatch economics
- **Benchmarking**: Cross-asset performance comparison

### 💰 Valuation Models
- **DCF Analysis**: Net Present Value, IRR calculations
- **Monte Carlo Simulation**: Risk assessment with probability distributions
- **Sensitivity Analysis**: Parameter impact on asset values
- **LCOE Calculation**: Levelized cost of energy analysis
- **Portfolio Valuation**: Aggregated portfolio metrics

### 🎯 Scenario Planning
- **Market Scenarios**: Base case, high/low price, green transition
- **Price Forecasting**: 7-90 day price predictions
- **Portfolio Optimization**: Strategic asset allocation
- **Hedging Strategy**: Risk management recommendations

## 🔧 Technical Architecture

### **Functional Programming Approach**
- Pure functions for calculations and data processing
- Modular design with clear separation of concerns
- Stateless operations for reliability and testing

### **Data Layer**
- **Real-time APIs**: ENTSO-E Transparency Platform, financial data
- **Fallback Data**: Synthetic data generation when APIs unavailable
- **Caching**: Streamlit caching for performance optimization

### **Models Layer**
- **Financial Models**: DCF, NPV, IRR, LCOE calculations
- **Market Analytics**: Price forecasting, correlation analysis
- **Optimization**: Portfolio optimization, dispatch scheduling

### **Visualization Layer**
- **Interactive Charts**: Plotly-based dashboards
- **Responsive Design**: Multi-column layouts, mobile-friendly
- **Real-time Updates**: Auto-refreshing market data

## 📊 Sample Data

The dashboard includes realistic sample data representing a typical European power generation portfolio:

### Assets Included:
- **Maasvlakte 3** (1,070 MW Gas) - Netherlands
- **Scholven B/C** (760 MW Gas) - Germany  
- **Grain Power** (1,275 MW Gas) - UK
- **Nord Stream Wind** (385 MW Wind) - Germany
- **Provence Solar** (150 MW Solar) - France

### Market Data:
- European electricity prices (Germany/Netherlands focus)
- Natural gas prices (TTF reference)
- EU ETS carbon prices
- Weather data affecting renewable generation

## 🔌 API Integration

### Live Data Sources:
- **ENTSO-E Transparency Platform**: European electricity market data
- **Financial APIs**: Gas and carbon price feeds  
- **OpenWeatherMap**: Weather data for renewable forecasting

### Getting API Keys:
1. **ENTSO-E Token**: Register at [transparency.entsoe.eu](https://transparency.entsoe.eu/)
2. **Weather API**: Get free key at [openweathermap.org](https://openweathermap.org/api)

*Note: Dashboard works with synthetic data if API keys not available*

## 💡 Business Value Demonstration

### For Asset Strategy Teams:
- **Market Intelligence**: Real-time price monitoring and trend analysis
- **Investment Analysis**: DCF models supporting business case development
- **Risk Management**: Monte Carlo simulations and scenario planning
- **Portfolio Optimization**: Data-driven asset allocation recommendations

### For Cross-functional Collaboration:
- **Interactive Dashboards**: Self-service analytics for stakeholders
- **Scenario Planning**: Strategic planning tools for management
- **Performance Monitoring**: KPI tracking and benchmarking
- **Decision Support**: Model-based analysis for investment decisions

## 🎯 Interview Demonstration Points

### Technical Capabilities:
- **End-to-end Implementation**: From data ingestion to strategic recommendations
- **Real-time Integration**: Live market data processing and analysis
- **Financial Modeling**: Sophisticated valuation techniques
- **Risk Analytics**: Monte Carlo, sensitivity, and scenario analysis

### Business Understanding:
- **Energy Market Knowledge**: European power markets, merit order, spark spreads
- **Asset Strategy**: Portfolio optimization, dispatch economics, hedging
- **Financial Analysis**: DCF modeling, LCOE calculations, risk assessment
- **Strategic Planning**: Scenario analysis, market forecasting

## 🚀 Future Enhancements

### Advanced Analytics:
- Machine learning price forecasting models
- Optimization algorithms for dispatch scheduling
- Real-time portfolio rebalancing
- Advanced risk metrics (VaR, CVaR)

### Extended Data Sources:
- Additional European markets
- Forward curve integration
- Fundamental market drivers
- Regulatory impact analysis

### Enhanced Visualization:
- Geographic mapping of assets
- 3D portfolio optimization surfaces  
- Interactive scenario builders
- Custom reporting tools

## 📞 Support & Documentation

For questions about implementation or energy market analysis:
- Review code comments for detailed explanations
- Check config.py for parameter adjustments
- Examine sample data structures in data/sample_assets.py
- Reference energy industry best practices in model implementations

---

*Built to demonstrate comprehensive energy asset strategy and valuation capabilities for interview purposes.*