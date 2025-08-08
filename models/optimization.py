"""
Portfolio optimization functions
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from models.financial_models import calculate_asset_dcf

def optimize_dispatch_schedule(assets_df, price_forecast, constraints=None):
    """
    Optimize asset dispatch schedule based on price forecast
    """
    results = []
    
    for hour_idx, (timestamp, price) in enumerate(zip(price_forecast['timestamp'], price_forecast['price_eur_mwh'])):
        hour_dispatch = []
        
        for _, asset in assets_df.iterrows():
            # Calculate marginal cost (simplified)
            marginal_cost = asset['variable_cost'] + 30  # Approximate fuel + carbon cost
            
            # Dispatch if profitable
            if price > marginal_cost and asset['type'] in ['Gas', 'Coal']:
                dispatch_mw = asset['capacity_mw'] * asset['availability']
                profit = (price - marginal_cost) * dispatch_mw
                
                hour_dispatch.append({
                    'timestamp': timestamp,
                    'asset': asset['name'],
                    'dispatch_mw': dispatch_mw,
                    'profit': profit
                })
        
        results.extend(hour_dispatch)
    
    return pd.DataFrame(results)

def calculate_portfolio_efficient_frontier(assets_df, market_scenarios, risk_free_rate=0.02):
    """
    Calculate efficient frontier for portfolio optimization
    """
    n_assets = len(assets_df)
    n_scenarios = len(market_scenarios)
    
    # Calculate returns for each asset under each scenario
    returns_matrix = np.zeros((n_assets, n_scenarios))
    
    for i, (_, asset) in enumerate(assets_df.iterrows()):
        for j, scenario in enumerate(market_scenarios):
            valuation = calculate_asset_dcf(asset, scenario)
            returns_matrix[i, j] = valuation['irr'] if not np.isnan(valuation['irr']) else 0.05
    
    # Calculate expected returns and covariance matrix
    expected_returns = np.mean(returns_matrix, axis=1)
    cov_matrix = np.cov(returns_matrix)
    
    # Generate efficient frontier points
    target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 20)
    efficient_frontier = []
    
    for target_return in target_returns:
        # Minimize portfolio variance subject to target return
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}  # Target return
        ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))  # No short selling
        
        result = minimize(objective, np.ones(n_assets) / n_assets, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            portfolio_return = target_return
            portfolio_risk = np.sqrt(result.fun)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
            
            efficient_frontier.append({
                'expected_return': portfolio_return,
                'volatility': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'weights': result.x
            })
    
    return efficient_frontier

def optimize_maintenance_schedule(assets_df, price_forecast, maintenance_requirements):
    """
    Optimize maintenance scheduling to minimize opportunity cost
    """
    optimization_results = []
    
    for _, asset in assets_df.iterrows():
        asset_name = asset['name']
        
        if asset_name not in maintenance_requirements:
            continue
            
        maintenance_duration = maintenance_requirements[asset_name].get('duration_hours', 72)
        
        # Find optimal maintenance window (lowest price period)
        price_forecast_sorted = price_forecast.sort_values('price_eur_mwh')
        
        # Find consecutive low-price periods
        best_start_time = price_forecast_sorted.iloc[0]['timestamp']
        min_opportunity_cost = float('inf')
        
        for i in range(len(price_forecast) - maintenance_duration):
            window = price_forecast.iloc[i:i+maintenance_duration]
            opportunity_cost = window['price_eur_mwh'].sum() * asset['capacity_mw']
            
            if opportunity_cost < min_opportunity_cost:
                min_opportunity_cost = opportunity_cost
                best_start_time = window.iloc[0]['timestamp']
        
        optimization_results.append({
            'asset': asset_name,
            'optimal_start_time': best_start_time,
            'opportunity_cost': min_opportunity_cost,
            'duration_hours': maintenance_duration
        })
    
    return pd.DataFrame(optimization_results)

def calculate_hedging_strategy(assets_df, price_forecast, hedge_ratio=0.5):
    """
    Calculate optimal hedging strategy for power generation portfolio
    """
    hedging_recommendations = []
    
    total_generation_forecast = 0
    
    for _, asset in assets_df.iterrows():
        # Estimate generation
        annual_generation = asset['capacity_mw'] * 8760 * asset['availability']
        total_generation_forecast += annual_generation
        
        # Calculate hedging recommendation
        hedge_volume = annual_generation * hedge_ratio
        
        # Estimate hedge price (simplified)
        avg_forecast_price = price_forecast['price_eur_mwh'].mean()
        hedge_price = avg_forecast_price * 0.95  # Slight discount for forward contracts
        
        hedging_recommendations.append({
            'asset': asset['name'],
            'generation_forecast_mwh': annual_generation,
            'recommended_hedge_volume_mwh': hedge_volume,
            'recommended_hedge_price': hedge_price,
            'hedge_value': hedge_volume * hedge_price
        })
    
    # Portfolio level hedging
    portfolio_hedge = {
        'total_generation_forecast': total_generation_forecast,
        'total_recommended_hedge': total_generation_forecast * hedge_ratio,
        'total_hedge_value': sum(h['hedge_value'] for h in hedging_recommendations)
    }
    
    return hedging_recommendations, portfolio_hedge

def scenario_optimization(assets_df, scenarios_list, weights_list):
    """
    Optimize portfolio performance across multiple scenarios
    """
    n_assets = len(assets_df)
    scenario_results = []
    
    for scenario_idx, (scenario, weight) in enumerate(zip(scenarios_list, weights_list)):
        scenario_valuations = []
        
        for _, asset in assets_df.iterrows():
            valuation = calculate_asset_dcf(asset, scenario)
            scenario_valuations.append(valuation['npv'])
        
        weighted_value = sum(val * weight for val in scenario_valuations)
        
        scenario_results.append({
            'scenario_id': scenario_idx,
            'scenario_name': f"Scenario_{scenario_idx + 1}",
            'weight': weight,
            'portfolio_value': weighted_value,
            'asset_values': scenario_valuations
        })
    
    # Calculate expected portfolio value across scenarios
    expected_portfolio_value = sum(sr['portfolio_value'] * sr['weight'] for sr in scenario_results)
    
    optimization_summary = {
        'expected_portfolio_value': expected_portfolio_value,
        'scenario_results': scenario_results,
        'value_at_risk_10': np.percentile([sr['portfolio_value'] for sr in scenario_results], 10)
    }
    
    return optimization_summary