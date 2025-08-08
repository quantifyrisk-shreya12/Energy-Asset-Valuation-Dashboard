"""
Visualization utility functions
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import PLOT_THEME, PRIMARY_COLOR, SECONDARY_COLOR

def create_price_chart(price_data, title="Electricity Prices"):
    """
    Create interactive price chart
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_data['timestamp'],
        y=price_data['price_eur_mwh'],
        mode='lines',
        name='Price (EUR/MWh)',
        line=dict(color=PRIMARY_COLOR, width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price (EUR/MWh)",
        template=PLOT_THEME,
        hovermode='x unified'
    )
    
    return fig

def create_spark_spread_chart(electricity_prices, gas_prices, efficiency=0.58):
    """
    Create spark spread visualization
    """
    spark_spreads = electricity_prices - (gas_prices / efficiency)
    
    fig = go.Figure()
    
    # Add electricity price
    fig.add_trace(go.Scatter(
        y=electricity_prices,
        mode='lines',
        name='Electricity Price',
        line=dict(color=PRIMARY_COLOR)
    ))
    
    # Add gas price (adjusted for efficiency)
    fig.add_trace(go.Scatter(
        y=gas_prices / efficiency,
        mode='lines',
        name='Gas Price (adj. for efficiency)',
        line=dict(color=SECONDARY_COLOR)
    ))
    
    # Add spark spread
    fig.add_trace(go.Scatter(
        y=spark_spreads,
        mode='lines',
        name='Spark Spread',
        fill='tozeroy',
        line=dict(color='green'),
        fillcolor='rgba(0,255,0,0.1)'
    ))
    
    fig.update_layout(
        title="Spark Spread Analysis",
        yaxis_title="Price (EUR/MWh)",
        template=PLOT_THEME
    )
    
    return fig

# def create_merit_order_chart(dispatch_data):
#     """
#     Create merit order stack chart
#     """
#     # Sort by marginal cost
#     sorted_data = dispatch_data.sort_values('marginal_cost')
    
#     # Calculate cumulative capacity
#     sorted_data['cumulative_capacity'] = sorted_data['capacity_mw'].cumsum()
    
#     fig = go.Figure()
    
#     # Create step chart for merit order
#     for i, row in sorted_data.iterrows():
#         start_capacity = row['cumulative_capacity'] - row['capacity_mw']
#         end_capacity = row['cumulative_capacity']
        
#         fig.add_trace(go.Scatter(
#             x=[start_capacity, end_capacity, end_capacity, start_capacity, start_capacity],
#             y=[row['marginal_cost'], row['marginal_cost'], 
#                row['marginal_cost'], row['marginal_cost'], row['marginal_cost']],
#             fill='toself',
#             name=row['name'],
#             mode='lines',
#             line=dict(width=0.5)
#         ))
    
#     fig.update_layout(
#         title="Merit Order Curve",
#         xaxis_title="Cumulative Capacity (MW)",
#         yaxis_title="Marginal Cost (EUR/MWh)",
#         template=PLOT_THEME,
#         showlegend=True
#     )
    
#     return fig



def create_merit_order_chart(dispatch_data):
    """
    Create merit order stack chart
    """
    if dispatch_data.empty:
        return go.Figure().add_annotation(text="No dispatch data", showarrow=False)
    
    # Sort by marginal cost
    sorted_data = dispatch_data.sort_values('marginal_cost')
    
    fig = go.Figure()
    
    # Create stepped merit order curve
    cumulative_capacity = 0
    
    for _, row in sorted_data.iterrows():
        start_capacity = cumulative_capacity
        end_capacity = cumulative_capacity + row['capacity_mw']
        
        fig.add_trace(go.Scatter(
            x=[start_capacity, end_capacity],
            y=[row['marginal_cost'], row['marginal_cost']],
            mode='lines',
            name=f"{row['name']} ({row['type']})",
            line=dict(width=3),
            hovertemplate=f"{row['name']}<br>Capacity: {row['capacity_mw']} MW<br>Cost: €{row['marginal_cost']:.1f}/MWh<extra></extra>"
        ))
        
        cumulative_capacity = end_capacity
    
    # Add current price line
    current_price = 60  # You can pass this as parameter
    fig.add_hline(
        y=current_price, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Market Price: €{current_price}/MWh"
    )
    
    fig.update_layout(
        title="Merit Order Curve - Asset Dispatch Economics",
        xaxis_title="Cumulative Capacity (MW)",
        yaxis_title="Marginal Cost (EUR/MWh)",
        template=PLOT_THEME,
        height=500,
        showlegend=False
    )
    
    return fig

def create_asset_performance_dashboard(asset_data, performance_metrics):
    """
    Create multi-panel asset performance dashboard
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Capacity Factor', 'Availability', 'Generation', 'Efficiency'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    assets = asset_data['name'].tolist()
    
    # Capacity factors
    fig.add_trace(
        go.Bar(x=assets, y=asset_data['availability'], name='Capacity Factor'),
        row=1, col=1
    )
    
    # Availability
    availability_data = asset_data['availability'] * 100
    fig.add_trace(
        go.Bar(x=assets, y=availability_data, name='Availability %'),
        row=1, col=2
    )
    
    # Generation
    generation = asset_data['capacity_mw'] * asset_data['availability'] * 8760
    fig.add_trace(
        go.Bar(x=assets, y=generation, name='Annual Generation (MWh)'),
        row=2, col=1
    )
    
    # Efficiency
    fig.add_trace(
        go.Bar(x=assets, y=asset_data['efficiency'] * 100, name='Efficiency %'),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Asset Performance Dashboard",
        template=PLOT_THEME,
        showlegend=False,
        height=600
    )
    
    return fig

def create_valuation_waterfall(base_value, adjustments, labels):
    """
    Create waterfall chart for valuation analysis
    """
    # Calculate cumulative values
    cumulative = [base_value]
    for adj in adjustments:
        cumulative.append(cumulative[-1] + adj)
    
    fig = go.Figure()
    
    # Base value
    fig.add_trace(go.Waterfall(
        name="NPV Waterfall",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(adjustments) + ["total"],
        x=["Base Case"] + labels + ["Final NPV"],
        textposition="outside",
        text=[f"€{base_value:.1f}M"] + [f"€{adj:+.1f}M" for adj in adjustments] + [f"€{cumulative[-1]:.1f}M"],
        y=[base_value] + adjustments + [cumulative[-1]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="NPV Sensitivity Waterfall",
        template=PLOT_THEME,
        yaxis_title="NPV (EUR Million)"
    )
    
    return fig

def create_monte_carlo_chart(simulation_results, title="Monte Carlo Simulation Results"):
    """
    Create histogram of Monte Carlo simulation results
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=simulation_results,
        nbinsx=50,
        name='NPV Distribution',
        opacity=0.7,
        marker=dict(color=PRIMARY_COLOR)
    ))
    
    # Add statistical lines
    mean_val = np.mean(simulation_results)
    p10_val = np.percentile(simulation_results, 10)
    p90_val = np.percentile(simulation_results, 90)
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: €{mean_val:.1f}M")
    fig.add_vline(x=p10_val, line_dash="dot", line_color="orange",
                  annotation_text=f"P10: €{p10_val:.1f}M")
    fig.add_vline(x=p90_val, line_dash="dot", line_color="green",
                  annotation_text=f"P90: €{p90_val:.1f}M")
    
    fig.update_layout(
        title=title,
        xaxis_title="NPV (EUR Million)",
        yaxis_title="Frequency",
        template=PLOT_THEME
    )
    
    return fig

def create_sensitivity_chart(sensitivity_data, asset_name):
    """
    Create tornado chart for sensitivity analysis
    """
    fig = go.Figure()
    
    variables = list(sensitivity_data.keys())
    colors = ['red' if x < 0 else 'green' for x in [sensitivity_data[var][-1]['npv_change'] - sensitivity_data[var][0]['npv_change'] for var in variables]]
    
    for i, variable in enumerate(variables):
        data = sensitivity_data[variable]
        low_impact = data[0]['npv_change'] * 100
        high_impact = data[-1]['npv_change'] * 100
        
        fig.add_trace(go.Bar(
            y=[variable],
            x=[high_impact - low_impact],
            orientation='h',
            name=variable,
            marker=dict(color=colors[i]),
            text=[f"{high_impact - low_impact:.1f}%"],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=f"Sensitivity Analysis - {asset_name}",
        xaxis_title="NPV Impact (%)",
        template=PLOT_THEME,
        showlegend=False
    )
    
    return fig

def create_portfolio_pie_chart(assets_df, value_column='capacity_mw', title="Portfolio Composition"):
    """
    Create pie chart for portfolio composition
    """
    fig = px.pie(
        assets_df,
        values=value_column,
        names='name',
        title=title,
        template=PLOT_THEME
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_price_heatmap(price_data):
    """
    Create heatmap of price patterns by hour and day
    """
    df = price_data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()
    
    # Create pivot table
    pivot = df.pivot_table(values='price_eur_mwh', index='day', columns='hour', aggfunc='mean')
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        text=pivot.values.round(1),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Price Heatmap by Hour and Day",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template=PLOT_THEME
    )
    
    return fig

def create_correlation_matrix(correlation_data):
    """
    Create correlation matrix heatmap
    """
    # Create correlation matrix
    variables = ['Power Price', 'Gas Price', 'Carbon Price']
    correlation_matrix = np.array([
        [1.0, 0.75, 0.45],
        [0.75, 1.0, 0.35],
        [0.45, 0.35, 1.0]
    ])
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=variables,
        y=variables,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix,
        texttemplate="%{text:.2f}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title="Price Correlation Matrix",
        template=PLOT_THEME
    )
    
    return fig

# def create_dispatch_optimization_chart(dispatch_schedule):
#     """
#     Create dispatch optimization visualization
#     """
#     fig = px.bar(
#         dispatch_schedule.groupby('asset')['profit'].sum().reset_index(),
#         x='asset',
#         y='profit',
#         title="Optimized Dispatch Profits by Asset",
#         template=PLOT_THEME
#     )
    
#     fig.update_layout(
#         xaxis_title="Asset",
#         yaxis_title="Total Profit (EUR)",
#         xaxis_tickangle=-45
#     )
    
#     return fig





def create_dispatch_optimization_chart(dispatch_schedule):
    """
    Create dispatch optimization visualization
    """
    if dispatch_schedule.empty:
        return go.Figure().add_annotation(text="No dispatch data available", showarrow=False)
    
    # Check if 'name' column exists (from dispatch_economics) or 'asset' (from optimization)
    asset_col = 'name' if 'name' in dispatch_schedule.columns else 'asset'
    
    # Filter only profitable assets
    profitable_assets = dispatch_schedule[dispatch_schedule['should_dispatch'] == True]
    
    if profitable_assets.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No profitable dispatch opportunities with current prices",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Group by asset/name and sum profits
    profit_data = profitable_assets.groupby(asset_col)['hourly_profit'].sum().reset_index()
    
    # Sort by profit descending
    profit_data = profit_data.sort_values('hourly_profit', ascending=True)
    
    fig = px.bar(
        profit_data,
        x=asset_col,
        y='hourly_profit',
        title="Dispatch Profits by Asset",
        template=PLOT_THEME,
        color='hourly_profit',
        color_continuous_scale=['red', 'yellow', 'green']
    )
    
    fig.update_layout(
        xaxis_title="Asset Name",
        yaxis_title="Hourly Profit (EUR)",
        xaxis_tickangle=-45,
        height=400
    )
    
    # Add value labels on bars
    fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
    
    return fig

def create_scenario_analysis_chart(scenario_results):
    """
    Create scenario analysis visualization
    """
    scenarios = [f"Scenario {i+1}" for i in range(len(scenario_results))]
    values = [result['portfolio_value'] for result in scenario_results]
    weights = [result['weight'] for result in scenario_results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scenarios,
        y=values,
        name='Portfolio Value',
        marker=dict(color=values, colorscale='Viridis'),
        text=[f"Weight: {w:.1%}" for w in weights],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Scenario Analysis Results",
        xaxis_title="Scenario",
        yaxis_title="Portfolio Value (EUR Million)",
        template=PLOT_THEME
    )
    
    return fig