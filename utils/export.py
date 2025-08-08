"""
Export functionality for reports, data, and visualizations
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import json
from typing import Dict, List, Any, Optional

class ReportGenerator:
    """
    Generate comprehensive reports for asset analysis and valuation
    """
    
    def __init__(self, title: str = "Energy Asset Analysis Report"):
        self.title = title
        self.report_date = datetime.now()
        self.sections = []
        
    def add_section(self, section_title: str, content: Dict[str, Any]):
        """Add a section to the report"""
        section = {
            'title': section_title,
            'content': content,
            'timestamp': datetime.now()
        }
        self.sections.append(section)
    
    def generate_executive_summary(self, portfolio_data: Dict, market_data: Dict) -> Dict:
        """Generate executive summary section"""
        total_capacity = portfolio_data.get('total_capacity_mw', 0)
        total_value = portfolio_data.get('total_portfolio_value', 0)
        asset_count = portfolio_data.get('asset_count', 0)
        
        summary = {
            'report_date': self.report_date.strftime('%Y-%m-%d'),
            'portfolio_overview': {
                'total_capacity_mw': total_capacity,
                'total_assets': asset_count,
                'portfolio_value_eur_m': total_value / 1e6 if total_value else 0,
                'value_per_mw_eur': total_value / total_capacity if total_capacity > 0 else 0
            },
            'market_conditions': {
                'power_price_eur_mwh': market_data.get('current_power_price', 0),
                'gas_price_eur_mwh': market_data.get('gas_price', 0),
                'carbon_price_eur_t': market_data.get('carbon_price', 0),
                'market_regime': 'Normal'  # Could be determined dynamically
            },
            'key_insights': [
                f"Portfolio comprises {asset_count} assets with {total_capacity:,.0f} MW total capacity",
                f"Current market conditions show power prices at €{market_data.get('current_power_price', 0):.1f}/MWh",
                "Gas assets represent the largest technology by capacity",
                "Portfolio positioned to benefit from current market dynamics"
            ]
        }
        
        return summary
    
    def generate_asset_performance_section(self, assets_data: pd.DataFrame, performance_metrics: Dict) -> Dict:
        """Generate asset performance analysis section"""
        if assets_data.empty:
            return {'error': 'No asset data available'}
        
        # Performance statistics
        avg_availability = assets_data['availability'].mean()
        avg_efficiency = assets_data['efficiency'].mean()
        capacity_range = {
            'min': assets_data['capacity_mw'].min(),
            'max': assets_data['capacity_mw'].max(),
            'mean': assets_data['capacity_mw'].mean()
        }
        
        # Technology breakdown
        tech_breakdown = assets_data.groupby('type').agg({
            'capacity_mw': ['count', 'sum'],
            'availability': 'mean',
            'efficiency': 'mean'
        }).round(3)
        
        performance_section = {
            'portfolio_statistics': {
                'average_availability': avg_availability,
                'average_efficiency': avg_efficiency,
                'capacity_statistics': capacity_range,
                'total_annual_generation_gwh': (
                    assets_data['capacity_mw'] * 8760 * assets_data['availability']
                ).sum() / 1000
            },
            'technology_breakdown': tech_breakdown.to_dict(),
            'top_performers': {
                'highest_availability': assets_data.nlargest(3, 'availability')[['name', 'availability']].to_dict('records'),
                'highest_efficiency': assets_data.nlargest(3, 'efficiency')[['name', 'efficiency']].to_dict('records'),
                'largest_capacity': assets_data.nlargest(3, 'capacity_mw')[['name', 'capacity_mw']].to_dict('records')
            },
            'performance_trends': {
                'generation_by_technology': assets_data.groupby('type')['capacity_mw'].sum().to_dict(),
                'age_distribution': {
                    'average_age': (2024 - assets_data['construction_year']).mean(),
                    'oldest_asset': 2024 - assets_data['construction_year'].min(),
                    'newest_asset': 2024 - assets_data['construction_year'].max()
                }
            }
        }
        
        return performance_section
    
    def generate_financial_analysis_section(self, valuation_results: Dict, market_scenarios: Dict) -> Dict:
        """Generate financial analysis section"""
        financial_section = {
            'valuation_summary': {
                'methodology': valuation_results.get('methodology', 'DCF'),
                'base_case_npv_eur_m': valuation_results.get('total_portfolio_value', 0) / 1e6,
                'key_assumptions': valuation_results.get('market_assumptions', {}),
                'valuation_date': valuation_results.get('valuation_date', datetime.now()).strftime('%Y-%m-%d')
            },
            'scenario_analysis': market_scenarios,
            'risk_metrics': {
                'npv_volatility': 'Medium',  # Would be calculated from Monte Carlo
                'probability_positive_npv': 0.75,  # Example value
                'key_sensitivities': ['Power prices', 'Gas prices', 'Carbon prices']
            },
            'financial_ratios': {
                'portfolio_irr': 0.08,  # Example - would be calculated
                'payback_period_years': 12,  # Example
                'debt_service_coverage': 1.4  # Example
            }
        }
        
        return financial_section
    
    def export_to_json(self) -> str:
        """Export report as JSON string"""
        report_dict = {
            'title': self.title,
            'report_date': self.report_date.isoformat(),
            'sections': self.sections,
            'metadata': {
                'generated_by': 'Energy Asset Valuation Dashboard',
                'version': '1.0',
                'section_count': len(self.sections)
            }
        }
        
        return json.dumps(report_dict, indent=2, default=str)
    
    def export_to_csv_data(self, data: pd.DataFrame, filename: str) -> io.StringIO:
        """Export DataFrame to CSV format"""
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return csv_buffer

def create_asset_summary_export(assets_data: pd.DataFrame, market_data: Dict) -> pd.DataFrame:
    """
    Create comprehensive asset summary for export
    """
    if assets_data.empty:
        return pd.DataFrame()
    
    export_data = assets_data.copy()
    
    # Add calculated fields
    export_data['annual_generation_mwh'] = (
        export_data['capacity_mw'] * 8760 * export_data['availability']
    )
    
    export_data['asset_age_years'] = 2024 - export_data['construction_year']
    
    # Add market-dependent calculations if market data available
    if market_data:
        power_price = market_data.get('current_power_price', 65)
        gas_price = market_data.get('gas_price', 45)
        carbon_price = market_data.get('carbon_price', 85)
        
        # Calculate marginal costs
        marginal_costs = []
        gross_margins = []
        
        for _, asset in export_data.iterrows():
            # Fuel cost
            if asset['fuel_type'] == 'Natural Gas':
                fuel_cost = gas_price / asset['efficiency']
            else:
                fuel_cost = 0
            
            # Carbon cost
            carbon_cost = carbon_price * asset['co2_intensity']
            
            # Total marginal cost
            marginal_cost = asset['variable_cost'] + fuel_cost + carbon_cost
            gross_margin = power_price - marginal_cost
            
            marginal_costs.append(marginal_cost)
            gross_margins.append(gross_margin)
        
        export_data['marginal_cost_eur_mwh'] = marginal_costs
        export_data['gross_margin_eur_mwh'] = gross_margins
        export_data['annual_gross_profit_eur_m'] = (
            export_data['gross_margin_eur_mwh'] * export_data['annual_generation_mwh'] / 1e6
        )
    
    # Reorder columns for better readability
    column_order = [
        'name', 'type', 'country', 'capacity_mw', 'efficiency', 'availability',
        'annual_generation_mwh', 'asset_age_years', 'construction_year',
        'fuel_type', 'co2_intensity', 'variable_cost', 'fixed_cost'
    ]
    
    # Add market-dependent columns if they exist
    if 'marginal_cost_eur_mwh' in export_data.columns:
        column_order.extend(['marginal_cost_eur_mwh', 'gross_margin_eur_mwh', 'annual_gross_profit_eur_m'])
    
    # Filter to existing columns
    available_columns = [col for col in column_order if col in export_data.columns]
    export_data = export_data[available_columns]
    
    return export_data

def create_valuation_export(valuation_results: List[Dict]) -> pd.DataFrame:
    """
    Create valuation results export table
    """
    if not valuation_results:
        return pd.DataFrame()
    
    export_records = []
    
    for result in valuation_results:
        if isinstance(result, dict):
            record = {
                'asset_name': result.get('asset_name', 'Unknown'),
                'methodology': result.get('methodology', 'DCF'),
                'npv_eur_m': result.get('npv_eur', 0) / 1e6 if result.get('npv_eur') else 0,
                'irr_percent': result.get('irr', 0) * 100 if result.get('irr') else 0,
                'annual_ebitda_eur_m': result.get('annual_ebitda', 0) / 1e6 if result.get('annual_ebitda') else 0,
                'valuation_date': result.get('valuation_date', datetime.now()).strftime('%Y-%m-%d') if result.get('valuation_date') else datetime.now().strftime('%Y-%m-%d')
            }
            
            # Add market assumptions if available
            if 'market_assumptions' in result:
                assumptions = result['market_assumptions']
                record.update({
                    'power_price_assumption_eur_mwh': assumptions.get('power_price', 0),
                    'gas_price_assumption_eur_mwh': assumptions.get('gas_price', 0),
                    'carbon_price_assumption_eur_t': assumptions.get('carbon_price', 0),
                    'discount_rate_percent': assumptions.get('discount_rate', 0.07) * 100
                })
            
            export_records.append(record)
    
    return pd.DataFrame(export_records)

def create_market_data_export(price_data: pd.DataFrame, market_summary: Dict) -> pd.DataFrame:
    """
    Create market data export
    """
    export_data = price_data.copy() if not price_data.empty else pd.DataFrame()
    
    if not export_data.empty:
        # Add market summary as additional columns (broadcast to all rows)
        if market_summary:
            export_data['gas_price_eur_mwh'] = market_summary.get('gas_price', 0)
            export_data['carbon_price_eur_t'] = market_summary.get('carbon_price', 0)
            export_data['market_regime'] = 'Normal'  # Example
    
    return export_data

def create_dashboard_export_package(assets_data: pd.DataFrame, valuation_results: List[Dict], 
                                   market_data: Dict, price_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create comprehensive export package with multiple data sets
    """
    export_package = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'dashboard_version': '1.0',
            'data_sources': ['Internal Asset Database', 'ENTSO-E', 'Market APIs'],
            'asset_count': len(assets_data) if not assets_data.empty else 0
        },
        'datasets': {}
    }
    
    # Asset summary
    asset_export = create_asset_summary_export(assets_data, market_data)
    if not asset_export.empty:
        export_package['datasets']['asset_summary'] = asset_export.to_dict('records')
    
    # Valuation results
    valuation_export = create_valuation_export(valuation_results)
    if not valuation_export.empty:
        export_package['datasets']['valuation_results'] = valuation_export.to_dict('records')
    
    # Market data
    market_export = create_market_data_export(price_data, market_data)
    if not market_export.empty:
        export_package['datasets']['market_data'] = market_export.to_dict('records')
    
    # Portfolio summary
    if not assets_data.empty:
        portfolio_summary = {
            'total_capacity_mw': assets_data['capacity_mw'].sum(),
            'asset_count': len(assets_data),
            'technology_mix': assets_data.groupby('type')['capacity_mw'].sum().to_dict(),
            'country_distribution': assets_data.groupby('country')['capacity_mw'].sum().to_dict(),
            'average_efficiency': assets_data['efficiency'].mean(),
            'average_availability': assets_data['availability'].mean(),
            'total_annual_generation_potential_gwh': (
                assets_data['capacity_mw'] * 8760 * assets_data['availability']
            ).sum() / 1000
        }
        export_package['datasets']['portfolio_summary'] = portfolio_summary
    
    return export_package

def export_visualization_data(chart_data: Dict, chart_type: str) -> Dict[str, Any]:
    """
    Export data used in visualizations for external use
    """
    visualization_export = {
        'chart_type': chart_type,
        'export_timestamp': datetime.now().isoformat(),
        'data': chart_data,
        'metadata': {
            'data_points': len(chart_data) if isinstance(chart_data, (list, dict)) else 0,
            'chart_library': 'plotly',
            'format': 'json'
        }
    }
    
    return visualization_export

def create_scenario_analysis_export(scenario_results: Dict) -> pd.DataFrame:
    """
    Create scenario analysis results export
    """
    if not scenario_results:
        return pd.DataFrame()
    
    export_records = []
    
    asset_name = scenario_results.get('asset_name', 'Portfolio')
    scenarios = scenario_results.get('scenarios', {})
    
    for scenario_name, scenario_data in scenarios.items():
        record = {
            'asset_name': asset_name,
            'scenario_name': scenario_name,
            'npv_eur_m': scenario_data.get('npv', 0) / 1e6,
            'irr_percent': scenario_data.get('irr', 0) * 100 if scenario_data.get('irr') else 0,
            'annual_ebitda_eur_m': scenario_data.get('annual_ebitda', 0) / 1e6,
        }
        
        # Add scenario assumptions
        if 'market_assumptions' in scenario_data:
            assumptions = scenario_data['market_assumptions']
            record.update({
                'power_price_eur_mwh': assumptions.get('power_price', 0),
                'gas_price_eur_mwh': assumptions.get('gas_price', 0),
                'carbon_price_eur_t': assumptions.get('carbon_price', 0)
            })
        
        # Add scenario factors if available
        if 'scenario_factors' in scenario_data:
            factors = scenario_data['scenario_factors']
            record.update({
                'power_price_factor': factors.get('power_price_factor', 1.0),
                'gas_price_factor': factors.get('gas_price_factor', 1.0),
                'carbon_price_factor': factors.get('carbon_price_factor', 1.0)
            })
        
        export_records.append(record)
    
    return pd.DataFrame(export_records)

def create_sensitivity_analysis_export(sensitivity_results: Dict, asset_name: str) -> pd.DataFrame:
    """
    Create sensitivity analysis results export
    """
    if not sensitivity_results:
        return pd.DataFrame()
    
    export_records = []
    
    for variable, sensitivity_data in sensitivity_results.items():
        for data_point in sensitivity_data:
            record = {
                'asset_name': asset_name,
                'sensitivity_variable': variable.replace('_', ' ').title(),
                'parameter_change_percent': data_point['parameter_change'] * 100,
                'npv_change_percent': data_point['npv_change'] * 100
            }
            export_records.append(record)
    
    return pd.DataFrame(export_records)

def create_dispatch_analysis_export(dispatch_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create dispatch analysis export
    """
    if dispatch_data.empty:
        return pd.DataFrame()
    
    export_data = dispatch_data.copy()
    
    # Round numerical columns for cleaner export
    numerical_columns = ['marginal_cost', 'profit_margin', 'hourly_profit', 'fuel_cost', 'carbon_cost']
    for col in numerical_columns:
        if col in export_data.columns:
            export_data[col] = export_data[col].round(2)
    
    # Convert boolean to text
    if 'should_dispatch' in export_data.columns:
        export_data['dispatch_recommended'] = export_data['should_dispatch'].map({True: 'Yes', False: 'No'})
        export_data = export_data.drop('should_dispatch', axis=1)
    
    return export_data

@st.cache_data
def generate_comprehensive_report(assets_data: pd.DataFrame, valuation_results: List[Dict], 
                                market_data: Dict, price_data: pd.DataFrame, 
                                performance_metrics: Dict) -> Dict[str, Any]:
    """
    Generate comprehensive analysis report
    """
    # Initialize report generator
    report_gen = ReportGenerator("Energy Asset Portfolio Analysis Report")
    
    # Calculate portfolio summary
    portfolio_summary = {
        'total_capacity_mw': assets_data['capacity_mw'].sum() if not assets_data.empty else 0,
        'asset_count': len(assets_data) if not assets_data.empty else 0,
        'total_portfolio_value': sum(v.get('npv_eur', 0) for v in valuation_results if isinstance(v, dict))
    }
    
    # Add executive summary
    exec_summary = report_gen.generate_executive_summary(portfolio_summary, market_data)
    report_gen.add_section("Executive Summary", exec_summary)
    
    # Add asset performance section
    if not assets_data.empty:
        performance_section = report_gen.generate_asset_performance_section(assets_data, performance_metrics)
        report_gen.add_section("Asset Performance Analysis", performance_section)
    
    # Add financial analysis section
    if valuation_results:
        financial_section = report_gen.generate_financial_analysis_section(
            {'total_portfolio_value': portfolio_summary['total_portfolio_value'],
             'methodology': 'DCF',
             'market_assumptions': market_data,
             'valuation_date': datetime.now()},
            {}  # Market scenarios would be passed here
        )
        report_gen.add_section("Financial Analysis", financial_section)
    
    # Add market analysis section
    market_section = {
        'current_conditions': market_data,
        'price_trends': {
            'current_power_price': market_data.get('current_power_price', 0),
            'price_volatility': market_data.get('power_price_volatility', 0),
            'trend_direction': 'Stable'  # Would be calculated from price data
        },
        'market_outlook': [
            "Power prices expected to remain volatile due to geopolitical factors",
            "Carbon prices likely to increase under EU Green Deal policies",
            "Gas price volatility expected to persist in medium term"
        ]
    }
    report_gen.add_section("Market Analysis", market_section)
    
    # Generate final report
    comprehensive_report = {
        'report_metadata': {
            'title': report_gen.title,
            'generation_date': report_gen.report_date.isoformat(),
            'sections_count': len(report_gen.sections),
            'data_quality': 'High' if not assets_data.empty and valuation_results else 'Medium'
        },
        'sections': {section['title']: section['content'] for section in report_gen.sections},
        'summary_metrics': {
            'portfolio_capacity_mw': portfolio_summary['total_capacity_mw'],
            'portfolio_value_eur_m': portfolio_summary['total_portfolio_value'] / 1e6,
            'asset_count': portfolio_summary['asset_count'],
            'report_confidence': 'High'
        },
        'export_formats_available': ['JSON', 'CSV', 'Excel'],
        'data_currency': market_data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if market_data.get('timestamp') else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return comprehensive_report

def create_excel_export_data(assets_data: pd.DataFrame, valuation_results: List[Dict], 
                           market_data: Dict, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create multiple DataFrames for Excel export (multiple sheets)
    """
    excel_data = {}
    
    # Sheet 1: Asset Summary
    if not assets_data.empty:
        excel_data['Asset Summary'] = create_asset_summary_export(assets_data, market_data)
    
    # Sheet 2: Valuation Results
    if valuation_results:
        excel_data['Valuation Results'] = create_valuation_export(valuation_results)
    
    # Sheet 3: Market Data
    if not price_data.empty:
        excel_data['Market Data'] = create_market_data_export(price_data, market_data)
    
    # Sheet 4: Portfolio Summary
    if not assets_data.empty:
        portfolio_df = pd.DataFrame([{
            'Metric': 'Total Capacity (MW)',
            'Value': assets_data['capacity_mw'].sum()
        }, {
            'Metric': 'Number of Assets',
            'Value': len(assets_data)
        }, {
            'Metric': 'Average Efficiency',
            'Value': assets_data['efficiency'].mean()
        }, {
            'Metric': 'Average Availability',
            'Value': assets_data['availability'].mean()
        }])
        excel_data['Portfolio Summary'] = portfolio_df
    
    return excel_data

def export_chart_to_json(fig) -> str:
    """
    Export Plotly chart to JSON format
    """
    return fig.to_json()

def export_chart_to_html(fig) -> str:
    """
    Export Plotly chart to HTML format
    """
    return fig.to_html(include_plotlyjs=True)

def create_kpi_dashboard_export(assets_data: pd.DataFrame, market_data: Dict, 
                              valuation_results: List[Dict]) -> Dict[str, Any]:
    """
    Create KPI dashboard export with key metrics
    """
    kpi_export = {
        'export_timestamp': datetime.now().isoformat(),
        'kpis': {}
    }
    
    if not assets_data.empty:
        # Operational KPIs
        kpi_export['kpis']['operational'] = {
            'total_capacity_mw': float(assets_data['capacity_mw'].sum()),
            'fleet_availability_percent': float(assets_data['availability'].mean() * 100),
            'fleet_efficiency_percent': float(assets_data['efficiency'].mean() * 100),
            'renewable_share_percent': float(
                assets_data[assets_data['type'].isin(['Wind', 'Solar'])]['capacity_mw'].sum() / 
                assets_data['capacity_mw'].sum() * 100
            ),
            'average_asset_age_years': float(2024 - assets_data['construction_year'].mean())
        }
        
        # Generation KPIs
        total_generation = (assets_data['capacity_mw'] * 8760 * assets_data['availability']).sum()
        kpi_export['kpis']['generation'] = {
            'annual_generation_potential_gwh': float(total_generation / 1000),
            'capacity_factor_weighted_avg': float(
                (assets_data['availability'] * assets_data['capacity_mw']).sum() / 
                assets_data['capacity_mw'].sum()
            ),
            'co2_emissions_potential_kt': float(
                (total_generation * assets_data['co2_intensity']).sum() / 1000
            )
        }
    
    # Market KPIs
    if market_data:
        kpi_export['kpis']['market'] = {
            'power_price_eur_mwh': float(market_data.get('current_power_price', 0)),
            'gas_price_eur_mwh': float(market_data.get('gas_price', 0)),
            'carbon_price_eur_t': float(market_data.get('carbon_price', 0)),
            'spark_spread_eur_mwh': float(
                market_data.get('current_power_price', 0) - 
                market_data.get('gas_price', 0) / 0.58  # Assuming 58% efficiency
            )
        }
    
    # Financial KPIs
    if valuation_results:
        total_npv = sum(v.get('npv_eur', 0) for v in valuation_results if isinstance(v, dict))
        positive_npv_count = sum(1 for v in valuation_results 
                               if isinstance(v, dict) and v.get('npv_eur', 0) > 0)
        
        kpi_export['kpis']['financial'] = {
            'portfolio_npv_eur_m': float(total_npv / 1e6),
            'positive_npv_assets_count': int(positive_npv_count),
            'positive_npv_assets_percent': float(
                positive_npv_count / len(valuation_results) * 100 if valuation_results else 0
            ),
            'average_asset_npv_eur_m': float(
                total_npv / len(valuation_results) / 1e6 if valuation_results else 0
            )
        }
    
    return kpi_export

def format_export_filename(base_name: str, export_type: str, timestamp: Optional[datetime] = None) -> str:
    """
    Generate standardized export filename
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp_str}.{export_type.lower()}"

def create_regulatory_report_export(assets_data: pd.DataFrame, market_data: Dict) -> Dict[str, Any]:
    """
    Create regulatory compliance report export
    """
    if assets_data.empty:
        return {}
    
    regulatory_export = {
        'report_date': datetime.now().isoformat(),
        'regulatory_metrics': {},
        'compliance_status': {}
    }
    
    # Emissions reporting
    total_emissions = (
        assets_data['capacity_mw'] * 8760 * assets_data['availability'] * assets_data['co2_intensity']
    ).sum()
    
    regulatory_export['regulatory_metrics']['emissions'] = {
        'total_annual_co2_tonnes': float(total_emissions),
        'emissions_intensity_kg_mwh': float(
            total_emissions * 1000 / 
            (assets_data['capacity_mw'] * 8760 * assets_data['availability']).sum()
        ),
        'renewable_generation_share': float(
            assets_data[assets_data['co2_intensity'] == 0]['capacity_mw'].sum() / 
            assets_data['capacity_mw'].sum()
        )
    }
    
    # Capacity adequacy
    regulatory_export['regulatory_metrics']['capacity'] = {
        'firm_capacity_mw': float(assets_data[assets_data['type'] != 'Wind']['capacity_mw'].sum()),
        'renewable_capacity_mw': float(
            assets_data[assets_data['type'].isin(['Wind', 'Solar'])]['capacity_mw'].sum()
        ),
        'dispatchable_capacity_mw': float(
            assets_data[assets_data['type'] == 'Gas']['capacity_mw'].sum()
        )
    }
    
    # Asset age compliance (example thresholds)
    old_assets = assets_data[2024 - assets_data['construction_year'] > 25]
    
    regulatory_export['compliance_status'] = {
        'assets_over_25_years': len(old_assets),
        'capacity_over_25_years_mw': float(old_assets['capacity_mw'].sum()) if not old_assets.empty else 0,
        'efficiency_below_threshold_count': len(assets_data[assets_data['efficiency'] < 0.45]),
        'emissions_intensive_assets': len(assets_data[assets_data['co2_intensity'] > 0.5])
    }
    
    return regulatory_export