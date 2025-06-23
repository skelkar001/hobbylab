# Financial Forecasting Web App
# Save this as: financial_app.py
# Run with: streamlit run financial_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

# Set page configuration
st.set_page_config(
    page_title="Financial Forecasting Tool",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class FinancialProfile:
    """Class to hold current financial information"""
    current_savings: float
    monthly_income: float
    monthly_expenses: Dict[str, float]
    annual_return_rate: float
    inflation_rate: float
    tax_rate: float = 0.0

class FinancialForecaster:
    """Financial forecasting engine"""

    def __init__(self, profile: FinancialProfile):
        self.profile = profile
        self.forecast_data = None

    def calculate_monthly_savings(self) -> float:
        """Calculate monthly savings (income - expenses)"""
        total_expenses = sum(self.profile.monthly_expenses.values())
        return self.profile.monthly_income - total_expenses

    def forecast(self, years: int = 10, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """Generate financial forecast for specified number of years"""
        if start_date is None:
            start_date = datetime.now()

        months = years * 12
        monthly_return_rate = self.profile.annual_return_rate / 12
        monthly_inflation_rate = self.profile.inflation_rate / 12

        # Initialize lists to store data
        dates = []
        savings_balance = []
        cumulative_savings = []
        real_value = []
        monthly_income = []
        monthly_expenses = []
        net_monthly_savings = []

        current_savings = self.profile.current_savings
        current_income = self.profile.monthly_income
        current_expense_total = sum(self.profile.monthly_expenses.values())

        for month in range(months + 1):
            current_date = start_date + timedelta(days=30 * month)
            dates.append(current_date)

            # Apply inflation to income and expenses
            inflated_income = current_income * ((1 + monthly_inflation_rate) ** month)
            inflated_expenses = current_expense_total * ((1 + monthly_inflation_rate) ** month)
            inflated_savings = inflated_income - inflated_expenses

            # Calculate investment growth
            if month == 0:
                current_balance = current_savings
            else:
                current_balance = (current_balance + inflated_savings) * (1 + monthly_return_rate)
                if self.profile.tax_rate > 0:
                    returns = current_balance - (previous_balance + inflated_savings)
                    taxes = returns * self.profile.tax_rate
                    current_balance -= taxes

            # Calculate real (inflation-adjusted) value
            real_balance = current_balance / ((1 + monthly_inflation_rate) ** month)

            # Store values
            savings_balance.append(current_balance)
            cumulative_savings.append(current_balance - self.profile.current_savings)
            real_value.append(real_balance)
            monthly_income.append(inflated_income)
            monthly_expenses.append(inflated_expenses)
            net_monthly_savings.append(inflated_savings)

            previous_balance = current_balance

        # Create DataFrame
        self.forecast_data = pd.DataFrame({
            'Date': dates,
            'Savings_Balance': savings_balance,
            'Cumulative_Growth': cumulative_savings,
            'Real_Value': real_value,
            'Monthly_Income': monthly_income,
            'Monthly_Expenses': monthly_expenses,
            'Net_Monthly_Savings': net_monthly_savings,
            'Year': [d.year for d in dates],
            'Month': [d.month for d in dates]
        })

        return self.forecast_data

    def get_summary_stats(self) -> Dict:
        """Get summary statistics from the forecast"""
        if self.forecast_data is None:
            raise ValueError("Must run forecast() first")

        final_row = self.forecast_data.iloc[-1]
        initial_savings = self.profile.current_savings

        return {
            'initial_savings': initial_savings,
            'final_balance': final_row['Savings_Balance'],
            'total_growth': final_row['Cumulative_Growth'],
            'growth_percentage': (final_row['Savings_Balance'] / initial_savings - 1) * 100 if initial_savings > 0 else 0,
            'final_real_value': final_row['Real_Value'],
            'real_growth_percentage': (final_row['Real_Value'] / initial_savings - 1) * 100 if initial_savings > 0 else 0,
            'average_monthly_savings': self.forecast_data['Net_Monthly_Savings'].mean(),
            'total_contributions': self.forecast_data['Net_Monthly_Savings'].sum()
        }

    def scenario_analysis(self, scenarios: Dict[str, Dict]) -> pd.DataFrame:
        """Run multiple scenarios with different parameters"""
        results = []

        for scenario_name, overrides in scenarios.items():
            # Create modified profile
            profile_dict = {
                'current_savings': self.profile.current_savings,
                'monthly_income': self.profile.monthly_income,
                'monthly_expenses': self.profile.monthly_expenses,
                'annual_return_rate': self.profile.annual_return_rate,
                'inflation_rate': self.profile.inflation_rate,
                'tax_rate': self.profile.tax_rate
            }
            profile_dict.update(overrides)

            temp_profile = FinancialProfile(**profile_dict)
            temp_forecaster = FinancialForecaster(temp_profile)
            temp_forecaster.forecast(years=10)
            stats = temp_forecaster.get_summary_stats()

            results.append({
                'Scenario': scenario_name,
                'Final_Balance': stats['final_balance'],
                'Real_Value': stats['final_real_value'],
                'Total_Growth': stats['total_growth'],
                'Growth_Percentage': stats['growth_percentage'],
                'Real_Growth_Percentage': stats['real_growth_percentage']
            })

        return pd.DataFrame(results)

def create_plotly_charts(forecaster):
    """Create interactive Plotly charts"""
    if forecaster.forecast_data is None:
        return None

    data = forecaster.forecast_data

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Savings Growth Over Time', 'Monthly Cash Flow',
                       'Annual Growth Rate', 'Final Balance Composition'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )

    # Plot 1: Savings Growth Over Time
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Savings_Balance'],
                  mode='lines', name='Nominal Value', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Real_Value'],
                  mode='lines', name='Real Value', line=dict(color='red', width=3, dash='dash')),
        row=1, col=1
    )

    # Plot 2: Monthly Cash Flow
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Monthly_Income'],
                  mode='lines', name='Income', fill='tonexty', fillcolor='rgba(0,255,0,0.3)'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data['Date'], y=-data['Monthly_Expenses'],
                  mode='lines', name='Expenses', fill='tozeroy', fillcolor='rgba(255,0,0,0.3)'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['Net_Monthly_Savings'],
                  mode='lines', name='Net Savings', line=dict(color='blue', width=3)),
        row=1, col=2
    )

    # Plot 3: Annual Growth Rate
    annual_data = data.groupby('Year').last()
    if len(annual_data) > 1:
        annual_growth = annual_data['Savings_Balance'].pct_change() * 100
        fig.add_trace(
            go.Bar(x=annual_data.index[1:], y=annual_growth[1:],
                  name='Annual Growth', marker_color='purple'),
            row=2, col=1
        )

    # Plot 4: Final Balance Composition
    stats = forecaster.get_summary_stats()
    labels = ['Initial Savings', 'Investment Returns', 'Contributions']
    values = [
        forecaster.profile.current_savings,
        max(0, stats['final_balance'] - stats['total_contributions'] - forecaster.profile.current_savings),
        max(0, stats['total_contributions'])
    ]

    # Filter out zero values
    non_zero = [(l, v) for l, v in zip(labels, values) if v > 0]
    if non_zero:
        labels, values = zip(*non_zero)
        fig.add_trace(
            go.Pie(labels=labels, values=values, name="Balance Composition"),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(height=800, showlegend=True, title_text="Financial Forecast Analysis")
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Balance ($)", row=1, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=2)
    fig.update_yaxes(title_text="Growth Rate (%)", row=2, col=1)

    return fig

def main():
    """Main Streamlit application"""

    # Header
    st.title("💰 Financial Forecasting Tool")
    st.markdown("Plan your financial future with interactive forecasting and scenario analysis")

    # Initialize session state
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None

    # Sidebar for inputs
    st.sidebar.header("📊 Configuration")

    # Load example data button
    if st.sidebar.button("🚀 Load Example Data"):
        st.session_state.example_loaded = True

    # Basic Information
    st.sidebar.subheader("💰 Basic Information")

    if 'example_loaded' in st.session_state:
        default_savings = 50000.0
        default_income = 8000.0
    else:
        default_savings = 0.0
        default_income = 0.0

    current_savings = st.sidebar.number_input(
        "Current Savings ($)",
        min_value=0.0,
        value=default_savings,
        step=1000.0,
        help="Your current total savings and investments"
    )

    monthly_income = st.sidebar.number_input(
        "Monthly Income ($)",
        min_value=0.0,
        value=default_income,
        step=100.0,
        help="Your monthly after-tax income"
    )

    # Monthly Expenses
    st.sidebar.subheader("💳 Monthly Expenses")

    # Define expense categories with example values
    if 'example_loaded' in st.session_state:
        expense_defaults = {
            'Housing': 2000.0,
            'Food': 800.0,
            'Transportation': 400.0,
            'Utilities': 300.0,
            'Insurance': 200.0,
            'Entertainment': 500.0,
            'Healthcare': 150.0,
            'Miscellaneous': 200.0
        }
    else:
        expense_defaults = {category: 0.0 for category in
                          ['Housing', 'Food', 'Transportation', 'Utilities',
                           'Insurance', 'Entertainment', 'Healthcare', 'Miscellaneous']}

    expenses = {}
    for category, default_value in expense_defaults.items():
        expenses[category] = st.sidebar.number_input(
            f"{category} ($)",
            min_value=0.0,
            value=default_value,
            step=50.0
        )

    # Investment Parameters
    st.sidebar.subheader("📈 Investment Parameters")

    return_rate = st.sidebar.slider(
        "Annual Return Rate (%)",
        min_value=-20.0,
        max_value=30.0,
        value=7.0 if 'example_loaded' in st.session_state else 7.0,
        step=0.5,
        help="Expected annual return on your investments"
    )

    inflation_rate = st.sidebar.slider(
        "Inflation Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0 if 'example_loaded' in st.session_state else 3.0,
        step=0.1,
        help="Expected annual inflation rate"
    )

    tax_rate = st.sidebar.slider(
        "Tax Rate on Returns (%)",
        min_value=0.0,
        max_value=50.0,
        value=15.0 if 'example_loaded' in st.session_state else 15.0,
        step=1.0,
        help="Tax rate applied to investment gains"
    )

    # Forecast Settings
    st.sidebar.subheader("⏰ Forecast Settings")

    forecast_years = st.sidebar.slider(
        "Years to Forecast",
        min_value=1,
        max_value=50,
        value=15 if 'example_loaded' in st.session_state else 15,
        help="Number of years to project into the future"
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("🎯 Quick Summary")

        # Calculate current monthly savings
        total_expenses = sum(expenses.values())
        monthly_savings = monthly_income - total_expenses

        st.metric("Monthly Income", f"${monthly_income:,.0f}")
        st.metric("Monthly Expenses", f"${total_expenses:,.0f}")
        st.metric("Monthly Savings", f"${monthly_savings:,.0f}",
                 delta=None if monthly_savings >= 0 else "⚠️ Negative!")

        if monthly_savings < 0:
            st.error("⚠️ You're spending more than you earn!")
        elif monthly_savings == 0:
            st.warning("💡 You're breaking even. Consider reducing expenses or increasing income.")
        else:
            st.success(f"✅ Great! You're saving ${monthly_savings:,.0f} per month")

    with col1:
        # Run Forecast Button
        if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
            if monthly_income <= 0:
                st.error("Please enter a positive monthly income")
            else:
                # Create profile
                profile = FinancialProfile(
                    current_savings=current_savings,
                    monthly_income=monthly_income,
                    monthly_expenses={k: v for k, v in expenses.items() if v > 0},
                    annual_return_rate=return_rate / 100,
                    inflation_rate=inflation_rate / 100,
                    tax_rate=tax_rate / 100
                )

                # Create forecaster and run forecast
                st.session_state.forecaster = FinancialForecaster(profile)
                with st.spinner("Running forecast..."):
                    st.session_state.forecaster.forecast(years=forecast_years)

                st.success("✅ Forecast completed!")

    # Display results if forecast has been run
    if st.session_state.forecaster and st.session_state.forecaster.forecast_data is not None:

        # Summary Statistics
        st.subheader("📊 Forecast Summary")
        stats = st.session_state.forecaster.get_summary_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Final Balance", f"${stats['final_balance']:,.0f}")
        with col2:
            st.metric("Total Growth", f"${stats['total_growth']:,.0f}")
        with col3:
            st.metric("Growth %", f"{stats['growth_percentage']:.1f}%")
        with col4:
            st.metric("Real Value", f"${stats['final_real_value']:,.0f}")

        # Interactive Charts
        st.subheader("📈 Interactive Charts")
        fig = create_plotly_charts(st.session_state.forecaster)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Scenario Analysis
        st.subheader("🎭 Scenario Analysis")

        if st.button("Run Scenario Analysis"):
            scenarios = {
                'Conservative': {'annual_return_rate': 0.04},
                'Base Case': {},
                'Optimistic': {'annual_return_rate': 0.10},
                'High Inflation': {'inflation_rate': 0.05},
                'Market Crash': {'annual_return_rate': -0.10},
                'Salary Boost': {'monthly_income': monthly_income * 1.2}
            }

            with st.spinner("Running scenarios..."):
                scenario_results = st.session_state.forecaster.scenario_analysis(scenarios)

            # Display scenario results
            st.dataframe(
                scenario_results.style.format({
                    'Final_Balance': '${:,.0f}',
                    'Real_Value': '${:,.0f}',
                    'Total_Growth': '${:,.0f}',
                    'Growth_Percentage': '{:.1f}%',
                    'Real_Growth_Percentage': '{:.1f}%'
                }),
                use_container_width=True
            )

            # Scenario comparison chart
            fig_scenario = px.bar(
                scenario_results,
                x='Scenario',
                y='Final_Balance',
                title="Final Balance by Scenario",
                color='Growth_Percentage',
                color_continuous_scale='RdYlGn'
            )
            fig_scenario.update_layout(height=400)
            st.plotly_chart(fig_scenario, use_container_width=True)

        # Data Export
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Download Forecast Data"):
                csv = st.session_state.forecaster.forecast_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"financial_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )

        with col2:
            if st.button("📋 Show Data Table"):
                # Show sample of data
                sample_data = st.session_state.forecaster.forecast_data.iloc[::12]  # Every year
                st.dataframe(
                    sample_data[['Date', 'Savings_Balance', 'Real_Value', 'Net_Monthly_Savings']].style.format({
                        'Savings_Balance': '${:,.0f}',
                        'Real_Value': '${:,.0f}',
                        'Net_Monthly_Savings': '${:,.0f}'
                    }),
                    use_container_width=True
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>💡 <strong>Tip:</strong> This tool provides estimates based on your inputs.
            Actual results may vary due to market conditions and life changes.</p>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
