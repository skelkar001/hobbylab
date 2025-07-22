# Financial Forecasting Web App with Monte Carlo Simulation
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
from typing import Dict, List, Optional, Tuple
import json
from scipy import stats
import sqlite3
import os

# Set page configuration
st.set_page_config(
    page_title="Financial Forecasting Tool",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Historical returns data (annual returns in decimal form)
HISTORICAL_RETURNS = {
    'equity': {
        'mean': 0.1021,  # S&P 500 average annual return (1957-2021)
        'std': 0.1605,   # S&P 500 standard deviation
        'annual_returns': [  # Sample of actual historical returns for better simulation
            0.2656, -0.0910, -0.1189, -0.2210, 0.2868, 0.0762, 0.1849, 0.0581, 0.1654, 0.3172,
            -0.0306, 0.0762, 0.1006, 0.0134, 0.3720, 0.2268, -0.0657, -0.1022, 0.2103, 0.1021,
            0.0503, 0.1675, 0.3149, 0.1838, 0.0581, -0.3700, 0.2639, 0.1506, 0.0221, 0.1596,
            0.1171, -0.0491, 0.2168, 0.1328, 0.1906, 0.0034, 0.1354, 0.1896, -0.1431, -0.2697,
            0.2868, 0.1488, 0.0600, -0.0918, 0.1056, 0.0491, 0.1579, 0.0549, -0.0177, 0.1575
        ]
    },
    'bonds': {
        'mean': 0.0564,   # US 10-year Treasury average return
        'std': 0.0886,    # US 10-year Treasury standard deviation
        'annual_returns': [  # Sample bond returns
            0.0845, 0.1540, 0.0384, 0.1175, -0.0796, 0.0969, 0.1216, 0.0881, -0.0251, -0.0899,
            0.1643, 0.0300, 0.0038, 0.2965, -0.0415, 0.0519, 0.0568, 0.2203, 0.0693, 0.1286,
            -0.1108, 0.0846, 0.2302, 0.0697, 0.1521, -0.0711, 0.1081, 0.1203, 0.0654, -0.0262,
            0.0664, 0.1484, -0.0496, 0.0584, 0.0735, 0.1311, -0.0013, -0.0758, 0.0515, 0.1398,
            -0.0334, 0.0847, 0.1747, 0.0993, -0.1318, 0.0456, 0.0274, 0.1028, 0.0845, 0.0564
        ]
    },
    'cash': {
        'mean': 0.0287,   # 3-month Treasury bill average
        'std': 0.0310,    # Very low volatility for cash
        'annual_returns': [0.0287] * 50  # Simplified constant returns for cash
    }
}

# Database setup for tracking
def init_database():
    """Initialize SQLite database for usage tracking and feedback"""
    conn = sqlite3.connect('app_data.db', check_same_thread=False)
    
    # Create usage tracking table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS usage_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            action_type TEXT,
            user_session TEXT,
            details TEXT
        )
    ''')
    
    # Create feedback table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            feedback_type TEXT,
            email TEXT,
            subject TEXT,
            message TEXT,
            user_session TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def track_usage(action_type: str, details: str = ""):
    """Track user actions for analytics"""
    try:
        # Get or create session ID
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        conn = sqlite3.connect('app_data.db', check_same_thread=False)
        conn.execute('''
            INSERT INTO usage_stats (action_type, user_session, details)
            VALUES (?, ?, ?)
        ''', (action_type, st.session_state.session_id, details))
        conn.commit()
        conn.close()
    except Exception as e:
        # Silently fail if database operations fail
        pass

def get_usage_stats():
    """Get usage statistics from database"""
    try:
        conn = sqlite3.connect('app_data.db', check_same_thread=False)
        
        # Total app visits
        total_visits = conn.execute('SELECT COUNT(DISTINCT user_session) FROM usage_stats').fetchone()[0]
        
        # Total simulations run
        total_simulations = conn.execute(
            'SELECT COUNT(*) FROM usage_stats WHERE action_type = "simulation_run"'
        ).fetchone()[0]
        
        # Today's visits
        today_visits = conn.execute('''
            SELECT COUNT(DISTINCT user_session) FROM usage_stats 
            WHERE date(timestamp) = date('now')
        ''').fetchone()[0]
        
        conn.close()
        return {
            'total_visits': total_visits,
            'total_simulations': total_simulations,
            'today_visits': today_visits
        }
    except Exception as e:
        return {'total_visits': 0, 'total_simulations': 0, 'today_visits': 0}

def submit_feedback(feedback_type: str, email: str, subject: str, message: str):
    """Submit user feedback to database"""
    try:
        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        conn = sqlite3.connect('app_data.db', check_same_thread=False)
        conn.execute('''
            INSERT INTO feedback (feedback_type, email, subject, message, user_session)
            VALUES (?, ?, ?, ?, ?)
        ''', (feedback_type, email, subject, message, st.session_state.session_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

@dataclass
class AssetAllocation:
    """Class to hold asset allocation percentages"""
    equity: float
    bonds: float
    cash: float
    
    def __post_init__(self):
        total = self.equity + self.bonds + self.cash
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Asset allocation must sum to 100%. Current total: {total*100:.1f}%")

@dataclass
class FinancialProfile:
    """Class to hold current financial information"""
    current_savings: float
    monthly_income: float
    monthly_expenses: Dict[str, float]
    asset_allocation: AssetAllocation
    inflation_rate: float
    tax_rate: float = 0.0

class MonteCarloForecaster:
    """Financial forecasting engine with Monte Carlo simulation"""

    def __init__(self, profile: FinancialProfile):
        self.profile = profile
        self.simulation_results = None
        self.forecast_percentiles = None

    def calculate_monthly_savings(self) -> float:
        """Calculate monthly savings (income - expenses)"""
        total_expenses = sum(self.profile.monthly_expenses.values())
        return self.profile.monthly_income - total_expenses

    def generate_portfolio_returns(self, years: int, num_simulations: int = 1000) -> np.ndarray:
        """Generate portfolio returns using historical data and Monte Carlo simulation"""
        np.random.seed(42)  # For reproducible results
        
        # Create return matrices for each asset class
        equity_returns = np.random.choice(
            HISTORICAL_RETURNS['equity']['annual_returns'], 
            size=(num_simulations, years)
        )
        
        bond_returns = np.random.choice(
            HISTORICAL_RETURNS['bonds']['annual_returns'], 
            size=(num_simulations, years)
        )
        
        cash_returns = np.random.choice(
            HISTORICAL_RETURNS['cash']['annual_returns'], 
            size=(num_simulations, years)
        )
        
        # Calculate weighted portfolio returns
        portfolio_returns = (
            self.profile.asset_allocation.equity * equity_returns +
            self.profile.asset_allocation.bonds * bond_returns +
            self.profile.asset_allocation.cash * cash_returns
        )
        
        return portfolio_returns

    def monte_carlo_forecast(self, years: int = 10, num_simulations: int = 1000, 
                           start_date: Optional[datetime] = None) -> Dict:
        """Generate Monte Carlo simulation for financial forecast"""
        if start_date is None:
            start_date = datetime.now()

        months = years * 12
        monthly_inflation_rate = self.profile.inflation_rate / 12

        # Generate annual portfolio returns for all simulations
        portfolio_returns = self.generate_portfolio_returns(years, num_simulations)
        
        # Initialize results arrays
        all_simulations = np.zeros((num_simulations, months + 1))
        monthly_dates = []
        
        # Create date array
        for month in range(months + 1):
            current_date = start_date + timedelta(days=30 * month)
            monthly_dates.append(current_date)

        current_income = self.profile.monthly_income
        current_expense_total = sum(self.profile.monthly_expenses.values())

        # Run Monte Carlo simulations
        for sim in range(num_simulations):
            current_savings = self.profile.current_savings
            all_simulations[sim, 0] = current_savings
            
            for month in range(1, months + 1):
                year_index = min((month - 1) // 12, years - 1)
                monthly_return_rate = portfolio_returns[sim, year_index] / 12
                
                # Apply inflation to income and expenses
                inflated_income = current_income * ((1 + monthly_inflation_rate) ** month)
                inflated_expenses = current_expense_total * ((1 + monthly_inflation_rate) ** month)
                inflated_savings = inflated_income - inflated_expenses
                
                # Calculate investment growth
                previous_balance = current_savings
                current_savings = (current_savings + inflated_savings) * (1 + monthly_return_rate)
                
                # Apply taxes on returns
                if self.profile.tax_rate > 0:
                    returns = current_savings - (previous_balance + inflated_savings)
                    if returns > 0:
                        taxes = returns * self.profile.tax_rate
                        current_savings -= taxes
                
                all_simulations[sim, month] = current_savings

        # Calculate percentiles
        percentiles = [3, 10, 50, 90, 97]
        percentile_results = {}
        
        for p in percentiles:
            percentile_results[f'p{p}'] = np.percentile(all_simulations, p, axis=0)

        # Create summary DataFrame for easy plotting
        summary_data = pd.DataFrame({
            'Date': monthly_dates,
            'P3': percentile_results['p3'],
            'P10': percentile_results['p10'],
            'P50': percentile_results['p50'],
            'P90': percentile_results['p90'],
            'P97': percentile_results['p97']
        })

        self.simulation_results = all_simulations
        self.forecast_percentiles = summary_data

        return {
            'percentiles': summary_data,
            'all_simulations': all_simulations,
            'simulation_stats': self._calculate_simulation_stats(all_simulations, monthly_dates)
        }

    def _calculate_simulation_stats(self, simulations: np.ndarray, dates: List[datetime]) -> Dict:
        """Calculate summary statistics from simulation results"""
        final_values = simulations[:, -1]
        initial_savings = self.profile.current_savings
        
        return {
            'initial_savings': initial_savings,
            'mean_final_balance': np.mean(final_values),
            'median_final_balance': np.median(final_values),
            'std_final_balance': np.std(final_values),
            'probability_positive': np.mean(final_values > initial_savings) * 100,
            'probability_double': np.mean(final_values > initial_savings * 2) * 100,
            'percentile_10': np.percentile(final_values, 10),
            'percentile_90': np.percentile(final_values, 90),
            'var_95': np.percentile(final_values, 5),  # Value at Risk (95% confidence)
            'expected_shortfall': np.mean(final_values[final_values <= np.percentile(final_values, 5)])
        }

def create_monte_carlo_charts(forecaster: MonteCarloForecaster) -> go.Figure:
    """Create interactive Monte Carlo simulation charts"""
    if forecaster.forecast_percentiles is None:
        return None

    data = forecaster.forecast_percentiles
    
    # Create main chart with percentile bands
    fig = go.Figure()

    # Add percentile bands
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['P97'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        name='97th Percentile'
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['P3'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fill='tonexty',
        fillcolor='rgba(255,182,193,0.2)',
        showlegend=True,
        name='3rd-97th Percentile Range'
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['P90'],
        mode='lines',
        line=dict(color='rgba(255,165,0,0.8)', width=2),
        showlegend=False,
        name='90th Percentile'
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['P10'],
        mode='lines',
        line=dict(color='rgba(255,165,0,0.8)', width=2),
        fill='tonexty',
        fillcolor='rgba(255,215,0,0.3)',
        showlegend=True,
        name='10th-90th Percentile Range'
    ))

    # Add median line
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['P50'],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Median (50th Percentile)'
    ))

    # Update layout
    fig.update_layout(
        title='Monte Carlo Simulation: Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=600,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Format y-axis as currency
    fig.update_yaxes(tickformat='$,.0f')

    return fig

def create_allocation_pie_chart(allocation: AssetAllocation) -> go.Figure:
    """Create pie chart for asset allocation"""
    labels = ['Equity', 'Bonds', 'Cash']
    values = [allocation.equity * 100, allocation.bonds * 100, allocation.cash * 100]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title='Asset Allocation',
        height=400
    )
    
    return fig

def create_risk_metrics_chart(stats: Dict) -> go.Figure:
    """Create chart showing risk metrics"""
    metrics = ['10th Percentile', 'Median', '90th Percentile', 'Value at Risk (5%)']
    values = [
        stats['percentile_10'],
        stats['median_final_balance'],
        stats['percentile_90'],
        stats['var_95']
    ]
    
    colors = ['red', 'blue', 'green', 'orange']
    
    fig = go.Figure(data=[go.Bar(
        x=metrics,
        y=values,
        marker_color=colors,
        text=[f'${v:,.0f}' for v in values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Risk Metrics: Final Portfolio Value',
        yaxis_title='Portfolio Value ($)',
        height=400
    )
    
    fig.update_yaxes(tickformat='$,.0f')
    
    return fig

def get_risk_level_allocation(risk_level: str) -> AssetAllocation:
    """Get predefined asset allocation based on risk level"""
    allocations = {
        'Conservative': AssetAllocation(equity=0.30, bonds=0.60, cash=0.10),
        'Moderate': AssetAllocation(equity=0.60, bonds=0.35, cash=0.05),
        'Aggressive': AssetAllocation(equity=0.80, bonds=0.15, cash=0.05),
        'Very Aggressive': AssetAllocation(equity=0.90, bonds=0.10, cash=0.00)
    }
    return allocations.get(risk_level, allocations['Moderate'])

def show_feedback_form():
    """Display feedback form in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 Feedback & Support")
        
        with st.expander("📝 Send Feedback", expanded=False):
            feedback_type = st.selectbox(
                "Type:",
                ["Feature Request", "Bug Report", "General Feedback", "Question"]
            )
            
            email = st.text_input(
                "Email (optional):",
                placeholder="your.email@example.com",
                help="We'll only use this to respond to your feedback"
            )
            
            subject = st.text_input(
                "Subject:",
                placeholder="Brief description of your feedback"
            )
            
            message = st.text_area(
                "Message:",
                placeholder="Please describe your feedback, feature request, or bug report in detail...",
                height=100
            )
            
            if st.button("📤 Submit Feedback", type="primary"):
                if message.strip() and subject.strip():
                    success = submit_feedback(feedback_type, email, subject, message)
                    if success:
                        st.success("✅ Thank you for your feedback!")
                        track_usage("feedback_submitted", f"Type: {feedback_type}")
                        st.rerun()
                    else:
                        st.error("❌ Error submitting feedback. Please try again.")
                else:
                    st.error("⚠️ Please fill in both subject and message fields.")

def show_usage_stats():
    """Display usage statistics in sidebar"""
    stats = get_usage_stats()
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 App Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Visits", f"{stats['total_visits']:,}")
            st.metric("Today's Visits", f"{stats['today_visits']:,}")
        with col2:
            st.metric("Simulations Run", f"{stats['total_simulations']:,}")

def main():
    """Main Streamlit application"""
    
    # Initialize database
    init_database()
    
    # Track page visit
    track_usage("page_visit", "Main page accessed")

    # Header
    st.title("💰 Advanced Retirement Planning Tool")
    st.markdown("Plan your financial future with Monte Carlo simulation and historical market data")

    # Initialize session state
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None

    # Sidebar for inputs
    st.sidebar.header("📊 Configuration")

    # Load example data button
    if st.sidebar.button("🚀 Load Example Data"):
        st.session_state.example_loaded = True
        track_usage("example_data_loaded", "User loaded example data")

    # Basic Information
    st.sidebar.subheader("💰 Basic Information")

    if 'example_loaded' in st.session_state:
        default_savings = 100000.0
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

    # Asset Allocation
    st.sidebar.subheader("📈 Asset Allocation")
    
    allocation_method = st.sidebar.radio(
        "Choose allocation method:",
        ['Predefined Risk Level', 'Custom Allocation']
    )
    
    if allocation_method == 'Predefined Risk Level':
        risk_level = st.sidebar.selectbox(
            "Risk Level",
            ['Conservative', 'Moderate', 'Aggressive', 'Very Aggressive'],
            index=1,
            help="Conservative: Lower risk, lower returns. Aggressive: Higher risk, higher potential returns."
        )
        asset_allocation = get_risk_level_allocation(risk_level)
        track_usage("allocation_selected", f"Risk level: {risk_level}")
    else:
        st.sidebar.write("**Custom Asset Allocation (must sum to 100%)**")
        equity_pct = st.sidebar.slider("Equity %", 0, 100, 60, 5, help="US Stock Market allocation")
        bonds_pct = st.sidebar.slider("Bonds %", 0, 100, 35, 5, help="US Bond Market allocation")
        cash_pct = st.sidebar.slider("Cash %", 0, 100, 5, 5, help="Cash/Money Market allocation")
        
        total_allocation = equity_pct + bonds_pct + cash_pct
        if total_allocation != 100:
            st.sidebar.error(f"Allocation must sum to 100%. Current: {total_allocation}%")
            st.sidebar.stop()
        
        asset_allocation = AssetAllocation(
            equity=equity_pct/100,
            bonds=bonds_pct/100,
            cash=cash_pct/100
        )
        track_usage("custom_allocation_set", f"Equity: {equity_pct}%, Bonds: {bonds_pct}%, Cash: {cash_pct}%")

    # Display expected returns based on allocation
    expected_return = (
        asset_allocation.equity * HISTORICAL_RETURNS['equity']['mean'] +
        asset_allocation.bonds * HISTORICAL_RETURNS['bonds']['mean'] +
        asset_allocation.cash * HISTORICAL_RETURNS['cash']['mean']
    )
    
    expected_volatility = np.sqrt(
        (asset_allocation.equity ** 2) * (HISTORICAL_RETURNS['equity']['std'] ** 2) +
        (asset_allocation.bonds ** 2) * (HISTORICAL_RETURNS['bonds']['std'] ** 2) +
        (asset_allocation.cash ** 2) * (HISTORICAL_RETURNS['cash']['std'] ** 2)
    )
    
    st.sidebar.info(f"**Expected Annual Return:** {expected_return:.1%}")
    st.sidebar.info(f"**Expected Volatility:** {expected_volatility:.1%}")

    # Other Parameters
    st.sidebar.subheader("⚙️ Other Parameters")

    inflation_rate = st.sidebar.slider(
        "Inflation Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Expected annual inflation rate"
    )

    tax_rate = st.sidebar.slider(
        "Tax Rate on Returns (%)",
        min_value=0.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
        help="Tax rate applied to investment gains"
    )

    # Simulation Settings
    st.sidebar.subheader("🎲 Simulation Settings")

    forecast_years = st.sidebar.slider(
        "Years to Forecast",
        min_value=1,
        max_value=50,
        value=25,
        help="Number of years to project into the future"
    )

    num_simulations = st.sidebar.slider(
        "Number of Simulations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="More simulations = more accurate results but slower computation"
    )

    # Show usage statistics and feedback form in sidebar
    show_usage_stats()
    show_feedback_form()

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

        # Show asset allocation pie chart
        st.plotly_chart(create_allocation_pie_chart(asset_allocation), use_container_width=True)

    with col1:
        # Run Forecast Button
        if st.button("🚀 Run Monte Carlo Simulation", type="primary", use_container_width=True):
            if monthly_income <= 0:
                st.error("Please enter a positive monthly income")
            else:
                # Track simulation run
                track_usage("simulation_run", f"Years: {forecast_years}, Simulations: {num_simulations}")
                
                # Create profile
                profile = FinancialProfile(
                    current_savings=current_savings,
                    monthly_income=monthly_income,
                    monthly_expenses={k: v for k, v in expenses.items() if v > 0},
                    asset_allocation=asset_allocation,
                    inflation_rate=inflation_rate / 100,
                    tax_rate=tax_rate / 100
                )

                # Create forecaster and run simulation
                st.session_state.forecaster = MonteCarloForecaster(profile)
                with st.spinner(f"Running {num_simulations:,} Monte Carlo simulations..."):
                    results = st.session_state.forecaster.monte_carlo_forecast(
                        years=forecast_years, 
                        num_simulations=num_simulations
                    )

                st.success("✅ Monte Carlo simulation completed!")

    # Display results if simulation has been run
    if st.session_state.forecaster and st.session_state.forecaster.forecast_percentiles is not None:

        # Track results viewing
        track_usage("results_viewed", "User viewed simulation results")

        # Get simulation stats
        stats = st.session_state.forecaster._calculate_simulation_stats(
            st.session_state.forecaster.simulation_results,
            st.session_state.forecaster.forecast_percentiles['Date'].tolist()
        )

        # Summary Statistics
        st.subheader("📊 Simulation Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Median Final Value", f"${stats['median_final_balance']:,.0f}")
        with col2:
            st.metric("10th Percentile", f"${stats['percentile_10']:,.0f}")
        with col3:
            st.metric("90th Percentile", f"${stats['percentile_90']:,.0f}")
        with col4:
            st.metric("Mean Final Value", f"${stats['mean_final_balance']:,.0f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Probability of Gain", f"{stats['probability_positive']:.1f}%")
        with col2:
            st.metric("Probability of Doubling", f"{stats['probability_double']:.1f}%")
        with col3:
            st.metric("Value at Risk (95%)", f"${stats['var_95']:,.0f}")

        # Main Monte Carlo Chart
        st.subheader("📈 Monte Carlo Simulation Results")
        fig = create_monte_carlo_charts(st.session_state.forecaster)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Risk Metrics Chart
        st.subheader("📊 Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            risk_fig = create_risk_metrics_chart(stats)
            st.plotly_chart(risk_fig, use_container_width=True)
        
        with col2:
            st.subheader("Key Statistics")
            st.write(f"**Initial Investment:** ${stats['initial_savings']:,.0f}")
            st.write(f"**Standard Deviation:** ${stats['std_final_balance']:,.0f}")
            st.write(f"**Expected Shortfall (5%):** ${stats['expected_shortfall']:,.0f}")
            
            # Calculate some additional metrics
            final_percentiles = st.session_state.forecaster.forecast_percentiles.iloc[-1]
            st.write("**Final Value Percentiles:**")
            st.write(f"- 3rd percentile: ${final_percentiles['P3']:,.0f}")
            st.write(f"- 10th percentile: ${final_percentiles['P10']:,.0f}")
            st.write(f"- 50th percentile: ${final_percentiles['P50']:,.0f}")
            st.write(f"- 90th percentile: ${final_percentiles['P90']:,.0f}")
            st.write(f"- 97th percentile: ${final_percentiles['P97']:,.0f}")

        # Data Export
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Download Simulation Results"):
                track_usage("data_downloaded", "Simulation results CSV downloaded")
                csv = st.session_state.forecaster.forecast_percentiles.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"monte_carlo_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )

        with col2:
            if st.button("📋 Show Percentile Data"):
                track_usage("data_table_viewed", "User viewed percentile data table")
                # Show sample of data
                sample_data = st.session_state.forecaster.forecast_percentiles.iloc[::12]  # Every year
                st.dataframe(
                    sample_data.style.format({
                        'P3': '${:,.0f}',
                        'P10': '${:,.0f}',
                        'P50': '${:,.0f}',
                        'P90': '${:,.0f}',
                        'P97': '${:,.0f}'
                    }),
                    use_container_width=True
                )

    # Educational Content
    with st.expander("📚 Understanding Monte Carlo Simulation"):
        track_usage("education_viewed", "User viewed educational content")
        st.markdown("""
        **What is Monte Carlo Simulation?**
        
        Monte Carlo simulation runs thousands of possible scenarios for your portfolio using historical market data. 
        Instead of assuming a fixed return rate, it accounts for market volatility and uncertainty.
        
        **How to Read the Results:**
        
        - **Median (50th Percentile):** Half of all simulations result in values above this line
        - **10th-90th Percentile Band:** 80% of outcomes fall within this range  
        - **3rd-97th Percentile Band:** 94% of outcomes fall within this range
        - **Value at Risk (95%):** There's only a 5% chance your portfolio will be worth less than this amount
        
        **Historical Data Sources:**
        - **Equity Returns:** Based on S&P 500 historical performance (1957-2021)
        - **Bond Returns:** Based on US 10-year Treasury bond historical performance
        - **Cash Returns:** Based on 3-month Treasury bill rates
        
        This simulation helps you understand the range of possible outcomes and make more informed decisions about your retirement planning.
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>💡 <strong>Disclaimer:</strong> This tool provides estimates based on historical data and Monte Carlo simulation.
            Past performance does not guarantee future results. Consult a financial advisor for personalized advice.</p>
            <p>Built with ❤️ using Streamlit | 
            <a href="#" onclick="document.querySelector('[data-testid=collapsedControl]').click()">📝 Send Feedback</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
