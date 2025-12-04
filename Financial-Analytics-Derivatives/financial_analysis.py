

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# DATA COLLECTION

def generate_synthetic_data(tickers, start_date, end_date):
    """Generate synthetic stock data for demonstration"""
    print("Generating synthetic stock data for demonstration...")
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Initial prices and parameters for each stock
    initial_prices = [150, 300, 120, 140, 200, 150, 220, 160, 140, 130]
    annual_returns = [0.15, 0.18, 0.16, 0.20, 0.25, 0.12, 0.14, 0.10, 0.11, 0.09]
    volatilities = [0.30, 0.28, 0.32, 0.35, 0.45, 0.25, 0.27, 0.20, 0.22, 0.19]
    
    stock_data = pd.DataFrame(index=dates)
    
    for i, ticker in enumerate(tickers):
        # Generate returns using geometric Brownian motion
        dt = 1/252  # Daily time step
        mu = annual_returns[i]
        sigma = volatilities[i]
        
        returns = np.random.normal((mu - 0.5 * sigma**2) * dt, 
                                   sigma * np.sqrt(dt), 
                                   len(dates))
        
        # Generate price path
        price_path = initial_prices[i] * np.exp(np.cumsum(returns))
        stock_data[ticker] = price_path
    
    return stock_data

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch stock data from Stooq with fallback to synthetic data"""
    print("Attempting to fetch stock data from Stooq...")
    stock_data = pd.DataFrame()
    
    for ticker in tickers:
        try:
            df = pdr.DataReader(ticker, 'stooq', start_date, end_date)
            stock_data[ticker] = df['Close']
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    # If no data was fetched, use synthetic data
    if stock_data.empty:
        print("\n⚠️  Unable to fetch real data. Using synthetic data for demonstration.")
        print("Note: To use real data, ensure network access to stooq.com\n")
        stock_data = generate_synthetic_data(tickers, start_date, end_date)
    else:
        stock_data = stock_data.sort_index()
    
    return stock_data

# PORTFOLIO OPTIMIZATION

def calculate_portfolio_metrics(weights, returns, cov_matrix):
    """Calculate portfolio return and risk"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio

def negative_sharpe(weights, returns, cov_matrix):
    """Negative Sharpe ratio for optimization"""
    return -calculate_portfolio_metrics(weights, returns, cov_matrix)[2]

def optimize_portfolio(returns, cov_matrix):
    """Find optimal portfolio weights using Efficient Frontier"""
    num_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1/num_assets] * num_assets)
    
    # Optimize for maximum Sharpe ratio
    optimal = minimize(negative_sharpe, initial_weights, 
                      args=(returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return optimal.x

def generate_efficient_frontier(returns, cov_matrix, num_portfolios=10000):
    """Generate random portfolios for Efficient Frontier"""
    num_assets = len(returns.columns)
    results = np.zeros((4, num_portfolios))
    
    print("Generating Efficient Frontier...")
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        portfolio_return, portfolio_std, sharpe = calculate_portfolio_metrics(
            weights, returns, cov_matrix)
        
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = sharpe
        results[3,i] = portfolio_return / portfolio_std  # Sharpe ratio
    
    return results

# MONTE CARLO SIMULATION

def monte_carlo_simulation(stock_data, optimal_weights, num_simulations=1000, days=252):
    """Run Monte Carlo simulation for portfolio"""
    print("Running Monte Carlo simulation...")
    returns = stock_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Portfolio initial value
    initial_portfolio = 100000
    
    # Run simulations
    simulation_results = np.zeros((num_simulations, days))
    
    for i in range(num_simulations):
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, days)
        portfolio_values = initial_portfolio * np.cumprod(1 + np.dot(daily_returns, optimal_weights))
        simulation_results[i] = portfolio_values
    
    return simulation_results

# BLACK-SCHOLES MODEL

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma):
    """Calculate option Greeks"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Delta
    call_delta = norm.cdf(d1)
    put_delta = -norm.cdf(-d1)
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Theta
    call_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                  - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
    put_theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    
    # Rho
    call_rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100
    put_rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100
    
    return {
        'Call Delta': call_delta, 'Put Delta': put_delta,
        'Gamma': gamma, 'Vega': vega,
        'Call Theta': call_theta, 'Put Theta': put_theta,
        'Call Rho': call_rho, 'Put Rho': put_rho
    }

# PUT-CALL PARITY 

def check_put_call_parity(S, K, T, r, call_price, put_price):
    """Check Put-Call Parity and detect arbitrage opportunities"""
    # Put-Call Parity: C - P = S - K*e^(-r*T)
    left_side = call_price - put_price
    right_side = S - K * np.exp(-r * T)
    difference = left_side - right_side
    
    # If difference > 0: Overpriced call or underpriced put (sell call, buy put)
    # If difference < 0: Underpriced call or overpriced put (buy call, sell put)
    
    return {
        'Left Side (C-P)': left_side,
        'Right Side (S-PV(K))': right_side,
        'Difference': difference,
        'Arbitrage': abs(difference) > 0.01
    }

# VISUALIZATION FUNCTIONS

def plot_stock_prices(stock_data, save_path='plots/stock_prices.png'):
    """Plot historical stock prices"""
    fig, ax = plt.subplots(figsize=(14, 7))
    for column in stock_data.columns:
        ax.plot(stock_data.index, stock_data[column], label=column, linewidth=2)
    ax.set_title('Historical Stock Prices', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_returns_distribution(returns, save_path='plots/returns_distribution.png'):
    """Plot returns distribution"""
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    axes = axes.flatten()
    
    for idx, column in enumerate(returns.columns):
        axes[idx].hist(returns[column], bins=50, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{column} Returns', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Daily Return', fontsize=8)
        axes[idx].set_ylabel('Frequency', fontsize=8)
        axes[idx].axvline(returns[column].mean(), color='red', 
                         linestyle='--', linewidth=2, label='Mean')
        axes[idx].legend(fontsize=7)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Daily Returns Distribution', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_correlation_matrix(returns, save_path='plots/correlation_matrix.png'):
    """Plot correlation matrix heatmap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation = returns.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    ax.set_title('Stock Returns Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_efficient_frontier(results, optimal_return, optimal_std, 
                           save_path='plots/efficient_frontier.png'):
    """Plot Efficient Frontier"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scatter = ax.scatter(results[1], results[0], c=results[2], 
                        cmap='viridis', alpha=0.5, s=10)
    ax.scatter(optimal_std, optimal_return, c='red', s=200, 
              marker='*', edgecolors='black', linewidths=2,
              label='Optimal Portfolio', zorder=5)
    
    ax.set_title('Efficient Frontier - Portfolio Optimization', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Volatility (Standard Deviation)', fontsize=12)
    ax.set_ylabel('Expected Annual Return', fontsize=12)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_portfolio_allocation(optimal_weights, tickers, 
                              save_path='plots/portfolio_allocation.png'):
    """Plot optimal portfolio allocation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
    wedges, texts, autotexts = ax1.pie(optimal_weights, labels=tickers, 
                                         autopct='%1.1f%%', colors=colors,
                                         startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Optimal Portfolio Allocation (Pie Chart)', 
                 fontsize=14, fontweight='bold')
    
    # Bar chart
    ax2.bar(tickers, optimal_weights, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_title('Optimal Portfolio Allocation (Bar Chart)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Stock Ticker', fontsize=12)
    ax2.set_ylabel('Weight', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_monte_carlo(simulation_results, save_path='plots/monte_carlo_simulation.png'):
    """Plot Monte Carlo simulation results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Simulation paths
    for i in range(min(100, len(simulation_results))):
        ax1.plot(simulation_results[i], alpha=0.1, color='blue')
    
    mean_path = simulation_results.mean(axis=0)
    ax1.plot(mean_path, color='red', linewidth=3, label='Mean Path')
    ax1.fill_between(range(len(mean_path)), 
                     np.percentile(simulation_results, 5, axis=0),
                     np.percentile(simulation_results, 95, axis=0),
                     alpha=0.2, color='red', label='5th-95th Percentile')
    
    ax1.set_title('Monte Carlo Portfolio Simulation (1000 Paths)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Final distribution
    final_values = simulation_results[:, -1]
    ax2.hist(final_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(final_values.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: ${final_values.mean():,.0f}')
    ax2.axvline(np.percentile(final_values, 5), color='orange', 
               linestyle='--', linewidth=2, label=f'5th Percentile: ${np.percentile(final_values, 5):,.0f}')
    ax2.axvline(np.percentile(final_values, 95), color='purple', 
               linestyle='--', linewidth=2, label=f'95th Percentile: ${np.percentile(final_values, 95):,.0f}')
    
    ax2.set_title('Final Portfolio Value Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Portfolio Value ($)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_option_prices(S_range, K, T, r, sigma, 
                       save_path='plots/option_prices.png'):
    """Plot option prices across stock prices"""
    call_prices = [black_scholes_call(S, K, T, r, sigma) for S in S_range]
    put_prices = [black_scholes_put(S, K, T, r, sigma) for S in S_range]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(S_range, call_prices, label='Call Option', linewidth=2.5, color='green')
    ax.plot(S_range, put_prices, label='Put Option', linewidth=2.5, color='red')
    ax.axvline(K, color='black', linestyle='--', linewidth=2, label=f'Strike Price: ${K}')
    
    ax.set_title('Black-Scholes Option Pricing', fontsize=16, fontweight='bold')
    ax.set_xlabel('Stock Price ($)', fontsize=12)
    ax.set_ylabel('Option Price ($)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_greeks(S_range, K, T, r, sigma, save_path='plots/option_greeks.png'):
    """Plot option Greeks"""
    greeks_data = {greek: [] for greek in ['Call Delta', 'Put Delta', 'Gamma', 'Vega']}
    
    for S in S_range:
        greeks = calculate_greeks(S, K, T, r, sigma)
        greeks_data['Call Delta'].append(greeks['Call Delta'])
        greeks_data['Put Delta'].append(greeks['Put Delta'])
        greeks_data['Gamma'].append(greeks['Gamma'])
        greeks_data['Vega'].append(greeks['Vega'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(S_range, greeks_data['Call Delta'], linewidth=2.5, color='green')
    axes[0, 0].set_title('Call Delta', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(K, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_ylabel('Delta', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(S_range, greeks_data['Put Delta'], linewidth=2.5, color='red')
    axes[0, 1].set_title('Put Delta', fontsize=12, fontweight='bold')
    axes[0, 1].axvline(K, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylabel('Delta', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(S_range, greeks_data['Gamma'], linewidth=2.5, color='blue')
    axes[1, 0].set_title('Gamma (Same for Call & Put)', fontsize=12, fontweight='bold')
    axes[1, 0].axvline(K, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Stock Price ($)', fontsize=10)
    axes[1, 0].set_ylabel('Gamma', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(S_range, greeks_data['Vega'], linewidth=2.5, color='purple')
    axes[1, 1].set_title('Vega (Same for Call & Put)', fontsize=12, fontweight='bold')
    axes[1, 1].axvline(K, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Stock Price ($)', fontsize=10)
    axes[1, 1].set_ylabel('Vega', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Option Greeks Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_put_call_parity(S_range, K, T, r, sigma, 
                         save_path='plots/put_call_parity.png'):
    """Plot Put-Call Parity analysis"""
    left_side = []
    right_side = []
    differences = []
    
    for S in S_range:
        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)
        parity = check_put_call_parity(S, K, T, r, call, put)
        
        left_side.append(parity['Left Side (C-P)'])
        right_side.append(parity['Right Side (S-PV(K))'])
        differences.append(parity['Difference'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Put-Call Parity
    ax1.plot(S_range, left_side, label='C - P', linewidth=2.5, color='blue')
    ax1.plot(S_range, right_side, label='S - PV(K)', linewidth=2.5, 
            color='orange', linestyle='--')
    ax1.axvline(K, color='black', linestyle='--', linewidth=2, 
               label=f'Strike: ${K}', alpha=0.5)
    ax1.set_title('Put-Call Parity: C - P = S - PV(K)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Stock Price ($)', fontsize=12)
    ax1.set_ylabel('Value ($)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Arbitrage opportunities
    ax2.plot(S_range, differences, linewidth=2.5, color='red')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(0.01, color='green', linestyle='--', linewidth=1.5, 
               label='Arbitrage Threshold (+$0.01)', alpha=0.7)
    ax2.axhline(-0.01, color='green', linestyle='--', linewidth=1.5, 
               label='Arbitrage Threshold (-$0.01)', alpha=0.7)
    ax2.fill_between(S_range, -0.01, 0.01, alpha=0.2, color='green')
    ax2.axvline(K, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    ax2.set_title('Pricing Inefficiencies (Arbitrage Opportunities)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Stock Price ($)', fontsize=12)
    ax2.set_ylabel('Price Difference ($)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# MAIN EXECUTION

def main():
    """Main execution function"""
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    print("="*60)
    print("FINANCIAL ANALYTICS & DERIVATIVES PROJECT")
    print("="*60)
    
    # Define 10 stocks for portfolio
    tickers = ['AAPL.US', 'MSFT.US', 'GOOGL.US', 'AMZN.US', 'TSLA.US',
               'JPM.US', 'V.US', 'JNJ.US', 'WMT.US', 'PG.US']
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    # Step 1: Fetch Data
    print("\n" + "="*60)
    print("STEP 1: DATA COLLECTION")
    print("="*60)
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    stock_data = stock_data.dropna()
    print(f"Data shape: {stock_data.shape}")
    print(f"Date range: {stock_data.index[0]} to {stock_data.index[-1]}")
    
    # Calculate returns
    returns = stock_data.pct_change().dropna()
    
    # Step 2: Visualize Data
    print("\n" + "="*60)
    print("STEP 2: DATA VISUALIZATION")
    print("="*60)
    plot_stock_prices(stock_data)
    plot_returns_distribution(returns)
    plot_correlation_matrix(returns)
    
    # Step 3: Portfolio Optimization
    print("\n" + "="*60)
    print("STEP 3: PORTFOLIO OPTIMIZATION")
    print("="*60)
    cov_matrix = returns.cov()
    optimal_weights = optimize_portfolio(returns, cov_matrix)
    
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight*100:.2f}%")
    
    optimal_return, optimal_std, optimal_sharpe = calculate_portfolio_metrics(
        optimal_weights, returns, cov_matrix)
    
    print(f"\nOptimal Portfolio Metrics:")
    print(f"Expected Annual Return: {optimal_return*100:.2f}%")
    print(f"Annual Volatility: {optimal_std*100:.2f}%")
    print(f"Sharpe Ratio: {optimal_sharpe:.3f}")
    
    # Step 4: Efficient Frontier
    print("\n" + "="*60)
    print("STEP 4: EFFICIENT FRONTIER")
    print("="*60)
    efficient_frontier_results = generate_efficient_frontier(returns, cov_matrix)
    plot_efficient_frontier(efficient_frontier_results, optimal_return, optimal_std)
    plot_portfolio_allocation(optimal_weights, tickers)
    
    # Step 5: Monte Carlo Simulation
    print("\n" + "="*60)
    print("STEP 5: MONTE CARLO SIMULATION")
    print("="*60)
    mc_results = monte_carlo_simulation(stock_data, optimal_weights, 
                                       num_simulations=1000, days=252)
    plot_monte_carlo(mc_results)
    
    final_values = mc_results[:, -1]
    print(f"\nMonte Carlo Results (1 Year Projection):")
    print(f"Mean Final Value: ${final_values.mean():,.2f}")
    print(f"5th Percentile: ${np.percentile(final_values, 5):,.2f}")
    print(f"95th Percentile: ${np.percentile(final_values, 95):,.2f}")
    print(f"Probability of Profit: {(final_values > 100000).sum() / len(final_values) * 100:.2f}%")
    
    # Step 6: Black-Scholes Option Pricing
    print("\n" + "="*60)
    print("STEP 6: BLACK-SCHOLES OPTION PRICING")
    print("="*60)
    
    # Use average stock price as example
    S = stock_data.iloc[-1].mean()  # Current average price
    K = S  # At-the-money option
    T = 0.25  # 3 months
    r = 0.05  # 5% risk-free rate
    sigma = returns.std().mean() * np.sqrt(252)  # Annualized volatility
    
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    
    print(f"\nOption Parameters:")
    print(f"Current Stock Price (S): ${S:.2f}")
    print(f"Strike Price (K): ${K:.2f}")
    print(f"Time to Maturity (T): {T} years")
    print(f"Risk-Free Rate (r): {r*100}%")
    print(f"Volatility (σ): {sigma*100:.2f}%")
    
    print(f"\nOption Prices:")
    print(f"Call Price: ${call_price:.2f}")
    print(f"Put Price: ${put_price:.2f}")
    
    # Calculate Greeks
    greeks = calculate_greeks(S, K, T, r, sigma)
    print(f"\nOption Greeks:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.4f}")
    
    # Visualize options
    S_range = np.linspace(S * 0.7, S * 1.3, 100)
    plot_option_prices(S_range, K, T, r, sigma)
    plot_greeks(S_range, K, T, r, sigma)
    
    # Step 7: Put-Call Parity Analysis
    print("\n" + "="*60)
    print("STEP 7: PUT-CALL PARITY ANALYSIS")
    print("="*60)
    
    parity_check = check_put_call_parity(S, K, T, r, call_price, put_price)
    print(f"\nPut-Call Parity Check:")
    for key, value in parity_check.items():
        if isinstance(value, bool):
            print(f"{key}: {value}")
        else:
            print(f"{key}: ${value:.4f}")
    
    if parity_check['Arbitrage']:
        print("\n⚠️ ARBITRAGE OPPORTUNITY DETECTED!")
        if parity_check['Difference'] > 0:
            print("Strategy: Sell Call + Buy Put + Buy Stock + Borrow PV(K)")
        else:
            print("Strategy: Buy Call + Sell Put + Sell Stock + Lend PV(K)")
    else:
        print("\n✓ No arbitrage opportunity - Market is efficient")
    
    plot_put_call_parity(S_range, K, T, r, sigma)
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("All visualizations saved in 'plots/' directory")
    print("="*60)

if __name__ == "__main__":
    main()
