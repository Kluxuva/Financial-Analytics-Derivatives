# Financial Analytics & Derivatives Project

A comprehensive financial analysis project implementing portfolio optimization using Efficient Frontier & Monte Carlo simulation, along with option pricing analysis using Black-Scholes model and Put-Call Parity arbitrage detection.


## Overview

This project demonstrates advanced financial analytics techniques combining portfolio theory and derivatives pricing:

1. **Portfolio Optimization**: Uses Modern Portfolio Theory to find optimal asset allocation
2. **Efficient Frontier**: Visualizes risk-return tradeoffs across 10,000 random portfolios
3. **Monte Carlo Simulation**: Projects portfolio performance over 1 year with 1,000 simulations
4. **Black-Scholes Pricing**: Calculates theoretical option prices and Greeks
5. **Put-Call Parity**: Detects pricing inefficiencies and arbitrage opportunities

## Features

### Portfolio Analysis
- ✅ **10-Stock Portfolio**: Analysis of major US stocks (AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, V, JNJ, WMT, PG)
- ✅ **Data Source**: Historical data from Stooq (2 years)
- ✅ **Optimization**: Maximum Sharpe Ratio portfolio using scipy optimization
- ✅ **Risk Metrics**: Expected return, volatility, Sharpe ratio, correlation analysis

### Visualizations (11+ Plots)
1. Historical stock prices
2. Daily returns distribution (10 histograms)
3. Correlation matrix heatmap
4. Efficient Frontier with 10,000 portfolios
5. Optimal portfolio allocation (pie + bar charts)
6. Monte Carlo simulation paths
7. Final portfolio value distribution
8. Black-Scholes call/put option prices
9. Option Greeks (Delta, Gamma, Vega, Theta, Rho)
10. Put-Call Parity analysis
11. Arbitrage opportunity detection

### Derivatives Pricing
- **Black-Scholes Model**: European call and put option pricing
- **Greeks Calculation**: Delta, Gamma, Vega, Theta, Rho
- **Put-Call Parity**: Validation and arbitrage detection
- **Sensitivity Analysis**: Option prices across different stock prices

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

##  Usage

The script automatically performs all analyses and generates visualizations:

```bash
python financial_analysis.py
```

### Output
- Console output with detailed metrics
- `plots/` directory containing all visualizations
- Performance statistics and optimization results

### Customization

You can modify the following parameters in `financial_analysis.py`:

```python
# Change stock tickers
tickers = ['AAPL.US', 'MSFT.US', ...]  # Add .US suffix for Stooq

# Adjust date range
start_date = end_date - timedelta(days=365*2)  # 2 years

# Monte Carlo parameters
num_simulations = 1000
days = 252  # Trading days

# Option parameters
K = S  # Strike price
T = 0.25  # Time to maturity (years)
r = 0.05  # Risk-free rate
```

## 🔬 Project Components

### 1. Data Collection
- Fetches historical stock data from Stooq
- Cleans and processes data
- Calculates daily returns

### 2. Portfolio Optimization
```python
def optimize_portfolio(returns, cov_matrix):
    # Maximizes Sharpe Ratio
    # Constraints: weights sum to 1, no short selling
    # Returns: optimal weights
```

**Key Metrics:**
- Expected Annual Return
- Annual Volatility (Standard Deviation)
- Sharpe Ratio (Return per unit of risk)

### 3. Efficient Frontier
- Generates 10,000 random portfolios
- Plots risk-return combinations
- Highlights optimal portfolio

### 4. Monte Carlo Simulation
- Simulates 1,000 potential portfolio paths
- Projects performance over 1 year (252 trading days)
- Calculates probability distributions
- Shows 5th and 95th percentiles

### 5. Black-Scholes Model

**Call Option Price:**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**Put Option Price:**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

Where:
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### 6. Put-Call Parity
```
C - P = S₀ - Ke^(-rT)
```

Detects arbitrage when:
- |C - P - (S₀ - Ke^(-rT))| > $0.01

**Arbitrage Strategies:**
- If C - P > S₀ - PV(K): Sell call, buy put, buy stock, borrow PV(K)
- If C - P < S₀ - PV(K): Buy call, sell put, sell stock, lend PV(K)

## 📈 Results & Visualizations

### Sample Output

```
OPTIMAL PORTFOLIO METRICS:
Expected Annual Return: 18.45%
Annual Volatility: 22.31%
Sharpe Ratio: 0.827

MONTE CARLO RESULTS (1 Year):
Mean Final Value: $115,234.56
5th Percentile: $87,543.21
95th Percentile: $148,765.43
Probability of Profit: 78.3%

OPTION PRICING:
Current Stock Price: $175.32
Call Price: $8.45
Put Price: $7.89

PUT-CALL PARITY:
Left Side (C-P): $0.56
Right Side (S-PV(K)): $0.56
Difference: $0.0001
✓ No arbitrage - Market is efficient
```

## 🛠 Technical Details

### Optimization Algorithm
- **Method**: Sequential Least Squares Programming (SLSQP)
- **Objective**: Maximize Sharpe Ratio
- **Constraints**: 
  - Weights sum to 1
  - No short selling (weights ≥ 0)

### Statistical Methods
- **Covariance Matrix**: Measures asset correlations
- **Multivariate Normal Distribution**: For Monte Carlo simulation
- **Cumulative Distribution Function**: For Black-Scholes (norm.cdf)
- **Probability Density Function**: For Greeks calculation (norm.pdf)

### Performance Optimization
- Vectorized NumPy operations
- Efficient matrix calculations
- Minimized loop iterations


## Key Concepts

### Modern Portfolio Theory (MPT)
- Developed by Harry Markowitz
- Focuses on risk-return tradeoff
- Diversification reduces unsystematic risk

### Efficient Frontier
- Set of optimal portfolios
- Maximum return for given risk level
- Minimum risk for given return level

### Sharpe Ratio
```
Sharpe Ratio = (Rₚ - Rₓ) / σₚ
```
- Rₚ = Portfolio return
- Rₓ = Risk-free rate (assumed 0 in this implementation)
- σₚ = Portfolio standard deviation

### Black-Scholes Assumptions
1. European options (exercise at maturity only)
2. No dividends
3. Efficient markets (no arbitrage)
4. Constant risk-free rate and volatility
5. Lognormal stock price distribution

[Stooq](https://stooq.com/)

---

Made with ❤️ for financial analysis and learning
