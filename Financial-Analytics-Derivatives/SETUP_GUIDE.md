# Setup Guide

## Quick Start (3 Steps)

### 1. Clone the Repository
```bash
git clone https://github.com/Kluxuva/Financial-Analytics-Derivatives.git
cd Financial-Analytics-Derivatives
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
```bash
python financial_analysis.py
```

That's it! All visualizations will be generated in the `plots/` directory.

---

## Detailed Setup Instructions

### Prerequisites

**Required:**
- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for real data from Stooq)

**Optional:**
- Virtual environment (recommended)
- Jupyter Notebook (for interactive analysis)

### Installation Methods

#### Method 1: Using pip (Recommended)

```bash
# Navigate to project directory
cd Financial-Analytics-Derivatives

# Install all dependencies
pip install -r requirements.txt

# Run the script
python financial_analysis.py
```

#### Method 2: Using Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the script
python financial_analysis.py
```

#### Method 3: Using Conda

```bash
# Create conda environment
conda create -n fintech python=3.10

# Activate environment
conda activate fintech

# Install dependencies
pip install -r requirements.txt

# Run the script
python financial_analysis.py
```

---

## Package Details

The project uses the following Python packages:

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation and analysis |
| numpy | Latest | Numerical computing |
| matplotlib | Latest | Plotting and visualization |
| seaborn | Latest | Statistical data visualization |
| pandas-datareader | Latest | Fetch financial data from Stooq |
| scipy | Latest | Scientific computing and optimization |

---

## Data Sources

### Real Data (Default)
- **Source:** Stooq (https://stooq.com/)
- **Format:** Historical daily prices
- **Tickers:** US stocks with .US suffix
- **Period:** Last 2 years

### Synthetic Data (Fallback)
- If Stooq is unavailable, the script automatically generates synthetic data
- Uses geometric Brownian motion to simulate realistic price movements
- Maintains similar statistical properties to real data

---

## Project Structure

```
Financial-Analytics-Derivatives/
│
├── financial_analysis.py       # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── SETUP_GUIDE.md              # This file
├── .gitignore                   # Git ignore rules
│
└── plots/                       # Generated visualizations
    ├── stock_prices.png
    ├── returns_distribution.png
    ├── correlation_matrix.png
    ├── efficient_frontier.png
    ├── portfolio_allocation.png
    ├── monte_carlo_simulation.png
    ├── option_prices.png
    ├── option_greeks.png
    └── put_call_parity.png
```

---

## Customization Guide

### 1. Change Stock Tickers

Edit `financial_analysis.py` line ~452:

```python
# Original
tickers = ['AAPL.US', 'MSFT.US', 'GOOGL.US', ...]

# Custom example
tickers = ['NFLX.US', 'DIS.US', 'BA.US', ...]
```

### 2. Adjust Time Period

Edit `financial_analysis.py` line ~456:

```python
# Original (2 years)
start_date = end_date - timedelta(days=365*2)

# Custom (5 years)
start_date = end_date - timedelta(days=365*5)
```

### 3. Monte Carlo Parameters

Edit `financial_analysis.py` line ~504:

```python
# Original
mc_results = monte_carlo_simulation(stock_data, optimal_weights, 
                                   num_simulations=1000, days=252)

# More simulations, longer period
mc_results = monte_carlo_simulation(stock_data, optimal_weights, 
                                   num_simulations=5000, days=504)
```

### 4. Option Parameters

Edit `financial_analysis.py` line ~524-528:

```python
# Original
K = S  # At-the-money
T = 0.25  # 3 months
r = 0.05  # 5% risk-free rate

# Custom
K = S * 1.1  # Out-of-the-money (10% above current price)
T = 1.0  # 1 year
r = 0.03  # 3% risk-free rate
```

---

## Troubleshooting

### Issue: Package Installation Fails

**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip

# Install packages one by one
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install pandas-datareader
pip install scipy
```

### Issue: Data Fetch Errors

**Problem:** Network issues or Stooq unavailable

**Solution:** The script automatically uses synthetic data as fallback. No action needed.

**For Real Data:**
- Check internet connection
- Verify firewall settings
- Try different network/VPN

### Issue: Plots Not Generating

**Solution:**
```bash
# Ensure plots directory exists
mkdir plots

# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# If using SSH/remote, set backend
export MPLBACKEND=Agg
python financial_analysis.py
```

### Issue: Import Errors

**Solution:**
```bash
# Verify all packages installed
pip list | grep -E "pandas|numpy|matplotlib|seaborn|scipy"

# Reinstall if missing
pip install -r requirements.txt --force-reinstall
```

---

## Running in Different Environments

### Jupyter Notebook

Create a new notebook and run:

```python
# Run the entire script
%run financial_analysis.py

# Or import functions individually
from financial_analysis import *

# Use functions interactively
optimal_weights = optimize_portfolio(returns, cov_matrix)
```

### Google Colab

```python
# Install dependencies
!pip install pandas numpy matplotlib seaborn pandas-datareader scipy

# Clone repository
!git clone https://github.com/Kluxuva/Financial-Analytics-Derivatives.git
%cd Financial-Analytics-Derivatives

# Run script
!python financial_analysis.py

# View plots
from IPython.display import Image, display
display(Image('plots/efficient_frontier.png'))
```

### Docker (Advanced)

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "financial_analysis.py"]
```

```bash
# Build and run
docker build -t financial-analytics .
docker run -v $(pwd)/plots:/app/plots financial-analytics
```

---

## Performance Tips

### Faster Execution

1. **Reduce simulations:**
   ```python
   num_portfolios = 5000  # Instead of 10000
   num_simulations = 500  # Instead of 1000
   ```

2. **Shorter data period:**
   ```python
   start_date = end_date - timedelta(days=365)  # 1 year
   ```

3. **Fewer stocks:**
   ```python
   tickers = ['AAPL.US', 'MSFT.US', 'GOOGL.US', 'AMZN.US', 'TSLA.US']
   ```

### Memory Optimization

For large datasets or many simulations:

```python
# Process in chunks
for i in range(0, num_simulations, 100):
    batch_results = monte_carlo_simulation(...)
    # Save or process batch
```

---

## Getting Help

### Resources
- **Documentation:** See README.md
- **Issues:** Open an issue on GitHub
- **Discussions:** Use GitHub Discussions

### Common Questions

**Q: Can I use different data sources?**  
A: Yes! Modify the `fetch_stock_data()` function to use Yahoo Finance, Alpha Vantage, or other sources.

**Q: How do I interpret the results?**  
A: See the "Results & Visualizations" section in README.md

**Q: Can I add more analysis?**  
A: Absolutely! The code is modular - just add new functions and call them in `main()`

**Q: Is this suitable for production?**  
A: This is an educational/demonstration project. For production, add error handling, logging, and data validation.

---

## Next Steps

After successful setup:

1. ✅ Run the basic analysis
2. ✅ Examine the generated plots
3. ✅ Customize parameters for your needs
4. ✅ Add your own stocks
5. ✅ Extend with new features

Happy analyzing! 📊📈
