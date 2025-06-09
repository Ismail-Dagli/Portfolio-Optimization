# Portfolio Optimization Pro

A sophisticated, production-grade portfolio optimization application built with Python for professional and educational use in quantitative finance.

## 🎯 Features

### Core Functionality
- **Multiple Optimization Strategies**: Max Sharpe Ratio, Minimum Volatility, Target Return, Equal Weight, Risk Parity
- **Real-time Data**: Fetch historical price data from Yahoo Finance
- **Advanced Analytics**: Backtesting, Monte Carlo simulation, risk analysis
- **Professional Visualizations**: Efficient frontier, allocation charts, performance plots
- **Export Capabilities**: CSV export for all results and data

### Technical Capabilities
- Modern Portfolio Theory (MPT) implementation
- Constraint-based optimization using CVXPY
- Rolling performance metrics
- Correlation analysis
- Value at Risk (VaR) calculations
- Benchmark comparison
- Transaction cost modeling

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## 📦 Dependencies

The application requires the following Python packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualization
- `yfinance` - Financial data
- `PyPortfolioOpt` - Portfolio optimization
- `cvxpy` - Convex optimization
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning utilities
- `plotly` - Interactive plotting
- `tkinter` - GUI framework
- `pillow` - Image processing
- `openpyxl` - Excel file support

## 🏗️ Architecture

The application follows a modular design with clear separation of concerns:

```
portfolio-optimization-app/
│
├── main.py              # Application entry point
├── gui.py               # Main GUI interface
├── data_loader.py       # Data acquisition and processing
├── optimizer.py         # Portfolio optimization algorithms
├── backtester.py        # Backtesting and simulation engine
├── visualizer.py        # Visualization and charting
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Module Overview

#### `data_loader.py`
- Fetches historical stock data from Yahoo Finance
- Validates ticker symbols
- Calculates returns (simple or logarithmic)
- Provides benchmark data
- Data cleaning and preprocessing

#### `optimizer.py`
- Multiple optimization strategies implementation
- Efficient frontier generation
- Risk parity optimization
- Constraint handling
- Modern Portfolio Theory calculations

#### `backtester.py`
- Historical backtesting with configurable rebalancing
- Monte Carlo simulation
- Performance metrics calculation
- Risk analysis (VaR, Expected Shortfall)
- Benchmark comparison

#### `visualizer.py`
- Professional chart generation
- Interactive Plotly dashboards
- Matplotlib integration with Tkinter
- Export capabilities (PDF, PNG)
- Correlation matrices and risk-return plots

#### `gui.py`
- Professional desktop interface
- Tabbed layout for different functions
- Real-time progress tracking
- Threaded operations for responsiveness
- Comprehensive error handling

## 🎮 Usage Guide

### 1. Data Input
- Enter stock tickers (comma-separated): `AAPL, GOOGL, MSFT, AMZN`
- Set date range for historical data
- Choose benchmark (default: S&P 500)
- Load and validate data

### 2. Portfolio Optimization
- Select optimization strategy:
  - **Max Sharpe Ratio**: Maximize risk-adjusted returns
  - **Min Volatility**: Minimize portfolio risk
  - **Target Return**: Achieve specific return target
  - **Equal Weight**: Equal allocation across assets
  - **Risk Parity**: Equal risk contribution
- Set constraints (weight limits, risk-free rate)
- Generate efficient frontier

### 3. Analysis & Backtesting
- Configure backtesting parameters:
  - Rebalancing frequency (daily, weekly, monthly, quarterly)
  - Transaction costs
  - Initial investment amount
- Run Monte Carlo simulations
- Perform comprehensive risk analysis

### 4. Results & Visualization
- View portfolio allocations and metrics
- Generate professional charts:
  - Efficient frontier plots
  - Portfolio allocation pie charts
  - Performance time series
  - Risk-return scatter plots
  - Correlation matrices
- Export results to CSV/PDF

## 📊 Output Examples

### Portfolio Metrics
```
Optimization Results - Max Sharpe Ratio
================================================
Expected Annual Return: 12.45%
Annual Volatility: 18.32%
Sharpe Ratio: 0.679

Portfolio Weights:
AAPL: 25.3%
GOOGL: 31.2%
MSFT: 28.1%
AMZN: 15.4%
```

### Backtest Results
```
Backtest Results
==============================
Total Return: 87.23%
Annualized Return: 13.21%
Volatility: 19.45%
Sharpe Ratio: 0.678
Maximum Drawdown: -15.67%
Win Rate: 54.32%
```

## 🔧 Advanced Features

### Custom Constraints
```python
# Example: Set maximum 30% allocation per asset
max_weight_per_asset = 0.30

# Example: Sector constraints (requires additional implementation)
sector_limits = {
    'Technology': 0.50,
    'Healthcare': 0.20
}
```

### Risk Models
The application supports various risk models:
- Sample covariance matrix
- Shrinkage estimators
- Exponentially weighted covariance
- Risk factor models (extensible)

### Simulation Parameters
```python
# Monte Carlo simulation settings
num_simulations = 10000
time_horizon_days = 252  # 1 year
confidence_levels = [0.05, 0.10, 0.95]
```

## 🛠️ Customization

### Adding New Optimization Strategies
Extend the `PortfolioOptimizer` class in `optimizer.py`:

```python
def custom_optimization_strategy(self) -> Dict:
    """
    Implement your custom optimization logic here.
    """
    # Your optimization code
    return {
        'weights': weights_dict,
        'expected_return': expected_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'method': 'Custom Strategy'
    }
```

### Adding New Visualizations
Extend the `PortfolioVisualizer` class in `visualizer.py`:

```python
def plot_custom_chart(self, data) -> plt.Figure:
    """
    Create custom visualization.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    # Your plotting code here
    return fig
```

## 🚨 Error Handling

The application includes comprehensive error handling:
- Data validation and cleaning
- Network error handling for data downloads
- Optimization convergence checks
- GUI responsiveness during long operations
- Detailed logging for debugging

## 📈 Performance Considerations

- Efficient data structures using pandas
- Vectorized calculations with NumPy
- Threaded operations for GUI responsiveness
- Memory-efficient handling of large datasets
- Optimized visualization rendering

## 🔒 Risk Disclaimer

This application is for educational and research purposes. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.

## 🤝 Contributing

Contributions are welcome! Please consider:
- Adding new optimization algorithms
- Implementing additional risk models
- Enhancing visualization capabilities
- Improving performance and efficiency
- Adding more comprehensive testing

## 📄 License

This project is open source. Please respect the licenses of all dependencies.

## 📞 Support

For questions or issues:
1. Check the application logs (`portfolio_optimizer.log`)
2. Verify all dependencies are correctly installed
3. Ensure internet connection for data downloads
4. Review error messages in the GUI status bar

## 🏆 Acknowledgments

Built using excellent open-source libraries:
- PyPortfolioOpt for optimization algorithms
- yfinance for financial data
- matplotlib/seaborn for visualization
- pandas/numpy for data processing
- tkinter for GUI framework

---

**Happy Optimizing! 📊💰**
