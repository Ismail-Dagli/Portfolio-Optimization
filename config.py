"""
Configuration file for Portfolio Optimization Application
"""

import os
from typing import Dict

# Application Configuration
APP_CONFIG = {
    'name': 'Portfolio Optimization Pro',
    'version': '1.0.0',
    'author': 'Portfolio Optimization Team',
    'description': 'Professional Portfolio Optimization Tool for Quantitative Finance'
}

# Data Configuration
DATA_CONFIG = {
    'default_tickers': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'default_benchmark': '^GSPC',
    'data_source': 'yahoo',
    'default_period_years': 3,
    'min_data_points': 100,
    'max_missing_data_ratio': 0.5
}

# Optimization Configuration
OPTIMIZATION_CONFIG = {
    'default_risk_free_rate': 0.02,
    'default_target_return': 0.10,
    'default_l2_gamma': 0.1,
    'max_weight_default': 0.4,
    'min_weight_default': 0.0,
    'strategies': [
        'Max Sharpe Ratio',
        'Min Volatility',
        'Target Return',
        'Equal Weight',
        'Risk Parity'
    ]
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'default_rebalance_frequency': 'monthly',
    'default_transaction_cost': 0.001,  # 0.1%
    'default_initial_investment': 100000,
    'rebalance_frequencies': ['daily', 'weekly', 'monthly', 'quarterly'],
    'lookback_window': 252  # 1 year
}

# Monte Carlo Configuration
MONTE_CARLO_CONFIG = {
    'default_simulations': 1000,
    'max_simulations': 10000,
    'default_time_horizon': 252,  # 1 year
    'confidence_levels': [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    'random_seed': 42
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'chart_types': [
        'Portfolio Allocation',
        'Efficient Frontier',
        'Performance Chart',
        'Risk-Return Scatter',
        'Correlation Matrix',
        'Monte Carlo Results',
        'Rolling Metrics'
    ]
}

# GUI Configuration
GUI_CONFIG = {
    'window_size': '1400x900',
    'theme': 'default',
    'font_size': 10,
    'tab_names': [
        'Data Input',
        'Optimization',
        'Analysis & Backtesting',
        'Results & Visualization'
    ]
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_name': 'portfolio_optimizer.log',
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5
}

# Export Configuration
EXPORT_CONFIG = {
    'csv_separator': ',',
    'decimal_places': 4,
    'date_format': '%Y-%m-%d',
    'supported_formats': ['csv', 'xlsx', 'pdf'],
    'default_export_path': os.path.expanduser('~/Documents/PortfolioOptimization')
}

# Risk Analysis Configuration
RISK_CONFIG = {
    'confidence_levels': [0.95, 0.99],
    'var_methods': ['historical', 'parametric', 'monte_carlo'],
    'stress_test_scenarios': {
        'market_crash': -0.20,
        'recession': -0.15,
        'volatility_spike': 2.0
    }
}

# Performance Metrics Configuration
METRICS_CONFIG = {
    'annualization_factor': 252,
    'risk_free_rate': 0.02,
    'benchmark_ticker': '^GSPC',
    'rolling_window_days': 252,
    'metrics_to_calculate': [
        'total_return',
        'annualized_return',
        'volatility',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'calmar_ratio',
        'omega_ratio',
        'var_95',
        'cvar_95'
    ]
}

# Validation Rules
VALIDATION_RULES = {
    'min_tickers': 2,
    'max_tickers': 50,
    'min_date_range_days': 30,
    'max_date_range_years': 20,
    'weight_tolerance': 1e-6,
    'optimization_tolerance': 1e-8
}


def get_config(section: str) -> Dict:
    """
    Get configuration for a specific section.
    
    Args:
        section: Configuration section name
        
    Returns:
        Dictionary with configuration parameters
    """
    config_map = {
        'app': APP_CONFIG,
        'data': DATA_CONFIG,
        'optimization': OPTIMIZATION_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'monte_carlo': MONTE_CARLO_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'gui': GUI_CONFIG,
        'logging': LOGGING_CONFIG,
        'export': EXPORT_CONFIG,
        'risk': RISK_CONFIG,
        'metrics': METRICS_CONFIG,
        'validation': VALIDATION_RULES
    }

    return config_map.get(section.lower(), {})


def validate_config():
    """Validate configuration parameters."""
    errors = []

    # Validate data config
    if DATA_CONFIG['min_data_points'] < 10:
        errors.append("Minimum data points should be at least 10")

    if not (0 < DATA_CONFIG['max_missing_data_ratio'] <= 1):
        errors.append("Max missing data ratio should be between 0 and 1")

    # Validate optimization config
    if not (0 <= OPTIMIZATION_CONFIG['default_risk_free_rate'] <= 1):
        errors.append("Risk-free rate should be between 0 and 1")

    # Validate Monte Carlo config
    if MONTE_CARLO_CONFIG['default_simulations'] <= 0:
        errors.append("Number of simulations should be positive")

    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    return True


# Validate configuration on import
validate_config()
