"""
Portfolio Optimization App - Utility Functions Module

This module contains utility functions used across the portfolio optimization application.
Includes data validation, formatting, statistical calculations, and helper functions.

Author: Portfolio Optimization App
Version: 1.0
"""

from typing import List, Dict, Union, Tuple, Optional
import re
import logging
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class PortfolioUtils:
    """Utility class containing static methods for portfolio optimization"""

    @staticmethod
    def validate_ticker_symbol(ticker: str) -> bool:
        """
        Validate ticker symbol format
        
        Args:
            ticker (str): Ticker symbol to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(ticker, str):
            return False

        # Basic ticker validation - alphanumeric characters and dots/hyphens
        pattern = r'^[A-Z0-9.-]{1,10}$'
        return bool(re.match(pattern, ticker.upper()))

    @staticmethod
    def clean_ticker_list(tickers: List[str]) -> List[str]:
        """
        Clean and validate a list of ticker symbols
        
        Args:
            tickers (List[str]): List of ticker symbols
            
        Returns:
            List[str]: Cleaned list of valid tickers
        """
        if not tickers:
            raise ValidationError("Ticker list cannot be empty")

        cleaned_tickers = []
        for ticker in tickers:
            ticker = ticker.strip().upper()
            if PortfolioUtils.validate_ticker_symbol(ticker):
                if ticker not in cleaned_tickers:  # Remove duplicates
                    cleaned_tickers.append(ticker)
            else:
                logger.warning(f"Invalid ticker symbol: {ticker}")

        if not cleaned_tickers:
            raise ValidationError("No valid ticker symbols found")

        return cleaned_tickers

    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
        """
        Validate and parse date range
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Tuple[datetime, datetime]: Parsed start and end dates
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            raise ValidationError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")

        if start_dt >= end_dt:
            raise ValidationError("Start date must be before end date")

        if end_dt > datetime.now():
            raise ValidationError("End date cannot be in the future")

        # Check if date range is reasonable (at least 30 days)
        if (end_dt - start_dt).days < 30:
            logger.warning("Date range is less than 30 days. Results may be unreliable.")

        return start_dt, end_dt

    @staticmethod
    def validate_weights(weights: np.ndarray) -> np.ndarray:
        """
        Validate portfolio weights
        
        Args:
            weights (np.ndarray): Portfolio weights
            
        Returns:
            np.ndarray: Validated weights
        """
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)

        if len(weights) == 0:
            raise ValidationError("Weights array cannot be empty")

        if np.any(weights < 0):
            raise ValidationError("Weights cannot be negative")

        if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
            logger.warning(f"Weights sum to {np.sum(weights):.6f}, not 1.0. Normalizing.")
            weights = weights / np.sum(weights)

        return weights

    @staticmethod
    def calculate_portfolio_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                                    risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio performance metrics
        
        Args:
            returns (pd.Series): Portfolio returns
            benchmark_returns (Optional[pd.Series]): Benchmark returns for comparison
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if returns.empty:
            raise ValidationError("Returns series cannot be empty")

        # Remove any infinite or NaN values
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) < 2:
            raise ValidationError("Need at least 2 return observations")

        metrics = {}

        # Basic metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['sharpe_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['volatility'] if metrics[
                                                                                                                 'volatility'] > 0 else 0

        # Risk metrics
        metrics['max_drawdown'] = PortfolioUtils.calculate_max_drawdown(returns)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()

        # Downside metrics
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics['downside_deviation'] = negative_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics[
                'downside_deviation'] if metrics['downside_deviation'] > 0 else 0
        else:
            metrics['downside_deviation'] = 0
            metrics['sortino_ratio'] = np.inf

        # Benchmark comparison
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.reindex(returns.index).dropna()
            if len(benchmark_returns) > 1:
                metrics['beta'] = PortfolioUtils.calculate_beta(returns, benchmark_returns)
                metrics['alpha'] = metrics['annualized_return'] - (
                            risk_free_rate + metrics['beta'] * (benchmark_returns.mean() * 252 - risk_free_rate))
                metrics['information_ratio'] = PortfolioUtils.calculate_information_ratio(returns, benchmark_returns)
                metrics['tracking_error'] = (returns - benchmark_returns).std() * np.sqrt(252)

        return metrics

    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            returns (pd.Series): Return series
            
        Returns:
            float: Maximum drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate portfolio beta relative to benchmark
        
        Args:
            portfolio_returns (pd.Series): Portfolio returns
            benchmark_returns (pd.Series): Benchmark returns
            
        Returns:
            float: Beta coefficient
        """
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 1.0

        covariance = np.cov(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])[0, 1]
        benchmark_variance = np.var(aligned_data.iloc[:, 1])

        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0

    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio
        
        Args:
            portfolio_returns (pd.Series): Portfolio returns
            benchmark_returns (pd.Series): Benchmark returns
            
        Returns:
            float: Information ratio
        """
        excess_returns = portfolio_returns - benchmark_returns
        excess_returns = excess_returns.dropna()

        if len(excess_returns) == 0:
            return 0.0

        tracking_error = excess_returns.std()
        return excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0

    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """
        Format a decimal value as percentage
        
        Args:
            value (float): Value to format
            decimals (int): Number of decimal places
            
        Returns:
            str: Formatted percentage string
        """
        return f"{value * 100:.{decimals}f}%"

    @staticmethod
    def format_currency(value: float, currency: str = "$") -> str:
        """
        Format value as currency
        
        Args:
            value (float): Value to format
            currency (str): Currency symbol
            
        Returns:
            str: Formatted currency string
        """
        return f"{currency}{value:,.2f}"

    @staticmethod
    def annualize_return(returns: Union[float, np.ndarray, pd.Series], periods: int = 252) -> Union[float, np.ndarray]:
        """
        Annualize returns
        
        Args:
            returns: Returns to annualize
            periods (int): Number of periods per year (252 for daily)
            
        Returns:
            Annualized returns
        """
        return (1 + returns) ** periods - 1

    @staticmethod
    def annualize_volatility(returns: Union[np.ndarray, pd.Series], periods: int = 252) -> float:
        """
        Annualize volatility
        
        Args:
            returns: Returns series
            periods (int): Number of periods per year
            
        Returns:
            float: Annualized volatility
        """
        return np.std(returns) * np.sqrt(periods)

    @staticmethod
    def create_summary_table(metrics: Dict[str, float]) -> pd.DataFrame:
        """
        Create a formatted summary table of metrics
        
        Args:
            metrics (Dict[str, float]): Dictionary of metrics
            
        Returns:
            pd.DataFrame: Formatted summary table
        """
        formatted_metrics = {}

        percentage_metrics = [
            'total_return', 'annualized_return', 'volatility', 'max_drawdown',
            'var_95', 'cvar_95', 'downside_deviation', 'tracking_error', 'alpha'
        ]

        ratio_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'information_ratio', 'beta',
            'skewness', 'kurtosis'
        ]

        for key, value in metrics.items():
            if key in percentage_metrics:
                formatted_metrics[key.replace('_', ' ').title()] = PortfolioUtils.format_percentage(value)
            elif key in ratio_metrics:
                formatted_metrics[key.replace('_', ' ').title()] = f"{value:.3f}"
            else:
                formatted_metrics[key.replace('_', ' ').title()] = f"{value:.3f}"

        return pd.DataFrame(list(formatted_metrics.items()), columns=['Metric', 'Value'])

    @staticmethod
    def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safely divide two numbers, returning default if denominator is zero
        
        Args:
            numerator (float): Numerator
            denominator (float): Denominator
            default (float): Default value if division by zero
            
        Returns:
            float: Result of division or default value
        """
        return numerator / denominator if abs(denominator) > 1e-10 else default

    @staticmethod
    def handle_missing_data(data: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """
        Handle missing data in DataFrame
        
        Args:
            data (pd.DataFrame): Data with potential missing values
            method (str): Method to handle missing data ('forward', 'drop', 'interpolate')
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        if method == 'forward':
            return data.fillna(method='ffill').dropna()
        elif method == 'drop':
            return data.dropna()
        elif method == 'interpolate':
            return data.interpolate().dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def log_operation(operation: str, details: str = "") -> None:
        """
        Log an operation with timestamp
        
        Args:
            operation (str): Operation description
            details (str): Additional details
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[{timestamp}] {operation} {details}")

    @staticmethod
    def suppress_warnings(func):
        """
        Decorator to suppress warnings for a function
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """

        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return func(*args, **kwargs)

        return wrapper


# Convenience functions for common operations
def validate_tickers(tickers: List[str]) -> List[str]:
    """Convenience function for ticker validation"""
    return PortfolioUtils.clean_ticker_list(tickers)


def calculate_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """
    Calculate returns from price data
    
    Args:
        prices (pd.DataFrame): Price data
        method (str): 'simple' or 'log' returns
        
    Returns:
        pd.DataFrame: Returns data
    """
    if method == 'simple':
        return prices.pct_change().dropna()
    elif method == 'log':
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'simple' or 'log'")


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix of returns
    
    Args:
        returns (pd.DataFrame): Returns data
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    return returns.corr()


def calculate_covariance_matrix(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Calculate covariance matrix of returns
    
    Args:
        returns (pd.DataFrame): Returns data
        annualize (bool): Whether to annualize the covariance matrix
        
    Returns:
        pd.DataFrame: Covariance matrix
    """
    cov_matrix = returns.cov()
    if annualize:
        cov_matrix *= 252  # Annualize assuming daily data
    return cov_matrix


def print_portfolio_summary(weights: np.ndarray, tickers: List[str], metrics: Dict[str, float]) -> None:
    """
    Print a formatted portfolio summary
    
    Args:
        weights (np.ndarray): Portfolio weights
        tickers (List[str]): Asset tickers
        metrics (Dict[str, float]): Performance metrics
    """
    print("=" * 50)
    print("PORTFOLIO SUMMARY")
    print("=" * 50)

    print("\nPortfolio Allocation:")
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker}: {PortfolioUtils.format_percentage(weight)}")

    print(f"\nPerformance Metrics:")
    summary_df = PortfolioUtils.create_summary_table(metrics)
    print(summary_df.to_string(index=False))
    print("=" * 50)


if __name__ == "__main__":
    # Example usage and testing
    print("Portfolio Optimization Utilities Module")
    print("Testing utility functions...")

    # Test ticker validation
    test_tickers = ["AAPL", "googl", "MSFT", "", "invalid!", "TSLA"]
    cleaned = validate_tickers(test_tickers)
    print(f"Cleaned tickers: {cleaned}")

    # Test date validation
    try:
        start, end = PortfolioUtils.validate_date_range("2020-01-01", "2021-01-01")
        print(f"Date range: {start} to {end}")
    except ValidationError as e:
        print(f"Date validation error: {e}")

    print("All tests completed successfully!")
