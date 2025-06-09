from datetime import datetime, timedelta
from typing import List, Tuple
import logging

import yfinance as yf
import pandas as pd
import numpy as np


class DataLoader:
    """
    Professional data loader for financial market data.
    Supports multiple data sources with error handling and validation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fetch_stock_data(self,
                         tickers: List[str],
                         start_date: str,
                         end_date: str,
                         interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            DataFrame with adjusted closing prices
        """
        try:
            # Clean and validate tickers
            tickers = [ticker.upper().strip() for ticker in tickers if ticker.strip()]

            if not tickers:
                raise ValueError("No valid tickers provided")

            # Fetch data
            data = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False)

            if data.empty:
                raise ValueError("No data retrieved for the specified tickers and date range")

            # Extract adjusted close prices
            if len(tickers) == 1:
                prices = data['Close'].to_frame()
                prices.columns = [tickers[0]]
            else:
                prices = data['Close']

            # Remove columns with too much missing data (>50%)
            missing_threshold = 0.5
            valid_columns = prices.columns[prices.isnull().mean() < missing_threshold]
            prices = prices[valid_columns]

            # Forward fill missing values
            prices = prices.fillna(method='ffill').dropna()

            if prices.empty:
                raise ValueError("No valid data after cleaning")

            self.logger.info(
                f"Successfully loaded data for {len(prices.columns)} assets from {start_date} to {end_date}")
            return prices

        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise

    def get_benchmark_data(self,
                           benchmark: str = '^GSPC',
                           start_date: str = None,
                           end_date: str = None) -> pd.Series:
        """
        Fetch benchmark data (default: S&P 500).
        
        Args:
            benchmark: Benchmark ticker (default: ^GSPC for S&P 500)
            start_date: Start date
            end_date: End date
            
        Returns:
            Series with benchmark prices
        """
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            benchmark_data = yf.download(benchmark, start=start_date, end=end_date, progress=False)

            if benchmark_data.empty:
                raise ValueError(f"No benchmark data retrieved for {benchmark}")

            return benchmark_data['Close']

        except Exception as e:
            self.logger.error(f"Error fetching benchmark data: {str(e)}")
            # Return a default benchmark if error occurs
            return pd.Series(index=pd.date_range(start_date, end_date), data=100.0)

    def calculate_returns(self, prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: DataFrame of price data
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame of returns
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        return returns.dropna()

    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate ticker symbols by attempting to fetch recent data.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Tuple of (valid_tickers, invalid_tickers)
        """
        valid_tickers = []
        invalid_tickers = []

        for ticker in tickers:
            try:
                # Try to fetch 5 days of data
                test_data = yf.download(ticker, period='5d', progress=False)
                if not test_data.empty:
                    valid_tickers.append(ticker.upper())
                else:
                    invalid_tickers.append(ticker)
            except:
                invalid_tickers.append(ticker)

        return valid_tickers, invalid_tickers

    def get_company_info(self, ticker: str) -> dict:
        """
        Get company information for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with company info
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0)
            }
        except:
            return {
                'name': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0
            }
