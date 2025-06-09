from typing import Dict, List
import logging
from datetime import datetime

import numpy as np
import pandas as pd


class PortfolioBacktester:
    """
    Comprehensive backtesting engine for portfolio strategies.
    """

    def __init__(self, returns_data: pd.DataFrame):
        self.returns_data = returns_data
        self.logger = logging.getLogger(__name__)

    def backtest_portfolio(self,
                           weights: Dict[str, float],
                           rebalance_freq: str = 'monthly',
                           transaction_cost: float = 0.001,
                           initial_value: float = 100000) -> Dict:
        """
        Backtest a portfolio with given weights.
        
        Args:
            weights: Portfolio weights dictionary
            rebalance_freq: Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly')
            transaction_cost: Transaction cost as percentage of trade value
            initial_value: Initial portfolio value
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Align weights with available data
            available_assets = [asset for asset in weights.keys() if asset in self.returns_data.columns]
            if not available_assets:
                raise ValueError("No assets from weights found in returns data")

            # Normalize weights for available assets
            total_weight = sum(weights[asset] for asset in available_assets)
            normalized_weights = {asset: weights[asset] / total_weight for asset in available_assets}

            # Get returns for available assets
            asset_returns = self.returns_data[available_assets].copy()

            # Calculate rebalancing dates
            rebalance_dates = self._get_rebalance_dates(asset_returns.index, rebalance_freq)

            # Initialize tracking variables
            portfolio_values = []
            portfolio_weights = []
            transaction_costs = []

            current_weights = normalized_weights.copy()
            portfolio_value = initial_value

            for date in asset_returns.index:
                # Check if rebalancing is needed
                if date in rebalance_dates:
                    # Calculate transaction costs
                    weight_changes = sum(abs(normalized_weights[asset] - current_weights.get(asset, 0))
                                         for asset in available_assets)
                    cost = portfolio_value * weight_changes * transaction_cost
                    transaction_costs.append(cost)
                    portfolio_value -= cost

                    # Rebalance to target weights
                    current_weights = normalized_weights.copy()
                else:
                    transaction_costs.append(0)

                # Calculate daily returns
                daily_returns = asset_returns.loc[date]
                portfolio_return = sum(current_weights.get(asset, 0) * daily_returns.get(asset, 0)
                                       for asset in available_assets)

                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)
                portfolio_values.append(portfolio_value)
                portfolio_weights.append(current_weights.copy())

                # Update weights based on price movements (drift)
                if not np.isnan(portfolio_return) and portfolio_return != -1:
                    for asset in available_assets:
                        if asset in daily_returns and not np.isnan(daily_returns[asset]):
                            current_weights[asset] *= (1 + daily_returns[asset]) / (1 + portfolio_return)

            # Create results DataFrame
            results_df = pd.DataFrame({
                'Date': asset_returns.index,
                'Portfolio_Value': portfolio_values,
                'Daily_Return': pd.Series(portfolio_values).pct_change(),
                'Transaction_Cost': transaction_costs
            }).set_index('Date')

            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(results_df['Portfolio_Value'], initial_value)

            return {
                'portfolio_values': results_df,
                'metrics': metrics,
                'weights_history': portfolio_weights,
                'total_transaction_costs': sum(transaction_costs)
            }

        except Exception as e:
            self.logger.error(f"Error in backtesting: {str(e)}")
            return {}

    def monte_carlo_simulation(self,
                               weights: Dict[str, float],
                               num_simulations: int = 1000,
                               time_horizon: int = 252,  # 1 year
                               initial_value: float = 100000) -> Dict:
        """
        Monte Carlo simulation for portfolio performance.
        
        Args:
            weights: Portfolio weights
            num_simulations: Number of simulation runs
            time_horizon: Time horizon in days
            initial_value: Initial portfolio value
            
        Returns:
            Dictionary with simulation results
        """
        try:
            # Get returns for assets in weights
            available_assets = [asset for asset in weights.keys() if asset in self.returns_data.columns]
            asset_returns = self.returns_data[available_assets]

            # Calculate mean returns and covariance matrix
            mean_returns = asset_returns.mean()
            cov_matrix = asset_returns.cov()

            # Normalize weights
            total_weight = sum(weights[asset] for asset in available_assets)
            normalized_weights = np.array([weights[asset] / total_weight for asset in available_assets])

            # Portfolio statistics
            portfolio_mean = np.dot(normalized_weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(normalized_weights, np.dot(cov_matrix, normalized_weights)))

            # Monte Carlo simulation
            final_values = []
            all_paths = []

            for _ in range(num_simulations):
                # Generate random returns
                random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
                portfolio_returns = np.dot(random_returns, normalized_weights)

                # Calculate cumulative portfolio value
                portfolio_path = initial_value * np.cumprod(1 + portfolio_returns)
                all_paths.append(portfolio_path)
                final_values.append(portfolio_path[-1])

            final_values = np.array(final_values)

            # Calculate statistics
            percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])

            return {
                'final_values': final_values,
                'all_paths': np.array(all_paths),
                'percentiles': {
                    '5th': percentiles[0],
                    '25th': percentiles[1],
                    '50th': percentiles[2],
                    '75th': percentiles[3],
                    '95th': percentiles[4]
                },
                'expected_value': np.mean(final_values),
                'probability_of_loss': np.mean(final_values < initial_value),
                'value_at_risk_5': percentiles[0] - initial_value,
                'expected_shortfall_5': np.mean(final_values[final_values <= percentiles[0]]) - initial_value
            }

        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {}

    def rolling_performance(self,
                            weights: Dict[str, float],
                            window_size: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            weights: Portfolio weights
            window_size: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        try:
            # Get portfolio returns
            available_assets = [asset for asset in weights.keys() if asset in self.returns_data.columns]
            asset_returns = self.returns_data[available_assets]

            # Normalize weights
            total_weight = sum(weights[asset] for asset in available_assets)
            normalized_weights = pd.Series({asset: weights[asset] / total_weight for asset in available_assets})

            # Calculate portfolio returns
            portfolio_returns = (asset_returns * normalized_weights).sum(axis=1)

            # Rolling calculations
            rolling_return = portfolio_returns.rolling(window_size).mean() * 252
            rolling_volatility = portfolio_returns.rolling(window_size).std() * np.sqrt(252)
            rolling_sharpe = rolling_return / rolling_volatility

            # Rolling maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.rolling(window_size).max()
            rolling_drawdown = (cumulative_returns - rolling_max) / rolling_max
            rolling_max_drawdown = rolling_drawdown.rolling(window_size).min()

            return pd.DataFrame({
                'Rolling_Return': rolling_return,
                'Rolling_Volatility': rolling_volatility,
                'Rolling_Sharpe': rolling_sharpe,
                'Rolling_Max_Drawdown': rolling_max_drawdown
            }).dropna()

        except Exception as e:
            self.logger.error(f"Error in rolling performance calculation: {str(e)}")
            return pd.DataFrame()

    def benchmark_comparison(self,
                             weights: Dict[str, float],
                             benchmark_returns: pd.Series,
                             initial_value: float = 100000) -> Dict:
        """
        Compare portfolio performance against a benchmark.
        
        Args:
            weights: Portfolio weights
            benchmark_returns: Benchmark return series
            initial_value: Initial portfolio value
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Get portfolio returns
            available_assets = [asset for asset in weights.keys() if asset in self.returns_data.columns]
            asset_returns = self.returns_data[available_assets]

            # Normalize weights
            total_weight = sum(weights[asset] for asset in available_assets)
            normalized_weights = pd.Series({asset: weights[asset] / total_weight for asset in available_assets})

            # Calculate portfolio returns
            portfolio_returns = (asset_returns * normalized_weights).sum(axis=1)

            # Align dates
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns.loc[common_dates]
            benchmark_returns = benchmark_returns.loc[common_dates]

            # Calculate cumulative values
            portfolio_value = initial_value * (1 + portfolio_returns).cumprod()
            benchmark_value = initial_value * (1 + benchmark_returns).cumprod()

            # Performance metrics
            portfolio_metrics = self._calculate_performance_metrics(portfolio_value, initial_value)
            benchmark_metrics = self._calculate_performance_metrics(benchmark_value, initial_value)

            # Alpha and Beta calculation
            excess_returns = portfolio_returns - benchmark_returns
            beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            alpha = portfolio_metrics['annualized_return'] - beta * benchmark_metrics['annualized_return']

            # Information ratio
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

            return {
                'portfolio_metrics': portfolio_metrics,
                'benchmark_metrics': benchmark_metrics,
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'portfolio_value': portfolio_value,
                'benchmark_value': benchmark_value
            }

        except Exception as e:
            self.logger.error(f"Error in benchmark comparison: {str(e)}")
            return {}

    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex, frequency: str) -> List[datetime]:
        """Get rebalancing dates based on frequency."""
        if frequency == 'daily':
            return date_index.tolist()
        elif frequency == 'weekly':
            return [date for date in date_index if date.weekday() == 0]  # Mondays
        elif frequency == 'monthly':
            return [date for date in date_index if date.day <= 7 and date.day == min(
                [d.day for d in date_index if d.month == date.month and d.year == date.year])]
        elif frequency == 'quarterly':
            return [date for date in date_index if date.month in [1, 4, 7, 10] and date.day <= 7]
        else:
            return [date_index[0]]  # Only rebalance at start

    def _calculate_performance_metrics(self, portfolio_values: pd.Series, initial_value: float) -> Dict:
        """Calculate comprehensive performance metrics."""
        try:
            # Returns
            returns = portfolio_values.pct_change().dropna()

            # Basic metrics
            total_return = (portfolio_values.iloc[-1] / initial_value) - 1
            num_years = len(portfolio_values) / 252
            annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0

            # Risk metrics
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            # Downside metrics
            negative_returns = returns[returns < 0]
            downside_volatility = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0

            # Maximum drawdown
            cumulative = portfolio_values / initial_value
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Additional metrics
            win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0

            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'downside_volatility': downside_volatility
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
