from typing import Dict, Tuple, Optional
import logging

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.objective_functions import L2_reg
import cvxpy as cp


class PortfolioOptimizer:
    """
    Advanced portfolio optimization engine supporting multiple strategies.
    """

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.assets = returns.columns.tolist()
        self.n_assets = len(self.assets)
        self.logger = logging.getLogger(__name__)

        # Calculate expected returns and covariance matrix
        self.mu = expected_returns.mean_historical_return(self.returns.dropna())
        self.S = risk_models.sample_cov(self.returns.dropna())

    def mean_variance_optimization(self,
                                   target_return: Optional[float] = None,
                                   target_volatility: Optional[float] = None,
                                   l2_gamma: float = 0.1) -> Dict:
        """
        Perform mean-variance optimization using PyPortfolioOpt.
        
        Args:
            target_return: Target portfolio return (annualized)
            target_volatility: Target portfolio volatility (annualized)
            l2_gamma: L2 regularization parameter
            
        Returns:
            Dictionary with optimization results
        """
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_objective(L2_reg, gamma=l2_gamma)

            if target_return is not None:
                weights = ef.efficient_return(target_return)
            elif target_volatility is not None:
                weights = ef.efficient_risk(target_volatility)
            else:
                # Default to max Sharpe ratio
                weights = ef.max_sharpe()

            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance()

            return {
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'method': 'Mean-Variance Optimization'
            }

        except Exception as e:
            self.logger.error(f"Error in mean-variance optimization: {str(e)}")
            return self._equal_weight_fallback()

    def max_sharpe_optimization(self, risk_free_rate: float = 0.02) -> Dict:
        """
        Optimize for maximum Sharpe ratio.
        
        Args:
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary with optimization results
        """
        try:
            ef = EfficientFrontier(self.mu, self.S)
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(risk_free_rate=risk_free_rate)

            return {
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'method': 'Maximum Sharpe Ratio'
            }

        except Exception as e:
            self.logger.error(f"Error in max Sharpe optimization: {str(e)}")
            return self._equal_weight_fallback()

    def min_volatility_optimization(self) -> Dict:
        """
        Optimize for minimum volatility.
        
        Returns:
            Dictionary with optimization results
        """
        try:
            ef = EfficientFrontier(self.mu, self.S)
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance()

            return {
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'method': 'Minimum Volatility'
            }

        except Exception as e:
            self.logger.error(f"Error in min volatility optimization: {str(e)}")
            return self._equal_weight_fallback()

    def equal_weight_portfolio(self) -> Dict:
        """
        Create equal-weight portfolio.
        
        Returns:
            Dictionary with portfolio results
        """
        weights = {asset: 1 / self.n_assets for asset in self.assets}

        # Calculate performance metrics
        weight_array = np.array(list(weights.values()))
        expected_return = np.dot(weight_array, self.mu)
        volatility = np.sqrt(np.dot(weight_array, np.dot(self.S, weight_array)))
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0

        return {
            'weights': weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'Equal Weight'
        }

    def risk_parity_optimization(self) -> Dict:
        """
        Risk parity optimization using CVXPY.
        
        Returns:
            Dictionary with optimization results
        """
        try:
            # Risk parity optimization
            w = cp.Variable(self.n_assets)

            # Objective: minimize sum of squared risk contributions
            risk_contrib = cp.multiply(w, self.S @ w)
            objective = cp.Minimize(cp.sum_squares(risk_contrib - cp.sum(risk_contrib) / self.n_assets))

            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Weights sum to 1
                w >= 0,  # Long-only
                w <= 1  # No single asset > 100%
            ]

            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if w.value is not None:
                weights = {self.assets[i]: w.value[i] for i in range(self.n_assets)}

                # Calculate performance metrics
                weight_array = np.array(list(weights.values()))
                expected_return = np.dot(weight_array, self.mu)
                volatility = np.sqrt(np.dot(weight_array, np.dot(self.S, weight_array)))
                sharpe_ratio = expected_return / volatility if volatility > 0 else 0

                return {
                    'weights': weights,
                    'expected_return': expected_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'method': 'Risk Parity'
                }
            else:
                return self._equal_weight_fallback()

        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {str(e)}")
            return self._equal_weight_fallback()

    def efficient_frontier(self, num_portfolios: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate efficient frontier data.
        
        Args:
            num_portfolios: Number of portfolios to generate
            
        Returns:
            Tuple of (returns, volatilities, sharpe_ratios)
        """
        try:
            # Generate target returns
            min_ret = self.mu.min()
            max_ret = self.mu.max()
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)

            returns = []
            volatilities = []
            sharpe_ratios = []

            for target_ret in target_returns:
                try:
                    ef = EfficientFrontier(self.mu, self.S)
                    weights = ef.efficient_return(target_ret)
                    performance = ef.portfolio_performance()

                    returns.append(performance[0])
                    volatilities.append(performance[1])
                    sharpe_ratios.append(performance[2])
                except:
                    continue

            return np.array(returns), np.array(volatilities), np.array(sharpe_ratios)

        except Exception as e:
            self.logger.error(f"Error generating efficient frontier: {str(e)}")
            return np.array([]), np.array([]), np.array([])

    def black_litterman_optimization(self,
                                     views: Dict[str, float] = None,
                                     picking_matrix: np.ndarray = None,
                                     omega: np.ndarray = None) -> Dict:
        """
        Black-Litterman optimization (simplified implementation).
        
        Args:
            views: Dictionary of expected returns views
            picking_matrix: Matrix identifying assets in views
            omega: Uncertainty matrix for views
            
        Returns:
            Dictionary with optimization results
        """
        try:
            from pypfopt import black_litterman

            # Market cap weights (simplified - use equal weights as proxy)
            market_caps = [1e9] * self.n_assets  # Placeholder market caps

            # Black-Litterman expected returns
            if views is not None and picking_matrix is not None:
                if omega is None:
                    omega = np.eye(len(views)) * 0.01  # Default uncertainty

                bl_mu = black_litterman.black_litterman(
                    self.S, pi=self.mu, Q=list(views.values()),
                    P=picking_matrix, omega=omega
                )
            else:
                bl_mu = self.mu

            # Optimize with Black-Litterman returns
            ef = EfficientFrontier(bl_mu, self.S)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance()

            return {
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'method': 'Black-Litterman'
            }

        except Exception as e:
            self.logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return self.max_sharpe_optimization()

    def _equal_weight_fallback(self) -> Dict:
        """
        Fallback to equal weight portfolio if optimization fails.
        
        Returns:
            Equal weight portfolio results
        """
        self.logger.warning("Optimization failed, falling back to equal weight portfolio")
        return self.equal_weight_portfolio()

    def add_constraints(self,
                        ef: EfficientFrontier,
                        max_weight: float = None,
                        min_weight: float = None,
                        sector_constraints: Dict = None) -> EfficientFrontier:
        """
        Add custom constraints to the optimizer.
        
        Args:
            ef: EfficientFrontier object
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            sector_constraints: Dictionary of sector weight constraints
            
        Returns:
            Modified EfficientFrontier object
        """
        if max_weight is not None:
            ef.add_constraint(lambda w: w <= max_weight)

        if min_weight is not None:
            ef.add_constraint(lambda w: w >= min_weight)

        # Additional sector constraints could be implemented here

        return ef
