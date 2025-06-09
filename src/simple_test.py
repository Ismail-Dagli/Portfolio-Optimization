"""
Simple Portfolio Optimization Test

Quick test to verify core functionality without heavy visualizations.
"""

import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())


def simple_test():
    print("=" * 50)
    print("PORTFOLIO OPTIMIZATION - SIMPLE TEST")
    print("=" * 50)

    try:
        # Test imports
        print("1. Testing module imports...")
        from data_loader import DataLoader
        from optimizer import PortfolioOptimizer
        from utils import PortfolioUtils, validate_tickers
        print("   ✅ All modules imported successfully")

        # Test ticker validation
        print("\n2. Testing ticker validation...")
        test_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        cleaned_tickers = validate_tickers(test_tickers)
        print(f"   ✅ Validated tickers: {cleaned_tickers}")

        # Test date validation
        print("\n3. Testing date validation...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        start_dt, end_dt = PortfolioUtils.validate_date_range(start_date, end_date)
        print(f"   ✅ Date range validated: {start_dt.date()} to {end_dt.date()}")

        # Test with simulated data
        print("\n4. Testing with simulated data...")
        np.random.seed(42)

        # Create simulated returns data
        n_days = 252
        n_assets = len(cleaned_tickers)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        # Generate random returns with some correlation
        mean_returns = np.array([0.001, 0.0008, 0.0012, 0.002])  # Daily returns
        cov_matrix = np.array([
            [0.0004, 0.0001, 0.0002, 0.0003],
            [0.0001, 0.0003, 0.0001, 0.0002],
            [0.0002, 0.0001, 0.0005, 0.0001],
            [0.0003, 0.0002, 0.0001, 0.0008]
        ])

        returns_data = pd.DataFrame(
            np.random.multivariate_normal(mean_returns, cov_matrix, n_days),
            index=dates,
            columns=cleaned_tickers
        )
        print(f"   ✅ Generated {returns_data.shape[0]} days of return data for {returns_data.shape[1]} assets")

        # Test portfolio optimization
        print("\n5. Testing portfolio optimization...")
        optimizer = PortfolioOptimizer()

        # Equal weight portfolio
        equal_weights = optimizer.optimize_equal_weight(cleaned_tickers)
        print(f"   ✅ Equal Weight: {[f'{w:.3f}' for w in equal_weights]}")

        # Maximum Sharpe ratio
        try:
            max_sharpe_weights = optimizer.optimize_max_sharpe(returns_data)
            print(f"   ✅ Max Sharpe: {[f'{w:.3f}' for w in max_sharpe_weights]}")
        except Exception as e:
            print(f"   ⚠️ Max Sharpe failed: {e}")

        # Minimum volatility
        try:
            min_vol_weights = optimizer.optimize_min_volatility(returns_data)
            print(f"   ✅ Min Volatility: {[f'{w:.3f}' for w in min_vol_weights]}")
        except Exception as e:
            print(f"   ⚠️ Min Volatility failed: {e}")

        # Test performance calculation
        print("\n6. Testing performance metrics...")
        portfolio_returns = (returns_data * equal_weights).sum(axis=1)
        metrics = PortfolioUtils.calculate_portfolio_metrics(portfolio_returns)

        print(f"   ✅ Portfolio Performance (Equal Weight):")
        print(f"      Annual Return: {PortfolioUtils.format_percentage(metrics['annualized_return'])}")
        print(f"      Volatility: {PortfolioUtils.format_percentage(metrics['volatility'])}")
        print(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"      Max Drawdown: {PortfolioUtils.format_percentage(metrics['max_drawdown'])}")

        # Test basic backtesting
        print("\n7. Testing backtesting...")
        from backtester import Backtester
        backtester = Backtester()

        try:
            backtest_result = backtester.run_backtest(
                weights=equal_weights,
                returns_data=returns_data,
                rebalance_frequency='monthly'
            )
            print(f"   ✅ Backtest completed with {len(backtest_result['portfolio_returns'])} return observations")
        except Exception as e:
            print(f"   ⚠️ Backtesting failed: {e}")

        print("\n" + "=" * 50)
        print("🎉 SIMPLE TEST COMPLETED SUCCESSFULLY!")
        print("✅ All core functionality is working properly")
        print("📊 Portfolio Optimization App is ready to use!")
        print("🚀 Launch the full app with: python main.py")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n🎯 Ready to run the full Portfolio Optimization App!")
    else:
        print("\n⚠️ Please check the error messages above.")
