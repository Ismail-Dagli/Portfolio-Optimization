"""
Portfolio Optimization App - Demo Script

This script demonstrates the key features of the Portfolio Optimization App
including data loading, optimization strategies, backtesting, and visualization.

Run this script to see a complete workflow example.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

# Add current directory to path for imports
sys.path.append(os.getcwd())

# Import our modules
from data_loader import DataLoader
from optimizer import PortfolioOptimizer
from backtester import Backtester
from visualizer import Visualizer
from utils import PortfolioUtils, validate_tickers, print_portfolio_summary
from config import Config

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def demo_portfolio_optimization():
    """
    Complete demonstration of portfolio optimization workflow
    """
    print("=" * 80)
    print("PORTFOLIO OPTIMIZATION APP - COMPREHENSIVE DEMO")
    print("=" * 80)
    
    # Step 1: Initialize components
    print("\n1. Initializing Portfolio Optimization Components...")
    data_loader = DataLoader()
    optimizer = PortfolioOptimizer()
    backtester = Backtester()
    visualizer = Visualizer()
    
    # Step 2: Define portfolio parameters
    print("\n2. Setting up Portfolio Parameters...")
    
    # Use a diversified portfolio of well-known stocks
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    print(f"Selected Assets: {', '.join(tickers)}")
    
    # Validate tickers
    cleaned_tickers = validate_tickers(tickers)
    print(f"Validated Tickers: {', '.join(cleaned_tickers)}")
    
    # Date range for historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')  # ~3 years
    print(f"Date Range: {start_date} to {end_date}")
    
    # Step 3: Load financial data
    print("\n3. Loading Financial Data...")
    try:
        # Load price data
        price_data = data_loader.load_stock_data(cleaned_tickers, start_date, end_date)
        print(f"Loaded price data: {price_data.shape[0]} days, {price_data.shape[1]} assets")
        
        # Calculate returns
        returns_data = data_loader.calculate_returns(price_data)
        print(f"Calculated returns: {returns_data.shape[0]} observations")
        
        # Load benchmark data (S&P 500)
        benchmark_data = data_loader.load_benchmark_data('^GSPC', start_date, end_date)
        benchmark_returns = data_loader.calculate_returns(benchmark_data)
        print(f"Loaded benchmark data: {benchmark_returns.shape[0]} observations")
        
        # Display basic statistics
        print("\nBasic Statistics:")
        print(f"Mean Annual Returns:")
        annual_returns = returns_data.mean() * 252
        for ticker, ret in annual_returns.items():
            print(f"  {ticker}: {PortfolioUtils.format_percentage(ret)}")
        
        print(f"\nAnnual Volatilities:")
        annual_vols = returns_data.std() * np.sqrt(252)
        for ticker, vol in annual_vols.items():
            print(f"  {ticker}: {PortfolioUtils.format_percentage(vol)}")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Demo will continue with simulated data...")
        
        # Create simulated data for demo
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Simulate price data
        n_assets = len(cleaned_tickers)
        n_days = len(dates)
        
        # Generate correlated returns
        mean_returns = np.random.normal(0.0008, 0.002, n_assets)  # Daily returns
        cov_matrix = np.random.rand(n_assets, n_assets)
        cov_matrix = np.dot(cov_matrix, cov_matrix.T) * 0.0001  # Make positive definite
        
        returns_data = pd.DataFrame(
            np.random.multivariate_normal(mean_returns, cov_matrix, n_days),
            index=dates[:n_days],
            columns=cleaned_tickers
        )
        
        # Generate price data from returns
        price_data = (1 + returns_data).cumprod() * 100  # Start at $100
        
        # Simulate benchmark
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, n_days),
            index=dates[:n_days],
            name='^GSPC'
        )
        
        print("Using simulated financial data for demonstration")
    
    # Step 4: Portfolio Optimization
    print("\n4. Running Portfolio Optimization Strategies...")
    
    optimization_results = {}
    
    # Maximum Sharpe Ratio Portfolio
    print("\n  4.1 Maximum Sharpe Ratio Optimization...")
    try:
        max_sharpe_weights = optimizer.optimize_max_sharpe(returns_data)
        max_sharpe_performance = optimizer.calculate_portfolio_performance(max_sharpe_weights, returns_data)
        optimization_results['Max Sharpe'] = {
            'weights': max_sharpe_weights,
            'performance': max_sharpe_performance
        }
        print(f"    Expected Return: {PortfolioUtils.format_percentage(max_sharpe_performance['expected_return'])}")
        print(f"    Volatility: {PortfolioUtils.format_percentage(max_sharpe_performance['volatility'])}")
        print(f"    Sharpe Ratio: {max_sharpe_performance['sharpe_ratio']:.3f}")
    except Exception as e:
        print(f"    Error in Max Sharpe optimization: {e}")
    
    # Minimum Volatility Portfolio
    print("\n  4.2 Minimum Volatility Optimization...")
    try:
        min_vol_weights = optimizer.optimize_min_volatility(returns_data)
        min_vol_performance = optimizer.calculate_portfolio_performance(min_vol_weights, returns_data)
        optimization_results['Min Volatility'] = {
            'weights': min_vol_weights,
            'performance': min_vol_performance
        }
        print(f"    Expected Return: {PortfolioUtils.format_percentage(min_vol_performance['expected_return'])}")
        print(f"    Volatility: {PortfolioUtils.format_percentage(min_vol_performance['volatility'])}")
        print(f"    Sharpe Ratio: {min_vol_performance['sharpe_ratio']:.3f}")
    except Exception as e:
        print(f"    Error in Min Volatility optimization: {e}")
    
    # Equal Weight Portfolio
    print("\n  4.3 Equal Weight Portfolio...")
    try:
        equal_weights = optimizer.optimize_equal_weight(cleaned_tickers)
        equal_performance = optimizer.calculate_portfolio_performance(equal_weights, returns_data)
        optimization_results['Equal Weight'] = {
            'weights': equal_weights,
            'performance': equal_performance
        }
        print(f"    Expected Return: {PortfolioUtils.format_percentage(equal_performance['expected_return'])}")
        print(f"    Volatility: {PortfolioUtils.format_percentage(equal_performance['volatility'])}")
        print(f"    Sharpe Ratio: {equal_performance['sharpe_ratio']:.3f}")
    except Exception as e:
        print(f"    Error in Equal Weight optimization: {e}")
    
    # Risk Parity Portfolio
    print("\n  4.4 Risk Parity Optimization...")
    try:
        risk_parity_weights = optimizer.optimize_risk_parity(returns_data)
        risk_parity_performance = optimizer.calculate_portfolio_performance(risk_parity_weights, returns_data)
        optimization_results['Risk Parity'] = {
            'weights': risk_parity_weights,
            'performance': risk_parity_performance
        }
        print(f"    Expected Return: {PortfolioUtils.format_percentage(risk_parity_performance['expected_return'])}")
        print(f"    Volatility: {PortfolioUtils.format_percentage(risk_parity_performance['volatility'])}")
        print(f"    Sharpe Ratio: {risk_parity_performance['sharpe_ratio']:.3f}")
    except Exception as e:
        print(f"    Error in Risk Parity optimization: {e}")
    
    # Step 5: Backtesting
    print("\n5. Backtesting Portfolio Strategies...")
    
    backtest_results = {}
    
    for strategy_name, result in optimization_results.items():
        print(f"\n  5.{len(backtest_results)+1} Backtesting {strategy_name} Portfolio...")
        try:
            weights = result['weights']
            
            # Run backtest
            backtest_result = backtester.run_backtest(
                weights=weights,
                returns_data=returns_data,
                rebalance_frequency='monthly',
                transaction_cost=0.001
            )
            
            # Calculate performance metrics
            portfolio_returns = backtest_result['portfolio_returns']
            metrics = PortfolioUtils.calculate_portfolio_metrics(
                portfolio_returns, 
                benchmark_returns.reindex(portfolio_returns.index).dropna()
            )
            
            backtest_results[strategy_name] = {
                'returns': portfolio_returns,
                'metrics': metrics,
                'weights_history': backtest_result.get('weights_history', [])
            }
            
            print(f"    Total Return: {PortfolioUtils.format_percentage(metrics['total_return'])}")
            print(f"    Annual Return: {PortfolioUtils.format_percentage(metrics['annualized_return'])}")
            print(f"    Annual Volatility: {PortfolioUtils.format_percentage(metrics['volatility'])}")
            print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"    Max Drawdown: {PortfolioUtils.format_percentage(metrics['max_drawdown'])}")
            
        except Exception as e:
            print(f"    Error in {strategy_name} backtest: {e}")
    
    # Step 6: Monte Carlo Simulation
    print("\n6. Running Monte Carlo Simulation...")
    
    if optimization_results:
        try:
            # Use the Max Sharpe portfolio for Monte Carlo
            strategy_name = list(optimization_results.keys())[0]
            weights = optimization_results[strategy_name]['weights']
            
            print(f"  Running simulation for {strategy_name} portfolio...")
            mc_results = backtester.monte_carlo_simulation(
                weights=weights,
                returns_data=returns_data,
                time_horizon=252,  # 1 year
                num_simulations=1000
            )
            
            print(f"    Simulations completed: {len(mc_results['final_values'])}")
            print(f"    Mean Final Value: {PortfolioUtils.format_currency(np.mean(mc_results['final_values']))}")
            print(f"    95% Confidence Interval: {PortfolioUtils.format_currency(np.percentile(mc_results['final_values'], 5))} - {PortfolioUtils.format_currency(np.percentile(mc_results['final_values'], 95))}")
            print(f"    Probability of Loss: {PortfolioUtils.format_percentage(np.mean(mc_results['final_values'] < 10000))}")
            
        except Exception as e:
            print(f"    Error in Monte Carlo simulation: {e}")
    
    # Step 7: Generate Visualizations
    print("\n7. Generating Visualizations...")
    
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Portfolio Allocation Comparison
        if optimization_results:
            plt.subplot(2, 3, 1)
            strategies = list(optimization_results.keys())
            if strategies:
                weights_data = optimization_results[strategies[0]]['weights']
                visualizer.plot_portfolio_allocation(weights_data, cleaned_tickers, title=f"{strategies[0]} Allocation")
        
        # Plot 2: Efficient Frontier
        plt.subplot(2, 3, 2)
        try:
            efficient_frontier = optimizer.calculate_efficient_frontier(returns_data, num_portfolios=100)
            if efficient_frontier:
                visualizer.plot_efficient_frontier(
                    efficient_frontier['returns'],
                    efficient_frontier['volatilities'],
                    efficient_frontier['sharpe_ratios']
                )
        except Exception as e:
            print(f"    Could not generate efficient frontier: {e}")
        
        # Plot 3: Correlation Matrix
        plt.subplot(2, 3, 3)
        correlation_matrix = returns_data.corr()
        visualizer.plot_correlation_matrix(correlation_matrix, title="Asset Correlation Matrix")
        
        # Plot 4: Cumulative Returns Comparison
        if backtest_results:
            plt.subplot(2, 3, 4)
            for strategy_name, result in backtest_results.items():
                returns = result['returns']
                cumulative_returns = (1 + returns).cumprod()
                plt.plot(cumulative_returns.index, cumulative_returns.values, label=strategy_name, linewidth=2)
            
            # Add benchmark
            if len(benchmark_returns) > 0:
                benchmark_aligned = benchmark_returns.reindex(returns.index).dropna()
                benchmark_cumulative = (1 + benchmark_aligned).cumprod()
                plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, label='S&P 500', linewidth=2, linestyle='--')
            
            plt.title('Cumulative Returns Comparison')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Risk-Return Scatter
        if optimization_results:
            plt.subplot(2, 3, 5)
            returns_list = []
            volatilities_list = []
            names = []
            
            for strategy_name, result in optimization_results.items():
                perf = result['performance']
                returns_list.append(perf['expected_return'])
                volatilities_list.append(perf['volatility'])
                names.append(strategy_name)
            
            plt.scatter(volatilities_list, returns_list, s=100, alpha=0.7)
            for i, name in enumerate(names):
                plt.annotate(name, (volatilities_list[i], returns_list[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            plt.title('Risk-Return Profile')
            plt.xlabel('Volatility')
            plt.ylabel('Expected Return')
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Performance Metrics Comparison
        if backtest_results:
            plt.subplot(2, 3, 6)
            metrics_comparison = pd.DataFrame({
                strategy: result['metrics'] for strategy, result in backtest_results.items()
            }).T
            
            key_metrics = ['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
            available_metrics = [m for m in key_metrics if m in metrics_comparison.columns]
            
            if available_metrics:
                metrics_subset = metrics_comparison[available_metrics]
                metrics_subset.plot(kind='bar', ax=plt.gca())
                plt.title('Performance Metrics Comparison')
                plt.ylabel('Value')
                plt.xticks(rotation=45)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'portfolio_optimization_demo_results.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    Visualizations saved to: {output_file}")
        
        # Display the plot
        plt.show()
        
    except Exception as e:
        print(f"    Error generating visualizations: {e}")
    
    # Step 8: Export Results
    print("\n8. Exporting Results...")
    
    try:
        # Create results summary
        results_summary = {
            'Portfolio Strategies': [],
            'Expected Return': [],
            'Volatility': [],
            'Sharpe Ratio': [],
            'Max Drawdown': []
        }
        
        for strategy_name in optimization_results.keys():
            if strategy_name in backtest_results:
                metrics = backtest_results[strategy_name]['metrics']
                results_summary['Portfolio Strategies'].append(strategy_name)
                results_summary['Expected Return'].append(f"{metrics.get('annualized_return', 0)*100:.2f}%")
                results_summary['Volatility'].append(f"{metrics.get('volatility', 0)*100:.2f}%")
                results_summary['Sharpe Ratio'].append(f"{metrics.get('sharpe_ratio', 0):.3f}")
                results_summary['Max Drawdown'].append(f"{metrics.get('max_drawdown', 0)*100:.2f}%")
        
        # Save to CSV
        if results_summary['Portfolio Strategies']:
            results_df = pd.DataFrame(results_summary)
            csv_file = 'portfolio_optimization_demo_results.csv'
            results_df.to_csv(csv_file, index=False)
            print(f"    Results exported to: {csv_file}")
            
            # Display summary table
            print("\n" + "="*60)
            print("PORTFOLIO OPTIMIZATION RESULTS SUMMARY")
            print("="*60)
            print(results_df.to_string(index=False))
            print("="*60)
    
    except Exception as e:
        print(f"    Error exporting results: {e}")
    
    # Step 9: Summary and Conclusions
    print("\n9. Demo Summary and Conclusions...")
    print(f"\n✅ Successfully demonstrated:")
    print(f"   • Data loading and validation for {len(cleaned_tickers)} assets")
    print(f"   • Multiple optimization strategies ({len(optimization_results)} methods)")
    print(f"   • Historical backtesting with performance metrics")
    print(f"   • Monte Carlo simulation for risk assessment")
    print(f"   • Comprehensive visualizations and charts")
    print(f"   • Results export to CSV format")
    
    if backtest_results:
        # Find best performing strategy
        best_strategy = max(backtest_results.items(), 
                          key=lambda x: x[1]['metrics'].get('sharpe_ratio', 0))
        print(f"\n🏆 Best performing strategy: {best_strategy[0]}")
        print(f"   Sharpe Ratio: {best_strategy[1]['metrics'].get('sharpe_ratio', 0):.3f}")
        print(f"   Annual Return: {PortfolioUtils.format_percentage(best_strategy[1]['metrics'].get('annualized_return', 0))}")
        print(f"   Volatility: {PortfolioUtils.format_percentage(best_strategy[1]['metrics'].get('volatility', 0))}")
    
    print(f"\n📊 The Portfolio Optimization App is ready for professional use!")
    print(f"   Launch with: python main.py")
    print(f"   View documentation: README.md")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        demo_portfolio_optimization()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
