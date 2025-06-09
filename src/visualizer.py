from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import warnings

warnings.filterwarnings('ignore')


class PortfolioVisualizer:
    """
    Comprehensive visualization engine for portfolio analysis.
    """

    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Configure matplotlib for better display
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    def plot_efficient_frontier(self,
                                returns: np.ndarray,
                                volatilities: np.ndarray,
                                sharpe_ratios: np.ndarray,
                                optimal_portfolio: Dict = None) -> plt.Figure:
        """
        Plot the efficient frontier with optimal portfolio highlighted.
        
        Args:
            returns: Array of portfolio returns
            volatilities: Array of portfolio volatilities
            sharpe_ratios: Array of Sharpe ratios
            optimal_portfolio: Dictionary with optimal portfolio data
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        if len(returns) > 0 and len(volatilities) > 0:
            # Create scatter plot colored by Sharpe ratio
            scatter = ax.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', alpha=0.6, s=50)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Sharpe Ratio', rotation=270, labelpad=15)

            # Highlight optimal portfolio if provided
            if optimal_portfolio:
                ax.scatter(optimal_portfolio['volatility'], optimal_portfolio['expected_return'],
                           marker='*', s=500, c='red', edgecolors='black', linewidth=2,
                           label=f"Optimal Portfolio (Sharpe: {optimal_portfolio['sharpe_ratio']:.3f})")
                ax.legend()

        ax.set_xlabel('Volatility (Risk)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_portfolio_allocation(self, weights: Dict[str, float]) -> plt.Figure:
        """
        Create a pie chart of portfolio allocation.
        
        Args:
            weights: Dictionary of asset weights
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Filter out zero weights
        non_zero_weights = {k: v for k, v in weights.items() if v > 0.001}

        if non_zero_weights:
            assets = list(non_zero_weights.keys())
            values = list(non_zero_weights.values())

            # Create pie chart
            wedges, texts, autotexts = ax.pie(values, labels=assets, autopct='%1.1f%%', startangle=90,
                                              explode=[0.05] * len(assets))

            # Enhance text
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

        ax.set_title('Portfolio Allocation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_portfolio_performance(self, portfolio_values: pd.Series, benchmark_values: pd.Series = None) -> plt.Figure:
        """
        Plot portfolio performance over time.
        
        Args:
            portfolio_values: Series of portfolio values
            benchmark_values: Series of benchmark values (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot cumulative values
        ax1.plot(portfolio_values.index, portfolio_values, linewidth=2, label='Portfolio', color='blue')

        if benchmark_values is not None:
            ax1.plot(benchmark_values.index, benchmark_values, linewidth=2, label='Benchmark', color='red', alpha=0.7)
            ax1.legend()

        ax1.set_title('Portfolio Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)

        # Plot drawdown
        cumulative_returns = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_risk_return_scatter(self, assets_data: pd.DataFrame, weights: Dict[str, float] = None) -> plt.Figure:
        """
        Plot risk-return scatter of individual assets.
        
        Args:
            assets_data: DataFrame with asset returns
            weights: Portfolio weights (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate risk and return for each asset
        returns = assets_data.mean() * 252
        volatilities = assets_data.std() * np.sqrt(252)

        # Create scatter plot
        scatter = ax.scatter(volatilities, returns, s=100, alpha=0.6)

        # Add asset labels
        for i, asset in enumerate(assets_data.columns):
            ax.annotate(asset, (volatilities.iloc[i], returns.iloc[i]), xytext=(5, 5), textcoords='offset points',
                        fontsize=9)

        # Highlight portfolio assets if weights provided
        if weights:
            portfolio_assets = list(weights.keys())
            mask = volatilities.index.isin(portfolio_assets)
            ax.scatter(volatilities[mask], returns[mask], s=200, alpha=0.8, c='red', edgecolors='black', linewidth=2,
                       label='Portfolio Assets')
            ax.legend()

        ax.set_xlabel('Volatility (Risk)')
        ax.set_ylabel('Expected Return')
        ax.set_title('Risk-Return Profile of Assets', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, returns_data: pd.DataFrame) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            returns_data: DataFrame with asset returns
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Calculate correlation matrix
        correlation_matrix = returns_data.corr()

        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, square=True, fmt='.2f', ax=ax,
                    cbar_kws={'label': 'Correlation'})

        ax.set_title('Asset Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_rolling_metrics(self, rolling_data: pd.DataFrame) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            rolling_data: DataFrame with rolling metrics
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Rolling return
        axes[0, 0].plot(rolling_data.index, rolling_data['Rolling_Return'])
        axes[0, 0].set_title('Rolling Annualized Return')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True, alpha=0.3)

        # Rolling volatility
        axes[0, 1].plot(rolling_data.index, rolling_data['Rolling_Volatility'], color='orange')
        axes[0, 1].set_title('Rolling Volatility')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].grid(True, alpha=0.3)

        # Rolling Sharpe ratio
        axes[1, 0].plot(rolling_data.index, rolling_data['Rolling_Sharpe'], color='green')
        axes[1, 0].set_title('Rolling Sharpe Ratio')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)

        # Rolling max drawdown
        axes[1, 1].fill_between(rolling_data.index, rolling_data['Rolling_Max_Drawdown'], 0, alpha=0.3, color='red')
        axes[1, 1].plot(rolling_data.index, rolling_data['Rolling_Max_Drawdown'], color='red')
        axes[1, 1].set_title('Rolling Maximum Drawdown')
        axes[1, 1].set_ylabel('Max Drawdown')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_monte_carlo_results(self, simulation_results: Dict) -> plt.Figure:
        """
        Plot Monte Carlo simulation results.
        
        Args:
            simulation_results: Dictionary with simulation data
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot simulation paths (sample)
        paths = simulation_results['all_paths']
        if len(paths) > 0:
            # Sample some paths to avoid overcrowding
            sample_paths = paths[::max(1, len(paths) // 50)]
            for path in sample_paths:
                ax1.plot(path, alpha=0.1, color='blue')

            # Plot percentiles
            percentiles_data = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
            ax1.plot(percentiles_data[2], color='red', linewidth=2, label='Median')
            ax1.fill_between(range(len(percentiles_data[0])), percentiles_data[0], percentiles_data[4], alpha=0.2,
                             color='gray', label='5th-95th Percentile')
            ax1.legend()
            ax1.set_title('Monte Carlo Simulation Paths')
            ax1.set_xlabel('Days')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True, alpha=0.3)

        # Plot final value distribution
        final_values = simulation_results['final_values']
        if len(final_values) > 0:
            ax2.hist(final_values, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(simulation_results['expected_value'], color='red', linestyle='--', linewidth=2,
                        label='Expected Value')
            ax2.axvline(simulation_results['percentiles']['5th'], color='orange', linestyle='--', linewidth=2,
                        label='5th Percentile (VaR)')
            ax2.legend()
            ax2.set_title('Distribution of Final Portfolio Values')
            ax2.set_xlabel('Final Portfolio Value ($)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_interactive_dashboard(self, portfolio_data: Dict) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            portfolio_data: Dictionary with portfolio analysis data
            
        Returns:
            Plotly figure
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Portfolio Performance', 'Asset Allocation', 'Risk Metrics', 'Correlation Matrix'),
                specs=[[{"secondary_y": False}, {"type": "domain"}], [{"secondary_y": False}, {"secondary_y": False}]]
            )

            # Portfolio performance (if available)
            if 'portfolio_values' in portfolio_data:
                values = portfolio_data['portfolio_values']
                fig.add_trace(
                    go.Scatter(x=values.index, y=values.values, name='Portfolio Value'),
                    row=1, col=1
                )

            # Asset allocation pie chart (if available)
            if 'weights' in portfolio_data:
                weights = portfolio_data['weights']
                non_zero_weights = {k: v for k, v in weights.items() if v > 0.001}
                if non_zero_weights:
                    fig.add_trace(
                        go.Pie(labels=list(non_zero_weights.keys()),
                               values=list(non_zero_weights.values()), name="Allocation"),
                        row=1, col=2
                    )

            # Update layout
            fig.update_layout(height=600, showlegend=True,
                              title_text="Portfolio Analysis Dashboard")

            return fig

        except Exception as e:
            # Return empty figure if error
            return go.Figure()

    def export_charts_to_file(self, figures: List[plt.Figure], filename: str):
        """
        Export multiple charts to a single PDF file.
        
        Args:
            figures: List of matplotlib figures
            filename: Output filename
        """
        try:
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(filename) as pdf:
                for fig in figures:
                    pdf.savefig(fig, bbox_inches='tight')

        except Exception as e:
            print(f"Error exporting charts: {e}")

    def embed_plot_in_tkinter(self, fig: plt.Figure, parent_frame) -> FigureCanvasTkAgg:
        """
        Embed matplotlib figure in Tkinter frame.
        
        Args:
            fig: Matplotlib figure
            parent_frame: Tkinter frame to embed in
            
        Returns:
            FigureCanvasTkAgg object
        """
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        return canvas
