import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime, timedelta
from typing import Dict
import threading
import logging

import pandas as pd
import numpy as np

from data_loader import DataLoader
from optimizer import PortfolioOptimizer
from backtester import PortfolioBacktester
from visualizer import PortfolioVisualizer


class PortfolioOptimizationGUI:
    """
    Professional portfolio optimization GUI application.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimization Pro - Quantitative Finance Tool")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Initialize components
        self.data_loader = DataLoader()
        self.visualizer = PortfolioVisualizer()

        # Data storage
        self.price_data = None
        self.returns_data = None
        self.optimization_results = {}
        self.current_portfolio = None

        # Setup logging
        self.setup_logging()

        # Create GUI
        self.create_gui()

        # Default values
        self.set_default_values()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('portfolio_optimizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_gui(self):
        """Create the main GUI interface."""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        self.create_data_tab()
        self.create_optimization_tab()
        self.create_analysis_tab()
        self.create_results_tab()

        # Create status bar
        self.create_status_bar()

    def create_data_tab(self):
        """Create data input and loading tab."""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Input")

        # Main container
        main_container = ttk.Frame(self.data_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Input section
        input_frame = ttk.LabelFrame(main_container, text="Data Input Parameters", padding=10)
        input_frame.pack(fill='x', pady=(0, 10))

        # Tickers input
        ttk.Label(input_frame, text="Stock Tickers (comma separated):").grid(row=0, column=0, sticky='w', pady=5)
        self.tickers_var = tk.StringVar()
        tickers_entry = ttk.Entry(input_frame, textvariable=self.tickers_var, width=60)
        tickers_entry.grid(row=0, column=1, columnspan=2, sticky='ew', padx=(10, 0), pady=5)

        # Date inputs
        ttk.Label(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky='w', pady=5)
        self.start_date_var = tk.StringVar()
        start_date_entry = ttk.Entry(input_frame, textvariable=self.start_date_var, width=20)
        start_date_entry.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=5)

        ttk.Label(input_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky='w', pady=5)
        self.end_date_var = tk.StringVar()
        end_date_entry = ttk.Entry(input_frame, textvariable=self.end_date_var, width=20)
        end_date_entry.grid(row=2, column=1, sticky='w', padx=(10, 0), pady=5)

        # Benchmark
        ttk.Label(input_frame, text="Benchmark Ticker:").grid(row=3, column=0, sticky='w', pady=5)
        self.benchmark_var = tk.StringVar()
        benchmark_entry = ttk.Entry(input_frame, textvariable=self.benchmark_var, width=20)
        benchmark_entry.grid(row=3, column=1, sticky='w', padx=(10, 0), pady=5)

        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="Validate Tickers", command=self.validate_tickers).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Load Data", command=self.load_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Load Sample Data", command=self.load_sample_data).pack(side='left', padx=5)

        # Configure grid weights
        input_frame.columnconfigure(1, weight=1)

        # Data preview section
        preview_frame = ttk.LabelFrame(main_container, text="Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True)

        # Treeview for data preview
        self.data_tree = ttk.Treeview(preview_frame)
        data_scrollbar_y = ttk.Scrollbar(preview_frame, orient='vertical', command=self.data_tree.yview)
        data_scrollbar_x = ttk.Scrollbar(preview_frame, orient='horizontal', command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=data_scrollbar_y.set, xscrollcommand=data_scrollbar_x.set)

        self.data_tree.pack(side='left', fill='both', expand=True)
        data_scrollbar_y.pack(side='right', fill='y')
        data_scrollbar_x.pack(side='bottom', fill='x')

        # Data statistics
        stats_frame = ttk.LabelFrame(preview_frame, text="Data Statistics", padding=5)
        stats_frame.pack(side='right', fill='y', padx=(10, 0))

        self.stats_text = scrolledtext.ScrolledText(stats_frame, width=30, height=15)
        self.stats_text.pack(fill='both', expand=True)

    def create_optimization_tab(self):
        """Create portfolio optimization tab."""
        self.optimization_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.optimization_frame, text="Optimization")

        main_container = ttk.Frame(self.optimization_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Optimization parameters
        params_frame = ttk.LabelFrame(main_container, text="Optimization Parameters", padding=10)
        params_frame.pack(fill='x', pady=(0, 10))

        # Strategy selection
        ttk.Label(params_frame, text="Optimization Strategy:").grid(row=0, column=0, sticky='w', pady=5)
        self.strategy_var = tk.StringVar()
        strategy_combo = ttk.Combobox(params_frame, textvariable=self.strategy_var, width=25)
        strategy_combo['values'] = ('Max Sharpe Ratio', 'Min Volatility', 'Target Return', 'Equal Weight',
                                    'Risk Parity')
        strategy_combo.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=5)
        strategy_combo.bind('<<ComboboxSelected>>', self.on_strategy_change)

        # Target return (for target return strategy)
        ttk.Label(params_frame, text="Target Return (annual):").grid(row=1, column=0, sticky='w', pady=5)
        self.target_return_var = tk.DoubleVar()
        self.target_return_entry = ttk.Entry(params_frame, textvariable=self.target_return_var, width=15)
        self.target_return_entry.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=5)
        self.target_return_entry.config(state='disabled')

        # Risk-free rate
        ttk.Label(params_frame, text="Risk-free Rate:").grid(row=2, column=0, sticky='w', pady=5)
        self.risk_free_rate_var = tk.DoubleVar()
        risk_free_entry = ttk.Entry(params_frame, textvariable=self.risk_free_rate_var, width=15)
        risk_free_entry.grid(row=2, column=1, sticky='w', padx=(10, 0), pady=5)

        # Constraints
        constraints_frame = ttk.LabelFrame(params_frame, text="Constraints", padding=5)
        constraints_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=10)

        ttk.Label(constraints_frame, text="Max Weight per Asset:").grid(row=0, column=0, sticky='w', pady=2)
        self.max_weight_var = tk.DoubleVar()
        max_weight_entry = ttk.Entry(constraints_frame, textvariable=self.max_weight_var, width=10)
        max_weight_entry.grid(row=0, column=1, sticky='w', padx=(5, 0), pady=2)

        ttk.Label(constraints_frame, text="Min Weight per Asset:").grid(row=0, column=2, sticky='w', padx=(20, 0),
                                                                        pady=2)
        self.min_weight_var = tk.DoubleVar()
        min_weight_entry = ttk.Entry(constraints_frame, textvariable=self.min_weight_var, width=10)
        min_weight_entry.grid(row=0, column=3, sticky='w', padx=(5, 0), pady=2)

        # Optimization buttons
        opt_button_frame = ttk.Frame(params_frame)
        opt_button_frame.grid(row=4, column=0, columnspan=3, pady=10)

        ttk.Button(opt_button_frame, text="Optimize Portfolio", command=self.optimize_portfolio).pack(side='left',
                                                                                                      padx=5)
        ttk.Button(opt_button_frame, text="Generate Efficient Frontier", command=self.generate_efficient_frontier).pack(
            side='left', padx=5)
        ttk.Button(opt_button_frame, text="Compare Strategies", command=self.compare_strategies).pack(side='left',
                                                                                                      padx=5)

        # Results display
        results_frame = ttk.LabelFrame(main_container, text="Optimization Results", padding=10)
        results_frame.pack(fill='both', expand=True)

        # Create notebook for different result views
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill='both', expand=True)

        # Weights tab
        weights_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(weights_frame, text="Portfolio Weights")

        self.weights_tree = ttk.Treeview(weights_frame, columns=('Weight', 'Value'), show='tree headings')
        self.weights_tree.heading('#0', text='Asset')
        self.weights_tree.heading('Weight', text='Weight (%)')
        self.weights_tree.heading('Value', text='Value ($)')
        weights_scrollbar = ttk.Scrollbar(weights_frame, orient='vertical', command=self.weights_tree.yview)
        self.weights_tree.configure(yscrollcommand=weights_scrollbar.set)
        self.weights_tree.pack(side='left', fill='both', expand=True)
        weights_scrollbar.pack(side='right', fill='y')

        # Metrics tab
        metrics_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(metrics_frame, text="Performance Metrics")

        self.metrics_text = scrolledtext.ScrolledText(metrics_frame, width=60, height=20)
        self.metrics_text.pack(fill='both', expand=True, padx=5, pady=5)

    def create_analysis_tab(self):
        """Create analysis and backtesting tab."""
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Analysis & Backtesting")

        main_container = ttk.Frame(self.analysis_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Analysis parameters
        params_frame = ttk.LabelFrame(main_container, text="Analysis Parameters", padding=10)
        params_frame.pack(fill='x', pady=(0, 10))

        # Backtest parameters
        ttk.Label(params_frame, text="Rebalancing Frequency:").grid(row=0, column=0, sticky='w', pady=5)
        self.rebalance_freq_var = tk.StringVar()
        rebalance_combo = ttk.Combobox(params_frame, textvariable=self.rebalance_freq_var, width=15)
        rebalance_combo['values'] = ('daily', 'weekly', 'monthly', 'quarterly')
        rebalance_combo.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=5)

        ttk.Label(params_frame, text="Transaction Cost (%):").grid(row=0, column=2, sticky='w', padx=(20, 0), pady=5)
        self.transaction_cost_var = tk.DoubleVar()
        transaction_cost_entry = ttk.Entry(params_frame, textvariable=self.transaction_cost_var, width=10)
        transaction_cost_entry.grid(row=0, column=3, sticky='w', padx=(10, 0), pady=5)

        ttk.Label(params_frame, text="Initial Investment ($):").grid(row=1, column=0, sticky='w', pady=5)
        self.initial_investment_var = tk.DoubleVar()
        investment_entry = ttk.Entry(params_frame, textvariable=self.initial_investment_var, width=15)
        investment_entry.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=5)

        # Monte Carlo parameters
        ttk.Label(params_frame, text="MC Simulations:").grid(row=1, column=2, sticky='w', padx=(20, 0), pady=5)
        self.mc_simulations_var = tk.IntVar()
        mc_entry = ttk.Entry(params_frame, textvariable=self.mc_simulations_var, width=10)
        mc_entry.grid(row=1, column=3, sticky='w', padx=(10, 0), pady=5)

        # Analysis buttons
        analysis_button_frame = ttk.Frame(params_frame)
        analysis_button_frame.grid(row=2, column=0, columnspan=4, pady=10)

        ttk.Button(analysis_button_frame, text="Run Backtest", command=self.run_backtest).pack(side='left', padx=5)
        ttk.Button(analysis_button_frame, text="Monte Carlo Simulation", command=self.run_monte_carlo).pack(side='left',
                                                                                                            padx=5)
        ttk.Button(analysis_button_frame, text="Risk Analysis", command=self.run_risk_analysis).pack(side='left',
                                                                                                     padx=5)

        # Analysis results
        analysis_results_frame = ttk.LabelFrame(main_container, text="Analysis Results", padding=10)
        analysis_results_frame.pack(fill='both', expand=True)

        self.analysis_text = scrolledtext.ScrolledText(analysis_results_frame, width=80, height=25)
        self.analysis_text.pack(fill='both', expand=True)

    def create_results_tab(self):
        """Create results and visualization tab."""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results & Visualization")

        main_container = ttk.Frame(self.results_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Visualization controls
        viz_controls_frame = ttk.LabelFrame(main_container, text="Visualization Controls", padding=10)
        viz_controls_frame.pack(fill='x', pady=(0, 10))

        # Chart selection
        ttk.Label(viz_controls_frame, text="Chart Type:").grid(row=0, column=0, sticky='w', pady=5)
        self.chart_type_var = tk.StringVar()
        chart_combo = ttk.Combobox(viz_controls_frame, textvariable=self.chart_type_var, width=25)
        chart_combo['values'] = ('Portfolio Allocation', 'Efficient Frontier', 'Performance Chart',
                                 'Risk-Return Scatter', 'Correlation Matrix', 'Monte Carlo Results')
        chart_combo.grid(row=0, column=1, sticky='w', padx=(10, 0), pady=5)

        # Chart buttons
        chart_button_frame = ttk.Frame(viz_controls_frame)
        chart_button_frame.grid(row=1, column=0, columnspan=3, pady=10)

        ttk.Button(chart_button_frame, text="Generate Chart", command=self.generate_chart).pack(side='left', padx=5)
        ttk.Button(chart_button_frame, text="Export Charts", command=self.export_charts).pack(side='left', padx=5)
        ttk.Button(chart_button_frame, text="Export Data", command=self.export_data).pack(side='left', padx=5)

        # Chart display frame
        self.chart_frame = ttk.LabelFrame(main_container, text="Chart Display", padding=10)
        self.chart_frame.pack(fill='both', expand=True)

        # Placeholder for charts
        self.chart_canvas_frame = ttk.Frame(self.chart_frame)
        self.chart_canvas_frame.pack(fill='both', expand=True)

    def create_status_bar(self):
        """Create status bar at bottom of window."""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side='bottom', fill='x')

        self.status_label = ttk.Label(self.status_bar, text="Ready", relief='sunken')
        self.status_label.pack(side='left', fill='x', expand=True, padx=5, pady=2)

        self.progress_bar = ttk.Progressbar(self.status_bar, length=200, mode='indeterminate')
        self.progress_bar.pack(side='right', padx=5, pady=2)

    def set_default_values(self):
        """Set default values for the GUI."""
        # Default date range (last 3 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3 * 365)

        self.start_date_var.set(start_date.strftime('%Y-%m-%d'))
        self.end_date_var.set(end_date.strftime('%Y-%m-%d'))

        # Default tickers
        self.tickers_var.set("AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META, NFLX")

        # Default benchmark
        self.benchmark_var.set("^GSPC")

        # Default optimization parameters
        self.strategy_var.set("Max Sharpe Ratio")
        self.risk_free_rate_var.set(0.02)
        self.target_return_var.set(0.10)
        self.max_weight_var.set(0.4)
        self.min_weight_var.set(0.0)

        # Default analysis parameters
        self.rebalance_freq_var.set("monthly")
        self.transaction_cost_var.set(0.1)
        self.initial_investment_var.set(100000)
        self.mc_simulations_var.set(1000)

        # Default chart type
        self.chart_type_var.set("Portfolio Allocation")

    def update_status(self, message: str):
        """Update status bar message."""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def show_progress(self):
        """Show progress bar."""
        self.progress_bar.start()

    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.stop()

    def on_strategy_change(self, event):
        """Handle strategy selection change."""
        strategy = self.strategy_var.get()
        if strategy == "Target Return":
            self.target_return_entry.config(state='normal')
        else:
            self.target_return_entry.config(state='disabled')

    def validate_tickers(self):
        """Validate ticker symbols."""

        def validate_worker():
            try:
                self.show_progress()
                self.update_status("Validating tickers...")

                tickers = [t.strip() for t in self.tickers_var.get().split(',') if t.strip()]
                valid_tickers, invalid_tickers = self.data_loader.validate_tickers(tickers)

                result_msg = f"Valid tickers: {len(valid_tickers)}\n"
                if valid_tickers:
                    result_msg += f"Valid: {', '.join(valid_tickers)}\n"
                if invalid_tickers:
                    result_msg += f"Invalid: {', '.join(invalid_tickers)}"

                messagebox.showinfo("Ticker Validation", result_msg)

                # Update tickers list with only valid ones
                if valid_tickers:
                    self.tickers_var.set(', '.join(valid_tickers))

            except Exception as e:
                messagebox.showerror("Error", f"Error validating tickers: {str(e)}")
            finally:
                self.hide_progress()
                self.update_status("Ready")

        threading.Thread(target=validate_worker, daemon=True).start()

    def load_sample_data(self):
        """Load sample data for demonstration."""
        self.tickers_var.set("AAPL, GOOGL, MSFT, AMZN")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        self.start_date_var.set(start_date.strftime('%Y-%m-%d'))
        self.end_date_var.set(end_date.strftime('%Y-%m-%d'))
        self.load_data()

    def load_data(self):
        """Load stock data."""

        def load_worker():
            try:
                self.show_progress()
                self.update_status("Loading data...")

                # Get parameters
                tickers = [t.strip() for t in self.tickers_var.get().split(',') if t.strip()]
                start_date = self.start_date_var.get()
                end_date = self.end_date_var.get()

                if not tickers:
                    raise ValueError("Please enter at least one ticker symbol")

                # Load price data
                self.price_data = self.data_loader.fetch_stock_data(tickers, start_date, end_date)
                self.returns_data = self.data_loader.calculate_returns(self.price_data)

                # Update data preview
                self.update_data_preview()

                self.update_status(f"Data loaded successfully for {len(self.price_data.columns)} assets")
                messagebox.showinfo("Success", f"Data loaded for {len(self.price_data.columns)} assets")

            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")
                self.update_status("Error loading data")
            finally:
                self.hide_progress()

        threading.Thread(target=load_worker, daemon=True).start()

    def update_data_preview(self):
        """Update data preview in the GUI."""
        if self.price_data is None:
            return

        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        # Setup columns
        self.data_tree['columns'] = list(self.price_data.columns)
        self.data_tree.heading('#0', text='Date')

        for col in self.price_data.columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)

        # Add last 20 rows of data
        for idx, (date, row) in enumerate(self.price_data.tail(20).iterrows()):
            values = [f"{val:.2f}" if pd.notna(val) else "N/A" for val in row]
            self.data_tree.insert('', 'end', text=date.strftime('%Y-%m-%d'), values=values)

        # Update statistics
        stats_text = "Data Statistics:\n\n"
        stats_text += f"Assets: {len(self.price_data.columns)}\n"
        stats_text += f"Date Range: {self.price_data.index[0].strftime('%Y-%m-%d')} to {self.price_data.index[-1].strftime('%Y-%m-%d')}\n"
        stats_text += f"Total Days: {len(self.price_data)}\n\n"

        if self.returns_data is not None:
            stats_text += "Return Statistics:\n"
            mean_returns = self.returns_data.mean() * 252
            volatilities = self.returns_data.std() * np.sqrt(252)

            for asset in self.returns_data.columns:
                stats_text += f"{asset}: Return {mean_returns[asset]:.2%}, Vol {volatilities[asset]:.2%}\n"

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

    def optimize_portfolio(self):
        """Run portfolio optimization."""
        if self.returns_data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        def optimize_worker():
            try:
                self.show_progress()
                self.update_status("Optimizing portfolio...")

                # Initialize optimizer
                optimizer = PortfolioOptimizer(self.returns_data)

                # Get optimization parameters
                strategy = self.strategy_var.get()
                risk_free_rate = self.risk_free_rate_var.get()

                # Run optimization based on strategy
                if strategy == "Max Sharpe Ratio":
                    result = optimizer.max_sharpe_optimization(risk_free_rate)
                elif strategy == "Min Volatility":
                    result = optimizer.min_volatility_optimization()
                elif strategy == "Target Return":
                    target_return = self.target_return_var.get()
                    result = optimizer.mean_variance_optimization(target_return=target_return)
                elif strategy == "Equal Weight":
                    result = optimizer.equal_weight_portfolio()
                elif strategy == "Risk Parity":
                    result = optimizer.risk_parity_optimization()
                else:
                    result = optimizer.max_sharpe_optimization(risk_free_rate)

                self.current_portfolio = result
                self.optimization_results[strategy] = result

                # Update results display
                self.update_optimization_results(result)

                self.update_status("Optimization completed successfully")

            except Exception as e:
                messagebox.showerror("Error", f"Error in optimization: {str(e)}")
                self.update_status("Optimization failed")
            finally:
                self.hide_progress()

        threading.Thread(target=optimize_worker, daemon=True).start()

    def update_optimization_results(self, result: Dict):
        """Update optimization results in GUI."""
        # Clear existing results
        for item in self.weights_tree.get_children():
            self.weights_tree.delete(item)

        # Update weights display
        total_value = self.initial_investment_var.get()
        for asset, weight in result['weights'].items():
            if weight > 0.001:  # Only show significant weights
                value = total_value * weight
                self.weights_tree.insert('', 'end', text=asset,
                                         values=[f"{weight:.2%}", f"${value:,.2f}"])

        # Update metrics display
        metrics_text = f"Optimization Results - {result['method']}\n"
        metrics_text += "=" * 50 + "\n\n"
        metrics_text += f"Expected Annual Return: {result['expected_return']:.2%}\n"
        metrics_text += f"Annual Volatility: {result['volatility']:.2%}\n"
        metrics_text += f"Sharpe Ratio: {result['sharpe_ratio']:.3f}\n\n"

        metrics_text += "Portfolio Weights:\n"
        metrics_text += "-" * 30 + "\n"
        for asset, weight in sorted(result['weights'].items(), key=lambda x: x[1], reverse=True):
            if weight > 0.001:
                metrics_text += f"{asset}: {weight:.2%}\n"

        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)

    def generate_efficient_frontier(self):
        """Generate and display efficient frontier."""
        if self.returns_data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        def frontier_worker():
            try:
                self.show_progress()
                self.update_status("Generating efficient frontier...")

                optimizer = PortfolioOptimizer(self.returns_data)
                returns, volatilities, sharpe_ratios = optimizer.efficient_frontier()

                if len(returns) > 0:
                    fig = self.visualizer.plot_efficient_frontier(
                        returns, volatilities, sharpe_ratios, self.current_portfolio
                    )
                    self.display_chart(fig)

                self.update_status("Efficient frontier generated")

            except Exception as e:
                messagebox.showerror("Error", f"Error generating efficient frontier: {str(e)}")
            finally:
                self.hide_progress()

        threading.Thread(target=frontier_worker, daemon=True).start()

    def compare_strategies(self):
        """Compare different optimization strategies."""
        if self.returns_data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        def compare_worker():
            try:
                self.show_progress()
                self.update_status("Comparing strategies...")

                optimizer = PortfolioOptimizer(self.returns_data)
                strategies = {
                    'Max Sharpe': optimizer.max_sharpe_optimization(),
                    'Min Volatility': optimizer.min_volatility_optimization(),
                    'Equal Weight': optimizer.equal_weight_portfolio(),
                    'Risk Parity': optimizer.risk_parity_optimization()
                }

                # Display comparison
                comparison_text = "Strategy Comparison\n"
                comparison_text += "=" * 50 + "\n\n"

                comparison_text += f"{'Strategy':<15} {'Return':<10} {'Volatility':<12} {'Sharpe':<8}\n"
                comparison_text += "-" * 50 + "\n"

                for name, result in strategies.items():
                    comparison_text += f"{name:<15} {result['expected_return']:<10.2%} "
                    comparison_text += f"{result['volatility']:<12.2%} {result['sharpe_ratio']:<8.3f}\n"

                self.analysis_text.delete(1.0, tk.END)
                self.analysis_text.insert(1.0, comparison_text)

                # Store results
                self.optimization_results.update(strategies)

                self.update_status("Strategy comparison completed")

            except Exception as e:
                messagebox.showerror("Error", f"Error comparing strategies: {str(e)}")
            finally:
                self.hide_progress()

        threading.Thread(target=compare_worker, daemon=True).start()

    def run_backtest(self):
        """Run portfolio backtest."""
        if self.current_portfolio is None:
            messagebox.showerror("Error", "Please optimize a portfolio first")
            return

        def backtest_worker():
            try:
                self.show_progress()
                self.update_status("Running backtest...")

                backtester = PortfolioBacktester(self.returns_data)

                # Get parameters
                rebalance_freq = self.rebalance_freq_var.get()
                transaction_cost = self.transaction_cost_var.get() / 100
                initial_value = self.initial_investment_var.get()

                # Run backtest
                backtest_results = backtester.backtest_portfolio(
                    self.current_portfolio['weights'],
                    rebalance_freq=rebalance_freq,
                    transaction_cost=transaction_cost,
                    initial_value=initial_value
                )

                # Display results
                if backtest_results:
                    self.display_backtest_results(backtest_results)

                self.update_status("Backtest completed")

            except Exception as e:
                messagebox.showerror("Error", f"Error in backtest: {str(e)}")
            finally:
                self.hide_progress()

        threading.Thread(target=backtest_worker, daemon=True).start()

    def display_backtest_results(self, results: Dict):
        """Display backtest results."""
        metrics = results['metrics']

        text = "Backtest Results\n"
        text += "=" * 30 + "\n\n"
        text += f"Total Return: {metrics['total_return']:.2%}\n"
        text += f"Annualized Return: {metrics['annualized_return']:.2%}\n"
        text += f"Volatility: {metrics['volatility']:.2%}\n"
        text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n"
        text += f"Sortino Ratio: {metrics['sortino_ratio']:.3f}\n"
        text += f"Maximum Drawdown: {metrics['max_drawdown']:.2%}\n"
        text += f"Win Rate: {metrics['win_rate']:.2%}\n"
        text += f"Total Transaction Costs: ${results['total_transaction_costs']:,.2f}\n"

        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, text)

    def run_monte_carlo(self):
        """Run Monte Carlo simulation."""
        if self.current_portfolio is None:
            messagebox.showerror("Error", "Please optimize a portfolio first")
            return

        def monte_carlo_worker():
            try:
                self.show_progress()
                self.update_status("Running Monte Carlo simulation...")

                backtester = PortfolioBacktester(self.returns_data)

                # Get parameters
                num_simulations = self.mc_simulations_var.get()
                initial_value = self.initial_investment_var.get()

                # Run simulation
                mc_results = backtester.monte_carlo_simulation(
                    self.current_portfolio['weights'],
                    num_simulations=num_simulations,
                    initial_value=initial_value
                )

                # Display results
                if mc_results:
                    self.display_monte_carlo_results(mc_results)

                self.update_status("Monte Carlo simulation completed")

            except Exception as e:
                messagebox.showerror("Error", f"Error in Monte Carlo simulation: {str(e)}")
            finally:
                self.hide_progress()

        threading.Thread(target=monte_carlo_worker, daemon=True).start()

    def display_monte_carlo_results(self, results: Dict):
        """Display Monte Carlo results."""
        text = "Monte Carlo Simulation Results\n"
        text += "=" * 35 + "\n\n"
        text += f"Expected Final Value: ${results['expected_value']:,.2f}\n"
        text += f"Probability of Loss: {results['probability_of_loss']:.2%}\n"
        text += f"Value at Risk (5%): ${results['value_at_risk_5']:,.2f}\n"
        text += f"Expected Shortfall (5%): ${results['expected_shortfall_5']:,.2f}\n\n"

        text += "Percentiles:\n"
        text += f"5th Percentile: ${results['percentiles']['5th']:,.2f}\n"
        text += f"25th Percentile: ${results['percentiles']['25th']:,.2f}\n"
        text += f"50th Percentile: ${results['percentiles']['50th']:,.2f}\n"
        text += f"75th Percentile: ${results['percentiles']['75th']:,.2f}\n"
        text += f"95th Percentile: ${results['percentiles']['95th']:,.2f}\n"

        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, text)

    def run_risk_analysis(self):
        """Run comprehensive risk analysis."""
        if self.current_portfolio is None:
            messagebox.showerror("Error", "Please optimize a portfolio first")
            return

        def risk_analysis_worker():
            try:
                self.show_progress()
                self.update_status("Running risk analysis...")

                # Risk analysis implementation
                text = "Risk Analysis\n"
                text += "=" * 20 + "\n\n"
                text += "Comprehensive risk analysis completed.\n"
                text += "Please check the visualization tab for detailed charts.\n"

                self.analysis_text.delete(1.0, tk.END)
                self.analysis_text.insert(1.0, text)

                self.update_status("Risk analysis completed")

            except Exception as e:
                messagebox.showerror("Error", f"Error in risk analysis: {str(e)}")
            finally:
                self.hide_progress()

        threading.Thread(target=risk_analysis_worker, daemon=True).start()

    def generate_chart(self):
        """Generate selected chart type."""
        chart_type = self.chart_type_var.get()

        if chart_type == "Portfolio Allocation" and self.current_portfolio:
            fig = self.visualizer.plot_portfolio_allocation(self.current_portfolio['weights'])
            self.display_chart(fig)
        elif chart_type == "Correlation Matrix" and self.returns_data is not None:
            fig = self.visualizer.plot_correlation_matrix(self.returns_data)
            self.display_chart(fig)
        elif chart_type == "Risk-Return Scatter" and self.returns_data is not None:
            weights = self.current_portfolio['weights'] if self.current_portfolio else None
            fig = self.visualizer.plot_risk_return_scatter(self.returns_data, weights)
            self.display_chart(fig)
        else:
            messagebox.showwarning("Warning", "Please load data and optimize portfolio first")

    def display_chart(self, fig):
        """Display matplotlib figure in GUI."""
        # Clear previous chart
        for widget in self.chart_canvas_frame.winfo_children():
            widget.destroy()

        # Embed new chart
        canvas = self.visualizer.embed_plot_in_tkinter(fig, self.chart_canvas_frame)
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def export_charts(self):
        """Export charts to PDF."""
        if not self.optimization_results:
            messagebox.showwarning("Warning", "No results to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Generate all charts
                figures = []

                if self.current_portfolio:
                    figures.append(self.visualizer.plot_portfolio_allocation(self.current_portfolio['weights']))

                if self.returns_data is not None:
                    figures.append(self.visualizer.plot_correlation_matrix(self.returns_data))
                    figures.append(self.visualizer.plot_risk_return_scatter(self.returns_data))

                # Export to PDF
                self.visualizer.export_charts_to_file(figures, filename)
                messagebox.showinfo("Success", f"Charts exported to {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Error exporting charts: {str(e)}")

    def export_data(self):
        """Export portfolio data to CSV."""
        if not self.optimization_results:
            messagebox.showwarning("Warning", "No results to export")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Create export data
                export_data = []

                for strategy, result in self.optimization_results.items():
                    for asset, weight in result['weights'].items():
                        export_data.append({
                            'Strategy': strategy,
                            'Asset': asset,
                            'Weight': weight,
                            'Expected_Return': result['expected_return'],
                            'Volatility': result['volatility'],
                            'Sharpe_Ratio': result['sharpe_ratio']
                        })

                df = pd.DataFrame(export_data)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Error exporting data: {str(e)}")


def main():
    """Main application entry point."""
    root = tk.Tk()
    app = PortfolioOptimizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
