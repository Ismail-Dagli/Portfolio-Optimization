"""
Portfolio Optimization Pro - Professional Quantitative Finance Tool

A comprehensive portfolio optimization application built with Python that provides:
- Modern Portfolio Theory optimization
- Multiple optimization strategies (Max Sharpe, Min Volatility, Risk Parity, etc.)
- Backtesting and Monte Carlo simulation
- Professional visualization and reporting
- CSV export capabilities

Author: Portfolio Optimization Pro Team
Version: 1.0.0
Date: 2025
"""

import sys
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def setup_logging():
    """Setup logging configuration for the application."""
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'portfolio_optimizer.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'yfinance',
        'pypfopt', 'cvxpy', 'scipy', 'sklearn', 'plotly', 'tkinter'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'pypfopt':
                import pypfopt
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False

    return True


def main():
    """Main application entry point."""
    print("Portfolio Optimization Pro - Starting Application...")
    print("=" * 60)

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Check dependencies
        if not check_dependencies():
            print("Error: Missing required dependencies. Please install them first.")
            sys.exit(1)

        # Import GUI after dependency check
        from gui import PortfolioOptimizationGUI
        import tkinter as tk

        # Create and run the application
        logger.info("Starting Portfolio Optimization GUI...")

        root = tk.Tk()
        app = PortfolioOptimizationGUI(root)

        print("Application started successfully!")
        print("Use the GUI to:")
        print("1. Load financial data")
        print("2. Optimize portfolios using various strategies")
        print("3. Perform backtesting and risk analysis")
        print("4. Generate professional visualizations")
        print("5. Export results to CSV files")
        print("\nEnjoy optimizing your portfolios!")

        root.mainloop()

    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure all required packages are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
