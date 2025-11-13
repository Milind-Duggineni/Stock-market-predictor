"""
Backtester Module

This module implements a backtesting framework for quantitative trading strategies.
It supports multiple assets, transaction costs, and various position sizing methods.
"""
import os
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

from .strategy import generate_signals
from .metrics import get_all_metrics

# Load configuration
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

class Backtester:
    """
    A backtesting engine for quantitative trading strategies.
    
    Features:
    - Multiple position sizing methods
    - Transaction costs and slippage modeling
    - Comprehensive performance metrics
    - Support for single and multiple assets
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        position_sizing: str = 'fixed',  # 'fixed', 'volatility', 'kelly'
        risk_per_trade: float = 0.01,   # 1% risk per trade
        max_position_size: float = 0.1,  # Max 10% of portfolio per position
        verbose: bool = False
    ):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting portfolio value
            transaction_cost: Transaction cost as a fraction of trade value
            slippage: Slippage as a fraction of trade value
            position_sizing: Position sizing method ('fixed', 'volatility', 'kelly')
            risk_per_trade: Risk per trade as a fraction of portfolio (for position sizing)
            max_position_size: Maximum position size as a fraction of portfolio
            verbose: Whether to print detailed logs
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.verbose = verbose
        
        # Will be set during backtest
        self.equity_curve = None
        self.returns = None
        self.positions = None
        self.trades = []
        self.metrics = {}
    
    def _calculate_position_size(
        self, 
        signal: float, 
        current_price: float, 
        volatility: float = None,
        account_equity: float = None
    ) -> float:
        """Calculate position size based on the selected method."""
        if signal == 0:
            return 0.0
            
        if self.position_sizing == 'fixed':
            return signal * self.max_position_size
            
        elif self.position_sizing == 'volatility' and volatility is not None:
            # Scale position inversely with volatility
            position_size = min(
                self.risk_per_trade / (volatility + 1e-8),  # Avoid division by zero
                self.max_position_size
            )
            return signal * position_size
            
        elif self.position_sizing == 'kelly' and volatility is not None and account_equity is not None:
            # Kelly criterion position sizing
            win_prob = 0.55  # This should be estimated from historical data
            win_loss_ratio = 1.5  # This should be estimated from historical data
            kelly_f = win_prob - (1 - win_prob) / win_loss_ratio
            position_size = kelly_f * self.risk_per_trade
            return signal * min(position_size, self.max_position_size)
            
        return signal * self.max_position_size  # Default fallback
    
    def _calculate_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility."""
        return prices.pct_change().rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def backtest(
        self, 
        data: pd.DataFrame, 
        model_pred_col: str = "Predicted",
        price_col: str = "Close"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run a backtest on the given data.
        
        Args:
            data: DataFrame with price data and predictions
            model_pred_col: Column name for model predictions
            price_col: Column name for price data
            
        Returns:
            Tuple of (results DataFrame, metrics dictionary)
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        df = data.copy()
        
        # Generate trading signals
        df = generate_signals(df, model_pred_col=model_pred_col)
        
        # Initialize columns
        df['Position'] = 0.0
        df['Trade'] = 0.0
        df['Cash'] = self.initial_capital
        df['Portfolio_Value'] = self.initial_capacity
        df['Returns'] = 0.0
        
        # Calculate volatility for position sizing
        if self.position_sizing == 'volatility':
            df['Volatility'] = self._calculate_volatility(df[price_col])
        
        # Run backtest
        for i in range(1, len(df)):
            current_price = df.loc[df.index[i], price_col]
            prev_price = df.loc[df.index[i-1], price_col]
            
            # Get signal and calculate position size
            signal = df.loc[df.index[i-1], 'Signal']
            
            # Calculate position size based on selected method
            if self.position_sizing == 'volatility':
                vol = df.loc[df.index[i-1], 'Volatility']
                position_size = self._calculate_position_size(
                    signal, current_price, 
                    volatility=vol,
                    account_equity=df.loc[df.index[i-1], 'Portfolio_Value']
                )
            else:
                position_size = self._calculate_position_size(
                    signal, current_price,
                    account_equity=df.loc[df.index[i-1], 'Portfolio_Value']
                )
            
            # Calculate position value in dollars
            position_value = position_size * df.loc[df.index[i-1], 'Portfolio_Value']
            
            # Calculate number of shares to buy/sell
            shares = position_value // current_price
            
            # Update position and cash
            position_change = shares - df.loc[df.index[i-1], 'Position']
            transaction_cost = abs(position_change * current_price) * self.transaction_cost
            slippage_cost = abs(position_change * current_price) * self.slippage
            
            df.loc[df.index[i], 'Position'] = shares
            df.loc[df.index[i], 'Cash'] = df.loc[df.index[i-1], 'Cash'] - \
                                        (position_change * current_price) - \
                                        transaction_cost - slippage_cost
            
            # Update portfolio value
            position_value = df.loc[df.index[i], 'Position'] * current_price
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i], 'Cash'] + position_value
            
            # Calculate returns
            df.loc[df.index[i], 'Returns'] = (df.loc[df.index[i], 'Portfolio_Value'] / 
                                            df.loc[df.index[i-1], 'Portfolio_Value'] - 1)
            
            # Track trades
            if position_change != 0:
                self.trades.append({
                    'date': df.index[i],
                    'price': current_price,
                    'shares': position_change,
                    'value': position_change * current_price,
                    'type': 'buy' if position_change > 0 else 'sell',
                    'transaction_cost': transaction_cost,
                    'slippage': slippage_cost
                })
        
        # Calculate metrics
        self.equity_curve = df['Portfolio_Value'].values
        self.returns = df['Returns'].values
        self.positions = df['Position'].values
        
        # Get comprehensive metrics
        self.metrics = get_all_metrics(
            returns=df['Returns'],
            equity_curve=df['Portfolio_Value'].values,
            risk_free_rate=CONFIG.get('risk_free_rate', 0.0),
            periods_per_year=CONFIG.get('periods_per_year', 252),
            confidence_level=CONFIG.get('confidence_level', 0.95)
        )
        
        # Add trade statistics
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            self.metrics.update({
                'total_trades': len(trades_df),
                'win_rate': (trades_df['value'] > 0).mean() if len(trades_df) > 0 else 0,
                'avg_trade_return': trades_df['value'].mean(),
                'profit_factor': (trades_df[trades_df['value'] > 0]['value'].sum() / 
                                abs(trades_df[trades_df['value'] < 0]['value'].sum() + 1e-10)),
                'max_consecutive_wins': self._max_consecutive(trades_df['value'] > 0),
                'max_consecutive_losses': self._max_consecutive(trades_df['value'] < 0)
            })
        
        return df, self.metrics
    
    def _max_consecutive(self, bool_series: pd.Series) -> int:
        """Calculate maximum number of consecutive True values in a boolean series."""
        if not len(bool_series):
            return 0
        bool_series = bool_series.astype(int)
        return max(sum(1 for _ in group) 
                  for key, group in bool_series.groupby((bool_series != bool_series.shift()).cumsum()) 
                  if key == 1)
    
    def generate_report(self, output_dir: str = "reports") -> None:
        """Generate a comprehensive backtest report."
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(os.path.join(output_dir, "backtest_metrics.csv"), index=False)
        
        # Save trades to CSV
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(os.path.join(output_dir, "trades.csv"), index=False)
        
        # Generate plots
        self._generate_plots(output_dir)
    
    def _generate_plots(self, output_dir: str) -> None:
        """Generate and save performance plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn')
        
        # Equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, label='Portfolio Value')
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
        plt.close()
        
        # Drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        
        plt.figure(figsize=(12, 4))
        plt.fill_between(range(len(drawdown)), drawdown * 100, 0, alpha=0.3, color='red')
        plt.title('Drawdown')
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'drawdown.png'))
        plt.close()

def run_backtest(
    data: Union[str, pd.DataFrame],
    model_pred_col: str = "Predicted",
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    position_sizing: str = 'fixed',
    output_dir: str = "reports"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run a backtest with the given parameters.
    
    Args:
        data: DataFrame or path to CSV with price and prediction data
        model_pred_col: Column name for model predictions
        initial_capital: Starting portfolio value
        transaction_cost: Transaction cost as a fraction of trade value
        slippage: Slippage as a fraction of trade value
        position_sizing: Position sizing method ('fixed', 'volatility', 'kelly')
        output_dir: Directory to save reports and plots
        
    Returns:
        Tuple of (results DataFrame, metrics dictionary)
    """
    # Load data if path is provided
    if isinstance(data, str):
        data = pd.read_csv(data)
    
    # Initialize and run backtester
    backtester = Backtester(
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        slippage=slippage,
        position_sizing=position_sizing
    )
    
    results, metrics = backtester.backtest(data, model_pred_col=model_pred_col)
    
    # Generate and save report
    backtester.generate_report(output_dir)
    
    return results, metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtest on trading strategy')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Transaction cost as fraction')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage as fraction')
    parser.add_argument('--position-sizing', type=str, default='fixed', 
                       choices=['fixed', 'volatility', 'kelly'], 
                       help='Position sizing method')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data and predictions
    test_data_path = os.path.join("data", "processed", f"{args.ticker}_test.csv")
    preds_path = os.path.join("reports", "plots", f"{args.ticker}_XGBoost_pred.csv")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions not found: {preds_path}")
    
    # Load and prepare data
    df_test = pd.read_csv(test_data_path)
    df_preds = pd.read_csv(preds_path)
    
    # Ensure dates are in datetime format and set as index
    if 'Date' in df_test.columns:
        df_test['Date'] = pd.to_datetime(df_test['Date'])
        df_test.set_index('Date', inplace=True)
    
    # Merge predictions with test data
    if 'Predicted' not in df_test.columns and 'Predicted' in df_preds.columns:
        df_test['Predicted'] = df_preds['Predicted'].values
    
    # Run backtest
    results, metrics = run_backtest(
        data=df_test,
        model_pred_col="Predicted",
        initial_capital=args.initial_capital,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        position_sizing=args.position_sizing,
        output_dir=os.path.join(args.output_dir, args.ticker)
    )
    
    # Print summary
    print(f"\nBacktest results for {args.ticker}:")
    print("-" * 50)
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${results['Portfolio_Value'].iloc[-1]:,.2f}")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Annualized Return (CAGR): {metrics['cagr']*100:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print("-" * 50)
    print(f"\nDetailed metrics and plots saved to: {os.path.join(args.output_dir, args.ticker)}")
