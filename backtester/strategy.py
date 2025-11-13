"""
Trading Strategy Module

This module implements various signal generation strategies for quantitative trading,
including mean reversion, momentum, and machine learning-based approaches.
"""
from typing import Tuple, Optional, Union, List, Dict
import pandas as pd
import numpy as np
from enum import Enum

class SignalType(Enum):
    """Signal types for trading strategies."""
    HOLD = 0
    BUY = 1
    SELL = -1

class StrategyType(Enum):
    """Available strategy types."""
    THRESHOLD = 'threshold'
    MEAN_REVERSION = 'mean_reversion'
    MOMENTUM = 'momentum'
    ML_PREDICTION = 'ml_prediction'
    COMBINATION = 'combination'

def generate_signals(
    df: pd.DataFrame,
    model_pred_col: str = "Predicted",
    strategy_type: Union[str, StrategyType] = StrategyType.THRESHOLD,
    threshold: float = 0.001,
    lookback: int = 20,
    zscore_threshold: float = 2.0,
    momentum_window: int = 10,
    **kwargs
) -> pd.DataFrame:
    """
    Generate trading signals based on the specified strategy.
    
    Args:
        df: DataFrame containing price data and optionally predictions
        model_pred_col: Column name for model predictions (used with ML_PREDICTION strategy)
        strategy_type: Type of strategy to use (from StrategyType)
        threshold: Minimum return threshold for signal generation (used with THRESHOLD strategy)
        lookback: Lookback period for mean reversion and momentum strategies
        zscore_threshold: Z-score threshold for mean reversion strategy
        momentum_window: Window size for momentum calculation
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        DataFrame with added 'Signal' column (+1 for buy, -1 for sell, 0 for hold)
    """
    if isinstance(strategy_type, str):
        strategy_type = StrategyType(strategy_type.lower())
    
    df = df.copy()
    
    if strategy_type == StrategyType.THRESHOLD:
        return _threshold_strategy(df, model_pred_col, threshold)
    elif strategy_type == StrategyType.MEAN_REVERSION:
        return _mean_reversion_strategy(df, lookback, zscore_threshold, **kwargs)
    elif strategy_type == StrategyType.MOMENTUM:
        return _momentum_strategy(df, momentum_window, **kwargs)
    elif strategy_type == StrategyType.ML_PREDICTION:
        return _ml_prediction_strategy(df, model_pred_col, **kwargs)
    elif strategy_type == StrategyType.COMBINATION:
        return _combination_strategy(df, model_pred_col, **kwargs)
    else:
        raise ValueError(f"Unsupported strategy type: {strategy_type}")

def _threshold_strategy(
    df: pd.DataFrame,
    model_pred_col: str,
    threshold: float = 0.001
) -> pd.DataFrame:
    """
    Generate signals based on a simple threshold rule.
    
    Args:
        df: Input DataFrame with price and prediction data
        model_pred_col: Column name for model predictions
        threshold: Minimum return threshold to generate a signal
        
    Returns:
        DataFrame with added 'Signal' column
    """
    df = df.copy()
    df["Pred_Return"] = df[model_pred_col].pct_change()
    df["Signal"] = SignalType.HOLD.value
    
    # Generate signals based on threshold
    df.loc[df["Pred_Return"] > threshold, "Signal"] = SignalType.BUY.value
    df.loc[df["Pred_Return"] < -threshold, "Signal"] = SignalType.SELL.value
    
    return df

def _mean_reversion_strategy(
    df: pd.DataFrame,
    lookback: int = 20,
    zscore_threshold: float = 2.0,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Generate signals based on mean reversion (Bollinger Bands).
    
    Args:
        df: Input DataFrame with price data
        lookback: Lookback period for moving average and standard deviation
        zscore_threshold: Z-score threshold for generating signals
        price_col: Column name for price data
        
    Returns:
        DataFrame with added 'Signal' column
    """
    df = df.copy()
    
    # Calculate rolling mean and standard deviation
    df['MA'] = df[price_col].rolling(window=lookback).mean()
    df['STD'] = df[price_col].rolling(window=lookback).std()
    
    # Calculate z-score
    df['Z-Score'] = (df[price_col] - df['MA']) / df['STD']
    
    # Generate signals
    df['Signal'] = SignalType.HOLD.value
    df.loc[df['Z-Score'] < -zscore_threshold, 'Signal'] = SignalType.BUY.value
    df.loc[df['Z-Score'] > zscore_threshold, 'Signal'] = SignalType.SELL.value
    
    return df

def _momentum_strategy(
    df: pd.DataFrame,
    momentum_window: int = 10,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Generate signals based on momentum strategy.
    
    Args:
        df: Input DataFrame with price data
        momentum_window: Window size for momentum calculation
        price_col: Column name for price data
        
    Returns:
        DataFrame with added 'Signal' column
    """
    df = df.copy()
    
    # Calculate momentum (rate of change)
    df['Momentum'] = df[price_col].pct_change(periods=momentum_window)
    
    # Generate signals
    df['Signal'] = SignalType.HOLD.value
    df.loc[df['Momentum'] > 0, 'Signal'] = SignalType.BUY.value
    df.loc[df['Momentum'] < 0, 'Signal'] = SignalType.SELL.value
    
    return df

def _ml_prediction_strategy(
    df: pd.DataFrame,
    model_pred_col: str,
    confidence_threshold: float = 0.7,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Generate signals based on machine learning model predictions.
    
    Args:
        df: Input DataFrame with model predictions
        model_pred_col: Column name for model predictions
        confidence_threshold: Minimum confidence threshold for taking action
        price_col: Column name for price data
        
    Returns:
        DataFrame with added 'Signal' column
    """
    df = df.copy()
    
    # Calculate prediction direction and confidence
    df['Pred_Return'] = df[model_pred_col].pct_change()
    df['Confidence'] = df[model_pred_col].rolling(window=5).std()  # Simple volatility as confidence proxy
    
    # Generate signals
    df['Signal'] = SignalType.HOLD.value
    
    # Only take signals when confidence is above threshold
    high_confidence = df['Confidence'] > confidence_threshold * df['Confidence'].mean()
    df.loc[high_confidence & (df['Pred_Return'] > 0), 'Signal'] = SignalType.BUY.value
    df.loc[high_confidence & (df['Pred_Return'] < 0), 'Signal'] = SignalType.SELL.value
    
    return df

def _combination_strategy(
    df: pd.DataFrame,
    model_pred_col: str,
    strategies: Optional[List[Dict]] = None,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Combine multiple strategies with voting mechanism.
    
    Args:
        df: Input DataFrame with price and prediction data
        model_pred_col: Column name for model predictions
        strategies: List of strategy configurations to combine
        price_col: Column name for price data
        
    Returns:
        DataFrame with added 'Signal' column
    """
    if strategies is None:
        strategies = [
            {'type': StrategyType.THRESHOLD, 'weight': 1.0},
            {'type': StrategyType.MEAN_REVERSION, 'weight': 1.0},
            {'type': StrategyType.ML_PREDICTION, 'weight': 2.0}
        ]
    
    # Initialize signal columns
    df['Signal_Sum'] = 0.0
    total_weight = 0.0
    
    # Apply each strategy and accumulate signals
    for i, strat in enumerate(strategies):
        strat_type = StrategyType(strat['type']) if isinstance(strat['type'], str) else strat['type']
        weight = float(strat.get('weight', 1.0))
        
        # Generate signals for this strategy
        strat_df = generate_signals(
            df.copy(),
            model_pred_col=model_pred_col,
            strategy_type=strat_type,
            **{k: v for k, v in strat.items() if k not in ['type', 'weight']}
        )
        
        # Add weighted signals to the sum
        df['Signal_Sum'] += strat_df['Signal'] * weight
        total_weight += weight
    
    # Calculate weighted average signal
    if total_weight > 0:
        df['Signal'] = np.sign(df['Signal_Sum'] / total_weight).astype(int)
    else:
        df['Signal'] = SignalType.HOLD.value
    
    return df

def filter_signals(
    df: pd.DataFrame,
    min_holding_period: int = 1,
    max_position_size: float = 0.1,
    volatility_threshold: Optional[float] = None,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Apply filters to trading signals to improve quality.
    
    Args:
        df: Input DataFrame with signals
        min_holding_period: Minimum bars to hold a position
        max_position_size: Maximum position size as fraction of portfolio
        volatility_threshold: Maximum volatility (as standard deviation) to take a position
        price_col: Column name for price data
        
    Returns:
        DataFrame with filtered signals
    """
    df = df.copy()
    
    # Initialize filtered signal column
    df['Filtered_Signal'] = SignalType.HOLD.value
    
    # Apply minimum holding period
    if min_holding_period > 1:
        # Only keep signals that are different from the previous signal
        signal_changes = df['Signal'] != df['Signal'].shift(1)
        df['Signal_Group'] = signal_changes.cumsum()
        
        # Count positions in each group
        position_counts = df.groupby('Signal_Group').cumcount() + 1
        
        # Only keep signals where we've held for at least min_holding_period
        df['Filtered_Signal'] = df['Signal'].where(position_counts >= min_holding_period, SignalType.HOLD.value)
    
    # Apply volatility filter
    if volatility_threshold is not None:
        # Calculate rolling volatility
        df['Volatility'] = df[price_col].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Only take signals when volatility is below threshold
        high_vol = df['Volatility'] > volatility_threshold
        df.loc[high_vol, 'Filtered_Signal'] = SignalType.HOLD.value
    
    # Apply position size limits (if implemented in your backtester)
    # This is a placeholder - actual position sizing should be handled in the backtester
    
    return df

def optimize_strategy_parameters(
    df: pd.DataFrame,
    strategy_type: Union[str, StrategyType],
    parameter_grid: Dict[str, List[float]],
    metric: str = 'sharpe_ratio',
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    price_col: str = "Close"
) -> Dict:
    """
    Optimize strategy parameters using grid search.
    
    Args:
        df: Input DataFrame with price data
        strategy_type: Type of strategy to optimize
        parameter_grid: Dictionary of parameter names and lists of values to try
        metric: Performance metric to optimize
        initial_capital: Initial capital for backtest
        transaction_cost: Transaction cost as a fraction
        price_col: Column name for price data
        
    Returns:
        Dictionary with best parameters and performance metrics
    """
    from itertools import product
    from .backtester import Backtester
    
    best_metrics = {metric: -np.inf}
    best_params = {}
    
    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = list(product(*parameter_grid.values()))
    
    for params in param_values:
        # Create parameter dictionary
        param_dict = dict(zip(param_names, params))
        
        try:
            # Generate signals with current parameters
            signals = generate_signals(
                df.copy(),
                strategy_type=strategy_type,
                price_col=price_col,
                **param_dict
            )
            
            # Run backtest
            backtester = Backtester(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost
            )
            
            # Add signals to a copy of the dataframe
            df_test = df.copy()
            df_test['Signal'] = signals['Signal']
            
            # Run backtest
            _, metrics = backtester.backtest(df_test, model_pred_col=price_col)
            
            # Update best parameters if current is better
            if metrics[metric] > best_metrics[metric]:
                best_metrics = metrics
                best_params = param_dict
                
        except Exception as e:
            print(f"Error with parameters {param_dict}: {str(e)}")
            continue
    
    return {
        'best_parameters': best_params,
        'best_metrics': best_metrics
    }
