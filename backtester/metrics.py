"""
Risk and Performance Metrics Module

This module provides various financial metrics for evaluating trading strategies,
including risk-adjusted returns, drawdown analysis, and performance statistics.
"""
import numpy as np
from typing import Union, Tuple

def annualized_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized volatility of returns.
    
    Args:
        returns: Array of periodic returns
        periods_per_year: Number of trading periods per year (default: 252)
        
    Returns:
        Annualized volatility (standard deviation of returns)
    """
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sharpe ratio.
    
    Args:
        returns: Array of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of trading periods per year (default: 252)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / (np.std(excess_returns, ddof=1) + 1e-10)

def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sortino ratio.
    
    Args:
        returns: Array of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of trading periods per year (default: 252)
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')  # No downside risk
        
    downside_std = np.std(downside_returns, ddof=1)
    if downside_std < 1e-10:
        return np.sign(np.mean(excess_returns)) * float('inf')
        
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / (downside_std + 1e-10)

def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate the maximum drawdown of an equity curve.
    
    Args:
        equity_curve: Array of portfolio values over time
        
    Returns:
        Maximum drawdown (as a negative number, e.g., -0.2 for 20% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0
        
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / (peak + 1e-10)
    return np.min(drawdowns)

def cagr(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR) of an equity curve.
    
    Args:
        equity_curve: Array of portfolio values over time
        periods_per_year: Number of trading periods per year (default: 252)
        
    Returns:
        Annualized return (e.g., 0.1 for 10% annual return)
    """
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return 0.0
        
    n_periods = len(equity_curve)
    return (equity_curve[-1] / equity_curve[0]) ** (periods_per_year / n_periods) - 1

def calmar_ratio(equity_curve: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar ratio (CAGR / Max Drawdown).
    
    Args:
        equity_curve: Array of portfolio values over time
        periods_per_year: Number of trading periods per year (default: 252)
        
    Returns:
        Calmar ratio (higher is better)
    """
    cagr_val = cagr(equity_curve, periods_per_year)
    max_dd = abs(max_drawdown(equity_curve))
    return cagr_val / (max_dd + 1e-10)

def value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate the Value at Risk (VaR) at a given confidence level.
    
    Args:
        returns: Array of periodic returns
        confidence_level: Confidence level (default: 0.95 for 95%)
        
    Returns:
        VaR as a positive number (e.g., 0.05 for 5% VaR)
    """
    if len(returns) == 0:
        return 0.0
    return -np.percentile(returns, 100 * (1 - confidence_level))

def expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate the Expected Shortfall (CVaR) at a given confidence level.
    
    Args:
        returns: Array of periodic returns
        confidence_level: Confidence level (default: 0.95 for 95%)
        
    Returns:
        Expected Shortfall as a positive number
    """
    if len(returns) == 0:
        return 0.0
    var = value_at_risk(returns, confidence_level)
    return -np.mean(returns[returns <= -var])

def win_rate(returns: np.ndarray) -> float:
    """
    Calculate the win rate (percentage of positive returns).
    
    Args:
        returns: Array of periodic returns
        
    Returns:
        Win rate between 0 and 1
    """
    if len(returns) == 0:
        return 0.0
    return np.mean(returns > 0)

def profit_factor(returns: np.ndarray) -> float:
    """
    Calculate the profit factor (gross profit / gross loss).
    
    Args:
        returns: Array of periodic returns
        
    Returns:
        Profit factor (values > 1 indicate profitable strategy)
    """
    if len(returns) == 0:
        return 0.0
    gross_profit = np.sum(np.maximum(returns, 0))
    gross_loss = abs(np.sum(np.minimum(returns, 0)))
    return gross_profit / (gross_loss + 1e-10)

def get_all_metrics(returns: np.ndarray, equity_curve: np.ndarray, risk_free_rate: float = 0.0, 
                   periods_per_year: int = 252, confidence_level: float = 0.95) -> dict:
    """
    Calculate and return a dictionary of all available metrics.
    
    Args:
        returns: Array of periodic returns
        equity_curve: Array of portfolio values over time
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of trading periods per year (default: 252)
        confidence_level: Confidence level for VaR and Expected Shortfall (default: 0.95)
        
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {
        'cagr': cagr(equity_curve, periods_per_year),
        'volatility': annualized_volatility(returns, periods_per_year),
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': sortino_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': max_drawdown(equity_curve),
        'calmar_ratio': calmar_ratio(equity_curve, periods_per_year),
        'var_95': value_at_risk(returns, confidence_level),
        'expected_shortfall_95': expected_shortfall(returns, confidence_level),
        'win_rate': win_rate(returns),
        'profit_factor': profit_factor(returns),
        'total_return': equity_curve[-1] / equity_curve[0] - 1 if len(equity_curve) > 0 else 0.0
    }
    
    return metrics
