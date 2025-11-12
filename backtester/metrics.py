# metrics.py
import numpy as np

def sharpe_ratio(returns, risk_free=0.0):
    excess_ret = returns - risk_free
    return np.mean(excess_ret) / (np.std(excess_ret) + 1e-8) * np.sqrt(252)

def sortino_ratio(returns, risk_free=0.0):
    excess_ret = returns - risk_free
    downside = returns[returns < 0]
    return np.mean(excess_ret) / (np.std(downside) + 1e-8) * np.sqrt(252)

def max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)

def cagr(equity_curve, periods_per_year=252):
    n_periods = len(equity_curve)
    start = equity_curve[0]
    end = equity_curve[-1]
    return (end / start) ** (periods_per_year / n_periods) - 1
