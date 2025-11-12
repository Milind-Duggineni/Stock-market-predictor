# features/factor_library.py
import pandas as pd

def momentum(df: pd.DataFrame, column="Close", window=5) -> pd.Series:
    """
    Momentum factor: current price / price n periods ago - 1
    """
    return df[column].pct_change(window)

def mean_reversion(df: pd.DataFrame, column="Close", window=5) -> pd.Series:
    """
    Mean reversion factor: rolling mean minus current price
    """
    return df[column] - df[column].rolling(window).mean()

def volatility(df: pd.DataFrame, column="Close", window=10) -> pd.Series:
    """
    Rolling standard deviation as a volatility factor
    """
    return df[column].rolling(window).std()

def bollinger_bands(df: pd.DataFrame, column="Close", window=20, n_std=2):
    """
    Bollinger Bands: returns upper, lower, and middle bands
    """
    sma = df[column].rolling(window).mean()
    std = df[column].rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    return upper, lower, sma

def exponential_moving_average(df: pd.DataFrame, column="Close", span=20):
    return df[column].ewm(span=span, adjust=False).mean()

def relative_strength_index(df: pd.DataFrame, column="Close", window=14):
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
