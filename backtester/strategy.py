# strategy.py
import pandas as pd
import numpy as np

def generate_signals(df, model_pred_col="Predicted", threshold=0.001):
    """
    df: dataframe with columns 'Close' and model predictions
    threshold: minimum predicted return to act
    Returns: dataframe with signals (+1 = buy, -1 = sell, 0 = hold)
    """
    df = df.copy()
    df["Pred_Return"] = df[model_pred_col].pct_change()
    df["Signal"] = 0
    df.loc[df["Pred_Return"] > threshold, "Signal"] = 1
    df.loc[df["Pred_Return"] < -threshold, "Signal"] = -1
    return df
