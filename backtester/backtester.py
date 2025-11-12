# backtester.py
import pandas as pd
import numpy as np
from strategy import generate_signals
from metrics import sharpe_ratio, sortino_ratio, max_drawdown, cagr

def backtest(df, model_pred_col="Predicted", initial_capital=100000, transaction_cost=0.001):
    df = generate_signals(df, model_pred_col=model_pred_col)
    df["Position"] = df["Signal"].shift(1).fillna(0)
    df["Returns"] = df["Close"].pct_change() * df["Position"]
    df["Returns"] -= np.abs(df["Position"].diff()) * transaction_cost
    df["Equity"] = initial_capital * (1 + df["Returns"]).cumprod()

    metrics = {
        "Sharpe": sharpe_ratio(df["Returns"].fillna(0)),
        "Sortino": sortino_ratio(df["Returns"].fillna(0)),
        "MaxDrawdown": max_drawdown(df["Equity"].values),
        "CAGR": cagr(df["Equity"].values)
    }

    return df, metrics

if __name__ == "__main__":
    tickers = ["AAPL","SPY","TSLA"]
    for ticker in tickers:
        df_test = pd.read_csv(f"data/processed/test/{ticker}_test.csv")
        # Assume 'Predicted' column exists from model_evaluator output
        df_test["Predicted"] = pd.read_csv(f"reports/plots/{ticker}_XGBoost_pred.csv")["Predicted"].values
        df_res, metrics = backtest(df_test)
        print(f"\nBacktest metrics for {ticker}: {metrics}")
        df_res.to_csv(f"reports/plots/{ticker}_backtest.csv", index=False)
