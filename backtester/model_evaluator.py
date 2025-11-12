# model_evaluator.py
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

# ---------------- Config ----------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

tickers = config.get("tickers", ["AAPL", "SPY", "TSLA"])
paths = config.get("paths", {})
test_data_path = paths.get("test_data", "data/processed")
model_path = paths.get("trained_models", "models/trained")
report_path = paths.get("reports", "reports/plots")
os.makedirs(report_path, exist_ok=True)

# ---------------- Helpers ----------------
def load_test_data(ticker):
    """Load test CSV, align features with training columns, and sanitize numeric values."""
    test_file = os.path.join(test_data_path, f"{ticker}_test.csv")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    df_test = pd.read_csv(test_file)

    # Load training feature names
    feature_file = os.path.join(model_path, f"{ticker}_features.pkl")
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    with open(feature_file, "rb") as f:
        feature_names = pickle.load(f)

    # Fill missing columns with 0
    for f in feature_names:
        if f not in df_test.columns:
            print(f"[Warning] {ticker} test data missing feature '{f}', filling with 0")
            df_test[f] = 0.0

    # Select features in the correct order
    X_test = df_test[feature_names].values.astype(np.float32)
    y_test = df_test["Close"].values.astype(np.float32)

    # Replace NaNs or infinite values with 0
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

    return X_test, y_test, df_test

def evaluate_model(ticker):
    """Evaluate both LSTM and XGBoost models for a ticker."""
    try:
        X_test, y_test, df_test = load_test_data(ticker)

        # -------- LSTM --------
        lstm_file = os.path.join(model_path, f"{ticker}_lstm")
        lstm_model = load_model(lstm_file)
        X_test_lstm = np.expand_dims(X_test, axis=1)  # (samples, timesteps=1, features)
        y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()

        # -------- XGBoost --------
        xgb_file = os.path.join(model_path, f"{ticker}_xgb.pkl")
        with open(xgb_file, "rb") as f:
            xgb_model = pickle.load(f)
        y_pred_xgb = xgb_model.predict(X_test)

        # Save XGBoost predictions for backtesting
        df_test["Predicted_XGBoost"] = y_pred_xgb
        df_test.to_csv(os.path.join(report_path, f"{ticker}_predictions.csv"), index=False)

        # -------- Metrics & Plots --------
        metrics = {}
        for name, y_pred in [("LSTM", y_pred_lstm), ("XGBoost", y_pred_xgb)]:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics[name] = {"MSE": mse, "MAE": mae, "R2": r2}

            plt.figure(figsize=(12,5))
            plt.plot(y_test, label="Actual")
            plt.plot(y_pred, label=f"Predicted ({name})")
            plt.title(f"{ticker} Prediction vs Actual ({name})")
            plt.legend()
            plt.savefig(os.path.join(report_path, f"{ticker}_{name}_pred.png"))
            plt.close()

        return metrics

    except Exception as e:
        print(f"Error evaluating {ticker}: {e}")
        return None

# ---------------- Main ----------------
if __name__ == "__main__":
    all_metrics = {}
    for ticker in tickers:
        print(f"\nEvaluating {ticker}...")
        metrics = evaluate_model(ticker)
        if metrics:
            all_metrics[ticker] = metrics
            for model_name, m in metrics.items():
                print(f"{model_name}: MSE={m['MSE']:.6f}, MAE={m['MAE']:.6f}, R2={m['R2']:.4f}")

    if all_metrics:
        metrics_df = pd.DataFrame(
            {(ticker, model): values for ticker, v in all_metrics.items() for model, values in v.items()}
        ).T
        metrics_df.index.names = ["Ticker", "Model"]
        metrics_df.to_csv(os.path.join(report_path, "model_metrics.csv"))
        print(f"\nSaved all metrics and plots to {report_path}")
