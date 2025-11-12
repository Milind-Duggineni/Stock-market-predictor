import os
import pandas as pd
import joblib
import yaml
import pickle
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from utils.data_utils import read_csv
from utils.logger import logger

# LSTM imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# XGBoost imports
import xgboost as xgb

# ----------------- Directories -----------------
FEATURE_DIR = "features/generated"
TRAINED_DIR = os.path.join("models", "trained")
CHECKPOINT_DIR = os.path.join("models", "checkpoints")

os.makedirs(TRAINED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------- Config -----------------
CONFIG_PATH = "models/model_config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        model_config = yaml.safe_load(f)
else:
    model_config = {
        "lstm_units": 50,
        "lstm_epochs": 10,
        "lstm_batch": 16,
        "xgb_params": {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
        "tscv_splits": 3,
    }

# ----------------- Helpers -----------------
def load_features(ticker: str):
    """Load and clean features for a specific ticker."""
    path = os.path.join(FEATURE_DIR, f"{ticker}_features.csv")
    df = read_csv(path)

    # Drop NaNs and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Features and target
    feature_cols = [c for c in df.columns if c not in ["Date", "Close"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Close"].values.astype(np.float32)

    # Sanitize numeric arrays
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extremes (optional safety)
    X = np.clip(X, -1e6, 1e6)
    y = np.clip(y, -1e6, 1e6)

    # Diagnostic logging
    logger.info(f"[{ticker}] Feature shape: {X.shape}, Target shape: {y.shape}")
    logger.info(
        f"[{ticker}] Any NaNs? {np.isnan(X).any() or np.isnan(y).any()} | Any infs? {np.isinf(X).any() or np.isinf(y).any()}"
    )
    return X, y, feature_cols

# ----------------- LSTM -----------------
def train_lstm(X, y, ticker):
    """Train and save an LSTM model."""
    X = np.expand_dims(X, axis=1)  # shape: [samples, timesteps=1, features]

    model = Sequential([
        LSTM(model_config["lstm_units"], input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{ticker}_lstm.ckpt")
    checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="loss")

    logger.info(f"Training LSTM for {ticker}")
    model.fit(
        X, y,
        epochs=model_config["lstm_epochs"],
        batch_size=model_config["lstm_batch"],
        callbacks=[checkpoint_cb],
        verbose=0
    )

    model.save(os.path.join(TRAINED_DIR, f"{ticker}_lstm"))
    logger.info(f"LSTM trained and saved for {ticker}")
    return model

# ----------------- XGBoost -----------------
def train_xgboost(X, y, ticker):
    """Train and save an XGBoost model using time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=model_config["tscv_splits"])
    best_model = None
    best_score = float("inf")

    logger.info(f"Training XGBoost for {ticker}")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Ensure data is finite before each fit
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

        model = xgb.XGBRegressor(**model_config["xgb_params"])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        score = mean_squared_error(y_val, y_pred)
        logger.info(f"[{ticker}] Fold {fold+1} MSE: {score:.6f}")

        if score < best_score:
            best_score = score
            best_model = model

    joblib.dump(best_model, os.path.join(TRAINED_DIR, f"{ticker}_xgb.pkl"))
    logger.info(f"XGBoost trained and saved for {ticker} | Best MSE: {best_score:.6f}")
    return best_model

# ----------------- Main -----------------
def main():
    tickers = [
        f.split("_features")[0]
        for f in os.listdir(FEATURE_DIR)
        if "_features" in f
    ]

    for ticker in tickers:
        X, y, feature_cols = load_features(ticker)

        # Save feature names for evaluation
        feature_file = os.path.join(TRAINED_DIR, f"{ticker}_features.pkl")
        with open(feature_file, "wb") as f:
            pickle.dump(feature_cols, f)

        train_lstm(X, y, ticker)
        train_xgboost(X, y, ticker)

if __name__ == "__main__":
    main()
