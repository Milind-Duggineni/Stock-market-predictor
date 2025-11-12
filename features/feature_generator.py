# features/feature_generator.py
import os
import pandas as pd
import yaml
from utils.data_utils import read_csv, save_csv
from utils.logger import logger
from .factor_library import (  # <-- relative import
    momentum, mean_reversion, volatility,
    bollinger_bands, exponential_moving_average,
    relative_strength_index
)

# Load configuration
with open(os.path.join(os.path.dirname(__file__), "feature_config.yaml"), "r") as f:
    feature_config = yaml.safe_load(f)

PROCESSED_DIR = os.path.join("data", "processed")
FEATURE_DIR = os.path.join("features", "generated")
os.makedirs(FEATURE_DIR, exist_ok=True)

def generate_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute all features defined in config
    """
    df = df.copy()
    
    for feature_name, params in config.items():
        if feature_name == "momentum":
            df[f"momentum_{params['window']}"] = momentum(df, params["column"], params["window"])
        elif feature_name == "mean_reversion":
            df[f"mean_rev_{params['window']}"] = mean_reversion(df, params["column"], params["window"])
        elif feature_name == "volatility":
            df[f"vol_{params['window']}"] = volatility(df, params["column"], params["window"])
        elif feature_name == "bollinger":
            upper, lower, sma = bollinger_bands(df, params["column"], params["window"], params.get("n_std", 2))
            df[f"bb_upper_{params['window']}"] = upper
            df[f"bb_lower_{params['window']}"] = lower
            df[f"bb_mid_{params['window']}"] = sma
        elif feature_name == "ema":
            df[f"ema_{params['span']}"] = exponential_moving_average(df, params["column"], params["span"])
        elif feature_name == "rsi":
            df[f"rsi_{params['window']}"] = relative_strength_index(df, params["column"], params["window"])
        else:
            logger.warning(f"Feature {feature_name} not recognized")
    
    return df

def process_ticker(ticker: str):
    path = os.path.join(PROCESSED_DIR, f"{ticker}_train.csv")
    df = read_csv(path)
    df_features = generate_features(df, feature_config)
    save_path = os.path.join(FEATURE_DIR, f"{ticker}_features.csv")
    save_csv(df_features, save_path)
    logger.info(f"Features generated for {ticker} at {save_path}")

def main():
    tickers = [f.split("_train")[0] for f in os.listdir(PROCESSED_DIR) if "_train" in f]
    for ticker in tickers:
        process_ticker(ticker)

if __name__ == "__main__":
    main()
