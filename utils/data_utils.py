# utils/data_utils.py
import pandas as pd
import os

def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def read_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)

def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def merge_dataframes(dfs: list, on: str = "Date") -> pd.DataFrame:
    """Merge multiple dataframes on a key column"""
    from functools import reduce
    return reduce(lambda left, right: pd.merge(left, right, on=on, how="outer"), dfs)

def check_missing(df: pd.DataFrame):
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing)
    return missing

# Example usage
if __name__ == "__main__":
    df1 = pd.DataFrame({"Date": ["2020-01-01"], "Price": [100]})
    df2 = pd.DataFrame({"Date": ["2020-01-01"], "Volume": [1000]})
    merged = merge_dataframes([df1, df2])
    print(merged)
