# utils/plotting.py
import matplotlib.pyplot as plt

def plot_equity_curve(dates, equity, title="Equity Curve"):
    plt.figure(figsize=(12,6))
    plt.plot(dates, equity, label="Equity")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_indicator(dates, price, indicator, indicator_name="Indicator"):
    plt.figure(figsize=(12,6))
    plt.plot(dates, price, label="Price")
    plt.plot(dates, indicator, label=indicator_name)
    plt.title(f"Price vs {indicator_name}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(df, title="Correlation Matrix"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title)
    plt.show()

# Example usage
if __name__ == "__main__":
    import pandas as pd
    dates = pd.date_range("2020-01-01", periods=50)
    prices = list(range(50))
    indicator = [x*0.9 for x in range(50)]
    plot_indicator(dates, prices, indicator, "Test Indicator")
