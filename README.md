\# QuantDev Simulator



\## Project Overview

QuantDev Simulator is a modular quantitative research and trading simulator. 

It allows for:

\- Historical and live market data ingestion

\- Feature engineering and alpha factor generation

\- ML/AI modeling for predictive signals

\- Backtesting with realistic trading logic

\- Portfolio risk management

\- Optional paper/live trading via Alpaca API

\- Dashboard visualization with Streamlit



\## Architecture

QuantDev/

│

├── 📄 README.md                  # Overview, setup guide, usage examples

├── 📄 requirements.txt           # All dependencies (pandas, numpy, sklearn, tensorflow, xgboost, etc.)

├── 📄 config.yaml                # Global config (tickers, dates, API keys, model params)

├── 📄 .env                       # Sensitive keys (Alpaca, Polygon.io)

├── 📄 .gitignore                 # Ignore cache, logs, and env files

│

├── 📂 data/                      # Raw and processed market data

│   ├── raw/                      # Direct downloads from APIs (e.g., Yahoo Finance)

│   ├── processed/                # Cleaned, merged, and feature-rich data

│   └── cache/                    # Cached datasets (Parquet/Feather for speed)

│

├── 📂 models/                    # All ML/AI models

│   ├── trained/                  # Serialized trained models (.pkl, .h5)

│   ├── checkpoints/              # Partial training checkpoints

│   └── scripts/                  # Scripts to train and test models

│       └── model\_trainer.py

│

├── 📂 features/                  # Feature generation logic

│   ├── feature\_generator.py      # Generates indicators, rolling stats, etc.

│   ├── factor\_library.py         # Custom alpha factors / factor models

│   └── feature\_config.yaml       # Defines which features to compute

│

├── 📂 backtester/                # Simulation and backtest engine

│   ├── backtester.py             # Core backtesting logic

│   ├── strategy.py               # Defines trading strategies (rules-based or ML-driven)

│   ├── metrics.py                # Sharpe, Sortino, Max Drawdown, etc.

│   └── reports/                  # Backtest logs, performance reports

│

├── 📂 risk/                      # Portfolio risk management

│   ├── risk\_manager.py           # VaR, volatility, beta, correlation tracking

│   ├── portfolio\_optimizer.py    # Mean-variance optimization, rebalancing

│   └── reports/                  # Daily/weekly risk reports

│

├── 📂 execution/                 # (Optional) Live or paper trading

│   ├── execution\_engine.py       # Converts model signals → API orders

│   ├── broker\_api.py             # Alpaca or Interactive Brokers integration

│   └── order\_log.csv             # Record of executed/paper trades

│

├── 📂 dashboard/                 # Streamlit or React-based analytics dashboard

│   ├── dashboard.py              # Portfolio visualization, live metrics

│   ├── components/               # Modular Streamlit/React widgets

│   └── assets/                   # Icons, plots, or CSS

│

├── 📂 utils/                     # Shared utilities and helper functions

│   ├── logger.py                 # Logging system

│   ├── timer.py                  # Benchmark decorators

│   ├── data\_utils.py             # Shared data helpers

│   └── plotting.py               # Visualization utilities

│

├── 📂 tests/                     # Unit \& integration tests

│   ├── test\_data\_pipeline.py

│   ├── test\_feature\_generator.py

│   ├── test\_backtester.py

│   └── \_\_init\_\_.py

│

└── 📂 notebooks/                 # Jupyter notebooks for exploration

&nbsp;   ├── exploratory\_analysis.ipynb

&nbsp;   ├── feature\_research.ipynb

&nbsp;   └── model\_validation.ipynb

\## Quick Start



\### 1. Setup Environment

```bash

python -m venv venv

source venv/bin/activate  # Mac/Linux

venv\\Scripts\\activate     # Windows

pip install -r requirements.txt

2\. Configure API Keys



Edit .env with Alpaca and Polygon API keys.



3\. Run Data Pipeline

python data/data\_pipeline.py



4\. Train Models

python models/scripts/model\_trainer.py



5\. Run Backtest

python backtester/backtester.py



6\. Launch Dashboard

streamlit run dashboard/dashboard.py







