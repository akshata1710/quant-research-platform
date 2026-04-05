# Quant Research Platform

A full end-to-end quantitative research platform for downloading market data,
building trading strategies, running backtests, and comparing results.

## What it does

- Downloads historical OHLCV data via yfinance
- Generates trading signals from 4 strategies
- Simulates trades with realistic costs and risk management
- Runs thousands of experiments in parallel
- Validates strategies on unseen data (walk-forward)
- Detects market regimes (bull/bear/sideways)
- Visualises everything in an interactive Streamlit dashboard

## Project structure
```
quant-research-platform/
├── data/
│   ├── raw/                      ← downloaded parquet files
│   └── processed/
├── ingestion/
│   └── fetch_market_data.py      ← yfinance downloader
├── strategies/
│   ├── base.py                   ← abstract Strategy class
│   ├── mean_reversion.py         ← MA crossover signals
│   ├── momentum.py               ← trend-following signals
│   ├── rsi.py                    ← RSI overbought/oversold
│   ├── bollinger_bands.py        ← band breakout signals
│   └── regime.py                 ← bull/bear/sideways detection
├── backtesting/
│   ├── engine.py                 ← trade simulation loop
│   ├── metrics.py                ← Sharpe, drawdown, volatility
│   └── position_sizing.py        ← Kelly criterion, fixed fractional
├── experiments/
│   ├── run_experiment.py         ← single + grid experiments
│   ├── grid_search.py            ← parallel param sweep
│   ├── portfolio_backtest.py     ← multi-ticker portfolio
│   ├── walk_forward.py           ← out-of-sample validation
│   └── results/                  ← saved CSV outputs
├── dashboard/
│   └── app.py                    ← Streamlit dashboard
├── config.py                     ← global paths
├── requirements.txt
└── README.md
```

## Quickstart
```bash
# 1. Clone and set up
git clone https://github.com/akshata1710/quant-research-platform.git
cd quant-research-platform
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download market data
python ingestion/fetch_market_data.py

# 3. Run experiments
python experiments/run_experiment.py
python experiments/portfolio_backtest.py
python experiments/walk_forward.py

# 4. Launch dashboard
streamlit run dashboard/app.py
```

## Strategies

| Strategy | Logic | Best on |
|----------|-------|---------|
| Mean reversion | Buy below MA, sell above | MSFT, SPY |
| Momentum | Buy on uptrend, sell on downtrend | AAPL |
| RSI | Buy oversold (<30), sell overbought (>70) | SPY |
| Bollinger Bands | Buy below lower band, sell above upper | SPY |

## Key findings

- Momentum dominates individual stocks (AAPL, MSFT) in bull markets
- Most strategies fail walk-forward validation — only SPY Bollinger Bands
  and MSFT Mean Reversion achieve avg Sharpe > 1.0 on unseen data
- Regime filtering helps selectively — improves MSFT momentum significantly
  but hurts strategies that were already working well
- 2022 (Fed rate hikes) destroyed every strategy — regime awareness is critical

## Levels completed

| Level | Feature |
|-------|---------|
| 1 | Transaction costs, slippage, lookahead bias fix, stop loss |
| 2 | 4 strategies, Streamlit dashboard |
| 3 | Position sizing, Kelly criterion, trailing stop |
| 4 | Portfolio backtesting with shared capital |
| 5 | Walk-forward validation |
| 6 | Market regime detection and filtering |

## Requirements

- Python 3.10+
- yfinance
- pandas
- pyarrow
- streamlit
- plotly
- numpy
