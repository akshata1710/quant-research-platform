import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

def compute_metrics(equity_curve: list) -> dict:
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_drawdown = drawdowns.min()

    positive = returns[returns > 0].sum()
    negative = abs(returns[returns < 0].sum())
    profit_factor = positive / negative if negative > 0 else float("inf")

    return {
        "sharpe_ratio":  round(sharpe, 3),
        "max_drawdown":  round(max_drawdown * 100, 2),
        "volatility":    round(np.std(returns) * np.sqrt(252) * 100, 2),
        "profit_factor": round(profit_factor, 3),
    }

if __name__ == "__main__":
    from ingestion.fetch_market_data import load
    from strategies.mean_reversion import MeanReversion
    from backtesting.engine import backtest

    df = load("AAPL")
    df = MeanReversion(window=20).generate_signals(df)
    df = df.dropna()

    result = backtest(df)
    metrics = compute_metrics(result["equity_curve"])

    print(f"Sharpe ratio:  {metrics['sharpe_ratio']}")
    print(f"Max drawdown:  {metrics['max_drawdown']}%")
    print(f"Volatility:    {metrics['volatility']}%")
    print(f"Profit factor: {metrics['profit_factor']}")