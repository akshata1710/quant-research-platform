import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from ingestion.fetch_market_data import load
from backtesting.engine import backtest
from backtesting.metrics import compute_metrics

from strategies.mean_reversion  import MeanReversion
from strategies.momentum        import Momentum
from strategies.rsi             import RSI
from strategies.bollinger_bands import BollingerBands

STRATEGIES = {
    "mean_reversion":  lambda w: MeanReversion(window=w),
    "momentum":        lambda w: Momentum(window=w),
    "rsi":             lambda w: RSI(window=w),
    "bollinger_bands": lambda w: BollingerBands(window=w),
}

WINDOWS = [10, 20, 30, 50]

def find_best_window(data: pd.DataFrame, strategy_name: str) -> int:
    """On training data, find the window with the best Sharpe ratio."""
    best_sharpe = -999
    best_window = WINDOWS[0]

    for window in WINDOWS:
        try:
            df = STRATEGIES[strategy_name](window).generate_signals(data).dropna()
            if len(df) < 30:
                continue
            result  = backtest(df)
            metrics = compute_metrics(result["equity_curve"])
            if metrics["sharpe_ratio"] > best_sharpe:
                best_sharpe = metrics["sharpe_ratio"]
                best_window = window
        except Exception:
            continue

    return best_window


def walk_forward(ticker: str, strategy_name: str,
                 train_years: int = 3,
                 test_years: int  = 1,
                 cash: float      = 10000.0) -> dict:
    """
    Split data into rolling train/test windows.
    Train: find best window parameter
    Test:  run strategy with that window on unseen data
    """
    df = load(ticker)
    df.index = pd.to_datetime(df.index)

    start_year = df.index.year.min()
    end_year   = df.index.year.max()

    all_test_equity = []
    folds = []

    year = start_year
    while year + train_years + test_years <= end_year + 1:
        train_start = f"{year}-01-01"
        train_end   = f"{year + train_years}-01-01"
        test_start  = f"{year + train_years}-01-01"
        test_end    = f"{year + train_years + test_years}-01-01"

        train_data = df[(df.index >= train_start) & (df.index < train_end)]
        test_data  = df[(df.index >= test_start)  & (df.index < test_end)]

        if len(train_data) < 60 or len(test_data) < 20:
            year += test_years
            continue

        # Find best window on TRAINING data only
        best_window = find_best_window(train_data, strategy_name)

        # Run on TEST data with best window — unseen data
        test_signals = STRATEGIES[strategy_name](best_window).generate_signals(test_data).dropna()
        if len(test_signals) < 10:
            year += test_years
            continue

        result  = backtest(test_signals, cash=cash)
        metrics = compute_metrics(result["equity_curve"])

        fold = {
            "train_period": f"{train_start} → {train_end}",
            "test_period":  f"{test_start} → {test_end}",
            "best_window":  best_window,
            "return_pct":   result["return_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
        }
        folds.append(fold)
        all_test_equity.extend(result["equity_curve"])

        year += test_years

    if not folds:
        return {}

    folds_df = pd.DataFrame(folds)
    return {
        "ticker":          ticker,
        "strategy":        strategy_name,
        "folds":           folds_df,
        "avg_return":      round(folds_df["return_pct"].mean(), 2),
        "avg_sharpe":      round(folds_df["sharpe_ratio"].mean(), 3),
        "avg_drawdown":    round(folds_df["max_drawdown"].mean(), 2),
        "equity_curve":    all_test_equity,
    }


if __name__ == "__main__":
    tickers   = ["AAPL", "MSFT", "SPY"]
    all_rows  = []

    for ticker in tickers:
        for strategy_name in STRATEGIES:
            r = walk_forward(ticker, strategy_name, train_years=3, test_years=1)
            if not r:
                continue

            print(f"\n{ticker} | {strategy_name}")
            print(r["folds"][["test_period","best_window","return_pct","sharpe_ratio"]].to_string(index=False))
            print(f"  Avg return: {r['avg_return']}%  |  Avg Sharpe: {r['avg_sharpe']}  |  Avg drawdown: {r['avg_drawdown']}%")

            all_rows.append({
                "ticker":       ticker,
                "strategy":     strategy_name,
                "avg_return":   r["avg_return"],
                "avg_sharpe":   r["avg_sharpe"],
                "avg_drawdown": r["avg_drawdown"],
            })

    summary = pd.DataFrame(all_rows)
    summary.to_csv("experiments/walk_forward_results.csv", index=False)

    print(f"\n{'='*60}")
    print("Top 5 strategies by avg Sharpe (walk-forward validated):")
    print(summary.sort_values("avg_sharpe", ascending=False).head().to_string(index=False))
    print(f"\nSaved → experiments/walk_forward_results.csv")