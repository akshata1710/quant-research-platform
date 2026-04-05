import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
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

def run_experiment(ticker: str, strategy_name: str, window: int, cash: float = 10000.0) -> dict:
    df = load(ticker)
    strategy = STRATEGIES[strategy_name](window)
    df = strategy.generate_signals(df)
    df = df.dropna()

    result  = backtest(df, cash=cash)
    metrics = compute_metrics(result["equity_curve"])

    return {
        "ticker":         ticker,
        "strategy":       strategy_name,
        "window":         window,
        "initial_value":  result["initial_value"],
        "final_value":    result["final_value"],
        "return_pct":     result["return_pct"],
        "stop_loss_hits": result["stop_loss_hits"],
        "total_trades":   result["total_trades"],
        **metrics,
    }

if __name__ == "__main__":
    tickers   = ["AAPL", "MSFT", "SPY"]
    windows   = [10, 20, 30]
    results   = []

    for ticker in tickers:
        for strategy_name in STRATEGIES:
            for window in windows:
                r = run_experiment(ticker, strategy_name, window)
                print(f"{ticker} | {strategy_name:<16} | w={window} | return={r['return_pct']:>7}% | sharpe={r['sharpe_ratio']}")
                results.append(r)

    df = pd.DataFrame(results)
    df.to_csv("experiments/results.csv", index=False)

    print(f"\nTop 10 by Sharpe ratio:")
    print(df[["ticker","strategy","window","return_pct","sharpe_ratio","max_drawdown"]]
          .sort_values("sharpe_ratio", ascending=False)
          .head(10)
          .to_string(index=False))

    print(f"\nBest strategy per ticker:")
    print(df.loc[df.groupby("ticker")["sharpe_ratio"].idxmax()]
            [["ticker","strategy","window","return_pct","sharpe_ratio"]]
            .to_string(index=False))

    print(f"\nSaved {len(df)} experiments → experiments/results.csv")