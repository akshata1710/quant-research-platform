import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.fetch_market_data import load
from strategies.mean_reversion import MeanReversion
from backtesting.engine import backtest
from backtesting.metrics import compute_metrics

def run_experiment(ticker: str, window: int, cash: float = 10000.0) -> dict:
    df = load(ticker)
    df = MeanReversion(window=window).generate_signals(df)
    df = df.dropna()

    result = backtest(df, cash=cash)
    metrics = compute_metrics(result["equity_curve"])

    return {
        "ticker":        ticker,
        "window":        window,
        "initial_value": result["initial_value"],
        "final_value":   result["final_value"],
        "return_pct":    result["return_pct"],
        **metrics,
    }

if __name__ == "__main__":
    import pandas as pd

    tickers = ["AAPL", "MSFT", "SPY"]
    windows = [10, 20, 30, 50]

    results = []
    for ticker in tickers:
        for window in windows:
            r = run_experiment(ticker, window)
            print(f"{ticker} | window={window:>2} | return={r['return_pct']:>7}% | sharpe={r['sharpe_ratio']}")
            results.append(r)

    df = pd.DataFrame(results)
    df.to_csv("experiments/results.csv", index=False)
    print(f"\nSaved {len(df)} experiments → experiments/results.csv")