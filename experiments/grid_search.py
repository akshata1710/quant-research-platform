import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from multiprocessing import Pool
from experiments.run_experiment import run_experiment

TICKERS = ["AAPL", "MSFT", "SPY"]
WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60]

def run_one(args):
    ticker, window = args
    try:
        result = run_experiment(ticker, window)
        print(f"  {ticker} window={window:>2} | return={result['return_pct']:>7}% | sharpe={result['sharpe_ratio']}")
        return result
    except Exception as e:
        print(f"  Skipping {ticker} window={window} — {e}")
        return None

if __name__ == "__main__":
    combos = [(t, w) for t in TICKERS for w in WINDOWS]
    print(f"Running {len(combos)} experiments across {len(TICKERS)} tickers...\n")

    with Pool() as pool:
        results = pool.map(run_one, combos)

    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    df.sort_values("sharpe_ratio", ascending=False, inplace=True)
    df.to_csv("experiments/results_grid.csv", index=False)

    print(f"\nTop 5 by Sharpe ratio:")
    print(df[["ticker","window","return_pct","sharpe_ratio","max_drawdown"]].head())
    print(f"\nSaved {len(df)} results → experiments/results_grid.csv")