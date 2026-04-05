import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from ingestion.fetch_market_data import load
from backtesting.engine import backtest
from backtesting.metrics import compute_metrics
from strategies.regime import detect_regime, filter_signals_by_regime

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

def run_experiment(ticker: str, strategy_name: str,
                   window: int, cash: float = 10000.0,
                   use_regime_filter: bool = False) -> dict:
    df = load(ticker)
    strategy = STRATEGIES[strategy_name](window)
    df = strategy.generate_signals(df)

    if use_regime_filter:
        df = detect_regime(df)
        df = filter_signals_by_regime(df, strategy_name)

    df = df.dropna()
    result  = backtest(df, cash=cash)
    metrics = compute_metrics(result["equity_curve"])

    return {
        "ticker":         ticker,
        "strategy":       strategy_name,
        "window":         window,
        "regime_filter":  use_regime_filter,
        "final_value":    result["final_value"],
        "return_pct":     result["return_pct"],
        "stop_loss_hits": result["stop_loss_hits"],
        "total_trades":   result["total_trades"],
        **metrics,
    }

if __name__ == "__main__":
    tickers  = ["AAPL", "MSFT", "SPY"]
    windows  = [10, 20, 30]
    results  = []

    print(f"{'Ticker':<6} {'Strategy':<18} {'W':>3} {'Filter':<8} {'Return':>8} {'Sharpe':>8}")
    print("-" * 60)

    for ticker in tickers:
        for strategy_name in STRATEGIES:
            for window in windows:
                for use_filter in [False, True]:
                    r = run_experiment(ticker, strategy_name, window, use_regime_filter=use_filter)
                    label = "yes" if use_filter else "no"
                    print(f"{ticker:<6} {strategy_name:<18} {window:>3} {label:<8} {r['return_pct']:>7}% {r['sharpe_ratio']:>8}")
                    results.append(r)

    df = pd.DataFrame(results)
    df.to_csv("experiments/results.csv", index=False)

    print(f"\nRegime filter impact (avg across all experiments):")
    no_filter  = df[df["regime_filter"] == False]
    yes_filter = df[df["regime_filter"] == True]
    print(f"  Without filter — avg return: {no_filter['return_pct'].mean():.2f}%  avg Sharpe: {no_filter['sharpe_ratio'].mean():.3f}")
    print(f"  With filter    — avg return: {yes_filter['return_pct'].mean():.2f}%  avg Sharpe: {yes_filter['sharpe_ratio'].mean():.3f}")

    print(f"\nTop 5 with regime filter:")
    top = df[df["regime_filter"] == True].sort_values("sharpe_ratio", ascending=False).head()
    print(top[["ticker","strategy","window","return_pct","sharpe_ratio","max_drawdown"]].to_string(index=False))

    print(f"\nSaved {len(df)} experiments → experiments/results.csv")