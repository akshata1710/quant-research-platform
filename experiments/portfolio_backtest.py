import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from backtesting.engine import backtest
from backtesting.metrics import compute_metrics
from ingestion.fetch_market_data import load

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

def portfolio_backtest(tickers: list, strategy_name: str,
                       window: int, total_cash: float = 100000.0) -> dict:
    """
    Run one strategy across multiple tickers simultaneously.
    Capital is split equally across all tickers.
    """
    cash_per_ticker = total_cash / len(tickers)
    results = []

    for ticker in tickers:
        df = load(ticker)
        strategy = STRATEGIES[strategy_name](window)
        df = strategy.generate_signals(df).dropna()
        result = backtest(df, cash=cash_per_ticker)
        metrics = compute_metrics(result["equity_curve"])
        results.append({
            "ticker":       ticker,
            "final_value":  result["final_value"],
            "return_pct":   result["return_pct"],
            "equity_curve": result["equity_curve"],
            **metrics,
        })

    # Combine equity curves into portfolio curve
    min_len  = min(len(r["equity_curve"]) for r in results)
    combined = [sum(r["equity_curve"][i] for r in results) for i in range(min_len)]
    port_metrics = compute_metrics(combined)

    total_final  = sum(r["final_value"] for r in results)
    total_return = round((total_final - total_cash) / total_cash * 100, 2)

    return {
        "strategy":      strategy_name,
        "window":        window,
        "total_cash":    total_cash,
        "total_final":   round(total_final, 2),
        "total_return":  total_return,
        "equity_curve":  combined,
        "per_ticker":    results,
        **port_metrics,
    }

if __name__ == "__main__":
    tickers  = ["AAPL", "MSFT", "SPY"]
    windows  = [10, 20, 30]
    all_results = []

    print(f"{'Strategy':<18} {'Window':>6} {'Return':>8} {'Sharpe':>8} {'Drawdown':>10}")
    print("-" * 56)

    for strategy_name in STRATEGIES:
        for window in windows:
            r = portfolio_backtest(tickers, strategy_name, window, total_cash=100000)
            print(f"{strategy_name:<18} {window:>6} {r['total_return']:>7}% {r['sharpe_ratio']:>8} {r['max_drawdown']:>9}%")
            all_results.append({
                "strategy":     strategy_name,
                "window":       window,
                "total_return": r["total_return"],
                "sharpe_ratio": r["sharpe_ratio"],
                "max_drawdown": r["max_drawdown"],
                "final_value":  r["total_final"],
            })

    df = pd.DataFrame(all_results)
    df.to_csv("experiments/portfolio_results.csv", index=False)

    print(f"\nBest portfolio by Sharpe:")
    best = df.loc[df["sharpe_ratio"].idxmax()]
    print(f"  {best['strategy']} w={int(best['window'])} | return={best['total_return']}% | sharpe={best['sharpe_ratio']}")
    print(f"\nSaved → experiments/portfolio_results.csv")