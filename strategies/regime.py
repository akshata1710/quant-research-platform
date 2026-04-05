import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

def detect_regime(data: pd.DataFrame, 
                  short_window: int = 50,
                  long_window:  int = 200) -> pd.DataFrame:
    """
    Detect market regime using moving average crossover.
    Bull:     short MA > long MA (trend is up)
    Bear:     short MA < long MA (trend is down)
    Sideways: short MA ≈ long MA (within 1% band)
    """
    data = data.copy()
    data["MA_short"] = data["Close"].rolling(short_window).mean()
    data["MA_long"]  = data["Close"].rolling(long_window).mean()

    ratio = (data["MA_short"] - data["MA_long"]) / data["MA_long"]

    data["regime"] = "sideways"
    data.loc[ratio >  0.01, "regime"] = "bull"
    data.loc[ratio < -0.01, "regime"] = "bear"

    return data


def filter_signals_by_regime(data: pd.DataFrame,
                              strategy_name: str) -> pd.DataFrame:
    """
    Apply regime filter to signals.
    Mean reversion + Bollinger: only trade in sideways/bull markets
    Momentum:                   only trade in bull markets
    RSI:                        trade in all regimes
    """
    data = data.copy()

    if strategy_name in ("mean_reversion", "bollinger_bands"):
        # Turn off signals in bear markets
        data.loc[data["regime"] == "bear", "signal"] = 0

    elif strategy_name == "momentum":
        # Only trade in bull markets
        data.loc[data["regime"] != "bull", "signal"] = 0

    return data


if __name__ == "__main__":
    from ingestion.fetch_market_data import load

    df = load("AAPL")
    df = detect_regime(df)

    regime_counts = df["regime"].value_counts()
    print("Regime distribution for AAPL (2018-2024):")
    print(regime_counts)
    print(f"\nBull:     {regime_counts.get('bull', 0)} days")
    print(f"Bear:     {regime_counts.get('bear', 0)} days")
    print(f"Sideways: {regime_counts.get('sideways', 0)} days")

    print("\nRecent regimes:")
    print(df[["Close", "MA_short", "MA_long", "regime"]].dropna().tail(10))