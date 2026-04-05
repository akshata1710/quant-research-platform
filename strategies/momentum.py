import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from strategies.base import Strategy

class Momentum(Strategy):
    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["returns"] = data["Close"].pct_change(self.window)
        data["signal"] = 0
        data.loc[data["returns"] > 0, "signal"] = 1   # price trended up → buy
        data.loc[data["returns"] < 0, "signal"] = -1  # price trended down → sell
        return data

if __name__ == "__main__":
    from ingestion.fetch_market_data import load
    df = load("AAPL")
    strategy = Momentum(window=20)
    result = strategy.generate_signals(df)
    print(result[["Close", "returns", "signal"]].dropna().tail(10))