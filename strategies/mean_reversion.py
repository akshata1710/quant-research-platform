import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from strategies.base import Strategy

class MeanReversion(Strategy):
    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["MA"] = data["Close"].rolling(self.window).mean()
        data["signal"] = 0
        data.loc[data["Close"] < data["MA"], "signal"] = 1
        data.loc[data["Close"] > data["MA"], "signal"] = -1
        return data

if __name__ == "__main__":
    from ingestion.fetch_market_data import load
    df = load("AAPL")
    strategy = MeanReversion(window=20)
    result = strategy.generate_signals(df)
    print(result[["Close", "MA", "signal"]].dropna().tail(10))