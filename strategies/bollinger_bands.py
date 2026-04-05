import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from strategies.base import Strategy

class BollingerBands(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window  = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["MA"]    = data["Close"].rolling(self.window).mean()
        data["STD"]   = data["Close"].rolling(self.window).std()
        data["upper"] = data["MA"] + self.num_std * data["STD"]
        data["lower"] = data["MA"] - self.num_std * data["STD"]
        data["signal"] = 0
        data.loc[data["Close"] < data["lower"], "signal"] = 1   # below lower band → buy
        data.loc[data["Close"] > data["upper"], "signal"] = -1  # above upper band → sell
        return data

if __name__ == "__main__":
    from ingestion.fetch_market_data import load
    df = load("AAPL")
    strategy = BollingerBands(window=20, num_std=2.0)
    result = strategy.generate_signals(df)
    print(result[["Close", "upper", "lower", "signal"]].dropna().tail(10))