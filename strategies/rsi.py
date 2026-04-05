import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from strategies.base import Strategy

class RSI(Strategy):
    def __init__(self, window: int = 14, oversold: int = 30, overbought: int = 70):
        self.window     = window
        self.oversold   = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        delta = data["Close"].diff()
        gain  = delta.clip(lower=0).rolling(self.window).mean()
        loss  = (-delta.clip(upper=0)).rolling(self.window).mean()
        rs    = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))
        data["signal"] = 0
        data.loc[data["RSI"] < self.oversold,   "signal"] = 1   # oversold → buy
        data.loc[data["RSI"] > self.overbought, "signal"] = -1  # overbought → sell
        return data

if __name__ == "__main__":
    from ingestion.fetch_market_data import load
    df = load("AAPL")
    strategy = RSI(window=14)
    result = strategy.generate_signals(df)
    print(result[["Close", "RSI", "signal"]].dropna().tail(10))