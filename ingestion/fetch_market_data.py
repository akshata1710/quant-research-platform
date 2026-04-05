import yfinance as yf
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_RAW


def fetch_single(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading {ticker}...")
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    data.dropna(inplace=True)

    data.to_parquet(DATA_RAW / f"{ticker}.parquet")
    data.to_csv(DATA_RAW / f"{ticker}.csv")

    print(f"  Saved {len(data)} rows → data/raw/{ticker}.parquet")
    return data


def fetch_multiple(tickers: list, start: str, end: str) -> dict:
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = fetch_single(ticker, start, end)
        except Exception as e:
            print(f"  Warning: skipping {ticker} — {e}")
    return results


def load(ticker: str) -> pd.DataFrame:
    path = DATA_RAW / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data for {ticker}. Run fetch_single() first.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "SPY"]
    fetch_multiple(tickers, start="2018-01-01", end="2024-01-01")