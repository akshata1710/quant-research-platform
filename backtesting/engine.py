import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

def backtest(data: pd.DataFrame, cash: float = 10000.0) -> dict:
    position = 0
    initial_cash = cash
    equity_curve = []

    for i in range(len(data)):
        price = data["Close"].iloc[i]
        signal = data["signal"].iloc[i]

        if signal == 1 and cash >= price:
            position += 1
            cash -= price
        elif signal == -1 and position > 0:
            position -= 1
            cash += price

        equity_curve.append(cash + position * price)

    final_value = equity_curve[-1]

    return {
        "initial_value": initial_cash,
        "final_value":   round(final_value, 2),
        "return_pct":    round((final_value - initial_cash) / initial_cash * 100, 2),
        "equity_curve":  equity_curve,
    }

if __name__ == "__main__":
    from ingestion.fetch_market_data import load
    from strategies.mean_reversion import MeanReversion

    df = load("AAPL")
    strategy = MeanReversion(window=20)
    df = strategy.generate_signals(df)
    df = df.dropna()

    result = backtest(df)
    print(f"Initial:  ${result['initial_value']:,.2f}")
    print(f"Final:    ${result['final_value']:,.2f}")
    print(f"Return:   {result['return_pct']}%")