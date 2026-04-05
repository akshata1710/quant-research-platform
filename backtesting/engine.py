import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

COMMISSION = 0.001
SLIPPAGE   = 0.0005

def backtest(data: pd.DataFrame, cash: float = 10000.0, stop_loss: float = 0.05) -> dict:
    position = 0
    initial_cash = cash
    equity_curve = []
    trades = []
    entry_price = None

    data = data.copy()
    data["signal"] = data["signal"].shift(1).fillna(0)

    for i in range(len(data)):
        price  = data["Close"].iloc[i]
        signal = data["signal"].iloc[i]

        buy_price  = price * (1 + SLIPPAGE)
        sell_price = price * (1 - SLIPPAGE)

        if position > 0 and entry_price is not None:
            if price < entry_price * (1 - stop_loss):
                proceeds = sell_price * (1 - COMMISSION)
                position -= 1
                cash += proceeds
                trades.append({"type": "stop_loss", "price": sell_price, "day": i})
                entry_price = None

        if signal == 1 and cash >= buy_price:
            cost = buy_price * (1 + COMMISSION)
            if cash >= cost:
                position += 1
                cash -= cost
                entry_price = buy_price
                trades.append({"type": "buy", "price": buy_price, "day": i})

        elif signal == -1 and position > 0:
            proceeds = sell_price * (1 - COMMISSION)
            position -= 1
            cash += proceeds
            entry_price = None
            trades.append({"type": "sell", "price": sell_price, "day": i})

        equity_curve.append(cash + position * price)

    final_value = equity_curve[-1]
    stop_loss_hits = len([t for t in trades if t["type"] == "stop_loss"])

    return {
        "initial_value":  initial_cash,
        "final_value":    round(final_value, 2),
        "return_pct":     round((final_value - initial_cash) / initial_cash * 100, 2),
        "equity_curve":   equity_curve,
        "total_trades":   len(trades),
        "stop_loss_hits": stop_loss_hits,
        "trades":         trades,
    }

if __name__ == "__main__":
    from ingestion.fetch_market_data import load
    from strategies.mean_reversion import MeanReversion

    df = load("AAPL")
    df = MeanReversion(window=20).generate_signals(df)
    df = df.dropna()

    result = backtest(df)
    print(f"Initial:        ${result['initial_value']:,.2f}")
    print(f"Final:          ${result['final_value']:,.2f}")
    print(f"Return:         {result['return_pct']}%")
    print(f"Total trades:   {result['total_trades']}")
    print(f"Stop loss hits: {result['stop_loss_hits']}")