import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from backtesting.position_sizing import fixed_fractional, kelly_fraction, compute_win_stats

COMMISSION = 0.001
SLIPPAGE   = 0.0005

def backtest(data: pd.DataFrame, cash: float = 10000.0,
             stop_loss: float = 0.05,
             trailing_stop: bool = True,
             sizing: str = "fixed_fractional",
             risk_pct: float = 0.02) -> dict:

    position    = 0
    initial_cash = cash
    equity_curve = []
    trades       = []
    entry_price  = None
    peak_price   = None

    data = data.copy()
    data["signal"] = data["signal"].shift(1).fillna(0)

    for i in range(len(data)):
        price  = data["Close"].iloc[i]
        signal = data["signal"].iloc[i]

        buy_price  = price * (1 + SLIPPAGE)
        sell_price = price * (1 - SLIPPAGE)

        # Trailing stop — tracks peak price since entry
        if position > 0 and entry_price is not None:
            if trailing_stop:
                peak_price = max(peak_price or entry_price, price)
                stop_price = peak_price * (1 - stop_loss)
            else:
                stop_price = entry_price * (1 - stop_loss)

            if price < stop_price:
                proceeds = sell_price * (1 - COMMISSION)
                cash += proceeds * position
                trades.append({"type": "stop_loss", "price": sell_price, "day": i})
                position    = 0
                entry_price = None
                peak_price  = None

        # Position sizing
        if sizing == "kelly" and len(trades) >= 10:
            stats    = compute_win_stats(trades)
            fraction = kelly_fraction(stats["win_rate"], stats["avg_win"], stats["avg_loss"])
            shares   = max(int((cash * fraction) / buy_price), 1)
        else:
            shares = fixed_fractional(cash, buy_price, risk_pct)
            shares = max(shares, 1)

        # Buy
        if signal == 1 and position == 0:
            cost = buy_price * shares * (1 + COMMISSION)
            if cash >= cost:
                position    += shares
                cash        -= cost
                entry_price  = buy_price
                peak_price   = buy_price
                trades.append({"type": "buy", "price": buy_price, "day": i, "shares": shares})

        # Sell
        elif signal == -1 and position > 0:
            proceeds = sell_price * position * (1 - COMMISSION)
            cash        += proceeds
            trades.append({"type": "sell", "price": sell_price, "day": i, "shares": position})
            position    = 0
            entry_price = None
            peak_price  = None

        equity_curve.append(cash + position * price)

    final_value      = equity_curve[-1]
    stop_loss_hits   = len([t for t in trades if t["type"] == "stop_loss"])

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
    from strategies.momentum import Momentum

    df = load("AAPL")
    df = Momentum(window=20).generate_signals(df)
    df = df.dropna()

    print("--- Fixed fractional sizing ---")
    r1 = backtest(df, sizing="fixed_fractional", trailing_stop=False)
    print(f"Return: {r1['return_pct']}%  |  Trades: {r1['total_trades']}  |  Stop losses: {r1['stop_loss_hits']}")

    print("\n--- Kelly sizing + trailing stop ---")
    r2 = backtest(df, sizing="kelly", trailing_stop=True)
    print(f"Return: {r2['return_pct']}%  |  Trades: {r2['total_trades']}  |  Stop losses: {r2['stop_loss_hits']}")