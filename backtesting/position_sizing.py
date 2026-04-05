import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def fixed_fractional(cash: float, price: float, risk_pct: float = 0.02) -> int:
    """Risk a fixed % of portfolio per trade."""
    amount = cash * risk_pct
    shares = int(amount / price)
    return max(shares, 0)

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly criterion — optimal fraction of capital to risk.
    win_rate: % of trades that are profitable (0-1)
    avg_win:  average profit per winning trade
    avg_loss: average loss per losing trade
    """
    if avg_loss == 0:
        return 0
    b = avg_win / avg_loss
    f = win_rate - ((1 - win_rate) / b)
    return max(min(f, 0.25), 0)  # cap at 25% to avoid overbetting

def compute_win_stats(trades: list) -> dict:
    """Compute win rate and avg win/loss from a list of trades."""
    if len(trades) < 2:
        return {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0}

    pnls = []
    buys = [t for t in trades if t["type"] == "buy"]
    sells = [t for t in trades if t["type"] in ("sell", "stop_loss")]

    for b, s in zip(buys, sells):
        pnls.append(s["price"] - b["price"])

    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    return {
        "win_rate": len(wins) / len(pnls) if pnls else 0.5,
        "avg_win":  sum(wins)   / len(wins)   if wins   else 1.0,
        "avg_loss": abs(sum(losses) / len(losses)) if losses else 1.0,
    }

if __name__ == "__main__":
    print("Fixed fractional (2% of $10,000 at $150/share):")
    shares = fixed_fractional(cash=10000, price=150, risk_pct=0.02)
    print(f"  Buy {shares} shares")

    print("\nKelly fraction (60% win rate, avg win $2, avg loss $1):")
    f = kelly_fraction(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
    print(f"  Risk {f:.1%} of capital")