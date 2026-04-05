import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ingestion.fetch_market_data import load
from strategies.mean_reversion import MeanReversion
from backtesting.engine import backtest
from backtesting.metrics import compute_metrics

st.set_page_config(page_title="Quant Research", layout="wide")
st.title("Quant Research Dashboard")

# --- Load experiment results ---
results_path = Path("experiments/results_grid.csv")
df = pd.read_csv(results_path)

# --- Sidebar filters ---
st.sidebar.header("Filters")
tickers = st.sidebar.multiselect("Tickers", df["ticker"].unique(), default=list(df["ticker"].unique()))
filtered = df[df["ticker"].isin(tickers)]

# --- Top metrics ---
st.subheader("Experiment overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total experiments", len(filtered))
col2.metric("Best return",   f"{filtered['return_pct'].max()}%")
col3.metric("Best Sharpe",   filtered['sharpe_ratio'].max())
col4.metric("Best drawdown", f"{filtered['max_drawdown'].max()}%")

# --- Results table ---
st.subheader("All experiments")
st.dataframe(
    filtered.sort_values("sharpe_ratio", ascending=False),
    use_container_width=True
)

# --- Sharpe heatmap ---
st.subheader("Sharpe ratio by ticker and window")
pivot = filtered.pivot(index="ticker", columns="window", values="sharpe_ratio")
fig = px.imshow(pivot, color_continuous_scale="Teal", text_auto=".2f", aspect="auto")
st.plotly_chart(fig, use_container_width=True)

# --- Equity curve explorer ---
st.subheader("Equity curve explorer")
col_t, col_w = st.columns(2)
ticker = col_t.selectbox("Ticker", df["ticker"].unique())
window = col_w.selectbox("Window", sorted(df["window"].unique()))

price_data = load(ticker)
signals = MeanReversion(window=window).generate_signals(price_data).dropna()
result = backtest(signals)
metrics = compute_metrics(result["equity_curve"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Return",        f"{result['return_pct']}%")
m2.metric("Sharpe",        metrics['sharpe_ratio'])
m3.metric("Max drawdown",  f"{metrics['max_drawdown']}%")
m4.metric("Profit factor", metrics['profit_factor'])

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    y=result["equity_curve"],
    mode="lines",
    name="Portfolio value",
    line=dict(color="#1D9E75", width=2)
))
fig2.update_layout(
    xaxis_title="Trading days",
    yaxis_title="Portfolio value ($)",
    height=400,
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig2, use_container_width=True)