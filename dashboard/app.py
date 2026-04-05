import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ingestion.fetch_market_data import load
from backtesting.engine import backtest
from backtesting.metrics import compute_metrics

from strategies.mean_reversion  import MeanReversion
from strategies.momentum        import Momentum
from strategies.rsi             import RSI
from strategies.bollinger_bands import BollingerBands

STRATEGIES = {
    "mean_reversion":  lambda w: MeanReversion(window=w),
    "momentum":        lambda w: Momentum(window=w),
    "rsi":             lambda w: RSI(window=w),
    "bollinger_bands": lambda w: BollingerBands(window=w),
}

st.set_page_config(page_title="Quant Research", layout="wide")
st.title("Quant Research Dashboard")

results_path = Path("experiments/results.csv")
df = pd.read_csv(results_path)

# --- Sidebar ---
st.sidebar.header("Filters")
tickers    = st.sidebar.multiselect("Tickers",    df["ticker"].unique(),   default=list(df["ticker"].unique()))
strategies = st.sidebar.multiselect("Strategies", df["strategy"].unique(), default=list(df["strategy"].unique()))
filtered   = df[df["ticker"].isin(tickers) & df["strategy"].isin(strategies)]

# --- Overview metrics ---
st.subheader("Experiment overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total experiments", len(filtered))
col2.metric("Best return",       f"{filtered['return_pct'].max()}%")
col3.metric("Best Sharpe",       round(filtered['sharpe_ratio'].max(), 3))
col4.metric("Lowest drawdown",   f"{filtered['max_drawdown'].max()}%")

# --- Results table ---
st.subheader("All experiments")
st.dataframe(
    filtered.sort_values("sharpe_ratio", ascending=False),
    use_container_width=True
)

# --- Sharpe heatmap ---
st.subheader("Sharpe ratio heatmap")
col_h1, col_h2 = st.columns(2)

with col_h1:
    st.caption("By ticker and strategy")
    pivot1 = filtered.pivot_table(index="ticker", columns="strategy", values="sharpe_ratio", aggfunc="max")
    fig1 = px.imshow(pivot1, color_continuous_scale="Teal", text_auto=".2f", aspect="auto")
    st.plotly_chart(fig1, use_container_width=True)

with col_h2:
    st.caption("By window and strategy")
    pivot2 = filtered.pivot_table(index="window", columns="strategy", values="sharpe_ratio", aggfunc="max")
    fig2 = px.imshow(pivot2, color_continuous_scale="Teal", text_auto=".2f", aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)

# --- Return comparison bar chart ---
st.subheader("Return comparison by strategy")
fig3 = px.bar(
    filtered.sort_values("return_pct", ascending=False),
    x="strategy", y="return_pct", color="ticker",
    barmode="group", height=400,
    labels={"return_pct": "Return (%)", "strategy": "Strategy"}
)
st.plotly_chart(fig3, use_container_width=True)

# --- Equity curve explorer ---
st.subheader("Equity curve explorer")
col_t, col_s, col_w = st.columns(3)
ticker        = col_t.selectbox("Ticker",   df["ticker"].unique())
strategy_name = col_s.selectbox("Strategy", list(STRATEGIES.keys()))
window        = col_w.selectbox("Window",   sorted(df["window"].unique()))

price_data = load(ticker)
strategy   = STRATEGIES[strategy_name](window)
signals    = strategy.generate_signals(price_data).dropna()
result     = backtest(signals)
metrics    = compute_metrics(result["equity_curve"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Return",        f"{result['return_pct']}%")
m2.metric("Sharpe",        metrics['sharpe_ratio'])
m3.metric("Max drawdown",  f"{metrics['max_drawdown']}%")
m4.metric("Profit factor", metrics['profit_factor'])

fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    y=result["equity_curve"],
    mode="lines",
    name="Portfolio value",
    line=dict(color="#1D9E75", width=2)
))
fig4.update_layout(
    xaxis_title="Trading days",
    yaxis_title="Portfolio value ($)",
    height=400,
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig4, use_container_width=True)

# --- Strategy comparison on same ticker ---
st.subheader("Compare all strategies on one ticker")
compare_ticker = st.selectbox("Select ticker", df["ticker"].unique(), key="compare")
compare_window = st.selectbox("Select window", sorted(df["window"].unique()), key="cwindow")

fig5 = go.Figure()
colors = {
    "mean_reversion":  "#1D9E75",
    "momentum":        "#7F77DD",
    "rsi":             "#D85A30",
    "bollinger_bands": "#BA7517",
}

for sname, sfunc in STRATEGIES.items():
    pdf = load(compare_ticker)
    sig = sfunc(compare_window).generate_signals(pdf).dropna()
    res = backtest(sig)
    fig5.add_trace(go.Scatter(
        y=res["equity_curve"],
        mode="lines",
        name=sname,
        line=dict(color=colors[sname], width=2)
    ))

fig5.update_layout(
    xaxis_title="Trading days",
    yaxis_title="Portfolio value ($)",
    height=450,
    margin=dict(l=0, r=0, t=20, b=0)
)
st.plotly_chart(fig5, use_container_width=True)