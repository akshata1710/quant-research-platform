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
from strategies.regime import detect_regime, filter_signals_by_regime

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

# --- Load results ---
results_path    = Path("experiments/results.csv")
wf_path         = Path("experiments/walk_forward_results.csv")
portfolio_path  = Path("experiments/portfolio_results.csv")

df         = pd.read_csv(results_path)
wf_df      = pd.read_csv(wf_path)
port_df    = pd.read_csv(portfolio_path)

# --- Sidebar ---
st.sidebar.header("Filters")
tickers    = st.sidebar.multiselect("Tickers",    df["ticker"].unique(),   default=list(df["ticker"].unique()))
strategies = st.sidebar.multiselect("Strategies", df["strategy"].unique(), default=list(df["strategy"].unique()))
use_filter = st.sidebar.toggle("Regime filter only", value=False)

filtered = df[df["ticker"].isin(tickers) & df["strategy"].isin(strategies)]
if use_filter:
    filtered = filtered[filtered["regime_filter"] == True]
else:
    filtered = filtered[filtered["regime_filter"] == False]

# ================================================================
# SECTION 1 — Overview
# ================================================================
st.header("1. Experiment overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total experiments", len(filtered))
col2.metric("Best return",       f"{filtered['return_pct'].max()}%")
col3.metric("Best Sharpe",       round(filtered['sharpe_ratio'].max(), 3))
col4.metric("Lowest drawdown",   f"{filtered['max_drawdown'].max()}%")

st.dataframe(
    filtered.sort_values("sharpe_ratio", ascending=False),
    use_container_width=True
)

# ================================================================
# SECTION 2 — Heatmaps
# ================================================================
st.header("2. Sharpe ratio heatmaps")
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

# ================================================================
# SECTION 3 — Regime filter comparison
# ================================================================
st.header("3. Regime filter impact")

no_filter  = df[df["ticker"].isin(tickers) & df["strategy"].isin(strategies) & (df["regime_filter"] == False)]
yes_filter = df[df["ticker"].isin(tickers) & df["strategy"].isin(strategies) & (df["regime_filter"] == True)]

col_r1, col_r2, col_r3, col_r4 = st.columns(4)
col_r1.metric("Avg return (no filter)",   f"{no_filter['return_pct'].mean():.2f}%")
col_r2.metric("Avg return (with filter)", f"{yes_filter['return_pct'].mean():.2f}%")
col_r3.metric("Avg Sharpe (no filter)",   round(no_filter['sharpe_ratio'].mean(), 3))
col_r4.metric("Avg Sharpe (with filter)", round(yes_filter['sharpe_ratio'].mean(), 3))

compare_df = pd.DataFrame({
    "Without filter": no_filter.groupby("strategy")["sharpe_ratio"].mean(),
    "With filter":    yes_filter.groupby("strategy")["sharpe_ratio"].mean(),
}).reset_index()

fig_regime = px.bar(
    compare_df.melt(id_vars="strategy", var_name="Filter", value_name="Avg Sharpe"),
    x="strategy", y="Avg Sharpe", color="Filter",
    barmode="group", height=350,
    color_discrete_map={"Without filter": "#1D9E75", "With filter": "#7F77DD"}
)
st.plotly_chart(fig_regime, use_container_width=True)

# ================================================================
# SECTION 4 — Walk-forward validation
# ================================================================
st.header("4. Walk-forward validation")
st.caption("Average performance on unseen test data — the honest measure of strategy quality")

col_w1, col_w2 = st.columns(2)

with col_w1:
    st.caption("Avg Sharpe by strategy and ticker")
    wf_pivot = wf_df.pivot_table(index="ticker", columns="strategy", values="avg_sharpe")
    fig_wf = px.imshow(wf_pivot, color_continuous_scale="Teal", text_auto=".2f", aspect="auto")
    st.plotly_chart(fig_wf, use_container_width=True)

with col_w2:
    st.caption("Top strategies — walk-forward validated")
    top_wf = wf_df.sort_values("avg_sharpe", ascending=False).head(8)
    fig_bar = px.bar(
        top_wf, x="avg_sharpe", y="strategy", color="ticker",
        orientation="h", height=350,
        labels={"avg_sharpe": "Avg Sharpe (out-of-sample)"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.dataframe(
    wf_df.sort_values("avg_sharpe", ascending=False),
    use_container_width=True
)

# ================================================================
# SECTION 5 — Portfolio results
# ================================================================
st.header("5. Portfolio backtesting")
st.caption("All tickers trading simultaneously with shared capital ($100,000)")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.caption("Return by strategy and window")
    fig_port1 = px.bar(
        port_df.sort_values("total_return", ascending=False),
        x="strategy", y="total_return", color="window",
        barmode="group", height=350,
        labels={"total_return": "Portfolio return (%)"}
    )
    st.plotly_chart(fig_port1, use_container_width=True)

with col_p2:
    st.caption("Sharpe by strategy and window")
    fig_port2 = px.bar(
        port_df.sort_values("sharpe_ratio", ascending=False),
        x="strategy", y="sharpe_ratio", color="window",
        barmode="group", height=350,
        labels={"sharpe_ratio": "Sharpe ratio"}
    )
    st.plotly_chart(fig_port2, use_container_width=True)

# ================================================================
# SECTION 6 — Equity curve explorer
# ================================================================
st.header("6. Equity curve explorer")

col_t, col_s, col_w = st.columns(3)
ticker        = col_t.selectbox("Ticker",   df["ticker"].unique())
strategy_name = col_s.selectbox("Strategy", list(STRATEGIES.keys()))
window        = col_w.selectbox("Window",   sorted(df["window"].unique()))
regime_on     = st.checkbox("Apply regime filter", value=False)

price_data = load(ticker)
strategy   = STRATEGIES[strategy_name](window)
signals    = strategy.generate_signals(price_data)

if regime_on:
    signals = detect_regime(signals)
    signals = filter_signals_by_regime(signals, strategy_name)

signals = signals.dropna()
result  = backtest(signals)
metrics = compute_metrics(result["equity_curve"])

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

# ================================================================
# SECTION 7 — Strategy comparison
# ================================================================
st.header("7. Compare all strategies on one ticker")

compare_ticker = st.selectbox("Ticker", df["ticker"].unique(), key="compare")
compare_window = st.selectbox("Window", sorted(df["window"].unique()), key="cwindow")
compare_regime = st.checkbox("Apply regime filter", value=False, key="cregime")

fig5   = go.Figure()
colors = {
    "mean_reversion":  "#1D9E75",
    "momentum":        "#7F77DD",
    "rsi":             "#D85A30",
    "bollinger_bands": "#BA7517",
}

for sname, sfunc in STRATEGIES.items():
    pdf = load(compare_ticker)
    sig = sfunc(compare_window).generate_signals(pdf)
    if compare_regime:
        sig = detect_regime(sig)
        sig = filter_signals_by_regime(sig, sname)
    sig = sig.dropna()
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