"""
S&P 500 Multi-Class Momentum Engine
Streamlit dashboard for analyzing a 75-ticker wishlist using three strategy classes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from pathlib import Path

from data_fetcher import (
    DEFAULT_WISHLIST, fetch_yfinance_data, fetch_sp500_data, fetch_av_verification,
)
from strategies import run_all_strategies, get_top15_tickers
from indicators import compute_indicators

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="S&P 500 Multi-Class Momentum Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

WISHLIST_FILE = Path(__file__).parent / "wishlist.json"


def load_wishlist() -> list[str]:
    if WISHLIST_FILE.exists():
        try:
            with open(WISHLIST_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_WISHLIST.copy()


def save_wishlist(tickers: list[str]):
    with open(WISHLIST_FILE, "w") as f:
        json.dump(sorted(set(tickers)), f, indent=2)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")

# Alpha Vantage API key
av_api_key = st.sidebar.text_input(
    "Alpha Vantage API Key",
    type="password",
    help="Optional. Used to verify top-15 signals (25 calls/day limit).",
)

# Wishlist management
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Watchlist Management")

wishlist = load_wishlist()

# Add ticker
col_add1, col_add2 = st.sidebar.columns([3, 1])
with col_add1:
    new_ticker = st.text_input("Add ticker", placeholder="e.g. AAPL", label_visibility="collapsed")
with col_add2:
    if st.button("Add", use_container_width=True):
        t = new_ticker.strip().upper()
        if t and t not in wishlist:
            wishlist.append(t)
            save_wishlist(wishlist)
            st.rerun()
        elif t in wishlist:
            st.sidebar.warning(f"{t} already in list")

# Remove ticker
remove_ticker = st.sidebar.selectbox(
    "Remove ticker",
    options=[""] + sorted(wishlist),
    index=0,
)
if remove_ticker:
    if st.sidebar.button(f"Remove {remove_ticker}", use_container_width=True):
        wishlist = [t for t in wishlist if t != remove_ticker]
        save_wishlist(wishlist)
        st.rerun()

st.sidebar.caption(f"Tracking **{len(wishlist)}** tickers")

with st.sidebar.expander("View full watchlist"):
    st.code("\n".join(sorted(wishlist)), language=None)

# Manual refresh
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Data auto-caches for 24h. Use button above to force refresh.")

# ── Main Header ──────────────────────────────────────────────────────────────

st.title("⚡ S&P 500 Multi-Class Momentum Engine")

# ── Market Regime Detection ──────────────────────────────────────────────────

sp500_df = fetch_sp500_data()
bearish_regime = False

if not sp500_df.empty and len(sp500_df) >= 200:
    sp500_close = float(sp500_df["Close"].iloc[-1])
    sp500_sma200 = float(sp500_df["Close"].rolling(200).mean().iloc[-1])
    deviation_pct = (sp500_close - sp500_sma200) / sp500_sma200 * 100

    if sp500_close < sp500_sma200:
        bearish_regime = True
        st.error(
            f"🐻 **BEARISH REGIME: NEW POSITIONS LOCKED** — "
            f"S&P 500 ({sp500_close:,.0f}) is below SMA200 ({sp500_sma200:,.0f}) "
            f"| Deviation: {deviation_pct:+.1f}%"
        )
    else:
        st.success(
            f"🟢 **BULLISH REGIME** — S&P 500 ({sp500_close:,.0f}) above SMA200 ({sp500_sma200:,.0f}) "
            f"| Deviation: {deviation_pct:+.1f}%"
        )
else:
    st.warning("⚠️ Could not fetch S&P 500 data for regime detection.")

# ── Fetch Data ───────────────────────────────────────────────────────────────

with st.spinner("Loading market data..."):
    all_data = fetch_yfinance_data(wishlist)

if not all_data:
    st.error("No data could be fetched. Check your internet connection or ticker symbols.")
    st.stop()

st.caption(f"Successfully loaded data for **{len(all_data)}** / {len(wishlist)} tickers")

# ── Run Strategies ───────────────────────────────────────────────────────────

with st.spinner("Running strategy scans..."):
    results = run_all_strategies(all_data)

# ── Alpha Vantage Verification ───────────────────────────────────────────────

av_verified = {}
if av_api_key:
    top15 = get_top15_tickers(results)
    if top15:
        av_verified = fetch_av_verification(top15, av_api_key)
        if av_verified:
            st.info(f"✅ Alpha Vantage verified {len(av_verified)} / {len(top15)} top signals")

# ── Toast Alerts ─────────────────────────────────────────────────────────────

for sig in results.get("class3", []):
    if sig["Status"] == "Entry Triggered":
        st.toast(f"🚀 CLASS III BREAKOUT: {sig['Ticker']} @ ${sig['Close']}", icon="🚀")

for sig in results.get("class1", []):
    if sig["Status"] == "Entry Triggered":
        st.toast(f"💎 CLASS I GOLDEN PIT: {sig['Ticker']} RSI={sig['RSI']}", icon="💎")

# ── Overview Metrics ─────────────────────────────────────────────────────────

c1_entries = [s for s in results["class1"] if s["Status"] == "Entry Triggered"]
c2_entries = [s for s in results["class2"] if s["Status"] == "Entry Triggered"]
c3_entries = [s for s in results["class3"] if s["Status"] == "Entry Triggered"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Class I Signals", len(results["class1"]), f"{len(c1_entries)} entries")
col2.metric("Class II Signals", len(results["class2"]), f"{len(c2_entries)} entries")
col3.metric("Class III Signals", len(results["class3"]), f"{len(c3_entries)} entries")
total_entries = len(c1_entries) + len(c2_entries) + len(c3_entries)
col4.metric("Total Entry Signals", total_entries)

# ── Signal Center (Tabbed View) ─────────────────────────────────────────────

st.markdown("---")
st.header("📡 Signal Center")

tab1, tab2, tab3, tab_scatter, tab_overview = st.tabs([
    "Class I: Mean Reversion",
    "Class II: Trend Resonance",
    "Class III: Breakout Engine",
    "📊 Signal Scatter Plot",
    "🌐 All Stocks Overview",
])

def status_color(status: str) -> str:
    if status == "Entry Triggered":
        return "🟢"
    elif status == "Hold":
        return "🟡"
    return "👁️"


def render_signal_table(signals: list[dict], class_name: str):
    if not signals:
        st.info(f"No {class_name} signals detected today.")
        return

    df = pd.DataFrame(signals)
    df.insert(0, "", df["Status"].map(status_color))

    # AV verification badge
    if av_verified:
        df["AV_Verified"] = df["Ticker"].apply(lambda t: "✅" if t in av_verified else "—")

    # Highlight entry triggered rows
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Kelly_%": st.column_config.NumberColumn("Kelly %", format="%.2f%%"),
            "Win_Rate": st.column_config.NumberColumn("Win Rate %", format="%.1f"),
            "Backtest_Trades": st.column_config.NumberColumn("BT Trades", format="%d"),
            "Dist_to_Mean_%": st.column_config.NumberColumn("Dist to Mean %", format="%.2f%%"),
            "ATR_%": st.column_config.NumberColumn("ATR %", format="%.2f%%"),
            "Signal_Strength": st.column_config.ProgressColumn(
                "Signal Strength", min_value=0, max_value=1.5, format="%.3f",
            ),
            "Volatility": st.column_config.NumberColumn("Vol %", format="%.2f%%"),
        },
    )


with tab1:
    st.subheader("Class I: High-Precision Mean Reversion")
    st.caption("Archetype: Oversold in Uptrend | Target Win Rate > 80% | Quarter Kelly")
    if bearish_regime:
        st.warning("⚠️ Bearish regime — new Class I entries suspended.")
    render_signal_table(results["class1"], "Class I")

with tab2:
    st.subheader("Class II: Balanced Trend Resonance")
    st.caption("Archetype: Pullback Continuation | Target Win Rate > 70% | Half Kelly")
    if bearish_regime:
        st.warning("⚠️ Bearish regime — new Class II entries suspended.")
    render_signal_table(results["class2"], "Class II")

with tab3:
    st.subheader("Class III: Explosive Breakout Engine")
    st.caption("Archetype: Fat-Tail Volatility Squeeze | Target Win Rate > 60% | Fractional Kelly")
    if bearish_regime:
        st.warning("⚠️ Bearish regime — new Class III entries suspended.")
    render_signal_table(results["class3"], "Class III")

# ── Scatter Plot: Signal Strength vs Volatility ─────────────────────────────

with tab_scatter:
    st.subheader("Signal Strength vs. Volatility")
    all_signals = results["class1"] + results["class2"] + results["class3"]

    if all_signals:
        scatter_df = pd.DataFrame(all_signals)
        fig = px.scatter(
            scatter_df,
            x="Volatility",
            y="Signal_Strength",
            color="Class",
            size="Kelly_%",
            hover_name="Ticker",
            hover_data=["Status", "Close", "ATR", "Win_Rate"],
            title="Signal Strength vs. Volatility (size = Kelly %)",
            color_discrete_map={
                "I - Mean Reversion": "#2ecc71",
                "II - Trend Resonance": "#3498db",
                "III - Breakout Engine": "#e74c3c",
            },
            labels={
                "Volatility": "Volatility (ATR/Price %)",
                "Signal_Strength": "Signal Strength",
            },
        )
        fig.update_layout(
            height=500,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_traces(marker=dict(line=dict(width=1, color="white")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No signals to plot.")

# ── All Stocks Overview ──────────────────────────────────────────────────────

with tab_overview:
    st.subheader("All Stocks Overview")
    st.caption("Every ticker in your watchlist with key indicators — regardless of signal status")

    overview_rows = []
    signal_tickers = set()
    for sig in (results["class1"] + results["class2"] + results["class3"]):
        signal_tickers.add(sig["Ticker"])

    for ticker, raw_df in all_data.items():
        df = compute_indicators(raw_df)
        if len(df) < 50:
            continue

        close = float(df["Close"].dropna().iloc[-1])
        rsi_val = float(df["RSI_14"].dropna().iloc[-1]) if df["RSI_14"].dropna().shape[0] > 0 else np.nan
        atr_val = float(df["ATR_14"].dropna().iloc[-1]) if df["ATR_14"].dropna().shape[0] > 0 else np.nan
        sma200 = float(df["SMA_200"].dropna().iloc[-1]) if df["SMA_200"].dropna().shape[0] > 0 else np.nan
        ema20 = float(df["EMA_20"].dropna().iloc[-1]) if df["EMA_20"].dropna().shape[0] > 0 else np.nan
        adx_val = float(df["ADX_14"].dropna().iloc[-1]) if df["ADX_14"].dropna().shape[0] > 0 else np.nan
        bb_width = float(df["BB_Width"].dropna().iloc[-1]) if df["BB_Width"].dropna().shape[0] > 0 else np.nan
        roc_val = float(df["ROC_12"].dropna().iloc[-1]) if df["ROC_12"].dropna().shape[0] > 0 else np.nan
        macd_hist = float(df["MACD_Hist"].dropna().iloc[-1]) if df["MACD_Hist"].dropna().shape[0] > 0 else np.nan

        vol_pct = (atr_val / close * 100) if (atr_val and close) else np.nan
        trend = "Above" if (not np.isnan(sma200) and close > sma200) else "Below"

        # Composite signal strength: normalized RSI proximity + momentum + trend strength
        rsi_score = max(0, (50 - abs(rsi_val - 30)) / 50) if not np.isnan(rsi_val) else 0
        adx_score = min(1.0, adx_val / 50) if not np.isnan(adx_val) else 0
        signal_strength = round((rsi_score + adx_score) / 2, 3)

        # Check if this ticker has an active signal
        active = "Yes" if ticker in signal_tickers else "No"

        overview_rows.append({
            "Ticker": ticker,
            "Close": round(close, 2),
            "RSI_14": round(rsi_val, 1) if not np.isnan(rsi_val) else None,
            "ADX_14": round(adx_val, 1) if not np.isnan(adx_val) else None,
            "ROC_12%": round(roc_val, 2) if not np.isnan(roc_val) else None,
            "MACD_Hist": round(macd_hist, 4) if not np.isnan(macd_hist) else None,
            "BB_Width": round(bb_width, 4) if not np.isnan(bb_width) else None,
            "Volatility%": round(vol_pct, 2) if not np.isnan(vol_pct) else None,
            "vs_SMA200": trend,
            "Signal_Strength": signal_strength,
            "Has_Signal": active,
        })

    if overview_rows:
        ov_df = pd.DataFrame(overview_rows).sort_values("Signal_Strength", ascending=False)

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            trend_filter = st.selectbox("Filter by SMA200 trend", ["All", "Above", "Below"])
        with col_f2:
            signal_filter = st.selectbox("Filter by signal status", ["All", "Has Signal", "No Signal"])

        if trend_filter != "All":
            ov_df = ov_df[ov_df["vs_SMA200"] == trend_filter]
        if signal_filter == "Has Signal":
            ov_df = ov_df[ov_df["Has_Signal"] == "Yes"]
        elif signal_filter == "No Signal":
            ov_df = ov_df[ov_df["Has_Signal"] == "No"]

        st.dataframe(
            ov_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Signal_Strength": st.column_config.ProgressColumn(
                    "Signal Strength", min_value=0, max_value=1.0, format="%.3f",
                ),
                "Volatility%": st.column_config.NumberColumn("Vol %", format="%.2f%%"),
                "ROC_12%": st.column_config.NumberColumn("ROC 12d %", format="%.2f%%"),
            },
        )

        # Full universe scatter plot
        st.markdown("#### All Stocks: Signal Strength vs Volatility")
        fig_all = px.scatter(
            ov_df,
            x="Volatility%",
            y="Signal_Strength",
            color="Has_Signal",
            hover_name="Ticker",
            hover_data=["Close", "RSI_14", "ADX_14", "ROC_12%", "vs_SMA200"],
            title="Full Watchlist: Signal Strength vs. Volatility",
            color_discrete_map={"Yes": "#e74c3c", "No": "#636e72"},
            labels={
                "Volatility%": "Volatility (ATR/Price %)",
                "Signal_Strength": "Signal Strength",
                "Has_Signal": "Active Signal",
            },
        )
        fig_all.update_layout(
            height=500,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_all.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        st.plotly_chart(fig_all, use_container_width=True)
    else:
        st.info("No stock data available for overview.")

# ── Kelly Sizing Summary ────────────────────────────────────────────────────

st.markdown("---")
st.header("📐 Kelly Sizing Summary")

all_active = [s for s in all_signals if s["Status"] in ("Entry Triggered", "Hold")]
if all_active:
    kelly_cols = ["Ticker", "Class", "Status", "Close", "Kelly_%", "Win_Rate", "Backtest_Trades", "Stop_Loss", "ATR"]
    # Only include columns that exist (varies by class)
    available_cols = [c for c in kelly_cols if c in pd.DataFrame(all_active).columns]
    kelly_df = pd.DataFrame(all_active)[available_cols]
    kelly_df = kelly_df.sort_values("Kelly_%", ascending=False)

    col_kelly, col_chart = st.columns([1, 1])
    with col_kelly:
        st.dataframe(
            kelly_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Kelly_%": st.column_config.NumberColumn("Kelly %", format="%.2f%%"),
                "Win_Rate": st.column_config.NumberColumn("Win Rate %", format="%.1f"),
                "Backtest_Trades": st.column_config.NumberColumn("BT Trades", format="%d"),
            },
        )
    with col_chart:
        fig_kelly = px.bar(
            kelly_df,
            x="Ticker",
            y="Kelly_%",
            color="Class",
            title="Position Sizing (Kelly %)",
            color_discrete_map={
                "I - Mean Reversion": "#2ecc71",
                "II - Trend Resonance": "#3498db",
                "III - Breakout Engine": "#e74c3c",
            },
        )
        fig_kelly.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_kelly, use_container_width=True)
else:
    st.info("No active entry or hold signals for Kelly sizing.")

# ── AV Verification Detail ──────────────────────────────────────────────────

if av_verified:
    st.markdown("---")
    st.header("🔍 Alpha Vantage Verification")
    st.caption("Adjusted close comparison between yfinance and Alpha Vantage for top signals")

    for ticker, av_df in av_verified.items():
        if ticker in all_data:
            with st.expander(f"{ticker} — AV vs YF Comparison"):
                yf_close = float(all_data[ticker]["Close"].iloc[-1])
                av_close = float(av_df["Close"].iloc[-1]) if "Close" in av_df.columns else np.nan
                diff_pct = (yf_close - av_close) / av_close * 100 if not np.isnan(av_close) else np.nan

                c1, c2, c3 = st.columns(3)
                c1.metric("YF Close", f"${yf_close:.2f}")
                c2.metric("AV Close", f"${av_close:.2f}")
                c3.metric("Difference", f"{diff_pct:+.2f}%" if not np.isnan(diff_pct) else "N/A")

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "**S&P 500 Multi-Class Momentum Engine** | "
    "Data: Yahoo Finance (primary) + Alpha Vantage (verification) | "
    "Not financial advice. For educational and research purposes only."
)
