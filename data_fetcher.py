"""
Data fetching module: yfinance primary + Alpha Vantage refinement.
Respects the 25-call/day AV limit by only verifying top-15 stocks.
"""

from __future__ import annotations

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional


DEFAULT_WISHLIST = [
    "SERV", "DXYZ", "IONQ", "IREN", "RDNT", "INTC", "SOXL", "NFLX", "NBIS",
    "CRDO", "ORCL", "CRM", "NVDA", "ADBE", "AMD", "AAPL", "VRT", "GOOG",
    "TSM", "AVGO", "TSLA", "MSFT", "APP", "STX", "META", "AZO", "DPRO",
    "ONDS", "UMAC", "PLTR", "RTX", "AVAV", "GD", "LMT", "NOC", "CAT",
    "VST", "TEM", "PANW", "AMZN", "KO", "TQQQ", "SGOV", "WMT", "SPMO",
    "BRK-B", "COST", "TCOM", "PDD", "BABA", "RBLX", "EOSE", "UAMY", "AES",
    "UUUU", "OXY", "MP", "CVX", "GEV", "UNH", "GILD", "NVO", "LLY",
    "LAES", "RGTI", "MU", "WDC", "SNDK",
]


@st.cache_data(ttl=86400, show_spinner="Fetching market data from Yahoo Finance...")
def fetch_yfinance_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Pull max available daily OHLCV for all tickers via yfinance."""
    data = {}
    # Deduplicate
    unique_tickers = sorted(set(tickers))
    # Batch download for speed
    try:
        raw = yf.download(unique_tickers, period="2y", group_by="ticker", progress=False)
        if raw.empty:
            return data
        for ticker in unique_tickers:
            try:
                if len(unique_tickers) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy()
                df = df.dropna(subset=["Close"])
                if len(df) >= 50:
                    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
                    data[ticker] = df
            except (KeyError, TypeError):
                continue
    except Exception:
        # Fallback: fetch individually
        for ticker in unique_tickers:
            try:
                df = yf.download(ticker, period="2y", progress=False)
                if df is not None and len(df) >= 50:
                    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
                    data[ticker] = df.dropna(subset=["Close"])
            except Exception:
                continue
    return data


@st.cache_data(ttl=86400, show_spinner="Fetching S&P 500 index data...")
def fetch_sp500_data() -> pd.DataFrame:
    """Fetch S&P 500 index data for regime detection."""
    try:
        df = yf.download("^GSPC", period="2y", progress=False)
        if df is not None and not df.empty:
            df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
            return df.dropna(subset=["Close"])
    except Exception:
        pass
    return pd.DataFrame()


def fetch_av_daily(ticker: str, api_key: str) -> pd.DataFrame | None:
    """
    Fetch TIME_SERIES_DAILY_ADJUSTED from Alpha Vantage for signal verification.
    Returns DataFrame or None on failure.
    """
    if not api_key:
        return None
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return None
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume",
        })
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner="Verifying top signals via Alpha Vantage...")
def fetch_av_verification(tickers: list[str], api_key: str) -> dict[str, pd.DataFrame]:
    """Fetch AV data for up to 15 tickers (respects daily limit)."""
    results = {}
    for t in tickers[:15]:
        df = fetch_av_daily(t, api_key)
        if df is not None:
            results[t] = df
    return results
