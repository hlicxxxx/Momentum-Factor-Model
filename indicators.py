"""
Technical indicators, preprocessing pipeline, and Kelly criterion sizing.
- Winsorization at 1st/99th percentiles
- Cross-sectional Z-score normalization
- Exponential performance weighting (half-life = 20 days)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import mstats


# ── Core Technical Indicators ────────────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    return upper, mid, lower, width


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # Zero out when one is not greater
    plus_dm[plus_dm <= minus_dm] = 0
    minus_dm[minus_dm <= plus_dm] = 0

    atr_val = atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_val.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.rolling(window=period, min_periods=period).mean()


def donchian(high: pd.Series, low: pd.Series, window: int = 20):
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    mid = (upper + lower) / 2
    return upper, mid, lower


def kdj(high: pd.Series, low: pd.Series, close: pd.Series,
        k_period: int = 9, d_period: int = 3):
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    rsv = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = rsv.ewm(alpha=1 / d_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    return close.pct_change(periods=period) * 100


def volume_sma(volume: pd.Series, window: int) -> pd.Series:
    return sma(volume, window)


# ── Preprocessing Pipeline ───────────────────────────────────────────────────

def winsorize_series(s: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
    """Winsorize at 1st and 99th percentiles."""
    arr = s.values.copy()
    mask = ~np.isnan(arr)
    if mask.sum() < 5:
        return s
    arr[mask] = mstats.winsorize(arr[mask], limits=limits)
    return pd.Series(arr, index=s.index, name=s.name)


def cross_sectional_zscore(factor_values: dict[str, float]) -> dict[str, float]:
    """
    Z-score normalization across all tickers for a single factor.
    Z = (Factor_i - mu) / sigma
    """
    vals = pd.Series(factor_values)
    vals = vals.dropna()
    if len(vals) < 3 or vals.std() == 0:
        return {k: 0.0 for k in factor_values}
    mu = vals.mean()
    sigma = vals.std()
    return {k: (v - mu) / sigma if not np.isnan(v) else 0.0
            for k, v in factor_values.items()}


def exponential_weights(n_days: int, half_life: int = 20) -> np.ndarray:
    """
    Exponential performance weighting.
    W_t = exp(-ln(2)/T_hl * (T_now - t))
    """
    t = np.arange(n_days)
    decay = np.log(2) / half_life
    weights = np.exp(-decay * (n_days - 1 - t))
    return weights / weights.sum()


# ── Kelly Criterion Sizing ───────────────────────────────────────────────────

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly Criterion: f* = (p * b - q) / b
    where p = win_rate, q = 1-p, b = avg_win/avg_loss
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = abs(avg_win / avg_loss)
    q = 1 - win_rate
    f = (win_rate * b - q) / b
    return max(0.0, f)


def quarter_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    return 0.25 * kelly_fraction(win_rate, avg_win, avg_loss)


def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    return 0.50 * kelly_fraction(win_rate, avg_win, avg_loss)


def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float,
                     fraction: float = 0.33) -> float:
    return fraction * kelly_fraction(win_rate, avg_win, avg_loss)


# ── Compute All Indicators for a Single Ticker ──────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a ticker's OHLCV DataFrame."""
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # Trend
    df["SMA_200"] = sma(c, 200)
    df["SMA_50"] = sma(c, 50)
    df["EMA_20"] = ema(c, 20)
    df["EMA_50"] = ema(c, 50)

    # Bollinger Bands
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"], df["BB_Width"] = bollinger_bands(c)

    # RSI
    df["RSI_14"] = rsi(c, 14)

    # KDJ
    df["KDJ_K"], df["KDJ_D"], df["KDJ_J"] = kdj(h, l, c)

    # MACD
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(c)

    # ATR
    df["ATR_14"] = atr(h, l, c, 14)

    # ADX
    df["ADX_14"] = adx(h, l, c, 14)

    # Donchian
    df["DC_Upper_20"], df["DC_Mid_20"], df["DC_Lower_20"] = donchian(h, l, 20)
    df["DC_Upper_10"], df["DC_Mid_10"], df["DC_Lower_10"] = donchian(h, l, 10)

    # ROC
    df["ROC_12"] = roc(c, 12)

    # Volume SMAs
    df["Vol_SMA_50"] = volume_sma(v, 50)
    df["Vol_SMA_20"] = volume_sma(v, 20)

    # BB Width 3-month low flag
    df["BB_Width_3m_Min"] = df["BB_Width"].rolling(window=63, min_periods=63).min()

    return df
