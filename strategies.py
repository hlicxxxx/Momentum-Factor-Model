"""
Three strategy classes for the Momentum Engine.
Each returns signals with entry/exit/stop/sizing info.

Fixes applied:
- Historical backtested win rates per ticker (not static placeholders)
- Dynamic asset-specific Kelly: f* = p - (1-p)/b
- Proper cross-sectional Winsorization before Z-scoring
- Universe filtering (exclude low-vol ETFs; ATR% > 2% for Class III)
- Distance to Mean indicator for Class I
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from indicators import (
    compute_indicators, cross_sectional_zscore, winsorize_series,
    exponential_weights, sma,
)

# ── Low-volatility / non-momentum assets to exclude from Class II & III ──────
LOW_VOL_EXCLUDE = {
    "SGOV", "BIL", "SHV", "SHY", "IEF", "TLT", "VGSH", "BSV", "AGG",
    "BND", "GOVT", "SCHO", "SPMO",  # bond/treasury ETFs + factor ETFs
}


def _latest(df: pd.DataFrame, col: str):
    """Get latest non-NaN value from a column."""
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) > 0 else np.nan


def _prev(df: pd.DataFrame, col: str):
    """Get second-to-last non-NaN value."""
    s = df[col].dropna()
    return float(s.iloc[-2]) if len(s) > 1 else np.nan


# ── Historical Backtest Engine ───────────────────────────────────────────────

def _backtest_mean_reversion(df: pd.DataFrame, max_hold_days: int = 5) -> dict:
    """
    Backtest Class I logic over history.
    Entry: Close <= BB_Lower AND RSI < 30 AND Price > SMA200 AND KDJ_J > KDJ_D
    Exit: Close >= BB_Mid OR RSI > 50 OR held for max_hold_days (time exit)
    Returns dict with win_rate, avg_win, avg_loss, n_trades.
    """
    trades = []
    in_trade = False
    entry_price = 0.0
    bars_held = 0

    close = df["Close"].values
    sma200 = df["SMA_200"].values
    bb_lower = df["BB_Lower"].values
    bb_mid = df["BB_Mid"].values
    rsi_arr = df["RSI_14"].values
    kdj_j = df["KDJ_J"].values
    kdj_d = df["KDJ_D"].values

    for i in range(200, len(df)):
        if np.isnan(sma200[i]) or np.isnan(bb_lower[i]) or np.isnan(rsi_arr[i]):
            continue
        if not in_trade:
            if (close[i] > sma200[i]
                    and close[i] <= bb_lower[i]
                    and rsi_arr[i] < 30
                    and not np.isnan(kdj_j[i]) and not np.isnan(kdj_d[i])
                    and kdj_j[i] > kdj_d[i]):
                in_trade = True
                entry_price = close[i]
                bars_held = 0
        else:
            bars_held += 1
            # Exit: target hit, RSI recovery, OR time-based exit
            if close[i] >= bb_mid[i] or rsi_arr[i] > 50 or bars_held >= max_hold_days:
                pnl = (close[i] - entry_price) / entry_price
                trades.append(pnl)
                in_trade = False

    if not trades:
        return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "n_trades": 0}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    return {"win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss, "n_trades": len(trades)}


def _backtest_trend_resonance(df: pd.DataFrame) -> dict:
    """
    Backtest Class II logic over history.
    Entry: Close near EMA20 (±2%), closes above prev high, Vol > 1.2x SMA50(Vol)
    Exit: 20% gain OR price drops 1.5 × ATR below EMA50
    """
    trades = []
    in_trade = False
    entry_price = 0.0

    close = df["Close"].values
    ema20 = df["EMA_20"].values
    ema50 = df["EMA_50"].values
    high = df["High"].values
    vol = df["Volume"].values
    vol_sma50 = df["Vol_SMA_50"].values
    atr_arr = df["ATR_14"].values

    for i in range(200, len(df)):
        if any(np.isnan(v) for v in [ema20[i], ema50[i], vol_sma50[i], atr_arr[i]]):
            continue
        if not in_trade:
            pullback = ema20[i] * 0.98 <= close[i] <= ema20[i] * 1.02
            above_prev = close[i] > high[i - 1] if i > 0 else False
            vol_surge = vol[i] > 1.2 * vol_sma50[i] if vol_sma50[i] > 0 else False
            if pullback and above_prev and vol_surge:
                in_trade = True
                entry_price = close[i]
        else:
            gain = (close[i] - entry_price) / entry_price
            stop_hit = close[i] < ema50[i] - 1.5 * atr_arr[i]
            if gain >= 0.20 or stop_hit:
                trades.append(gain)
                in_trade = False

    if not trades:
        return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "n_trades": 0}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    return {
        "win_rate": len(wins) / len(trades),
        "avg_win": np.mean(wins) if wins else 0.0,
        "avg_loss": abs(np.mean(losses)) if losses else 0.0,
        "n_trades": len(trades),
    }


def _backtest_breakout(df: pd.DataFrame) -> dict:
    """
    Backtest Class III logic over history.
    Entry: Close > DC_Upper_20 AND ADX > 30 (rising) AND Vol > 1.5x avg
    Exit: Close < DC_Mid_10 OR trailing stop (2.5 × ATR from peak)
    """
    trades = []
    in_trade = False
    entry_price = 0.0
    peak_price = 0.0

    close = df["Close"].values
    dc_upper = df["DC_Upper_20"].values
    dc_mid_10 = df["DC_Mid_10"].values
    adx_arr = df["ADX_14"].values
    atr_arr = df["ATR_14"].values
    vol = df["Volume"].values
    vol_sma20 = df["Vol_SMA_20"].values

    for i in range(200, len(df)):
        if any(np.isnan(v) for v in [dc_upper[i], adx_arr[i], atr_arr[i], vol_sma20[i]]):
            continue
        if not in_trade:
            adx_rising = adx_arr[i] > adx_arr[i - 1] if i > 0 and not np.isnan(adx_arr[i - 1]) else False
            if (close[i] > dc_upper[i]
                    and adx_arr[i] > 30 and adx_rising
                    and vol[i] > 1.5 * vol_sma20[i]):
                in_trade = True
                entry_price = close[i]
                peak_price = close[i]
        else:
            peak_price = max(peak_price, close[i])
            trailing_stop = peak_price - 2.5 * atr_arr[i]
            if close[i] < dc_mid_10[i] or close[i] < trailing_stop:
                pnl = (close[i] - entry_price) / entry_price
                trades.append(pnl)
                in_trade = False

    if not trades:
        return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "n_trades": 0}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    return {
        "win_rate": len(wins) / len(trades),
        "avg_win": np.mean(wins) if wins else 0.0,
        "avg_loss": abs(np.mean(losses)) if losses else 0.0,
        "n_trades": len(trades),
    }


# ── Dynamic Kelly ────────────────────────────────────────────────────────────

def _dynamic_kelly(win_rate: float, target_return: float, stop_distance: float,
                   fraction: float = 1.0) -> float:
    """
    Asset-specific Kelly: f* = p - (1-p)/b
    where b = target_return / stop_distance (reward/risk ratio).
    Clamped to [0, 0.25] for safety. Multiplied by fraction for fractional Kelly.
    """
    if stop_distance <= 0 or target_return <= 0:
        return 0.0
    b = target_return / stop_distance
    if b <= 0:
        return 0.0
    f = win_rate - (1 - win_rate) / b
    f = max(0.0, min(0.25, f))  # hard cap at 25%
    return f * fraction


# ── CLASS I: HIGH-PRECISION MEAN REVERSION ───────────────────────────────────

def class1_scan(all_data: dict[str, pd.DataFrame], max_picks: int = 10,
                z_factors: dict = None) -> list[dict]:
    """
    Oversold in Uptrend.
    Entry: Price > SMA200 AND Close <= BB_Lower(20,2σ) AND RSI14 < 30 AND KDJ_J > KDJ_D
    Exit: BB Mid (SMA20) or RSI > 50 or 5-day time exit
    Stop: 1.0 × ATR(14) below entry low
    Sizing: Quarter Kelly (fraction=0.25)
    """
    signals = []
    for ticker, raw_df in all_data.items():
        df = compute_indicators(raw_df)
        if len(df) < 200:
            continue

        close = _latest(df, "Close")
        sma200 = _latest(df, "SMA_200")
        bb_lower = _latest(df, "BB_Lower")
        bb_mid = _latest(df, "BB_Mid")
        rsi_val = _latest(df, "RSI_14")
        kdj_j = _latest(df, "KDJ_J")
        kdj_d = _latest(df, "KDJ_D")
        atr_val = _latest(df, "ATR_14")
        low = _latest(df, "Low")

        if any(np.isnan(v) for v in [close, sma200, bb_lower, bb_mid, rsi_val, kdj_j, kdj_d, atr_val]):
            continue

        # ── Signal status ────────────────────────────────────────────────
        entry_triggered = (
            close > sma200
            and close <= bb_lower
            and rsi_val < 30
            and kdj_j > kdj_d
        )
        hold = (
            close > sma200
            and close > bb_lower
            and close < bb_mid
            and rsi_val < 50
        )
        watch = (
            close > sma200
            and rsi_val < 40
            and close < bb_mid
        )

        if not (entry_triggered or hold or watch):
            continue

        # ── Distance to Mean (margin of safety) ─────────────────────────
        dist_to_mean_pct = (bb_mid - close) / close * 100
        dist_to_lower_pct = (close - bb_lower) / close * 100

        # ── Historical backtest ──────────────────────────────────────────
        bt = _backtest_mean_reversion(df)
        # Use backtested rate if sufficient trades, else fallback to target
        if bt["n_trades"] >= 5:
            win_rate = bt["win_rate"]
            avg_win = bt["avg_win"]
            avg_loss = bt["avg_loss"]
        else:
            win_rate = 0.80
            avg_win = float(bb_mid - close) / close if close > 0 else 0.02
            avg_loss = float(atr_val) / close if close > 0 else 0.01

        # ── Dynamic Kelly (Quarter) ──────────────────────────────────────
        target_return = float(bb_mid - close) / close if close > 0 else 0.0
        stop_distance = float(atr_val) / close if close > 0 else 0.0
        kelly_pct = _dynamic_kelly(win_rate, target_return, stop_distance, fraction=0.25) * 100

        status = "Entry Triggered" if entry_triggered else ("Hold" if hold else "Watch")
        stop_loss = float(low - 1.0 * atr_val)
        target = float(bb_mid)

        # Signal strength using winsorized Z-score of RSI (inverted: lower RSI = stronger)
        rsi_z = z_factors.get("RSI_14", {}).get(ticker, 0) if z_factors else 0
        sig_strength = round(max(0, -rsi_z), 3)  # negative Z = more oversold = stronger signal

        signals.append({
            "Ticker": ticker,
            "Class": "I - Mean Reversion",
            "Status": status,
            "Close": round(close, 2),
            "RSI": round(rsi_val, 1),
            "BB_Lower": round(bb_lower, 2),
            "BB_Mid": round(bb_mid, 2),
            "SMA_200": round(sma200, 2),
            "Dist_to_Mean_%": round(dist_to_mean_pct, 2),
            "Target": round(target, 2),
            "Stop_Loss": round(stop_loss, 2),
            "ATR": round(atr_val, 2),
            "Kelly_%": round(kelly_pct, 2),
            "Win_Rate": round(win_rate * 100, 1),
            "Backtest_Trades": bt["n_trades"],
            "Signal_Strength": sig_strength,
            "Volatility": round(atr_val / close * 100, 2),
        })

    signals.sort(key=lambda x: (-1 if x["Status"] == "Entry Triggered" else 0, -x["Signal_Strength"]))
    return signals[:max_picks]


# ── CLASS II: BALANCED TREND RESONANCE ───────────────────────────────────────

def class2_scan(all_data: dict[str, pd.DataFrame], max_picks: int = 10,
                z_factors: dict = None) -> list[dict]:
    """
    Pullback Continuation.
    Momentum Score: 0.6 * Z(ROC12) + 0.4 * Z(MACD_Hist). Pick top 10%.
    Entry: Price pulls back to EMA20, closes above prev high, Volume > 1.2 × SMA50(Vol).
    Exit: 2.0 × ATR(14) or 20% gain.
    Stop: 1.5 × ATR(14) below EMA50.
    Sizing: Half Kelly (fraction=0.50).

    Universe filter: excludes LOW_VOL_EXCLUDE set.
    Uses pre-winsorized Z-scores from _preprocess_factors.
    """
    # Phase 1: compute indicators for eligible tickers
    ticker_dfs = {}
    for ticker, raw_df in all_data.items():
        if ticker in LOW_VOL_EXCLUDE:
            continue
        df = compute_indicators(raw_df)
        if len(df) < 200:
            continue
        ticker_dfs[ticker] = df

    if not ticker_dfs:
        return []

    # Phase 2: Use pre-computed winsorized Z-scores for momentum ranking
    z_roc = z_factors.get("ROC_12", {}) if z_factors else {}
    z_macd = z_factors.get("MACD_Hist", {}) if z_factors else {}

    # Combined momentum score
    momentum_scores = {}
    for ticker in ticker_dfs:
        r = z_roc.get(ticker, 0)
        m = z_macd.get(ticker, 0)
        momentum_scores[ticker] = 0.6 * r + 0.4 * m

    # Top 10% by momentum
    sorted_tickers = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
    top_n = max(1, len(sorted_tickers) // 10)
    top_tickers = sorted_tickers[:top_n]

    # Phase 3: check entry conditions for top momentum tickers
    signals = []
    for ticker in top_tickers:
        df = ticker_dfs[ticker]
        close = _latest(df, "Close")
        ema20 = _latest(df, "EMA_20")
        ema50 = _latest(df, "EMA_50")
        atr_val = _latest(df, "ATR_14")
        vol = _latest(df, "Volume")
        vol_sma50 = _latest(df, "Vol_SMA_50")
        prev_high = _prev(df, "High")

        if any(np.isnan(v) for v in [close, ema20, ema50, atr_val, vol, vol_sma50]):
            continue

        pullback_to_ema = ema20 * 0.98 <= close <= ema20 * 1.02
        closes_above_prev_high = close > prev_high if not np.isnan(prev_high) else False
        volume_surge = vol > 1.2 * vol_sma50

        entry_triggered = pullback_to_ema and closes_above_prev_high and volume_surge
        hold = close > ema20 and close > ema50
        watch = momentum_scores[ticker] > 0

        if not (entry_triggered or hold or watch):
            continue

        # ── Historical backtest ──────────────────────────────────────────
        bt = _backtest_trend_resonance(df)
        if bt["n_trades"] >= 5:
            win_rate = bt["win_rate"]
        else:
            win_rate = 0.70

        # ── Dynamic Kelly (Half) ─────────────────────────────────────────
        target_return = min(2.0 * atr_val / close, 0.20)
        stop_distance = 1.5 * atr_val / close
        kelly_pct = _dynamic_kelly(win_rate, target_return, stop_distance, fraction=0.50) * 100

        status = "Entry Triggered" if entry_triggered else ("Hold" if hold else "Watch")
        stop_loss = float(ema50 - 1.5 * atr_val)
        target_atr = float(close + 2.0 * atr_val)
        target_20pct = float(close * 1.20)

        signals.append({
            "Ticker": ticker,
            "Class": "II - Trend Resonance",
            "Status": status,
            "Close": round(close, 2),
            "Momentum_Score": round(momentum_scores[ticker], 3),
            "ROC_12_Z": round(z_roc.get(ticker, 0), 3),
            "MACD_Z": round(z_macd.get(ticker, 0), 3),
            "EMA_20": round(ema20, 2),
            "Target_ATR": round(target_atr, 2),
            "Target_20pct": round(target_20pct, 2),
            "Stop_Loss": round(stop_loss, 2),
            "ATR": round(atr_val, 2),
            "Kelly_%": round(kelly_pct, 2),
            "Win_Rate": round(win_rate * 100, 1),
            "Backtest_Trades": bt["n_trades"],
            "Signal_Strength": round(momentum_scores[ticker], 3),
            "Volatility": round(atr_val / close * 100, 2),
        })

    signals.sort(key=lambda x: (-1 if x["Status"] == "Entry Triggered" else 0, -x["Signal_Strength"]))
    return signals[:max_picks]


# ── CLASS III: EXPLOSIVE BREAKOUT ENGINE ─────────────────────────────────────

def class3_scan(all_data: dict[str, pd.DataFrame], max_picks: int = 10,
                z_factors: dict = None) -> list[dict]:
    """
    Fat-Tail Volatility Squeeze.
    Entry: Price > Upper Donchian(20) AND ADX14 > 30 (rising) AND BB Width at 3-month low.
    Validation: Volume > 1.5 × monthly average.
    Exit: Close < 10-day Donchian Mid or 2.5 × ATR(14) trailing stop from peak.
    Sizing: Fractional Kelly (fraction=0.33).

    Universe filter: excludes LOW_VOL_EXCLUDE; requires ATR% > 2%.
    Uses pre-winsorized ADX Z-score for signal strength.
    """
    signals = []
    for ticker, raw_df in all_data.items():
        if ticker in LOW_VOL_EXCLUDE:
            continue
        df = compute_indicators(raw_df)
        if len(df) < 200:
            continue

        close = _latest(df, "Close")
        atr_val = _latest(df, "ATR_14")
        if np.isnan(close) or np.isnan(atr_val) or close <= 0:
            continue

        # ── ATR% gate: only high-volatility assets for breakout ──────────
        atr_pct = atr_val / close * 100
        if atr_pct < 2.0:
            continue

        dc_upper = _latest(df, "DC_Upper_20")
        dc_mid_10 = _latest(df, "DC_Mid_10")
        adx_val = _latest(df, "ADX_14")
        adx_prev = _prev(df, "ADX_14")
        bb_width = _latest(df, "BB_Width")
        bb_width_3m_min = _latest(df, "BB_Width_3m_Min")
        vol = _latest(df, "Volume")
        vol_sma_20 = _latest(df, "Vol_SMA_20")
        high = _latest(df, "High")

        if any(np.isnan(v) for v in [dc_upper, adx_val, bb_width, vol, vol_sma_20]):
            continue

        # ── Entry conditions ─────────────────────────────────────────────
        above_donchian = close > dc_upper
        adx_strong = adx_val > 30
        adx_rising = adx_val > adx_prev if not np.isnan(adx_prev) else False
        bb_squeeze = (bb_width <= bb_width_3m_min * 1.05) if not np.isnan(bb_width_3m_min) else False
        volume_confirm = vol > 1.5 * vol_sma_20

        entry_triggered = above_donchian and adx_strong and adx_rising and bb_squeeze and volume_confirm

        # Near-breakout watch (use parentheses to fix operator precedence bug)
        near_breakout = (
            close > dc_upper * 0.98
            and adx_val > 25
            and (bb_width < bb_width_3m_min * 1.2 if not np.isnan(bb_width_3m_min) else False)
        )

        hold = above_donchian and adx_strong and close > dc_mid_10 if not np.isnan(dc_mid_10) else False

        if not (entry_triggered or hold or near_breakout):
            continue

        # ── Historical backtest ──────────────────────────────────────────
        bt = _backtest_breakout(df)
        if bt["n_trades"] >= 5:
            win_rate = bt["win_rate"]
        else:
            win_rate = 0.60

        # ── Dynamic Kelly (Fractional 0.33) ──────────────────────────────
        target_return = 2.5 * atr_val / close
        stop_distance = 2.5 * atr_val / close
        kelly_pct = _dynamic_kelly(win_rate, target_return, stop_distance, fraction=0.33) * 100

        status = "Entry Triggered" if entry_triggered else ("Hold" if hold else "Watch")
        trailing_stop = float(high - 2.5 * atr_val)
        exit_level = float(dc_mid_10) if not np.isnan(dc_mid_10) else float(close * 0.95)

        # Signal strength using winsorized ADX Z-score (higher = stronger trend)
        adx_z = z_factors.get("ADX_14", {}).get(ticker, 0) if z_factors else 0
        sig_strength = round(max(0, adx_z), 3)

        signals.append({
            "Ticker": ticker,
            "Class": "III - Breakout Engine",
            "Status": status,
            "Close": round(close, 2),
            "Donchian_Upper": round(dc_upper, 2),
            "ADX": round(adx_val, 1),
            "BB_Width": round(bb_width, 4),
            "ATR_%": round(atr_pct, 2),
            "Volume_Ratio": round(vol / vol_sma_20, 2) if vol_sma_20 > 0 else 0,
            "Trailing_Stop": round(trailing_stop, 2),
            "Exit_DC_Mid": round(exit_level, 2),
            "ATR": round(atr_val, 2),
            "Kelly_%": round(kelly_pct, 2),
            "Win_Rate": round(win_rate * 100, 1),
            "Backtest_Trades": bt["n_trades"],
            "Signal_Strength": sig_strength,
            "Volatility": round(atr_pct, 2),
        })

    signals.sort(key=lambda x: (-1 if x["Status"] == "Entry Triggered" else 0, -x["Signal_Strength"]))
    return signals[:max_picks]


# ── Cross-Sectional Factor Preprocessing ─────────────────────────────────────

def _preprocess_factors(all_data: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Winsorize ALL raw factors (RSI, ADX, ROC, MACD_Hist) at [1%, 99%]
    cross-sectionally, then compute Z-scores.
    Returns dict: ticker -> {factor_name: z_score}.
    """
    factor_names = ["RSI_14", "ADX_14", "ROC_12", "MACD_Hist"]
    raw_factors = {f: {} for f in factor_names}
    ticker_dfs = {}

    for ticker, raw_df in all_data.items():
        df = compute_indicators(raw_df)
        if len(df) < 50:
            continue
        ticker_dfs[ticker] = df
        for f in factor_names:
            val = _latest(df, f)
            if not np.isnan(val):
                raw_factors[f][ticker] = val

    # Winsorize each factor cross-sectionally, then Z-score
    z_scores = {f: {} for f in factor_names}
    for f in factor_names:
        if len(raw_factors[f]) < 3:
            z_scores[f] = {k: 0.0 for k in raw_factors[f]}
            continue
        series = pd.Series(raw_factors[f])
        ws = winsorize_series(series, limits=(0.01, 0.01))
        z_scores[f] = cross_sectional_zscore(ws.to_dict())

    return z_scores


# ── UNIFIED SCAN ─────────────────────────────────────────────────────────────

def run_all_strategies(all_data: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    """Run all three strategy classes and return results."""
    # Pre-compute winsorized Z-scores for all factors
    z_factors = _preprocess_factors(all_data)

    return {
        "class1": class1_scan(all_data, z_factors=z_factors),
        "class2": class2_scan(all_data, z_factors=z_factors),
        "class3": class3_scan(all_data, z_factors=z_factors),
    }


def get_top15_tickers(results: dict[str, list[dict]]) -> list[str]:
    """Get top 15 unique tickers across all classes for AV verification."""
    seen = set()
    top = []
    for cls in ["class1", "class2", "class3"]:
        for sig in results.get(cls, []):
            t = sig["Ticker"]
            if t not in seen:
                seen.add(t)
                top.append(t)
            if len(top) >= 15:
                return top
    return top
