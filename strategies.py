"""
Three strategy classes for the Momentum Engine.

Professional quant logic:
- Empirical backtested win rates (n_trades shown, <30 = Unconfirmed)
- Dynamic Kelly via actual RRR: b = (Target-Close)/(Close-StopLoss)
- Cross-sectional Winsorization [1%,99%] on all factors before Z-scoring
- 5-day time exit for Class I mean reversion
- Liquidity filter: AvgDailyVolume > $10M, ATR% > 1% for Class II/III
- ATR% > 2% gate for Class III breakout candidates
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from indicators import (
    compute_indicators, cross_sectional_zscore, winsorize_series,
    exponential_weights, sma,
)

# ── Assets excluded from momentum strategies ────────────────────────────────
LOW_VOL_EXCLUDE = {
    "SGOV", "BIL", "SHV", "SHY", "IEF", "TLT", "VGSH", "BSV", "AGG",
    "BND", "GOVT", "SCHO", "SPMO",
}

# Minimum sample size for confirmed signals
MIN_CONFIRMED_TRADES = 30


def _latest(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    return float(s.iloc[-1]) if len(s) > 0 else np.nan


def _prev(df: pd.DataFrame, col: str):
    s = df[col].dropna()
    return float(s.iloc[-2]) if len(s) > 1 else np.nan


def _avg_dollar_volume(df: pd.DataFrame, window: int = 20) -> float:
    """Average daily dollar volume over last `window` days."""
    if len(df) < window:
        return 0.0
    recent = df.tail(window)
    dv = (recent["Close"] * recent["Volume"]).mean()
    return float(dv) if not np.isnan(dv) else 0.0


# ── Historical Backtest Engine ───────────────────────────────────────────────

def _backtest_mean_reversion(df: pd.DataFrame, max_hold_days: int = 5) -> dict:
    """
    Backtest Class I: Oversold-in-uptrend mean reversion.
    Entry: Close <= BB_Lower AND RSI < 30 AND Price > SMA200 AND KDJ_J > KDJ_D
    Exit: Close >= BB_Mid OR RSI > 50 OR 5-day time exit OR stop loss hit
    Stop: 1.0 × ATR below entry low
    """
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_stop = 0.0
    bars_held = 0

    close = df["Close"].values
    low_arr = df["Low"].values
    sma200 = df["SMA_200"].values
    bb_lower = df["BB_Lower"].values
    bb_mid = df["BB_Mid"].values
    rsi_arr = df["RSI_14"].values
    kdj_j = df["KDJ_J"].values
    kdj_d = df["KDJ_D"].values
    atr_arr = df["ATR_14"].values

    for i in range(200, len(df)):
        if np.isnan(sma200[i]) or np.isnan(bb_lower[i]) or np.isnan(rsi_arr[i]) or np.isnan(atr_arr[i]):
            continue
        if not in_trade:
            if (close[i] > sma200[i]
                    and close[i] <= bb_lower[i]
                    and rsi_arr[i] < 30
                    and not np.isnan(kdj_j[i]) and not np.isnan(kdj_d[i])
                    and kdj_j[i] > kdj_d[i]):
                in_trade = True
                entry_price = close[i]
                entry_stop = low_arr[i] - 1.0 * atr_arr[i]
                bars_held = 0
        else:
            bars_held += 1
            if (close[i] >= bb_mid[i]
                    or rsi_arr[i] > 50
                    or bars_held >= max_hold_days
                    or close[i] <= entry_stop):
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


def _backtest_trend_resonance(df: pd.DataFrame) -> dict:
    """
    Backtest Class II: Pullback continuation.
    Entry: Close near EMA20 (±2%), closes above prev high, Vol > 1.2x SMA50(Vol)
    Exit: 2.0 × ATR profit OR 20% gain OR stop (1.5 × ATR below EMA50)
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
            atr_target = (entry_price + 2.0 * atr_arr[i])
            stop_hit = close[i] < ema50[i] - 1.5 * atr_arr[i]
            if close[i] >= atr_target or gain >= 0.20 or stop_hit:
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
    Backtest Class III: Volatility squeeze breakout.
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
            if not np.isnan(dc_mid_10[i]) and (close[i] < dc_mid_10[i] or close[i] < trailing_stop):
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


# ── Dynamic Kelly via RRR ────────────────────────────────────────────────────

def _kelly_rrr(win_rate: float, target: float, close: float, stop_loss: float,
               fraction: float = 1.0) -> float:
    """
    Kelly% = p - (1-p)/b
    where b = (Target - Close) / (Close - StopLoss)  [Reward-to-Risk Ratio]
    Clamped to [0, 25%]. Multiplied by fraction for fractional Kelly.
    """
    risk = close - stop_loss
    reward = target - close
    if risk <= 0 or reward <= 0:
        return 0.0
    b = reward / risk
    f = win_rate - (1 - win_rate) / b
    f = max(0.0, min(0.25, f))
    return f * fraction


# ── CLASS I: HIGH-PRECISION MEAN REVERSION ───────────────────────────────────

def class1_scan(all_data: dict[str, pd.DataFrame], max_picks: int = 10,
                z_factors: dict = None) -> list[dict]:
    """
    Oversold in Uptrend. 5-day time exit.
    Entry: Price > SMA200 AND Close <= BB_Lower AND RSI14 < 30 AND KDJ_J > KDJ_D
    Exit: BB Mid or RSI > 50 or 5-day forced exit
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

        entry_triggered = (
            close > sma200
            and close <= bb_lower
            and rsi_val < 30
            and kdj_j > kdj_d
        )
        hold = (close > sma200 and close > bb_lower and close < bb_mid and rsi_val < 50)
        watch = (close > sma200 and rsi_val < 40 and close < bb_mid)

        if not (entry_triggered or hold or watch):
            continue

        # ── Distance to Mean ─────────────────────────────────────────────
        dist_to_mean_pct = (bb_mid - close) / close * 100

        # ── Empirical backtest ───────────────────────────────────────────
        bt = _backtest_mean_reversion(df, max_hold_days=5)
        win_rate = bt["win_rate"] if bt["n_trades"] >= 3 else 0.80
        confirmed = bt["n_trades"] >= MIN_CONFIRMED_TRADES

        # ── Kelly via RRR ────────────────────────────────────────────────
        target = float(bb_mid)
        stop_loss = float(low - 1.0 * atr_val)
        kelly_pct = _kelly_rrr(win_rate, target, close, stop_loss, fraction=0.25) * 100

        status = "Entry Triggered" if entry_triggered else ("Hold" if hold else "Watch")

        rsi_z = z_factors.get("RSI_14", {}).get(ticker, 0) if z_factors else 0
        sig_strength = round(max(0, -rsi_z), 3)

        signals.append({
            "Ticker": ticker,
            "Class": "I - Mean Reversion",
            "Status": status,
            "Confirmed": "Yes" if confirmed else "Unconfirmed",
            "Close": round(close, 2),
            "RSI": round(rsi_val, 1),
            "BB_Lower": round(bb_lower, 2),
            "BB_Mid": round(bb_mid, 2),
            "SMA_200": round(sma200, 2),
            "Dist_to_Mean_%": round(dist_to_mean_pct, 2),
            "Target": round(target, 2),
            "Stop_Loss": round(stop_loss, 2),
            "RRR": round((target - close) / (close - stop_loss), 2) if close > stop_loss else 0,
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
    Momentum Score: 0.6 * Z(ROC12) + 0.4 * Z(MACD_Hist). Top 10%.
    Liquidity filter: AvgDailyVolume > $10M, ATR% > 1%.
    """
    ticker_dfs = {}
    for ticker, raw_df in all_data.items():
        if ticker in LOW_VOL_EXCLUDE:
            continue
        df = compute_indicators(raw_df)
        if len(df) < 200:
            continue

        # ── Liquidity filter ─────────────────────────────────────────────
        adv = _avg_dollar_volume(df)
        if adv < 10_000_000:
            continue
        atr_pct = _latest(df, "ATR_14") / _latest(df, "Close") * 100
        if np.isnan(atr_pct) or atr_pct < 1.0:
            continue

        ticker_dfs[ticker] = df

    if not ticker_dfs:
        return []

    z_roc = z_factors.get("ROC_12", {}) if z_factors else {}
    z_macd = z_factors.get("MACD_Hist", {}) if z_factors else {}

    momentum_scores = {}
    for ticker in ticker_dfs:
        r = z_roc.get(ticker, 0)
        m = z_macd.get(ticker, 0)
        momentum_scores[ticker] = 0.6 * r + 0.4 * m

    sorted_tickers = sorted(momentum_scores, key=momentum_scores.get, reverse=True)
    top_n = max(1, len(sorted_tickers) // 10)
    top_tickers = sorted_tickers[:top_n]

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

        bt = _backtest_trend_resonance(df)
        win_rate = bt["win_rate"] if bt["n_trades"] >= 3 else 0.70
        confirmed = bt["n_trades"] >= MIN_CONFIRMED_TRADES

        stop_loss = float(ema50 - 1.5 * atr_val)
        target_atr = float(close + 2.0 * atr_val)
        target_20pct = float(close * 1.20)
        target = min(target_atr, target_20pct)

        kelly_pct = _kelly_rrr(win_rate, target, close, stop_loss, fraction=0.50) * 100

        status = "Entry Triggered" if entry_triggered else ("Hold" if hold else "Watch")

        signals.append({
            "Ticker": ticker,
            "Class": "II - Trend Resonance",
            "Status": status,
            "Confirmed": "Yes" if confirmed else "Unconfirmed",
            "Close": round(close, 2),
            "Momentum_Score": round(momentum_scores[ticker], 3),
            "ROC_12_Z": round(z_roc.get(ticker, 0), 3),
            "MACD_Z": round(z_macd.get(ticker, 0), 3),
            "EMA_20": round(ema20, 2),
            "Target": round(target, 2),
            "Stop_Loss": round(stop_loss, 2),
            "RRR": round((target - close) / (close - stop_loss), 2) if close > stop_loss else 0,
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
    Liquidity filter: AvgDailyVolume > $10M, ATR% > 2%.
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

        # ── Liquidity + volatility filter ────────────────────────────────
        adv = _avg_dollar_volume(df)
        if adv < 10_000_000:
            continue
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

        above_donchian = close > dc_upper
        adx_strong = adx_val > 30
        adx_rising = adx_val > adx_prev if not np.isnan(adx_prev) else False
        bb_squeeze = (bb_width <= bb_width_3m_min * 1.05) if not np.isnan(bb_width_3m_min) else False
        volume_confirm = vol > 1.5 * vol_sma_20

        entry_triggered = above_donchian and adx_strong and adx_rising and bb_squeeze and volume_confirm

        near_breakout = (
            close > dc_upper * 0.98
            and adx_val > 25
            and (bb_width < bb_width_3m_min * 1.2 if not np.isnan(bb_width_3m_min) else False)
        )

        hold = (above_donchian and adx_strong
                and (close > dc_mid_10 if not np.isnan(dc_mid_10) else False))

        if not (entry_triggered or hold or near_breakout):
            continue

        bt = _backtest_breakout(df)
        win_rate = bt["win_rate"] if bt["n_trades"] >= 3 else 0.60
        confirmed = bt["n_trades"] >= MIN_CONFIRMED_TRADES

        trailing_stop = float(high - 2.5 * atr_val)
        exit_level = float(dc_mid_10) if not np.isnan(dc_mid_10) else float(close * 0.95)
        target = float(close + 2.5 * atr_val)
        stop_loss = trailing_stop

        kelly_pct = _kelly_rrr(win_rate, target, close, stop_loss, fraction=0.33) * 100

        status = "Entry Triggered" if entry_triggered else ("Hold" if hold else "Watch")

        adx_z = z_factors.get("ADX_14", {}).get(ticker, 0) if z_factors else 0
        sig_strength = round(max(0, adx_z), 3)

        signals.append({
            "Ticker": ticker,
            "Class": "III - Breakout Engine",
            "Status": status,
            "Confirmed": "Yes" if confirmed else "Unconfirmed",
            "Close": round(close, 2),
            "Donchian_Upper": round(dc_upper, 2),
            "ADX": round(adx_val, 1),
            "BB_Width": round(bb_width, 4),
            "ATR_%": round(atr_pct, 2),
            "Volume_Ratio": round(vol / vol_sma_20, 2) if vol_sma_20 > 0 else 0,
            "Target": round(target, 2),
            "Trailing_Stop": round(trailing_stop, 2),
            "Exit_DC_Mid": round(exit_level, 2),
            "RRR": round((target - close) / (close - stop_loss), 2) if close > stop_loss else 0,
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
    """
    factor_names = ["RSI_14", "ADX_14", "ROC_12", "MACD_Hist"]
    raw_factors = {f: {} for f in factor_names}

    for ticker, raw_df in all_data.items():
        df = compute_indicators(raw_df)
        if len(df) < 50:
            continue
        for f in factor_names:
            val = _latest(df, f)
            if not np.isnan(val):
                raw_factors[f][ticker] = val

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
    z_factors = _preprocess_factors(all_data)
    return {
        "class1": class1_scan(all_data, z_factors=z_factors),
        "class2": class2_scan(all_data, z_factors=z_factors),
        "class3": class3_scan(all_data, z_factors=z_factors),
    }


def get_top15_tickers(results: dict[str, list[dict]]) -> list[str]:
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
