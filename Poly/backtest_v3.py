"""
Polymarket 5-Min BTC Bot v3 - 7-Day Backtest
=============================================
Key improvements:
  - 7 days of 1-minute BTC/USDT data from Binance
  - Entry at minute 3 of each 5-min window (last 2 minutes)
  - Uses first 3 candles of the window + historical indicators to predict
  - Intra-window momentum, candle patterns, and volume analysis
  - Walk-forward ML with much larger training set (~2000 windows)

Logic:
  Each 5-min window (e.g. 10:00-10:05):
    - Candle 1: :00-:01  (observe)
    - Candle 2: :01-:02  (observe)
    - Candle 3: :02-:03  (observe)
    → DECISION POINT at :03 → enter trade if confident
    - Candle 4: :03-:04  (in trade)
    - Candle 5: :04-:05  (in trade, resolve)
    
  Outcome: Did the window close price > window open price?

Usage:
    py backtest_v3.py
"""

import asyncio
import aiohttp
import time
import math
import json
import csv
import statistics
import warnings
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# ============================================================================
#  DATA STRUCTURES
# ============================================================================

@dataclass
class Candle:
    ts_open: float    # open timestamp (unix sec)
    ts_close: float   # close timestamp (unix sec)
    open: float
    high: float
    low: float
    close: float
    volume: float     # base asset volume
    trades: int       # number of trades
    taker_buy_vol: float  # taker buy volume


@dataclass
class Window:
    idx: int
    start_ts: float
    end_ts: float
    candles: list       # 5 x 1-min candles in this window
    open_price: float
    close_price: float
    outcome: str        # "UP" or "DOWN"


@dataclass
class TradeSignal:
    direction: str
    confidence: float
    reasoning: str


@dataclass
class TradeResult:
    window_idx: int
    window_start: str
    window_end: str
    open_price: float
    close_price: float
    price_at_entry: float    # price when we enter (at minute 3)
    actual_outcome: str
    predicted: str
    confidence: float
    strategy: str
    pnl: float
    reasoning: str


# ============================================================================
#  DATA FETCHING — 1-Minute Klines from Binance
# ============================================================================

async def fetch_1m_klines(days: int = 7) -> list[Candle]:
    """Fetch 1-minute BTC/USDT klines from Binance for the last N days."""
    base_url = "https://api.binance.com/api/v3/klines"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86400 * 1000

    candles = []
    current_start = start_ms
    batch = 0

    async with aiohttp.ClientSession() as session:
        while current_start < now_ms:
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": current_start,
                "limit": 1000,
            }
            try:
                async with session.get(base_url, params=params) as resp:
                    data = await resp.json()
                    if not data or not isinstance(data, list):
                        break
                    for c in data:
                        candles.append(Candle(
                            ts_open=c[0] / 1000.0,
                            ts_close=c[6] / 1000.0,
                            open=float(c[1]),
                            high=float(c[2]),
                            low=float(c[3]),
                            close=float(c[4]),
                            volume=float(c[5]),
                            trades=int(c[8]),
                            taker_buy_vol=float(c[9]),
                        ))
                    current_start = data[-1][6] + 1
                    batch += 1
                    if batch % 5 == 0:
                        hrs = len(candles) / 60
                        print(f"  [*] Fetched {len(candles):,} candles ({hrs:.1f} hours)...")
                    await asyncio.sleep(0.1)
            except Exception as e:
                print(f"  [!] Fetch error: {e}, retrying...")
                await asyncio.sleep(2)

    print(f"  [+] Total: {len(candles):,} 1-min candles ({len(candles)/1440:.1f} days)")
    return candles


# ============================================================================
#  TECHNICAL INDICATOR LIBRARY
# ============================================================================

def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    mult = 2.0 / (period + 1)
    e = [values[0]]
    for v in values[1:]:
        e.append(v * mult + e[-1] * (1 - mult))
    return e


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    recent = deltas[-period:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    ag = sum(gains) / period if gains else 1e-8
    al = sum(losses) / period if losses else 1e-8
    return 100 - 100 / (1 + ag / al)


def bollinger(closes: list[float], period: int = 20, nstd: float = 2.0):
    if len(closes) < period:
        return 0.5, 0.1
    w = closes[-period:]
    mid = statistics.mean(w)
    sd = statistics.stdev(w) if len(w) > 1 else 1e-8
    upper = mid + nstd * sd
    lower = mid - nstd * sd
    pct_b = (closes[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    bw = (upper - lower) / mid * 100
    return pct_b, bw


def atr(candles_list: list[Candle], period: int = 14) -> float:
    if len(candles_list) < 2:
        return 0.0
    trs = []
    for i in range(1, len(candles_list)):
        tr = max(
            candles_list[i].high - candles_list[i].low,
            abs(candles_list[i].high - candles_list[i-1].close),
            abs(candles_list[i].low - candles_list[i-1].close)
        )
        trs.append(tr)
    return statistics.mean(trs[-period:]) if trs else 0.0


def macd(closes: list[float]):
    if len(closes) < 26:
        return 0, 0, 0
    e12 = ema(closes, 12)
    e26 = ema(closes, 26)
    line = [e12[i] - e26[i] for i in range(len(e26))]
    sig = ema(line, 9)
    return line[-1], sig[-1], line[-1] - sig[-1]


def stochastic(candles_list: list[Candle], period: int = 14) -> float:
    if len(candles_list) < period:
        return 50.0
    r = candles_list[-period:]
    hi = max(c.high for c in r)
    lo = min(c.low for c in r)
    if hi == lo:
        return 50.0
    return (r[-1].close - lo) / (hi - lo) * 100


def williams_r(candles_list: list[Candle], period: int = 14) -> float:
    if len(candles_list) < period:
        return -50.0
    r = candles_list[-period:]
    hi = max(c.high for c in r)
    lo = min(c.low for c in r)
    if hi == lo:
        return -50.0
    return (hi - r[-1].close) / (hi - lo) * -100


def obv_slope(candles_list: list[Candle], lookback: int = 20) -> float:
    if len(candles_list) < lookback:
        return 0.0
    r = candles_list[-lookback:]
    obv_vals = [0.0]
    for i in range(1, len(r)):
        if r[i].close > r[i-1].close:
            obv_vals.append(obv_vals[-1] + r[i].volume)
        elif r[i].close < r[i-1].close:
            obv_vals.append(obv_vals[-1] - r[i].volume)
        else:
            obv_vals.append(obv_vals[-1])
    x = list(range(len(obv_vals)))
    xm = statistics.mean(x)
    ym = statistics.mean(obv_vals)
    num = sum((x[i] - xm) * (obv_vals[i] - ym) for i in range(len(obv_vals)))
    den = sum((x[i] - xm) ** 2 for i in range(len(obv_vals)))
    return num / den if den > 0 else 0.0


def vwap(candles_list: list[Candle], period: int = 20) -> float:
    r = candles_list[-period:]
    tv = sum(c.volume for c in r)
    if tv == 0:
        return r[-1].close
    return sum(((c.high + c.low + c.close) / 3) * c.volume for c in r) / tv


def taker_buy_ratio(candles_list: list[Candle], period: int = 5) -> float:
    """Ratio of taker buy volume to total volume (buy pressure)."""
    r = candles_list[-period:]
    tv = sum(c.volume for c in r)
    tbv = sum(c.taker_buy_vol for c in r)
    return tbv / tv if tv > 0 else 0.5


def trade_intensity(candles_list: list[Candle], period: int = 5) -> float:
    """Average trade count per candle, normalized."""
    r = candles_list[-period:]
    return statistics.mean([c.trades for c in r])


# ============================================================================
#  FEATURE ENGINEERING
# ============================================================================

def build_features(
    hist_candles: list[Candle],     # Historical 1-min candles BEFORE window
    window_candles_3m: list[Candle], # First 3 candles of the current window
) -> dict:
    """
    Build feature vector at the decision point (3 minutes into window).
    Two feature groups:
      A) Historical context (from candles before window open)
      B) Intra-window signals (from first 3 candles of this window)
    """
    f = {}

    if len(hist_candles) < 30 or len(window_candles_3m) < 2:
        return {}

    # Combine for indicators that need the freshest data
    combined = hist_candles[-60:] + window_candles_3m
    closes = [c.close for c in combined]
    hist_closes = [c.close for c in hist_candles]

    # ====== A) HISTORICAL CONTEXT ======

    # 1. RSI (multiple lookbacks)
    f["rsi_14"] = rsi(hist_closes, 14)
    f["rsi_7"] = rsi(hist_closes, 7)

    # 2. EMA alignment
    e9 = ema(hist_closes[-30:], 9)
    e21 = ema(hist_closes[-50:], 21)
    f["ema9_slope"] = (e9[-1] - e9[-3]) / e9[-3] * 1e4 if len(e9) > 3 else 0
    f["ema21_slope"] = (e21[-1] - e21[-3]) / e21[-3] * 1e4 if len(e21) > 3 else 0
    f["ema_cross"] = (e9[-1] - e21[-1]) / hist_closes[-1] * 1e4 if e9 and e21 else 0

    # 3. Bollinger Bands
    bb_pos, bb_bw = bollinger(hist_closes, 20, 2.0)
    f["bb_pct_b"] = bb_pos
    f["bb_bandwidth"] = bb_bw

    # 4. MACD
    ml, sl, mh = macd(hist_closes)
    f["macd_hist"] = mh / hist_closes[-1] * 1e4

    # 5. ATR / Volatility
    a = atr(hist_candles, 14)
    f["atr_pct"] = a / hist_closes[-1] * 100

    # 6. Multi-period returns
    for p in [5, 10, 20, 60]:
        if len(hist_closes) > p:
            f[f"ret_{p}m"] = (hist_closes[-1] - hist_closes[-p]) / hist_closes[-p] * 100
        else:
            f[f"ret_{p}m"] = 0

    # 7. Stochastic + Williams %R
    f["stoch_k"] = stochastic(hist_candles, 14)
    f["williams_r"] = williams_r(hist_candles, 14)

    # 8. OBV slope
    f["obv_slope"] = obv_slope(hist_candles, 20)

    # 9. VWAP deviation
    v = vwap(hist_candles, 20)
    f["vwap_dev"] = (hist_closes[-1] - v) / v * 1e4

    # 10. Taker buy pressure (order flow proxy)
    f["taker_buy_ratio"] = taker_buy_ratio(hist_candles, 10)
    f["taker_buy_ratio_5"] = taker_buy_ratio(hist_candles, 5)

    # 11. Trade intensity
    f["trade_intensity"] = trade_intensity(hist_candles, 10)

    # 12. Volatility regime
    if len(hist_closes) > 60:
        recent_vol = statistics.stdev(hist_closes[-10:]) if len(hist_closes[-10:]) > 1 else 1e-8
        longer_vol = statistics.stdev(hist_closes[-60:]) if len(hist_closes[-60:]) > 1 else 1e-8
        f["vol_ratio"] = recent_vol / longer_vol
    else:
        f["vol_ratio"] = 1.0

    # 13. Z-score
    if len(hist_closes) > 20:
        ma20 = statistics.mean(hist_closes[-20:])
        sd20 = statistics.stdev(hist_closes[-20:]) if len(hist_closes[-20:]) > 1 else 1e-8
        f["z_score"] = (hist_closes[-1] - ma20) / sd20
    else:
        f["z_score"] = 0

    # 14. Time features
    hour_utc = datetime.fromtimestamp(hist_candles[-1].ts_close, tz=timezone.utc).hour
    minute_utc = datetime.fromtimestamp(hist_candles[-1].ts_close, tz=timezone.utc).minute
    f["hour_sin"] = math.sin(2 * math.pi * hour_utc / 24)
    f["hour_cos"] = math.cos(2 * math.pi * hour_utc / 24)
    dow = datetime.fromtimestamp(hist_candles[-1].ts_close, tz=timezone.utc).weekday()
    f["dow_sin"] = math.sin(2 * math.pi * dow / 7)
    f["dow_cos"] = math.cos(2 * math.pi * dow / 7)

    # 15. Previous window outcomes (mean reversion / momentum context)
    # Did the last few 5-min windows go UP or DOWN?
    if len(hist_closes) >= 15:
        prev_5m = (hist_closes[-1] - hist_closes[-5]) / hist_closes[-5] * 100
        prev_10m = (hist_closes[-1] - hist_closes[-10]) / hist_closes[-10] * 100
        prev_15m = (hist_closes[-1] - hist_closes[-15]) / hist_closes[-15] * 100
        f["prev_5m_dir"] = 1 if prev_5m > 0 else -1
        f["prev_10m_dir"] = 1 if prev_10m > 0 else -1
        f["prev_15m_dir"] = 1 if prev_15m > 0 else -1
        f["prev_5m_ret"] = prev_5m
    else:
        f["prev_5m_dir"] = 0
        f["prev_10m_dir"] = 0
        f["prev_15m_dir"] = 0
        f["prev_5m_ret"] = 0

    # ====== B) INTRA-WINDOW FEATURES (first 3 minutes) ======

    wc = window_candles_3m
    window_open = wc[0].open

    # 16. 3-minute return from window open
    f["intra_3m_ret"] = (wc[-1].close - window_open) / window_open * 100

    # 17. Intra-window momentum direction
    f["intra_dir"] = 1 if wc[-1].close > window_open else -1

    # 18. Candle-by-candle direction (pattern)
    dirs = [1 if c.close >= c.open else -1 for c in wc]
    f["intra_candle_consistency"] = sum(dirs) / len(dirs)  # -1 to +1

    # 19. Intra-window high/low relative to open
    intra_high = max(c.high for c in wc)
    intra_low = min(c.low for c in wc)
    f["intra_high_dev"] = (intra_high - window_open) / window_open * 100
    f["intra_low_dev"] = (window_open - intra_low) / window_open * 100

    # 20. Intra-window taker buy ratio (buy pressure during this window)
    intra_vol = sum(c.volume for c in wc)
    intra_buy = sum(c.taker_buy_vol for c in wc)
    f["intra_buy_ratio"] = intra_buy / intra_vol if intra_vol > 0 else 0.5

    # 21. Intra-window volume trend (increasing = conviction)
    if len(wc) >= 3:
        f["intra_vol_trend"] = wc[-1].volume / (wc[0].volume + 1e-8)
    else:
        f["intra_vol_trend"] = 1.0

    # 22. Intra-window trade intensity
    f["intra_trade_intensity"] = sum(c.trades for c in wc) / len(wc)

    # 23. Current price position relative to intra-range
    intra_range = intra_high - intra_low
    if intra_range > 0:
        f["intra_position"] = (wc[-1].close - intra_low) / intra_range
    else:
        f["intra_position"] = 0.5

    # 24. Momentum acceleration (are we speeding up or slowing?)
    if len(wc) >= 3:
        m1 = wc[1].close - wc[0].close
        m2 = wc[2].close - wc[1].close
        f["intra_accel"] = (m2 - m1) / (abs(m1) + 1e-8)
    else:
        f["intra_accel"] = 0

    # 25. Body/wick ratio (conviction of moves)
    bodies = [abs(c.close - c.open) for c in wc]
    wicks = [c.high - c.low for c in wc]
    f["intra_body_wick"] = sum(bodies) / (sum(wicks) + 1e-8)

    # 26. Combined signal: intra direction + historical momentum alignment
    hist_mom = f.get("ret_5m", 0)
    f["alignment"] = f["intra_dir"] * (1 if hist_mom > 0 else -1)

    return f


# ============================================================================
#  STRATEGY A: CONFLUENCE (Enhanced Rule-Based)
# ============================================================================

class ConfluenceStrategy:
    """
    Multi-indicator scoring with intra-window confirmation.
    Key: use intra-window data as PRIMARY signal, historical as FILTER.
    """

    def __init__(self):
        self.name = "Confluence"
        self.min_score = 2.8
        self.min_confidence = 0.54

    def evaluate(self, features: dict) -> Optional[TradeSignal]:
        if not features:
            return None

        score = 0.0
        reasons = []

        # ── PRIMARY: Intra-window signals (3 min of the current window)

        # P1. 3-minute intra return: the biggest signal
        intra_ret = features.get("intra_3m_ret", 0)
        if abs(intra_ret) > 0.02:
            s = 1.5 * (1 if intra_ret > 0 else -1)
            score += s
            reasons.append(f"3m move {intra_ret:+.3f}%")

        # P2. Candle consistency: all 3 candles same direction = strong
        consistency = features.get("intra_candle_consistency", 0)
        if abs(consistency) > 0.6:
            s = 1.2 * (1 if consistency > 0 else -1)
            score += s
            reasons.append(f"Candle streak {consistency:+.1f}")

        # P3. Intra buy pressure: >55% = bullish, <45% = bearish
        buy_ratio = features.get("intra_buy_ratio", 0.5)
        if buy_ratio > 0.55:
            score += 0.8
            reasons.append(f"Buy pressure {buy_ratio:.0%}")
        elif buy_ratio < 0.45:
            score -= 0.8
            reasons.append(f"Sell pressure {buy_ratio:.0%}")

        # P4. Momentum acceleration
        accel = features.get("intra_accel", 0)
        if abs(accel) > 0.5:
            s = 0.6 * (1 if accel > 0 else -1)
            # Only add if it agrees with current intra direction
            if s * (1 if intra_ret > 0 else -1 if intra_ret < 0 else 0) > 0:
                score += s
                reasons.append(f"Accelerating {'up' if accel > 0 else 'down'}")

        # P5. Intra-window price position (close to high = bullish)
        pos = features.get("intra_position", 0.5)
        if pos > 0.75:
            score += 0.5
        elif pos < 0.25:
            score -= 0.5

        # ── FILTERS: Historical indicators (must not contradict)

        # F1. RSI extreme filter: skip if RSI would counter our direction
        rsi_val = features.get("rsi_14", 50)
        if score > 0 and rsi_val > 78:  # Betting UP but RSI overbought → caution
            score -= 1.0
            reasons.append(f"RSI caution ({rsi_val:.0f})")
        elif score < 0 and rsi_val < 22:  # Betting DOWN but RSI oversold → caution
            score += 1.0
            reasons.append(f"RSI caution ({rsi_val:.0f})")

        # F2. Bollinger Band confirmation
        bb_pos_val = features.get("bb_pct_b", 0.5)
        if score > 0 and bb_pos_val > 0.95:
            score -= 0.5  # Already at top of band
        elif score < 0 and bb_pos_val < 0.05:
            score += 0.5  # Already at bottom

        # F3. Volatility filter: skip extremely dead or choppy markets
        atr_pct = features.get("atr_pct", 0)
        if atr_pct < 0.002 or atr_pct > 0.25:
            return None

        # F4. Volume confirmation
        vol_trend = features.get("intra_vol_trend", 1.0)
        if vol_trend > 1.5 and abs(score) > 1:
            score *= 1.1  # Volume confirming the move
            reasons.append("Volume confirming")

        # F5. Historical alignment bonus
        alignment = features.get("alignment", 0)
        if alignment > 0:
            score *= 1.08  # Historical trend matches intra-window
            reasons.append("Trend aligned")

        # ── Decision
        abs_score = abs(score)
        if abs_score < self.min_score:
            return None

        direction = "UP" if score > 0 else "DOWN"
        confidence = min(0.88, 0.50 + abs_score * 0.05)

        if confidence < self.min_confidence:
            return None

        return TradeSignal(
            direction=direction,
            confidence=confidence,
            reasoning=f"Score {score:+.1f}: " + ", ".join(reasons[:5]),
        )


# ============================================================================
#  STRATEGY B: ML ENSEMBLE (Walk-Forward)
# ============================================================================

FEATURE_NAMES = [
    "rsi_14", "rsi_7", "ema9_slope", "ema21_slope", "ema_cross",
    "bb_pct_b", "bb_bandwidth", "macd_hist", "atr_pct",
    "ret_5m", "ret_10m", "ret_20m", "ret_60m",
    "stoch_k", "williams_r", "obv_slope", "vwap_dev",
    "taker_buy_ratio", "taker_buy_ratio_5", "trade_intensity",
    "vol_ratio", "z_score",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "prev_5m_dir", "prev_10m_dir", "prev_15m_dir", "prev_5m_ret",
    # Intra-window features (the key differentiators)
    "intra_3m_ret", "intra_dir", "intra_candle_consistency",
    "intra_high_dev", "intra_low_dev", "intra_buy_ratio",
    "intra_vol_trend", "intra_trade_intensity", "intra_position",
    "intra_accel", "intra_body_wick", "alignment",
]


class MLEnsembleStrategy:
    """
    Walk-forward ML with larger training window and better model tuning.
    Uses GB + RF + AdaBoost with soft voting, calibrated probabilities.
    """

    def __init__(self):
        self.name = "ML_Ensemble"

        self.gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.06,
            subsample=0.8, min_samples_leaf=8, random_state=42
        )
        self.rf = RandomForestClassifier(
            n_estimators=250, max_depth=5, min_samples_leaf=8,
            random_state=42, n_jobs=-1
        )
        self.ada = AdaBoostClassifier(
            n_estimators=100, learning_rate=0.08, random_state=42
        )

        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_train = 120            # Need 120 windows before first prediction
        self.retrain_every = 40         # Retrain every 40 windows
        self.confidence_threshold = 0.55
        self.X_history = []
        self.y_history = []
        self._since_train = 0

    def _to_array(self, features: dict) -> np.ndarray:
        return np.array([features.get(f, 0.0) for f in FEATURE_NAMES], dtype=np.float64)

    def add_sample(self, features: dict, outcome: str):
        if not features:
            return
        self.X_history.append(self._to_array(features))
        self.y_history.append(1 if outcome == "UP" else 0)
        self._since_train += 1

    def should_retrain(self) -> bool:
        return (len(self.X_history) >= self.min_train
                and self._since_train >= self.retrain_every)

    def train(self):
        X = np.array(self.X_history)
        y = np.array(self.y_history)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler.fit(X)
        Xs = self.scaler.transform(X)

        self.gb.fit(Xs, y)
        self.rf.fit(Xs, y)
        self.ada.fit(Xs, y)

        self.is_trained = True
        self._since_train = 0

    def predict(self, features: dict) -> Optional[TradeSignal]:
        if not self.is_trained or not features:
            return None

        x = self._to_array(features).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xs = self.scaler.transform(x)

        # Ensemble: weighted soft vote
        gb_p = self.gb.predict_proba(xs)[0]
        rf_p = self.rf.predict_proba(xs)[0]
        ada_p = self.ada.predict_proba(xs)[0]

        # Weights: GB=40%, RF=35%, AdaBoost=25%
        ensemble = 0.40 * gb_p + 0.35 * rf_p + 0.25 * ada_p
        prob_up = ensemble[1]

        if prob_up > 0.5:
            direction = "UP"
            confidence = prob_up
        else:
            direction = "DOWN"
            confidence = 1 - prob_up

        if confidence < self.confidence_threshold:
            return None

        # Feature importance for reasoning
        try:
            imp = self.gb.feature_importances_
            top3 = np.argsort(imp)[-3:][::-1]
            top_str = ", ".join(f"{FEATURE_NAMES[i]}" for i in top3)
            reason = f"ML conf={confidence:.1%} top:[{top_str}]"
        except Exception:
            reason = f"ML conf={confidence:.1%}"

        return TradeSignal(direction=direction, confidence=confidence, reasoning=reason)


# ============================================================================
#  BACKTEST ENGINE
# ============================================================================

def build_windows(candles: list[Candle]) -> list[Window]:
    """Build 5-minute windows aligned to 5-min boundaries."""
    first_ts = candles[0].ts_open
    last_ts = candles[-1].ts_close

    # Align to 5-minute boundary
    start = first_ts - (first_ts % 300) + 300
    windows = []

    while start + 300 <= last_ts:
        end = start + 300
        wc = [c for c in candles if start <= c.ts_open < end]

        if len(wc) >= 4:  # Need at least 4 candles (we use first 3 for prediction)
            windows.append(Window(
                idx=len(windows),
                start_ts=start,
                end_ts=end,
                candles=wc,
                open_price=wc[0].open,
                close_price=wc[-1].close,
                outcome="UP" if wc[-1].close >= wc[0].open else "DOWN",
            ))
        start += 300

    return windows


def run_full_backtest(candles: list[Candle]):
    """Run both strategies with intra-window entry at minute 3."""

    windows = build_windows(candles)
    print(f"  [*] Total 5-min windows: {len(windows)}")
    up = sum(1 for w in windows if w.outcome == "UP")
    print(f"  [*] Base rate: {up}/{len(windows)} UP ({up/len(windows)*100:.1f}%)")

    confluence = ConfluenceStrategy()
    ml_strat = MLEnsembleStrategy()

    results = {"Confluence": [], "ML_Ensemble": []}

    # P&L model: bet $10 per trade
    # Polymarket vig ~2%: buy at 0.52, win pays 1.00
    BET = 10.0
    WIN_PROFIT = BET * (1.0 / 0.52 - 1.0)  # $9.23
    LOSS = -BET

    for i, w in enumerate(windows):
        wc = w.candles

        # First 3 candles = our observation period
        first_3 = wc[:3]
        if len(first_3) < 2:
            continue

        # Historical candles: everything before this window
        hist = [c for c in candles if c.ts_close <= w.start_ts]
        hist = hist[-120:]  # Last 2 hours of 1-min candles

        if len(hist) < 30:
            # Add sample anyway for ML warmup
            features = build_features(hist if len(hist) >= 30 else hist + first_3, first_3)
            ml_strat.add_sample(features, w.outcome)
            if ml_strat.should_retrain():
                ml_strat.train()
            continue

        # Build features
        features = build_features(hist, first_3)

        if not features:
            ml_strat.add_sample({}, w.outcome)
            if ml_strat.should_retrain():
                ml_strat.train()
            continue

        entry_price = first_3[-1].close  # Price at minute 3

        ts_start = datetime.fromtimestamp(w.start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        ts_end = datetime.fromtimestamp(w.end_ts, tz=timezone.utc).strftime("%H:%M")

        def make_result(sig, strat_name):
            if sig is None:
                return None
            win = sig.direction == w.outcome
            pnl = round(WIN_PROFIT if win else LOSS, 2)
            return TradeResult(
                window_idx=w.idx,
                window_start=ts_start,
                window_end=ts_end,
                open_price=round(w.open_price, 2),
                close_price=round(w.close_price, 2),
                price_at_entry=round(entry_price, 2),
                actual_outcome=w.outcome,
                predicted=sig.direction,
                confidence=round(sig.confidence, 4),
                strategy=strat_name,
                pnl=pnl,
                reasoning=sig.reasoning,
            )

        # Confluence
        sig_c = confluence.evaluate(features)
        res = make_result(sig_c, "Confluence")
        if res:
            results["Confluence"].append(res)

        # ML: predict first, then add to training
        sig_ml = ml_strat.predict(features)
        res = make_result(sig_ml, "ML_Ensemble")
        if res:
            results["ML_Ensemble"].append(res)

        ml_strat.add_sample(features, w.outcome)
        if ml_strat.should_retrain():
            ml_strat.train()
            if (i + 1) % 200 == 0:
                print(f"  [*] ML retrained at window {i+1} (samples: {len(ml_strat.X_history)})")

        if (i + 1) % 500 == 0:
            print(f"  [*] Processed {i+1}/{len(windows)} windows...")

    return results, windows


# ============================================================================
#  REPORTING
# ============================================================================

def strategy_stats(trades: list[TradeResult]) -> dict:
    if not trades:
        return {}
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    wr = len(wins) / len(trades) * 100

    running = []
    cum = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cum += t.pnl
        running.append(cum)
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    # Streaks
    max_ws = max_ls = 0
    cs = 0; ct = None
    for t in trades:
        if t.pnl > 0:
            if ct == "w": cs += 1
            else: cs = 1; ct = "w"
            max_ws = max(max_ws, cs)
        else:
            if ct == "l": cs += 1
            else: cs = 1; ct = "l"
            max_ls = max(max_ls, cs)

    tw = sum(t.pnl for t in wins)
    tl = sum(t.pnl for t in losses)
    pf = abs(tw / tl) if tl != 0 else float('inf')
    exp = total_pnl / len(trades)

    pstd = statistics.stdev([t.pnl for t in trades]) if len(trades) > 1 else 1
    sharpe = (exp / pstd * math.sqrt(len(trades))) if pstd > 0 else 0

    # Daily breakdown
    daily = {}
    for t in trades:
        day = t.window_start[:10]
        if day not in daily:
            daily[day] = {"trades": 0, "wins": 0, "pnl": 0.0}
        daily[day]["trades"] += 1
        if t.pnl > 0:
            daily[day]["wins"] += 1
        daily[day]["pnl"] += t.pnl

    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": wr, "total_pnl": total_pnl, "expectancy": exp,
        "profit_factor": pf, "sharpe": sharpe,
        "max_dd": max_dd, "max_ws": max_ws, "max_ls": max_ls,
        "avg_win": statistics.mean([t.pnl for t in wins]) if wins else 0,
        "avg_loss": statistics.mean([t.pnl for t in losses]) if losses else 0,
        "running": running, "daily": daily,
        "avg_conf": statistics.mean([t.confidence for t in trades]),
    }


def print_report(all_results: dict, windows: list[Window]):
    total = len(windows)
    up = sum(1 for w in windows if w.outcome == "UP")
    sep = "=" * 72

    print(f"\n{sep}")
    print(f"  POLYMARKET 5-MIN BTC BACKTEST v3 -- 7-DAY RESULTS")
    start_dt = datetime.fromtimestamp(windows[0].start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(windows[-1].end_ts, tz=timezone.utc)
    print(f"  Period: {start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Windows: {total} | UP: {up} ({up/total*100:.1f}%) | DOWN: {total-up} ({(total-up)/total*100:.1f}%)")
    print(f"  Entry: Minute 3 of each 5-min window (last 2 min for trade)")
    print(f"  Bet size: $10 per trade | Polymarket vig: 2%")
    print(f"{sep}")

    strats = ["Confluence", "ML_Ensemble"]
    all_stats = {}

    for name in strats:
        trades = all_results.get(name, [])
        stats = strategy_stats(trades)
        all_stats[name] = stats

        print(f"\n  {'='*64}")
        print(f"  STRATEGY: {name}")
        print(f"  {'='*64}")

        if not stats:
            print(f"  No trades generated.\n")
            continue

        print(f"  Trades:           {stats['trades']:>6}  ({stats['trades']/total*100:.1f}% of windows)")
        print(f"  Win/Loss:         {stats['wins']:>3}/{stats['losses']:<3}")
        print(f"  Win Rate:         {stats['win_rate']:>6.1f}%")
        print(f"  Total P&L:        ${stats['total_pnl']:>+9.2f}")
        print(f"  Expectancy:       ${stats['expectancy']:>+9.2f} / trade")
        print(f"  Profit Factor:    {stats['profit_factor']:>9.2f}")
        print(f"  Sharpe (trade):   {stats['sharpe']:>9.2f}")
        print(f"  Avg Win:          ${stats['avg_win']:>+9.2f}")
        print(f"  Avg Loss:         ${stats['avg_loss']:>+9.2f}")
        print(f"  Avg Confidence:   {stats['avg_conf']:>8.1%}")
        print(f"  Max Drawdown:     ${stats['max_dd']:>9.2f}")
        print(f"  Max Win Streak:   {stats['max_ws']:>6}")
        print(f"  Max Loss Streak:  {stats['max_ls']:>6}")

        # Daily breakdown
        if stats["daily"]:
            print(f"\n  --- Daily Breakdown ---")
            print(f"  {'Date':<12} {'Trades':>7} {'Win%':>7} {'P&L':>10}")
            print(f"  {'-'*38}")
            for day in sorted(stats["daily"]):
                d = stats["daily"][day]
                wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
                print(f"  {day:<12} {d['trades']:>7} {wr:>6.1f}% ${d['pnl']:>+9.2f}")

    # Head-to-head
    print(f"\n{sep}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{sep}")
    print(f"  {'Metric':<22} {'Confluence':>14} {'ML Ensemble':>14}")
    print(f"  {'-'*50}")

    for label, key, fmt in [
        ("Trades", "trades", "{}"),
        ("Win Rate", "win_rate", "{:.1f}%"),
        ("Total P&L", "total_pnl", "${:+.2f}"),
        ("Expectancy", "expectancy", "${:+.2f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Max Drawdown", "max_dd", "${:.2f}"),
        ("Avg Confidence", "avg_conf", "{:.1%}"),
    ]:
        vals = []
        for name in strats:
            s = all_stats.get(name, {})
            v = s.get(key, 0)
            vals.append(fmt.format(v))
        print(f"  {label:<22} {vals[0]:>14} {vals[1]:>14}")

    # Equity curves
    print(f"\n{sep}")
    print(f"  EQUITY CURVES")
    print(f"{sep}")

    for name in strats:
        trades = all_results.get(name, [])
        stats = all_stats.get(name, {})
        running = stats.get("running", [])
        if not running:
            print(f"\n  [{name}] No trades")
            continue

        mn = min(running)
        mx = max(running)
        rng = mx - mn if mx != mn else 1
        cw = 40

        print(f"\n  [{name}] ({len(trades)} trades)")
        step = max(1, len(running) // 20)
        for i in range(0, len(running), step):
            val = running[i]
            pos = int((val - mn) / rng * cw)
            bar = " " * pos + "*"
            day_ts = trades[i].window_start[-5:]
            print(f"    {trades[i].window_start[:10]} {day_ts} ${val:>+9.2f}  |{bar}")
        val = running[-1]
        pos = int((val - mn) / rng * cw)
        bar = " " * pos + "*"
        print(f"    {'FINAL':>16} ${val:>+9.2f}  |{bar}")

    print(f"\n{sep}")
    print(f"  BACKTEST COMPLETE")
    print(f"{sep}\n")


def save_csv(all_results: dict, base: str):
    keys = [
        "window_idx", "window_start", "window_end", "open_price", "close_price",
        "price_at_entry", "actual_outcome", "predicted", "confidence",
        "strategy", "pnl", "reasoning"
    ]
    for name, trades in all_results.items():
        if not trades:
            continue
        fp = f"{base}_{name.lower()}.csv"
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for t in trades:
                w.writerow({k: getattr(t, k) for k in keys})
        print(f"  [+] Saved {len(trades)} trades -> {fp}")


# ============================================================================
#  MAIN
# ============================================================================

async def main():
    print("=" * 72)
    print("  POLYMARKET 5-MIN BTC BACKTESTER v3")
    print("  7-Day Test | Entry at Minute 3 | 1-Min Candles")
    print("  Confluence + ML Ensemble")
    print("=" * 72)
    print()

    print("  [1/4] Fetching 7 days of 1-minute BTC/USDT data...")
    candles = await fetch_1m_klines(days=7)

    if len(candles) < 500:
        print("  [!] Not enough data!")
        return

    prices = [c.close for c in candles]
    ft = datetime.fromtimestamp(candles[0].ts_open, tz=timezone.utc)
    lt = datetime.fromtimestamp(candles[-1].ts_close, tz=timezone.utc)
    print(f"\n  Period: {ft.strftime('%Y-%m-%d %H:%M')} -> {lt.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  BTC Range: ${min(prices):,.2f} - ${max(prices):,.2f}")
    print(f"  Candles: {len(candles):,}\n")

    print("  [2/4] Running backtest (Confluence + ML)...")
    results, windows = run_full_backtest(candles)

    print("\n  [3/4] Generating report...")
    print_report(results, windows)

    print("  [4/4] Saving CSVs...")
    save_csv(results, str(Path(__file__).parent / "backtest_v3"))


if __name__ == "__main__":
    asyncio.run(main())
