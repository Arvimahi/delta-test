"""
Polymarket 5-Min BTC Bot v2 - Refined Backtest
===============================================
Two improved strategies backtested against the original:

  Strategy A: "Confluence" - Multi-indicator rule-based
     RSI, EMA trend, Bollinger Bands, momentum, volatility filter,
     time-of-day weighting, strict confluence scoring (3+ agree to trade)

  Strategy B: "ML Ensemble" - Walk-forward machine learning
     Gradient Boosting + Random Forest ensemble with 15+ features,
     walk-forward training (no look-ahead bias), confidence threshold

Usage:
    py backtest_v2.py
"""

import asyncio
import aiohttp
import time
import math
import json
import csv
import statistics
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")

# ============================================================================
#  DATA STRUCTURES
# ============================================================================

@dataclass
class Candle:
    """Aggregated candle from raw ticks."""
    open: float
    high: float
    low: float
    close: float
    ts: float         # close timestamp
    volume: int = 0   # tick count


@dataclass
class WindowResult:
    idx: int
    start_ts: float
    end_ts: float
    open_price: float
    close_price: float
    outcome: str           # "UP" or "DOWN"


@dataclass
class TradeSignal:
    direction: str         # "UP" or "DOWN"
    confidence: float      # 0.0 - 1.0
    reasoning: str


@dataclass
class TradeResult:
    window_idx: int
    window_start: str
    window_end: str
    open_price: float
    close_price: float
    actual_outcome: str
    predicted: str
    confidence: float
    strategy: str
    pnl: float
    reasoning: str


# ============================================================================
#  DATA FETCHING (Binance 1-second klines)
# ============================================================================

async def fetch_1s_klines(hours: int = 24) -> list[Candle]:
    """Fetch 1-second BTC/USDT klines from Binance."""
    base_url = "https://api.binance.com/api/v3/klines"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - hours * 3600 * 1000

    candles = []
    current_start = start_ms
    batch = 0

    async with aiohttp.ClientSession() as session:
        while current_start < now_ms:
            params = {
                "symbol": "BTCUSDT",
                "interval": "1s",
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
                            open=float(c[1]),
                            high=float(c[2]),
                            low=float(c[3]),
                            close=float(c[4]),
                            ts=c[6] / 1000.0,
                            volume=int(float(c[5]) * 1000),  # approx tick volume
                        ))

                    current_start = data[-1][6] + 1
                    batch += 1
                    if batch % 10 == 0:
                        print(f"  [*] Fetched {len(candles):,} candles ({len(candles)/60:.0f} mins)...")
                    await asyncio.sleep(0.08)

            except Exception as e:
                print(f"  [!] Fetch error: {e}, retrying...")
                await asyncio.sleep(1)

    print(f"  [+] Total: {len(candles):,} candles over {len(candles)/3600:.1f} hours")
    return candles


# ============================================================================
#  FEATURE ENGINEERING
# ============================================================================

def aggregate_candles(candles_1s: list[Candle], period: int) -> list[Candle]:
    """Aggregate 1-second candles into N-second candles."""
    result = []
    for i in range(0, len(candles_1s) - period + 1, period):
        chunk = candles_1s[i:i + period]
        result.append(Candle(
            open=chunk[0].open,
            high=max(c.high for c in chunk),
            low=min(c.low for c in chunk),
            close=chunk[-1].close,
            ts=chunk[-1].ts,
            volume=sum(c.volume for c in chunk),
        ))
    return result


def compute_rsi(closes: list[float], period: int = 14) -> float:
    """Compute RSI from close prices."""
    if len(closes) < period + 1:
        return 50.0

    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    recent = deltas[-period:]

    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]

    avg_gain = sum(gains) / period if gains else 0.0001
    avg_loss = sum(losses) / period if losses else 0.0001

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_ema(values: list[float], period: int) -> list[float]:
    """Compute EMA series."""
    if not values:
        return []
    multiplier = 2.0 / (period + 1)
    ema = [values[0]]
    for i in range(1, len(values)):
        ema.append(values[i] * multiplier + ema[-1] * (1 - multiplier))
    return ema


def compute_bollinger(closes: list[float], period: int = 20, num_std: float = 2.0):
    """Compute Bollinger Bands. Returns (upper, middle, lower, %B, bandwidth)."""
    if len(closes) < period:
        return None

    window = closes[-period:]
    middle = statistics.mean(window)
    std = statistics.stdev(window) if len(window) > 1 else 0.0001

    upper = middle + num_std * std
    lower = middle - num_std * std

    current = closes[-1]
    pct_b = (current - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    bandwidth = (upper - lower) / middle * 100

    return upper, middle, lower, pct_b, bandwidth


def compute_atr(candles: list[Candle], period: int = 14) -> float:
    """Average True Range."""
    if len(candles) < period + 1:
        return 0.0

    trs = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i-1].close),
            abs(candles[i].low - candles[i-1].close)
        )
        trs.append(tr)

    return statistics.mean(trs[-period:])


def compute_macd(closes: list[float]):
    """MACD: (macd_line, signal_line, histogram)."""
    if len(closes) < 26:
        return 0, 0, 0

    ema12 = compute_ema(closes, 12)
    ema26 = compute_ema(closes, 26)

    macd_line = [ema12[i] - ema26[i] for i in range(len(ema26))]
    signal = compute_ema(macd_line, 9)

    return macd_line[-1], signal[-1], macd_line[-1] - signal[-1]


def compute_stochastic(candles: list[Candle], period: int = 14) -> tuple[float, float]:
    """Stochastic %K and %D."""
    if len(candles) < period:
        return 50.0, 50.0

    recent = candles[-period:]
    highest = max(c.high for c in recent)
    lowest = min(c.low for c in recent)

    if highest == lowest:
        return 50.0, 50.0

    k = (recent[-1].close - lowest) / (highest - lowest) * 100

    # %D = 3-period SMA of %K (simplified: use current K)
    return k, k


def compute_obv_slope(candles: list[Candle], lookback: int = 20) -> float:
    """On-Balance Volume slope (approximated from tick volume)."""
    if len(candles) < lookback:
        return 0.0

    recent = candles[-lookback:]
    obv = [0.0]
    for i in range(1, len(recent)):
        if recent[i].close > recent[i-1].close:
            obv.append(obv[-1] + recent[i].volume)
        elif recent[i].close < recent[i-1].close:
            obv.append(obv[-1] - recent[i].volume)
        else:
            obv.append(obv[-1])

    # Slope of OBV (linear regression simplified)
    if len(obv) < 2:
        return 0.0
    x = list(range(len(obv)))
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(obv)
    numerator = sum((x[i] - x_mean) * (obv[i] - y_mean) for i in range(len(obv)))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(len(obv)))

    return numerator / denominator if denominator > 0 else 0.0


def compute_vwap(candles: list[Candle], period: int = 60) -> float:
    """Volume-weighted average price."""
    recent = candles[-period:]
    total_vol = sum(c.volume for c in recent)
    if total_vol == 0:
        return recent[-1].close

    vwap = sum(((c.high + c.low + c.close) / 3) * c.volume for c in recent) / total_vol
    return vwap


def build_feature_vector(candles_5s: list[Candle], candles_15s: list[Candle],
                          candles_30s: list[Candle], candles_60s: list[Candle]) -> dict:
    """
    Build a comprehensive feature vector from multi-timeframe candle data.
    All features are computed from data BEFORE the window opens (no look-ahead).
    """
    features = {}

    if len(candles_5s) < 30:
        return {}

    closes_5s = [c.close for c in candles_5s]
    closes_15s = [c.close for c in candles_15s] if candles_15s else closes_5s
    closes_30s = [c.close for c in candles_30s] if candles_30s else closes_5s

    # ── 1. RSI (multiple timeframes)
    features["rsi_5s"] = compute_rsi(closes_5s, 14)
    features["rsi_15s"] = compute_rsi(closes_15s, 14)

    # ── 2. EMA slopes (trend direction)
    ema9 = compute_ema(closes_5s, 9)
    ema21 = compute_ema(closes_5s, 21)
    features["ema9_slope"] = (ema9[-1] - ema9[-3]) / ema9[-3] * 10000 if len(ema9) > 3 else 0
    features["ema21_slope"] = (ema21[-1] - ema21[-3]) / ema21[-3] * 10000 if len(ema21) > 3 else 0
    features["ema_cross"] = (ema9[-1] - ema21[-1]) / closes_5s[-1] * 10000 if len(ema9) > 0 and len(ema21) > 0 else 0

    # ── 3. Bollinger Bands
    bb = compute_bollinger(closes_5s, 20, 2.0)
    if bb:
        features["bb_pct_b"] = bb[3]          # Where price is in band (0=lower, 1=upper)
        features["bb_bandwidth"] = bb[4]       # Band width (volatility)
    else:
        features["bb_pct_b"] = 0.5
        features["bb_bandwidth"] = 0.1

    # ── 4. MACD
    macd_line, signal_line, histogram = compute_macd(closes_5s)
    features["macd_hist"] = histogram / closes_5s[-1] * 10000  # Normalized
    features["macd_hist_accel"] = 0  # Will compute if enough data

    # ── 5. ATR (volatility)
    atr = compute_atr(candles_5s, 14)
    features["atr_pct"] = atr / closes_5s[-1] * 100

    # ── 6. Returns at different lookbacks
    for lb in [6, 12, 30, 60]:  # 30s, 60s, 150s, 300s
        if len(closes_5s) > lb:
            features[f"ret_{lb*5}s"] = (closes_5s[-1] - closes_5s[-lb]) / closes_5s[-lb] * 100
        else:
            features[f"ret_{lb*5}s"] = 0

    # ── 7. Stochastic
    k, d = compute_stochastic(candles_5s, 14)
    features["stoch_k"] = k
    features["stoch_d"] = d

    # ── 8. OBV slope (volume momentum)
    features["obv_slope"] = compute_obv_slope(candles_5s, 20)

    # ── 9. VWAP deviation
    vwap = compute_vwap(candles_5s, 60)
    features["vwap_dev"] = (closes_5s[-1] - vwap) / vwap * 10000

    # ── 10. Price acceleration (2nd derivative of price)
    if len(closes_5s) > 6:
        vel1 = closes_5s[-1] - closes_5s[-3]
        vel2 = closes_5s[-3] - closes_5s[-6]
        features["price_accel"] = (vel1 - vel2) / closes_5s[-1] * 10000
    else:
        features["price_accel"] = 0

    # ── 11. Candle patterns (last few candles)
    if len(candles_5s) >= 3:
        last3 = candles_5s[-3:]
        # Consecutive direction
        dirs = [1 if c.close > c.open else -1 for c in last3]
        features["candle_streak"] = sum(dirs)
        # Body-to-wick ratio (momentum quality)
        bodies = [abs(c.close - c.open) for c in last3]
        wicks = [c.high - c.low for c in last3]
        features["body_to_wick"] = sum(bodies) / (sum(wicks) + 0.0001)
    else:
        features["candle_streak"] = 0
        features["body_to_wick"] = 0.5

    # ── 12. Volatility regime (recent vs longer-term)
    if len(candles_5s) > 60:
        recent_vol = statistics.stdev(closes_5s[-12:]) if len(closes_5s[-12:]) > 1 else 0.001
        longer_vol = statistics.stdev(closes_5s[-60:]) if len(closes_5s[-60:]) > 1 else 0.001
        features["vol_ratio"] = recent_vol / longer_vol
    else:
        features["vol_ratio"] = 1.0

    # ── 13. Higher timeframe trend (30s candles)
    if candles_30s and len(candles_30s) > 5:
        ema_ht = compute_ema([c.close for c in candles_30s], 5)
        features["ht_trend"] = (ema_ht[-1] - ema_ht[-3]) / ema_ht[-3] * 10000 if len(ema_ht) > 3 else 0
    else:
        features["ht_trend"] = 0

    # ── 14. Mean reversion signal
    if len(closes_5s) > 20:
        ma20 = statistics.mean(closes_5s[-20:])
        std20 = statistics.stdev(closes_5s[-20:]) if len(closes_5s[-20:]) > 1 else 0.001
        features["z_score"] = (closes_5s[-1] - ma20) / std20
    else:
        features["z_score"] = 0

    # ── 15. Time features (hour of day in UTC)
    hour_utc = datetime.fromtimestamp(candles_5s[-1].ts, tz=timezone.utc).hour
    features["hour_sin"] = math.sin(2 * math.pi * hour_utc / 24)
    features["hour_cos"] = math.cos(2 * math.pi * hour_utc / 24)

    return features


# ============================================================================
#  STRATEGY A: CONFLUENCE (Multi-Indicator Rule-Based)
# ============================================================================

class ConfluenceStrategy:
    """
    Scores multiple indicators and only trades when 3+ indicators
    agree on direction with sufficient strength. Includes strict filters.
    """

    def __init__(self):
        self.name = "Confluence"
        self.min_score = 3.0          # Minimum confluence score to trade
        self.min_confidence = 0.55    # Minimum confidence
        self.max_atr_pct = 0.15      # Skip if volatility too extreme
        self.min_atr_pct = 0.005     # Skip if market is dead flat

    def evaluate(self, features: dict) -> Optional[TradeSignal]:
        if not features:
            return None

        atr = features.get("atr_pct", 0)

        # ── FILTER 1: Volatility gate
        if atr > self.max_atr_pct or atr < self.min_atr_pct:
            return None

        # ── FILTER 2: Skip extreme BB bandwidth (choppy markets)
        if features.get("bb_bandwidth", 0) > 0.5:
            return None

        # ── Score each indicator for UP or DOWN
        score = 0.0  # Positive = UP, Negative = DOWN
        reasons = []

        # 1. RSI: Oversold → UP, Overbought → DOWN
        rsi = features.get("rsi_5s", 50)
        if rsi < 30:
            score += 1.5
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 40:
            score += 0.5
        elif rsi > 70:
            score -= 1.5
            reasons.append(f"RSI overbought ({rsi:.0f})")
        elif rsi > 60:
            score -= 0.5

        # 2. EMA trend alignment
        ema_slope = features.get("ema9_slope", 0)
        ema_cross = features.get("ema_cross", 0)
        if ema_slope > 0.5 and ema_cross > 0:
            score += 1.0
            reasons.append("EMA bullish alignment")
        elif ema_slope < -0.5 and ema_cross < 0:
            score -= 1.0
            reasons.append("EMA bearish alignment")

        # 3. Bollinger Band position
        bb_pos = features.get("bb_pct_b", 0.5)
        if bb_pos < 0.1:
            score += 1.2  # Near lower band → reversal UP
            reasons.append(f"Near BB lower ({bb_pos:.2f})")
        elif bb_pos > 0.9:
            score -= 1.2  # Near upper band → reversal DOWN
            reasons.append(f"Near BB upper ({bb_pos:.2f})")

        # 4. MACD histogram
        macd_h = features.get("macd_hist", 0)
        if macd_h > 0.3:
            score += 0.8
            reasons.append("MACD bullish")
        elif macd_h < -0.3:
            score -= 0.8
            reasons.append("MACD bearish")

        # 5. Stochastic
        stoch = features.get("stoch_k", 50)
        if stoch < 20:
            score += 1.0
            reasons.append(f"Stoch oversold ({stoch:.0f})")
        elif stoch > 80:
            score -= 1.0
            reasons.append(f"Stoch overbought ({stoch:.0f})")

        # 6. Recent momentum (30s return)
        ret_30 = features.get("ret_30s", 0)
        if ret_30 > 0.03:
            score += 0.7
            reasons.append(f"30s momentum +{ret_30:.3f}%")
        elif ret_30 < -0.03:
            score -= 0.7
            reasons.append(f"30s momentum {ret_30:.3f}%")

        # 7. Higher timeframe trend alignment
        ht = features.get("ht_trend", 0)
        if abs(ht) > 0.5:
            if ht > 0:
                score += 0.6
            else:
                score -= 0.6
            reasons.append(f"HT trend {'up' if ht > 0 else 'down'}")

        # 8. Z-score mean reversion (contrarian)
        z = features.get("z_score", 0)
        if z < -1.5:
            score += 0.8
            reasons.append(f"Z-score extreme low ({z:.1f})")
        elif z > 1.5:
            score -= 0.8
            reasons.append(f"Z-score extreme high ({z:.1f})")

        # 9. Volume confirmation (OBV slope)
        obv = features.get("obv_slope", 0)
        if abs(obv) > 100:
            if obv > 0 and score > 0:
                score += 0.5  # Volume confirms bullish
            elif obv < 0 and score < 0:
                score -= 0.5  # Volume confirms bearish

        # ── Decision
        abs_score = abs(score)
        if abs_score < self.min_score:
            return None

        direction = "UP" if score > 0 else "DOWN"
        # Map score to confidence (3.0 → 0.55, 6.0+ → 0.85)
        confidence = min(0.90, 0.50 + abs_score * 0.06)

        if confidence < self.min_confidence:
            return None

        return TradeSignal(
            direction=direction,
            confidence=confidence,
            reasoning=f"Score {score:+.1f}: " + ", ".join(reasons[:4]),
        )


# ============================================================================
#  STRATEGY B: ML ENSEMBLE (Walk-Forward)
# ============================================================================

class MLEnsembleStrategy:
    """
    Walk-forward ML ensemble:
    - GradientBoosting (captures non-linear patterns)
    - RandomForest (reduces overfitting, captures interactions)
    - Weighted average prediction
    - Retrain every N windows on expanding window
    - Confidence threshold to filter low-quality predictions
    """

    FEATURE_NAMES = [
        "rsi_5s", "rsi_15s", "ema9_slope", "ema21_slope", "ema_cross",
        "bb_pct_b", "bb_bandwidth", "macd_hist", "atr_pct",
        "ret_30s", "ret_60s", "ret_150s", "ret_300s",
        "stoch_k", "obv_slope", "vwap_dev", "price_accel",
        "candle_streak", "body_to_wick", "vol_ratio",
        "ht_trend", "z_score", "hour_sin", "hour_cos",
    ]

    def __init__(self):
        self.name = "ML_Ensemble"
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_train_samples = 60      # Need 60 windows before first prediction
        self.retrain_interval = 25       # Retrain every 25 new windows
        self.confidence_threshold = 0.56 # Minimum ML confidence to trade
        self.train_history_X = []
        self.train_history_y = []
        self._windows_since_train = 0

    def _features_to_array(self, features: dict) -> np.ndarray:
        return np.array([features.get(f, 0.0) for f in self.FEATURE_NAMES])

    def add_training_sample(self, features: dict, outcome: str):
        """Add a resolved window to training set."""
        if not features:
            return
        x = self._features_to_array(features)
        y = 1 if outcome == "UP" else 0
        self.train_history_X.append(x)
        self.train_history_y.append(y)
        self._windows_since_train += 1

    def should_retrain(self) -> bool:
        return (
            len(self.train_history_X) >= self.min_train_samples
            and self._windows_since_train >= self.retrain_interval
        )

    def train(self):
        """Train/retrain models on accumulated history."""
        X = np.array(self.train_history_X)
        y = np.array(self.train_history_y)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.gb_model.fit(X_scaled, y)
        self.rf_model.fit(X_scaled, y)

        self.is_trained = True
        self._windows_since_train = 0

    def predict(self, features: dict) -> Optional[TradeSignal]:
        """Predict direction for current window."""
        if not self.is_trained or not features:
            return None

        x = self._features_to_array(features).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_scaled = self.scaler.transform(x)

        # Ensemble probabilities (weighted average)
        gb_proba = self.gb_model.predict_proba(x_scaled)[0]  # [P(DOWN), P(UP)]
        rf_proba = self.rf_model.predict_proba(x_scaled)[0]

        # Weight GB slightly higher (better at capturing sequential patterns)
        ensemble_proba = 0.55 * gb_proba + 0.45 * rf_proba
        prob_up = ensemble_proba[1]

        # Direction and confidence
        if prob_up > 0.5:
            direction = "UP"
            confidence = prob_up
        else:
            direction = "DOWN"
            confidence = 1 - prob_up

        if confidence < self.confidence_threshold:
            return None

        # Get top feature importances for reasoning
        try:
            importances = self.gb_model.feature_importances_
            top_idx = np.argsort(importances)[-3:][::-1]
            top_features = [f"{self.FEATURE_NAMES[i]}({importances[i]:.2f})" for i in top_idx]
            reason = f"ML conf {confidence:.1%}, top: {', '.join(top_features)}"
        except Exception:
            reason = f"ML ensemble confidence {confidence:.1%}"

        return TradeSignal(
            direction=direction,
            confidence=confidence,
            reasoning=reason,
        )


# ============================================================================
#  ORIGINAL STRATEGY (from v1 for comparison)
# ============================================================================

class OriginalStrategy:
    """Simplified version of the v1 bot logic for fair comparison."""

    def __init__(self):
        self.name = "Original_v1"

    def evaluate(self, features: dict, candles_5s: list[Candle]) -> Optional[TradeSignal]:
        if not features or len(candles_5s) < 30:
            return None

        # Regime detection (same logic)
        prices = [c.close for c in candles_5s[-60:]]
        ranges = [abs(prices[i] - prices[i-5]) for i in range(5, len(prices))]
        avg_range = sum(ranges) / len(ranges) if ranges else 0
        atr_pct = avg_range / prices[-1] * 100

        net_move = (prices[-1] - prices[0]) / prices[0] * 100
        directions = [1 if prices[i] > prices[i-1] else -1 for i in range(1, len(prices))]
        net_dir = 1 if net_move > 0 else -1
        consistency = sum(1 for d in directions if d == net_dir) / len(directions) if directions else 0

        if atr_pct > 0.08:
            regime = "VOLATILE"
        elif abs(net_move) > 0.05 and consistency > 0.58:
            regime = "TRENDING"
        else:
            regime = "RANGING"

        # Momentum strategy
        if regime == "TRENDING" and abs(net_move) > 0.04 and consistency > 0.55:
            direction = "UP" if net_move > 0 else "DOWN"
            conf = 0.65 if consistency > 0.65 else 0.55
            return TradeSignal(direction=direction, confidence=conf,
                              reasoning=f"v1 momentum: {net_move:+.3f}%, {consistency:.0%} consistent")

        # Spread fade (simplified: just check if recent move was extreme)
        if regime == "RANGING":
            z = features.get("z_score", 0)
            if abs(z) > 1.0:
                direction = "UP" if z < 0 else "DOWN"
                return TradeSignal(direction=direction, confidence=0.52,
                                  reasoning=f"v1 spread fade: z={z:.1f}")

        return None


# ============================================================================
#  BACKTEST ENGINE
# ============================================================================

def run_backtest(candles_1s: list[Candle]):
    """
    Run all three strategies across 5-minute windows.
    Returns results for each strategy.
    """
    # Pre-compute multi-timeframe candles
    candles_5s = aggregate_candles(candles_1s, 5)
    candles_15s = aggregate_candles(candles_1s, 15)
    candles_30s = aggregate_candles(candles_1s, 30)
    candles_60s = aggregate_candles(candles_1s, 60)

    print(f"  [*] Aggregated: {len(candles_5s)} x5s, {len(candles_15s)} x15s, "
          f"{len(candles_30s)} x30s, {len(candles_60s)} x60s candles")

    # Build 5-minute windows
    first_ts = candles_1s[0].ts
    last_ts = candles_1s[-1].ts
    window_start = first_ts - (first_ts % 300) + 300
    window_duration = 300

    windows = []
    while window_start + window_duration <= last_ts:
        window_end = window_start + window_duration
        wc = [c for c in candles_1s if window_start <= c.ts <= window_end]
        if len(wc) >= 30:
            windows.append(WindowResult(
                idx=len(windows),
                start_ts=window_start,
                end_ts=window_end,
                open_price=wc[0].open,
                close_price=wc[-1].close,
                outcome="UP" if wc[-1].close >= wc[0].open else "DOWN",
            ))
        window_start += window_duration

    print(f"  [*] Total windows: {len(windows)}")
    up_count = sum(1 for w in windows if w.outcome == "UP")
    print(f"  [*] Base rate: {up_count}/{len(windows)} UP ({up_count/len(windows)*100:.1f}%)")

    # Initialize strategies
    confluence = ConfluenceStrategy()
    ml_ensemble = MLEnsembleStrategy()
    original = OriginalStrategy()

    results = {"Original_v1": [], "Confluence": [], "ML_Ensemble": []}

    # Polymarket cost model:
    #   Buy at 0.52 (2% spread/vig), win pays $1.00
    #   Win PnL  = +$0.48 per $1 risked (or $4.80 per $10)
    #   Loss PnL = -$0.52 per $1 risked (or $5.20 per $10)
    BET_SIZE = 10.0
    VIG = 0.02  # 2% vig (buy at 0.52 instead of 0.50)

    for i, window in enumerate(windows):
        # Get candles up to window start (NO look-ahead)
        c5s_before = [c for c in candles_5s if c.ts <= window.start_ts]
        c15s_before = [c for c in candles_15s if c.ts <= window.start_ts]
        c30s_before = [c for c in candles_30s if c.ts <= window.start_ts]
        c60s_before = [c for c in candles_60s if c.ts <= window.start_ts]

        # Take last N candles for feature computation
        c5s_window = c5s_before[-120:]   # Last 600s of 5s candles
        c15s_window = c15s_before[-40:]
        c30s_window = c30s_before[-20:]
        c60s_window = c60s_before[-10:]

        # Build features
        features = build_feature_vector(c5s_window, c15s_window, c30s_window, c60s_window)

        window_start_str = datetime.fromtimestamp(window.start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        window_end_str = datetime.fromtimestamp(window.end_ts, tz=timezone.utc).strftime("%H:%M")

        def make_result(signal, strategy_name):
            if signal is None:
                return None
            win = signal.direction == window.outcome
            # Higher confidence bets get better/worse odds
            if win:
                pnl = BET_SIZE * (1.0 / (0.50 + VIG) - 1.0)  # ~$9.23 per $10
            else:
                pnl = -BET_SIZE

            return TradeResult(
                window_idx=window.idx,
                window_start=window_start_str,
                window_end=window_end_str,
                open_price=round(window.open_price, 2),
                close_price=round(window.close_price, 2),
                actual_outcome=window.outcome,
                predicted=signal.direction,
                confidence=round(signal.confidence, 3),
                strategy=strategy_name,
                pnl=round(pnl, 2),
                reasoning=signal.reasoning,
            )

        # ── Strategy 1: Original v1
        sig_orig = original.evaluate(features, c5s_window)
        res = make_result(sig_orig, "Original_v1")
        if res:
            results["Original_v1"].append(res)

        # ── Strategy 2: Confluence
        sig_conf = confluence.evaluate(features)
        res = make_result(sig_conf, "Confluence")
        if res:
            results["Confluence"].append(res)

        # ── Strategy 3: ML Ensemble (walk-forward)
        # First predict (if trained), then add to training set
        sig_ml = ml_ensemble.predict(features)
        res = make_result(sig_ml, "ML_Ensemble")
        if res:
            results["ML_Ensemble"].append(res)

        # Add this window's features + outcome to ML training set
        ml_ensemble.add_training_sample(features, window.outcome)

        # Retrain periodically
        if ml_ensemble.should_retrain():
            ml_ensemble.train()
            trained_on = len(ml_ensemble.train_history_X)
            if i < len(windows) - 1:  # Don't print on last window
                pass  # Silent retraining

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  [*] Processed {i+1}/{len(windows)} windows...")

    return results, windows


# ============================================================================
#  REPORTING
# ============================================================================

def print_strategy_report(name: str, trades: list[TradeResult], total_windows: int):
    """Print detailed report for one strategy."""
    if not trades:
        print(f"\n  [{name}] No trades generated.\n")
        return

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    win_rate = len(wins) / len(trades) * 100

    avg_win = statistics.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = statistics.mean([t.pnl for t in losses]) if losses else 0
    total_wins_pnl = sum(t.pnl for t in wins)
    total_loss_pnl = sum(t.pnl for t in losses)
    profit_factor = abs(total_wins_pnl / total_loss_pnl) if total_loss_pnl != 0 else float('inf')

    # Streaks
    max_ws = max_ls = 0
    cs = 0
    ct = None
    for t in trades:
        if t.pnl > 0:
            if ct == "w": cs += 1
            else: cs = 1; ct = "w"
            max_ws = max(max_ws, cs)
        else:
            if ct == "l": cs += 1
            else: cs = 1; ct = "l"
            max_ls = max(max_ls, cs)

    # Drawdown
    running = []
    cum = 0
    for t in trades:
        cum += t.pnl
        running.append(cum)
    peak = running[0]
    max_dd = 0
    for p in running:
        if p > peak: peak = p
        dd = peak - p
        if dd > max_dd: max_dd = dd

    # Expectancy
    expectancy = total_pnl / len(trades)

    # Sharpe-like ratio (using trade PnLs)
    if len(trades) > 1:
        pnl_std = statistics.stdev([t.pnl for t in trades])
        sharpe = (expectancy / pnl_std * math.sqrt(len(trades))) if pnl_std > 0 else 0
    else:
        sharpe = 0

    print(f"  Trades:         {len(trades):>6}  ({len(trades)/total_windows*100:.1f}% of windows)")
    print(f"  Wins/Losses:    {len(wins):>3}/{len(losses):<3}")
    print(f"  Win Rate:       {win_rate:>6.1f}%")
    print(f"  Total P&L:      ${total_pnl:>+8.2f}")
    print(f"  Avg Win:        ${avg_win:>+8.2f}")
    print(f"  Avg Loss:       ${avg_loss:>+8.2f}")
    print(f"  Expectancy:     ${expectancy:>+8.2f} / trade")
    print(f"  Profit Factor:  {profit_factor:>8.2f}")
    print(f"  Sharpe (trade): {sharpe:>8.2f}")
    print(f"  Max Drawdown:   ${max_dd:>8.2f}")
    print(f"  Max Win Streak: {max_ws:>6}")
    print(f"  Max Loss Streak:{max_ls:>6}")


def print_full_report(all_results: dict, windows: list[WindowResult]):
    total_windows = len(windows)
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  POLYMARKET 5-MIN BTC BACKTEST v2 -- STRATEGY COMPARISON")
    print(f"  Period: {datetime.fromtimestamp(windows[0].start_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} - "
          f"{datetime.fromtimestamp(windows[-1].end_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    up_ct = sum(1 for w in windows if w.outcome == "UP")
    print(f"  Windows: {total_windows} | UP: {up_ct} ({up_ct/total_windows*100:.1f}%) | "
          f"DOWN: {total_windows - up_ct} ({(total_windows-up_ct)/total_windows*100:.1f}%)")
    print(f"{sep}")

    # ── Side-by-side comparison
    strategies = ["Original_v1", "Confluence", "ML_Ensemble"]
    for name in strategies:
        trades = all_results.get(name, [])
        print(f"\n  {'='*60}")
        print(f"  STRATEGY: {name}")
        print(f"  {'='*60}")
        print_strategy_report(name, trades, total_windows)

    # ── Head-to-head comparison table  
    print(f"\n{sep}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{sep}")
    print(f"  {'Metric':<20} {'Original':>12} {'Confluence':>12} {'ML Ensemble':>12}")
    print(f"  {'-'*56}")

    for metric_name, metric_fn in [
        ("Trades", lambda t: f"{len(t)}"),
        ("Win Rate", lambda t: f"{sum(1 for x in t if x.pnl>0)/len(t)*100:.1f}%" if t else "N/A"),
        ("Total P&L", lambda t: f"${sum(x.pnl for x in t):+.2f}" if t else "$0.00"),
        ("Expectancy", lambda t: f"${sum(x.pnl for x in t)/len(t):+.2f}" if t else "$0.00"),
        ("Profit Factor", lambda t: f"{abs(sum(x.pnl for x in t if x.pnl>0)/(sum(x.pnl for x in t if x.pnl<=0) or -0.01)):.2f}" if t else "0.00"),
    ]:
        vals = []
        for name in strategies:
            trades = all_results.get(name, [])
            vals.append(metric_fn(trades))
        print(f"  {metric_name:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # ── Hourly P&L comparison
    print(f"\n{sep}")
    print(f"  HOURLY P&L BREAKDOWN")
    print(f"{sep}")
    print(f"  {'Hour':<7} {'Original':>10} {'Confluence':>12} {'ML Ensemble':>12}")
    print(f"  {'-'*41}")

    all_hours = sorted(set(
        t.window_start.split(" ")[1][:2] + ":00"
        for strat_trades in all_results.values()
        for t in strat_trades
    ))

    for hour in all_hours:
        vals = []
        for name in strategies:
            trades = all_results.get(name, [])
            hour_trades = [t for t in trades if t.window_start.split(" ")[1][:2] + ":00" == hour]
            pnl = sum(t.pnl for t in hour_trades)
            vals.append(f"${pnl:+.2f}")
        print(f"  {hour:<7} {vals[0]:>10} {vals[1]:>12} {vals[2]:>12}")

    # ── Equity curves
    print(f"\n{sep}")
    print(f"  EQUITY CURVES")
    print(f"{sep}")

    for name in strategies:
        trades = all_results.get(name, [])
        if not trades:
            print(f"\n  [{name}] No trades")
            continue

        running = []
        cum = 0
        for t in trades:
            cum += t.pnl
            running.append(cum)

        min_eq = min(running)
        max_eq = max(running)
        rng = max_eq - min_eq if max_eq != min_eq else 1
        chart_w = 35

        print(f"\n  [{name}]")
        step = max(1, len(running) // 15)
        for i in range(0, len(running), step):
            val = running[i]
            pos = int((val - min_eq) / rng * chart_w)
            bar = " " * pos + "*"
            ts = trades[i].window_start[-5:]
            print(f"    {ts}  ${val:>+8.2f}  |{bar}")
        val = running[-1]
        pos = int((val - min_eq) / rng * chart_w)
        bar = " " * pos + "*"
        print(f"    FINAL ${val:>+8.2f}  |{bar}")

    # ── ML Feature Importance
    print(f"\n{sep}")
    print(f"  ML MODEL INSIGHTS")
    print(f"{sep}")

    print(f"\n  The ML ensemble was trained on {len(all_results.get('ML_Ensemble', []))} prediction windows")
    print(f"  First prediction after {60} windows of training data")

    print(f"\n{sep}")
    print(f"  BACKTEST COMPLETE")
    print(f"{sep}\n")


def save_all_csv(all_results: dict, base_path: str):
    """Save trade results to CSV files."""
    keys = [
        "window_idx", "window_start", "window_end", "open_price", "close_price",
        "actual_outcome", "predicted", "confidence", "strategy", "pnl", "reasoning"
    ]
    for name, trades in all_results.items():
        if not trades:
            continue
        filepath = f"{base_path}_{name.lower()}.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for t in trades:
                w.writerow({k: getattr(t, k) for k in keys})
        print(f"  [+] Saved {len(trades)} trades to {filepath}")


# ============================================================================
#  MAIN
# ============================================================================

async def main():
    print("=" * 70)
    print("  POLYMARKET 5-MIN BTC BACKTESTER v2")
    print("  Original vs Confluence vs ML Ensemble")
    print("=" * 70)
    print()

    # 1. Fetch data
    print("  [1/4] Downloading 24h of 1-second BTC/USDT data from Binance...")
    candles = await fetch_1s_klines(hours=24)

    if len(candles) < 600:
        print("  [!] Not enough data. Exiting.")
        return

    first_time = datetime.fromtimestamp(candles[0].ts, tz=timezone.utc)
    last_time = datetime.fromtimestamp(candles[-1].ts, tz=timezone.utc)
    prices = [c.close for c in candles]
    print(f"\n  Period: {first_time.strftime('%Y-%m-%d %H:%M')} - {last_time.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  BTC Range: ${min(prices):,.2f} - ${max(prices):,.2f}")
    print(f"  Candles: {len(candles):,}\n")

    # 2. Build features & run backtest
    print("  [2/4] Running backtest (3 strategies in parallel)...")
    all_results, windows = run_backtest(candles)

    # 3. Report
    print("\n  [3/4] Generating reports...")
    print_full_report(all_results, windows)

    # 4. Save CSV
    print("  [4/4] Saving trade data...")
    save_all_csv(all_results, str(Path(__file__).parent / "backtest_v2"))


if __name__ == "__main__":
    asyncio.run(main())
