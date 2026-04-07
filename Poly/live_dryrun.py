"""
Polymarket 5-Min BTC Bot - LIVE DRY RUN
========================================
Runs both Confluence + ML Ensemble strategies in real-time.
No real money - paper trades only. Shows live signals in terminal.

Flow:
  1. Bootstrap: fetch 2h of historical 1-min candles for indicators
  2. Stream: Coinbase ticker + Binance kline + Chainlink oracle poll
  3. Every 5-min window: observe first 3 candles, decide at minute 3
  4. At minute 5: resolve outcome, update P&L
  5. ML model trains on accumulated live data + bootstrap history

Usage:
    py live_dryrun.py
"""

import asyncio
import aiohttp
import aiohttp.web
import time
import math
import json
import os
import statistics
import warnings
import sys
import joblib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from collections import deque

# Fix Windows encoding so it works with 'py live_dryrun.py' directly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # Enable ANSI escape codes on Windows 10+
    os.system("")

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
)
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load .env credentials
load_dotenv(Path(__file__).parent / ".env")
TG_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TG_CHAT_ID", "")

# ============================================================================
#  DATA STRUCTURES
# ============================================================================

@dataclass
class Candle:
    ts_open: float
    ts_close: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    taker_buy_vol: float
    binance_price: float = 0.0
    chainlink_price: float = 0.0
    is_closed: bool = True


@dataclass
class PaperTrade:
    window_start: str
    window_end: str
    strategy: str
    direction: str
    confidence: float
    entry_price: float
    open_price: float
    close_price: float = 0.0
    outcome: str = ""
    pnl: float = 0.0
    resolved: bool = False
    reasoning: str = ""


# ============================================================================
#  TECHNICAL INDICATORS (same as backtest_v3)
# ============================================================================

def ema(values, period):
    if not values: return []
    mult = 2.0 / (period + 1)
    e = [values[0]]
    for v in values[1:]:
        e.append(v * mult + e[-1] * (1 - mult))
    return e

def rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    recent = deltas[-period:]
    ag = sum(d for d in recent if d > 0) / period or 1e-8
    al = sum(-d for d in recent if d < 0) / period or 1e-8
    return 100 - 100 / (1 + ag / al)

def bollinger(closes, period=20, nstd=2.0):
    if len(closes) < period: return 0.5, 0.1
    w = closes[-period:]
    mid = statistics.mean(w)
    sd = statistics.stdev(w) if len(w) > 1 else 1e-8
    upper, lower = mid + nstd * sd, mid - nstd * sd
    pct_b = (closes[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    return pct_b, (upper - lower) / mid * 100

def atr(candles_list, period=14):
    if len(candles_list) < 2: return 0.0
    trs = []
    for i in range(1, len(candles_list)):
        tr = max(candles_list[i].high - candles_list[i].low,
                 abs(candles_list[i].high - candles_list[i-1].close),
                 abs(candles_list[i].low - candles_list[i-1].close))
        trs.append(tr)
    return statistics.mean(trs[-period:]) if trs else 0.0

def macd(closes):
    if len(closes) < 26: return 0, 0, 0
    e12, e26 = ema(closes, 12), ema(closes, 26)
    line = [e12[i] - e26[i] for i in range(len(e26))]
    sig = ema(line, 9)
    return line[-1], sig[-1], line[-1] - sig[-1]

def stochastic(candles_list, period=14):
    if len(candles_list) < period: return 50.0
    r = candles_list[-period:]
    hi, lo = max(c.high for c in r), min(c.low for c in r)
    return (r[-1].close - lo) / (hi - lo) * 100 if hi != lo else 50.0

def williams_r(candles_list, period=14):
    if len(candles_list) < period: return -50.0
    r = candles_list[-period:]
    hi, lo = max(c.high for c in r), min(c.low for c in r)
    return (hi - r[-1].close) / (hi - lo) * -100 if hi != lo else -50.0

def obv_slope(candles_list, lookback=20):
    if len(candles_list) < lookback: return 0.0
    r = candles_list[-lookback:]
    obv = [0.0]
    for i in range(1, len(r)):
        if r[i].close > r[i-1].close: obv.append(obv[-1] + r[i].volume)
        elif r[i].close < r[i-1].close: obv.append(obv[-1] - r[i].volume)
        else: obv.append(obv[-1])
    x = list(range(len(obv)))
    xm, ym = statistics.mean(x), statistics.mean(obv)
    num = sum((x[i]-xm)*(obv[i]-ym) for i in range(len(obv)))
    den = sum((x[i]-xm)**2 for i in range(len(obv)))
    return num / den if den > 0 else 0.0

def vwap(candles_list, period=20):
    r = candles_list[-period:]
    tv = sum(c.volume for c in r)
    if tv == 0: return r[-1].close
    return sum(((c.high+c.low+c.close)/3)*c.volume for c in r) / tv

def taker_buy_ratio(candles_list, period=5):
    r = candles_list[-period:]
    tv = sum(c.volume for c in r)
    return sum(c.taker_buy_vol for c in r) / tv if tv > 0 else 0.5


# ============================================================================
#  FEATURE ENGINEERING (identical to backtest_v3)
# ============================================================================

def build_features(hist: list[Candle], window_3m: list[Candle]) -> dict:
    f = {}
    if len(hist) < 30 or len(window_3m) < 2:
        return {}

    closes = [c.close for c in hist]

    f["rsi_14"] = rsi(closes, 14)
    f["rsi_7"] = rsi(closes, 7)

    e9 = ema(closes[-30:], 9)
    e21 = ema(closes[-50:], 21)
    f["ema9_slope"] = (e9[-1]-e9[-3])/e9[-3]*1e4 if len(e9)>3 else 0
    f["ema21_slope"] = (e21[-1]-e21[-3])/e21[-3]*1e4 if len(e21)>3 else 0
    f["ema_cross"] = (e9[-1]-e21[-1])/closes[-1]*1e4 if e9 and e21 else 0

    bb_p, bb_bw = bollinger(closes, 20, 2.0)
    f["bb_pct_b"] = bb_p; f["bb_bandwidth"] = bb_bw

    ml, sl, mh = macd(closes)
    f["macd_hist"] = mh / closes[-1] * 1e4
    f["atr_pct"] = atr(hist, 14) / closes[-1] * 100

    for p in [5, 10, 20, 60]:
        f[f"ret_{p}m"] = (closes[-1]-closes[-p])/closes[-p]*100 if len(closes)>p else 0

    f["stoch_k"] = stochastic(hist, 14)
    f["williams_r"] = williams_r(hist, 14)
    f["obv_slope"] = obv_slope(hist, 20)

    v = vwap(hist, 20)
    f["vwap_dev"] = (closes[-1]-v)/v*1e4

    f["taker_buy_ratio"] = taker_buy_ratio(hist, 10)
    f["taker_buy_ratio_5"] = taker_buy_ratio(hist, 5)
    f["trade_intensity"] = statistics.mean([c.trades for c in hist[-10:]])

    if len(closes) > 60:
        rv = statistics.stdev(closes[-10:]) if len(closes[-10:])>1 else 1e-8
        lv = statistics.stdev(closes[-60:]) if len(closes[-60:])>1 else 1e-8
        f["vol_ratio"] = rv / lv
    else:
        f["vol_ratio"] = 1.0

    if len(closes) > 20:
        ma20 = statistics.mean(closes[-20:])
        sd20 = statistics.stdev(closes[-20:]) if len(closes[-20:])>1 else 1e-8
        f["z_score"] = (closes[-1]-ma20)/sd20
    else:
        f["z_score"] = 0

    hour = datetime.fromtimestamp(hist[-1].ts_close, tz=timezone.utc).hour
    f["hour_sin"] = math.sin(2*math.pi*hour/24)
    f["hour_cos"] = math.cos(2*math.pi*hour/24)
    dow = datetime.fromtimestamp(hist[-1].ts_close, tz=timezone.utc).weekday()
    f["dow_sin"] = math.sin(2*math.pi*dow/7)
    f["dow_cos"] = math.cos(2*math.pi*dow/7)

    if len(closes) >= 15:
        f["prev_5m_dir"] = 1 if closes[-1]>closes[-5] else -1
        f["prev_10m_dir"] = 1 if closes[-1]>closes[-10] else -1
        f["prev_15m_dir"] = 1 if closes[-1]>closes[-15] else -1
        f["prev_5m_ret"] = (closes[-1]-closes[-5])/closes[-5]*100
    else:
        f["prev_5m_dir"]=0; f["prev_10m_dir"]=0; f["prev_15m_dir"]=0; f["prev_5m_ret"]=0

    # Intra-window
    wc = window_3m
    wo = wc[0].open
    f["intra_3m_ret"] = (wc[-1].close-wo)/wo*100
    f["intra_dir"] = 1 if wc[-1].close > wo else -1
    dirs = [1 if c.close>=c.open else -1 for c in wc]
    f["intra_candle_consistency"] = sum(dirs)/len(dirs)
    if wc[-1].binance_price > 0 and wc[-1].close > 0:
        f["spread_divergence"] = (wc[-1].binance_price - wc[-1].close) / wc[-1].close * 1e4
    else:
        f["spread_divergence"] = 0.0
    ih, il = max(c.high for c in wc), min(c.low for c in wc)
    f["intra_high_dev"] = (ih-wo)/wo*100
    f["intra_low_dev"] = (wo-il)/wo*100
    iv = sum(c.volume for c in wc)
    ib = sum(c.taker_buy_vol for c in wc)
    f["intra_buy_ratio"] = ib/iv if iv>0 else 0.5
    f["intra_vol_trend"] = wc[-1].volume/(wc[0].volume+1e-8) if len(wc)>=3 else 1.0
    f["intra_trade_intensity"] = sum(c.trades for c in wc)/len(wc)
    ir = ih - il
    f["intra_position"] = (wc[-1].close-il)/ir if ir>0 else 0.5
    if len(wc) >= 3:
        m1 = wc[1].close-wc[0].close
        m2 = wc[2].close-wc[1].close
        f["intra_accel"] = (m2-m1)/(abs(m1)+1e-8)
    else:
        f["intra_accel"] = 0
    bodies = [abs(c.close-c.open) for c in wc]
    wicks = [c.high-c.low for c in wc]
    f["intra_body_wick"] = sum(bodies)/(sum(wicks)+1e-8)
    f["alignment"] = f["intra_dir"] * (1 if f.get("ret_5m",0)>0 else -1)

    # -- Chainlink oracle lag features --
    cl_price = getattr(wc[-1], 'chainlink_price', 0.0)
    if cl_price > 0 and wc[-1].close > 0:
        f["chainlink_lag"] = (wc[-1].close - cl_price) / wc[-1].close * 1e4
        f["chainlink_divergence"] = abs(f["chainlink_lag"])
    else:
        f["chainlink_lag"] = 0.0
        f["chainlink_divergence"] = 0.0

    # -- Three-feed consensus --
    cb_dir = 1 if wc[-1].close > wo else -1
    bn_dir = 0
    if wc[-1].binance_price > 0 and len(wc) > 0 and wc[0].binance_price > 0:
        bn_dir = 1 if wc[-1].binance_price > wc[0].binance_price else -1
    cl_dir = 0
    if cl_price > 0 and len(hist) > 1:
        prev_cl = getattr(hist[-2], 'chainlink_price', 0.0)
        if prev_cl > 0:
            cl_dir = 1 if cl_price > prev_cl else -1
    if cb_dir == bn_dir == cl_dir and cb_dir != 0:
        f["three_feed_consensus"] = float(cb_dir)
    else:
        f["three_feed_consensus"] = 0.0

    return f


# ============================================================================
#  CONFLUENCE STRATEGY
# ============================================================================

class ConfluenceStrategy:
    def __init__(self):
        self.min_score = 2.8

    def evaluate(self, features: dict):
        if not features: return None
        score = 0.0; reasons = []

        intra_ret = features.get("intra_3m_ret", 0)
        if abs(intra_ret) > 0.02:
            score += 1.5 * (1 if intra_ret > 0 else -1)
            reasons.append(f"3m {intra_ret:+.3f}%")

        cons = features.get("intra_candle_consistency", 0)
        if abs(cons) > 0.6:
            score += 1.2 * (1 if cons > 0 else -1)
            reasons.append(f"Streak {cons:+.1f}")

        br = features.get("intra_buy_ratio", 0.5)
        if br > 0.55: score += 0.8; reasons.append(f"Buy {br:.0%}")
        elif br < 0.45: score -= 0.8; reasons.append(f"Sell {br:.0%}")

        accel = features.get("intra_accel", 0)
        if abs(accel) > 0.5:
            s = 0.6 * (1 if accel > 0 else -1)
            if s * (1 if intra_ret > 0 else -1 if intra_ret < 0 else 0) > 0:
                score += s; reasons.append(f"Accel {'up' if accel>0 else 'dn'}")

        pos = features.get("intra_position", 0.5)
        if pos > 0.75: score += 0.5
        elif pos < 0.25: score -= 0.5

        rsi_val = features.get("rsi_14", 50)
        if score > 0 and rsi_val > 78: score -= 1.0; reasons.append(f"RSI {rsi_val:.0f}")
        elif score < 0 and rsi_val < 22: score += 1.0; reasons.append(f"RSI {rsi_val:.0f}")

        atr_pct = features.get("atr_pct", 0)
        if atr_pct < 0.002 or atr_pct > 0.25: return None

        vt = features.get("intra_vol_trend", 1.0)
        if vt > 1.5 and abs(score) > 1: score *= 1.1

        al = features.get("alignment", 0)
        if al > 0: score *= 1.08; reasons.append("Aligned")

        # -- Oracle Lag Boost (Chainlink vs Coinbase) --
        cl_lag = features.get("chainlink_lag", 0)
        if abs(cl_lag) > 2.0:
            lag_dir = 1 if cl_lag > 0 else -1
            if lag_dir * (1 if score > 0 else -1 if score < 0 else 0) > 0:
                score += 1.0 * lag_dir
                reasons.append(f"OracleLag {cl_lag:+.1f}bp")

        # -- Three-Feed Consensus Boost --
        tfc = features.get("three_feed_consensus", 0)
        if tfc != 0:
            if tfc * (1 if score > 0 else -1 if score < 0 else 0) > 0:
                score *= 1.12
                reasons.append("3-Feed")

        if abs(score) < self.min_score: return None
        direction = "UP" if score > 0 else "DOWN"
        conf = min(0.88, 0.50 + abs(score) * 0.05)
        if conf < 0.54: return None
        return {"direction": direction, "confidence": conf,
                "reasoning": f"[{score:+.1f}] " + ", ".join(reasons[:5])}


# ============================================================================
#  ML ENSEMBLE STRATEGY
# ============================================================================

FEATURE_NAMES = [
    "rsi_14","rsi_7","ema9_slope","ema21_slope","ema_cross",
    "bb_pct_b","bb_bandwidth","macd_hist","atr_pct",
    "ret_5m","ret_10m","ret_20m","ret_60m",
    "stoch_k","williams_r","obv_slope","vwap_dev",
    "taker_buy_ratio","taker_buy_ratio_5","trade_intensity",
    "vol_ratio","z_score","hour_sin","hour_cos","dow_sin","dow_cos",
    "prev_5m_dir","prev_10m_dir","prev_15m_dir","prev_5m_ret",
    "intra_3m_ret","intra_dir","intra_candle_consistency",
    "intra_high_dev","intra_low_dev","intra_buy_ratio",
    "intra_vol_trend","intra_trade_intensity","intra_position",
    "intra_accel","intra_body_wick","alignment",
    "chainlink_lag","chainlink_divergence","three_feed_consensus",
]

class MLStrategy:
    def __init__(self):
        self.gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
            learning_rate=0.06, subsample=0.8, min_samples_leaf=8, random_state=42)
        self.rf = RandomForestClassifier(n_estimators=250, max_depth=5,
            min_samples_leaf=8, random_state=42, n_jobs=1)
        self.ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.08, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.min_train = 120
        self.retrain_every = 40
        self.conf_threshold = 0.55
        self.X = []; self.y = []
        self._since = 0

    def _vec(self, f):
        return np.array([f.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float64)

    def add(self, features, outcome):
        if not features: return
        self.X.append(self._vec(features))
        self.y.append(1 if outcome == "UP" else 0)
        self._since += 1

    def should_retrain(self):
        return len(self.X) >= self.min_train and self._since >= self.retrain_every

    def train(self):
        X = np.nan_to_num(np.array(self.X), nan=0, posinf=0, neginf=0)
        y = np.array(self.y)
        self.scaler.fit(X); Xs = self.scaler.transform(X)
        self.gb.fit(Xs, y); self.rf.fit(Xs, y); self.ada.fit(Xs, y)
        self.is_trained = True; self._since = 0

    def predict(self, features):
        if not self.is_trained or not features: return None
        x = np.nan_to_num(self._vec(features).reshape(1,-1), nan=0, posinf=0, neginf=0)
        # Handle feature count mismatch (old model trained on fewer features)
        expected = self.scaler.n_features_in_
        if x.shape[1] > expected:
            x = x[:, :expected]  # trim new features old model doesn't know
        elif x.shape[1] < expected:
            x = np.pad(x, ((0,0),(0, expected - x.shape[1])))
        xs = self.scaler.transform(x)
        p = 0.40*self.gb.predict_proba(xs)[0] + 0.35*self.rf.predict_proba(xs)[0] + 0.25*self.ada.predict_proba(xs)[0]
        prob_up = p[1]
        if prob_up > 0.5: d, c = "UP", prob_up
        else: d, c = "DOWN", 1-prob_up
        if c < self.conf_threshold: return None
        return {"direction": d, "confidence": c, "reasoning": f"ML conf={c:.1%}"}


# ============================================================================
#  TELEGRAM NOTIFIER
# ============================================================================

class TelegramNotifier:
    """Lightweight Telegram bot using aiohttp — no external bot framework needed."""

    def __init__(self, token: str, chat_id: str, bot_ref):
        self.token   = token
        self.chat_id = chat_id
        self.bot     = bot_ref          # reference back to LiveDryRun
        self._base   = f"https://api.telegram.org/bot{token}"
        self._offset = 0
        self.enabled = bool(token and chat_id)
        self._paused = False

    # ── Send ──────────────────────────────────────────────────────────────
    async def send(self, text: str):
        """Send a message (silently fails if not configured)."""
        if not self.enabled:
            return
        try:
            async with aiohttp.ClientSession() as s:
                await s.post(f"{self._base}/sendMessage", json={
                    "chat_id":    self.chat_id,
                    "text":       text,
                    "parse_mode": "HTML",
                }, timeout=aiohttp.ClientTimeout(total=8))
        except Exception as e:
            print(f"  [TG] send error: {e}")

    # ── Canned messages ───────────────────────────────────────────────────
    async def notify_startup(self, btc_price: float, ml_samples: int):
        await self.send(
            f"🟢 <b>Bot Online</b>\n"
            f"Mode: DRY RUN | $10/trade\n"
            f"BTC: <b>${btc_price:,.2f}</b>\n"
            f"ML samples: {ml_samples} (need 120 to activate)\n"
            f"Dashboard: http://localhost:8080\n"
            f"Commands: /status /pnl /trades /pause /resume"
        )

    async def notify_signal(self, trade):
        if self._paused:
            return
        arrow = "🟢 UP" if trade.direction == "UP" else "🔴 DOWN"
        await self.send(
            f"⚡ <b>SIGNAL — {trade.strategy}</b>\n"
            f"Direction: {arrow}\n"
            f"Confidence: {trade.confidence:.1%}\n"
            f"Entry: <b>${trade.entry_price:,.2f}</b>\n"
            f"Window: {trade.window_start} → {trade.window_end} UTC\n"
            f"Reasoning: {trade.reasoning[:60]}"
        )

    async def notify_result(self, trade, c_pnl: float, m_pnl: float):
        if self._paused:
            return
        icon   = "✅" if trade.pnl > 0 else "❌"
        result = "WIN" if trade.pnl > 0 else "LOSS"
        await self.send(
            f"{icon} <b>{result} — {trade.strategy}</b>\n"
            f"Direction: {trade.direction} | Outcome: {trade.outcome}\n"
            f"P&amp;L: <b>${trade.pnl:+.2f}</b>\n"
            f"Open: ${trade.open_price:,.2f} → Close: ${trade.close_price:,.2f}\n"
            f"― Running Totals ―\n"
            f"Confluence: ${c_pnl:+.2f}  |  ML: ${m_pnl:+.2f}"
        )

    async def notify_shutdown(self, completed_trades: list):
        c_t = [t for t in completed_trades if t.strategy == "Confluence"]
        m_t = [t for t in completed_trades if t.strategy == "ML_Ensemble"]
        c_pnl = sum(t.pnl for t in c_t)
        m_pnl = sum(t.pnl for t in m_t)
        c_wr  = sum(1 for t in c_t if t.pnl > 0) / len(c_t) * 100 if c_t else 0
        m_wr  = sum(1 for t in m_t if t.pnl > 0) / len(m_t) * 100 if m_t else 0
        await self.send(
            f"🔴 <b>Bot Stopped</b>\n"
            f"Total trades: {len(completed_trades)}\n"
            f"Confluence:  {len(c_t)} trades | {c_wr:.1f}% WR | ${c_pnl:+.2f}\n"
            f"ML Ensemble: {len(m_t)} trades | {m_wr:.1f}% WR | ${m_pnl:+.2f}\n"
            f"Combined P&amp;L: <b>${c_pnl+m_pnl:+.2f}</b>"
        )

    async def notify_skip(self, window_start: str, window_end: str, reasons: list[str]):
        """Notify when a 5-min window passes with no trade taken."""
        if self._paused:
            return
        reasons_text = "\n".join(f"  • {r}" for r in reasons) if reasons else "  • No reason recorded"
        await self.send(
            f"⏭ <b>Window Skipped</b>\n"
            f"🕐 {window_start} → {window_end} UTC\n"
            f"<b>Reason(s):</b>\n{reasons_text}"
        )

    async def notify_order_placed(self, trade, shares: float, cost_usdc: float,
                                   order_id: str, market_slug: str):
        """Notify when a real Polymarket order is successfully placed."""
        arrow = "🟢 UP" if trade.direction == "UP" else "🔴 DOWN"
        strat_short = "Conf" if trade.strategy == "Confluence" else "ML"
        await self.send(
            f"🔥 <b>ORDER PLACED — {strat_short}</b>\n"
            f"Direction: {arrow}\n"
            f"Confidence: {trade.confidence:.1%}\n"
            f"Market: <code>{market_slug}</code>\n"
            f"Shares: <b>{shares:.0f}</b> @ ${trade.order_price:.2f} = <b>${cost_usdc:.2f}</b>\n"
            f"BTC Entry: <b>${trade.entry_price:,.2f}</b>\n"
            f"Window: {trade.window_start} → {trade.window_end} UTC\n"
            f"Reason: {trade.reasoning[:80]}"
        )

    async def notify_order_failed(self, strategy: str, direction: str, error: str):
        """Notify when an order placement fails."""
        await self.send(
            f"❌ <b>ORDER FAILED — {strategy}</b>\n"
            f"Direction: {'🟢 UP' if direction == 'UP' else '🔴 DOWN'}\n"
            f"Error: <code>{error[:120]}</code>"
        )

    # ── Status reply (for /status command) ───────────────────────────────
    def _status_text(self) -> str:
        b = self.bot
        c_t = [t for t in b.completed_trades if t.strategy == "Confluence"]
        m_t = [t for t in b.completed_trades if t.strategy == "ML_Ensemble"]
        all_t = c_t + m_t
        c_pnl = sum(t.pnl for t in c_t)
        m_pnl = sum(t.pnl for t in m_t)
        c_wr  = sum(1 for t in c_t if t.pnl > 0) / len(c_t) * 100 if c_t else 0
        m_wr  = sum(1 for t in m_t if t.pnl > 0) / len(m_t) * 100 if m_t else 0
        o_wr  = sum(1 for t in all_t if t.pnl > 0) / len(all_t) * 100 if all_t else 0
        now   = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        paused_str = " ⏸ PAUSED" if self._paused else ""
        return (
            f"📊 <b>Bot Status{paused_str}</b> — {now}\n"
            f"BTC: <b>${b.btc_price:,.2f}</b>\n"
            f"Windows: {b.total_windows} | ML: {'Active' if b.ml.is_trained else 'Warming up'} ({len(b.ml.X)} smpl)\n"
            f"\n"
            f"<b>Confluence</b>: {len(c_t)} trades | {c_wr:.1f}% WR | ${c_pnl:+.2f}\n"
            f"<b>ML Ensemble</b>: {len(m_t)} trades | {m_wr:.1f}% WR | ${m_pnl:+.2f}\n"
            f"<b>Overall WR</b>: {o_wr:.1f}% | <b>Combined P&amp;L: ${c_pnl+m_pnl:+.2f}</b>\n"
            f"\n"
            f"Pending trades: {len(b.pending_trades)}"
        )

    def _trades_text(self) -> str:
        trades = self.bot.completed_trades[-10:]
        if not trades:
            return "No completed trades yet."
        lines = ["📋 <b>Last 10 Trades</b>"]
        for t in reversed(trades):
            icon = "✅" if t.pnl > 0 else "❌"
            lines.append(
                f"{icon} [{t.strategy[:4]}] {t.direction} | "
                f"${t.pnl:+.2f} | {t.window_start[-5:]}"
            )
        return "\n".join(lines)

    # ── Command polling loop ───────────────────────────────────────────────
    async def poll_commands(self):
        """Long-poll Telegram for commands. Runs as a background task."""
        if not self.enabled:
            print("  [TG] Not configured — skipping Telegram polling")
            return
        print("  [TG] Telegram command listener active")
        while True:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        f"{self._base}/getUpdates",
                        params={"offset": self._offset, "timeout": 20,
                                "allowed_updates": ["message"]},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as r:
                        data = await r.json()
                for upd in data.get("result", []):
                    self._offset = upd["update_id"] + 1
                    msg  = upd.get("message", {})
                    text = msg.get("text", "").strip().lower()
                    cid  = str(msg.get("chat", {}).get("id", ""))
                    if cid != self.chat_id:
                        continue  # ignore other chats
                    if text in ("/status", "/s"):
                        await self.send(self._status_text())
                    elif text in ("/pnl", "/p"):
                        b = self.bot
                        c_pnl = sum(t.pnl for t in b.completed_trades if t.strategy == "Confluence")
                        m_pnl = sum(t.pnl for t in b.completed_trades if t.strategy == "ML_Ensemble")
                        await self.send(
                            f"💰 <b>P&amp;L Summary</b>\n"
                            f"Confluence:  ${c_pnl:+.2f}\n"
                            f"ML Ensemble: ${m_pnl:+.2f}\n"
                            f"Combined:    <b>${c_pnl+m_pnl:+.2f}</b>"
                        )
                    elif text in ("/trades", "/t"):
                        await self.send(self._trades_text())
                    elif text == "/pause":
                        self._paused = True
                        await self.send("⏸ Bot <b>paused</b> — signals suppressed. Send /resume to reactivate.")
                    elif text == "/resume":
                        self._paused = False
                        await self.send("▶️ Bot <b>resumed</b> — signals active.")
                    elif text == "/help":
                        await self.send(
                            "🤖 <b>Available Commands</b>\n"
                            "/status — Full bot status & stats\n"
                            "/pnl — Quick P&amp;L summary\n"
                            "/trades — Last 10 completed trades\n"
                            "/pause — Suppress trade notifications\n"
                            "/resume — Re-enable notifications\n"
                            "/help — This message"
                        )
            except Exception as e:
                print(f"  [TG] Poll error: {e}")
            await asyncio.sleep(1)


# ============================================================================
#  LIVE DRY-RUN BOT
# ============================================================================

# -- Polygon RPC endpoints for Chainlink (free, with fallbacks) --

POLYGON_RPCS = [

    "https://polygon-bor-rpc.publicnode.com",

    "https://1rpc.io/matic",

    "https://polygon.drpc.org",

]

CHAINLINK_BTC_USD = "0xc907E116054Ad103354f2D350FD2514433D57F6f"

LATEST_ROUND_DATA_SIG = "0xfeaf968c"





class LiveDryRun:
    def __init__(self):
        self.candle_history: list[Candle] = []  # Completed 1-min candles
        self.current_candle: Optional[Candle] = None

        self.confluence = ConfluenceStrategy()
        
        ml_path = Path(__file__).parent / 'ml_model.joblib'
        if ml_path.exists():
            self.ml = joblib.load(ml_path)
            self.ml.retrain_every = 99999999  # Disable online retraining
            print(f"  [+] Loaded offline trained ML model from {ml_path}")
        else:
            self.ml = MLStrategy()

        # Telegram
        self.tg = TelegramNotifier(TG_TOKEN, TG_CHAT, bot_ref=self)

        # Active window tracking
        self.window_candles: list[Candle] = []  # Candles in current 5-min window
        self.window_start_ts: float = 0
        self.window_end_ts: float = 0
        self.window_open_price: float = 0
        self.signal_emitted: bool = False

        # Paper trades
        self.pending_trades: list[PaperTrade] = []
        self.completed_trades: list[PaperTrade] = []

        # Stats
        self.total_windows = 0
        self.btc_price = 0.0
        self.binance_cache = {}
        self.chainlink_price = 0.0   # Latest Chainlink BTC/USD oracle price

        # Live candle tracking for chart
        self.current_tick = None  # Current forming 1-min candle (dict)

        # P&L
        self.BET = 10.0
        self.WIN = 10.0 * (1.0/0.52 - 1.0)  # $9.23

    def _get_window_boundaries(self, ts: float):
        """Get the 5-min window boundaries for a given timestamp."""
        start = ts - (ts % 300)
        return start, start + 300


    async def bootstrap(self):
        print("  [*] Bootstrapping 2h of historical candles (Coinbase + Binance)...")
        now = datetime.now(timezone.utc)
        start_ms = int((now.timestamp() - 2 * 3600))
        c_url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
        b_url = "https://api.binance.com/api/v3/klines"
        
        async with aiohttp.ClientSession() as session:
            # Fetch Coinbase
            cb_c = {}
            for _ in range(3):
                try:
                    async with session.get(c_url, params={"granularity": 60, "start": datetime.fromtimestamp(start_ms, tz=timezone.utc).isoformat(), "end": now.isoformat()}) as r:
                        if r.status == 200:
                            data = await r.json()
                            for row in data: cb_c[float(row[0])] = [float(row[3]), float(row[2]), float(row[1]), float(row[4])]
                            break
                except: await asyncio.sleep(1)
                
            # Fetch Binance
            bn_c = {}
            for _ in range(3):
                try:
                    async with session.get(b_url, params={"symbol": "BTCUSDC", "interval": "1m", "startTime": start_ms * 1000, "limit": 1000}) as r:
                        if r.status == 200:
                            data = await r.json()
                            for row in data: bn_c[float(row[0])/1000.0] = [float(row[5]), int(row[8]), float(row[9]), float(row[4])]
                            break
                except: await asyncio.sleep(1)
                
        keys = sorted(list(cb_c.keys()))
        for ts in keys:
            if ts in bn_c and ts < time.time() - 60:
                c, b = cb_c[ts], bn_c[ts]
                self.candle_history.append(Candle(
                    ts_open=ts, ts_close=ts+59.9, open=c[0], high=c[1], low=c[2], close=c[3],
                    volume=b[0], trades=b[1], taker_buy_vol=b[2], binance_price=b[3]
                ))
                
        self.btc_price = self.candle_history[-1].close if self.candle_history else 0
        print(f"  [+] Loaded {len(self.candle_history)} hybrid historical candles | BTC: ${self.btc_price:,.2f}")
        self._pretrain_ml()

    def _pretrain_ml(self):
        """Use historical candles to pre-train the ML model."""
        if self.ml.is_trained:
            print("  [+] Using loaded offline ML model; skipping bootstrap retraining")
            return
        print("  [*] Pre-training ML model from historical data...")
        hist = self.candle_history
        count = 0
        for i in range(35, len(hist) - 5, 5):
            window_c = hist[i:i+5]
            if len(window_c) < 4: continue
            first_3 = window_c[:3]
            h = hist[:i]
            if len(h) < 30: continue

            features = build_features(h[-120:], first_3)
            if not features: continue

            outcome = "UP" if window_c[-1].close >= window_c[0].open else "DOWN"
            self.ml.add(features, outcome)
            count += 1

        if self.ml.should_retrain() or len(self.ml.X) >= self.ml.min_train:
            if len(self.ml.X) >= self.ml.min_train:
                self.ml.train()
                print(f"  [+] ML pre-trained on {count} windows (samples: {len(self.ml.X)})")
            else:
                print(f"  [*] ML has {len(self.ml.X)} samples, need {self.ml.min_train} to start predicting")

    def _print_log(self, message: str):
        """Emit a lightweight timestamped runtime log line."""
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [{now}] {message}", flush=True)

    def _print_header(self):
        """Print the live dashboard header."""
        if sys.platform == "win32":
            os.system("cls")
        else:
            sys.stdout.write("\033[2J\033[H")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        c_trades = [t for t in self.completed_trades if t.strategy == "Confluence"]
        m_trades = [t for t in self.completed_trades if t.strategy == "ML_Ensemble"]

        c_pnl = sum(t.pnl for t in c_trades)
        m_pnl = sum(t.pnl for t in m_trades)
        c_wins = sum(1 for t in c_trades if t.pnl > 0)
        m_wins = sum(1 for t in m_trades if t.pnl > 0)
        c_wr = c_wins/len(c_trades)*100 if c_trades else 0
        m_wr = m_wins/len(m_trades)*100 if m_trades else 0

        lines = [
            "",
            "=" * 72,
            "  POLYMARKET 5-MIN BTC BOT -- LIVE DRY RUN",
            f"  {now}    BTC: ${self.btc_price:,.2f}",
            "=" * 72,
            "",
            f"  Windows Processed: {self.total_windows:<6}  ML Trained: {'YES' if self.ml.is_trained else 'NO'} ({len(self.ml.X)} samples)",
            "",
            "  +-----------------+--------+-------+----------+------------+",
            "  | Strategy        | Trades | Win%  | P&L      | Expectancy |",
            "  +-----------------+--------+-------+----------+------------+",
            f"  | Confluence      | {len(c_trades):>6} | {c_wr:>4.1f}% | ${c_pnl:>+7.2f} | ${c_pnl/len(c_trades) if c_trades else 0:>+7.2f}/t |",
            f"  | ML Ensemble     | {len(m_trades):>6} | {m_wr:>4.1f}% | ${m_pnl:>+7.2f} | ${m_pnl/len(m_trades) if m_trades else 0:>+7.2f}/t |",
            "  +-----------------+--------+-------+----------+------------+",
            "",
        ]

        # Pending trades
        if self.pending_trades:
            lines.append("  >> PENDING TRADES (waiting for window close):")
            for pt in self.pending_trades:
                lines.append(f"     [{pt.strategy:<12}] {pt.direction:<4} | "
                            f"Conf: {pt.confidence:.1%} | Entry: ${pt.entry_price:,.2f} | "
                            f"Window: {pt.window_start}-{pt.window_end}")
            lines.append("")

        # Current window status
        if self.window_start_ts > 0:
            ws = datetime.fromtimestamp(self.window_start_ts, tz=timezone.utc).strftime("%H:%M:%S")
            we = datetime.fromtimestamp(self.window_end_ts, tz=timezone.utc).strftime("%H:%M:%S")
            elapsed = int(time.time() - self.window_start_ts)
            candle_count = len(self.window_candles)
            status = "OBSERVING" if candle_count < 3 else ("DECISION MADE" if self.signal_emitted else "ANALYZING")
            lines.append(f"  >> CURRENT WINDOW: {ws} - {we} | Candle {candle_count}/5 | {elapsed}s elapsed | {status}")
            if self.window_candles:
                wp = self.window_candles[0].open
                cp = self.window_candles[-1].close
                ret = (cp - wp) / wp * 100
                lines.append(f"     Window Open: ${wp:,.2f} | Current: ${cp:,.2f} | Move: {ret:+.4f}%")
            lines.append("")

        # Recent completed trades (last 10)
        recent = self.completed_trades[-10:]
        if recent:
            lines.append("  -- RECENT TRADES " + "-" * 53)
            lines.append(f"  {'Time':<12} {'Strat':<14} {'Dir':<5} {'Outcome':<8} {'P&L':>8} {'Conf':>6} {'Reasoning'}")
            lines.append(f"  {'-'*12} {'-'*14} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*20}")
            for t in recent:
                icon = "+" if t.pnl > 0 else "X"
                lines.append(
                    f"  {t.window_start[-8:]:<12} {t.strategy:<14} {t.direction:<5} "
                    f"{'WIN ' + icon if t.pnl > 0 else 'LOSS ' + icon:<8} "
                    f"${t.pnl:>+7.2f} {t.confidence:>5.1%} {t.reasoning[:30]}"
                )
            lines.append("")

        lines.append("  Press Ctrl+C to stop")
        lines.append("")

        print("\n".join(lines), flush=True)

    def _on_candle_close(self, candle: Candle):
        """Called when a 1-minute candle closes."""
        self.candle_history.append(candle)
        self.btc_price = candle.close

        # Limit history to 200 candles
        if len(self.candle_history) > 200:
            self.candle_history = self.candle_history[-200:]

        # Determine which 5-min window this candle belongs to
        ws, we = self._get_window_boundaries(candle.ts_open)

        # New window?
        if ws != self.window_start_ts:
            # Resolve previous window
            if self.window_start_ts > 0 and self.window_candles:
                self._resolve_window()

            # Start new window
            self.window_start_ts = ws
            self.window_end_ts = we
            self.window_candles = [candle]
            self.window_open_price = candle.open
            self.signal_emitted = False
            self.total_windows += 1
        else:
            self.window_candles.append(candle)

        # At candle 3: make decision
        if len(self.window_candles) == 3 and not self.signal_emitted:
            self._make_decision()
            self.signal_emitted = True

    def _make_decision(self):
        """At minute 3 of the window, evaluate both strategies."""
        hist = self.candle_history[:-3]  # Everything before this window
        if len(hist) < 30:
            return

        first_3 = self.window_candles[:3]
        features = build_features(hist[-120:], first_3)
        if not features:
            return

        entry_price = first_3[-1].close
        ws = datetime.fromtimestamp(self.window_start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        we = datetime.fromtimestamp(self.window_end_ts, tz=timezone.utc).strftime("%H:%M")

        # Confluence
        sig_c = self.confluence.evaluate(features)
        if sig_c:
            trade = PaperTrade(
                window_start=ws, window_end=we, strategy="Confluence",
                direction=sig_c["direction"], confidence=sig_c["confidence"],
                entry_price=entry_price, open_price=self.window_open_price,
                reasoning=sig_c["reasoning"],
            )
            self.pending_trades.append(trade)

        # ML
        sig_ml = self.ml.predict(features)
        if sig_ml:
            trade = PaperTrade(
                window_start=ws, window_end=we, strategy="ML_Ensemble",
                direction=sig_ml["direction"], confidence=sig_ml["confidence"],
                entry_price=entry_price, open_price=self.window_open_price,
                reasoning=sig_ml["reasoning"],
            )
            self.pending_trades.append(trade)

        # Store features for ML training (will be resolved when window closes)
        self._pending_features = features

    def _resolve_window(self):
        """Window ended - resolve pending trades and train ML."""
        if not self.window_candles:
            return

        close_price = self.window_candles[-1].close
        outcome = "UP" if close_price >= self.window_open_price else "DOWN"

        # Resolve pending trades for this window
        ws = datetime.fromtimestamp(self.window_start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        resolved = []
        remaining = []

        for trade in self.pending_trades:
            if trade.window_start == ws:
                trade.close_price = close_price
                trade.outcome = outcome
                trade.resolved = True
                if trade.direction == outcome:
                    trade.pnl = round(self.WIN, 2)
                else:
                    trade.pnl = round(-self.BET, 2)
                self.completed_trades.append(trade)
                resolved.append(trade)
            else:
                remaining.append(trade)

        self.pending_trades = remaining

        # Train ML on this resolved window
        if hasattr(self, '_pending_features') and self._pending_features:
            self.ml.add(self._pending_features, outcome)
            if self.ml.should_retrain():
                self.ml.train()
            self._pending_features = None


    async def stream_binance(self):
        url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        self._print_log("Binance BTCUSDT volume stream connected (secondary)")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                k = data.get("k", {})
                                ts = int(k["t"]) // 1000
                                self.binance_cache[ts] = {'v': float(k["v"]), 'n': int(k["n"]), 'V': float(k["V"]), 'c': float(k["c"])}
            except Exception as e:
                await asyncio.sleep(3)



    async def stream_chainlink(self):

        """Poll Chainlink BTC/USD on Polygon every 10s via free RPC."""

        rpc_idx = 0

        cl_logged = False

        while True:

            try:

                rpc_url = POLYGON_RPCS[rpc_idx % len(POLYGON_RPCS)]

                payload = {

                    "jsonrpc": "2.0", "id": 1, "method": "eth_call",

                    "params": [{"to": CHAINLINK_BTC_USD, "data": LATEST_ROUND_DATA_SIG}, "latest"]

                }

                async with aiohttp.ClientSession() as session:

                    async with session.post(rpc_url, json=payload,

                                            timeout=aiohttp.ClientTimeout(total=6)) as r:

                        resp = await r.json()

                        result = resp.get("result", "0x")

                        if result and len(result) >= 66:

                            hex_answer = result[2 + 64:2 + 128]

                            answer = int(hex_answer, 16)

                            if answer > 2**255:

                                answer -= 2**256

                            price = answer / 1e8

                            if price > 1000:

                                self.chainlink_price = price

                                if not cl_logged:

                                    self._print_log(f"Chainlink BTC/USD oracle connected: ${price:,.2f} (tertiary)")

                                    cl_logged = True

            except Exception as e:

                rpc_idx += 1

            await asyncio.sleep(10)



    async def stream_coinbase(self):
        url = "wss://ws-feed.exchange.coinbase.com"
        min_start = 0
        cur_open = 0; cur_high = 0; cur_low = 0; cur_close = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        await ws.send_json({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]})
                        self._print_log("Coinbase BTC-USD primary price stream connected")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if data.get("type") == "ticker":
                                    p = float(data["price"])
                                    self.btc_price = p
                                    evt_ts = datetime.fromisoformat(data["time"].replace("Z", "+00:00")).timestamp()
                                    c_min = int(evt_ts) - (int(evt_ts) % 60)
                                    
                                    if c_min != min_start:
                                        if min_start > 0:
                                            bcache = getattr(self, 'binance_cache', {}).get(min_start, {'v': 0, 'n': 0, 'V': 0, 'c': cur_close})
                                            candle = Candle(
                                                ts_open=min_start, ts_close=min_start+59.9,
                                                open=cur_open, high=cur_high, low=cur_low, close=cur_close,
                                                volume=bcache['v'], trades=bcache['n'], taker_buy_vol=bcache['V'],
                                                binance_price=bcache['c'],
                                                chainlink_price=self.chainlink_price,
                                            )
                                            self._on_candle_close(candle)
                                            
                                        min_start = c_min
                                        cur_open = p; cur_high = p; cur_low = p; cur_close = p
                                    else:
                                        cur_high = max(cur_high, p)
                                        cur_low = min(cur_low, p)
                                        cur_close = p
                                        
                                    self.current_tick = {
                                        "time": c_min, "open": cur_open, "high": cur_high,
                                        "low": cur_low, "close": cur_close
                                    }
            except Exception as e:
                self._print_log(f"Coinbase WS error: {e}")
                await asyncio.sleep(3)

    def _build_state_json(self):
        c_trades = [t for t in self.completed_trades if t.strategy == "Confluence"]
        m_trades = [t for t in self.completed_trades if t.strategy == "ML_Ensemble"]
        def td(t):
            return {"window_idx": t.window_start, "window_start": t.window_start,
                    "window_end": t.window_end, "strategy": t.strategy,
                    "direction": t.direction, "confidence": t.confidence,
                    "entry_price": t.entry_price, "open_price": t.open_price,
                    "close_price": t.close_price, "outcome": t.outcome,
                    "pnl": t.pnl, "reasoning": t.reasoning,
                    "is_real": getattr(t, 'is_real', False)}
        ws_str = we_str = ""
        w_candles = w_elapsed = 0
        w_move = w_open = 0.0
        w_status = "WAITING"
        if self.window_start_ts > 0:
            ws_str = datetime.fromtimestamp(self.window_start_ts, tz=timezone.utc).strftime("%H:%M:%S")
            we_str = datetime.fromtimestamp(self.window_end_ts, tz=timezone.utc).strftime("%H:%M:%S")
            w_candles = len(self.window_candles)
            w_elapsed = int(time.time() - self.window_start_ts)
            if self.window_candles:
                w_open = self.window_candles[0].open
                cp = self.window_candles[-1].close
                w_move = (cp - w_open) / w_open * 100 if w_open else 0
            w_status = "OBSERVING" if w_candles < 3 else ("DECISION MADE" if self.signal_emitted else "ANALYZING")

        chart_candles = []
        for c in self.candle_history[-60:]:
            chart_candles.append({"time": int(c.ts_open), "open": round(c.open, 2),
                "high": round(c.high, 2), "low": round(c.low, 2),
                "close": round(c.close, 2), "volume": round(c.volume, 2)})

        # Get latest binance price from cache
        last_binance = 0
        if hasattr(self, 'binance_cache') and self.binance_cache:
            try:
                last_binance = self.binance_cache[max(self.binance_cache.keys())]['c']
            except: pass

        return {
            "btc_price": self.btc_price, 
            "binance_price": last_binance,
            "chainlink_price": self.chainlink_price,
            "total_windows": self.total_windows,
            "ml_trained": self.ml.is_trained, "ml_samples": len(self.ml.X),
            "confluence_trades": [td(t) for t in c_trades],
            "ml_trades": [td(t) for t in m_trades],
            "pending_trades": [{"strategy": p.strategy, "direction": p.direction,
                "confidence": p.confidence, "entry_price": p.entry_price,
                "window_start": p.window_start, "window_end": p.window_end,
                "is_real": getattr(p, 'is_real', False)} for p in self.pending_trades],
            "window_start": ws_str, "window_end": we_str,
            "window_candles": w_candles, "window_elapsed": w_elapsed,
            "window_move": round(w_move, 4), "window_open_price": round(w_open, 2),
            "window_status": w_status,
            "chart_candles": chart_candles, "current_tick": self.current_tick,
            "mode": getattr(self, 'MODE', 'DRY RUN'),
            "trade_capital": 10.0,
            "total_real_trades": sum(1 for t in self.completed_trades if getattr(t, 'is_real', False)),
        }

    async def _ws_broadcast_loop(self):
        while True:
            if hasattr(self, '_ws_clients') and self._ws_clients:
                try:
                    state = json.dumps(self._build_state_json())
                    dead = set()
                    for ws in self._ws_clients:
                        try: await ws.send_str(state)
                        except: dead.add(ws)
                    self._ws_clients -= dead
                except: pass
            await asyncio.sleep(1.0)

    async def _handle_ws(self, request):
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)
        if not hasattr(self, '_ws_clients'): self._ws_clients = set()
        self._ws_clients.add(ws)
        async for msg in ws: pass
        self._ws_clients.discard(ws)
        return ws

    async def _handle_index(self, request):
        html_path = Path(__file__).parent / "dashboard.html"
        return aiohttp.web.FileResponse(html_path)

    async def _start_web_server(self, port=8080):
        app = aiohttp.web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        # Auto-detect free port
        for p in range(port, port + 10):
            try:
                site = aiohttp.web.TCPSite(runner, "0.0.0.0", p)
                await site.start()
                self._print_log(f"Dashboard live at http://localhost:{p}")
                return runner
            except OSError:
                self._print_log(f"Port {p} in use, trying next...")
        raise RuntimeError(f"Could not bind to any port in range {port}-{port+9}")
    async def run(self):
        """Main entry point — starts bot + web dashboard + Telegram."""
        print()
        print("=" * 64)
        print("  POLYMARKET 5-MIN BTC BOT -- LIVE DRY RUN (Triple-Feed)")
        print("  Coinbase + Binance + Chainlink | Paper Trading | $10/trade")
        print("=" * 64)
        print()

        await self.bootstrap()

        # Start web dashboard
        runner = await self._start_web_server(port=8080)

        # Telegram startup message
        if self.tg.enabled:
            print(f"  [+] Telegram notifications active (chat: {TG_CHAT})")
            await self.tg.notify_startup(self.btc_price, len(self.ml.X))
        else:
            print("  [!] Telegram not configured — check Poly/.env")

        print()
        print("  [*] Feeds: Coinbase (primary) + Binance (volume) + Chainlink (oracle)")
        print("  [*] Dashboard: open the URL shown above in your browser")
        print("  [*] Press Ctrl+C to stop")
        print()
        print("  " + "-" * 60)
        print()

        await asyncio.gather(
            self.stream_coinbase(),
            self.stream_binance(),
            self.stream_chainlink(),
            self._ws_broadcast_loop(),
            self.tg.poll_commands(),
        )


# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    bot = LiveDryRun()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n\n  [*] Bot stopped by user.\n")
        for strat in ["Confluence", "ML_Ensemble"]:
            trades = [t for t in bot.completed_trades if t.strategy == strat]
            if trades:
                w = sum(1 for t in trades if t.pnl > 0)
                pnl = sum(t.pnl for t in trades)
                print(f"  [{strat}] {len(trades)} trades | {w}/{len(trades)} wins "
                      f"({w/len(trades)*100:.1f}%) | P&L: ${pnl:+.2f}")
        # Send shutdown notification
        if bot.tg.enabled:
            try:
                asyncio.run(bot.tg.notify_shutdown(bot.completed_trades))
            except Exception:
                pass
        print()

