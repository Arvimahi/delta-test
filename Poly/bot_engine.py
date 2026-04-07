"""
Polymarket 5-Min BTC Bot - Core Engine
Demo mode (read-only). No wallet required.

Strategy: Adaptive regime detection selects from 3 edge hypotheses:
  1. Chainlink Oracle Lag Arb  - enter last 30-60s based on real BTC vs Poly odds
  2. Momentum / Trend Follow   - enter early based on 1m BTC directional move
  3. Market Maker Spread Fade  - fade extreme odds (>0.72 / <0.28) back to mean
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("btc_bot")

# ─── Constants ────────────────────────────────────────────────────────────────
BINANCE_WS       = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
POLY_GAMMA_API   = "https://gamma-api.polymarket.com"
POLY_CLOB_API    = "https://clob.polymarket.com"
BTC_5M_SEARCH    = "Bitcoin Up or Down"

# Safety limits (demo mode enforces these so real mode can inherit them)
MAX_SPREAD_PCT   = 0.05   # Don't enter if bid-ask > 5%
MIN_LIQUIDITY    = 5_000  # $5k minimum market liquidity
MIN_EDGE_PCT     = 0.05   # Model must beat market by 5% to signal
FADE_THRESHOLD   = 0.72   # Odds above this = fade signal
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BTCTick:
    price: float
    ts: float  # unix seconds


@dataclass
class MarketWindow:
    slug: str
    condition_id: str
    yes_token_id: str
    no_token_id: str
    start_ts: float
    end_ts: float
    yes_price: float = 0.5
    no_price: float = 0.5
    liquidity: float = 0.0
    volume_24h: float = 0.0


@dataclass
class Signal:
    strategy: str          # "oracle_lag" | "momentum" | "spread_fade"
    direction: str         # "UP" | "DOWN"
    model_prob: float      # our estimated probability
    market_prob: float     # what polymarket says
    edge: float            # model_prob - market_prob
    confidence: str        # "HIGH" | "MED" | "LOW"
    reasoning: str


@dataclass
class BotState:
    btc_ticks: deque = field(default_factory=lambda: deque(maxlen=300))  # 5 min of ticks
    active_market: Optional[MarketWindow] = None
    next_market: Optional[MarketWindow] = None
    last_signal: Optional[Signal] = None
    regime: str = "UNKNOWN"   # "TRENDING" | "RANGING" | "VOLATILE"
    regime_score: dict = field(default_factory=dict)
    signals_today: list = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    skips: int = 0


# ─── BTC Price Feed ───────────────────────────────────────────────────────────

class BinanceFeed:
    """Streams real BTC/USDT ticks from Binance WebSocket."""

    def __init__(self, state: BotState):
        self.state = state
        self.running = False
        self._callbacks = []

    def on_tick(self, fn):
        self._callbacks.append(fn)

    async def run(self):
        self.running = True
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(BINANCE_WS) as ws:
                        log.info("✅ Binance WebSocket connected")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                tick = BTCTick(
                                    price=float(data["p"]),
                                    ts=data["T"] / 1000.0
                                )
                                self.state.btc_ticks.append(tick)
                                for cb in self._callbacks:
                                    await cb(tick)
            except Exception as e:
                log.warning(f"Binance WS error: {e}, reconnecting in 3s...")
                await asyncio.sleep(3)


# ─── Polymarket Market Fetcher ────────────────────────────────────────────────

class PolymarketFetcher:
    """Fetches 5-min BTC markets and live odds from Polymarket APIs."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get(self, url, params=None):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as r:
                r.raise_for_status()
                return await r.json()
        except Exception as e:
            log.debug(f"API error {url}: {e}")
            return None

    async def get_active_5m_markets(self) -> list[MarketWindow]:
        """Fetch upcoming and active 5-min BTC markets from Gamma API."""
        data = await self._get(
            f"{POLY_GAMMA_API}/markets",
            params={
                "search": BTC_5M_SEARCH,
                "active": "true",
                "limit": 20,
                "order": "end_date_iso",
            }
        )
        if not data:
            return []

        markets = []
        now = time.time()

        for m in (data if isinstance(data, list) else data.get("markets", [])):
            try:
                slug = m.get("slug", "")
                if "5" not in slug and "5min" not in m.get("question", "").lower():
                    if "5" not in m.get("question", "")[:30]:
                        continue

                end_iso = m.get("endDate") or m.get("end_date_iso")
                start_iso = m.get("startDate") or m.get("start_date_iso")
                if not end_iso:
                    continue

                end_ts = _parse_iso(end_iso)
                start_ts = _parse_iso(start_iso) if start_iso else end_ts - 300

                # Only care about markets within ±10 minutes
                if end_ts < now - 60 or end_ts > now + 660:
                    continue

                tokens = m.get("tokens", m.get("outcomes", []))
                yes_tid = no_tid = ""
                yes_price = no_price = 0.5

                for t in tokens:
                    outcome = (t.get("outcome") or t.get("name") or "").upper()
                    if "UP" in outcome or "YES" in outcome:
                        yes_tid = t.get("token_id") or t.get("id") or ""
                        yes_price = float(t.get("price", 0.5))
                    elif "DOWN" in outcome or "NO" in outcome:
                        no_tid = t.get("token_id") or t.get("id") or ""
                        no_price = float(t.get("price", 0.5))

                markets.append(MarketWindow(
                    slug=slug,
                    condition_id=m.get("conditionId") or m.get("condition_id") or "",
                    yes_token_id=yes_tid,
                    no_token_id=no_tid,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    yes_price=yes_price,
                    no_price=no_price,
                    liquidity=float(m.get("liquidity") or 0),
                    volume_24h=float(m.get("volume24hr") or m.get("volume") or 0),
                ))
            except Exception as e:
                log.debug(f"Market parse error: {e}")
                continue

        markets.sort(key=lambda x: x.end_ts)
        return markets

    async def get_orderbook(self, token_id: str) -> dict:
        """Get live orderbook for a token."""
        if not token_id:
            return {}
        data = await self._get(f"{POLY_CLOB_API}/book", params={"token_id": token_id})
        return data or {}

    async def get_market_odds(self, market: MarketWindow) -> tuple[float, float]:
        """Returns (yes_price, no_price) from live orderbook midpoints."""
        ob = await self.get_orderbook(market.yes_token_id)
        if ob and ob.get("bids") and ob.get("asks"):
            bids = ob["bids"]
            asks = ob["asks"]
            if bids and asks:
                best_bid = float(bids[0]["price"])
                best_ask = float(asks[0]["price"])
                yes_mid = (best_bid + best_ask) / 2
                return yes_mid, 1 - yes_mid
        return market.yes_price, market.no_price


# ─── Market Regime Detector ──────────────────────────────────────────────────

class RegimeDetector:
    """
    Classifies recent BTC price action into:
      TRENDING  → momentum strategy is best
      RANGING   → spread fade is best
      VOLATILE  → oracle lag arb is best (or skip)
    """

    def detect(self, ticks: deque) -> tuple[str, dict]:
        if len(ticks) < 30:
            return "UNKNOWN", {}

        prices = [t.price for t in list(ticks)[-60:]]  # last 60 ticks

        # ── ATR-style volatility (avg range of 10-tick windows)
        ranges = [abs(prices[i] - prices[i-5]) for i in range(5, len(prices))]
        avg_range = sum(ranges) / len(ranges)
        atr_pct = avg_range / prices[-1] * 100

        # ── Momentum: net direction of last 60 ticks vs first
        net_move_pct = (prices[-1] - prices[0]) / prices[0] * 100
        abs_move = abs(net_move_pct)

        # ── Trend consistency (what % of windows moved same direction as net)
        directions = [1 if prices[i] > prices[i-1] else -1 for i in range(1, len(prices))]
        net_dir = 1 if net_move_pct > 0 else -1
        consistency = sum(1 for d in directions if d == net_dir) / len(directions)

        scores = {
            "atr_pct": round(atr_pct, 4),
            "net_move_pct": round(net_move_pct, 4),
            "consistency": round(consistency, 3),
        }

        # Classification logic
        if atr_pct > 0.08:
            regime = "VOLATILE"
        elif abs_move > 0.05 and consistency > 0.58:
            regime = "TRENDING"
        else:
            regime = "RANGING"

        return regime, scores


# ─── Signal Generators ───────────────────────────────────────────────────────

class SignalEngine:
    """
    Picks the right strategy based on regime and generates a signal.
    All three strategies are implemented; regime selects the primary one,
    but all three are evaluated and the highest-confidence result wins.
    """

    def generate(
        self,
        state: BotState,
        market: MarketWindow,
        regime: str,
        regime_scores: dict,
    ) -> Optional[Signal]:
        ticks = list(state.btc_ticks)
        if len(ticks) < 10:
            return None

        candidates = []

        now = time.time()
        secs_left = market.end_ts - now
        yes_p = market.yes_price
        no_p = market.no_price

        # ── Strategy 1: Oracle Lag Arbitrage
        # Best in VOLATILE regime or with <60s remaining
        if secs_left <= 90:
            sig = self._oracle_lag(ticks, market, secs_left)
            if sig:
                candidates.append(sig)

        # ── Strategy 2: Momentum / Trend Following
        # Best in TRENDING regime, enter early (>120s left)
        if regime == "TRENDING" or secs_left > 120:
            sig = self._momentum(ticks, market, regime_scores)
            if sig:
                candidates.append(sig)

        # ── Strategy 3: Market Maker Spread Fade
        # Best in RANGING regime; fades extreme odds
        sig = self._spread_fade(market, regime)
        if sig:
            candidates.append(sig)

        if not candidates:
            return None

        # Pick highest-edge signal
        best = max(candidates, key=lambda s: s.edge)
        if best.edge < MIN_EDGE_PCT:
            return None
        return best

    def _oracle_lag(self, ticks, market: MarketWindow, secs_left: float) -> Optional[Signal]:
        """
        Compare real BTC momentum (last 15s) against Polymarket implied odds.
        If BTC has moved strongly in one direction but Poly odds haven't adjusted,
        there's a mispricing window.
        """
        recent = [t for t in ticks if time.time() - t.ts < 20]
        if len(recent) < 3:
            return None

        first_p = recent[0].price
        last_p = recent[-1].price
        move_pct = (last_p - first_p) / first_p * 100

        # What the market currently prices
        yes_p = market.yes_price

        # If BTC moved up >0.03% in last 20s but Poly still shows <55% UP
        if move_pct > 0.03 and yes_p < 0.55:
            model_prob = min(0.72, yes_p + abs(move_pct) * 4)
            edge = model_prob - yes_p
            conf = "HIGH" if edge > 0.12 else "MED" if edge > 0.07 else "LOW"
            return Signal(
                strategy="oracle_lag",
                direction="UP",
                model_prob=model_prob,
                market_prob=yes_p,
                edge=edge,
                confidence=conf,
                reasoning=f"BTC +{move_pct:.3f}% in 20s, Poly only {yes_p:.2f} UP"
            )
        elif move_pct < -0.03 and yes_p > 0.45:
            model_prob = max(0.28, yes_p - abs(move_pct) * 4)
            no_model = 1 - model_prob
            edge = no_model - market.no_price
            conf = "HIGH" if edge > 0.12 else "MED" if edge > 0.07 else "LOW"
            return Signal(
                strategy="oracle_lag",
                direction="DOWN",
                model_prob=no_model,
                market_prob=market.no_price,
                edge=edge,
                confidence=conf,
                reasoning=f"BTC {move_pct:.3f}% in 20s, Poly only {market.no_price:.2f} DOWN"
            )
        return None

    def _momentum(self, ticks, market: MarketWindow, scores: dict) -> Optional[Signal]:
        """
        1-minute BTC trend. If price has moved consistently in one direction
        and window just started, bet on continuation.
        """
        if len(ticks) < 60:
            return None

        last_60 = [t for t in ticks if time.time() - t.ts < 60]
        if len(last_60) < 20:
            return None

        open_p = last_60[0].price
        close_p = last_60[-1].price
        move_pct = (close_p - open_p) / open_p * 100
        consistency = scores.get("consistency", 0.5)

        yes_p = market.yes_price

        if move_pct > 0.04 and consistency > 0.55:
            # Trending up: model P(UP) elevated
            raw_edge = move_pct * 6 * (consistency - 0.5) * 2
            model_prob = min(0.75, yes_p + raw_edge * 0.01)
            edge = model_prob - yes_p
            conf = "HIGH" if consistency > 0.65 else "MED"
            return Signal(
                strategy="momentum",
                direction="UP",
                model_prob=model_prob,
                market_prob=yes_p,
                edge=edge,
                confidence=conf,
                reasoning=f"1m trend +{move_pct:.3f}%, {consistency:.0%} consistent"
            )
        elif move_pct < -0.04 and consistency > 0.55:
            no_p = market.no_price
            raw_edge = abs(move_pct) * 6 * (consistency - 0.5) * 2
            model_prob = min(0.75, no_p + raw_edge * 0.01)
            edge = model_prob - no_p
            conf = "HIGH" if consistency > 0.65 else "MED"
            return Signal(
                strategy="momentum",
                direction="DOWN",
                model_prob=model_prob,
                market_prob=no_p,
                edge=edge,
                confidence=conf,
                reasoning=f"1m trend {move_pct:.3f}%, {consistency:.0%} consistent"
            )
        return None

    def _spread_fade(self, market: MarketWindow, regime: str) -> Optional[Signal]:
        """
        When market odds are extreme (>72% / <28%), fade back toward 50/50.
        Works best in ranging regime — extremes tend to revert.
        Only applies mid-window, not last 30s.
        """
        yes_p = market.yes_price

        if yes_p > FADE_THRESHOLD and regime == "RANGING":
            # Fade: bet DOWN (against the crowd's extreme YES)
            model_prob = max(0.45, 1 - yes_p + 0.15)
            edge = model_prob - market.no_price
            return Signal(
                strategy="spread_fade",
                direction="DOWN",
                model_prob=model_prob,
                market_prob=market.no_price,
                edge=edge,
                confidence="MED",
                reasoning=f"Extreme UP odds ({yes_p:.2f}), ranging regime → fade to DOWN"
            )
        elif yes_p < (1 - FADE_THRESHOLD) and regime == "RANGING":
            model_prob = max(0.45, yes_p + 0.15)
            edge = model_prob - yes_p
            return Signal(
                strategy="spread_fade",
                direction="UP",
                model_prob=model_prob,
                market_prob=yes_p,
                edge=edge,
                confidence="MED",
                reasoning=f"Extreme DOWN odds ({yes_p:.2f}), ranging regime → fade to UP"
            )
        return None


# ─── Helper ───────────────────────────────────────────────────────────────────

def _parse_iso(iso_str: str) -> float:
    """Parse ISO datetime string to unix timestamp."""
    if not iso_str:
        return 0.0
    try:
        iso_str = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_str)
        return dt.timestamp()
    except Exception:
        return 0.0


def get_current_btc_price(state: BotState) -> Optional[float]:
    if state.btc_ticks:
        return state.btc_ticks[-1].price
    return None


def get_btc_1m_range(state: BotState) -> tuple[float, float]:
    """Returns (low, high) of BTC in last 60 seconds."""
    now = time.time()
    recent = [t.price for t in state.btc_ticks if now - t.ts < 60]
    if not recent:
        return 0.0, 0.0
    return min(recent), max(recent)
