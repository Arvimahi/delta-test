"""
Polymarket 5-Min BTC Bot - Main Runner
Demo mode: reads markets + generates signals, NO real trades placed.

Usage:
    python bot_runner.py              # Run bot (demo mode)
    python bot_runner.py --once       # Single scan and exit
"""

import asyncio
import argparse
import json
import logging
import time
import sys
from datetime import datetime
from pathlib import Path

from bot_engine import (
    BotState, BinanceFeed, PolymarketFetcher,
    RegimeDetector, SignalEngine,
    get_current_btc_price, BTCTick
)

log = logging.getLogger("btc_bot.runner")

STATE_FILE = Path("bot_state.json")


class BotRunner:
    def __init__(self, demo: bool = True):
        self.demo = demo
        self.state = BotState()
        self.feed = BinanceFeed(self.state)
        self.fetcher = PolymarketFetcher()
        self.regime_detector = RegimeDetector()
        self.signal_engine = SignalEngine()
        self._running = False

    # ─── Market Loop ──────────────────────────────────────────────────────────

    async def market_loop(self):
        """Every 10s: fetch markets, update odds, run regime + signal."""
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                log.error(f"Market loop error: {e}")
            await asyncio.sleep(10)

    async def _tick(self):
        now = time.time()

        # Fetch active 5-min BTC markets
        markets = await self.fetcher.get_active_5m_markets()

        if not markets:
            log.info("⏳ No active 5-min BTC markets found. Waiting...")
            return

        # Find the currently active window (started, not yet ended)
        active = next(
            (m for m in markets if m.start_ts <= now <= m.end_ts), None
        )
        # Find the next upcoming window
        upcoming = next(
            (m for m in markets if m.start_ts > now), None
        )

        if active:
            # Refresh live odds
            fresh_yes, fresh_no = await self.fetcher.get_market_odds(active)
            active.yes_price = fresh_yes
            active.no_price = fresh_no
            self.state.active_market = active

        self.state.next_market = upcoming

        # Regime detection
        regime, scores = self.regime_detector.detect(self.state.btc_ticks)
        self.state.regime = regime
        self.state.regime_score = scores

        # Signal generation
        target = active or upcoming
        if target:
            secs_left = target.end_ts - now
            signal = self.signal_engine.generate(self.state, target, regime, scores)

            if signal:
                self.state.last_signal = signal
                self.state.signals_today.append({
                    "ts": now,
                    "strategy": signal.strategy,
                    "direction": signal.direction,
                    "edge": round(signal.edge, 4),
                    "confidence": signal.confidence,
                    "reasoning": signal.reasoning,
                    "secs_left": round(secs_left, 1),
                    "btc_price": get_current_btc_price(self.state),
                })
                self._log_signal(signal, secs_left)
            else:
                self.state.skips += 1

        self._save_state()

    def _log_signal(self, signal, secs_left):
        arrow = "🟢 UP" if signal.direction == "UP" else "🔴 DOWN"
        log.info(
            f"[SIGNAL] {arrow} | Strategy: {signal.strategy} | "
            f"Edge: {signal.edge:.1%} | Conf: {signal.confidence} | "
            f"{secs_left:.0f}s left | {signal.reasoning}"
        )
        if self.demo:
            log.info("  ↳ [DEMO MODE] Signal logged. No order placed.")

    # ─── State Persistence ────────────────────────────────────────────────────

    def _save_state(self):
        btc_price = get_current_btc_price(self.state)
        active = self.state.active_market
        next_m = self.state.next_market
        sig = self.state.last_signal

        out = {
            "ts": time.time(),
            "btc_price": btc_price,
            "regime": self.state.regime,
            "regime_scores": self.state.regime_score,
            "active_market": {
                "slug": active.slug if active else None,
                "yes_price": active.yes_price if active else None,
                "no_price": active.no_price if active else None,
                "end_ts": active.end_ts if active else None,
                "secs_left": round(active.end_ts - time.time(), 1) if active else None,
                "liquidity": active.liquidity if active else None,
            } if active else None,
            "next_market": {
                "slug": next_m.slug if next_m else None,
                "starts_in": round(next_m.start_ts - time.time(), 1) if next_m else None,
            } if next_m else None,
            "last_signal": {
                "strategy": sig.strategy,
                "direction": sig.direction,
                "model_prob": round(sig.model_prob, 4),
                "market_prob": round(sig.market_prob, 4),
                "edge": round(sig.edge, 4),
                "confidence": sig.confidence,
                "reasoning": sig.reasoning,
            } if sig else None,
            "stats": {
                "signals_today": len(self.state.signals_today),
                "wins": self.state.wins,
                "losses": self.state.losses,
                "skips": self.state.skips,
            },
            "recent_signals": self.state.signals_today[-20:],
        }

        try:
            STATE_FILE.write_text(json.dumps(out, indent=2))
        except Exception as e:
            log.debug(f"State save error: {e}")

    # ─── Main Entry ──────────────────────────────────────────────────────────

    async def run(self):
        self._running = True
        log.info("=" * 60)
        log.info("  Polymarket 5-Min BTC Bot  |  DEMO MODE" if self.demo else "  Polymarket 5-Min BTC Bot  |  LIVE MODE")
        log.info("=" * 60)
        log.info("Starting Binance price feed + market scanner...")

        # Run both concurrently
        await asyncio.gather(
            self.feed.run(),
            self._delayed_market_loop(),
        )

    async def _delayed_market_loop(self):
        """Wait for a few ticks before starting market loop."""
        log.info("Waiting 5s for BTC price feed to warm up...")
        await asyncio.sleep(5)
        await self.market_loop()

    async def run_once(self):
        """Single scan - useful for testing."""
        log.info("Running single scan...")
        # Populate with a single real price fetch first
        async with __import__("aiohttp").ClientSession() as s:
            async with s.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT") as r:
                d = await r.json()
                self.state.btc_ticks.append(BTCTick(float(d["price"]), time.time()))
                log.info(f"BTC Price: ${float(d['price']):,.2f}")

        await self._tick()

        state_data = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
        print("\n" + json.dumps(state_data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Polymarket 5-Min BTC Bot")
    parser.add_argument("--once", action="store_true", help="Single scan and exit")
    parser.add_argument("--live", action="store_true", help="Enable live trading (requires wallet)")
    args = parser.parse_args()

    demo = not args.live
    runner = BotRunner(demo=demo)

    if args.once:
        asyncio.run(runner.run_once())
    else:
        try:
            asyncio.run(runner.run())
        except KeyboardInterrupt:
            log.info("Bot stopped.")


if __name__ == "__main__":
    main()
