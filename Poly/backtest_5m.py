"""
Polymarket 5-Min BTC Bot — 24-Hour Backtester
=============================================
Downloads 1-second BTC/USDT klines from Binance for the last 24 hours,
simulates 5-minute market windows (as Polymarket would), runs all three
strategies against each window, and produces a detailed performance report.

Usage:
    py backtest_5m.py
"""

import asyncio
import aiohttp
import time
import json
import csv
import statistics
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ─── Re-use core engine types & logic ────────────────────────────────────────

# Copy constants from bot_engine
MIN_EDGE_PCT     = 0.05
FADE_THRESHOLD   = 0.72


@dataclass
class BTCTick:
    price: float
    ts: float


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
    liquidity: float = 50000.0
    volume_24h: float = 100000.0
    # Backtest additions
    open_price: float = 0.0
    close_price: float = 0.0
    outcome: str = ""  # "UP" or "DOWN"


@dataclass
class Signal:
    strategy: str
    direction: str
    model_prob: float
    market_prob: float
    edge: float
    confidence: str
    reasoning: str


@dataclass
class TradeResult:
    window_idx: int
    window_start: str
    window_end: str
    open_price: float
    close_price: float
    actual_outcome: str
    signal_direction: str
    strategy: str
    edge: float
    confidence: str
    market_prob: float
    model_prob: float
    pnl: float  # +1 win, -1 loss (unit bet on model_prob side)
    reasoning: str


# ─── Regime Detector (same as bot_engine) ─────────────────────────────────────

class RegimeDetector:
    def detect(self, ticks: list[BTCTick]) -> tuple[str, dict]:
        if len(ticks) < 30:
            return "UNKNOWN", {}

        prices = [t.price for t in ticks[-60:]]

        ranges = [abs(prices[i] - prices[i - 5]) for i in range(5, len(prices))]
        avg_range = sum(ranges) / len(ranges) if ranges else 0
        atr_pct = avg_range / prices[-1] * 100 if prices[-1] else 0

        net_move_pct = (prices[-1] - prices[0]) / prices[0] * 100 if prices[0] else 0
        abs_move = abs(net_move_pct)

        directions = [1 if prices[i] > prices[i - 1] else -1 for i in range(1, len(prices))]
        net_dir = 1 if net_move_pct > 0 else -1
        consistency = sum(1 for d in directions if d == net_dir) / len(directions) if directions else 0

        scores = {
            "atr_pct": round(atr_pct, 4),
            "net_move_pct": round(net_move_pct, 4),
            "consistency": round(consistency, 3),
        }

        if atr_pct > 0.08:
            regime = "VOLATILE"
        elif abs_move > 0.05 and consistency > 0.58:
            regime = "TRENDING"
        else:
            regime = "RANGING"

        return regime, scores


# ─── Signal Engine (same logic as bot_engine, but uses sim_time) ──────────────

class BacktestSignalEngine:
    """Same as SignalEngine but uses sim_time instead of time.time()."""

    def generate(
        self,
        ticks: list[BTCTick],
        market: MarketWindow,
        regime: str,
        regime_scores: dict,
        sim_time: float,
    ) -> Optional[Signal]:
        if len(ticks) < 10:
            return None

        candidates = []
        secs_left = market.end_ts - sim_time

        # Strategy 1: Oracle Lag Arbitrage (last 90 seconds)
        if secs_left <= 90:
            sig = self._oracle_lag(ticks, market, secs_left, sim_time)
            if sig:
                candidates.append(sig)

        # Strategy 2: Momentum (trending or >120s left)
        if regime == "TRENDING" or secs_left > 120:
            sig = self._momentum(ticks, market, regime_scores, sim_time)
            if sig:
                candidates.append(sig)

        # Strategy 3: Spread Fade
        sig = self._spread_fade(market, regime)
        if sig:
            candidates.append(sig)

        if not candidates:
            return None

        best = max(candidates, key=lambda s: s.edge)
        if best.edge < MIN_EDGE_PCT:
            return None
        return best

    def _oracle_lag(self, ticks, market, secs_left, sim_time):
        recent = [t for t in ticks if sim_time - t.ts < 20]
        if len(recent) < 3:
            return None

        first_p = recent[0].price
        last_p = recent[-1].price
        move_pct = (last_p - first_p) / first_p * 100

        yes_p = market.yes_price

        if move_pct > 0.03 and yes_p < 0.55:
            model_prob = min(0.72, yes_p + abs(move_pct) * 4)
            edge = model_prob - yes_p
            conf = "HIGH" if edge > 0.12 else "MED" if edge > 0.07 else "LOW"
            return Signal(
                strategy="oracle_lag", direction="UP",
                model_prob=model_prob, market_prob=yes_p, edge=edge,
                confidence=conf,
                reasoning=f"BTC +{move_pct:.3f}% in 20s, Poly only {yes_p:.2f} UP"
            )
        elif move_pct < -0.03 and yes_p > 0.45:
            model_prob = max(0.28, yes_p - abs(move_pct) * 4)
            no_model = 1 - model_prob
            edge = no_model - market.no_price
            conf = "HIGH" if edge > 0.12 else "MED" if edge > 0.07 else "LOW"
            return Signal(
                strategy="oracle_lag", direction="DOWN",
                model_prob=no_model, market_prob=market.no_price, edge=edge,
                confidence=conf,
                reasoning=f"BTC {move_pct:.3f}% in 20s, Poly only {market.no_price:.2f} DOWN"
            )
        return None

    def _momentum(self, ticks, market, scores, sim_time):
        if len(ticks) < 60:
            return None

        last_60 = [t for t in ticks if sim_time - t.ts < 60]
        if len(last_60) < 20:
            return None

        open_p = last_60[0].price
        close_p = last_60[-1].price
        move_pct = (close_p - open_p) / open_p * 100
        consistency = scores.get("consistency", 0.5)

        yes_p = market.yes_price

        if move_pct > 0.04 and consistency > 0.55:
            raw_edge = move_pct * 6 * (consistency - 0.5) * 2
            model_prob = min(0.75, yes_p + raw_edge * 0.01)
            edge = model_prob - yes_p
            conf = "HIGH" if consistency > 0.65 else "MED"
            return Signal(
                strategy="momentum", direction="UP",
                model_prob=model_prob, market_prob=yes_p, edge=edge,
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
                strategy="momentum", direction="DOWN",
                model_prob=model_prob, market_prob=no_p, edge=edge,
                confidence=conf,
                reasoning=f"1m trend {move_pct:.3f}%, {consistency:.0%} consistent"
            )
        return None

    def _spread_fade(self, market, regime):
        yes_p = market.yes_price

        if yes_p > FADE_THRESHOLD and regime == "RANGING":
            model_prob = max(0.45, 1 - yes_p + 0.15)
            edge = model_prob - market.no_price
            return Signal(
                strategy="spread_fade", direction="DOWN",
                model_prob=model_prob, market_prob=market.no_price, edge=edge,
                confidence="MED",
                reasoning=f"Extreme UP odds ({yes_p:.2f}), ranging → fade DOWN"
            )
        elif yes_p < (1 - FADE_THRESHOLD) and regime == "RANGING":
            model_prob = max(0.45, yes_p + 0.15)
            edge = model_prob - yes_p
            return Signal(
                strategy="spread_fade", direction="UP",
                model_prob=model_prob, market_prob=yes_p, edge=edge,
                confidence="MED",
                reasoning=f"Extreme DOWN odds ({yes_p:.2f}), ranging → fade UP"
            )
        return None


# ─── Synthetic Odds Generator ─────────────────────────────────────────────────

def compute_synthetic_odds(ticks_in_window: list[BTCTick], sim_time: float, window_start: float, window_end: float) -> tuple[float, float]:
    """
    Simulates what Polymarket odds would look like based on actual BTC
    price movement within the window. In reality the market reacts to
    price changes, so we model odds as a blend of:
        - 50/50 base at window open
        - Shifted toward the direction BTC has moved so far
        - With a lag (simulating slow market-maker updates)
    """
    if not ticks_in_window:
        return 0.50, 0.50

    open_price = ticks_in_window[0].price
    current_price = ticks_in_window[-1].price

    # How far through the window are we (0..1)
    progress = min(1.0, (sim_time - window_start) / (window_end - window_start))

    move_pct = (current_price - open_price) / open_price * 100

    # Odds shift more aggressively as the window progresses
    # But we add a "lag" — odds only adjust ~60-80% of the true implied probability
    lag_factor = 0.6 + 0.2 * progress  # 0.6 at start → 0.8 near end

    # Convert move to implied probability (logistic-ish)
    # A 0.05% move ≈ 60% implied, 0.1% ≈ 70%, 0.2% ≈ 85%
    import math
    raw_prob_up = 1 / (1 + math.exp(-move_pct * 30))  # sensitivity tuned

    # Apply lag: odds converge toward raw_prob but lagged
    yes_p = 0.50 + (raw_prob_up - 0.50) * lag_factor
    yes_p = max(0.05, min(0.95, yes_p))  # clamp

    return round(yes_p, 4), round(1 - yes_p, 4)


# ─── Data Fetching ────────────────────────────────────────────────────────────

async def fetch_1s_klines(hours: int = 24) -> list[BTCTick]:
    """
    Fetch 1-second BTC/USDT klines from Binance for the last N hours.
    Binance allows max 1000 per request, so we paginate.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - hours * 3600 * 1000

    ticks = []
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

                    for candle in data:
                        # Use close price and close time
                        close_ts = candle[6] / 1000.0  # close time ms → s
                        close_price = float(candle[4])  # close price
                        ticks.append(BTCTick(price=close_price, ts=close_ts))

                    last_ts = data[-1][6]  # last close time
                    current_start = last_ts + 1

                    batch += 1
                    if batch % 10 == 0:
                        fetched_mins = len(ticks) / 60
                        print(f"  📡 Fetched {len(ticks):,} ticks ({fetched_mins:.0f} mins)...")

                    # Rate limit (Binance allows 1200/min but be cautious)
                    await asyncio.sleep(0.08)

            except Exception as e:
                print(f"  ⚠️  Fetch error: {e}, retrying...")
                await asyncio.sleep(1)

    print(f"  ✅ Total: {len(ticks):,} ticks over {len(ticks)/3600:.1f} hours")
    return ticks


# ─── Backtest Engine ──────────────────────────────────────────────────────────

def run_backtest(all_ticks: list[BTCTick]) -> list[TradeResult]:
    """
    Simulate 5-minute windows across the entire tick dataset.
    For each window:
      1. Determine open/close price → actual outcome (UP/DOWN)
      2. Every 10 seconds within the window, build a tick buffer,
         compute synthetic odds, run regime + signal engine
      3. Take the FIRST valid signal per window (earliest entry)
      4. Evaluate P&L: win if signal direction matches outcome
    """
    if not all_ticks:
        return []

    regime_detector = RegimeDetector()
    signal_engine = BacktestSignalEngine()

    # Define 5-minute windows aligned to the data
    first_ts = all_ticks[0].ts
    last_ts = all_ticks[-1].ts

    # Align to 5-min boundaries
    window_start = first_ts - (first_ts % 300) + 300  # next clean 5-min boundary
    window_duration = 300  # 5 minutes

    results = []
    window_idx = 0

    while window_start + window_duration <= last_ts:
        window_end = window_start + window_duration

        # Get all ticks in this window
        window_ticks = [t for t in all_ticks if window_start <= t.ts <= window_end]
        if len(window_ticks) < 30:
            window_start += window_duration
            continue

        # Determine actual outcome
        open_price = window_ticks[0].price
        close_price = window_ticks[-1].price
        actual_outcome = "UP" if close_price >= open_price else "DOWN"

        # Simulate checking every 10 seconds within the window
        best_signal = None
        best_signal_time = None

        for check_offset in range(0, window_duration, 10):
            sim_time = window_start + check_offset

            # Build tick buffer: up to 300 ticks before sim_time (for regime/momentum)
            buffer_start = sim_time - 300
            tick_buffer = [t for t in all_ticks if buffer_start <= t.ts <= sim_time]

            if len(tick_buffer) < 30:
                continue

            # Get ticks within this window up to sim_time (for odds)
            window_ticks_so_far = [t for t in all_ticks if window_start <= t.ts <= sim_time]

            # Compute synthetic polymarket odds
            yes_p, no_p = compute_synthetic_odds(window_ticks_so_far, sim_time, window_start, window_end)

            # Build market window
            market = MarketWindow(
                slug=f"btc-5min-{window_idx}",
                condition_id="",
                yes_token_id="",
                no_token_id="",
                start_ts=window_start,
                end_ts=window_end,
                yes_price=yes_p,
                no_price=no_p,
                open_price=open_price,
                close_price=close_price,
                outcome=actual_outcome,
            )

            # Regime detection
            regime, scores = regime_detector.detect(tick_buffer)

            # Signal generation
            signal = signal_engine.generate(tick_buffer, market, regime, scores, sim_time)

            if signal:
                best_signal = signal
                best_signal_time = sim_time
                break  # take the first signal (earliest entry)

        if best_signal:
            # P&L: flat $10 bet on the signaled side
            # Win pays out at (1/market_prob) * bet - bet  ≈  edge-proportional
            # Loss loses the bet
            bet_size = 10.0
            if best_signal.direction == actual_outcome:
                # Win: profit = bet * (1/market_prob - 1), i.e. inverse odds payout
                payout = bet_size * (1.0 / best_signal.market_prob - 1.0)
                pnl = round(payout, 2)
            else:
                pnl = -bet_size

            results.append(TradeResult(
                window_idx=window_idx,
                window_start=datetime.fromtimestamp(window_start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                window_end=datetime.fromtimestamp(window_end, tz=timezone.utc).strftime("%H:%M"),
                open_price=round(open_price, 2),
                close_price=round(close_price, 2),
                actual_outcome=actual_outcome,
                signal_direction=best_signal.direction,
                strategy=best_signal.strategy,
                edge=round(best_signal.edge, 4),
                confidence=best_signal.confidence,
                market_prob=round(best_signal.market_prob, 4),
                model_prob=round(best_signal.model_prob, 4),
                pnl=pnl,
                reasoning=best_signal.reasoning,
            ))

        window_start += window_duration
        window_idx += 1

    return results


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_report(results: list[TradeResult], total_windows: int):
    if not results:
        print("\n❌ No signals generated during this period.")
        return

    wins = [r for r in results if r.pnl > 0]
    losses = [r for r in results if r.pnl <= 0]
    total_pnl = sum(r.pnl for r in results)
    win_rate = len(wins) / len(results) * 100

    # Strategy breakdown
    strategies = {}
    for r in results:
        s = r.strategy
        if s not in strategies:
            strategies[s] = {"trades": 0, "wins": 0, "pnl": 0.0}
        strategies[s]["trades"] += 1
        if r.pnl > 0:
            strategies[s]["wins"] += 1
        strategies[s]["pnl"] += r.pnl

    # Confidence breakdown
    confidences = {}
    for r in results:
        c = r.confidence
        if c not in confidences:
            confidences[c] = {"trades": 0, "wins": 0, "pnl": 0.0}
        confidences[c]["trades"] += 1
        if r.pnl > 0:
            confidences[c]["wins"] += 1
        confidences[c]["pnl"] += r.pnl

    # Streak tracking
    max_win_streak = max_loss_streak = 0
    current_streak = 0
    streak_type = None
    for r in results:
        if r.pnl > 0:
            if streak_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if streak_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    # Running P&L for drawdown
    running_pnl = []
    cum = 0
    for r in results:
        cum += r.pnl
        running_pnl.append(cum)
    peak = running_pnl[0]
    max_drawdown = 0
    for p in running_pnl:
        if p > peak:
            peak = p
        dd = peak - p
        if dd > max_drawdown:
            max_drawdown = dd

    avg_win = statistics.mean([r.pnl for r in wins]) if wins else 0
    avg_loss = statistics.mean([r.pnl for r in losses]) if losses else 0
    profit_factor = abs(sum(r.pnl for r in wins) / sum(r.pnl for r in losses)) if losses and sum(r.pnl for r in losses) != 0 else float('inf')
    avg_edge = statistics.mean([r.edge for r in results])

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  📊  POLYMARKET 5-MIN BTC BACKTEST — 24 HOUR REPORT")
    print(f"{sep}\n")

    print(f"  Total 5-min Windows:     {total_windows}")
    print(f"  Signals Generated:       {len(results)}")
    print(f"  Windows Skipped:         {total_windows - len(results)}")
    print(f"  Signal Rate:             {len(results)/total_windows*100:.1f}%\n")

    print(f"  ─── Performance ($10 flat bet) ───────────────────────")
    print(f"  Wins:                    {len(wins)}")
    print(f"  Losses:                  {len(losses)}")
    print(f"  Win Rate:                {win_rate:.1f}%")
    print(f"  Total P&L:               ${total_pnl:+.2f}")
    print(f"  Avg Win:                 ${avg_win:+.2f}")
    print(f"  Avg Loss:                ${avg_loss:.2f}")
    print(f"  Profit Factor:           {profit_factor:.2f}")
    print(f"  Avg Edge:                {avg_edge:.2%}")
    print(f"  Max Drawdown:            ${max_drawdown:.2f}")
    print(f"  Max Win Streak:          {max_win_streak}")
    print(f"  Max Loss Streak:         {max_loss_streak}\n")

    print(f"  ─── Strategy Breakdown ───────────────────────────────")
    for s, d in sorted(strategies.items()):
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        print(f"  {s:16s}  Trades: {d['trades']:3d}  Win: {wr:5.1f}%  P&L: ${d['pnl']:+.2f}")

    print(f"\n  ─── Confidence Breakdown ─────────────────────────────")
    for c in ["HIGH", "MED", "LOW"]:
        if c in confidences:
            d = confidences[c]
            wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
            print(f"  {c:16s}  Trades: {d['trades']:3d}  Win: {wr:5.1f}%  P&L: ${d['pnl']:+.2f}")

    # Hourly breakdown
    print(f"\n  ─── Hourly Breakdown ─────────────────────────────────")
    hourly = {}
    for r in results:
        hour = r.window_start.split(" ")[1][:2] + ":00"
        if hour not in hourly:
            hourly[hour] = {"trades": 0, "pnl": 0.0, "wins": 0}
        hourly[hour]["trades"] += 1
        hourly[hour]["pnl"] += r.pnl
        if r.pnl > 0:
            hourly[hour]["wins"] += 1

    for h in sorted(hourly):
        d = hourly[h]
        wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
        bar = "█" * max(1, int(abs(d["pnl"]) / 3))
        sign = "+" if d["pnl"] >= 0 else "-"
        print(f"  {h}  {d['trades']:2d} trades  {wr:5.1f}% WR  ${d['pnl']:+7.2f}  {bar}")

    # Equity curve (text-based)
    print(f"\n  ─── Equity Curve ─────────────────────────────────────")
    min_eq = min(running_pnl)
    max_eq = max(running_pnl)
    range_eq = max_eq - min_eq if max_eq != min_eq else 1
    chart_width = 40
    for i in range(0, len(running_pnl), max(1, len(running_pnl) // 20)):
        val = running_pnl[i]
        pos = int((val - min_eq) / range_eq * chart_width)
        bar = " " * pos + "●"
        print(f"  {results[i].window_start[-5:]}  ${val:+7.2f}  {bar}")
    # Final point
    val = running_pnl[-1]
    pos = int((val - min_eq) / range_eq * chart_width)
    bar = " " * pos + "●"
    print(f"  FINAL  ${val:+7.2f}  {bar}")

    print(f"\n{sep}")
    print(f"  ✅ Backtest complete.")
    print(f"{sep}\n")


def save_trades_csv(results: list[TradeResult], filepath: str = "backtest_trades.csv"):
    if not results:
        return
    keys = [
        "window_idx", "window_start", "window_end", "open_price", "close_price",
        "actual_outcome", "signal_direction", "strategy", "edge", "confidence",
        "market_prob", "model_prob", "pnl", "reasoning"
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in keys})
    print(f"  📁 Trades saved to {filepath}")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("  📈 Polymarket 5-Min BTC Backtester — Last 24 Hours")
    print("=" * 60)
    print()

    # 1. Fetch data
    print("  🔄 Downloading 24h of 1-second BTC/USDT data from Binance...")
    ticks = await fetch_1s_klines(hours=24)

    if len(ticks) < 300:
        print("  ❌ Not enough data. Exiting.")
        return

    first_time = datetime.fromtimestamp(ticks[0].ts, tz=timezone.utc)
    last_time = datetime.fromtimestamp(ticks[-1].ts, tz=timezone.utc)
    price_range = (min(t.price for t in ticks), max(t.price for t in ticks))
    print(f"\n  📅 Period: {first_time.strftime('%Y-%m-%d %H:%M')} → {last_time.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  💰 BTC Range: ${price_range[0]:,.2f} — ${price_range[1]:,.2f}")
    print(f"  📊 Ticks: {len(ticks):,}\n")

    # 2. Run backtest
    print("  ⏱️  Running backtest across all 5-minute windows...")
    total_windows = int((ticks[-1].ts - ticks[0].ts) // 300)
    results = run_backtest(ticks)

    # 3. Report
    print_report(results, total_windows)

    # 4. Save CSV
    save_trades_csv(results, str(Path(__file__).parent / "backtest_trades.csv"))


if __name__ == "__main__":
    asyncio.run(main())
