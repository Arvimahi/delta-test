"""
Polymarket 30-Day Setup Backtester
==================================
Simulates the exact live bot architecture (Confluence & ML) over 
the last 30 days of 1-minute Binance historical BTC/USDT data.
"""

import asyncio
import aiohttp
import time
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
import sys

# Import core engine modules from the live environment
from live_dryrun import (
    Candle, build_features, ConfluenceStrategy, MLStrategy
)

# ── Backtest Config ──
DAYS = 14
TRADE_SIZE = 3.0  # $3 as requested


class TradeSim:
    def __init__(self):
        self.strategy = ""
        self.direction = ""
        self.confidence = 0.0
        self.entry_price = 0.0
        self.synthetic_prob = 0.0
        self.pnl = 0.0
        self.reasoning = ""
        self.window_start = ""
        self.window_end = ""


def compute_synthetic_odds(open_price: float, current_price: float) -> tuple[float, float]:
    """
    Simulates what Polymarket odds would look like at Minute 3 based on 
    actual BTC price movement within the first 3 minutes of the window.
    """
    move_pct = (current_price - open_price) / open_price * 100
    # Logistic curve: ~0.05% move scales odds heavily (50c -> ~58c)
    raw_prob_up = 1 / (1 + math.exp(-move_pct * 30)) 
    
    # Lag factor (market maker inefficiency + fees simulation)
    # Market rarely goes above 95c or below 5c unless extremely close to expiry
    yes_p = 0.50 + (raw_prob_up - 0.50) * 0.8
    yes_p = max(0.05, min(0.95, yes_p))
    return round(yes_p, 4), round(1 - yes_p, 4)


async def fetch_30d_klines() -> list[Candle]:
    base_url = "https://api.binance.com/api/v3/klines"
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (DAYS * 24 * 3600 * 1000)
    
    candles = []
    current_start = start_ms
    
    print(f"  🔄 Downloading {DAYS} days of 1m BTC/USDT data from Binance...")
    
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
                    
                    for row in data:
                        # [open_ts, open, high, low, close, vol, close_ts, quote_vol, trades...]
                        c = Candle(
                            ts_open=row[0] / 1000.0,
                            ts_close=row[6] / 1000.0,
                            open=float(row[1]),
                            high=float(row[2]),
                            low=float(row[3]),
                            close=float(row[4]),
                            volume=float(row[5]),
                            trades=int(row[8]),
                            taker_buy_vol=float(row[9])
                        )
                        candles.append(c)
                        
                    current_start = data[-1][6] + 1
                    
                    fetched_days = len(candles) / (24 * 60)
                    print(f"  📡 Fetched {len(candles):,} candles (~{fetched_days:.1f} days)...", end="\r")
                    
                    await asyncio.sleep(0.05)
            except Exception as e:
                print(f"\n  ⚠️ Fetch error: {e}, retrying...")
                await asyncio.sleep(1)
                
    print(f"\n  ✅ Total: {len(candles):,} 1m candles.")
    return candles


def run_simulation(all_candles: list[Candle]):
    confluence = ConfluenceStrategy()
    ml = MLStrategy()
    
    c_trades = []
    m_trades = []
    
    candle_history = []
    
    print("  ⏱️  Running historical 5-minute simulation loop...")
    
    # Ensure aligned to full 5-minute bounds (00, 05, 10...)
    # The actual bot processes window_candles chunked by alignment
    
    window_start_ts = 0
    window_candles = []
    window_open_price = 0
    
    c_pending = None
    m_pending = None
    c3_features = None
    
    for i, c in enumerate(all_candles):
        candle_history.append(c)
        if len(candle_history) > 200:
            candle_history.pop(0)
            
        ms = int(c.ts_open * 1000)
        
        # Determine 5-minute boundary logic similar to live_dryrun
        remainder = ms % 300000
        ws = (ms - remainder) / 1000.0
        we = ws + 300.0
        
        if ws != window_start_ts:
            if c3_features is not None and window_start_ts > 0:
                # Resolve using last known close price before window change
                last_c = candle_history[-2] if len(candle_history) > 1 else c
                outcome = "UP" if last_c.close >= window_open_price else "DOWN"
                
                ml.add(c3_features, outcome)
                if ml.should_retrain():
                    ml.train()
                
                def resolve(pending_trade: TradeSim, trade_list: list):
                    if not pending_trade: return
                    bet = TRADE_SIZE
                    if pending_trade.direction == outcome:
                        payout_ratio = 1.0 / pending_trade.synthetic_prob
                        pending_trade.pnl = round(bet * (payout_ratio - 1.0), 2)
                    else:
                        pending_trade.pnl = round(-bet, 2)
                    trade_list.append(pending_trade)

                resolve(c_pending, c_trades)
                resolve(m_pending, m_trades)

            window_start_ts = ws
            window_candles = [c]
            window_open_price = c.open
            c_pending = None
            m_pending = None
            c3_features = None
        else:
            window_candles.append(c)
            
        # At >= Minute 3
        if remainder >= 120000 and c3_features is None:
            if len(candle_history) >= 130 and len(window_candles) >= 1:
                past_history = candle_history[:-len(window_candles)]
                c3_features = build_features(past_history[-120:], window_candles)
                
                if c3_features:
                    entry_price = c.close
                    
                    # Predict
                    sig_c = confluence.evaluate(c3_features)
                    sig_m = ml.predict(c3_features)
                    
                    up_odds, down_odds = compute_synthetic_odds(window_open_price, entry_price)
                    
                    if sig_c:
                        t = TradeSim()
                        t.strategy = "Confluence"
                        t.direction = sig_c["direction"]
                        t.confidence = sig_c["confidence"]
                        t.reasoning = sig_c["reasoning"]
                        t.entry_price = entry_price
                        t.synthetic_prob = up_odds if t.direction == "UP" else down_odds
                        t.window_start = ws
                        t.window_end = we
                        c_pending = t
                        
                    if sig_m:
                        t = TradeSim()
                        t.strategy = "ML"
                        t.direction = sig_m["direction"]
                        t.confidence = sig_m["confidence"]
                        t.reasoning = sig_m["reasoning"]
                        t.entry_price = entry_price
                        t.synthetic_prob = up_odds if t.direction == "UP" else down_odds
                        t.window_start = ws
                        t.window_end = we
                        m_pending = t

    # Final resolution for the last window
    if c3_features is not None and window_start_ts > 0:
        last_c = candle_history[-1]
        outcome = "UP" if last_c.close >= window_open_price else "DOWN"
        
        ml.add(c3_features, outcome)
        if ml.should_retrain():
            ml.train()
        
        def resolve_final(pending_trade: TradeSim, trade_list: list):
            if not pending_trade: return
            bet = TRADE_SIZE
            if pending_trade.direction == outcome:
                payout_ratio = 1.0 / pending_trade.synthetic_prob
                pending_trade.pnl = round(bet * (payout_ratio - 1.0), 2)
            else:
                pending_trade.pnl = round(-bet, 2)
            trade_list.append(pending_trade)

        resolve_final(c_pending, c_trades)
        resolve_final(m_pending, m_trades)

    return c_trades, m_trades


def print_report(name: str, trades: list[TradeSim], total_windows: int):
    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  📊  {name.upper()} RESULTS — LAST {DAYS} DAYS")
    print(f"{sep}\n")

    if not trades:
        print("  ❌ No trades taken.")
        return

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in trades)
    wr = len(wins) / len(trades) * 100

    max_dd = 0
    peak = 0
    cum = 0
    for t in trades:
        cum += t.pnl
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    profit_f = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if sum(r.pnl for r in losses) != 0 else float('inf')

    # Breakdown by confidence bands
    bands = {"HIGH (>70%)": [], "MED (60-70%)": [], "LOW (<60%)": []}
    for t in trades:
        if t.confidence >= 0.70: bands["HIGH (>70%)"].append(t)
        elif t.confidence >= 0.60: bands["MED (60-70%)"].append(t)
        else: bands["LOW (<60%)"].append(t)

    print(f"  Windows Scanned:   {total_windows:,}")
    print(f"  Total Trades:      {len(trades)}  ({len(trades)/total_windows*100:.1f}% activity)")
    print(f"  Win / Loss:        {len(wins)} / {len(losses)}  ({wr:.1f}% WR)")
    print(f"  Total P&L:         ${total_pnl:+.2f}  ($3 / trade)")
    print(f"  Profit Factor:     {profit_f:.2f}")
    print(f"  Max Drawdown:      ${max_dd:.2f}")

    print(f"\n  ── Confidence Breakdown ──")
    for b_name, b_trades in bands.items():
        if b_trades:
            b_wr = sum(1 for t in b_trades if t.pnl > 0) / len(b_trades) * 100
            b_pnl = sum(t.pnl for t in b_trades)
            print(f"  {b_name:12s} | Trades: {len(b_trades):<3} | WR: {b_wr:4.1f}% | P&L: ${b_pnl:+.2f}")


async def main():
    print("=" * 60)
    print(f"  📈 {DAYS}-Day Polymarket BTC 5m Simulator (Live Logic Base)")
    print("=" * 60)
    
    candles = await fetch_30d_klines()
    if len(candles) < 300:
        return
        
    c_trades, m_trades = run_simulation(candles)
    total_w = len(candles) // 5
    
    print_report("Confluence Strategy", c_trades, total_w)
    print_report("ML Ensemble", m_trades, total_w)

    print("\n  ✅ Backtest Complete.\n")


if __name__ == "__main__":
    asyncio.run(main())
