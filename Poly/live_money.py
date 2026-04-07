"""
Polymarket 5-Min BTC Bot - LIVE MONEY
======================================
Real execution on Polymarket CLOB.
$1/trade | 1-2¢ slippage | Hold to expiry.

Usage:
    py live_money.py
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

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.system("")

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
)
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv(Path(__file__).parent / ".env")
TG_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TG_CHAT_ID", "")
POLY_API_KEY = os.getenv("POLY_API_KEY", "")
POLY_API_SECRET = os.getenv("POLY_API_SECRET", "")
POLY_API_PASSPHRASE = os.getenv("POLY_API_PASSPHRASE", "")
POLY_WALLET = os.getenv("POLY_WALLET", "")
POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", "")
if POLY_PRIVATE_KEY and not POLY_PRIVATE_KEY.startswith("0x"):
    POLY_PRIVATE_KEY = "0x" + POLY_PRIVATE_KEY

# ── Trade Config ──
BET_SIZE_USDC = 3.0       # $3/trade capital
MIN_SHARES    = 5         # 5 shares per trade
MAX_BUY_PRICE = 0.92      # Never buy above 92c
SELL_PRICE    = 0.99      # Auto-sell TP price
SELL_SHARES   = 4.98      # Sell 4.98 shares (Polymarket fee buffer)
MAX_SLIPPAGE  = 0.01      # 1 cent max slippage on limit price
POLY_GAMMA_API = "https://gamma-api.polymarket.com"
POLY_CLOB_HOST = "https://clob.polymarket.com"

# Import everything from live_dryrun (shared code)
sys.path.insert(0, str(Path(__file__).parent))
from live_dryrun import (
    Candle, PaperTrade, build_features, ConfluenceStrategy, MLStrategy,
    TelegramNotifier, FEATURE_NAMES,
    ema, rsi, bollinger, atr, macd, stochastic, williams_r,
    obv_slope, vwap, taker_buy_ratio,
    POLYGON_RPCS, CHAINLINK_BTC_USD, LATEST_ROUND_DATA_SIG,
)


# ============================================================================
#  POLYMARKET EXECUTION ENGINE
# ============================================================================

@dataclass
class ActiveMarket:
    """Represents a live 5-min BTC market on Polymarket."""
    condition_id: str
    yes_token_id: str
    no_token_id: str
    start_ts: float
    end_ts: float
    yes_price: float = 0.5
    no_price: float = 0.5
    slug: str = ""


class PolyExecutor:
    """Finds active markets & places real orders via py_clob_client."""

    def __init__(self):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        creds = ApiCreds(
            api_key=POLY_API_KEY,
            api_secret=POLY_API_SECRET,
            api_passphrase=POLY_API_PASSPHRASE,
        )
        self.client = ClobClient(
            POLY_CLOB_HOST, key=POLY_PRIVATE_KEY, chain_id=137,
            creds=creds, signature_type=2, funder=POLY_WALLET,
        )
        try:
            self.client.get_api_keys()
            self._session = None
            print(f"  [POLY] Executor initialized (wallet: {POLY_WALLET[:10]}...)")
        except Exception:
            print(f"  [POLY] Provided API credentials invalid! Deriving new ones...")
            new_creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(new_creds)
            self._session = None
            print(f"  [POLY] Derived new API credentials (wallet: {POLY_WALLET[:10]}...)")

    async def _get(self, url, params=None):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.get(url, params=params,
                                          timeout=aiohttp.ClientTimeout(total=8)) as r:
                r.raise_for_status()
                return await r.json()
        except Exception as e:
            print(f"  [POLY] API error: {e}")
            return None

    async def find_active_market(self) -> Optional[ActiveMarket]:
        """Find the currently active 5-min BTC market using Gamma API."""
        from datetime import datetime as dt

        now_utc = dt.now(timezone.utc)
        now = time.time()

        # Use date-range filtering: markets ending within next 7 minutes
        data = await self._get(f"{POLY_GAMMA_API}/markets", params={
            "end_date_min": now_utc.isoformat(),
            "end_date_max": (now_utc + __import__('datetime').timedelta(minutes=7)).isoformat(),
            "closed": "false",
            "limit": 20,
            "order": "endDate",
            "ascending": "true",
        })
        if not data:
            return None

        markets_raw = data if isinstance(data, list) else data.get("markets", [])

        for m in markets_raw:
            try:
                slug = m.get("slug", "")
                q = m.get("question", "")

                # Match BTC 5-min markets: slug like "btc-updown-5m-..."
                # or question like "Bitcoin Up or Down ... 5 min"
                is_btc_5m = "btc-updown-5m" in slug and "15m" not in slug
                if not is_btc_5m:
                    q_lower = q.lower()
                    is_btc_5m = ("bitcoin" in q_lower and
                                 ("-5m" in slug or " 5 min" in q_lower or "5:" in q) and
                                 ("15m" not in slug and "15 min" not in q_lower) and
                                 ("up" in q_lower and "down" in q_lower))
                if not is_btc_5m:
                    continue

                end_iso = m.get("endDate") or m.get("endDateIso")
                if not end_iso:
                    continue
                end_ts = dt.fromisoformat(end_iso.replace("Z", "+00:00")).timestamp()
                start_iso = m.get("startDate") or m.get("startDateIso")
                start_ts = (dt.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp()
                            if start_iso else end_ts - 300)

                # Parse clobTokenIds (JSON string like '["id1","id2"]')
                clob_ids_raw = m.get("clobTokenIds", "")
                outcomes_raw = m.get("outcomes", "")
                prices_raw = m.get("outcomePrices", "")

                try:
                    clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw or []
                    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw or []
                    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw or []
                except (json.JSONDecodeError, TypeError):
                    continue

                if len(clob_ids) < 2 or len(outcomes) < 2:
                    continue

                yes_tid = no_tid = ""
                yes_price = no_price = 0.5

                for i, outcome in enumerate(outcomes):
                    out_upper = str(outcome).upper()
                    tid = str(clob_ids[i]) if i < len(clob_ids) else ""
                    price = float(prices[i]) if i < len(prices) else 0.5
                    if "UP" in out_upper or "YES" in out_upper:
                        yes_tid = tid
                        yes_price = price
                    elif "DOWN" in out_upper or "NO" in out_upper:
                        no_tid = tid
                        no_price = price

                if yes_tid and no_tid:
                    print(f"  [POLY] Found market: {slug} | Up={yes_price:.3f} Down={no_price:.3f}")
                    return ActiveMarket(
                        condition_id=m.get("conditionId", ""),
                        yes_token_id=yes_tid,
                        no_token_id=no_tid,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        yes_price=yes_price,
                        no_price=no_price,
                        slug=slug,
                    )
            except Exception as e:
                print(f"  [POLY] Parse error: {e}")
                continue
        return None

    async def get_live_price(self, token_id: str) -> float:
        """Get best ask price for a token from the orderbook."""
        data = await self._get(f"{POLY_CLOB_HOST}/book", params={"token_id": token_id})
        if data and data.get("asks"):
            return float(data["asks"][0]["price"])
        return 0.5

    def place_order(self, token_id: str, price: float, size_usdc: float) -> dict:
        """Place a BUY limit order. Enforces Polymarket minimum of 5 shares."""
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY

        # Round price to valid tick (0.01), clamp to valid range
        price_rounded = round(max(0.01, min(0.99, price)), 2)

        # Compute shares from USDC budget, enforce minimum of 5
        desired_shares = size_usdc / price_rounded
        shares = max(desired_shares, MIN_SHARES)
        shares = round(shares, 0)  # whole shares
        actual_cost = round(shares * price_rounded, 2)

        order_args = OrderArgs(
            token_id=token_id,
            price=price_rounded,
            size=shares,
            side=BUY,
        )

        try:
            result = self.client.create_and_post_order(order_args)
            return {"success": True, "result": result, "price": price_rounded,
                    "shares": shares, "cost_usdc": actual_cost}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def place_sell_order(self, token_id: str, price: float, shares: float) -> dict:
        """Place a SELL limit order to take profit."""
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import SELL

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=shares,
            side=SELL,
        )

        try:
            result = self.client.create_and_post_order(order_args)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
#  LIVE MONEY BOT
# ============================================================================

@dataclass
class LiveTrade:
    window_start: str
    window_end: str
    strategy: str
    direction: str
    confidence: float
    entry_price: float       # BTC price at signal
    open_price: float        # BTC price at window open
    close_price: float = 0.0
    outcome: str = ""
    pnl: float = 0.0
    resolved: bool = False
    reasoning: str = ""
    # Polymarket-specific
    token_id: str = ""
    order_price: float = 0.0   # What we paid per share
    shares: float = 0.0
    order_id: str = ""
    cost_usdc: float = 0.0
    is_real: bool = False
    fill_verified: bool = False  # Only True if order confirmed filled


class LiveMoneyBot:
    def __init__(self):
        self.candle_history: list[Candle] = []
        self.confluence = ConfluenceStrategy()
        
        ml_path = Path(__file__).parent / 'ml_model.joblib'
        if ml_path.exists():
            self.ml = joblib.load(ml_path)
            self.ml.retrain_every = 99999999
            print(f"  [+] Loaded offline trained ML model from {ml_path}")
        else:
            self.ml = MLStrategy()
            
        self.executor = PolyExecutor()
        self.tg = TelegramNotifier(TG_TOKEN, TG_CHAT, bot_ref=self)

        self.window_candles: list[Candle] = []
        self.window_start_ts: float = 0
        self.window_end_ts: float = 0
        self.window_open_price: float = 0
        self.signal_emitted: bool = False

        self.pending_trades: list[LiveTrade] = []
        self.completed_trades: list[LiveTrade] = []
        self.total_windows = 0
        self.btc_price = 0.0
        self.binance_cache = {}
        self.chainlink_price = 0.0
        self.current_tick = None

        # Active market for current window
        self._active_market: Optional[ActiveMarket] = None

        # Dashboard WebSocket clients
        self._ws_clients = set()

    def _print_log(self, msg):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [{ts}] {msg}", flush=True)

    def _get_window_boundaries(self, ts):
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
        print("  [*] Pre-training ML model...")
        hist = self.candle_history
        for i in range(35, len(hist) - 5, 5):
            wc = hist[i:i+5]
            if len(wc) < 4: continue
            h = hist[:i]
            if len(h) < 30: continue
            features = build_features(h[-120:], wc[:3])
            if not features: continue
            outcome = "UP" if wc[-1].close >= wc[0].open else "DOWN"
            self.ml.add(features, outcome)
        if len(self.ml.X) >= self.ml.min_train:
            self.ml.train()
            print(f"  [+] ML pre-trained ({len(self.ml.X)} samples)")

    def _on_candle_close(self, candle):
        self.candle_history.append(candle)
        self.btc_price = candle.close
        if len(self.candle_history) > 200:
            self.candle_history = self.candle_history[-200:]

        ws, we = self._get_window_boundaries(candle.ts_open)
        if ws != self.window_start_ts:
            if self.window_start_ts > 0 and self.window_candles:
                self._resolve_window()
            self.window_start_ts = ws
            self.window_end_ts = we
            self.window_candles = [candle]
            self.window_open_price = candle.open
            self.signal_emitted = False
            self.total_windows += 1
            self._active_market = None
        else:
            self.window_candles.append(candle)

        if len(self.window_candles) == 2 and not self.signal_emitted:
            asyncio.ensure_future(self._make_decision_live())
            self.signal_emitted = True

    async def _place_take_profit(self, token_id: str, approx_shares: float, delay: int = 4):
        """Places a SELL order at $0.99 a few seconds after entry to instantly recycle capital."""
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        import math
        
        await asyncio.sleep(delay)
        
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
            # Fetch real balance since partial fills or fees can result in fractional shares (e.g. 4.98)
            b = self.executor.client.get_balance_allowance(params=params)
            raw_balance = int(b.get("balance", "0"))
            if raw_balance > 0:
                actual_shares = math.floor((raw_balance / 10**6) * 100) / 100.0
            else:
                actual_shares = approx_shares
        except Exception as e:
            self._print_log(f"⚠ Could not fetch exact balance, defaulting to {approx_shares} ({e})")
            actual_shares = approx_shares

        # Fallback safeguard
        if actual_shares < 0.01:
            self._print_log(f"❌ AUTO-SELL TP ABORTED | Insufficient balance ({actual_shares} shares)")
            return

        sell_result = self.executor.place_sell_order(token_id, SELL_PRICE, SELL_SHARES)
        if sell_result.get("success"):
            self._print_log(f"✅ AUTO-SELL TP PLACED | {actual_shares:.2f} shares @ $0.99")
        else:
            self._print_log(f"❌ AUTO-SELL TP FAILED | {sell_result.get('error')}")

    async def _make_decision_live(self):
        """At minute 2: Confluence signal + ML confirmation = unified trade."""
        hist = self.candle_history[:-3]
        if len(hist) < 30:
            self._print_log(f"Not enough history ({len(hist)} candles), need 30")
            return
        first_n = self.window_candles[:2]  # 2 candles at minute 2
        features = build_features(hist[-120:], first_n)
        if not features:
            self._print_log("build_features returned None -- skipping")
            return

        entry_price = first_n[-1].close
        ws = datetime.fromtimestamp(self.window_start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        we = datetime.fromtimestamp(self.window_end_ts, tz=timezone.utc).strftime("%H:%M")

        skip_reasons = []

        # -- Step 1: Confluence generates the primary signal --
        sig_c = self.confluence.evaluate(features)
        if not sig_c:
            skip_reasons.append("Confluence: no signal (weak/mixed)")
            self._print_log(f"Window {ws}-{we}: Confluence skip")
            asyncio.ensure_future(self.tg.notify_skip(ws, we, skip_reasons))
            self._pending_features = features
            return

        # -- Step 2: ML must CONFIRM the Confluence direction --
        sig_ml = self.ml.predict(features)
        if not sig_ml:
            skip_reasons.append(f"ML did not confirm (not trained or low conf)")
            self._print_log(f"Window {ws}-{we}: ML did not confirm Confluence {sig_c['direction']}")
            asyncio.ensure_future(self.tg.notify_skip(ws, we, skip_reasons))
            self._pending_features = features
            return

        if sig_ml["direction"] != sig_c["direction"]:
            skip_reasons.append(f"ML disagrees: Conf={sig_c['direction']} vs ML={sig_ml['direction']}")
            self._print_log(f"Window {ws}-{we}: ML disagrees (Conf={sig_c['direction']}, ML={sig_ml['direction']})")
            asyncio.ensure_future(self.tg.notify_skip(ws, we, skip_reasons))
            self._pending_features = features
            return

        # -- Both agree! Merge into unified signal --
        direction = sig_c["direction"]
        conf = (sig_c["confidence"] * 0.45 + sig_ml["confidence"] * 0.55)  # ML-weighted
        reasoning = f"Conf+ML | {sig_c['reasoning']} | ML={sig_ml['confidence']:.1%}"

        self._print_log(
            f"CONFIRMED | {direction} | Confluence={sig_c['confidence']:.1%} + "
            f"ML={sig_ml['confidence']:.1%} = Merged {conf:.1%}"
        )

        # -- Step 3: Find active Polymarket market --
        market = await self.executor.find_active_market()
        if not market:
            self._print_log(f"Window {ws}-{we}: confirmed signal but no market!")
            asyncio.ensure_future(self.tg.notify_skip(ws, we, ["No active BTC 5-min market"]))
            self._pending_features = features
            return

        # -- Step 3b: Check time-to-expiry (skip if < 90s to resolution) --
        time_to_expiry = market.end_ts - time.time()
        if time_to_expiry < 90:
            self._print_log(
                f"SKIP | Market expires in {time_to_expiry:.0f}s (<90s) - too late for fill"
            )
            self._pending_features = features
            return

        # -- Step 3c: Check if market is already effectively resolved --
        # If YES or NO price is > 0.93, the market has basically decided
        if market.yes_price > 0.93 or market.no_price > 0.93:
            self._print_log(
                f"SKIP | Market already resolved (YES={market.yes_price:.2f} NO={market.no_price:.2f})"
            )
            self._pending_features = features
            return

        # -- Step 4: Check price and execute --
        trade = LiveTrade(
            window_start=ws, window_end=we, strategy="Confirmed",
            direction=direction, confidence=conf,
            entry_price=entry_price, open_price=self.window_open_price,
            reasoning=reasoning,
        )

        token_id = market.yes_token_id if direction == "UP" else market.no_token_id
        ask_price = await self.executor.get_live_price(token_id)
        limit_price = min(ask_price + MAX_SLIPPAGE, MAX_BUY_PRICE)

        # Check if ask is too high (max buy price cap)
        if ask_price > MAX_BUY_PRICE:
            self._print_log(
                f"SKIP EXECUTION | Ask ${ask_price:.3f} > max ${MAX_BUY_PRICE:.2f} | "
                f"Signal was {direction} {conf:.1%}"
            )
            # Still record as paper trade for tracking
            trade.is_real = False
            self.pending_trades.append(trade)
            self._pending_features = features
            return

        self._print_log(
            f"EXECUTING {direction} | Conf={conf:.1%} | "
            f"Ask=${ask_price:.3f} Limit=${limit_price:.3f} | "
            f"{MIN_SHARES} shares @ ~${MIN_SHARES * ask_price:.2f} | "
            f"Market: {market.slug}"
        )

        result = self.executor.place_order(token_id, limit_price, BET_SIZE_USDC)
        if result["success"]:
            trade.is_real = True
            trade.token_id = token_id
            trade.order_price = result["price"]
            trade.shares = result["shares"]
            trade.cost_usdc = result["cost_usdc"]
            trade.order_id = str(result.get("result", {}).get("orderID", ""))
            self._print_log(
                f"ORDER PLACED | {result['shares']:.0f} shares @ "
                f"${result['price']:.3f} = ${result['cost_usdc']:.2f} | "
                f"ID: {trade.order_id[:12]}..."
            )
            asyncio.ensure_future(self._place_take_profit(token_id, SELL_SHARES))
            asyncio.ensure_future(
                self.tg.notify_order_placed(
                    trade, result["shares"], result["cost_usdc"],
                    trade.order_id, market.slug
                )
            )
        else:
            self._print_log(f"ORDER FAILED: {result['error']}")
            asyncio.ensure_future(
                self.tg.notify_order_failed("Confirmed", direction, result["error"])
            )

        self.pending_trades.append(trade)
        self._pending_features = features



    async def _verify_fill_and_tp(self, trade, token_id: str, max_wait: int = 30):
        """Check if order filled, then place TP. Only count P&L if filled."""
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        import math

        # Poll for fill (check balance every 3s for up to max_wait seconds)
        for attempt in range(max_wait // 3):
            await asyncio.sleep(3)
            try:
                params = BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL, token_id=token_id
                )
                b = self.executor.client.get_balance_allowance(params=params)
                raw_balance = int(b.get("balance", "0"))
                if raw_balance > 0:
                    actual_shares = math.floor((raw_balance / 10**6) * 100) / 100.0
                    if actual_shares >= 1.0:  # meaningful fill
                        trade.fill_verified = True
                        trade.shares = actual_shares
                        trade.cost_usdc = round(actual_shares * trade.order_price, 2)
                        self._print_log(
                            f"FILL CONFIRMED | {actual_shares:.2f} shares held"
                        )
                        # Now place take-profit sell
                        sell_shares = min(SELL_SHARES, actual_shares)
                        sell_result = self.executor.place_sell_order(
                            token_id, SELL_PRICE, sell_shares
                        )
                        if sell_result.get("success"):
                            self._print_log(
                                f"AUTO-SELL TP PLACED | {sell_shares:.2f} shares @ ${SELL_PRICE}"
                            )
                        else:
                            self._print_log(
                                f"AUTO-SELL TP FAILED | {sell_result.get('error')}"
                            )
                        return
            except Exception as e:
                pass  # retry

        # If we get here, order did not fill
        trade.fill_verified = False
        trade.is_real = False  # Downgrade to paper trade
        self._print_log(
            f"ORDER NOT FILLED after {max_wait}s | Downgrading to paper trade"
        )
        # Try to cancel the unfilled order
        try:
            if trade.order_id:
                self.executor.client.cancel(trade.order_id)
                self._print_log(f"Cancelled unfilled order {trade.order_id[:12]}...")
        except Exception:
            pass

    def _resolve_window(self):
        if not self.window_candles:
            return
        close_price = self.window_candles[-1].close
        outcome = "UP" if close_price >= self.window_open_price else "DOWN"
        ws = datetime.fromtimestamp(self.window_start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        remaining = []

        for trade in self.pending_trades:
            if trade.window_start == ws:
                trade.close_price = close_price
                trade.outcome = outcome
                trade.resolved = True
                if trade.is_real and trade.fill_verified and trade.order_price > 0:
                    # REAL P&L: only for verified fills
                    # win = shares * $1 - cost, lose = -cost
                    if trade.direction == outcome:
                        trade.pnl = round(trade.shares * 1.0 - trade.cost_usdc, 2)
                    else:
                        trade.pnl = round(-trade.cost_usdc, 2)
                    self._print_log(
                        f"REAL P&L | {trade.direction} {'WIN' if trade.pnl > 0 else 'LOSS'} "
                        f"| ${trade.pnl:+.2f} ({trade.shares:.0f}sh @ {trade.order_price:.2f}c)"
                    )
                else:
                    # Paper/unfilled - track for stats but clearly mark
                    if trade.direction == outcome:
                        trade.pnl = round(BET_SIZE_USDC * (1.0/0.52 - 1.0), 2)
                    else:
                        trade.pnl = round(-BET_SIZE_USDC, 2)
                    if not trade.fill_verified:
                        trade.is_real = False  # ensure unfilled are paper
                self.completed_trades.append(trade)
            else:
                remaining.append(trade)
        self.pending_trades = remaining

        # Log results + notify telegram
        total_pnl = sum(t.pnl for t in self.completed_trades)
        real_pnl = sum(t.pnl for t in self.completed_trades if t.is_real and getattr(t, 'fill_verified', False))
        for ct in self.completed_trades[-2:]:
            if ct.window_start == ws:
                icon = "WIN" if ct.pnl > 0 else "LOSS"
                filled = getattr(ct, 'fill_verified', False)
                fill_tag = " [FILLED]" if filled else " [UNFILLED]"
                self._print_log(
                    f"{icon}{fill_tag} {ct.strategy} {ct.direction} | "
                    f"P&L=${ct.pnl:+.2f} | Total=${total_pnl:+.2f} (Real=${real_pnl:+.2f})"
                )
                asyncio.ensure_future(self.tg.notify_result(ct, total_pnl, real_pnl))

        # Train ML
        if hasattr(self, '_pending_features') and self._pending_features:
            self.ml.add(self._pending_features, outcome)
            if self.ml.should_retrain():
                self.ml.train()
            self._pending_features = None

    # ── WebSocket broadcast for dashboard ──
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

        last_binance = 0.0
        if self.binance_cache:
            last_key = max(self.binance_cache.keys())
            last_binance = self.binance_cache[last_key].get('c', 0)

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
            "mode": "LIVE",
            "trade_capital": BET_SIZE_USDC,
            "total_real_trades": sum(1 for t in self.completed_trades if getattr(t, 'is_real', False)),
        }

    async def _ws_broadcast_loop(self):
        while True:
            if self._ws_clients:
                state = json.dumps(self._build_state_json())
                dead = set()
                for ws in self._ws_clients:
                    try: await ws.send_str(state)
                    except: dead.add(ws)
                self._ws_clients -= dead
            await asyncio.sleep(1.5)

    async def _handle_ws(self, request):
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.add(ws)
        self._print_log(f"Dashboard client connected ({len(self._ws_clients)} total)")
        async for msg in ws:
            pass
        self._ws_clients.discard(ws)
        return ws

    async def _handle_index(self, request):
        html_path = Path(__file__).parent / "dashboard_live.html"
        return aiohttp.web.FileResponse(html_path)

    async def _start_web_server(self, port=8081):
        app = aiohttp.web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/ws", self._handle_ws)
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        for p in range(port, port + 10):
            try:
                site = aiohttp.web.TCPSite(runner, "0.0.0.0", p)
                await site.start()
                self._print_log(f"Dashboard live at http://localhost:{p}")
                return runner
            except OSError:
                self._print_log(f"Port {p} in use, trying next...")
        raise RuntimeError(f"Could not bind to any port in range {port}-{port+9}")


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
        """Poll Chainlink BTC/USD on Polygon every 10s."""
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
                                    self._print_log(f"Chainlink BTC/USD oracle: ${price:,.2f} (tertiary)")
                                    cl_logged = True
            except Exception:
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
                        self._print_log("Coinbase BTC-USD primary stream connected")
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

    async def run(self):
        print()
        print("=" * 64)
        print("  POLYMARKET 5-MIN BTC BOT -- 💰 LIVE MONEY 💰")
        print(f"  Confluence+ML Confirmed | {MIN_SHARES} shares @ max ${MAX_BUY_PRICE} | ${BET_SIZE_USDC}/trade¢")
        print("=" * 64)
        print()

        await self.bootstrap()
        runner = await self._start_web_server(port=8082)

        if self.tg.enabled:
            print(f"  [+] Telegram active (chat: {TG_CHAT})")
            await self.tg.send(
                f"💰 <b>LIVE BOT Online</b>\n"
                f"Mode: <b>REAL MONEY</b>\n"
                f"Trade Size: <b>{MIN_SHARES} shares @ max ${MAX_BUY_PRICE}</b>\n"
                f"Capital: <b>${BET_SIZE_USDC}/trade¢</b>\n"
                f"BTC: <b>${self.btc_price:,.2f}</b>\n"
                f"Wallet: {POLY_WALLET[:10]}...\n"
                f"Dashboard: http://localhost:8082"
            )

        print()
        print("  [*] Dashboard: http://localhost:8082")
        print("  [*] Ctrl+C to stop")
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
    bot = LiveMoneyBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n\n  [*] Bot stopped by user.\n")
        trades = bot.completed_trades
        if trades:
            w = sum(1 for t in trades if t.pnl > 0)
            pnl = sum(t.pnl for t in trades)
            real_count = sum(1 for t in trades if t.is_real)
            print(f"  [TOTAL] {len(trades)} trades ({real_count} real) | "
                  f"{w}/{len(trades)} wins ({w/len(trades)*100:.0f}%) | P&L: ${pnl:+.2f}")
        if bot.tg.enabled:
            try:
                asyncio.run(bot.tg.notify_shutdown(bot.completed_trades))
            except: pass
        print()

