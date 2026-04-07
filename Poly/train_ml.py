import asyncio
import aiohttp
import time
import math
import joblib
from datetime import datetime, timezone, timedelta
import numpy as np
from pathlib import Path

# Import our shared structure
from live_dryrun import Candle, build_features, MLStrategy

DAYS = 60

async def fetch_coinbase(session, start_iso, end_iso):
    """Fetch 1m candles (max 300) from Coinbase BTC-USD."""
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    params = {"granularity": 60, "start": start_iso, "end": end_iso}
    for attempt in range(5):
        try:
            async with session.get(url, params=params) as r:
                if r.status == 429:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                data = await r.json()
                return data if isinstance(data, list) else []
        except:
            await asyncio.sleep(0.5)
    return []

async def fetch_binance(session, start_ms, end_ms):
    """Fetch 1m candles from Binance BTCUSDT."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "startTime": start_ms, "endTime": end_ms, "limit": 1000}
    for attempt in range(5):
        try:
            async with session.get(url, params=params) as r:
                data = await r.json()
                return data if isinstance(data, list) else []
        except:
            await asyncio.sleep(0.5)
    return []

async def fetch_dual_data():
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=DAYS)
    
    cb_candles = {} # ts -> [o, h, l, c]
    bn_candles = {} # ts -> [v, trades, taker_vol, binance_close]
    
    print(f"🔄 Fetching {DAYS} days of parallel data (Coinbase + Binance)...")
    
    async with aiohttp.ClientSession() as session:
        # Fetch Coinbase (300 mins at a time)
        curr_cb = start_dt
        total_cb = 0
        while curr_cb < now:
            c_start = curr_cb
            c_end = min(c_start + timedelta(minutes=300), now)
            data = await fetch_coinbase(session, c_start.isoformat(), c_end.isoformat())
            for row in data:
                ts = float(row[0])
                cb_candles[ts] = [float(row[3]), float(row[2]), float(row[1]), float(row[4])] # open, high, low, close
            total_cb += len(data)
            print(f"  [Coinbase] Fetched {total_cb} candles...", end="\r")
            curr_cb = c_end
            
        print(f"\n  [Coinbase] Completed. {len(cb_candles)} unique candles.")
        
        # Fetch Binance
        curr_bn = int(start_dt.timestamp() * 1000)
        now_ms = int(now.timestamp() * 1000)
        total_bn = 0
        while curr_bn < now_ms:
            data = await fetch_binance(session, curr_bn, min(curr_bn + 1000 * 60000, now_ms))
            if not data: break
            for row in data:
                ts = float(row[0]) / 1000.0
                # v, trades, taker, binance_close
                bn_candles[ts] = [float(row[5]), int(row[8]), float(row[9]), float(row[4])]
            total_bn += len(data)
            print(f"  [Binance] Fetched {total_bn} candles...", end="\r")
            curr_bn = data[-1][0] + 60000
            
        print(f"\n  [Binance] Completed. {len(bn_candles)} unique candles.")
        
    print("🧹 Stitching streams into hybrid candles...")
    
    merged = []
    # Intersection of both feeds
    keys = sorted(list(cb_candles.keys()))
    for ts in keys:
        if ts in bn_candles:
            c = cb_candles[ts]
            b = bn_candles[ts]
            # Construct Super-Candle
            candle = Candle(
                ts_open=ts, ts_close=ts+59.9,
                open=c[0], high=c[1], low=c[2], close=c[3], # Coinbase Core
                volume=b[0], trades=b[1], taker_buy_vol=b[2], # Binance Volumes
                binance_price=b[3] # Binance secondary proxy
            )
            merged.append(candle)
            
    print(f"✅ Generated {len(merged)} hybrid candles perfectly aligned.")
    return merged

def run_training_prep(all_candles: list[Candle]):
    ml = MLStrategy()
    ml.rf.class_weight = "balanced"
    
    candle_history = []
    window_start_ts = 0
    window_candles = []
    window_open_price = 0
    c3_features = None
    
    print("⏱️ Extracting ML target features from historical 5-minute windows...")
    
    for i, c in enumerate(all_candles):
        candle_history.append(c)
        if len(candle_history) > 200:
            candle_history.pop(0)

        ms = int(c.ts_open * 1000)
        remainder = ms % 300000
        ws = (ms - remainder) / 1000.0
        
        if ws != window_start_ts:
            if c3_features is not None and window_start_ts > 0:
                last_c = candle_history[-2] if len(candle_history) > 1 else c
                # Predict Coinbase direction!
                outcome = "UP" if last_c.close >= window_open_price else "DOWN"
                
                ml.X.append(ml._vec(c3_features))
                ml.y.append(1 if outcome == "UP" else 0)

            window_start_ts = ws
            window_candles = [c]
            window_open_price = c.open # Coinbase open
            c3_features = None
        else:
            window_candles.append(c)
            
        if remainder >= 120000 and c3_features is None:
            if len(candle_history) >= 130 and len(window_candles) >= 1:
                past_history = candle_history[:-len(window_candles)]
                c3_features = build_features(past_history[-120:], window_candles)

    return ml

async def main():
    print("=" * 60)
    print(" 🧠 TRIPLE-FEED HYBRID ML OFFLINE HUB ")
    print("=" * 60)
    
    candles = await fetch_dual_data()
    if len(candles) < 300:
        print("❌ Not enough data.")
        return
        
    ml_instance = run_training_prep(candles)
    
    print(f"📊 Gathered {len(ml_instance.X):,} viable training samples.")
    ups = sum(ml_instance.y)
    downs = len(ml_instance.y) - ups
    print(f"⚖️ Targets: UP ({ups}) vs DOWN ({downs}) | Win Rate Baseline: {ups/len(ml_instance.y)*100:.1f}%")
    
    print("🚀 Running massive offline model training...")
    start_train = time.time()
    
    X = np.nan_to_num(np.array(ml_instance.X), nan=0, posinf=0, neginf=0)
    y = np.array(ml_instance.y)
    ml_instance.scaler.fit(X); Xs = ml_instance.scaler.transform(X)
    ml_instance.gb.fit(Xs, y)
    ml_instance.rf.fit(Xs, y)
    ml_instance.ada.fit(Xs, y)
    
    ml_instance.is_trained = True
    print(f"✅ Training completed in {time.time()-start_train:.2f} seconds.")
    
    path = Path(__file__).parent / 'ml_model.joblib'
    joblib.dump(ml_instance, path)
    print("💾 Triple-Feed Coinbase-Driven Model saved to 'ml_model.joblib'.")

if __name__ == "__main__":
    asyncio.run(main())
