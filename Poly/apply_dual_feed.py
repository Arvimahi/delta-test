import os

BOOTSTRAP_CODE = """
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
"""

STREAM_CODE = """
    async def stream_binance(self):
        url = "wss://stream.binance.com:9443/ws/btcusdc@kline_1m"
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        self._print_log("Binance background volume stream connected")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                k = data.get("k", {})
                                ts = int(k["t"]) // 1000
                                self.binance_cache[ts] = {'v': float(k["v"]), 'n': int(k["n"]), 'V': float(k["V"]), 'c': float(k["c"])}
            except Exception as e:
                await asyncio.sleep(3)

    async def stream_coinbase(self):
        url = "wss://ws-feed.exchange.coinbase.com"
        min_start = 0
        cur_open = 0; cur_high = 0; cur_low = 0; cur_close = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        await ws.send_json({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]})
                        self._print_log("Coinbase primary price stream connected")
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
                                                        volume=bcache['v'], trades=bcache['n'], taker_buy_vol=bcache['V'], binance_price=bcache['c']
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
"""

for target in ["live_dryrun.py", "live_money.py"]:
    with open(target, 'r', encoding='utf-8') as f:
        src = f.read()

    # Add binance_cache to init
    src = src.replace("self.btc_price = 0.0\n", "self.btc_price = 0.0\n        self.binance_cache = {}\n")
    
    # Replace bootstrap
    s1, s2 = src.split("    async def bootstrap(self):")
    _, s3 = s2.split("    def _pretrain_ml(self):")
    src = s1 + BOOTSTRAP_CODE + "\n    def _pretrain_ml(self):" + s3

    # Replace stream_candles
    s1, s2 = src.split("    async def stream_candles(self):")
    _, s3 = s2.split("    async def run(self):")
    src = s1 + STREAM_CODE + "\n    async def run(self):" + s3
    
    # Update gathers
    src = src.replace("self.stream_candles(),", "self.stream_coinbase(),\n            self.stream_binance(),")

    with open(target, 'w', encoding='utf-8') as f:
        f.write(src)

print("Patching complete!")
