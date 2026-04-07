import os
import re

# Correct Coinbase WebSocket logic for public feed
NEW_CB_STREAM = """    async def stream_coinbase(self):
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
                await asyncio.sleep(3)"""

def patch_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the stream_coinbase method
    start_match = re.search(r'    async def stream_coinbase\(self\):', content)
    if start_match:
        # Find the next method or end of class
        end_match = re.search(r'    async def run\(self\):', content)
        if end_match:
            new_content = content[:start_match.start()] + NEW_CB_STREAM + "\n\n" + content[end_match.start():]
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Successfully patched {filename}")
        else:
            print(f"Could not find run method in {filename}")
    else:
        print(f"Could not find stream_coinbase method in {filename}")

patch_file('live_dryrun.py')
patch_file('live_money.py')
