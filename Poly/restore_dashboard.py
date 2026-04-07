import os
import re

# Comprehensive method set for the dual-feed engine
HYBRID_METHODS = '''    def _build_state_json(self):
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
        site = aiohttp.web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        self._print_log(f"Dashboard live at http://localhost:{port}")
        return runner
'''

UPDATE_RUN = '''        await asyncio.gather(
            self.stream_coinbase(),
            self.stream_binance(),
            self._ws_broadcast_loop(),
            self.tg.poll_commands(),
        )'''

def patch_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the insertion point before run()
    run_def = "    async def run(self):"
    if run_def in content and "def _build_state_json" not in content:
        content = content.replace(run_def, HYBRID_METHODS + "\\n" + run_def)
    
    # Fix the gather in run()
    new_content = re.sub(r'await asyncio\\.gather\\(.*?\\)', UPDATE_RUN, content, flags=re.DOTALL)

    # Mode fix
    if 'money' in filename:
        new_content = new_content.replace("'DRY RUN'", "'LIVE'")
    else:
        new_content = new_content.replace("'LIVE'", "'DRY RUN'")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Deep patched {filename}")

patch_file('live_dryrun.py')
patch_file('live_money.py')
