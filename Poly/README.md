# Polymarket 5-Min BTC Bot

Adaptive prediction market bot for Polymarket's "Bitcoin Up or Down - 5 Minutes" markets.
**Demo mode by default** — no wallet, no real money, no risk.

---

## What This Bot Does

1. **Streams live BTC/USDT prices** from Binance WebSocket
2. **Fetches active 5-min markets** from Polymarket's Gamma + CLOB APIs
3. **Detects market regime** (Trending / Ranging / Volatile) every 10 seconds
4. **Selects the best strategy** for that regime and generates signals:

| Regime | Strategy | Logic |
|--------|----------|-------|
| VOLATILE | **Oracle Lag Arb** | BTC moved but Poly odds haven't caught up yet (last 30–90s) |
| TRENDING | **Momentum Follow** | 1-min BTC direction with consistency filter |
| RANGING  | **Spread Fade** | Extreme odds (>72%) tend to revert in choppy markets |

5. **Live dashboard** shows BTC price, timer, odds bar, signal, and log in real-time

---

## Prerequisites

- Python 3.10+
- `pip install aiohttp websockets`

---

## Quick Start

### Step 1 — Run the bot
```bash
cd polymarket_btc_bot
pip install aiohttp --break-system-packages   # if needed

# Demo scan (single run, see state output)
python bot_runner.py --once

# Full demo mode (continuous, every 10s)
python bot_runner.py
```

### Step 2 — Open the dashboard
Open `dashboard.html` in your browser. It reads `bot_state.json` every 3 seconds.

> **Tip**: Serve it locally so fetch() works:
> ```bash
> python -m http.server 8080
> # then open http://localhost:8080/dashboard.html
> ```

---

## Claude Desktop MCP Integration

For deeper market research via Claude Desktop chat:

### Install the MCP server
```bash
git clone https://github.com/caiovicentino/polymarket-mcp-server.git
cd polymarket-mcp-server
python -m venv venv && source venv/bin/activate
pip install -e .
```

### Configure Claude Desktop
Copy the `mcpServers` block from `claude_desktop_config.json` into:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Restart Claude Desktop, then use these prompts:

```
"Search for Bitcoin Up or Down markets closing in the next 10 minutes. Show me odds, liquidity, and spread for each."

"Get orderbook depth for the next BTC 5-min market. Where are the liquidity walls?"

"Based on current ATR and trend consistency, what regime are we in and which strategy should I use?"

"Compare all active BTC 5-min markets side-by-side and flag any that look mispriced."
```

---

## Files

```
polymarket_btc_bot/
├── bot_engine.py            # Core: BTC feed, market fetcher, regime detector, signal engine
├── bot_runner.py            # Orchestrator: event loop, state persistence
├── dashboard.html           # Live browser dashboard (reads bot_state.json)
├── claude_desktop_config.json  # MCP config + Claude Desktop prompts
├── README.md                # This file
└── bot_state.json           # Generated at runtime (bot output)
```

---

## Safety (Demo → Live transition)

The signal engine enforces these filters regardless of mode:
- **Min edge**: Signal only fires if model probability beats market by ≥5%
- **Max spread**: Skip market if bid-ask > 5%
- **Min liquidity**: Skip market if < $5,000
- **No entry in last 10s**: Oracle lag only; avoid execution risk at resolution

When you're ready for live trading, you'll need:
1. A Polygon wallet with USDC
2. `PRIVATE_KEY` and `POLYMARKET_FUNDER` in `claude_desktop_config.json`
3. Set `ENABLE_AUTONOMOUS_TRADING: "true"` and `DEMO_MODE: "false"`

**Start with $10–20 max. These are 5-min binary markets. Variance is high.**

---

## Key Facts About These Markets

- Resolution source: **Chainlink BTC/USD oracle** (not Binance spot)
- Chainlink updates every ~10–30s or on 0.5% deviations
- Markets run 24/7, new window every 5 minutes
- Average arbitrage window: ~2.7 seconds (bots dominate)
- Your edge comes from **regime + oracle timing**, not speed

---

*Not financial advice. Demo mode is the right place to start.*
