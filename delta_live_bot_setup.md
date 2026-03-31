# Delta Live Bot Setup

1. Copy [delta_bot_config.example.json](C:\Users\Aravind\Trading\Crypto\delta_bot_config.example.json) to `delta_bot_config.json`.
2. Fill in:
   - `api_key`
   - `api_secret`
   - `environment`: `production` or `testnet`
   - `product_symbol`: for example `ETHUSDT`
   - `order_size`: set this manually after checking Delta product specs
3. Start in `DRY_RUN` mode first.
4. Run:

```powershell
python .\delta_live_bot.py
```

Files written by the bot:
- `delta_bot_state.json`
- `delta_bot_log.csv`

Notes:
- The bot uses the refined `5m` entry logic with `1h` bias.
- Default live logic is the best balanced setup from the backtest: `0.6%` stop and `1.8%` target.
- Delta authenticated REST requests require `api-key`, `signature`, `timestamp`, and `User-Agent`.
- Delta docs note signatures expire quickly, so keep your system clock synced.
- Do not switch to `LIVE` until you have verified the dry-run entries and exits.
