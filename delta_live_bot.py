from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from delta_exchange_client import DeltaRestClient
from refine_5m_ict_intraday import RefinedConfig, confluence_score, prepare_entry_frame, prepare_htf_trend


ENTRY_CONFIG = RefinedConfig(
    name="delta_live_setup2_entry",
    trend_tf="1h",
    session_name="broad_day",
    session_windows=((6, 20),),
    stop_loss_pct=0.006,
    rr_multiple=3.0,
    min_score=5,
    max_trades_per_day=2,
    max_hold_bars=36,
)

LEVERAGE = 10.0
STAGED_STOP_PRICE_PCT = 0.025
INITIAL_TARGET_PCT = 0.05
EXTENDED_TARGET_PCT = 0.10
BE_TRIGGER_PCT = 0.05
PROFIT_LOCK_TRIGGER_PCT = 0.065
PROFIT_LOCK_PCT = 0.0275
MAX_HOLD_BARS = 72
IST = ZoneInfo("Asia/Kolkata")
ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_BOLD = "\033[1m"

BASE = Path(__file__).resolve().parent
CONFIG_PATH = BASE / "delta_bot_config.json"


@dataclass
class TradeState:
    active: bool
    side: str | None = None
    entry_price: float | None = None
    entry_time: str | None = None
    stop_price: float | None = None
    target_price: float | None = None
    product_symbol: str | None = None
    product_id: int | None = None
    order_id: int | None = None
    signal_score: int | None = None
    bars_held: int = 0
    moved_be: bool = False
    last_processed_candle: str | None = None


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing {CONFIG_PATH}. Copy delta_bot_config.example.json to delta_bot_config.json and fill in values.")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def state_path(config: dict[str, Any]) -> Path:
    return BASE / config.get("state_path", "delta_bot_state.json")


def log_path(config: dict[str, Any]) -> Path:
    return BASE / config.get("log_path", "delta_bot_log.csv")


def load_state(config: dict[str, Any]) -> TradeState:
    path = state_path(config)
    if not path.exists():
        return TradeState(active=False)
    with path.open("r", encoding="utf-8") as f:
        return TradeState(**json.load(f))


def save_state(config: dict[str, Any], state: TradeState) -> None:
    with state_path(config).open("w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2)


def append_log(config: dict[str, Any], row: dict[str, Any]) -> None:
    path = log_path(config)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def chunked_candles(client: DeltaRestClient, symbol: str, resolution: str, start_s: int, end_s: int) -> pd.DataFrame:
    step_s = {"5m": 300, "1h": 3600}[resolution]
    max_candles = 2000
    window = step_s * max_candles
    rows: list[dict[str, Any]] = []
    cursor = start_s

    while cursor < end_s:
        chunk_end = min(end_s, cursor + window)
        payload = client.get_candles(symbol, resolution, cursor, chunk_end)
        result = payload.get("result", payload)
        candles = result.get("candles") if isinstance(result, dict) else result
        if not candles:
            cursor = chunk_end
            continue
        for item in candles:
            if isinstance(item, dict):
                rows.append(
                    {
                        "timestamp": pd.to_datetime(int(item.get("time", item.get("timestamp"))) , unit="s", utc=True),
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"]),
                        "Volume": float(item.get("volume", item.get("Volume", 0.0))),
                    }
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 6:
                rows.append(
                    {
                        "timestamp": pd.to_datetime(int(item[0]), unit="s", utc=True),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "Volume": float(item[5]),
                    }
                )
        cursor = chunk_end + step_s
        time.sleep(0.1)

    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        return df
    return df.set_index("timestamp")


def fetch_market_frame(client: DeltaRestClient, symbol: str) -> pd.DataFrame:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = int((now - timedelta(days=15)).timestamp())
    end = int(now.timestamp())
    df = chunked_candles(client, symbol, "5m", start, end)
    if df.empty:
        raise RuntimeError("No candle data returned from Delta.")
    return df


def daily_trade_count(config: dict[str, Any], day: datetime.date) -> int:
    path = log_path(config)
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if df.empty or "entry_time" not in df.columns:
        return 0
    if "event" in df.columns:
        df = df[df["event"] == "opened"]
    return int((pd.to_datetime(df["entry_time"], utc=True).dt.date == day).sum())


def realized_stats(config: dict[str, Any]) -> dict[str, float]:
    path = log_path(config)
    if not path.exists():
        return {"closed_trades": 0, "realized_pnl_usd": 0.0, "win_rate_pct": 0.0}
    df = pd.read_csv(path)
    if df.empty or "event" not in df.columns:
        return {"closed_trades": 0, "realized_pnl_usd": 0.0, "win_rate_pct": 0.0}
    df = df[df["event"] == "closed"].copy()
    if df.empty:
        return {"closed_trades": 0, "realized_pnl_usd": 0.0, "win_rate_pct": 0.0}
    df["net_return_pct"] = pd.to_numeric(df["net_return_pct"], errors="coerce").fillna(0.0)
    notional = float(config["order_size"]) * float(config["leverage"])
    df["net_pnl_usd"] = notional * (df["net_return_pct"] / 100.0)
    closed_trades = int(len(df))
    realized_pnl = float(df["net_pnl_usd"].sum())
    win_rate = float((df["net_pnl_usd"] > 0).mean() * 100.0)
    return {
        "closed_trades": closed_trades,
        "realized_pnl_usd": realized_pnl,
        "win_rate_pct": win_rate,
    }


def in_trade_session(ts: pd.Timestamp) -> bool:
    ts_ist = ts.tz_convert(IST)
    return any(start <= ts_ist.hour < end for start, end in ENTRY_CONFIG.session_windows)


def unrealized_pnl_text(state: TradeState, latest_close: float, config: dict[str, Any]) -> str:
    if not state.active or state.entry_price is None or state.side is None:
        return "-"
    gross_return = (latest_close / state.entry_price - 1.0) if state.side == "buy" else (state.entry_price / latest_close - 1.0)
    net_return_pct = (gross_return - (2 * 0.0004)) * 100.0
    pnl = float(config["order_size"]) * float(config["leverage"]) * (net_return_pct / 100.0)
    color = ANSI_GREEN if pnl > 0 else ANSI_RED if pnl < 0 else ANSI_YELLOW
    return f"{color}{pnl:.2f} USD ({net_return_pct:.2f}%){ANSI_RESET}"


def render_dashboard(
    config: dict[str, Any],
    state: TradeState,
    latest_closed_ts: pd.Timestamp,
    latest_close: float,
    today_count: int,
    latest_event: str,
) -> None:
    stats = realized_stats(config)
    latest_closed_ist = latest_closed_ts.tz_convert(IST)
    now_ist = datetime.now(IST)
    realized_color = ANSI_GREEN if stats["realized_pnl_usd"] > 0 else ANSI_RED if stats["realized_pnl_usd"] < 0 else ANSI_YELLOW
    event_color = ANSI_RED if "error" in latest_event.lower() else ANSI_GREEN if "opened" in latest_event.lower() else ANSI_CYAN
    position_color = ANSI_GREEN if state.active else ANSI_YELLOW
    lines = [
        f"{ANSI_BOLD}{ANSI_CYAN}=== Delta Bot Dashboard ==={ANSI_RESET}",
        f"{ANSI_BOLD}Time IST:{ANSI_RESET} {now_ist.isoformat()}",
        f"{ANSI_BOLD}Mode:{ANSI_RESET} {config['mode']} | {ANSI_BOLD}Env:{ANSI_RESET} {config['environment']} | {ANSI_BOLD}Symbol:{ANSI_RESET} {config['product_symbol']}",
        f"{ANSI_BOLD}Latest Closed Candle IST:{ANSI_RESET} {latest_closed_ist.isoformat()} | {ANSI_BOLD}Close:{ANSI_RESET} {latest_close:.2f}",
        f"{ANSI_BOLD}Daily Trades:{ANSI_RESET} {today_count}/{min(config.get('max_trades_per_day', 2), ENTRY_CONFIG.max_trades_per_day)}",
        f"{ANSI_BOLD}Position:{ANSI_RESET} {position_color}{'OPEN' if state.active else 'FLAT'}{ANSI_RESET}",
        f"{ANSI_BOLD}Side:{ANSI_RESET} {state.side or '-'} | {ANSI_BOLD}Entry:{ANSI_RESET} {state.entry_price or '-'} | {ANSI_BOLD}Stop:{ANSI_RESET} {state.stop_price or '-'} | {ANSI_BOLD}Target:{ANSI_RESET} {state.target_price or '-'}",
        f"{ANSI_BOLD}Bars Held:{ANSI_RESET} {state.bars_held} | {ANSI_BOLD}Moved BE:{ANSI_RESET} {state.moved_be} | {ANSI_BOLD}Signal Score:{ANSI_RESET} {state.signal_score or '-'}",
        f"Unrealized: {unrealized_pnl_text(state, latest_close, config)}",
        f"{ANSI_BOLD}Closed Trades:{ANSI_RESET} {stats['closed_trades']} | {ANSI_BOLD}Realized PnL:{ANSI_RESET} {realized_color}{stats['realized_pnl_usd']:.2f} USD{ANSI_RESET} | {ANSI_BOLD}Win Rate:{ANSI_RESET} {stats['win_rate_pct']:.2f}%",
        f"{ANSI_BOLD}Latest Event:{ANSI_RESET} {event_color}{latest_event}{ANSI_RESET}",
        "Press Ctrl+C to stop.",
    ]
    os.system("cls")
    print("\n".join(lines), flush=True)


def signal_on_latest_closed_bar(frame_5m: pd.DataFrame) -> dict[str, Any] | None:
    closed = frame_5m.iloc[:-1].copy()
    if len(closed) < 500:
        return None
    entry_df = prepare_entry_frame(closed)
    trend = prepare_htf_trend(closed, ENTRY_CONFIG.trend_tf)
    latest_ts = entry_df.index[-1]
    row = entry_df.iloc[-1]
    if not in_trade_session(latest_ts):
        return None
    if trend.loc[latest_ts] == "bullish":
        last_bos_idx = row["last_bull_bos_idx"]
        valid = pd.notna(last_bos_idx) and len(entry_df) - 1 - int(last_bos_idx) <= ENTRY_CONFIG.structure_expiry_bars
        score = confluence_score(row, "long", ENTRY_CONFIG)
        if valid and score >= ENTRY_CONFIG.min_score:
            return {"side": "buy", "entry_time": latest_ts.isoformat(), "entry_price": float(row["close"]), "signal_score": score}
    if trend.loc[latest_ts] == "bearish":
        last_bos_idx = row["last_bear_bos_idx"]
        valid = pd.notna(last_bos_idx) and len(entry_df) - 1 - int(last_bos_idx) <= ENTRY_CONFIG.structure_expiry_bars
        score = confluence_score(row, "short", ENTRY_CONFIG)
        if valid and score >= ENTRY_CONFIG.min_score:
            return {"side": "sell", "entry_time": latest_ts.isoformat(), "entry_price": float(row["close"]), "signal_score": score}
    return None


def market_order_payload(product_id: int, size: int, side: str, leverage: int, client_order_id: str) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "size": size,
        "side": side,
        "order_type": "market_order",
        "leverage": str(leverage),
        "time_in_force": "ioc",
        "client_order_id": client_order_id[:32],
    }


def open_trade(config: dict[str, Any], client: DeltaRestClient, product: dict[str, Any], signal: dict[str, Any]) -> TradeState:
    entry_price = float(signal["entry_price"])
    side = signal["side"]
    stop_price = entry_price * (1 - STAGED_STOP_PRICE_PCT) if side == "buy" else entry_price * (1 + STAGED_STOP_PRICE_PCT)
    target_price = entry_price * (1 + INITIAL_TARGET_PCT) if side == "buy" else entry_price * (1 - INITIAL_TARGET_PCT)
    client_order_id = f"ict5m_{int(time.time())}"
    order_id = None

    if config["mode"].upper() == "LIVE":
        response = client.place_order(
            market_order_payload(
                product_id=int(product["id"]),
                size=int(config["order_size"]),
                side=side,
                leverage=int(config["leverage"]),
                client_order_id=client_order_id,
            )
        )
        result = response.get("result", {})
        order_id = result.get("id")

    return TradeState(
        active=True,
        side=side,
        entry_price=entry_price,
        entry_time=signal["entry_time"],
        stop_price=float(stop_price),
        target_price=float(target_price),
        product_symbol=product["symbol"],
        product_id=int(product["id"]),
        order_id=order_id,
        signal_score=int(signal["signal_score"]),
        moved_be=False,
        last_processed_candle=signal["entry_time"],
    )


def close_trade(config: dict[str, Any], client: DeltaRestClient, state: TradeState, exit_price: float, exit_reason: str, exit_time: str) -> TradeState:
    if config["mode"].upper() == "LIVE" and state.product_id is not None:
        close_side = "sell" if state.side == "buy" else "buy"
        client.place_order(
            market_order_payload(
                product_id=state.product_id,
                size=int(config["order_size"]),
                side=close_side,
                leverage=int(config["leverage"]),
                client_order_id=f"ict5m_exit_{int(time.time())}",
            )
        )

    gross_return = ((exit_price / state.entry_price) - 1.0) if state.side == "buy" else ((state.entry_price / exit_price) - 1.0)
    net_return_pct = (gross_return - (2 * 0.0004)) * 100
    row = {
        "event": "closed",
        "entry_time": state.entry_time,
        "exit_time": exit_time,
        "side": state.side,
        "entry_price": state.entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "signal_score": state.signal_score,
        "net_return_pct": net_return_pct,
    }
    append_log(config, row)
    return TradeState(active=False)


def manage_open_trade(config: dict[str, Any], client: DeltaRestClient, state: TradeState, latest_closed: pd.Series, latest_ts: pd.Timestamp) -> TradeState:
    if not state.active or state.entry_price is None:
        return state

    bars_held = state.bars_held + 1
    side = state.side
    exit_price = None
    exit_reason = None

    if side == "buy":
        move_pct = latest_closed["high"] / state.entry_price - 1.0
        if move_pct >= BE_TRIGGER_PCT and not state.moved_be:
            state.stop_price = max(state.stop_price, state.entry_price)
            state.target_price = state.entry_price * (1 + EXTENDED_TARGET_PCT)
            state.moved_be = True
        if move_pct >= PROFIT_LOCK_TRIGGER_PCT:
            state.stop_price = max(state.stop_price, state.entry_price * (1 + PROFIT_LOCK_PCT))
        if latest_closed["low"] <= state.stop_price:
            exit_price, exit_reason = state.stop_price, "stop_loss"
        elif latest_closed["high"] >= state.target_price:
            exit_price, exit_reason = state.target_price, "take_profit"
        elif bars_held >= MAX_HOLD_BARS or not in_trade_session(latest_ts):
            exit_price, exit_reason = float(latest_closed["close"]), "time_exit"
    else:
        move_pct = state.entry_price / latest_closed["low"] - 1.0
        if move_pct >= BE_TRIGGER_PCT and not state.moved_be:
            state.stop_price = min(state.stop_price, state.entry_price)
            state.target_price = state.entry_price * (1 - EXTENDED_TARGET_PCT)
            state.moved_be = True
        if move_pct >= PROFIT_LOCK_TRIGGER_PCT:
            state.stop_price = min(state.stop_price, state.entry_price * (1 - PROFIT_LOCK_PCT))
        if latest_closed["high"] >= state.stop_price:
            exit_price, exit_reason = state.stop_price, "stop_loss"
        elif latest_closed["low"] <= state.target_price:
            exit_price, exit_reason = state.target_price, "take_profit"
        elif bars_held >= MAX_HOLD_BARS or not in_trade_session(latest_ts):
            exit_price, exit_reason = float(latest_closed["close"]), "time_exit"

    if exit_price is not None and exit_reason is not None:
        return close_trade(config, client, state, float(exit_price), exit_reason, latest_ts.isoformat())

    state.bars_held = bars_held
    state.last_processed_candle = latest_ts.isoformat()
    return state


def run_loop() -> None:
    config = load_config()
    client = DeltaRestClient(
        api_key=config["api_key"],
        api_secret=config["api_secret"],
        environment=config.get("environment", "production"),
        user_agent=config.get("user_agent", "python-delta-bot"),
    )
    product = client.get_product(config["product_symbol"])
    if product is None:
        raise RuntimeError(f"Could not resolve product for symbol {config['product_symbol']}")

    state = load_state(config)
    latest_event = "starting"

    while True:
        try:
            frame = fetch_market_frame(client, config["product_symbol"])
            latest_closed_ts = frame.index[-2]
            latest_closed = frame.iloc[-2]
            today_count = daily_trade_count(config, latest_closed_ts.date())

            if state.active and state.last_processed_candle != latest_closed_ts.isoformat():
                state = manage_open_trade(config, client, state, latest_closed, latest_closed_ts)
                save_state(config, state)
                latest_event = f"managed trade active={state.active} bars_held={state.bars_held if state.active else 0}"

            if not state.active:
                if today_count < min(config.get("max_trades_per_day", 2), ENTRY_CONFIG.max_trades_per_day):
                    signal = signal_on_latest_closed_bar(frame)
                    if signal and signal["entry_time"] == latest_closed_ts.isoformat():
                        state = open_trade(config, client, product, signal)
                        save_state(config, state)
                        append_log(
                            config,
                            {
                                "event": "opened",
                                "entry_time": state.entry_time,
                                "exit_time": "",
                                "side": state.side,
                                "entry_price": state.entry_price,
                                "exit_price": "",
                                "exit_reason": "opened",
                                "signal_score": state.signal_score,
                                "net_return_pct": "",
                            },
                        )
                        latest_event = f"opened {state.side} at {state.entry_price} on {state.entry_time}"
                    else:
                        latest_event = f"no entry signal on latest closed bar; today_count={today_count}"
                else:
                    latest_event = f"daily trade cap reached; today_count={today_count}"

            render_dashboard(
                config=config,
                state=state,
                latest_closed_ts=latest_closed_ts,
                latest_close=float(latest_closed["close"]),
                today_count=today_count,
                latest_event=latest_event,
            )

            time.sleep(int(config.get("poll_seconds", 30)))
        except KeyboardInterrupt:
            print("Stopping Delta bot.", flush=True)
            break
        except Exception as exc:
            latest_event = f"loop error: {exc}"
            print(f"Loop error: {exc}", flush=True)
            time.sleep(int(config.get("poll_seconds", 30)))


if __name__ == "__main__":
    run_loop()
