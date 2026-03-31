from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import backtest_eth_smc as smc


BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "BINANCE_ETHUSDT, 5.csv"
STARTING_BALANCE = 1000.0
MARGIN_PER_TRADE = 50.0
LEVERAGE = 5.0


@dataclass(frozen=True)
class RefinedConfig:
    name: str
    trend_tf: str
    session_name: str
    session_windows: tuple[tuple[int, int], ...]
    stop_loss_pct: float
    rr_multiple: float
    min_score: int
    max_trades_per_day: int
    max_hold_bars: int
    sweep_lookback: int = 12
    pd_lookback: int = 48
    structure_expiry_bars: int = 18
    atr_floor: float = 0.0018

    @property
    def take_profit_pct(self) -> float:
        return self.stop_loss_pct * self.rr_multiple


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    return (
        df.resample(timeframe)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )


def prepare_htf_trend(raw_5m: pd.DataFrame, timeframe: str) -> pd.Series:
    rule = {"1h": "1h", "4h": "4h"}[timeframe]
    htf = resample_ohlcv(raw_5m, rule)
    cfg = smc.StrategyConfig(name=f"{timeframe}_trend", min_score=5, structure_expiry_bars=24, structure_lookback=24)
    prepared = smc.prepare_dataset(htf, cfg)
    aligned = pd.merge_asof(
        raw_5m.reset_index()[["timestamp"]],
        prepared.reset_index()[["timestamp", "trend"]],
        on="timestamp",
        direction="backward",
    )
    return pd.Series(aligned["trend"].values, index=raw_5m.index).fillna("neutral")


def prepare_entry_frame(raw_5m: pd.DataFrame) -> pd.DataFrame:
    cfg = smc.StrategyConfig(
        name="5m_refined_base",
        min_score=4,
        structure_expiry_bars=24,
        structure_lookback=24,
        rsi_bull=52.0,
        rsi_bear=48.0,
        entry_rsi_long=50.0,
        entry_rsi_short=50.0,
        atr_floor=0.0015,
    )
    df = smc.prepare_dataset(raw_5m, cfg)
    df["prev_swing_low"] = df["low"].shift(1).rolling(12).min()
    df["prev_swing_high"] = df["high"].shift(1).rolling(12).max()
    df["range_low"] = df["low"].rolling(48).min()
    df["range_high"] = df["high"].rolling(48).max()
    df["equilibrium"] = (df["range_low"] + df["range_high"]) / 2
    df["candle_body"] = (df["close"] - df["open"]).abs()
    df["candle_range"] = (df["high"] - df["low"]).replace(0, pd.NA)
    df["body_ratio"] = (df["candle_body"] / df["candle_range"]).fillna(0.0)
    return df


def in_session(ts: pd.Timestamp, windows: tuple[tuple[int, int], ...]) -> bool:
    hour = ts.hour
    return any(start <= hour < end for start, end in windows)


def liquidity_sweep_long(row: pd.Series) -> bool:
    return pd.notna(row["prev_swing_low"]) and row["low"] < row["prev_swing_low"] and row["close"] > row["prev_swing_low"]


def liquidity_sweep_short(row: pd.Series) -> bool:
    return pd.notna(row["prev_swing_high"]) and row["high"] > row["prev_swing_high"] and row["close"] < row["prev_swing_high"]


def displacement_long(row: pd.Series, atr_floor: float) -> bool:
    return (
        row["close"] > row["open"]
        and row["body_ratio"] >= 0.55
        and pd.notna(row["atr"])
        and (row["high"] - row["low"]) / row["close"] >= atr_floor
    )


def displacement_short(row: pd.Series, atr_floor: float) -> bool:
    return (
        row["close"] < row["open"]
        and row["body_ratio"] >= 0.55
        and pd.notna(row["atr"])
        and (row["high"] - row["low"]) / row["close"] >= atr_floor
    )


def confluence_score(row: pd.Series, direction: str, cfg: RefinedConfig) -> int:
    if direction == "long":
        checks = [
            liquidity_sweep_long(row),
            pd.notna(row["last_bull_bos_idx"]),
            smc.zone_touched(row, "long"),
            row["close"] <= row["equilibrium"] if pd.notna(row["equilibrium"]) else False,
            row["rsi"] >= 50,
            row["macd_line"] > row["macd_signal"],
            displacement_long(row, cfg.atr_floor),
        ]
    else:
        checks = [
            liquidity_sweep_short(row),
            pd.notna(row["last_bear_bos_idx"]),
            smc.zone_touched(row, "short"),
            row["close"] >= row["equilibrium"] if pd.notna(row["equilibrium"]) else False,
            row["rsi"] <= 50,
            row["macd_line"] < row["macd_signal"],
            displacement_short(row, cfg.atr_floor),
        ]
    return int(sum(checks))


def backtest_refined(df: pd.DataFrame, htf_trend: pd.Series, cfg: RefinedConfig) -> pd.DataFrame:
    trades = []
    position = None
    start_index = 200

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        ts = df.index[i]
        trend = htf_trend.loc[ts]

        if position is not None:
            bars_held = i - position["entry_index"]
            direction = position["direction"]
            entry = position["entry_price"]
            exit_price = None
            exit_reason = None

            if direction == "long":
                stop_price = entry * (1 - cfg.stop_loss_pct)
                target_price = entry * (1 + cfg.take_profit_pct)
                if row["low"] <= stop_price and row["high"] >= target_price:
                    exit_price, exit_reason = stop_price, "both_hit_stop_assumed"
                elif row["low"] <= stop_price:
                    exit_price, exit_reason = stop_price, "stop_loss"
                elif row["high"] >= target_price:
                    exit_price, exit_reason = target_price, "take_profit"
                elif bars_held >= cfg.max_hold_bars or not in_session(ts, cfg.session_windows):
                    exit_price, exit_reason = row["close"], "time_exit"
                elif trend == "bearish" and row["bearish_bos"]:
                    exit_price, exit_reason = row["close"], "htf_invalidation"
                if exit_price is not None:
                    gross_return = (exit_price / entry) - 1.0
            else:
                stop_price = entry * (1 + cfg.stop_loss_pct)
                target_price = entry * (1 - cfg.take_profit_pct)
                if row["high"] >= stop_price and row["low"] <= target_price:
                    exit_price, exit_reason = stop_price, "both_hit_stop_assumed"
                elif row["high"] >= stop_price:
                    exit_price, exit_reason = stop_price, "stop_loss"
                elif row["low"] <= target_price:
                    exit_price, exit_reason = target_price, "take_profit"
                elif bars_held >= cfg.max_hold_bars or not in_session(ts, cfg.session_windows):
                    exit_price, exit_reason = row["close"], "time_exit"
                elif trend == "bullish" and row["bullish_bos"]:
                    exit_price, exit_reason = row["close"], "htf_invalidation"
                if exit_price is not None:
                    gross_return = (entry / exit_price) - 1.0

            if exit_price is not None:
                net_return = (gross_return - (2 * 0.0004)) * 100
                trades.append(
                    {
                        "direction": direction,
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "entry_price": entry,
                        "exit_price": float(exit_price),
                        "exit_reason": exit_reason,
                        "signal_score": position["signal_score"],
                        "gross_return_pct": gross_return * 100,
                        "net_return_pct": net_return,
                        "bars_held": bars_held,
                        "setup_name": cfg.name,
                    }
                )
                position = None
                continue

        if position is not None or not in_session(ts, cfg.session_windows):
            continue

        trade_date = ts.date()
        existing_today = sum(pd.Timestamp(t["entry_time"]).date() == trade_date for t in trades)
        if existing_today >= cfg.max_trades_per_day:
            continue

        if trend == "bullish":
            last_bos_idx = row["last_bull_bos_idx"]
            valid_structure = pd.notna(last_bos_idx) and i - int(last_bos_idx) <= cfg.structure_expiry_bars
            score = confluence_score(row, "long", cfg)
            if valid_structure and score >= cfg.min_score:
                position = {
                    "direction": "long",
                    "entry_index": i,
                    "entry_time": ts,
                    "entry_price": float(row["close"]),
                    "signal_score": score,
                }
        elif trend == "bearish":
            last_bos_idx = row["last_bear_bos_idx"]
            valid_structure = pd.notna(last_bos_idx) and i - int(last_bos_idx) <= cfg.structure_expiry_bars
            score = confluence_score(row, "short", cfg)
            if valid_structure and score >= cfg.min_score:
                position = {
                    "direction": "short",
                    "entry_index": i,
                    "entry_time": ts,
                    "entry_price": float(row["close"]),
                    "signal_score": score,
                }

    return pd.DataFrame(trades)


def apply_sizing(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True)
    notional = MARGIN_PER_TRADE * LEVERAGE
    out["net_pnl_usd"] = notional * (out["net_return_pct"] / 100.0)
    out["gross_pnl_usd"] = notional * (out["gross_return_pct"] / 100.0)
    out["fees_usd"] = notional * 2 * 0.0004
    out["equity_usd"] = STARTING_BALANCE + out["net_pnl_usd"].cumsum()
    out["trade_date"] = out["entry_time"].dt.strftime("%Y-%m-%d")
    return out


def summarize(trades: pd.DataFrame, cfg: RefinedConfig) -> dict:
    if trades.empty:
        return {
            "config_name": cfg.name,
            "trend_tf": cfg.trend_tf,
            "session_name": cfg.session_name,
            "rr_multiple": cfg.rr_multiple,
            "trades": 0,
        }
    daily = trades.groupby("trade_date", as_index=False).agg(trades_taken=("trade_date", "size"), net_pnl_usd=("net_pnl_usd", "sum"))
    wins = trades[trades["net_pnl_usd"] > 0]
    losses = trades[trades["net_pnl_usd"] <= 0]
    gross_profit = wins["net_pnl_usd"].sum()
    gross_loss = abs(losses["net_pnl_usd"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else 999.0
    equity_curve = trades["net_pnl_usd"].cumsum()
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve - rolling_peak
    return {
        "config_name": cfg.name,
        "trend_tf": cfg.trend_tf,
        "session_name": cfg.session_name,
        "stop_loss_pct": cfg.stop_loss_pct,
        "rr_multiple": cfg.rr_multiple,
        "take_profit_pct": cfg.take_profit_pct,
        "min_score": cfg.min_score,
        "max_trades_per_day": cfg.max_trades_per_day,
        "max_hold_bars": cfg.max_hold_bars,
        "trades": len(trades),
        "trading_days": int(daily["trade_date"].nunique()),
        "avg_trades_per_day": round(daily["trades_taken"].mean(), 2),
        "win_rate_pct": round((trades["net_pnl_usd"] > 0).mean() * 100, 2),
        "net_pnl_usd": round(trades["net_pnl_usd"].sum(), 2),
        "avg_trade_pnl_usd": round(trades["net_pnl_usd"].mean(), 2),
        "profit_factor": round(float(profit_factor), 2),
        "max_drawdown_usd": round(abs(float(drawdown.min())), 2),
        "ending_balance_usd": round(float(STARTING_BALANCE + trades["net_pnl_usd"].sum()), 2),
        "best_day_usd": round(daily["net_pnl_usd"].max(), 2),
        "worst_day_usd": round(daily["net_pnl_usd"].min(), 2),
    }


def build_daily(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    daily = (
        trades.groupby("trade_date", as_index=False)
        .agg(
            trades_taken=("trade_date", "size"),
            wins=("net_pnl_usd", lambda s: int((s > 0).sum())),
            losses=("net_pnl_usd", lambda s: int((s <= 0).sum())),
            net_pnl_usd=("net_pnl_usd", "sum"),
            avg_trade_pnl_usd=("net_pnl_usd", "mean"),
        )
    )
    daily["cum_net_pnl_usd"] = daily["net_pnl_usd"].cumsum()
    return daily


def candidate_configs() -> list[RefinedConfig]:
    sessions = {
        "london_ny": ((6, 11), (13, 17)),
        "broad_day": ((6, 20),),
        "ny_only": ((13, 18),),
    }
    configs = []
    for trend_tf in ["1h", "4h"]:
        for session_name, windows in sessions.items():
            for stop_loss_pct, rr_multiple in [(0.005, 2.0), (0.005, 3.0), (0.006, 2.5), (0.006, 3.0), (0.0075, 2.0)]:
                for min_score in [5, 6]:
                    for max_trades_per_day in [2, 3]:
                        max_hold = 24 if session_name != "broad_day" else 36
                        name = f"{trend_tf}_{session_name}_sl{stop_loss_pct}_rr{rr_multiple}_s{min_score}_d{max_trades_per_day}"
                        configs.append(
                            RefinedConfig(
                                name=name,
                                trend_tf=trend_tf,
                                session_name=session_name,
                                session_windows=windows,
                                stop_loss_pct=stop_loss_pct,
                                rr_multiple=rr_multiple,
                                min_score=min_score,
                                max_trades_per_day=max_trades_per_day,
                                max_hold_bars=max_hold,
                            )
                        )
    return configs


def main() -> None:
    raw = smc.load_data(DATA_PATH)
    raw = smc.slice_lookback(raw, 180)
    entry_df = prepare_entry_frame(raw)
    trends = {tf: prepare_htf_trend(raw, tf) for tf in ["1h", "4h"]}

    result_rows = []
    best_balanced = None
    best_high_win = None
    best_high_rr = None
    best_balanced_score = -10**9
    best_high_win_score = -10**9
    best_high_rr_score = -10**9

    for cfg in candidate_configs():
        trades = backtest_refined(entry_df, trends[cfg.trend_tf], cfg)
        sized = apply_sizing(trades)
        summary = summarize(sized, cfg)
        result_rows.append(summary)

        if summary["trades"] < 80:
            continue

        balanced_score = summary["net_pnl_usd"] + summary["profit_factor"] * 15 - summary["max_drawdown_usd"] * 1.5
        if balanced_score > best_balanced_score:
            best_balanced_score = balanced_score
            best_balanced = (cfg, sized, summary)

        high_win_score = summary["win_rate_pct"] + summary["net_pnl_usd"] * 0.3 - summary["max_drawdown_usd"] * 0.5
        if high_win_score > best_high_win_score:
            best_high_win_score = high_win_score
            best_high_win = (cfg, sized, summary)

        if cfg.rr_multiple >= 3.0:
            high_rr_score = summary["net_pnl_usd"] + summary["profit_factor"] * 10 - summary["max_drawdown_usd"]
            if high_rr_score > best_high_rr_score:
                best_high_rr_score = high_rr_score
                best_high_rr = (cfg, sized, summary)

    results_df = pd.DataFrame(result_rows).sort_values(["net_pnl_usd", "profit_factor"], ascending=False)
    results_df.to_csv(BASE / "eth_5m_refined_candidates.csv", index=False)

    named_sets = {
        "best_balanced": best_balanced,
        "best_high_win": best_high_win,
        "best_high_rr": best_high_rr,
    }

    summary_rows = []
    for label, payload in named_sets.items():
        if payload is None:
            continue
        cfg, trades, summary = payload
        daily = build_daily(trades)
        trades.to_csv(BASE / f"eth_5m_{label}_trades.csv", index=False)
        daily.to_csv(BASE / f"eth_5m_{label}_daily.csv", index=False)
        output_summary = {**summary, "label": label}
        summary_rows.append(output_summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(BASE / "eth_5m_refined_best_summary.csv", index=False)
    print("BEST_SETUPS")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
