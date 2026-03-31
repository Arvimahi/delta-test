from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Direction = Literal["long", "short"]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    take_profit_pct: float = 0.05
    stop_loss_pct: float = 0.025
    fee_pct: float = 0.0004
    min_score: int = 5
    structure_expiry_bars: int = 48
    fast_ema: int = 20
    mid_ema: int = 50
    slow_ema: int = 200
    rsi_length: int = 14
    rsi_bull: float = 55.0
    rsi_bear: float = 45.0
    entry_rsi_long: float = 52.0
    entry_rsi_short: float = 48.0
    atr_floor: float = 0.003
    structure_lookback: int = 36
    swing_left: int = 2
    swing_right: int = 2


@dataclass
class Trade:
    direction: Direction
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    exit_reason: str
    signal_score: int
    gross_return_pct: float
    net_return_pct: float
    bars_held: int
    setup_name: str


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, 12) - ema(series, 26)
    signal_line = ema(macd_line, 9)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    df[["open", "high", "low", "close", "Volume"]] = df[
        ["open", "high", "low", "close", "Volume"]
    ].astype(float)
    return df


def slice_lookback(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    available_start = df.index.min()
    desired_start = df.index.max() - pd.Timedelta(days=lookback_days)
    return df[df.index >= max(available_start, desired_start)].copy()


def find_swings(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    highs = df["high"]
    lows = df["low"]
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)

    for i in range(left, len(df) - right):
        window_highs = highs.iloc[i - left : i + right + 1]
        window_lows = lows.iloc[i - left : i + right + 1]
        if highs.iloc[i] == window_highs.max() and (window_highs == highs.iloc[i]).sum() == 1:
            swing_high.iloc[i] = True
        if lows.iloc[i] == window_lows.min() and (window_lows == lows.iloc[i]).sum() == 1:
            swing_low.iloc[i] = True

    out = df.copy()
    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    return out


def add_indicators(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], config.fast_ema)
    out["ema_mid"] = ema(out["close"], config.mid_ema)
    out["ema_slow"] = ema(out["close"], config.slow_ema)
    out["rsi"] = rsi(out["close"], config.rsi_length)
    macd_line, macd_signal, macd_hist = macd(out["close"])
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    out["volume_sma"] = out["Volume"].rolling(20).mean()
    out["atr"] = (
        pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - out["close"].shift()).abs(),
                (out["low"] - out["close"].shift()).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(14)
        .mean()
    )
    return find_swings(out, config.swing_left, config.swing_right)


def add_smc_features(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = df.copy()
    last_swing_high = np.nan
    last_swing_low = np.nan
    bullish_bos = np.zeros(len(out), dtype=bool)
    bearish_bos = np.zeros(len(out), dtype=bool)
    bull_fvg_low = np.full(len(out), np.nan)
    bull_fvg_high = np.full(len(out), np.nan)
    bear_fvg_low = np.full(len(out), np.nan)
    bear_fvg_high = np.full(len(out), np.nan)
    bull_ob_low = np.full(len(out), np.nan)
    bull_ob_high = np.full(len(out), np.nan)
    bear_ob_low = np.full(len(out), np.nan)
    bear_ob_high = np.full(len(out), np.nan)
    bull_bos_index = np.full(len(out), -1, dtype=int)
    bear_bos_index = np.full(len(out), -1, dtype=int)

    for i in range(len(out)):
        if out["swing_high"].iloc[i]:
            last_swing_high = out["high"].iloc[i]
        if out["swing_low"].iloc[i]:
            last_swing_low = out["low"].iloc[i]

        if not np.isnan(last_swing_high) and out["close"].iloc[i] > last_swing_high:
            bullish_bos[i] = True
            bull_bos_index[i] = i
        elif i > 0:
            bull_bos_index[i] = bull_bos_index[i - 1]

        if not np.isnan(last_swing_low) and out["close"].iloc[i] < last_swing_low:
            bearish_bos[i] = True
            bear_bos_index[i] = i
        elif i > 0:
            bear_bos_index[i] = bear_bos_index[i - 1]

        if 0 < i < len(out) - 1:
            prev_high = out["high"].iloc[i - 1]
            prev_low = out["low"].iloc[i - 1]
            next_high = out["high"].iloc[i + 1]
            next_low = out["low"].iloc[i + 1]
            if next_low > prev_high:
                bull_fvg_low[i] = prev_high
                bull_fvg_high[i] = next_low
            if next_high < prev_low:
                bear_fvg_low[i] = next_high
                bear_fvg_high[i] = prev_low

        if bullish_bos[i]:
            start = max(0, i - config.structure_lookback)
            bearish_candles = out.iloc[start:i]
            bearish_candles = bearish_candles[bearish_candles["close"] < bearish_candles["open"]]
            if not bearish_candles.empty:
                candle = bearish_candles.iloc[-1]
                bull_ob_low[i] = candle["low"]
                bull_ob_high[i] = candle["open"]

        if bearish_bos[i]:
            start = max(0, i - config.structure_lookback)
            bullish_candles = out.iloc[start:i]
            bullish_candles = bullish_candles[bullish_candles["close"] > bullish_candles["open"]]
            if not bullish_candles.empty:
                candle = bullish_candles.iloc[-1]
                bear_ob_low[i] = candle["open"]
                bear_ob_high[i] = candle["high"]

    out["bullish_bos"] = bullish_bos
    out["bearish_bos"] = bearish_bos
    out["last_bull_bos_idx"] = pd.Series(bull_bos_index, index=out.index).replace(-1, np.nan).ffill()
    out["last_bear_bos_idx"] = pd.Series(bear_bos_index, index=out.index).replace(-1, np.nan).ffill()
    out["bull_fvg_low"] = pd.Series(bull_fvg_low, index=out.index).ffill()
    out["bull_fvg_high"] = pd.Series(bull_fvg_high, index=out.index).ffill()
    out["bear_fvg_low"] = pd.Series(bear_fvg_low, index=out.index).ffill()
    out["bear_fvg_high"] = pd.Series(bear_fvg_high, index=out.index).ffill()
    out["bull_ob_low"] = pd.Series(bull_ob_low, index=out.index).ffill()
    out["bull_ob_high"] = pd.Series(bull_ob_high, index=out.index).ffill()
    out["bear_ob_low"] = pd.Series(bear_ob_low, index=out.index).ffill()
    out["bear_ob_high"] = pd.Series(bear_ob_high, index=out.index).ffill()
    return out


def trend_label(row: pd.Series, config: StrategyConfig) -> str:
    bullish = (
        row["close"] > row["ema_slow"]
        and row["ema_fast"] > row["ema_mid"] > row["ema_slow"]
        and row["rsi"] >= config.rsi_bull
        and row["macd_hist"] > 0
    )
    bearish = (
        row["close"] < row["ema_slow"]
        and row["ema_fast"] < row["ema_mid"] < row["ema_slow"]
        and row["rsi"] <= config.rsi_bear
        and row["macd_hist"] < 0
    )
    if bullish:
        return "bullish"
    if bearish:
        return "bearish"
    return "neutral"


def zone_touched(row: pd.Series, direction: Direction) -> bool:
    if direction == "long":
        in_fvg = pd.notna(row["bull_fvg_low"]) and row["low"] <= row["bull_fvg_high"] and row["close"] >= row["bull_fvg_low"]
        in_ob = pd.notna(row["bull_ob_low"]) and row["low"] <= row["bull_ob_high"] and row["close"] >= row["bull_ob_low"]
        return in_fvg or in_ob
    in_fvg = pd.notna(row["bear_fvg_high"]) and row["high"] >= row["bear_fvg_low"] and row["close"] <= row["bear_fvg_high"]
    in_ob = pd.notna(row["bear_ob_high"]) and row["high"] >= row["bear_ob_low"] and row["close"] <= row["bear_ob_high"]
    return in_fvg or in_ob


def signal_score(row: pd.Series, direction: Direction, config: StrategyConfig) -> int:
    vol_ok = row["Volume"] >= row["volume_sma"] if pd.notna(row["volume_sma"]) else False
    atr_ok = row["atr"] / row["close"] >= config.atr_floor if pd.notna(row["atr"]) else False
    if direction == "long":
        checks = [
            row["close"] > row["ema_fast"],
            row["rsi"] >= config.entry_rsi_long,
            row["macd_line"] > row["macd_signal"],
            vol_ok,
            row["close"] >= row["open"],
            atr_ok,
        ]
    else:
        checks = [
            row["close"] < row["ema_fast"],
            row["rsi"] <= config.entry_rsi_short,
            row["macd_line"] < row["macd_signal"],
            vol_ok,
            row["close"] <= row["open"],
            atr_ok,
        ]
    return int(sum(checks))


def prepare_dataset(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    out = add_indicators(df, config)
    out = add_smc_features(out, config)
    out["trend"] = out.apply(lambda row: trend_label(row, config), axis=1)
    return out


def backtest(
    df: pd.DataFrame,
    config: StrategyConfig,
    trend_override: pd.Series | None = None,
) -> tuple[list[Trade], pd.DataFrame]:
    trades: list[Trade] = []
    position: dict | None = None
    start_index = max(config.slow_ema, 200)

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        timestamp = df.index[i]
        active_trend = trend_override.loc[timestamp] if trend_override is not None else row["trend"]

        if position is not None:
            direction: Direction = position["direction"]
            entry_price = position["entry_price"]
            bars_held = i - position["entry_index"]

            if direction == "long":
                stop_price = entry_price * (1 - config.stop_loss_pct)
                target_price = entry_price * (1 + config.take_profit_pct)
                hit_stop = row["low"] <= stop_price
                hit_target = row["high"] >= target_price
                structure_break = row["trend"] == "bearish" and row["bearish_bos"]
                exit_price = None
                exit_reason = None
                if hit_stop and hit_target:
                    exit_price, exit_reason = stop_price, "both_hit_stop_assumed"
                elif hit_stop:
                    exit_price, exit_reason = stop_price, "stop_loss"
                elif hit_target:
                    exit_price, exit_reason = target_price, "take_profit"
                elif structure_break:
                    exit_price, exit_reason = row["close"], "bearish_structure_shift"
            else:
                stop_price = entry_price * (1 + config.stop_loss_pct)
                target_price = entry_price * (1 - config.take_profit_pct)
                hit_stop = row["high"] >= stop_price
                hit_target = row["low"] <= target_price
                structure_break = row["trend"] == "bullish" and row["bullish_bos"]
                exit_price = None
                exit_reason = None
                if hit_stop and hit_target:
                    exit_price, exit_reason = stop_price, "both_hit_stop_assumed"
                elif hit_stop:
                    exit_price, exit_reason = stop_price, "stop_loss"
                elif hit_target:
                    exit_price, exit_reason = target_price, "take_profit"
                elif structure_break:
                    exit_price, exit_reason = row["close"], "bullish_structure_shift"

            if exit_price is not None and exit_reason is not None:
                gross_return = (
                    (exit_price / entry_price - 1.0)
                    if direction == "long"
                    else (entry_price / exit_price - 1.0)
                )
                net_return = gross_return - (2 * config.fee_pct)
                trades.append(
                    Trade(
                        direction=direction,
                        entry_time=str(position["entry_time"]),
                        exit_time=str(timestamp),
                        entry_price=entry_price,
                        exit_price=float(exit_price),
                        exit_reason=exit_reason,
                        signal_score=position["signal_score"],
                        gross_return_pct=gross_return * 100,
                        net_return_pct=net_return * 100,
                        bars_held=bars_held,
                        setup_name=config.name,
                    )
                )
                position = None
                continue

        if position is not None:
            continue

        if active_trend == "bullish":
            last_bos_idx = row["last_bull_bos_idx"]
            valid_structure = pd.notna(last_bos_idx) and i - int(last_bos_idx) <= config.structure_expiry_bars
            if valid_structure and zone_touched(row, "long"):
                score = signal_score(row, "long", config)
                if score >= config.min_score:
                    position = {
                        "direction": "long",
                        "entry_index": i,
                        "entry_time": timestamp,
                        "entry_price": float(row["close"]),
                        "signal_score": score,
                    }
        elif active_trend == "bearish":
            last_bos_idx = row["last_bear_bos_idx"]
            valid_structure = pd.notna(last_bos_idx) and i - int(last_bos_idx) <= config.structure_expiry_bars
            if valid_structure and zone_touched(row, "short"):
                score = signal_score(row, "short", config)
                if score >= config.min_score:
                    position = {
                        "direction": "short",
                        "entry_index": i,
                        "entry_time": timestamp,
                        "entry_price": float(row["close"]),
                        "signal_score": score,
                    }

    trade_df = pd.DataFrame([asdict(t) for t in trades])
    return trades, trade_df


def summarize(trade_df: pd.DataFrame, data_start: pd.Timestamp, data_end: pd.Timestamp) -> dict:
    if trade_df.empty:
        return {
            "data_start": str(data_start),
            "data_end": str(data_end),
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "win_rate_pct": 0.0,
            "net_return_pct_sum": 0.0,
            "avg_net_return_pct": 0.0,
            "median_net_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_bars_held": 0.0,
            "take_profit_exits": 0,
            "stop_loss_exits": 0,
            "structure_shift_exits": 0,
        }

    wins = trade_df[trade_df["net_return_pct"] > 0]
    losses = trade_df[trade_df["net_return_pct"] <= 0]
    equity_curve = trade_df["net_return_pct"].cumsum()
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve - rolling_peak
    gross_profit = wins["net_return_pct"].sum()
    gross_loss = abs(losses["net_return_pct"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else np.inf

    return {
        "data_start": str(data_start),
        "data_end": str(data_end),
        "trades": int(len(trade_df)),
        "long_trades": int((trade_df["direction"] == "long").sum()),
        "short_trades": int((trade_df["direction"] == "short").sum()),
        "win_rate_pct": round(len(wins) / len(trade_df) * 100, 2),
        "net_return_pct_sum": round(trade_df["net_return_pct"].sum(), 2),
        "avg_net_return_pct": round(trade_df["net_return_pct"].mean(), 2),
        "median_net_return_pct": round(trade_df["net_return_pct"].median(), 2),
        "profit_factor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else "inf",
        "max_drawdown_pct": round(abs(drawdown.min()), 2),
        "avg_bars_held": round(trade_df["bars_held"].mean(), 2),
        "take_profit_exits": int((trade_df["exit_reason"] == "take_profit").sum()),
        "stop_loss_exits": int((trade_df["exit_reason"] == "stop_loss").sum()),
        "structure_shift_exits": int(trade_df["exit_reason"].str.contains("structure_shift").sum()),
    }


def apply_fixed_sizing(
    trade_df: pd.DataFrame,
    margin_per_trade: float = 50.0,
    leverage: float = 5.0,
    starting_balance: float = 1000.0,
) -> pd.DataFrame:
    out = trade_df.copy()
    if out.empty:
        return out
    notional = margin_per_trade * leverage
    out["margin_used_usd"] = margin_per_trade
    out["notional_usd"] = notional
    out["gross_pnl_usd"] = notional * (out["gross_return_pct"] / 100.0)
    out["net_pnl_usd"] = notional * (out["net_return_pct"] / 100.0)
    out["fees_usd"] = notional * 2 * 0.0004
    out["equity_usd"] = starting_balance + out["net_pnl_usd"].cumsum()
    return out


def apply_compounding(
    trade_df: pd.DataFrame,
    risk_fraction: float = 0.05,
    leverage: float = 5.0,
    starting_balance: float = 1000.0,
) -> pd.DataFrame:
    out = trade_df.copy()
    if out.empty:
        return out

    balances = []
    margins = []
    notionals = []
    gross_pnls = []
    net_pnls = []
    fees = []
    balance = starting_balance

    for _, row in out.iterrows():
        margin = balance * risk_fraction
        notional = margin * leverage
        gross_pnl = notional * (row["gross_return_pct"] / 100.0)
        net_pnl = notional * (row["net_return_pct"] / 100.0)
        fee_paid = notional * 2 * 0.0004
        balance += net_pnl
        margins.append(round(margin, 2))
        notionals.append(round(notional, 2))
        gross_pnls.append(round(gross_pnl, 2))
        net_pnls.append(round(net_pnl, 2))
        fees.append(round(fee_paid, 2))
        balances.append(round(balance, 2))

    out["margin_used_usd"] = margins
    out["notional_usd"] = notionals
    out["gross_pnl_usd"] = gross_pnls
    out["net_pnl_usd"] = net_pnls
    out["fees_usd"] = fees
    out["equity_usd"] = balances
    return out


def summarize_sized(trade_df: pd.DataFrame, starting_balance: float) -> dict:
    if trade_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl_usd": 0.0,
            "avg_net_pnl_usd": 0.0,
            "best_trade_usd": 0.0,
            "worst_trade_usd": 0.0,
            "ending_balance_usd": starting_balance,
            "max_equity_drawdown_usd": 0.0,
            "max_equity_drawdown_pct": 0.0,
        }
    equity = pd.Series([starting_balance] + trade_df["equity_usd"].tolist())
    running_peak = equity.cummax()
    drawdown_usd = equity - running_peak
    drawdown_pct = np.where(running_peak > 0, drawdown_usd / running_peak * 100, 0.0)
    return {
        "trades": int(len(trade_df)),
        "wins": int((trade_df["net_pnl_usd"] > 0).sum()),
        "losses": int((trade_df["net_pnl_usd"] <= 0).sum()),
        "net_pnl_usd": round(trade_df["net_pnl_usd"].sum(), 2),
        "avg_net_pnl_usd": round(trade_df["net_pnl_usd"].mean(), 2),
        "best_trade_usd": round(trade_df["net_pnl_usd"].max(), 2),
        "worst_trade_usd": round(trade_df["net_pnl_usd"].min(), 2),
        "ending_balance_usd": round(float(equity.iloc[-1]), 2),
        "max_equity_drawdown_usd": round(abs(float(drawdown_usd.min())), 2),
        "max_equity_drawdown_pct": round(abs(float(drawdown_pct.min())), 2),
    }


def optimize_strategy(df: pd.DataFrame, timeframe_label: str) -> tuple[StrategyConfig, pd.DataFrame]:
    param_grid = {
        "min_score": [4, 5, 6],
        "structure_expiry_bars": [24, 48],
        "rsi_bull": [52.0, 55.0],
        "rsi_bear": [48.0, 45.0],
        "entry_rsi_long": [50.0],
        "entry_rsi_short": [50.0],
        "atr_floor": [0.0025, 0.0030],
        "structure_lookback": [24, 36],
    }
    keys = list(param_grid)
    results = []
    best_score = -np.inf
    best_config = None

    for values in itertools.product(*(param_grid[key] for key in keys)):
        overrides = dict(zip(keys, values))
        config = StrategyConfig(name=f"{timeframe_label}_opt", **overrides)
        prepared = prepare_dataset(df, config)
        _, trades = backtest(prepared, config)
        summary = summarize(trades, prepared.index.min(), prepared.index.max())
        if summary["trades"] < 20:
            continue
        profit_factor = summary["profit_factor"] if isinstance(summary["profit_factor"], float) else 3.0
        objective = summary["net_return_pct_sum"] + profit_factor * 10 - summary["max_drawdown_pct"] * 1.5
        results.append({**overrides, **summary, "objective": round(objective, 2)})
        if objective > best_score:
            best_score = objective
            best_config = config

    if best_config is None:
        return StrategyConfig(name=f"{timeframe_label}_opt"), pd.DataFrame()
    results_df = pd.DataFrame(results).sort_values(["objective", "net_return_pct_sum"], ascending=False)
    return best_config, results_df


def align_higher_timeframe_trend(entry_df: pd.DataFrame, higher_df: pd.DataFrame) -> pd.Series:
    trend = higher_df["trend"].rename("higher_trend")
    aligned = pd.merge_asof(
        entry_df.reset_index()[["timestamp"]],
        trend.reset_index(),
        on="timestamp",
        direction="backward",
    )
    return pd.Series(aligned["higher_trend"].values, index=entry_df.index).fillna("neutral")


def run_single_timeframe(
    csv_path: Path,
    lookback_days: int,
    config: StrategyConfig,
    label: str,
    starting_balance: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = load_data(csv_path)
    sliced = slice_lookback(raw, lookback_days)
    prepared = prepare_dataset(sliced, config)
    _, trades = backtest(prepared, config)
    fixed = apply_fixed_sizing(trades, starting_balance=starting_balance)
    summary = {
        **summarize(trades, prepared.index.min(), prepared.index.max()),
        **summarize_sized(fixed, starting_balance),
        "timeframe": label,
        "config_name": config.name,
    }
    return prepared, fixed, summary


def run_multi_timeframe(
    entry_csv: Path,
    trend_csv: Path,
    lookback_days: int,
    config: StrategyConfig,
    label: str,
    starting_balance: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    entry_raw = slice_lookback(load_data(entry_csv), lookback_days)
    trend_raw = slice_lookback(load_data(trend_csv), lookback_days)
    entry_prepared = prepare_dataset(entry_raw, config)
    trend_prepared = prepare_dataset(trend_raw, config)
    trend_override = align_higher_timeframe_trend(entry_prepared, trend_prepared)
    _, trades = backtest(entry_prepared, config, trend_override=trend_override)
    fixed = apply_fixed_sizing(trades, starting_balance=starting_balance)
    summary = {
        **summarize(trades, entry_prepared.index.min(), entry_prepared.index.max()),
        **summarize_sized(fixed, starting_balance),
        "timeframe": label,
        "config_name": config.name,
    }
    return entry_prepared, fixed, summary


def save_outputs(
    base_dir: Path,
    prefix: str,
    trade_df: pd.DataFrame,
    summary: dict,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    trade_df.to_csv(base_dir / f"{prefix}_trades.csv", index=False)
    pd.DataFrame([summary]).to_csv(base_dir / f"{prefix}_summary.csv", index=False)
    if extra_tables:
        for name, table in extra_tables.items():
            table.to_csv(base_dir / f"{prefix}_{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest and optimize ETH EMA/RSI/MACD + SMC-inspired strategies.")
    parser.add_argument("--mode", choices=["single", "optimize", "compound", "mtf", "all"], default="all")
    parser.add_argument("--csv", default="BINANCE_ETHUSDT, 5.csv")
    parser.add_argument("--entry-csv", default="BINANCE_ETHUSDT, 5.csv")
    parser.add_argument("--trend-csv", default="BINANCE_ETHUSDT, 60.csv")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--starting-balance", type=float, default=1000.0)
    parser.add_argument("--compound-risk-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path.cwd()
    default_configs = {
        "5m": StrategyConfig(name="5m_default", min_score=5, structure_expiry_bars=48, structure_lookback=36),
        "15m": StrategyConfig(name="15m_default", min_score=5, structure_expiry_bars=36, structure_lookback=36),
        "1h": StrategyConfig(name="1h_default", min_score=5, structure_expiry_bars=24, structure_lookback=24),
    }

    if args.mode == "single":
        _, fixed, summary = run_single_timeframe(Path(args.csv), args.lookback_days, default_configs["5m"], "single", args.starting_balance)
        save_outputs(base_dir, "eth_smc_single", fixed, summary)
        print(pd.DataFrame([summary]).to_string(index=False))
        return

    if args.mode == "optimize":
        label = Path(args.csv).stem.split(",")[-1].strip()
        raw = slice_lookback(load_data(Path(args.csv)), args.lookback_days)
        best_config, grid = optimize_strategy(raw, label)
        prepared = prepare_dataset(raw, best_config)
        _, trades = backtest(prepared, best_config)
        fixed = apply_fixed_sizing(trades, starting_balance=args.starting_balance)
        summary = {**summarize(trades, prepared.index.min(), prepared.index.max()), **summarize_sized(fixed, args.starting_balance)}
        save_outputs(base_dir, f"eth_smc_opt_{label}", fixed, summary, {"grid": grid})
        print("BEST_CONFIG")
        print(best_config)
        print(pd.DataFrame([summary]).to_string(index=False))
        return

    if args.mode == "compound":
        raw = slice_lookback(load_data(Path(args.csv)), args.lookback_days)
        prepared = prepare_dataset(raw, default_configs["5m"])
        _, trades = backtest(prepared, default_configs["5m"])
        compounded = apply_compounding(trades, risk_fraction=args.compound_risk_fraction, starting_balance=args.starting_balance)
        summary = {**summarize(trades, prepared.index.min(), prepared.index.max()), **summarize_sized(compounded, args.starting_balance)}
        save_outputs(base_dir, "eth_smc_compound", compounded, summary)
        print(pd.DataFrame([summary]).to_string(index=False))
        return

    if args.mode == "mtf":
        config = StrategyConfig(name="mtf_default", min_score=5, structure_expiry_bars=48, structure_lookback=36)
        _, fixed, summary = run_multi_timeframe(Path(args.entry_csv), Path(args.trend_csv), args.lookback_days, config, "mtf", args.starting_balance)
        save_outputs(base_dir, "eth_smc_mtf", fixed, summary)
        print(pd.DataFrame([summary]).to_string(index=False))
        return

    timeframe_files = {
        "5m": base_dir / "BINANCE_ETHUSDT, 5.csv",
        "15m": base_dir / "BINANCE_ETHUSDT, 15.csv",
        "1h": base_dir / "BINANCE_ETHUSDT, 60.csv",
    }
    optimization_rows = []
    compound_rows = []
    mtf_rows = []

    for label, csv_path in timeframe_files.items():
        raw = slice_lookback(load_data(csv_path), args.lookback_days)
        best_config, grid = optimize_strategy(raw, label)
        prepared = prepare_dataset(raw, best_config)
        _, trades = backtest(prepared, best_config)

        fixed = apply_fixed_sizing(trades, starting_balance=args.starting_balance)
        fixed_summary = {
            **summarize(trades, prepared.index.min(), prepared.index.max()),
            **summarize_sized(fixed, args.starting_balance),
            "timeframe": label,
            "config_name": best_config.name,
            "min_score": best_config.min_score,
            "structure_expiry_bars": best_config.structure_expiry_bars,
            "rsi_bull": best_config.rsi_bull,
            "rsi_bear": best_config.rsi_bear,
            "entry_rsi_long": best_config.entry_rsi_long,
            "entry_rsi_short": best_config.entry_rsi_short,
            "atr_floor": best_config.atr_floor,
            "structure_lookback": best_config.structure_lookback,
        }
        optimization_rows.append(fixed_summary)
        save_outputs(base_dir, f"eth_smc_opt_{label}", fixed, fixed_summary, {"grid": grid})

        compounded = apply_compounding(
            trades,
            risk_fraction=args.compound_risk_fraction,
            starting_balance=args.starting_balance,
        )
        compound_summary = {
            **summarize(trades, prepared.index.min(), prepared.index.max()),
            **summarize_sized(compounded, args.starting_balance),
            "timeframe": label,
            "config_name": best_config.name,
        }
        compound_rows.append(compound_summary)
        save_outputs(base_dir, f"eth_smc_compound_{label}", compounded, compound_summary)

    mtf_setups = [
        ("1h_trend_5m_entry", base_dir / "BINANCE_ETHUSDT, 5.csv", base_dir / "BINANCE_ETHUSDT, 60.csv"),
        ("1h_trend_15m_entry", base_dir / "BINANCE_ETHUSDT, 15.csv", base_dir / "BINANCE_ETHUSDT, 60.csv"),
    ]
    mtf_config = StrategyConfig(name="mtf_1h_bias", min_score=5, structure_expiry_bars=48, structure_lookback=36)
    for label, entry_csv, trend_csv in mtf_setups:
        _, fixed, summary = run_multi_timeframe(entry_csv, trend_csv, args.lookback_days, mtf_config, label, args.starting_balance)
        summary["entry_timeframe"] = label.split("_")[-2]
        summary["trend_timeframe"] = "1h"
        mtf_rows.append(summary)
        save_outputs(base_dir, f"eth_smc_{label}", fixed, summary)

    optimization_df = pd.DataFrame(optimization_rows).sort_values("net_pnl_usd", ascending=False)
    compound_df = pd.DataFrame(compound_rows).sort_values("ending_balance_usd", ascending=False)
    mtf_df = pd.DataFrame(mtf_rows).sort_values("net_pnl_usd", ascending=False)
    optimization_df.to_csv(base_dir / "eth_smc_optimization_compare.csv", index=False)
    compound_df.to_csv(base_dir / "eth_smc_compound_compare.csv", index=False)
    mtf_df.to_csv(base_dir / "eth_smc_mtf_compare.csv", index=False)

    print("OPTIMIZATION_RESULTS")
    print(optimization_df.to_string(index=False))
    print("COMPOUND_RESULTS")
    print(compound_df.to_string(index=False))
    print("MTF_RESULTS")
    print(mtf_df.to_string(index=False))


if __name__ == "__main__":
    main()
