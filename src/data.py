"""Binance 1-second OHLCV download helpers."""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable

import warnings

import pandas as pd
import requests

API_URL = "https://api.binance.com/api/v3/aggTrades"

REQUIRED_COLUMNS = {
    "timestamp": ["timestamp", "open_time", "ts", "time", "date"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c"],
    "volume": ["volume", "v"],
}


def to_milliseconds(value: str | int | float | datetime) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        parsed = pd.to_datetime(value, utc=True, errors="raise")
        return int(parsed.timestamp() * 1000)
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    raise TypeError(f"Cannot convert {value!r} to milliseconds.")


def fetch_agg_trades(
    symbol: str,
    start_time: int | None = None,
    end_time: int | None = None,
    from_id: int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "limit": limit}
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time
    if from_id is not None:
        params["fromId"] = from_id
    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()
    trades = response.json()
    if not trades:
        return pd.DataFrame(
            columns=["aggId", "price", "qty", "firstId", "lastId", "timestamp", "isBuyerMaker"]
        )
    df = pd.DataFrame(trades)
    df = df.rename(
        columns={
            "a": "aggId",
            "p": "price",
            "q": "qty",
            "f": "firstId",
            "l": "lastId",
            "T": "timestamp",
            "m": "isBuyerMaker",
        }
    )
    df["price"] = pd.to_numeric(df["price"])
    df["qty"] = pd.to_numeric(df["qty"])
    return df


def iter_agg_trades(
    symbol: str, start_time: int, end_time: int, limit: int = 1000
) -> Iterable[pd.DataFrame]:
    next_id: int | None = None
    requested_end = end_time
    while True:
        batch = fetch_agg_trades(symbol, start_time=start_time, end_time=requested_end, from_id=next_id, limit=limit)
        if batch.empty:
            break
        yield batch
        last_row = batch.iloc[-1]
        last_ts = int(last_row["timestamp"])
        if last_ts >= requested_end - 1:
            break
        next_id = int(last_row["aggId"]) + 1
        start_time = last_ts + 1
        time.sleep(0.15)


def trades_to_candles(trades: pd.DataFrame, interval: str = "1S") -> pd.DataFrame:
    if trades.empty:
        raise ValueError("No trades provided for resampling.")
    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    trades = trades.set_index("timestamp")
    ohlc = trades["price"].resample(interval).agg(["first", "max", "min", "last"])
    volume = trades["qty"].resample(interval).sum()
    candles = pd.concat(
        {
            "open": ohlc["first"],
            "high": ohlc["max"],
            "low": ohlc["min"],
            "close": ohlc["last"],
            "volume": volume,
        },
        axis=1,
    ).dropna()
    candles.index = pd.DatetimeIndex(candles.index).tz_convert(None)
    return candles


def download_per_second_ohlcv(
    symbol: str,
    start_time: str | int | float | datetime,
    end_time: str | int | float | datetime,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    start_ms = to_milliseconds(start_time)
    end_ms = to_milliseconds(end_time)
    if start_ms >= end_ms:
        raise ValueError("start_time must precede end_time.")
    batches = []
    for batch in iter_agg_trades(symbol, start_ms, end_ms):
        batches.append(batch)
    if not batches:
        raise RuntimeError("Binance returned no aggregated trades.")
    trades = pd.concat(batches, ignore_index=True)
    candles = trades_to_candles(trades, interval="1S")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        candles.to_csv(cache_path)
    return candles


def _find_best_match(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _align_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for target, candidates in REQUIRED_COLUMNS.items():
        match = _find_best_match(df.columns, candidates)
        if match is None:
            raise ValueError(f"Missing column for {target}: tried {candidates}")
        rename_map[match] = target
    return df.rename(columns=rename_map)


def _detect_timestamp_unit(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "auto"
    numeric = pd.to_numeric(series, errors="coerce")
    sample = numeric.dropna()
    if sample.empty:
        # allow string timestamps (ISO) by falling back to auto
        return "auto"
    median = sample.median()
    if median > 1e15:
        return "us"
    if median > 1e12:
        return "ms"
    if median > 1e9:
        return "s"
    return "s"


def _read_last_rows(path: Path, tail_rows: int) -> pd.DataFrame:
    path = Path(path)
    if tail_rows <= 0:
        raise ValueError("tail_rows must be a positive integer.")
    chunksize = max(tail_rows, 200_000)
    reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
    recent: Deque[pd.DataFrame] = deque()
    rows_in_cache = 0
    columns = None
    for chunk in reader:
        if columns is None:
            columns = chunk.columns
        recent.append(chunk)
        rows_in_cache += len(chunk)
        while recent and rows_in_cache - len(recent[0]) >= tail_rows:
            rows_in_cache -= len(recent[0])
            recent.popleft()
    if not recent:
        if columns is None:
            return pd.DataFrame()
        return pd.DataFrame(columns=columns)
    combined = pd.concat(recent, ignore_index=True)
    if tail_rows < len(combined):
        combined = combined.tail(tail_rows)
    return combined


def _read_cached_csv(path: Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cached OHLCV file not found at {path}")
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Cached OHLCV at {path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Failed to parse cached OHLCV at {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not decode cached OHLCV at {path}: {exc}") from exc


def load_ohlcv(path: Path, tail_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    kwargs = {"low_memory": False}
    if tail_rows is not None:
        df = _read_last_rows(path, tail_rows)
    else:
        df = _read_cached_csv(path, **kwargs)
    df = _align_columns(df)

    ts_unit = _detect_timestamp_unit(df["timestamp"])
    if ts_unit == "auto":
        # 문자열/이미 datetime 컬럼은 unit 없이 자동 파싱
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        # 숫자형 에포크 값은 추정한 unit으로 파싱
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=ts_unit, errors="coerce")
    mask = df["timestamp"].notna()
    if not mask.any():
        raise ValueError("Parsed timestamps contain NaN after conversion.")
    if not mask.all():
        warnings.warn(
            "Dropped rows with unparseable timestamps from cached data; "
            "see `dataset` for the original rows.",
            stacklevel=2,
        )
        df = df.loc[mask]
    df = df.set_index("timestamp").sort_index()
    return (
        df[["open", "high", "low", "close", "volume"]]
        .resample("1s")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
