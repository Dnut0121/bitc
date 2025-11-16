#!/usr/bin/env python3
"""Join downloaded Binance 1s kline CSVs into a single OHLCV dataset."""
from __future__ import annotations

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


STANDARD_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def find_csv_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*.csv"))


def _guess_time_unit(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "ms"
    median = numeric.median()
    if median > 1e15:
        return "us"
    if median > 1e12:
        return "ms"
    return "s"


def read_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] > len(STANDARD_COLUMNS):
        df = df.iloc[:, : len(STANDARD_COLUMNS)]
    df.columns = STANDARD_COLUMNS[: df.shape[1]]
    unit = _guess_time_unit(df["open_time"])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit=unit, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def build_dataset(source_dir: Path) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for csv_path in find_csv_files(source_dir):
        try:
            frame = read_daily_csv(csv_path)
        except Exception as exc:
            print(f"warning: could not read {csv_path.name}: {exc}")
            continue
        blocks.append(frame)
    if not blocks:
        raise RuntimeError(f"no CSV files found in {source_dir}")
    result = pd.concat(blocks)
    result = result[~result.index.duplicated(keep="first")]
    return result.sort_index()


def _load_cached_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError(f"Cached CSV at {path} is missing a timestamp column.")
    df = df.set_index("timestamp").sort_index()
    missing = set(STANDARD_COLUMNS[1:6]) - set(df.columns)
    if missing:
        raise ValueError(f"Cached CSV at {path} is missing columns: {sorted(missing)}")
    return df[["open", "high", "low", "close", "volume"]]


def prepare_dataset(
    source_dir: Path,
    destination: Path,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    cache_exists = destination.exists()
    if cache_exists:
        print(f"cached dataset found at {destination}; skipping raw CSV merge")
        combined = _load_cached_dataset(destination)
    else:
        combined = build_dataset(source_dir)
    if start:
        combined = combined[combined.index >= pd.Timestamp(start)]
    if end:
        combined = combined[combined.index <= pd.Timestamp(end)]
    needs_write = not cache_exists or bool(start) or bool(end)
    if needs_write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(destination)
        print(f"saved consolidated dataset to {destination}")
    else:
        print(f"reused cached dataset without rebuilding at {destination}")
    return destination


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate downloaded Binance 1s CSV into usable OHLCV.")
    parser.add_argument("source_dir", type=Path, help="Folder containing downloaded ZIP-extracted CSVs.")
    parser.add_argument("destination", type=Path, help="Output CSV path (e.g., dataset/binance_ohlcv.csv).")
    parser.add_argument("--start", help="Optional inclusive start date (ISO YYYY-MM-DD).")
    parser.add_argument("--end", help="Optional inclusive end date (ISO YYYY-MM-DD).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    prepare_dataset(args.source_dir, args.destination, args.start, args.end)


if __name__ == "__main__":
    main()
