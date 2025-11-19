#!/usr/bin/env python3
"""Split a consolidated per-second OHLCV CSV into per-day CSV files.

Input:
    - A CSV like dataset/binance_ohlcv.csv with columns:
      timestamp, open, high, low, close, volume

Output:
    - One CSV per day under the specified output directory, named:
      binance_ohlcv_YYYY-MM-DD.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def split_ohlcv_by_day(source: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError(f"{source} is missing a 'timestamp' column.")
    if df.empty:
        raise ValueError(f"{source} has no rows.")

    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date

    for date, day_frame in df.groupby("date"):
        day_str = str(date)  # YYYY-MM-DD
        out_path = out_dir / f"binance_ohlcv_{day_str}.csv"
        # timestamp를 컬럼으로 유지한 채 저장
        day_frame.drop(columns=["date"]).to_csv(out_path, index=False)
        print(f"wrote {len(day_frame)} rows to {out_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Split consolidated Binance per-second OHLCV CSV into per-day CSV files."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Input CSV path (e.g., dataset/binance_ohlcv.csv).",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Output directory where per-day CSVs will be written (e.g., dataset/daily).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    split_ohlcv_by_day(args.source, args.out_dir)


if __name__ == "__main__":
    main()

