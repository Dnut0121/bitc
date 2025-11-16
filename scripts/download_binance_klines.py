#!/usr/bin/env python3
"""Download Binance daily klines ZIPs by date for a given symbol/interval."""
from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urljoin

import requests


BASE_URL = "https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/"


def daterange(start: date, end: date) -> Iterable[date]:
    current = end
    while current >= start:
        yield current
        current -= timedelta(days=1)


def build_zip_url(symbol: str, interval: str, target_date: date) -> str:
    base = BASE_URL.format(symbol=symbol.upper(), interval=interval)
    filename = f"{symbol.upper()}-{interval}-{target_date:%Y-%m-%d}.zip"
    return urljoin(base, filename)


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"skip (cached): {dest.name}")
        return True
    print(f"downloading {dest.name}")
    try:
        with requests.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as out:
                shutil.copyfileobj(resp.raw, out)
        return True
    except requests.HTTPError as exc:
        if exc.response.status_code == 404:
            print(f"missing file: {dest.name}")
            return False
        raise


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)
    print(f"extracted {zip_path.name} -> {target_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download Binance daily klines zip files.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol on Binance Vision.")
    parser.add_argument("--interval", default="1s", help="Interval folder name (e.g., 1s, 1m).")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("dataset/binance_raw"),
        help="Directory to store zip files and extracted CSVs.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Unzip each downloaded archive into a subdirectory of the same name.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Stop after downloading this many files (0 = no limit).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if start > end:
        raise ValueError("start must be on or before end")

    count = 0
    for target_date in daterange(start, end):
        if args.max_files and count >= args.max_files:
            break
        url = build_zip_url(args.symbol, args.interval, target_date)
        zip_path = args.dest / url.split("/")[-1]
        succeeded = download_file(url, zip_path)
        if succeeded:
            count += 1
            if args.extract:
                extract_zip(zip_path, args.dest / zip_path.stem)


if __name__ == "__main__":
    main()
