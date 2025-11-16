from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scripts.prepare_dataset import prepare_dataset
from src.cli import main as cli_main


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset and run the LSTM CLI.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("dataset/binance_raw"),
        help="Directory where raw CSVs exist after extracting the ZIPs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/binance_ohlcv.csv"),
        help="Destination path for the consolidated CSV.",
    )
    parser.add_argument("--start", help="Optional inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Optional inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--window", type=int, default=60, help="LSTM lookback window.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Train batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--plot", action="store_true", help="Show final plots.")
    parser.add_argument(
        "--tail-rows",
        type=_positive_int,
        help="When supplied, tell the CLI to load only the last N rows from the cached CSV.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    dataset_path = prepare_dataset(args.raw_dir, args.output, args.start, args.end)
    cli_args = [
        str(dataset_path),
        "--window",
        str(args.window),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--grad-clip",
        str(args.grad_clip),
    ]
    if args.tail_rows:
        cli_args.extend(["--tail-rows", str(args.tail_rows)])
    if args.plot:
        cli_args.append("--plot")
    cli_main(cli_args)


if __name__ == "__main__":
    main()
