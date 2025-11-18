#!/usr/bin/env python3
"""Evaluate next-second direction prediction accuracy on a daily 1s CSV.

This script loads a trained LSTM checkpoint (e.g., models/btc_lstm.pt),
applies the same feature engineering as training, and computes the
accuracy of "next-second up/down" predictions on a validation CSV.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# 프로젝트 루트(src 패키지)를 import 경로에 추가
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# run_pipeline.py처럼 프로젝트 루트에서 실행할 때와
# scripts 디렉토리 안에서 직접 실행할 때 모두 동작하도록 import를 유연하게 처리한다.
try:
    # 루트에서 "python scripts/eval_next_second.py"로 실행하는 경우
    from scripts.prepare_dataset import read_daily_csv  # type: ignore[import]
except ImportError:  # pragma: no cover
    # scripts 디렉토리 안에서 직접 실행하는 경우
    from prepare_dataset import read_daily_csv  # type: ignore[import]

from src.model import SequenceDataset, build_sequences, load_model_checkpoint
from src.pressure import compute_pressure


def _extract_date_from_validate_name(path: Path) -> datetime:
    """Extract date from a filename like BTCUSDT-1s-2025-11-16.csv."""
    stem = path.stem  # e.g., BTCUSDT-1s-2025-11-16
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse date from filename: {path.name}")
    # Take the last three components as YYYY-MM-DD
    date_str = "-".join(parts[-3:])
    return datetime.fromisoformat(date_str)


def _pick_latest_validate_csv(validate_dir: Path) -> Path:
    candidates = list(validate_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under {validate_dir}")
    dated = []
    for p in candidates:
        try:
            dt = _extract_date_from_validate_name(p)
        except Exception:
            continue
        dated.append((dt, p))
    if not dated:
        raise RuntimeError(f"No dated CSVs (like BTCUSDT-1s-YYYY-MM-DD.csv) found under {validate_dir}")
    dated.sort()
    return dated[-1][1]


def evaluate_next_second(
    csv_path: Path,
    model_path: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    # 1) 하루치 1초 봉 CSV를 OHLCV DataFrame으로 로드
    # read_daily_csv는 timestamp 인덱스를 가진 open/high/low/close/volume DataFrame을 반환한다.
    ohlcv = read_daily_csv(csv_path)
    print(f"[eval] Loaded {len(ohlcv)} rows from {csv_path}")

    # 2) 푸쉬/임밸런스 피처 생성 (학습과 동일)
    pressured = compute_pressure(ohlcv)

    # 3) 학습된 모델과 정규화 통계, lookback 길이 로드
    model, feature_mean, feature_std, lookback = load_model_checkpoint(model_path, device=device)
    print(f"[eval] Loaded checkpoint from {model_path} (lookback={lookback})")

    # 4) 시퀀스/타깃 생성 (학습과 동일한 방식)
    sequences, targets, indexes = build_sequences(pressured, lookback)
    total_sequences = len(sequences)
    if total_sequences < 5:
        raise ValueError(f"Not enough sequences ({total_sequences}) to evaluate.")
    print(f"[eval] Built {total_sequences} sequences for next-second prediction.")

    # 5) 학습 시점의 정규화 통계를 사용해 시퀀스를 정규화
    feature_mean = feature_mean.astype(np.float32)
    feature_std = feature_std.astype(np.float32)
    std_safe = np.where(feature_std < 1e-6, 1.0, feature_std)
    # sequences: (N, L, F), mean/std: (F,)
    normalized = (sequences - feature_mean) / std_safe

    dataset = SequenceDataset(normalized, targets)
    loader = DataLoader(dataset, batch_size=batch_size)

    # 6) 모델을 평가 모드로 두고, 전체 시퀀스에 대한 정확도 계산
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seq_batch, target_batch in loader:
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(seq_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == target_batch).sum().item()
            total += target_batch.size(0)

    accuracy = correct / total if total > 0 else 0.0
    print()
    print("=== Next-second direction evaluation ===")
    print(f"- File: {csv_path.name}")
    print(f"- Sequences evaluated: {total}")
    print(f"- Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate next-second up/down prediction accuracy on a daily validation CSV."
    )
    parser.add_argument(
        "--validate-dir",
        type=Path,
        default=Path("dataset/validate"),
        help="Directory containing raw 1s kline CSVs (e.g., BTCUSDT-1s-YYYY-MM-DD.csv).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Specific validation CSV to use. When omitted, automatically picks the latest date file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/btc_lstm.pt"),
        help="Trained LSTM checkpoint path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.file is not None:
        csv_path = args.file
    else:
        csv_path = _pick_latest_validate_csv(args.validate_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")

    evaluate_next_second(csv_path, args.model, args.batch_size, device)


if __name__ == "__main__":
    main()
