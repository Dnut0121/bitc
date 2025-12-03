#!/usr/bin/env python3
"""Analyze cost-aware labels across multiple horizons and train a simple meta-model.

1) 하루치 1초 봉 CSV와 학습된 LSTM 체크포인트를 불러온다.
2) compute_pressure + add_microstructure_features로 마이크로구조 피처를 계산한다.
3) LSTM으로 각 시점에 대한 prob_up을 계산한다.
4) 여러 horizon에 대해 cost-aware 라벨을 만들고, 각 horizon에서의 품질(precision/recall 등)을 요약한다.
5) triple-barrier 라벨을 타깃으로 하는 간단한 메타 모델(로지스틱 회귀 형태)을 학습한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 프로젝트 루트(src 패키지)를 import 경로에 추가
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_ohlcv
from src.model import build_sequences, load_model_checkpoint
from src.pressure import compute_pressure
from src.features import add_microstructure_features
from src.labels import multi_horizon_cost_labels, triple_barrier_labels


class NumpyDataset(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        self.data = torch.from_numpy(data.astype(np.float32))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _extract_date_from_name(path: Path) -> pd.Timestamp:
    stem = path.stem  # e.g., BTCUSDT-1s-2025-01-01
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse date from filename: {path.name}")
    date_str = "-".join(parts[-3:])
    return pd.to_datetime(date_str)


def _pick_latest_file(d: Path) -> Path:
    candidates = sorted(d.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSVs found under {d}")
    dated = []
    for p in candidates:
        try:
            dt = _extract_date_from_name(p)
        except Exception:
            continue
        dated.append((dt, p))
    if not dated:
        raise RuntimeError(f"No dated CSVs found under {d}")
    dated.sort()
    return dated[-1][1]


def compute_prob_up_series(
    ohlcv: pd.DataFrame,
    model_path: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    """학습된 LSTM 체크포인트를 사용해 하루치 데이터에 대한 prob_up 시계열을 계산한다."""
    pressured = compute_pressure(ohlcv)
    model, feature_mean, feature_std, lookback = load_model_checkpoint(model_path, device=device)

    # 체크포인트에 저장된 feature_mean 길이에 맞춰 사용할 피처 컬럼을 결정한다.
    # - 5차원: 기본 조합 ["buy_ratio", "volume", "range", "price_change", "imbalance"]
    # - 6차원: 강화된 order-flow 조합
    if feature_mean.ndim == 1 and feature_mean.shape[0] == 5:
        feature_columns = ["buy_ratio", "volume", "range", "price_change", "imbalance"]
    elif feature_mean.ndim == 1 and feature_mean.shape[0] == 6:
        feature_columns = [
            "buy_ratio",
            "price_change",
            "range",
            "taker_imbalance",
            "log_trades",
            "volume_per_trade",
        ]
    else:  # pragma: no cover - 방어적 코드
        raise ValueError(f"Unexpected feature_mean shape in checkpoint: {feature_mean.shape!r}")

    sequences, _, indexes = build_sequences(
        pressured,
        lookback=lookback,
        max_sequences=None,
        use_cost_labels=False,
        label_horizon=1,
        feature_columns=feature_columns,
    )
    feature_mean = feature_mean.astype(np.float32)
    feature_std = feature_std.astype(np.float32)
    std_safe = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = (sequences - feature_mean) / std_safe

    dataset = NumpyDataset(normalized)
    loader = DataLoader(dataset, batch_size=512)

    model.eval()
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    prob_up = np.concatenate(all_probs, axis=0)

    # index 정렬 및 DataFrame 구성
    prob_df = pressured.reindex(indexes).copy()
    prob_df["prob_up"] = prob_up
    return prob_df, feature_mean, feature_std, lookback


def analyze_horizons(
    prob_df: pd.DataFrame,
    horizons: Sequence[int],
    fee_rate: float,
    slippage_rate: float,
    margin_rate: float,
    base_threshold: float,
) -> None:
    """여러 horizon 에 대해 cost-aware 라벨과 base LSTM의 품질을 요약 출력."""
    close = prob_df["close"]
    labels_df = multi_horizon_cost_labels(
        close,
        horizons=list(horizons),
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        margin_rate=margin_rate,
    ).reindex(prob_df.index)

    prob_up = prob_df["prob_up"].to_numpy()
    pred_long = prob_up >= base_threshold

    print()
    print("=== Multi-horizon cost-aware label analysis ===")
    print(f"- Base long threshold (prob_up): {base_threshold:.2f}")
    for h in horizons:
        col = f"H_{h}s"
        y = labels_df[col].to_numpy()
        mask_valid = ~np.isnan(y)
        y_valid = y[mask_valid]
        pred_valid = pred_long[mask_valid]
        if y_valid.size == 0:
            continue
        pos_rate = (y_valid == 1).mean()
        if pos_rate == 0:
            precision = 0.0
            recall = 0.0
        else:
            tp = ((pred_valid) & (y_valid == 1)).sum()
            fp = ((pred_valid) & (y_valid != 1)).sum()
            fn = ((~pred_valid) & (y_valid == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"H={h:>4}s | label(+1) 비율={pos_rate:6.3f}  precision={precision:6.3f}  recall={recall:6.3f}")


def train_meta_model(
    prob_df: pd.DataFrame,
    base_threshold: float,
    tb_horizon: int,
    tb_up_pct: float,
    tb_down_pct: float,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> None:
    """triple-barrier 라벨을 타깃으로 하는 간단한 메타 모델(로지스틱)을 학습한다.

    - 입력: prob_up + 마이크로구조 피처 일부
    - 타깃: triple-barrier 라벨에서 +1 (좋은 트레이드) vs 나머지(0/-1)
    - 학습은 base 시그널이 롱( prob_up >= base_threshold )인 시점들에 대해서만 수행한다.
    """
    # 마이크로구조 피처 추가
    micro = add_microstructure_features(prob_df)
    df = micro.reindex(prob_df.index).copy()
    df["prob_up"] = prob_df["prob_up"]

    tb = triple_barrier_labels(
        close=df["close"],
        high=df["high"],
        low=df["low"],
        horizon=tb_horizon,
        up_pct=tb_up_pct,
        down_pct=tb_down_pct,
    )
    df["tb_label"] = tb

    base_long = df["prob_up"] >= base_threshold
    # 메타 라벨: triple-barrier가 +1이면 1, 나머지는 0
    y = (df["tb_label"] == 1).astype(float)

    # 학습에 사용할 피처 선택 (원하면 조합을 조정해도 된다)
    feature_cols = [
        "prob_up",
        "ret_1s",
        "ret_5s",
        "ret_10s",
        "rv_30s",
        "rv_120s",
        "pos_in_range",
        "taker_imbalance",
        "log_trades",
        "volume_per_trade",
        "regime_vol",
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Meta-model features missing from DataFrame: {missing}")

    feature_data = df[feature_cols].to_numpy(dtype=np.float32)
    target_data = y.to_numpy(dtype=np.float32)

    # base 시그널이 롱인 시점만 메타 모델 학습에 사용 (메타 라벨링 아이디어)
    mask = base_long.to_numpy(dtype=bool)
    X = feature_data[mask, :]
    y_meta = target_data[mask]

    if X.shape[0] < 100:
        print("[meta] Not enough samples where base signal is long; skip meta-model training.")
        return

    # 시간 순서 보존 train/val split
    total = X.shape[0]
    train_n = int(total * 0.7)
    X_train, X_val = X[:train_n], X[train_n:]
    y_train, y_val = y_meta[:train_n], y_meta[train_n:]

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    model = nn.Linear(X.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print()
    print("=== Meta-model training (triple-barrier) ===")
    print(
        f"- Samples for meta-labeling (base long): total={total}, train={train_n}, val={total-train_n}, "
        f"positive_rate={y_meta.mean():.3f}"
    )

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t).squeeze(-1)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze(-1)
            val_loss = criterion(val_logits, y_val_t).item()
            train_prob = torch.sigmoid(logits).cpu().numpy()
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        # 간단한 accuracy/precision/recall
        train_pred = (train_prob >= 0.5).astype(int)
        val_pred = (val_prob >= 0.5).astype(int)
        train_acc = (train_pred == y_train).mean()
        val_acc = (val_pred == y_val).mean()

        print(
            f"Epoch {epoch+1}/{epochs}: train_loss={loss.item():.4f} "
            f"val_loss={val_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze cost-aware labels across multiple horizons and train a simple meta-model "
            "on triple-barrier labels using an existing LSTM checkpoint."
        )
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=Path("dataset/daily"),
        help="하루치 1초 봉 CSV들이 들어있는 디렉토리 (예: dataset/daily 또는 dataset/binance_raw/하루 폴더).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="하루치 CSV를 직접 지정 (지정하지 않으면 daily-dir에서 가장 최근 날짜 파일을 자동 선택).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/btc_lstm.pt"),
        help="학습된 LSTM 체크포인트 경로.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 10, 30, 60],
        help="분석에 사용할 cost-aware 라벨 horizon 리스트 (초 단위).",
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=0.5,
        help="LSTM prob_up 기반 롱 진입 기준 임계값.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0006,
        help="한 번 체결될 때 수수료 비율 (예: 0.0005 = 0.05%).",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0003,
        help="한 번 체결될 때 슬리피지 비율 (예: 0.0001 = 0.01%).",
    )
    parser.add_argument(
        "--margin-rate",
        type=float,
        default=0.0,
        help="수수료/슬리피지를 모두 제하고도 이만큼(비율) 이상 남을 때만 1로 라벨링합니다.",
    )
    parser.add_argument(
        "--tb-horizon",
        type=int,
        default=60,
        help="triple-barrier 라벨을 만들 때 사용할 horizon(초).",
    )
    parser.add_argument(
        "--tb-up-pct",
        type=float,
        default=0.001,
        help="triple-barrier 위 배리어(익절) 퍼센트 (예: 0.001 = 0.1%).",
    )
    parser.add_argument(
        "--tb-down-pct",
        type=float,
        default=0.001,
        help="triple-barrier 아래 배리어(손절) 퍼센트 (예: 0.001 = 0.1%).",
    )
    parser.add_argument(
        "--meta-epochs",
        type=int,
        default=5,
        help="메타 모델(로지스틱) 학습 epoch 수.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.file is not None:
        csv_path = args.file
    else:
        csv_path = _pick_latest_file(args.daily_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Daily CSV not found: {csv_path}")
    if not args.model.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[analyze-meta] Using device: {device}")
    print(f"[analyze-meta] Daily CSV: {csv_path}")
    print(f"[analyze-meta] Model checkpoint: {args.model}")

    ohlcv = load_ohlcv(csv_path)
    print(f"[analyze-meta] Loaded {len(ohlcv)} rows from {csv_path.name}")

    prob_df, _, _, _ = compute_prob_up_series(ohlcv, args.model, device=device)

    analyze_horizons(
        prob_df,
        horizons=args.horizons,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        margin_rate=args.margin_rate,
        base_threshold=args.base_threshold,
    )

    train_meta_model(
        prob_df,
        base_threshold=args.base_threshold,
        tb_horizon=args.tb_horizon,
        tb_up_pct=args.tb_up_pct,
        tb_down_pct=args.tb_down_pct,
        device=device,
        epochs=args.meta_epochs,
        lr=1e-3,
    )


if __name__ == "__main__":
    main()
