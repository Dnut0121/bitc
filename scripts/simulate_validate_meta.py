#!/usr/bin/env python3
"""Simulate trading on dataset/validate using base LSTM + meta-model."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable
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
from src.pressure import compute_pressure
from src.features import add_microstructure_features
from src.model import build_sequences, load_model_checkpoint
from src.backtest import BacktestResult


class NumpyDataset(Dataset):
    def __init__(self, data: np.ndarray) -> None:
        self.data = torch.from_numpy(data.astype(np.float32))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _extract_validate_date(path: Path) -> datetime:
    # 기대 형식: BTCUSDT-1s-YYYY-MM-DD.csv
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse date from filename: {path.name}")
    date_str = "-".join(parts[-3:])
    return datetime.fromisoformat(date_str)


def _pick_latest_validate_csv(validate_dir: Path) -> Path:
    candidates = sorted(validate_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found under {validate_dir}")
    dated = []
    for p in candidates:
        try:
            dt = _extract_validate_date(p)
        except Exception:
            continue
        dated.append((dt, p))
    if not dated:
        raise RuntimeError(f"No dated CSVs (like BTCUSDT-1s-YYYY-MM-DD.csv) found under {validate_dir}")
    dated.sort()
    return dated[-1][1]


def compute_prob_up_series_with_checkpoint(
    ohlcv: pd.DataFrame,
    model_path: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    """베이스 LSTM 체크포인트를 사용해 하루치 데이터에 대한 prob_up 시계열을 계산합니다."""
    pressured = compute_pressure(ohlcv)
    model, feature_mean, feature_std, lookback = load_model_checkpoint(model_path, device=device)

    # 체크포인트에 저장된 feature_mean 길이에 맞춰 사용할 피처 컬럼을 결정한다.
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
    else:  # pragma: no cover
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

    prob_df = pressured.reindex(indexes).copy()
    prob_df["prob_up"] = prob_up
    # 디버그용: 베이스 LSTM 확률 분포 요약
    print(
        f"[simulate-meta] prob_up stats — min={prob_up.min():.4f}, "
        f"max={prob_up.max():.4f}, mean={prob_up.mean():.4f}"
    )
    return prob_df, feature_mean, feature_std, lookback


def run_meta_backtest(
    df: pd.DataFrame,
    meta_model: nn.Module,
    feature_cols: list[str],
    base_threshold: float,
    meta_threshold: float,
    window: int,
    horizon: int,
    fee_rate: float,
    slippage_rate: float,
    device: torch.device,
) -> BacktestResult:
    """베이스 LSTM + 메타모델을 함께 사용하는 이벤트 기반 백테스트."""
    required_cols = {"close", "prob_up"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"run_meta_backtest: DataFrame is missing columns: {sorted(missing)}")
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"run_meta_backtest: required meta feature '{c}' not found in DataFrame")

    if len(df) < window + horizon + 5:
        raise ValueError("run_meta_backtest: not enough rows for the requested window/horizon.")

    close = df["close"].to_numpy(dtype=float)
    prob_up = df["prob_up"].to_numpy(dtype=float)
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("run_meta_backtest: DataFrame index must be a DatetimeIndex.")

    # 메타 피처 행렬 및 메타 확률 사전 계산
    X_meta = df[feature_cols].to_numpy(dtype=np.float32)
    X_meta_t = torch.from_numpy(X_meta).to(device)
    meta_model.eval()
    with torch.no_grad():
        logits_meta = meta_model(X_meta_t).squeeze(-1)
        meta_prob = torch.sigmoid(logits_meta).cpu().numpy()
    # NaN/inf 방어: 메타 확률에서 문제가 되는 값은 0으로 치환
    nan_count = int(np.isnan(meta_prob).sum())
    if nan_count > 0:
        print(f"[simulate-meta] WARNING: meta_prob contains {nan_count} NaN values; replacing with 0.0")
    meta_prob = np.nan_to_num(meta_prob, nan=0.0, posinf=0.0, neginf=0.0)
    # 디버그용: 메타 모델 확률 분포 요약
    print(
        f"[simulate-meta] meta_prob stats — min={meta_prob.min():.4f}, "
        f"max={meta_prob.max():.4f}, mean={meta_prob.mean():.4f}"
    )

    total_cost_rate = max(0.0, fee_rate + slippage_rate)

    in_position = False
    entry_price = 0.0
    entry_idx = 0
    target_price = 0.0
    stop_price = 0.0

    pnl_per_step = np.zeros(len(df), dtype=float)
    trade_pnls: list[float] = []
    trade_rs: list[float] = []
    wins = 0
    losses = 0
    max_daily_loss = -3.0
    max_consec_losses = 5
    daily_pnl: dict[pd.Timestamp, float] = {}
    consec_losses = 0

    for i in range(len(df)):
        ts = index[i]
        price = close[i]
        p = prob_up[i]
        mp = meta_prob[i]

        day = ts.normalize()
        day_pnl = daily_pnl.get(day, 0.0)

        allow_new_trade = (day_pnl > max_daily_loss) and (consec_losses < max_consec_losses)

        if not in_position and allow_new_trade:
            # 베이스 신호 + 메타모델 필터 동시 만족 시에만 롱 진입
            if p >= base_threshold and mp >= meta_threshold:
                in_position = True
                entry_price = price
                entry_idx = i
                # 최근 window초 평균 변동폭 기반 목표/손절
                if "range" in df.columns:
                    avg_range = float(df["range"].iloc[max(0, i - window + 1) : i + 1].mean())
                else:
                    avg_range = float(
                        (df["high"] - df["low"]).iloc[max(0, i - window + 1) : i + 1].mean()
                    )
                tp = avg_range * 2.0
                sl = avg_range
                target_price = entry_price + tp
                stop_price = entry_price - sl
            continue

        if in_position:
            unrealized = price - entry_price
            pnl_per_step[i] = unrealized

            holding = i - entry_idx
            exit_reason = None
            if price >= target_price:
                exit_reason = "tp"
            elif price <= stop_price:
                exit_reason = "sl"
            elif holding >= horizon:
                exit_reason = "time"

            if exit_reason is not None:
                gross = unrealized
                roundtrip_cost = total_cost_rate * (entry_price + price)
                net = gross - roundtrip_cost
                trade_pnls.append(net)
                r_multiple = net / (abs(stop_price - entry_price) + 1e-9)
                trade_rs.append(r_multiple)

                if net > 0:
                    wins += 1
                    consec_losses = 0
                else:
                    losses += 1
                    consec_losses += 1

                daily_pnl[day] = daily_pnl.get(day, 0.0) + net

                in_position = False
                entry_price = 0.0
                target_price = 0.0
                stop_price = 0.0

    # 성과 지표 계산
    pnl_series = pd.Series(pnl_per_step, index=index)
    from src.backtest import _compute_performance  # type: ignore[import]

    total_ret, sharpe = _compute_performance(pnl_series)

    num_trades = len(trade_pnls)
    hit_ratio = wins / num_trades if num_trades > 0 else 0.0
    avg_r = float(np.mean(trade_rs)) if trade_rs else 0.0
    profits = [p for p in trade_pnls if p > 0]
    losses_list = [p for p in trade_pnls if p < 0]
    gross_profit = float(sum(profits))
    gross_loss = -float(sum(losses_list)) if losses_list else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    equity = pnl_series.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    return BacktestResult(
        total_return=float(total_ret),
        sharpe=float(sharpe),
        max_drawdown=max_dd,
        hit_ratio=float(hit_ratio),
        avg_r_multiple=avg_r,
        profit_factor=float(profit_factor),
        num_trades=int(num_trades),
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Simulate trading on dataset/validate using base LSTM + meta-model."
    )
    parser.add_argument(
        "--validate-dir",
        type=Path,
        default=Path("dataset/validate"),
        help="Directory containing validate CSVs (e.g., BTCUSDT-1s-YYYY-MM-DD.csv).",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Specific validate CSV to use. When omitted, automatically picks the latest date file.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("models/btc_lstm.pt"),
        help="Base LSTM checkpoint path.",
    )
    parser.add_argument(
        "--meta-model",
        type=Path,
        default=Path("models/btc_meta.pt"),
        help="Meta-model checkpoint path (trained via run_meta_pipeline).",
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        help="Base LSTM prob_up threshold. When omitted, uses the value stored in the meta-model checkpoint.",
    )
    parser.add_argument(
        "--meta-threshold",
        type=float,
        default=0.5,
        help="Meta-model good-trade probability threshold.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Lookback window (seconds) used for range-based TP/SL.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
        help="Maximum holding horizon (seconds).",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0006,
        help="Per-trade fee rate (e.g., 0.0005 = 0.05%).",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0003,
        help="Per-trade slippage rate.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.file is not None:
        csv_path = args.file
    else:
        csv_path = _pick_latest_validate_csv(args.validate_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"Validate CSV not found: {csv_path}")
    if not args.base_model.exists():
        raise FileNotFoundError(f"Base LSTM checkpoint not found: {args.base_model}")
    if not args.meta_model.exists():
        raise FileNotFoundError(f"Meta-model checkpoint not found: {args.meta_model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[simulate-meta] Using device: {device}")
    print(f"[simulate-meta] Validate CSV: {csv_path}")
    print(f"[simulate-meta] Base model:  {args.base_model}")
    print(f"[simulate-meta] Meta model:  {args.meta_model}")

    ohlcv = load_ohlcv(csv_path)
    print(f"[simulate-meta] Loaded {len(ohlcv)} rows from {csv_path.name}")

    prob_df, _, _, _ = compute_prob_up_series_with_checkpoint(ohlcv, args.base_model, device=device)

    # 메타 모델 체크포인트 로드
    meta_ckpt = torch.load(args.meta_model, map_location=device)
    feature_cols = list(meta_ckpt.get("feature_cols", []))
    base_threshold = (
        float(args.base_threshold)
        if args.base_threshold is not None
        else float(meta_ckpt.get("base_threshold", 0.5))
    )

    df_micro = add_microstructure_features(prob_df)
    df_full = df_micro.reindex(prob_df.index).copy()
    df_full["prob_up"] = prob_df["prob_up"]

    # 메타 모델 구성 및 state 복원
    meta_model = nn.Linear(len(feature_cols), 1).to(device)
    meta_model.load_state_dict(meta_ckpt["state_dict"])

    # 디버그용: threshold와 조건 만족 개수 출력
    p = df_full["prob_up"].to_numpy(dtype=float)
    print(
        f"[simulate-meta] Using base_threshold={base_threshold:.4f}, "
        f"meta_threshold={args.meta_threshold:.4f}"
    )
    # meta_prob는 run_meta_backtest 내부에서 계산되지만, base_threshold 기준으로만 먼저 찍어본다.
    print(
        f"[simulate-meta] count(prob_up >= base_threshold) = "
        f"{int((p >= base_threshold).sum())} / {len(p)}"
    )

    bt = run_meta_backtest(
        df_full,
        meta_model=meta_model,
        feature_cols=feature_cols,
        base_threshold=base_threshold,
        meta_threshold=args.meta_threshold,
        window=args.window,
        horizon=args.horizon,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        device=device,
    )

    print()
    print("=== LSTM + Meta-model backtest on validate ===")
    print(f"- File:           {csv_path.name}")
    print(f"- Total return:   {bt.total_return:.4f} (price diff units)")
    print(f"- Sharpe:         {bt.sharpe:.3f}")
    print(f"- Max drawdown:   {bt.max_drawdown:.4f}")
    print(f"- Hit ratio:      {bt.hit_ratio:.3f}")
    print(f"- Avg R-multiple: {bt.avg_r_multiple:.3f}")
    print(f"- Profit factor:  {bt.profit_factor:.3f}")
    print(f"- Num trades:     {bt.num_trades}")


if __name__ == "__main__":
    main()
