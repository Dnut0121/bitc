#!/usr/bin/env python3
"""Backtest profit from next-horizon direction predictions on a daily 1s CSV.

전처리/특징/모델은 학습 때와 동일하게 사용하고,
validate 폴더의 하루치 CSV를 대상으로 간단한 매매 전략들을 백테스트합니다.

지원 전략:
  - long           : prob_up >= threshold 일 때 롱 진입, horizon초 뒤 청산
  - long-short     : prob_up >= long_threshold 이면 롱,
                     prob_up <= short_threshold 이면 숏
    (두 경우 모두 horizon초 뒤 청산)
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable
import sys

import numpy as np
import torch

# 프로젝트 루트(src 패키지)를 import 경로에 추가
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # 루트에서 "python scripts/eval_profit.py"로 실행하는 경우
    from scripts.prepare_dataset import read_daily_csv  # type: ignore[import]
except ImportError:  # pragma: no cover
    # scripts 디렉토리 안에서 직접 실행하는 경우
    from prepare_dataset import read_daily_csv  # type: ignore[import]

from src.model import load_model_checkpoint
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


def build_horizon_sequences_with_prices(
    df,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """lookback 길이의 시퀀스를 만들고, horizon 초 뒤의 방향/가격 정보를 함께 반환합니다.

    반환:
        sequences   : (N, lookback, F)
        targets     : (N,) 0/1 (horizon 뒤 상승 여부)
        entry_price : (N,) 시퀀스 마지막 시점의 종가
        exit_price  : (N,) horizon초 뒤의 종가
    """
    import pandas as pd  # local import

    if not hasattr(df, "columns"):
        raise TypeError("df must be a pandas DataFrame.")

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer (seconds).")

    features: list[str] = ["buy_ratio", "volume", "range", "price_change", "imbalance"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing}")

    total_rows = len(df)
    if total_rows < lookback + horizon + 1:
        raise ValueError(
            f"Not enough rows ({total_rows}) for lookback={lookback} and horizon={horizon}."
        )

    close = df["close"].to_numpy(dtype=np.float32)
    future_close = close[horizon:]
    curr_close = close[:-horizon]
    usable_len = len(curr_close)

    targets_all = (future_close > curr_close).astype(np.float32)  # (usable_len,)
    feature_array = df[features].to_numpy(dtype=np.float32, copy=True)[:usable_len]

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    entry_prices: list[float] = []
    exit_prices: list[float] = []

    max_start = usable_len - lookback
    for start in range(max_start + 1):
        end = start + lookback
        sequences.append(feature_array[start:end])
        entry_idx = end - 1
        entry_prices.append(float(close[entry_idx]))
        exit_prices.append(float(close[entry_idx + horizon]))
        targets.append(float(targets_all[entry_idx]))

    return (
        np.stack(sequences),
        np.array(targets, dtype=np.float32),
        np.array(entry_prices, dtype=np.float32),
        np.array(exit_prices, dtype=np.float32),
    )


def evaluate_profit(
    csv_path: Path,
    model_path: Path,
    batch_size: int,
    horizon: int,
    prob_threshold: float,
    prob_short_threshold: float,
    strategy: str,
    min_consecutive: int,
    min_expected_pct: float,
    single_position: bool,
    fee_rate: float,
    slippage_rate: float,
    position_size: float,
    device: torch.device,
) -> None:
    import pandas as pd  # local import
    from torch import nn

    # 1) 하루치 1초 봉 CSV를 OHLCV DataFrame으로 로드
    ohlcv = read_daily_csv(csv_path)
    print(f"[eval-profit] Loaded {len(ohlcv)} rows from {csv_path}")

    # 2) 푸쉬/임밸런스 피처 생성 (학습과 동일)
    pressured = compute_pressure(ohlcv)

    # 3) 학습된 모델과 정규화 통계, lookback 길이 로드
    model, feature_mean, feature_std, lookback = load_model_checkpoint(model_path, device=device)
    print(f"[eval-profit] Loaded checkpoint from {model_path} (lookback={lookback})")

    # 4) horizon(초) 뒤 방향/가격을 타깃으로 하는 시퀀스 생성
    sequences, targets, entry_prices, exit_prices = build_horizon_sequences_with_prices(
        pressured, lookback=lookback, horizon=horizon
    )
    total_sequences = len(sequences)
    if total_sequences < 5:
        raise ValueError(f"Not enough sequences ({total_sequences}) to evaluate.")
    print(f"[eval-profit] Built {total_sequences} sequences for +{horizon}s backtest.")

    # 5) 학습 시점의 정규화 통계를 사용해 시퀀스를 정규화
    feature_mean = feature_mean.astype(np.float32)
    feature_std = feature_std.astype(np.float32)
    std_safe = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = (sequences - feature_mean) / std_safe

    # 6) 모델을 평가 모드로 두고, 전체 시퀀스에 대한 확률 예측
    model.eval()
    sigmoid = nn.Sigmoid()

    probs_all = np.empty(total_sequences, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, total_sequences, batch_size):
            end = min(start + batch_size, total_sequences)
            batch_np = normalized[start:end]
            seq_batch = torch.from_numpy(batch_np).to(device)
            logits = model(seq_batch)
            probs = sigmoid(logits).cpu().numpy().astype(np.float32)
            probs_all[start:end] = probs

    # 7) 추가 필터를 위한 준비
    #    - 연속 신호 카운터
    #    - 기대 수익률(%) 계산을 위한 과거 평균 변동폭
    total_cost_rate = max(0.0, fee_rate + slippage_rate)
    # horizon 뒤 실제 움직임을 기준으로 한 절대 변동률 (entry 대비)
    move_pct_all = np.abs(exit_prices - entry_prices) / np.clip(entry_prices, 1e-6, None)
    # 기대 변동률 계산을 위한 롤링 윈도우 크기 (대략 horizon 초)
    exp_window = max(horizon, 10)

    consec_long = 0
    consec_short = 0
    next_available_idx = 0  # single_position 모드에서 다음 포지션을 열 수 있는 최소 시퀀스 인덱스

    total_trades = 0
    win_trades = 0
    long_trades = 0
    short_trades = 0
    net_profit = 0.0  # in quote currency (USDT assumed)
    gross_profit_sum = 0.0
    cost_sum = 0.0

    for i in range(total_sequences):
        p = probs_all[i]

        # 포지션 하나만 유지하는 모드에서는 아직 포지션이 끝나지 않은 기간에는 신호를 무시
        if single_position and i < next_available_idx:
            continue

        # 연속 신호 카운터 업데이트
        if p >= prob_threshold:
            consec_long += 1
        else:
            consec_long = 0

        if p <= prob_short_threshold:
            consec_short += 1
        else:
            consec_short = 0

        side = None  # "long" or "short"
        if strategy == "long":
            # 롱 전략: 상승 확률이 threshold 이상이고, 연속 신호 조건을 만족할 때만 진입
            if p >= prob_threshold and consec_long >= min_consecutive:
                side = "long"
        elif strategy == "long-short":
            # long-short 전략: 상승/하락 각각에 대해 연속 신호 조건을 적용
            if p >= prob_threshold and consec_long >= min_consecutive:
                side = "long"
            elif p <= prob_short_threshold and consec_short >= min_consecutive:
                side = "short"
        else:
            # 알 수 없는 전략 이름이면 아무 트레이드도 하지 않는다.
            continue

        if side is None:
            continue

        entry = float(entry_prices[i])
        exit = float(exit_prices[i])

        # 기대 수익률 필터:
        # 과거 exp_window 개 시퀀스의 평균 절대 변동률을 이용하여
        # (예상변동률 * 방향확률 - 비용률*2)이 min_expected_pct(%) 이상일 때만 진입
        window_start = max(0, i - exp_window + 1)
        avg_move_pct = float(move_pct_all[window_start : i + 1].mean())
        if avg_move_pct <= 0:
            continue

        if side == "long":
            dir_prob = float(p)
        else:  # short
            dir_prob = float(1.0 - p)

        expected_net_pct = (dir_prob * avg_move_pct - total_cost_rate * 2.0) * 100.0
        if expected_net_pct < min_expected_pct:
            # 기대 순수익률이 기준에 미달하면 진입하지 않는다.
            continue

        # gross profit for long/short 1 * position_size
        if side == "long":
            gross = (exit - entry) * position_size
            long_trades += 1
        else:  # short
            gross = (entry - exit) * position_size
            short_trades += 1

        # approximate roundtrip cost as (fee+slippage) * notional on entry+exit
        roundtrip_cost = total_cost_rate * (entry + exit) * position_size
        trade_net = gross - roundtrip_cost

        net_profit += trade_net
        gross_profit_sum += gross
        cost_sum += roundtrip_cost
        total_trades += 1
        if trade_net > 0:
            win_trades += 1

        # single_position 모드에서는 현재 포지션이 끝나는 시점까지 신규 진입을 막는다.
        if single_position:
            next_available_idx = i + horizon

    # 8) 통계 출력
    print()
    print(f"=== +{horizon}s backtest profit evaluation ===")
    print(f"- File: {csv_path.name}")
    print(f"- Total sequences available: {total_sequences}")
    print(f"- Threshold: prob_up >= {prob_threshold:.2f}")
    if strategy == "long-short":
        print(f"- Short threshold: prob_up <= {prob_short_threshold:.2f}")
    print(f"- Strategy: {strategy}")
    print(f"- Min consecutive signals: {min_consecutive}")
    print(f"- Min expected net pct: {min_expected_pct:.4f}%")
    print(f"- Position size: {position_size:.4f} BTC (가정)")
    print(f"- Fee rate: {fee_rate:.4f}, Slippage rate: {slippage_rate:.4f}")
    print(f"- Trades executed: {total_trades}")
    if total_trades > 0:
        win_rate = win_trades / total_trades
        avg_net = net_profit / total_trades
        print(f"- Long trades:  {long_trades}")
        print(f"- Short trades: {short_trades}")
        print(f"- Win rate: {win_rate:.4f} ({win_rate * 100:.2f}%)")
        print(f"- Total gross profit: {gross_profit_sum:.4f}")
        print(f"- Total trading cost: {cost_sum:.4f}")
        print(f"- Total net profit: {net_profit:.4f}")
        print(f"- Avg net profit per trade: {avg_net:.4f}")
    else:
        print("- No trades executed at this threshold.")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backtest profit of +N seconds ahead (default 60s) direction predictions on a daily validation CSV."
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
    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
        help="Prediction horizon in seconds (default: 60s = 1 minute).",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.6,
        help="매수 신호로 간주할 최소 상승 확률 (예: 0.6 = 60%).",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0006,
        help="한 번 체결될 때 수수료 비율 (예: 0.0006 = 0.06%).",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0003,
        help="한 번 체결될 때 슬리피지 비율 (예: 0.0003 = 0.03%).",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=1.0,
        help="한 번 진입할 때 매수하는 BTC 수량 (기본 1 BTC).",
    )
    parser.add_argument(
        "--strategy",
        choices=["long", "long-short"],
        default="long",
        help="매매 전략 유형: long 또는 long-short.",
    )
    parser.add_argument(
        "--prob-short-threshold",
        type=float,
        default=0.4,
        help="long-short 전략에서 숏 신호로 간주할 최대 상승 확률 (예: 0.4 = 40%).",
    )
    parser.add_argument(
        "--min-consecutive",
        type=int,
        default=1,
        help="동일 방향 연속 신호 개수 (예: 3이면 3초 연속일 때만 진입).",
    )
    parser.add_argument(
        "--min-expected-pct",
        type=float,
        default=0.0,
        help="예상 순수익률(%)이 이 값 이상일 때만 진입 (예: 0.05 = 0.05%).",
    )
    parser.add_argument(
        "--single-position",
        action="store_true",
        help="포지션 하나만 유지 (포지션 보유 중에는 새로운 신호를 무시).",
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
    print(f"[eval-profit] Using device: {device}")

    evaluate_profit(
        csv_path=csv_path,
        model_path=args.model,
        batch_size=args.batch_size,
        horizon=args.horizon,
        prob_threshold=args.prob_threshold,
        prob_short_threshold=args.prob_short_threshold,
        strategy=args.strategy,
        min_consecutive=args.min_consecutive,
        min_expected_pct=args.min_expected_pct,
        single_position=args.single_position,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        position_size=args.position_size,
        device=device,
    )


if __name__ == "__main__":
    main()
