"""Command-line orchestrator that wires download, preprocessing, and LSTM prediction."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .data import download_per_second_ohlcv, load_ohlcv
from .model import ModelResult, build_sequences, train_lstm
from .pressure import compute_pressure


def prepare_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        return load_ohlcv(args.csv, tail_rows=getattr(args, "tail_rows", None))

    if args.cache and args.cache.exists() and not args.force_download:
        return load_ohlcv(args.cache, tail_rows=getattr(args, "tail_rows", None))

    if not args.start or not args.end:
        raise ValueError("When not reusing a CSV file you must supply --start and --end timestamps.")
    return download_per_second_ohlcv(args.symbol, args.start, args.end, cache_path=args.cache)


def plot_summary(result: ModelResult, window: int, limit: int = 300) -> None:
    data = result.data.tail(limit)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(data.index, data["close"], label="Price")
    axes[0].set_title("Close Price with Aggressive Pressure")
    axes[0].legend()

    axes[1].plot(data.index, data["buy_ratio"], label="Buy ratio")
    axes[1].plot(data.index, data["signal"], label="Rolling signal", alpha=0.6)
    axes[1].set_title(f"Buy ratio and {window}s rolling imbalance")
    axes[1].legend()

    axes[2].plot(data.index, data["prob_up"], label="LSTM prob up", color="tab:green")
    axes[2].plot(data.index, data["pred_direction"], label="Predicted direction", alpha=0.8)
    axes[2].set_title("Next-second direction prediction")
    axes[2].legend()

    fig.tight_layout()
    plt.show()


def print_snapshot(result: ModelResult, window: int) -> None:
    latest = result.data.iloc[-1]
    print(f"Signal window: {window}s rolling imbalance (LSTM forecast)")
    print(f"Latest candle @ {latest.name}")
    print(f"  Buy pressure: {latest['buy_pressure']:.6f}")
    print(f"  Sell pressure: {latest['sell_pressure']:.6f}")
    print(f"  Buy ratio: {latest['buy_ratio']:.3f}")
    print(f"  Imbalance: {latest['imbalance']:.6f}")
    print(f"  Predicted probability of up: {latest['prob_up']:.3f}")
    print(f"  Val accuracy: {result.accuracy:.3f}, Val loss: {result.loss:.3f}")
    print("Recent signal direction (1=buys dominated, -1=sells dominated):")
    print(result.data["signal"].tail(5).to_string())


def suggest_trades(
    result: ModelResult,
    window: int,
    prob_threshold: float,
    horizon: int,
    risk_reward: float,
    fee_rate: float,
    slippage_rate: float,
    min_expected_pct: float,
) -> None:
    """과거 데이터와 현재 시점을 기준으로 단순 매수/매도 시나리오를 출력합니다.

    prob_threshold   : 매수 후보로 볼 최소 상승 확률
    horizon          : 최대 보유 시간(초)
    risk_reward      : 손익비 (예: 2.0 = 2:1)
    fee_rate         : 한 번 체결될 때 수수료 비율 (0.0005 = 0.05%)
    slippage_rate    : 한 번 체결될 때 슬리피지 비율
    min_expected_pct : 현재 시점 기대 수익률이 이 값(%) 이상일 때만 “의미 있는 기회”로 간주
    """
    df = result.data
    if "prob_up" not in df.columns or "close" not in df.columns:
        print("trade suggestion: prob_up 또는 close 컬럼이 없어 추천을 계산할 수 없습니다.")
        return

    # 1) 과거 데이터 기준: 높은 상승 확률 구간에서 최대 horizon 초 보유했을 때
    #    수수료/슬리피지를 반영한 순익 기준으로 가장 좋은 트레이드 찾기
    df_reset = df.reset_index()  # timestamp 인덱스를 컬럼으로 유지
    timestamp_col = df_reset.columns[0]
    candidates = df_reset[df_reset["prob_up"] >= prob_threshold].copy()

    total_cost_rate = max(0.0, fee_rate + slippage_rate)
    best_trade: dict | None = None
    for start_idx_obj in candidates.index:
        start_idx: int = int(start_idx_obj)
        if start_idx + 1 >= len(df_reset):
            continue
        end_idx = min(len(df_reset) - 1, start_idx + horizon)
        future = df_reset.iloc[start_idx + 1 : end_idx + 1]
        if future.empty:
            continue
        # 미래 구간에서 최고 종가가 되는 위치(상대 위치)를 찾는다.
        rel_best_pos: int = int(future["close"].to_numpy().argmax())
        best_row = future.iloc[rel_best_pos]
        entry_row = df_reset.iloc[start_idx]
        entry_price = float(entry_row["close"])
        exit_price = float(best_row["close"])
        gross_profit = exit_price - entry_price
        # 왕복(진입+청산) 시 수수료+슬리피지 비용을 단순히 가격 비율로 차감
        roundtrip_cost = total_cost_rate * (entry_price + exit_price)
        net_profit = gross_profit - roundtrip_cost
        net_profit_pct = net_profit / entry_price * 100.0 if entry_price != 0 else 0.0
        if best_trade is None or net_profit_pct > best_trade["net_profit_pct"]:
            best_trade = {
                "entry_time": entry_row[timestamp_col],
                "entry_price": entry_price,
                "exit_time": best_row[timestamp_col],
                "exit_price": exit_price,
                "profit": net_profit,
                "profit_pct": net_profit_pct,
            }

    print()
    print("=== 과거 데이터 기준 매수/매도 기회 (단순 시뮬레이션, 수수료/슬리피지 반영) ===")
    if best_trade is None:
        print(f"- prob_up >= {prob_threshold:.2f} 조건을 만족하는 고확률 매수 구간을 찾지 못했습니다.")
    else:
        print(f"- 조건: prob_up >= {prob_threshold:.2f}, 보유 기간 최대 {horizon}초")
        print(f"        손익비(목표:손절) ≈ {risk_reward:.2f}:1, 수수료+슬리피지 비율 ≈ {total_cost_rate:.4f}")
        print(f"  진입 시점: {best_trade['entry_time']}, 가격: {best_trade['entry_price']:.2f}")
        print(f"  청산 시점: {best_trade['exit_time']}, 가격: {best_trade['exit_price']:.2f}")
        print(f"  실현 수익: {best_trade['profit']:.2f} ({best_trade['profit_pct']:.2f}%)")

    # 2) 현재 시점 기준: 마지막 캔들을 지금 매수한다고 가정한 단순 기대 수익
    latest = df.iloc[-1]
    entry_price = float(latest["close"])
    prob_up = float(latest["prob_up"])
    # 최근 window초 평균 변동폭을 이용해 목표/손절 폭을 가정
    if "range" in df.columns:
        avg_range = float(df["range"].rolling(window, min_periods=1).mean().iloc[-1])
    else:
        avg_range = float((df["high"] - df["low"]).rolling(window, min_periods=1).mean().iloc[-1])
    rr = max(risk_reward, 0.1)
    target_move = rr * avg_range
    stop_move = avg_range
    target_price = entry_price + target_move
    stop_price = entry_price - stop_move
    # 수수료/슬리피지 포함 기대값 (목표/손절 각각 다른 가격에서 비용 계산)
    target_cost = total_cost_rate * (entry_price + target_price)
    stop_cost = total_cost_rate * (entry_price + stop_price)
    expected_profit = prob_up * (target_move - target_cost) - (1.0 - prob_up) * (stop_move + stop_cost)
    expected_profit_pct = expected_profit / entry_price * 100.0 if entry_price != 0 else 0.0

    print()
    print("=== 현재 시점 가정 매수/매도 시나리오 (단순 모델 기반, 수수료/슬리피지 반영) ===")
    print(f"- 현재 캔들 시점: {latest.name}, 종가(가정 매수가): {entry_price:.2f}")
    print(f"  다음 1초 상승 확률(prob_up): {prob_up:.3f}")
    print(f"  최근 {window}초 평균 변동폭(range): {avg_range:.6f}")
    print(f"  손익비(목표:손절): {rr:.2f}:1")
    print(f"  가정 목표가(위): {target_price:.2f}")
    print(f"  가정 손절가(아래): {stop_price:.2f}")
    print(f"  수수료+슬리피지 비율(왕복 기준 근사): {total_cost_rate:.4f}")
    print(f"  단순 기대 수익(수수료/슬리피지 반영): {expected_profit:.6f} ({expected_profit_pct:.4f}%)")
    if expected_profit_pct < min_expected_pct:
        print(f"  → 기대 수익률이 설정한 최소값({min_expected_pct:.2f}%)보다 낮아서, 보수적으로 보면 지금은 매수 신호가 약한 편입니다.")
    print("  ※ 위 수치는 매우 단순한 가정이며, 실제 매매 판단은 포지션 사이즈, 리스크 관리, 마켓 컨디션 등을 함께 고려해야 합니다.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Deep-learning driven aggressive pressure signal from Binance 1-second candles."
    )
    parser.add_argument("csv", nargs="?", type=Path, help="Existing per-second OHLCV CSV file to reuse.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Market symbol to download via aggTrades.")
    parser.add_argument("--start", help="Start time for download (ISO or epoch in ms).")
    parser.add_argument("--end", help="End time for download (ISO or epoch in ms).")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/binance_ohlcv.csv"),
        help="Cached path to store downloaded per-second candles.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached CSV and always fetch fresh data.",
    )
    parser.add_argument("--window", type=int, default=60, help="Lookback in seconds for imbalance signal.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for the LSTM.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max norm for gradient clipping.")
    parser.add_argument("--plot", action="store_true", help="Show plots at the end.")
    parser.add_argument(
        "--tail-rows",
        type=int,
        help="When supplied, load only the last N rows from the cached CSV (useful for testing).",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.6,
        help="매수 후보로 볼 최소 상승 확률 (예: 0.6 = 60%).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=60,
        help="과거 시뮬레이션에서 한 트레이드를 최대 몇 초까지 보유할지.",
    )
    parser.add_argument(
        "--risk-reward",
        type=float,
        default=1.0,
        help="손익비 (목표:손절 비율, 예: 2.0 = 2:1).",
    )
    parser.add_argument(
        "--min-expected-pct",
        type=float,
        default=0.0,
        help="현재 시점 기대 수익률이 이 값(%) 이상일 때만 의미 있는 기회로 간주.",
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
    args = parser.parse_args(list(argv) if argv is not None else None)
    # configure logging for CLI
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
    logger = logging.getLogger("bitc.cli")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"기기 사용: {device}")

    df = prepare_dataframe(args)
    pressured = compute_pressure(df)
    sequences, targets, indexes = build_sequences(pressured, args.window)
    result = train_lstm(
        sequences,
        targets,
        indexes,
        pressured,
        args.window,
        args.epochs,
        args.batch_size,
        args.lr,
        device=device,
        grad_clip=args.grad_clip,
    )
    print_snapshot(result, args.window)
    suggest_trades(
        result,
        args.window,
        prob_threshold=args.prob_threshold,
        horizon=args.horizon,
        risk_reward=args.risk_reward,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        min_expected_pct=args.min_expected_pct,
    )
    if args.plot:
        plot_summary(result, args.window)
