#!/usr/bin/env python3
"""Grid-search style backtest of base LSTM strategies on validate data.

1) 하루치 validate CSV를 불러와서 prob_up 시계열을 계산한다.
2) 여러 quantile 기반 threshold(q_long/q_short)와 (horizon, risk_reward) 조합에 대해
   run_quantile_long_short_backtest를 실행한다.
3) 각 조합에 대한 Sharpe, 승률, 트레이드 수 등을 표 형태로 출력한다.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
import sys

import numpy as np
import pandas as pd
import torch

# 프로젝트 루트(src 패키지)를 import 경로에 추가
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader, Dataset

from src.data import load_ohlcv
from src.pressure import compute_pressure
from src.model import build_sequences, load_model_checkpoint
from src.features import add_microstructure_features
from src.backtest import run_quantile_long_short_backtest, TradeRecord
from torch import nn


def _extract_validate_date(path: Path) -> datetime:
    stem = path.stem  # BTCUSDT-1s-YYYY-MM-DD
    parts = stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse date from filename: {path.name}")
    date_str = "-".join(parts[-3:])
    return datetime.fromisoformat(date_str)


def compute_prob_up_series(
    ohlcv: pd.DataFrame,
    model_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    pressured = compute_pressure(ohlcv)
    model, feature_mean, feature_std, lookback = load_model_checkpoint(model_path, device=device)

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
    else:
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

    class _NumpyDataset(Dataset):
        def __init__(self, data: np.ndarray) -> None:
            self.data = torch.from_numpy(data.astype(np.float32))

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.data[idx]

    dataset = _NumpyDataset(normalized)
    loader = DataLoader(dataset, batch_size=512)

    model.eval()
    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
    prob_up = np.concatenate(probs_list, axis=0)

    prob_df = pressured.reindex(indexes).copy()
    prob_df["prob_up"] = prob_up
    return prob_df


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backtest multiple base LSTM strategies with quantile-based thresholds on validate data."
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
        help="Specific validate CSV to use. When omitted, runs on all files in validate-dir.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("models/btc_lstm.pt"),
        help="Base LSTM checkpoint path.",
    )
    parser.add_argument(
        "--long-quantiles",
        type=float,
        nargs="+",
        default=[0.6, 0.7],
        help="Long entry quantiles for prob_up (e.g., 0.6 0.7).",
    )
    parser.add_argument(
        "--short-quantiles",
        type=float,
        nargs="+",
        default=[0.4, 0.3],
        help="Short entry quantiles for prob_up (mirrors long-quantiles).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[60, 120, 180],
        help="Holding horizons (seconds) to test.",
    )
    parser.add_argument(
        "--risk-rewards",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0],
        help="Risk-reward ratios (TP:SL) to test.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Lookback window for range-based filters/TP/SL.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0006,
        help="Per-trade fee rate.",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0003,
        help="Per-trade slippage rate.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print first trade details for each strategy combination (for debugging).",
    )
    parser.add_argument(
        "--meta-thresholds",
        type=float,
        nargs="+",
        help="List of meta thresholds to grid search (overrides --meta-threshold when set).",
    )
    parser.add_argument(
        "--min-entry-gaps",
        type=int,
        nargs="+",
        default=[0],
        help="Minimum bars between entries (grid search list; default=0).",
    )
    parser.add_argument(
        "--ma-window",
        type=int,
        default=None,
        help="Optional moving-average window (bars) filter; long above MA, short below.",
    )
    parser.add_argument(
        "--rsi-window",
        type=int,
        default=None,
        help="Optional RSI window (bars) filter.",
    )
    parser.add_argument(
        "--rsi-overbought",
        type=float,
        default=None,
        help="RSI overbought threshold for short entries.",
    )
    parser.add_argument(
        "--rsi-oversold",
        type=float,
        default=None,
        help="RSI oversold threshold for long entries.",
    )
    parser.add_argument(
        "--meta-model",
        type=Path,
        help="Optional meta-model checkpoint (trained via run_meta_pipeline) to filter entries (long/short).",
    )
    parser.add_argument(
        "--meta-threshold",
        type=float,
        default=None,
        help="Meta-model probability threshold; defaults to 0.5 or checkpoint meta_threshold if present.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.base_model.exists():
        raise FileNotFoundError(f"Base LSTM checkpoint not found: {args.base_model}")

    if len(args.long_quantiles) != len(args.short_quantiles):
        raise ValueError("long-quantiles and short-quantiles must have the same length.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[base-backtest] Using device: {device}")
    print(f"[base-backtest] Base model: {args.base_model}")
    if args.meta_model:
        print(f"[base-backtest] Meta model: {args.meta_model}")

    if args.file is not None:
        files = [args.file]
    else:
        files = sorted(args.validate_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found under {args.validate_dir}")

    for csv_path in files:
        if not csv_path.exists():
            continue
        dt = _extract_validate_date(csv_path)
        print()
        print(f"=== Backtest on {csv_path.name} (date={dt.date()}) ===")
        ohlcv = load_ohlcv(csv_path)
        prob_df = compute_prob_up_series(ohlcv, args.base_model, device=device)
        meta_prob = None
        meta_prob_long = None
        meta_prob_short = None
        meta_feature_cols: list[str] | None = None
        base_thr_from_meta: float | None = None
        base_short_thr_from_meta: float | None = None
        meta_threshold = args.meta_threshold
        meta_thresholds: list[float] | None = None

        if args.meta_model:
            if not args.meta_model.exists():
                print(f"[base-backtest] WARNING: meta-model not found: {args.meta_model}; skipping meta filter.")
            else:
                ckpt = torch.load(args.meta_model, map_location=device)
                meta_feature_cols = list(ckpt.get("feature_cols", []))
                base_thr_from_meta = float(ckpt.get("base_long_threshold", ckpt.get("base_threshold", 0.5)))
                base_short_thr_from_meta = float(
                    ckpt.get("base_short_threshold", 1.0 - base_thr_from_meta if base_thr_from_meta else 0.4)
                )
                meta_model = nn.Linear(len(meta_feature_cols), 1).to(device)
                meta_model.load_state_dict(ckpt["state_dict"])
                meta_model.eval()

                df_feat = add_microstructure_features(prob_df)
                df_feat = df_feat.reindex(prob_df.index).copy()
                df_feat["prob_up"] = prob_df["prob_up"]
                if "direction" in meta_feature_cols and "direction" not in df_feat.columns:
                    df_feat["direction"] = 0.0
                missing = [c for c in meta_feature_cols if c not in df_feat.columns]
                if missing:
                    print(f"[base-backtest] WARNING: missing meta features {missing}; meta filter disabled.")
                else:
                    # 결측이 있으면 간단히 보간/0으로 메우고 메타 확률 계산
                    df_filled = df_feat.copy()
                    df_filled[meta_feature_cols] = (
                        df_filled[meta_feature_cols].ffill().bfill().fillna(0.0)
                    )

                    # 메타 모델 피처에 direction이 포함되어 있으면 롱/숏 별로 각각 점수 계산
                    uses_direction = "direction" in meta_feature_cols
                    if uses_direction:
                        df_long = df_filled.copy()
                        df_long["direction"] = 1.0
                        X_long = df_long[meta_feature_cols].to_numpy(dtype=np.float32)

                        df_short = df_filled.copy()
                        df_short["direction"] = -1.0
                        X_short = df_short[meta_feature_cols].to_numpy(dtype=np.float32)

                        with torch.no_grad():
                            logits_long = meta_model(torch.from_numpy(X_long).to(device)).squeeze(-1)
                            logits_short = meta_model(torch.from_numpy(X_short).to(device)).squeeze(-1)
                        meta_prob_long = torch.sigmoid(logits_long).cpu().numpy()
                        meta_prob_short = torch.sigmoid(logits_short).cpu().numpy()
                        prob_df = prob_df.copy()
                        prob_df["meta_prob_long"] = meta_prob_long
                        prob_df["meta_prob_short"] = meta_prob_short
                        msg_stats = (
                            f"meta_long min={meta_prob_long.min():.4f} max={meta_prob_long.max():.4f} "
                            f"mean={meta_prob_long.mean():.4f}; "
                            f"meta_short min={meta_prob_short.min():.4f} max={meta_prob_short.max():.4f} "
                            f"mean={meta_prob_short.mean():.4f}"
                        )
                    else:
                        X_meta = df_filled[meta_feature_cols].to_numpy(dtype=np.float32)
                        with torch.no_grad():
                            logits = meta_model(torch.from_numpy(X_meta).to(device)).squeeze(-1)
                        meta_prob = torch.sigmoid(logits).cpu().numpy()
                        prob_df = prob_df.copy()
                        prob_df["meta_prob"] = meta_prob
                        msg_stats = (
                            f"meta min={meta_prob.min():.4f} max={meta_prob.max():.4f} "
                            f"mean={meta_prob.mean():.4f}"
                        )
                    meta_thr_default = args.meta_threshold
                    if meta_thr_default is None:
                        meta_thr_default = float(ckpt.get("meta_threshold", 0.5))
                    meta_threshold = meta_thr_default
                    if args.meta_thresholds:
                        meta_thresholds = list(args.meta_thresholds)
                    else:
                        meta_thresholds = [meta_thr_default]
                    print(
                        f"[base-backtest] {msg_stats} "
                        f"(thr={meta_threshold:.4f}, base_long_thr_from_meta={base_thr_from_meta}, "
                        f"base_short_thr_from_meta={base_short_thr_from_meta})"
                    )
                meta_model = None

        # meta threshold grid fallback
        if meta_thresholds is None:
            if hasattr(args, 'meta_thresholds') and args.meta_thresholds:
                meta_thresholds = list(args.meta_thresholds)
            elif meta_threshold is not None:
                meta_thresholds = [meta_threshold]
            else:
                meta_thresholds = [0.5]

        rows: list[dict[str, object]] = []
        for q_long, q_short in zip(args.long_quantiles, args.short_quantiles):
            for horizon in args.horizons:
                for rr in args.risk_rewards:
                    for meta_thr_val in meta_thresholds:
                        for entry_gap in args.min_entry_gaps:
                            try:
                                bt = run_quantile_long_short_backtest(
                                    prob_df,
                                    window=args.window,
                                    horizon=horizon,
                                    risk_reward=rr,
                                    fee_rate=args.fee_rate,
                                    slippage_rate=args.slippage_rate,
                                    q_long=q_long,
                                    q_short=q_short,
                                    min_range_quantile=0.1,
                                    min_volume_quantile=0.1,
                                    consistency_lookback=5,
                                    meta_prob=meta_prob,
                                    meta_prob_long=meta_prob_long,
                                    meta_prob_short=meta_prob_short,
                                    meta_threshold=meta_thr_val,
                                    min_entry_gap=entry_gap,
                                    ma_window=args.ma_window,
                                    rsi_window=args.rsi_window,
                                    rsi_overbought=args.rsi_overbought,
                                    rsi_oversold=args.rsi_oversold,
                                )
                            except Exception as exc:
                                print(
                                    f"[base-backtest] Skipped combo q_long={q_long} q_short={q_short} "
                                    f"horizon={horizon} rr={rr} meta_thr={meta_thr_val} gap={entry_gap}: {exc}"
                                )
                                continue
                            rows.append(
                                {
                                    "q_long": q_long,
                                    "q_short": q_short,
                                    "horizon": horizon,
                                    "risk_reward": rr,
                                    "meta_thr": meta_thr_val,
                                    "min_entry_gap": entry_gap,
                                    "ma_window": args.ma_window,
                                    "rsi_window": args.rsi_window,
                                    "total_return": bt.total_return,
                                    "sharpe": bt.sharpe,
                                    "hit_ratio": bt.hit_ratio,
                                    "num_trades": bt.num_trades,
                                    "profit_factor": bt.profit_factor,
                                }
                            )

                            if args.debug and getattr(bt, "num_trades", 0) > 0:
                                # run_quantile_long_short_backtest 기록에서 첫 트레이드를 복원
                                # TradeRecord 리스트는 별도 확장 API로만 접근 가능하므로 내부 구조를 가정한다.
                                try:
                                    trade_records: list[TradeRecord] = getattr(bt, "trades", [])  # type: ignore[attr-defined]
                                except Exception:
                                    trade_records = []
                                if trade_records:
                                    tr = trade_records[0]
                                    print("  [DEBUG first trade]")
                                    print(f"    q_long={q_long:.3f}, q_short={q_short:.3f}, horizon={horizon}, RR={rr}")
                                    print(f"    entry_time={tr.entry_time}, direction={tr.direction}, entry_price={tr.entry_price:.2f}")
                                    print(f"    tp_level={tr.tp_price:.2f}, sl_level={tr.sl_price:.2f}, horizon={tr.horizon}")
                                    print(
                                        f"    exit_time={tr.exit_time}, exit_price={tr.exit_price:.2f}, "
                                        f"exit_reason={tr.exit_reason}"
                                    )
                                    print(f"    net_pnl={tr.net_pnl:.2f}, R={tr.r_multiple:.4f}")

        if not rows:
            print("[base-backtest] No valid backtest results for this file.")
            continue

        result_df = pd.DataFrame(rows)
        # Sharpe 기준으로 상위 몇 개만 정렬해서 보여준다.
        result_df = result_df.sort_values("sharpe", ascending=False)
        print(result_df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))


if __name__ == "__main__":
    main()
