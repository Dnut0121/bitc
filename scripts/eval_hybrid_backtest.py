#!/usr/bin/env python3
"""Evaluate a saved hybrid (LSTM + XGBoost) ensemble on a CSV and run a simple backtest.

Usage example:
  python scripts/eval_hybrid_backtest.py \
    --csv dataset/validate/BTCUSDT-1m-2025-01-01.csv \
    --checkpoint models/hybrid_ensemble_1m \
    --resample none \
    --prob-long 0.55 --prob-short 0.45
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

# Allow running without installing as a package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import run_event_backtest
from src.data import load_ohlcv
from src.features import add_microstructure_features, add_technical_features
from src.hybrid_model import load_hybrid_checkpoint
from src.model import build_sequences
from src.pressure import compute_pressure


def _clean_resample_arg(raw: str | None) -> str | None:
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in {"none", "null", "off", ""}:
        return None
    return raw


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a hybrid ensemble on a CSV and backtest.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV path with OHLCV (or resampled) data.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/hybrid_ensemble_1m"),
        help="Directory containing lstm.pt, xgb.json, meta.json.",
    )
    parser.add_argument(
        "--resample",
        default="none",
        help="Pandas resample rule (e.g., 1s). Use 'none' to keep the CSV interval as-is.",
    )
    parser.add_argument(
        "--prob-long",
        type=float,
        default=0.55,
        help="Threshold to enter long (prob_up >= threshold).",
    )
    parser.add_argument(
        "--prob-short",
        type=float,
        default=0.45,
        help="Threshold to enter short (prob_up <= threshold).",
    )
    parser.add_argument(
        "--tail-rows",
        type=int,
        default=0,
        help="Use only the last N rows from CSV (0 = use all).",
    )
    parser.add_argument(
        "--infer-batch",
        type=int,
        default=2048,
        help="Batch size for LSTM inference to control memory usage.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU for LSTM inference.",
    )
    args = parser.parse_args(argv)

    resample_rule = _clean_resample_arg(args.resample)
    tail_rows = None if args.tail_rows <= 0 else args.tail_rows

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[eval] device={device} resample={resample_rule or 'none'} csv={args.csv}")

    # 1) Load and feature-engineer
    df = load_ohlcv(args.csv, tail_rows=tail_rows, resample_rule=resample_rule)
    df = add_technical_features(add_microstructure_features(compute_pressure(df)))

    # 2) Load checkpoint/meta
    lstm_result, booster, meta = load_hybrid_checkpoint(args.checkpoint, device=None if args.cpu else device)
    feature_columns = meta["feature_columns"]
    lookback = int(meta["lookback"])
    label_horizon = int(meta["label_horizon"])
    w_lstm, w_xgb = meta.get("ensemble_weights", [0.5, 0.5])
    feature_mean = np.asarray(meta["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(meta["feature_std"], dtype=np.float32)
    std_safe = np.where(feature_std < 1e-6, 1.0, feature_std)

    # 3) Build sequences aligned with LSTM/XGB expectations
    df_clean = df.dropna(subset=[*feature_columns, "close"])
    sequences, targets, indexes = build_sequences(
        df_clean,
        lookback=lookback,
        label_horizon=label_horizon,
        feature_columns=feature_columns,
    )
    if len(sequences) == 0:
        raise RuntimeError("No sequences constructed; check lookback/label_horizon and data length.")
    print(f"[eval] sequences={len(sequences)} lookback={lookback} horizon={label_horizon}")

    # 4) Normalize sequences for LSTM
    norm_sequences = (sequences - feature_mean) / std_safe

    # 5) LSTM inference (batched)
    model = lstm_result.model
    if model is None:
        raise RuntimeError("Checkpoint does not contain an LSTM model.")
    model = model.to(device)
    model.eval()
    probs_lstm_chunks: list[np.ndarray] = []
    bs = max(1, int(args.infer_batch))
    with torch.no_grad():
        for start in range(0, len(norm_sequences), bs):
            batch_np = norm_sequences[start : start + bs].astype(np.float32, copy=False)
            batch = torch.from_numpy(batch_np).to(device)
            logits = model(batch)
            probs_lstm_chunks.append(torch.sigmoid(logits).cpu().numpy())
    probs_lstm = np.concatenate(probs_lstm_chunks)

    # 6) XGBoost inference
    feature_matrix = df_clean.reindex(indexes)[feature_columns].to_numpy(dtype=np.float32)
    feature_matrix = (feature_matrix - feature_mean) / std_safe
    dmatrix = xgb.DMatrix(feature_matrix)
    best_it = meta.get("xgb_best_iteration", None)
    iter_range = None
    if best_it is not None:
        iter_range = (0, int(best_it) + 1)
    probs_xgb = booster.predict(dmatrix, iteration_range=iter_range)

    # 7) Ensemble
    w_sum = w_lstm + w_xgb
    w_lstm_norm = w_lstm / w_sum if w_sum > 0 else 0.5
    w_xgb_norm = w_xgb / w_sum if w_sum > 0 else 0.5
    probs_ens = w_lstm_norm * probs_lstm + w_xgb_norm * probs_xgb

    # 8) Backtest on ensemble prob_up
    eval_df = df_clean.reindex(indexes).copy()
    eval_df["prob_up"] = probs_ens
    bt = run_event_backtest(
        eval_df,
        window=lookback,
        prob_threshold_long=args.prob_long,
        prob_threshold_short=args.prob_short,
        horizon=label_horizon,
        fee_rate=0.0006,
        slippage_rate=0.0003,
        record_trades=True,
    )

    print("=== Hybrid ensemble backtest ===")
    print(
        f"trades={bt.num_trades} total_return={bt.total_return:.4f} "
        f"sharpe={bt.sharpe:.3f} hit_ratio={bt.hit_ratio:.3f} "
        f"max_dd={bt.max_drawdown:.4f} profit_factor={bt.profit_factor:.3f}"
    )
    if bt.trades:
        print("\nRecent trades (tail=5):")
        for tr in bt.trades[-5:]:
            print(
                f"{tr.entry_time} -> {tr.exit_time}  {tr.direction} "
                f"entry={tr.entry_price:.4f} exit={tr.exit_price:.4f} "
                f"tp={tr.tp_price:.4f} sl={tr.sl_price:.4f} "
                f"pnl={tr.net_pnl:.6f} r={tr.r_multiple:.3f} reason={tr.exit_reason}"
            )
    print("\nTail predictions (ensemble):")
    preview_cols = ["close", "prob_up"]
    print(eval_df[preview_cols].tail(5).to_string())


if __name__ == "__main__":
    main()
