#!/usr/bin/env python3
"""Train an LSTM + XGBoost ensemble with richer technical/momentum features."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch

# Local imports
from src.data import load_ohlcv
from src.features import add_microstructure_features, add_technical_features
from src.hybrid_model import (
    DEFAULT_HYBRID_FEATURES,
    save_hybrid_checkpoint,
    train_hybrid_ensemble,
)
from src.pressure import compute_pressure


def build_feature_frame(csv_path: Path, tail_rows: int | None):
    ohlcv = load_ohlcv(csv_path, tail_rows=tail_rows)
    pressured = compute_pressure(ohlcv)
    micro = add_microstructure_features(pressured)
    technical = add_technical_features(micro)
    return technical


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hybrid LSTM + XGBoost ensemble trainer with technical factors.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/binance_ohlcv.csv"),
        help="Consolidated OHLCV CSV path.",
    )
    parser.add_argument(
        "--tail-rows",
        type=int,
        default=400_000,
        help="Use only the last N rows from the CSV to keep training manageable.",
    )
    parser.add_argument("--lookback", type=int, default=120, help="Sequence length for the LSTM.")
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=60,
        help="Prediction horizon in seconds (used for both LSTM and XGBoost labels).",
    )
    parser.add_argument("--epochs", type=int, default=3, help="LSTM training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="LSTM batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="LSTM learning rate.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="LSTM gradient clipping norm.")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=300_000,
        help="Cap the number of training sequences to manage memory/compute.",
    )
    parser.add_argument("--xgb-rounds", type=int, default=300, help="XGBoost boosting rounds.")
    parser.add_argument("--xgb-depth", type=int, default=4, help="XGBoost max tree depth.")
    parser.add_argument("--xgb-eta", type=float, default=0.05, help="XGBoost learning rate (eta).")
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=0.8,
        help="XGBoost subsample ratio for rows.",
    )
    parser.add_argument(
        "--xgb-colsample",
        type=float,
        default=0.8,
        help="XGBoost colsample_bytree ratio for columns.",
    )
    parser.add_argument(
        "--ensemble-weight-lstm",
        type=float,
        default=0.55,
        help="Weight for LSTM probability in the ensemble vote.",
    )
    parser.add_argument(
        "--ensemble-weight-xgb",
        type=float,
        default=0.45,
        help="Weight for XGBoost probability in the ensemble vote.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("models/hybrid_ensemble"),
        help="Directory to save the hybrid checkpoint (lstm.pt + xgb.json + meta.json).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing checkpoints to disk.",
    )
    parser.add_argument(
        "--cost-aware-label",
        action="store_true",
        help="Use cost-aware labels (fee/slippage margin) instead of simple up/down.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.0006,
        help="Per-fill fee rate used when cost-aware labels are enabled.",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0003,
        help="Per-fill slippage rate used when cost-aware labels are enabled.",
    )
    parser.add_argument(
        "--margin-rate",
        type=float,
        default=0.0,
        help="Additional margin threshold for cost-aware labels.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU for LSTM training even when CUDA is available.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    device = torch.device("cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[hybrid] Using device: {device}")

    feature_df = build_feature_frame(args.csv, tail_rows=args.tail_rows)
    print(f"[hybrid] Loaded {len(feature_df)} rows from {args.csv} (after feature engineering).")

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": args.xgb_depth,
        "eta": args.xgb_eta,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "verbosity": 0,
    }

    try:
        result = train_hybrid_ensemble(
            feature_df,
            feature_columns=DEFAULT_HYBRID_FEATURES,
            lookback=args.lookback,
            label_horizon=args.label_horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            grad_clip=args.grad_clip,
            device=device,
            max_sequences=args.max_sequences,
            use_cost_labels=args.cost_aware_label,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            margin_rate=args.margin_rate,
            xgb_params=xgb_params,
            xgb_rounds=args.xgb_rounds,
            ensemble_weights=(args.ensemble_weight_lstm, args.ensemble_weight_xgb),
        )
    except RuntimeError as exc:
        if "xgboost" in str(exc).lower():
            print("[hybrid] xgboost is missing. Install it inside your virtualenv: `pip install xgboost`.")
        raise

    print()
    print("=== Hybrid training summary ===")
    print(f"- LSTM val_acc: {result.lstm_result.accuracy:.4f}  val_loss: {result.lstm_result.loss:.4f}")
    if result.xgb_val_accuracy is not None:
        print(f"- XGBoost val_acc: {result.xgb_val_accuracy:.4f}  best_iteration: {result.xgb_best_iteration}")
    if result.xgb_val_logloss is not None:
        print(f"- XGBoost val_logloss: {result.xgb_val_logloss:.4f}")
    if result.ensemble_val_accuracy is not None:
        print(f"- Ensemble val_acc (weighted vote): {result.ensemble_val_accuracy:.4f}")
    else:
        print("- Ensemble val_acc: n/a (not enough samples for validation split)")

    print("\nRecent predictions (tail=5):")
    preview_cols = ["close", "prob_up", "prob_up_xgb", "prob_up_ensemble", "pred_ensemble", "target"]
    print(result.data[preview_cols].tail(5).to_string())

    if not args.no_save and args.save_dir:
        save_hybrid_checkpoint(args.save_dir, result)
        print(f"\n[hybrid] Saved hybrid checkpoint to {args.save_dir} (lstm.pt, xgb.json, meta.json).")
    else:
        print("\n[hybrid] Skipped saving checkpoint (requested by flag).")


if __name__ == "__main__":
    main()
