from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from src.data import load_ohlcv
from src.features import add_microstructure_features, add_technical_features
from src.hybrid_model import DEFAULT_HYBRID_FEATURES, save_hybrid_checkpoint, train_hybrid_ensemble
from src.pressure import compute_pressure


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _load_env(path: Path | None = None) -> dict[str, str]:
    """Lightweight .env reader (KEY=VALUE, ignores blank/# lines)."""
    if path is None:
        path = Path(__file__).resolve().with_name(".env")
    if not path.exists():
        return {}
    env: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key:
            env[key] = value
    return env


def build_feature_frame(csv_path: Path, tail_rows: int | None, resample_rule: str | None):
    """Load one CSV or a directory of daily CSVs, optionally resampling."""

    def _load_one(path: Path) -> pd.DataFrame:
        return load_ohlcv(path, tail_rows=None, resample_rule=resample_rule)

    if csv_path.is_dir():
        files = sorted(csv_path.rglob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found under directory: {csv_path}")
        frames: list[pd.DataFrame] = []
        for f in files:
            frames.append(_load_one(f))
        ohlcv = pd.concat(frames).sort_index()
        if tail_rows is not None:
            ohlcv = ohlcv.tail(tail_rows)
    else:
        ohlcv = load_ohlcv(csv_path, tail_rows=tail_rows, resample_rule=resample_rule)

    pressured = compute_pressure(ohlcv)
    micro = add_microstructure_features(pressured)
    return add_technical_features(micro)


def main(argv: Sequence[str] | None = None) -> None:
    env = _load_env()

    def _get_env_path(name: str, default: str) -> Path:
        value = env.get(name)
        return Path(value) if value else Path(default)

    def _get_env_int(name: str, default: int) -> int:
        value = env.get(name)
        if value is None or value == "":
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _get_env_float(name: str, default: float) -> float:
        value = env.get(name)
        if value is None or value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def _get_env_bool(name: str, default: bool) -> bool:
        value = env.get(name)
        if value is None or value == "":
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    def _get_env_str(name: str, default: str) -> str:
        value = env.get(name)
        if value is None or value == "":
            return default
        return value

    csv_default = _get_env_path("BITC_HYBRID_CSV", "dataset/binance_ohlcv.csv")
    tail_rows_default = _get_env_int("BITC_HYBRID_TAIL_ROWS", 400_000)
    lookback_default = _get_env_int("BITC_HYBRID_LOOKBACK", 120)
    label_horizon_default = _get_env_int("BITC_HYBRID_LABEL_HORIZON", 60)
    epochs_default = _get_env_int("BITC_HYBRID_EPOCHS", 3)
    batch_size_default = _get_env_int("BITC_HYBRID_BATCH_SIZE", 64)
    lr_default = _get_env_float("BITC_HYBRID_LR", 1e-3)
    grad_clip_default = _get_env_float("BITC_HYBRID_GRAD_CLIP", 1.0)
    max_sequences_default = _get_env_int("BITC_HYBRID_MAX_SEQUENCES", 300_000)
    xgb_rounds_default = _get_env_int("BITC_HYBRID_XGB_ROUNDS", 300)
    xgb_depth_default = _get_env_int("BITC_HYBRID_XGB_DEPTH", 4)
    xgb_eta_default = _get_env_float("BITC_HYBRID_XGB_ETA", 0.05)
    xgb_subsample_default = _get_env_float("BITC_HYBRID_XGB_SUBSAMPLE", 0.8)
    xgb_colsample_default = _get_env_float("BITC_HYBRID_XGB_COLSAMPLE", 0.8)
    ensemble_weight_lstm_default = _get_env_float("BITC_HYBRID_WEIGHT_LSTM", 0.55)
    ensemble_weight_xgb_default = _get_env_float("BITC_HYBRID_WEIGHT_XGB", 0.45)
    save_dir_default = _get_env_path("BITC_HYBRID_SAVE_DIR", "models/hybrid_ensemble")
    no_save_default = _get_env_bool("BITC_HYBRID_NO_SAVE", False)
    cost_aware_default = _get_env_bool("BITC_HYBRID_COST_AWARE_LABEL", False)
    fee_rate_default = _get_env_float("BITC_HYBRID_FEE_RATE", 0.0006)
    slippage_rate_default = _get_env_float("BITC_HYBRID_SLIPPAGE_RATE", 0.0003)
    margin_rate_default = _get_env_float("BITC_HYBRID_MARGIN_RATE", 0.0)
    force_cpu_default = _get_env_bool("BITC_HYBRID_FORCE_CPU", False)
    resample_default = _get_env_str("BITC_HYBRID_RESAMPLE", "1s")

    parser = argparse.ArgumentParser(
        description="Train a hybrid LSTM + XGBoost ensemble with feature-engineered OHLCV data."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=csv_default,
        help="OHLCV CSV path or directory containing daily CSVs (e.g., dataset/binance_raw).",
    )
    parser.add_argument(
        "--tail-rows",
        type=int,
        default=tail_rows_default,
        help="Use only the last N rows from the CSV (<=0 uses all rows).",
    )
    parser.add_argument(
        "--lookback",
        type=_positive_int,
        default=max(1, lookback_default),
        help="Sequence length for the LSTM.",
    )
    parser.add_argument(
        "--label-horizon",
        type=_positive_int,
        default=max(1, label_horizon_default),
        help="Prediction horizon in seconds for labels.",
    )
    parser.add_argument("--epochs", type=_positive_int, default=max(1, epochs_default), help="LSTM training epochs.")
    parser.add_argument("--batch-size", type=_positive_int, default=max(1, batch_size_default), help="LSTM batch size.")
    parser.add_argument("--lr", type=float, default=lr_default, help="LSTM learning rate.")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=max(0.0, grad_clip_default),
        help="LSTM gradient clipping norm.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=max_sequences_default,
        help="Cap the number of training sequences to manage memory/compute (<=0 to disable cap).",
    )
    parser.add_argument(
        "--xgb-rounds", type=_positive_int, default=max(1, xgb_rounds_default), help="XGBoost boosting rounds."
    )
    parser.add_argument(
        "--xgb-depth", type=_positive_int, default=max(1, xgb_depth_default), help="XGBoost max tree depth."
    )
    parser.add_argument("--xgb-eta", type=float, default=xgb_eta_default, help="XGBoost learning rate (eta).")
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=xgb_subsample_default,
        help="XGBoost subsample ratio for rows.",
    )
    parser.add_argument(
        "--xgb-colsample",
        type=float,
        default=xgb_colsample_default,
        help="XGBoost colsample_bytree ratio for columns.",
    )
    parser.add_argument(
        "--ensemble-weight-lstm",
        type=float,
        default=ensemble_weight_lstm_default,
        help="Weight for LSTM probability in the ensemble vote.",
    )
    parser.add_argument(
        "--ensemble-weight-xgb",
        type=float,
        default=ensemble_weight_xgb_default,
        help="Weight for XGBoost probability in the ensemble vote.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=save_dir_default,
        help="Directory to save the hybrid checkpoint (lstm.pt + xgb.json + meta.json).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=no_save_default,
        help="Skip writing checkpoints to disk.",
    )
    parser.add_argument(
        "--cost-aware-label",
        action="store_true",
        default=cost_aware_default,
        help="Use cost-aware labels (fee/slippage margin) instead of simple up/down.",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=fee_rate_default,
        help="Per-fill fee rate used when cost-aware labels are enabled.",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=slippage_rate_default,
        help="Per-fill slippage rate used when cost-aware labels are enabled.",
    )
    parser.add_argument(
        "--margin-rate",
        type=float,
        default=margin_rate_default,
        help="Additional margin threshold for cost-aware labels.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        default=force_cpu_default,
        help="Force CPU for LSTM training even when CUDA is available.",
    )
    parser.add_argument(
        "--resample",
        default=resample_default,
        help="Pandas resample rule (e.g., 1s). Set to 'none' to keep the original interval (e.g., 1m klines).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    device = torch.device("cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[hybrid] Using device: {device}")

    tail_rows = args.tail_rows
    if tail_rows is not None and tail_rows <= 0:
        tail_rows = None

    resample_rule: str | None
    if args.resample is None:
        resample_rule = None
    else:
        resample_clean = str(args.resample).strip().lower()
        if resample_clean in {"none", "null", "off", ""}:
            resample_rule = None
        else:
            resample_rule = args.resample

    max_sequences = args.max_sequences
    if max_sequences is not None and max_sequences <= 0:
        max_sequences = None

    feature_df = build_feature_frame(args.csv, tail_rows=tail_rows, resample_rule=resample_rule)
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
            max_sequences=max_sequences,
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
