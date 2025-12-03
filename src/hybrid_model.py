from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .model import (
    ModelResult,
    build_sequences,
    load_model_checkpoint,
    save_model_checkpoint,
    train_lstm,
)


DEFAULT_HYBRID_FEATURES: list[str] = [
    "buy_ratio",
    "imbalance",
    "volume_ratio_20",
    "volume_ratio_60",
    "vwap_pct_60",
    "vwap_dev_60",
    "price_change_pct_1",
    "price_change_pct_5",
    "roc_30",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "bb_percent_20",
    "bb_width_20",
    "roll_close_mean_20",
    "roll_close_std_20",
    "roll_volume_mean_20",
    "roll_volume_std_20",
    "close_lag_1",
    "close_lag_5",
    "volume_lag_1",
    "volume_lag_5",
]


def _require_xgboost():
    try:
        import xgboost as xgb  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "xgboost is required for the hybrid ensemble. Install it with `pip install xgboost`."
        ) from exc
    return xgb


def _train_val_counts(total: int) -> tuple[int, int]:
    train_count = int(total * 0.7)
    if train_count < 2:
        train_count = max(1, total - 1)
    val_count = total - train_count
    if val_count < 1 and total > 1:
        val_count = 1
        train_count = total - 1
    return train_count, val_count


@dataclass
class HybridModelResult:
    data: pd.DataFrame
    lstm_result: ModelResult
    booster: object
    feature_columns: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    lookback: int
    label_horizon: int
    xgb_best_iteration: int | None
    xgb_val_logloss: float | None
    xgb_val_accuracy: float | None
    ensemble_val_accuracy: float | None
    ensemble_weights: tuple[float, float]


def train_hybrid_ensemble(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    lookback: int = 60,
    label_horizon: int = 1,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    max_sequences: int | None = None,
    use_cost_labels: bool = False,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    margin_rate: float = 0.0,
    xgb_params: dict | None = None,
    xgb_rounds: int = 200,
    xgb_early_stopping_rounds: int = 20,
    ensemble_weights: tuple[float, float] = (0.5, 0.5),
) -> HybridModelResult:
    """Train an LSTM + XGBoost ensemble on an already feature-engineered DataFrame."""
    if feature_columns is None:
        feature_columns = DEFAULT_HYBRID_FEATURES
    feature_columns = list(feature_columns)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"train_hybrid_ensemble: DataFrame is missing feature columns: {missing}")

    # dropna subset은 리스트/Index를 기대하므로 list로 정리한다.
    required_for_clean = list({*feature_columns, "close"})
    df_clean = df.dropna(subset=required_for_clean)
    if len(df_clean) < lookback + label_horizon + 2:
        raise ValueError("Not enough rows after feature engineering to build sequences for the hybrid model.")

    # Build sequential dataset (targets shared with XGBoost)
    sequences, targets, indexes = build_sequences(
        df_clean,
        lookback=lookback,
        max_sequences=max_sequences,
        use_cost_labels=use_cost_labels,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        margin_rate=margin_rate,
        label_horizon=label_horizon,
        feature_columns=feature_columns,
    )
    total_samples = len(sequences)
    if total_samples < 5:
        raise ValueError(f"Need at least 5 sequences; got {total_samples}.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train LSTM
    print(f"[hybrid] Training LSTM on {total_samples} sequences (lookback={lookback}, horizon={label_horizon})")
    lstm_result = train_lstm(
        sequences,
        targets,
        indexes,
        df=df_clean,
        window=lookback,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        grad_clip=grad_clip,
    )
    if lstm_result.feature_mean is None or lstm_result.feature_std is None:
        raise RuntimeError("LSTM training did not return feature normalization stats.")

    feature_mean = np.asarray(lstm_result.feature_mean, dtype=np.float32)
    feature_std = np.asarray(lstm_result.feature_std, dtype=np.float32)
    std_safe = np.where(feature_std < 1e-6, 1.0, feature_std)

    # Prepare tabular data for XGBoost aligned to the same timestamps as LSTM sequences
    tabular = df_clean.reindex(indexes)
    feature_matrix = tabular[feature_columns].to_numpy(dtype=np.float32, copy=True)
    feature_matrix = (feature_matrix - feature_mean) / std_safe
    target_array = targets.astype(np.float32)

    train_count, val_count = _train_val_counts(total_samples)
    xgb = _require_xgboost()
    dtrain = xgb.DMatrix(feature_matrix[:train_count], label=target_array[:train_count])
    evals = [(dtrain, "train")]
    dval = None
    if val_count > 0:
        dval = xgb.DMatrix(feature_matrix[train_count:], label=target_array[train_count:])
        evals.append((dval, "val"))

    if xgb_params is None:
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "lambda": 1.0,
            "verbosity": 0,
        }

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=xgb_rounds,
        evals=evals,
        early_stopping_rounds=xgb_early_stopping_rounds if dval is not None else None,
        verbose_eval=False,
    )
    print(f"[hybrid] XGBoost training done (rounds={xgb_rounds}, val_split={val_count})")

    best_iteration = getattr(booster, "best_iteration", None)
    predict_kwargs: dict = {}
    if best_iteration is not None:
        predict_kwargs["iteration_range"] = (0, best_iteration + 1)

    all_probs_xgb = booster.predict(xgb.DMatrix(feature_matrix), **predict_kwargs)

    xgb_val_logloss: float | None = None
    xgb_val_accuracy: float | None = None
    if dval is not None:
        if hasattr(booster, "best_score"):
            try:
                xgb_val_logloss = float(booster.best_score)  # type: ignore[arg-type]
            except Exception:
                xgb_val_logloss = None
        val_probs = all_probs_xgb[train_count:]
        val_targets = target_array[train_count:]
        if len(val_targets) > 0:
            val_preds = (val_probs >= 0.5).astype(int)
            xgb_val_accuracy = float((val_preds == val_targets).mean())

    # Ensemble voting (weighted average of probabilities)
    w_lstm, w_xgb = ensemble_weights
    weight_sum = w_lstm + w_xgb
    if weight_sum <= 0:
        w_lstm = 0.5
        w_xgb = 0.5
        weight_sum = 1.0
    w_lstm /= weight_sum
    w_xgb /= weight_sum

    lstm_probs = lstm_result.data["prob_up"].to_numpy(dtype=float)
    ensemble_probs = w_lstm * lstm_probs + w_xgb * all_probs_xgb
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    ensemble_val_accuracy: float | None = None
    if val_count > 0:
        ens_val = ensemble_preds[train_count:]
        val_targets = target_array[train_count:]
        if len(val_targets) > 0:
            ensemble_val_accuracy = float((ens_val == val_targets).mean())

    enriched = lstm_result.data.copy()
    enriched["target"] = target_array
    enriched["prob_up_xgb"] = all_probs_xgb
    enriched["prob_up_ensemble"] = ensemble_probs
    enriched["pred_ensemble"] = ensemble_preds

    return HybridModelResult(
        data=enriched,
        lstm_result=lstm_result,
        booster=booster,
        feature_columns=feature_columns,
        feature_mean=feature_mean,
        feature_std=feature_std,
        lookback=lookback,
        label_horizon=label_horizon,
        xgb_best_iteration=best_iteration,
        xgb_val_logloss=xgb_val_logloss,
        xgb_val_accuracy=xgb_val_accuracy,
        ensemble_val_accuracy=ensemble_val_accuracy,
        ensemble_weights=(w_lstm, w_xgb),
    )


def save_hybrid_checkpoint(path: Path, result: HybridModelResult) -> None:
    """Persist the hybrid ensemble (LSTM checkpoint + XGBoost model + metadata)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    lstm_path = path / "lstm.pt"
    save_model_checkpoint(result.lstm_result, result.lookback, lstm_path)

    xgb = _require_xgboost()
    booster = result.booster
    if isinstance(booster, xgb.Booster):
        booster_path = path / "xgb.json"
        booster.save_model(booster_path)
    else:  # pragma: no cover - defensive
        raise TypeError("Unexpected booster type when saving hybrid checkpoint.")

    meta = {
        "feature_columns": result.feature_columns,
        "feature_mean": result.feature_mean.tolist(),
        "feature_std": result.feature_std.tolist(),
        "lookback": result.lookback,
        "label_horizon": result.label_horizon,
        "ensemble_weights": list(result.ensemble_weights),
        "xgb_best_iteration": result.xgb_best_iteration,
    }
    meta_path = path / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))


def load_hybrid_checkpoint(path: Path, device: torch.device | None = None) -> tuple[ModelResult, object, dict]:
    """Load a previously saved hybrid ensemble components."""
    path = Path(path)
    lstm_path = path / "lstm.pt"
    xgb_path = path / "xgb.json"
    meta_path = path / "meta.json"

    model, feature_mean, feature_std, lookback = load_model_checkpoint(lstm_path, device=device)
    meta = json.loads(meta_path.read_text())
    meta.setdefault("lookback", lookback)
    meta.setdefault("feature_mean", feature_mean.tolist())
    meta.setdefault("feature_std", feature_std.tolist())

    xgb = _require_xgboost()
    booster = xgb.Booster()
    booster.load_model(str(xgb_path))

    dummy_result = ModelResult(
        data=pd.DataFrame(),
        loss=0.0,
        accuracy=0.0,
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    return dummy_result, booster, meta
