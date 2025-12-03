from __future__ import annotations

import argparse
from datetime import datetime, date
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.data import load_ohlcv
from src.features import add_microstructure_features
from src.labels import triple_barrier_labels
from scripts.analyze_horizons_meta import compute_prob_up_series  # type: ignore[import]


def _load_env(path: Path | None = None) -> dict[str, str]:
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


def main(argv: Sequence[str] | None = None) -> None:
    env = _load_env()

    def _get_env_path(name: str, default: str) -> Path:
        value = env.get(name)
        return Path(value) if value else Path(default)

    def _get_env_str(name: str, default: str | None = None) -> str | None:
        value = env.get(name)
        if value is None or value == "":
            return default
        return value

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

    def _get_env_optional_path(name: str) -> Path | None:
        value = env.get(name)
        if value is None or value == "":
            return None
        return Path(value)

    parser = argparse.ArgumentParser(
        description="Train a simple triple-barrier meta-model on top of an existing LSTM using daily CSVs."
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=_get_env_path("BITC_DAILY_DIR", "dataset/daily"),
        help="Directory containing per-day CSVs (either raw binance_raw subfolders or binance_ohlcv_*.csv).",
    )
    parser.add_argument(
        "--use-raw-daily",
        action="store_true",
        default=_get_env_bool("BITC_USE_RAW_DAILY", False),
        help="Interpret daily-dir as raw binance_1s subfolders (dataset/binance_raw/...).",
    )
    parser.add_argument(
        "--single-day",
        help="When set, only this YYYY-MM-DD day is used to collect meta samples.",
        default=_get_env_str("BITC_META_SINGLE_DAY"),
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=_get_env_int("BITC_HOLDOUT_DAYS", 0),
        help="Last N days will be skipped (kept for final OOS evaluation).",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=_get_env_optional_path("BITC_SAVE_MODEL") or Path("models/btc_lstm.pt"),
        help="Base LSTM checkpoint to use for prob_up generation.",
    )
    parser.add_argument(
        "--save-meta-model",
        type=Path,
        default=_get_env_optional_path("BITC_META_MODEL") or Path("models/btc_meta.pt"),
        help="Path to save the trained meta-model parameters.",
    )
    parser.add_argument(
        "--base-long-threshold",
        type=float,
        default=_get_env_float("BITC_META_BASE_LONG_THRESHOLD", 0.6),
        help="Base LSTM prob_up threshold to consider a long candidate for meta training.",
    )
    # backward compatibility alias
    parser.add_argument(
        "--base-threshold",
        dest="base_long_threshold",
        type=float,
        default=None,
        help="Alias for --base-long-threshold.",
    )
    parser.add_argument(
        "--base-short-threshold",
        type=float,
        default=_get_env_float("BITC_META_BASE_SHORT_THRESHOLD", 0.4),
        help="Base LSTM prob_up threshold to consider a short candidate for meta training.",
    )
    parser.add_argument(
        "--tb-horizon",
        type=int,
        default=_get_env_int("BITC_META_TB_HORIZON", 60),
        help="Horizon (seconds) for triple-barrier labels.",
    )
    parser.add_argument(
        "--tb-up-pct",
        type=float,
        default=_get_env_float("BITC_META_TB_UP_PCT", 0.001),
        help="Upper barrier percentage for triple-barrier (e.g., 0.001 = 0.1%).",
    )
    parser.add_argument(
        "--tb-down-pct",
        type=float,
        default=_get_env_float("BITC_META_TB_DOWN_PCT", 0.001),
        help="Lower barrier percentage for triple-barrier (e.g., 0.001 = 0.1%).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_get_env_int("BITC_META_EPOCHS", 5),
        help="Training epochs for the meta-model.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # handle alias defaulting
    if args.base_long_threshold is None:
        args.base_long_threshold = _get_env_float("BITC_META_BASE_LONG_THRESHOLD", 0.6)

    daily_dir = args.daily_dir
    use_raw_daily = args.use_raw_daily
    base_model_path = args.base_model

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base LSTM checkpoint not found: {base_model_path}")

    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily directory not found: {daily_dir}")

    # 1) 수집할 일일 CSV 리스트(날짜 포함) 구성
    if use_raw_daily:
        daily_files: list[Path] = []
        for sub in sorted(daily_dir.iterdir()):
            if not sub.is_dir():
                continue
            csv_candidates = sorted(sub.glob("*.csv"))
            if not csv_candidates:
                continue
            daily_files.append(csv_candidates[0])

        if not daily_files:
            raise RuntimeError(f"No raw daily CSVs found under {daily_dir}")

        def _extract_date(path: Path) -> date:
            stem = path.stem  # BTCUSDT-1s-YYYY-MM-DD
            parts = stem.split("-")
            if len(parts) < 3:
                raise ValueError(f"Cannot parse date from filename: {path.name}")
            date_str = "-".join(parts[-3:])
            return datetime.fromisoformat(date_str).date()

    else:
        daily_files = sorted(daily_dir.glob("binance_ohlcv_*.csv"))
        if not daily_files:
            raise RuntimeError(f"No daily CSVs found under {daily_dir}")

        def _extract_date(path: Path) -> date:
            stem = path.stem  # binance_ohlcv_YYYY-MM-DD
            parts = stem.split("_")
            if not parts:
                raise ValueError(f"Cannot parse date from filename: {path.name}")
            date_str = parts[-1]
            return datetime.fromisoformat(date_str).date()

    files_with_dates = sorted((p, _extract_date(p)) for p in daily_files)

    # 2) single-day 또는 holdout 설정으로 meta 학습 대상 날짜 범위 결정
    single_day_str = args.single_day
    holdout_days = max(0, int(args.holdout_days))

    if single_day_str:
        try:
            single_day = datetime.fromisoformat(single_day_str).date()
        except ValueError as exc:
            raise ValueError(f"Invalid --single-day format: {single_day_str} (expected YYYY-MM-DD)") from exc
        train_files_with_dates = [(p, d) for (p, d) in files_with_dates if d == single_day]
        if not train_files_with_dates:
            print(f"[meta-pipeline] No daily CSV found for single day {single_day} under {daily_dir}.")
            return
    else:
        if holdout_days > 0 and len(files_with_dates) > holdout_days:
            split_idx = len(files_with_dates) - holdout_days
            train_files_with_dates = files_with_dates[:split_idx]
        else:
            train_files_with_dates = files_with_dates

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[meta-pipeline] Using device: {device}")
    print(f"[meta-pipeline] Base model: {base_model_path}")

    # 3) 모든 train 일자에서 메타 학습용 샘플 수집
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
        "direction",  # +1 for long candidate, -1 for short candidate
    ]

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for path, d in train_files_with_dates:
        print(f"[meta-pipeline] Collecting meta samples from {path.name} (date={d})")
        ohlcv = load_ohlcv(path)
        prob_df, _, _, _ = compute_prob_up_series(ohlcv, base_model_path, device=device)

        # 마이크로 피처 + triple-barrier 라벨 구성 (analyze_horizons_meta.train_meta_model 로직과 동일)
        micro = add_microstructure_features(prob_df)
        df = micro.reindex(prob_df.index).copy()
        df["prob_up"] = prob_df["prob_up"]

        tb = triple_barrier_labels(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            horizon=args.tb_horizon,
            up_pct=args.tb_up_pct,
            down_pct=args.tb_down_pct,
        )
        df["tb_label"] = tb

        # 방향 피처를 위해 기본 컬럼 추가
        df["direction"] = 0.0

        base_long = df["prob_up"] >= args.base_long_threshold
        base_short = df["prob_up"] <= args.base_short_threshold
        # 타깃: 방향과 triple-barrier 결과가 일치할 때 1
        y_long = (df["tb_label"] == 1).astype(float)
        y_short = (df["tb_label"] == -1).astype(float)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"[meta-pipeline] Skipping {path.name}: missing features {missing}")
            continue

        # 결측을 간단히 처리해 학습과 추론 시 일관성을 유지
        df_filled = df.copy()
        df_filled[feature_cols] = df_filled[feature_cols].ffill().bfill().fillna(0.0)

        # 롱 후보 샘플
        df_long = df_filled.loc[base_long].copy()
        df_long["direction"] = 1.0
        X_long = df_long[feature_cols].to_numpy(dtype=np.float32)
        y_long_masked = y_long.loc[df_long.index].to_numpy(dtype=np.float32)

        # 숏 후보 샘플
        df_short = df_filled.loc[base_short].copy()
        df_short["direction"] = -1.0
        X_short = df_short[feature_cols].to_numpy(dtype=np.float32)
        y_short_masked = y_short.loc[df_short.index].to_numpy(dtype=np.float32)

        X_day = np.concatenate([X_long, X_short], axis=0)
        y_day = np.concatenate([y_long_masked, y_short_masked], axis=0)

        if X_day.shape[0] < 50:
            print(f"[meta-pipeline] Skipping {path.name}: not enough base-long/short samples ({X_day.shape[0]})")
            continue

        X_list.append(X_day)
        y_list.append(y_day)

    if not X_list:
        print("[meta-pipeline] No meta samples collected; aborting meta-model training.")
        return

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    total = X_all.shape[0]

    # 4) 메타 모델 학습 (간단한 로지스틱 회귀)
    train_n = int(total * 0.7)
    X_train, X_val = X_all[:train_n], X_all[train_n:]
    y_train, y_val = y_all[:train_n], y_all[train_n:]

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    model = torch.nn.Linear(X_all.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(
        f"[meta-pipeline] Training meta-model: total_samples={total}, "
        f"train={train_n}, val={total-train_n}, positive_rate={y_all.mean():.3f}"
    )

    for epoch in range(args.epochs):
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

        train_pred = (train_prob >= 0.5).astype(int)
        val_pred = (val_prob >= 0.5).astype(int)
        train_acc = (train_pred == y_train).mean()
        val_acc = (val_pred == y_val).mean()

        print(
            f"[meta-pipeline] Epoch {epoch+1}/{args.epochs}: "
            f"train_loss={loss.item():.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

    # 5) 메타 모델 저장 (state_dict + 메타데이터)
    meta_checkpoint = {
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "base_long_threshold": float(args.base_long_threshold),
        "base_short_threshold": float(args.base_short_threshold),
        "tb_horizon": int(args.tb_horizon),
        "tb_up_pct": float(args.tb_up_pct),
        "tb_down_pct": float(args.tb_down_pct),
    }
    save_path = args.save_meta_model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta_checkpoint, save_path)
    print(f"[meta-pipeline] Saved meta-model checkpoint to {save_path}")


if __name__ == "__main__":
    main()
