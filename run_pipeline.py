from __future__ import annotations

import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Sequence

from scripts.split_ohlcv_daily import split_ohlcv_by_day
from src.cli import main as cli_main
from src.model import load_checkpoint_metadata


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _load_env(path: Path | None = None) -> dict[str, str]:
    """간단한 .env 파일 로더.

    KEY=VALUE 형식의 줄을 읽어서 dict로 반환합니다.
    - # 으로 시작하는 줄과 빈 줄은 무시
    - 양 끝의 작은/큰따옴표는 제거
    """
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
    # 1) .env에서 기본값 불러오기
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
        description="Run LSTM training on one day-per-CSV files, continuing the model across days."
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=_get_env_path("BITC_DAILY_DIR", "dataset/daily"),
        help="Directory containing per-day OHLCV CSVs (e.g., from scripts/split_ohlcv_daily.py).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_get_env_path("BITC_OUTPUT", "dataset/binance_ohlcv.csv"),
        help="Consolidated OHLCV CSV to split when daily CSVs are missing.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=_get_env_optional_path("BITC_SAVE_MODEL"),
        help="학습된 LSTM 체크포인트를 저장할 경로 (예: models/btc_lstm.pt).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=_get_env_int("BITC_WINDOW", 60),
        help="LSTM lookback window.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=_get_env_int("BITC_EPOCHS", 1),
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_get_env_int("BITC_BATCH_SIZE", 16),
        help="Train batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=_get_env_float("BITC_LR", 1e-3),
        help="Learning rate.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=_get_env_float("BITC_GRAD_CLIP", 1.0),
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=_get_env_bool("BITC_PLOT", False),
        help="Show plots at the end.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=_get_env_bool("BITC_WANDB_ENABLED", False),
        help="Weights & Biases에 학습 결과를 로깅합니다.",
    )
    parser.add_argument(
        "--wandb-project",
        default=_get_env_str("BITC_WANDB_PROJECT"),
        help="Weights & Biases 프로젝트 이름.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=_get_env_str("BITC_WANDB_ENTITY"),
        help="Weights & Biases 엔터티(팀/사용자 이름).",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=_get_env_str("BITC_WANDB_RUN_NAME"),
        help="Weights & Biases 런 이름.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    daily_dir = args.daily_dir
    source_csv = args.source

    # 1) 일일 CSV가 없으면 자동으로 split_ohlcv_daily를 실행해 생성
    needs_split = (not daily_dir.exists()) or (not any(daily_dir.glob("binance_ohlcv_*.csv")))
    if needs_split:
        if not source_csv.exists():
            raise FileNotFoundError(
                f"No daily CSVs in {daily_dir} and source CSV not found at {source_csv}."
            )
        print(f"[daily-csv] No daily files found. Splitting {source_csv} into {daily_dir} ...")
        split_ohlcv_by_day(source_csv, daily_dir)

    # 2) 날짜별 CSV 리스트 수집 (파일명 형식: binance_ohlcv_YYYY-MM-DD.csv)
    daily_files = sorted(daily_dir.glob("binance_ohlcv_*.csv"))
    if not daily_files:
        raise RuntimeError(f"No daily CSVs found under {daily_dir} even after splitting.")

    def _extract_date_from_name(path: Path) -> date:
        # 기대 형식: binance_ohlcv_YYYY-MM-DD.csv
        stem = path.stem  # e.g., "binance_ohlcv_2025-01-01"
        parts = stem.split("_")
        if not parts:
            raise ValueError(f"Cannot parse date from filename: {path.name}")
        date_str = parts[-1]
        return datetime.fromisoformat(date_str).date()

    files_with_dates = sorted((p, _extract_date_from_name(p)) for p in daily_files)

    # 3) src.cli를 통해 하루치 CSV들을 순서대로 모두 학습
    #    (한 번 실행하면 처리 가능한 일일 CSV를 끝까지 처리)
    # 체크포인트의 마지막 날짜 이후부터 순차적으로 진행한다.
    start_idx = 0
    if args.save_model is not None and args.save_model.exists():
        meta = load_checkpoint_metadata(args.save_model)
        last_ts = meta.get("trained_until")
        if last_ts is not None:
            last_dt = datetime.fromisoformat(str(last_ts))
            last_date = last_dt.date()
            target_date = last_date + timedelta(days=1)
            for i, (_, d) in enumerate(files_with_dates):
                if d == target_date:
                    start_idx = i
                    break
            else:
                print("[daily-csv] No next daily CSV to train on (all days have been processed).")
                return

    for i in range(start_idx, len(files_with_dates)):
        next_file, file_date = files_with_dates[i]
        print(f"[daily-csv] Using daily CSV: {next_file.name}")

        cli_args = [
            str(next_file),
            "--window",
            str(args.window),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--grad-clip",
            str(args.grad_clip),
            "--quiet",
        ]
        if args.save_model is not None:
            # 이어서 학습할 수 있도록 동일 경로에 저장/로드
            if args.save_model.exists():
                cli_args.extend(["--load-model", str(args.save_model)])
            cli_args.extend(["--save-model", str(args.save_model)])
        if args.plot:
            cli_args.append("--plot")
        if args.wandb:
            cli_args.append("--wandb")
            if args.wandb_project:
                cli_args.extend(["--wandb-project", args.wandb_project])
            if args.wandb_entity:
                cli_args.extend(["--wandb-entity", args.wandb_entity])
            if args.wandb_run_name:
                cli_args.extend(["--wandb-run-name", args.wandb_run_name])

        cli_main(cli_args)


if __name__ == "__main__":
    main()
