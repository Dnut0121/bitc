from __future__ import annotations

import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Sequence

from scripts.split_ohlcv_daily import split_ohlcv_by_day
from scripts.eval_next_minute import main as eval_next_minute_main
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

    def _get_env_optional_str(name: str) -> str | None:
        value = env.get(name)
        if value is None or value == "":
            return None
        return value

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
        "--single-day",
        help=(
            "하루치 데이터만 학습할 경우 사용할 날짜 (형식: YYYY-MM-DD). "
            "지정하면 해당 날짜의 daily CSV 한 개만 사용합니다."
        ),
        default=_get_env_optional_str("BITC_SINGLE_DAY"),
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=_get_env_int("BITC_HOLDOUT_DAYS", 0),
        help="마지막 N일은 학습에서 제외하고, 그 구간은 오직 평가/백테스트에만 사용합니다.",
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
        "--fee-rate",
        type=float,
        default=_get_env_float("BITC_FEE_RATE", 0.0006),
        help="한 번 체결될 때 수수료 비율 (예: 0.0005 = 0.05%).",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=_get_env_float("BITC_SLIPPAGE_RATE", 0.0003),
        help="한 번 체결될 때 슬리피지 비율 (예: 0.0001 = 0.01%).",
    )
    parser.add_argument(
        "--label-margin-rate",
        type=float,
        default=_get_env_float("BITC_LABEL_MARGIN_RATE", 0.0),
        help="수수료/슬리피지를 모두 제하고도 이만큼(비율) 이상 남을 때만 1로 라벨링합니다.",
    )
    parser.add_argument(
        "--cost-aware-label",
        action="store_true",
        default=_get_env_bool("BITC_COST_AWARE_LABEL", False),
        help="훈련 타깃을 단순 up/down이 아니라, 수수료/슬리피지 이후 롱이 유리한 구간(1) vs 그 외(0)로 설정합니다.",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=_get_env_int("BITC_LABEL_HORIZON", 1),
        help="라벨을 만들 때 볼 미래 horizon(초). 기본은 1초.",
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
    parser.add_argument(
        "--eval-after",
        action="store_true",
        default=_get_env_bool("BITC_EVAL_AFTER", False),
        help="모든 일일 학습이 끝난 후, 최근 체크포인트에 대해 1분 앞(+N초) 평가를 한 번 실행합니다.",
    )
    parser.add_argument(
        "--eval-cost-aware-label",
        action="store_true",
        default=_get_env_bool("BITC_EVAL_COST_AWARE_LABEL", False),
        help="평가(eval_next_minute) 시 라벨을 코스트-어웨어 정의(롱 유리=1, 그 외=0)로 사용합니다.",
    )
    parser.add_argument(
        "--eval-horizon",
        type=_positive_int,
        default=_get_env_int("BITC_EVAL_HORIZON", 60),
        help="평가 시 사용할 horizon(초). 기본 60초(1분).",
    )
    parser.add_argument(
        "--run-backtest",
        action="store_true",
        default=_get_env_bool("BITC_RUN_BACKTEST", False),
        help="각 일일 학습 시 src.cli의 이벤트 기반 백테스트를 함께 실행합니다.",
    )
    parser.add_argument(
        "--use-raw-daily",
        action="store_true",
        default=_get_env_bool("BITC_USE_RAW_DAILY", False),
        help=(
            "dataset/binance_raw 처럼 하루 폴더 안 하루 CSV 구조를 그대로 사용합니다. "
            "이 경우 split_ohlcv_daily를 사용하지 않고, "
            "각 서브 디렉터리 안의 1초 봉 CSV를 src.cli에 직접 전달합니다."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    daily_dir = args.daily_dir
    source_csv = args.source
    use_raw_daily = args.use_raw_daily

    # 1) 일일 CSV 준비
    if use_raw_daily:
        # dataset/binance_raw 처럼 하루 폴더 안에 하루치 CSV가 있는 구조를 기대한다.
        if not daily_dir.exists():
            raise FileNotFoundError(f"Raw daily directory not found: {daily_dir}")
        # 예: BTCUSDT-1s-2025-01-01/BTCUSDT-1s-2025-01-01.csv
        daily_files: list[Path] = []
        for sub in sorted(daily_dir.iterdir()):
            if not sub.is_dir():
                continue
            csv_candidates = sorted(sub.glob("*.csv"))
            if not csv_candidates:
                continue
            # 하루 폴더 안 첫 번째 CSV를 사용
            daily_files.append(csv_candidates[0])
        if not daily_files:
            raise RuntimeError(f"No raw daily CSVs found under {daily_dir}.")

        def _extract_date_from_name(path: Path) -> date:
            # 기대 형식: BTCUSDT-1s-YYYY-MM-DD.csv
            stem = path.stem  # e.g., "BTCUSDT-1s-2025-01-01"
            parts = stem.split("-")
            if len(parts) < 3:
                raise ValueError(f"Cannot parse date from filename: {path.name}")
            date_str = "-".join(parts[-3:])
            return datetime.fromisoformat(date_str).date()

    else:
        # 기존 모드: binance_ohlcv.csv 를 하루씩 split 해서 사용
        needs_split = (not daily_dir.exists()) or (not any(daily_dir.glob("binance_ohlcv_*.csv")))
        if needs_split:
            if not source_csv.exists():
                raise FileNotFoundError(
                    f"No daily CSVs in {daily_dir} and source CSV not found at {source_csv}."
                )
            print(f"[daily-csv] No daily files found. Splitting {source_csv} into {daily_dir} ...")
            split_ohlcv_by_day(source_csv, daily_dir)

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

    # --single-day (또는 BITC_SINGLE_DAY)가 지정되면 해당 날짜 하루만 학습한다.
    single_day_str = args.single_day
    holdout_days = max(0, int(args.holdout_days))
    start_idx = 0

    holdout_files: list[tuple[Path, date]] = []

    if single_day_str:
        try:
            single_day = datetime.fromisoformat(single_day_str).date()
        except ValueError as exc:  # pragma: no cover - 잘못된 입력 방어
            raise ValueError(f"Invalid --single-day format: {single_day_str} (expected YYYY-MM-DD)") from exc
        train_files_with_dates = [(p, d) for (p, d) in files_with_dates if d == single_day]
        if not train_files_with_dates:
            print(f"[daily-csv] No daily CSV found for single day {single_day} under {daily_dir}.")
            return
    else:
        # holdout-days가 지정되면 마지막 N일을 holdout으로 떼어낸다.
        if holdout_days > 0 and len(files_with_dates) > holdout_days:
            split_idx = len(files_with_dates) - holdout_days
            train_files_with_dates = files_with_dates[:split_idx]
            holdout_files = files_with_dates[split_idx:]
        else:
            train_files_with_dates = files_with_dates
            holdout_files = []

        # 3) src.cli를 통해 학습 구간만 순서대로 모두 학습
        # 체크포인트의 마지막 날짜 이후부터 순차적으로 진행한다.
        if args.save_model is not None and args.save_model.exists():
            meta = load_checkpoint_metadata(args.save_model)
            last_ts = meta.get("trained_until")
            if last_ts is not None:
                last_dt = datetime.fromisoformat(str(last_ts))
                last_date = last_dt.date()
                target_date = last_date + timedelta(days=1)
                for i, (_, d) in enumerate(train_files_with_dates):
                    if d == target_date:
                        start_idx = i
                        break
                else:
                    print("[daily-csv] No next daily CSV to train on (all days have been processed).")
                    return

    # 3-a) 학습 구간에 대해서만 LSTM 학습(walk-forward 스타일로 이어서 학습)
    for i in range(start_idx, len(train_files_with_dates)):
        next_file, file_date = train_files_with_dates[i]
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
        # 수수료/슬리피지/라벨 마진 및 코스트-어웨어 라벨 옵션을 그대로 넘긴다.
        cli_args.extend(
            [
                "--fee-rate",
                str(args.fee_rate),
                "--slippage-rate",
                str(args.slippage_rate),
                "--label-margin-rate",
                str(args.label_margin_rate),
                "--label-horizon",
                str(args.label_horizon),
            ]
        )
        if args.cost_aware_label:
            cli_args.append("--cost-aware-label")
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
        if args.run_backtest:
            cli_args.append("--run-backtest")

        cli_main(cli_args)

    # 3-b) holdout 구간에 대해서는 학습 없이(epochs=0) 평가/백테스트만 수행한다.
    if holdout_files and args.save_model is not None and args.save_model.exists():
        print(
            f"[run_pipeline] Starting hold-out evaluation on last {len(holdout_files)} day(s) "
            f"using checkpoint at {args.save_model}"
        )
        for holdout_file, holdout_date in holdout_files:
            print(f"[run_pipeline] Hold-out day: {holdout_file.name}")
            cli_args = [
                str(holdout_file),
                "--window",
                str(args.window),
                "--epochs",
                "0",  # 학습 없이 기존 체크포인트로만 평가
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--grad-clip",
                str(args.grad_clip),
                "--quiet",
                "--load-model",
                str(args.save_model),
            ]
            cli_args.extend(
                [
                    "--fee-rate",
                    str(args.fee_rate),
                    "--slippage-rate",
                    str(args.slippage_rate),
                    "--label-margin-rate",
                    str(args.label_margin_rate),
                    "--label-horizon",
                    str(args.label_horizon),
                ]
            )
            if args.run_backtest:
                cli_args.append("--run-backtest")
            # hold-out 평가는 wandb에는 굳이 중복 로깅하지 않는다.
            cli_main(cli_args)

    # 모든 일일 학습이 끝난 뒤, 선택적으로 1분 앞(+N초) 평가를 수행한다.
    if args.save_model is not None and args.eval_after:
        if not args.save_model.exists():
            print(f"[run_pipeline] 지정한 체크포인트가 존재하지 않아 평가를 건너뜁니다: {args.save_model}")
            return
        eval_argv: list[str] = [
            "--model",
            str(args.save_model),
            "--horizon",
            str(args.eval_horizon),
        ]
        # 평가 라벨을 cost-aware 로 보고 싶으면 학습 라벨과 무관하게 별도 플래그로 제어한다.
        if args.eval_cost_aware_label:
            eval_argv.append("--cost-aware-label")
            eval_argv.extend(
                [
                    "--fee-rate",
                    str(args.fee_rate),
                    "--slippage-rate",
                    str(args.slippage_rate),
                    "--margin-rate",
                    str(args.label_margin_rate),
                ]
            )
        print(
            f"[run_pipeline] Running +{args.eval_horizon}s evaluation "
            f"on latest checkpoint at {args.save_model} (cost-aware={args.cost_aware_label})"
        )
        eval_next_minute_main(eval_argv)


if __name__ == "__main__":
    main()
