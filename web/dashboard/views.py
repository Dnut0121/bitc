from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from torch.utils.data import DataLoader

from src.model import SequenceDataset, build_sequences, load_model_checkpoint
from src.pressure import compute_pressure
from scripts.prepare_dataset import read_daily_csv


DEFAULT_PROB_THRESHOLD = 0.6
DEFAULT_HORIZON = 60
DEFAULT_RISK_REWARD = 1.0
DEFAULT_MIN_EXPECTED_PCT = 0.0
DEFAULT_FEE_RATE = 0.0006
DEFAULT_SLIPPAGE_RATE = 0.0003
RECENT_ROWS = 30
CHART_CANDLES = 300


def _load_model_result() -> tuple[dict, pd.DataFrame, dict]:
    """CLI 학습 결과(model_result.csv, meta.json)를 읽어 현재 스냅샷/데이터/메타 정보를 반환합니다."""
    csv_path = Path(settings.BITC_MODEL_RESULT_CSV)
    meta_path = Path(settings.BITC_MODEL_META_JSON)
    if not csv_path.exists():
        raise FileNotFoundError(f"모델 결과 CSV를 찾을 수 없습니다: {csv_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"모델 메타 JSON을 찾을 수 없습니다: {meta_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("model_result.csv에 'timestamp' 컬럼이 없습니다.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.set_index("timestamp").sort_index()

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    if df.empty:
        raise ValueError("model_result.csv에 데이터가 없습니다.")

    latest = df.iloc[-1]
    snapshot = {
        "timestamp": latest.name,
        "price_open": float(latest.get("open", float("nan"))),
        "price_close": float(latest.get("close", float("nan"))),
        "price_high": float(latest.get("high", float("nan"))),
        "price_low": float(latest.get("low", float("nan"))),
        "buy_pressure": float(latest.get("buy_pressure", float("nan"))),
        "sell_pressure": float(latest.get("sell_pressure", float("nan"))),
        "buy_ratio": float(latest.get("buy_ratio", float("nan"))),
        "imbalance": float(latest.get("imbalance", float("nan"))),
        "prob_up": float(latest.get("prob_up", float("nan"))),
        "pred_direction": int(latest.get("pred_direction", 0)),
        "val_accuracy": float(meta.get("accuracy", float("nan"))),
        "val_loss": float(meta.get("loss", float("nan"))),
        "lookback": int(meta.get("window", 0)),
        "tail_rows": meta.get("tail_rows"),
        "device": meta.get("device", "offline"),
    }
    return snapshot, df, meta


def _load_validate_predictions(model_path: Path, csv_path: Path) -> tuple[dict, pd.DataFrame, dict, dict]:
    """지정한 모델/검증 CSV에 대해 LSTM으로 prob_up을 예측해 붙인 DataFrame을 반환합니다."""
    if not model_path.exists():
        raise FileNotFoundError(f"LSTM 체크포인트를 찾을 수 없습니다: {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"검증 CSV를 찾을 수 없습니다: {csv_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) CSV 로드 및 피처 생성
    ohlcv = read_daily_csv(csv_path)
    df = compute_pressure(ohlcv)

    # 2) 모델/정규화 통계/룩백 로드
    model, feature_mean, feature_std, lookback = load_model_checkpoint(model_path, device=device)

    # 3) 시퀀스 구축 및 정규화
    sequences, targets, indexes = build_sequences(df, lookback)
    total_sequences = len(sequences)
    if total_sequences < 5:
        raise ValueError(f"평가에 필요한 시퀀스가 부족합니다. ({total_sequences})")

    feature_mean = feature_mean.astype(np.float32)
    feature_std = feature_std.astype(np.float32)
    std_safe = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = (sequences - feature_mean) / std_safe

    dataset = SequenceDataset(normalized, targets)
    loader = DataLoader(dataset, batch_size=256)

    # 4) 예측 및 정확도 계산
    model.eval()
    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for seq_batch, target_batch in loader:
            seq_batch = seq_batch.to(device)
            logits = model(seq_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(np.int8)
            probs_list.append(probs.astype(np.float32))
            preds_list.append(preds.astype(np.int8))

    probs_all = np.concatenate(probs_list) if probs_list else np.array([], dtype=np.float32)
    preds_all = np.concatenate(preds_list) if preds_list else np.array([], dtype=np.int8)
    accuracy = float((preds_all == targets[: len(preds_all)]).mean()) if len(preds_all) else float("nan")

    # 5) prob_up/신호 붙이기
    df_pred = df.copy()
    df_pred["prob_up"] = np.nan
    df_pred.loc[indexes, "prob_up"] = probs_all
    df_pred["pred_direction"] = np.where(df_pred["prob_up"] >= 0.5, 1, -1)
    df_pred["signal"] = np.where(df_pred["prob_up"] >= 0.55, 1, np.where(df_pred["prob_up"] <= 0.45, -1, 0))

    # 6) 스냅샷/메타 정보
    latest = df_pred.iloc[-1]
    snapshot = {
        "timestamp": df_pred.index[-1],
        "price_open": float(latest.get("open", float("nan"))),
        "price_close": float(latest.get("close", float("nan"))),
        "price_high": float(latest.get("high", float("nan"))),
        "price_low": float(latest.get("low", float("nan"))),
        "buy_pressure": float(latest.get("buy_pressure", float("nan"))),
        "sell_pressure": float(latest.get("sell_pressure", float("nan"))),
        "buy_ratio": float(latest.get("buy_ratio", float("nan"))),
        "imbalance": float(latest.get("imbalance", float("nan"))),
        "prob_up": float(latest.get("prob_up", float("nan"))),
        "pred_direction": int(latest.get("pred_direction", 0)),
        "val_accuracy": accuracy,
        "val_loss": float("nan"),
        "lookback": lookback,
        "tail_rows": len(df_pred),
        "device": str(device),
        "file": csv_path.name,
    }

    meta = {
        "accuracy": accuracy,
        "loss": float("nan"),
        "file": csv_path.name,
        "lookback": lookback,
        "tail_rows": len(df_pred),
        "device": str(device),
    }

    eval_dict = {
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100.0 if not np.isnan(accuracy) else None,
        "total_sequences": total_sequences,
        "file": csv_path.name,
        "lookback": lookback,
        "checkpoint": str(model_path),
        "device": str(device),
        "error": None,
        "samples": [],
    }
    # 최근 200개 예측을 담아 템플릿에서 재사용
    sample_limit = 200
    n = min(sample_limit, len(probs_all))
    if n > 0:
        slice_indexes = indexes[-n:]
        slice_targets = targets[-n:]
        slice_preds = preds_all[-n:]
        closes = df_pred.loc[slice_indexes, "close"]
        samples: list[dict] = []
        for ts, tgt, pred in zip(slice_indexes, slice_targets[-n:], slice_preds):
            samples.append(
                {
                    "timestamp": ts.isoformat(),
                    "close": float(closes.get(ts, float("nan"))),
                    "actual_up": int(tgt),
                    "pred_up": int(pred),
                    "correct": bool(int(tgt) == int(pred)),
                }
            )
        eval_dict["samples"] = samples
    return snapshot, df_pred, meta, eval_dict


def _prepare_candles(df: pd.DataFrame, limit: int = CHART_CANDLES) -> list[dict]:
    """차트용 OHLCV 캔들 시퀀스를 반환합니다."""
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return []

    data = df.tail(limit)
    candles: list[dict] = []
    for ts, row in data.iterrows():
        candles.append(
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "open": float(row.get("open", float("nan"))),
                "high": float(row.get("high", float("nan"))),
                "low": float(row.get("low", float("nan"))),
                "close": float(row.get("close", float("nan"))),
                "volume": float(row.get("volume", float("nan"))),
            }
        )
    return candles


def _pick_latest_validate_csv(validate_dir: Path) -> Path:
    candidates = list(validate_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"검증 CSV를 찾을 수 없습니다: {validate_dir}")
    candidates.sort()
    return candidates[-1]


def _evaluate_lstm_on_validate() -> dict:
    """btc_lstm.pt 모델을 dataset/validate 최신 CSV에 적용해 정확도를 계산합니다."""
    result = {
        "accuracy": None,
        "accuracy_pct": None,
        "total_sequences": 0,
        "file": None,
        "lookback": None,
        "checkpoint": settings.BITC_LSTM_CHECKPOINT,
        "device": None,
        "error": None,
        "samples": [],  # 최근 예측 결과 일부를 시각화용으로 담는다.
    }
    try:
        model_path = Path(settings.BITC_LSTM_CHECKPOINT)
        validate_dir = Path(settings.BITC_VALIDATE_DIR)
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM 체크포인트를 찾을 수 없습니다: {model_path}")
        csv_path = _pick_latest_validate_csv(validate_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result["device"] = str(device)

        # 1) 검증 CSV 로드 및 피처 생성
        ohlcv = read_daily_csv(csv_path)
        pressured = compute_pressure(ohlcv)

        # 2) 체크포인트 로드
        model, feature_mean, feature_std, lookback = load_model_checkpoint(model_path, device=device)
        result["lookback"] = lookback
        result["file"] = csv_path.name

        # 3) 시퀀스/타깃 생성 및 정규화
        sequences, targets, indexes = build_sequences(pressured, lookback)
        total_sequences = len(sequences)
        result["total_sequences"] = total_sequences
        if total_sequences < 5:
            raise ValueError(f"평가에 필요한 시퀀스가 부족합니다. ({total_sequences})")

        feature_mean = feature_mean.astype(np.float32)
        feature_std = feature_std.astype(np.float32)
        std_safe = np.where(feature_std < 1e-6, 1.0, feature_std)
        normalized = (sequences - feature_mean) / std_safe

        dataset = SequenceDataset(normalized, targets)
        loader = DataLoader(dataset, batch_size=256)

        # 4) 정확도 계산
        model.eval()
        correct = 0
        total = 0
        preds_list: list[np.ndarray] = []
        with torch.no_grad():
            for seq_batch, target_batch in loader:
                seq_batch = seq_batch.to(device)
                target_batch = target_batch.to(device)
                logits = model(seq_batch)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == target_batch).sum().item()
                total += target_batch.size(0)
                preds_list.append(preds.cpu().numpy().astype(np.int8))

        if preds_list:
            preds_all = np.concatenate(preds_list)
        else:
            preds_all = np.array([], dtype=np.int8)

        accuracy = correct / total if total > 0 else 0.0
        result["accuracy"] = accuracy
        result["accuracy_pct"] = accuracy * 100.0

        # 5) 최근 일부 예측을 시각화용으로 준비 (최대 200개)
        sample_limit = 200
        n = min(sample_limit, len(preds_all))
        if n > 0:
            # build_sequences가 반환한 indexes는 각 시퀀스의 기준 시점이다.
            slice_indexes = indexes[-n:]
            slice_targets = targets[-n:]
            slice_preds = preds_all[-n:]
            closes = pressured.loc[slice_indexes, "close"]
            samples: list[dict] = []
            for ts, tgt, pred in zip(slice_indexes, slice_targets[-n:], slice_preds):
                close_val = float(closes.get(ts, float("nan")))
                samples.append(
                    {
                        "timestamp": ts.isoformat(),
                        "close": close_val,
                        "actual_up": int(tgt),
                        "pred_up": int(pred),
                        "correct": bool(int(tgt) == int(pred)),
                    }
                )
            result["samples"] = samples
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result


def _compute_trade_analysis(
    df: pd.DataFrame,
    prob_threshold: float,
    horizon: int,
    risk_reward: float,
    fee_rate: float,
    slippage_rate: float,
    min_expected_pct: float,
) -> dict:
    analysis: dict = {
        "config": {
            "prob_threshold": prob_threshold,
            "horizon": horizon,
            "risk_reward": risk_reward,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "min_expected_pct": min_expected_pct,
        },
        "best_trade": None,
        "current": None,
    }
    if "prob_up" not in df.columns or "close" not in df.columns:
        return analysis

    df_reset = df.reset_index()
    timestamp_col = df_reset.columns[0]
    candidates = df_reset[df_reset["prob_up"] >= prob_threshold].copy()

    total_cost_rate = max(0.0, fee_rate + slippage_rate)
    best_trade: dict | None = None
    for start_idx_obj in candidates.index:
        start_idx = int(start_idx_obj)
        if start_idx + 1 >= len(df_reset):
            continue
        end_idx = min(len(df_reset) - 1, start_idx + horizon)
        future = df_reset.iloc[start_idx + 1 : end_idx + 1]
        if future.empty:
            continue
        rel_best_pos = int(future["close"].to_numpy().argmax())
        best_row = future.iloc[rel_best_pos]
        entry_row = df_reset.iloc[start_idx]
        entry_price = float(entry_row["close"])
        exit_price = float(best_row["close"])
        gross_profit = exit_price - entry_price
        trade_cost = total_cost_rate * (entry_price + exit_price)
        net_profit = gross_profit - trade_cost
        net_profit_pct = net_profit / entry_price * 100.0 if entry_price != 0 else 0.0
        if best_trade is None or net_profit_pct > best_trade["profit_pct"]:
            best_trade = {
                "entry_time": entry_row[timestamp_col],
                "entry_price": entry_price,
                "exit_time": best_row[timestamp_col],
                "exit_price": exit_price,
                "profit": net_profit,
                "profit_pct": net_profit_pct,
            }

    analysis["best_trade"] = best_trade

    latest = df.iloc[-1]
    entry_price = float(latest["close"])
    prob_up = float(latest["prob_up"])
    if "range" in df.columns:
        avg_range = float(df["range"].rolling(RECENT_ROWS, min_periods=1).mean().iloc[-1])
    else:
        avg_range = float((df["high"] - df["low"]).rolling(RECENT_ROWS, min_periods=1).mean().iloc[-1])
    rr = max(risk_reward, 0.1)
    target_move = rr * avg_range
    stop_move = avg_range
    target_price = entry_price + target_move
    stop_price = entry_price - stop_move

    target_cost = total_cost_rate * (entry_price + target_price)
    stop_cost = total_cost_rate * (entry_price + stop_price)
    expected_profit = prob_up * (target_move - target_cost) - (1.0 - prob_up) * (stop_move + stop_cost)
    expected_profit_pct = expected_profit / entry_price * 100.0 if entry_price != 0 else 0.0
    is_opportunity = expected_profit_pct >= min_expected_pct

    analysis["current"] = {
        "entry_time": latest.name,
        "entry_price": entry_price,
        "prob_up": prob_up,
        "avg_range": avg_range,
        "risk_reward": rr,
        "target_price": target_price,
        "stop_price": stop_price,
        "total_cost_rate": total_cost_rate,
        "expected_profit": expected_profit,
        "expected_profit_pct": expected_profit_pct,
        "is_opportunity": is_opportunity,
    }
    return analysis


def _compute_recent_view(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    data = df.copy()
    if data.empty:
        return {"window": RECENT_ROWS}, []

    window = len(data)
    avg_prob_up = float(data["prob_up"].mean()) if "prob_up" in data.columns else float("nan")
    bullish_ratio = float((data["prob_up"] >= 0.55).sum()) / len(data) * 100.0 if "prob_up" in data.columns else 0.0
    signal_up_ratio = float((data.get("signal", 0) > 0).sum()) / len(data) * 100.0

    summary = {
        "window": window,
        "avg_prob_up": avg_prob_up,
        "bullish_ratio": bullish_ratio,
        "signal_up_ratio": signal_up_ratio,
    }
    rows: list[dict] = []
    for idx, row in data.iterrows():
        rows.append(
            {
                "timestamp": idx,
                "close": float(row.get("close", float("nan"))),
                "prob_up": float(row.get("prob_up", float("nan"))),
                "signal": int(row.get("signal", 0)),
                "buy_ratio": float(row.get("buy_ratio", float("nan"))),
            }
        )
    return summary, rows


def index(request: HttpRequest) -> HttpResponse:
    context: dict = {}
    error_message: str | None = None
    snapshot: dict | None = None
    trade: dict | None = None
    recent_summary: dict | None = None
    recent_rows: list[dict] | None = None
    lstm_eval: dict | None = None

    def _get_float(name: str, default: float) -> float:
        raw = request.GET.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _get_int(name: str, default: int) -> int:
        raw = request.GET.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    prob_threshold = _get_float("prob_threshold", DEFAULT_PROB_THRESHOLD)
    horizon = _get_int("horizon", DEFAULT_HORIZON)
    risk_reward = _get_float("risk_reward", DEFAULT_RISK_REWARD)
    min_expected_pct = _get_float("min_expected_pct", DEFAULT_MIN_EXPECTED_PCT)
    fee_rate = _get_float("fee_rate", DEFAULT_FEE_RATE)
    slippage_rate = _get_float("slippage_rate", DEFAULT_SLIPPAGE_RATE)
    show_flow = request.GET.get("show_flow") == "1"
    model_choice = request.GET.get("model_path")
    validate_choice = request.GET.get("validate_file")

    models_dir = Path(settings.BITC_MODELS_DIR)
    validate_dir = Path(settings.BITC_VALIDATE_DIR)
    model_files = sorted([p.name for p in models_dir.glob("*.pt")]) if models_dir.exists() else []
    validate_files = sorted([p.name for p in validate_dir.glob("*.csv")]) if validate_dir.exists() else []

    selected_model: Path | None = None
    selected_validate: Path | None = None
    if model_choice:
        p = Path(model_choice)
        selected_model = p if p.is_absolute() else (models_dir / p)
    if validate_choice:
        p = Path(validate_choice)
        selected_validate = p if p.is_absolute() else (validate_dir / p)

    active_tab = request.GET.get("tab")
    if not active_tab:
        active_tab = "model" if (selected_model and selected_validate) else "binance"

    try:
        if selected_model and selected_validate:
            snapshot, df, meta, lstm_eval = _load_validate_predictions(selected_model, selected_validate)
            trade = _compute_trade_analysis(
                df,
                prob_threshold=prob_threshold,
                horizon=horizon,
                risk_reward=risk_reward,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                min_expected_pct=min_expected_pct,
            )
            if show_flow:
                recent_summary, recent_rows = _compute_recent_view(df)
            context["candles"] = _prepare_candles(df)
        else:
            lstm_eval = None
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)

    context["snapshot"] = snapshot
    context["trade"] = trade
    context["recent_summary"] = recent_summary
    context["recent_rows"] = recent_rows
    context["lstm_eval"] = lstm_eval
    context["model_files"] = model_files
    context["validate_files"] = validate_files
    context["selected_model"] = str(selected_model) if selected_model else ""
    context["selected_validate"] = str(selected_validate) if selected_validate else ""
    context["active_tab"] = active_tab
    context["config"] = {
        "prob_threshold": prob_threshold,
        "horizon": horizon,
        "risk_reward": risk_reward,
        "min_expected_pct": min_expected_pct,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
    }
    context["show_flow"] = show_flow
    context["error_message"] = error_message
    return render(request, "dashboard/index.html", context)


def candles_api(request: HttpRequest) -> HttpResponse:
    """현재 model_result 기반 OHLCV 캔들을 JSON으로 반환합니다.

    프런트엔드에서 주기적으로 호출해 실시간 차트처럼 사용합니다.
    """
    try:
        _, df, _ = _load_model_result()
        limit_raw = request.GET.get("limit")
        try:
            limit = int(limit_raw) if limit_raw is not None else CHART_CANDLES
        except ValueError:
            limit = CHART_CANDLES
        candles = _prepare_candles(df, limit=limit)
        return JsonResponse({"candles": candles})
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({"error": str(exc)}, status=500)

