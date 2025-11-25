from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render


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
    data = df.tail(RECENT_ROWS).copy()
    if data.empty:
        return {"window": RECENT_ROWS}, []

    avg_prob_up = float(data["prob_up"].mean()) if "prob_up" in data.columns else float("nan")
    bullish_ratio = float((data["prob_up"] >= 0.55).sum()) / len(data) * 100.0 if "prob_up" in data.columns else 0.0
    signal_up_ratio = float((data.get("signal", 0) > 0).sum()) / len(data) * 100.0

    summary = {
        "window": RECENT_ROWS,
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

    try:
        snapshot, df, meta = _load_model_result()
        trade = _compute_trade_analysis(
            df,
            prob_threshold=prob_threshold,
            horizon=horizon,
            risk_reward=risk_reward,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            min_expected_pct=min_expected_pct,
        )
        recent_summary, recent_rows = _compute_recent_view(df)
        context["candles"] = _prepare_candles(df)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)

    context["snapshot"] = snapshot
    context["trade"] = trade
    context["recent_summary"] = recent_summary
    context["recent_rows"] = recent_rows
    context["config"] = {
        "prob_threshold": prob_threshold,
        "horizon": horizon,
        "risk_reward": risk_reward,
        "min_expected_pct": min_expected_pct,
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
    }
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

