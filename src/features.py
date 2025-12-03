from __future__ import annotations

"""Additional microstructure and regime-style feature engineering."""

from typing import Sequence

import numpy as np
import pandas as pd


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV 기반으로 간단한 마이크로구조/레짐 피처를 추가합니다.

    추가되는 컬럼 (가능한 경우):
        - ret_1s, ret_5s, ret_10s, ret_30s : 수익률
        - pos_in_range                     : (close - low) / (high - low)
        - rv_30s, rv_120s                  : 로그 수익률 기반 롤링 변동성
        - vol_roll_30s, vol_roll_120s      : 거래량 롤링 평균
        - vol_zscore_120s                  : 거래량 z-score (120초 기준)
        - regime_vol                       : 변동성 레벨 (0=low, 1=mid, 2=high)
    """
    required: Sequence[str] = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_microstructure_features: missing required columns: {missing}")

    out = df.copy()

    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    # 단기/중기 수익률
    for h in (1, 5, 10, 30):
        out[f"ret_{h}s"] = close.pct_change(h)

    # 당일 high/low 대비 위치
    rng = (out["high"] - out["low"]).astype(float)
    out["pos_in_range"] = (close - out["low"]) / (rng + 1e-9)

    # 로그 수익률 기반 롤링 변동성
    log_price = np.log(close.to_numpy())
    log_ret = pd.Series(log_price, index=out.index).diff()
    out["rv_30s"] = log_ret.rolling(30, min_periods=10).std()
    out["rv_120s"] = log_ret.rolling(120, min_periods=20).std()

    # 거래량 롤링 평균 및 z-score
    out["vol_roll_30s"] = volume.rolling(30, min_periods=10).mean()
    out["vol_roll_120s"] = volume.rolling(120, min_periods=20).mean()
    vol_mean_120 = out["vol_roll_120s"]
    vol_std_120 = volume.rolling(120, min_periods=20).std()
    out["vol_zscore_120s"] = (volume - vol_mean_120) / (vol_std_120 + 1e-9)

    # 변동성 레짐: rv_120s 기준 분위수로 3단계 구분
    rv = out["rv_120s"]
    q_low = rv.quantile(0.33)
    q_high = rv.quantile(0.66)
    regime = np.zeros(len(out), dtype=np.int8)
    regime[rv > q_high] = 2
    # 중간 레벨
    mid_mask = (rv > q_low) & (rv <= q_high)
    regime[mid_mask] = 1
    out["regime_vol"] = regime

    return out


def add_technical_features(
    df: pd.DataFrame,
    *,
    volume_windows: Sequence[int] = (20, 60),
    vwap_window: int = 60,
    price_pct_windows: Sequence[int] = (1, 5, 15, 60),
    roc_window: int = 30,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_window: int = 14,
    bb_window: int = 20,
    bb_k: float = 2.0,
    lag_steps: Sequence[int] = (1, 5, 10),
    rolling_windows: Sequence[int] = (20, 60),
) -> pd.DataFrame:
    """Add richer OHLCV-based technical features for hybrid models.

    Included signals:
      - Volume: ratio to rolling mean, VWAP deviation
      - Momentum: price change %, ROC
      - Indicators: MACD (fast/slow/signal), RSI, Bollinger bands
      - Lags: close/volume/(buy_ratio/imbalance if present)
      - Rolling stats: mean/std for close, volume, buy_ratio
    """
    required: Sequence[str] = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_technical_features: missing required columns: {missing}")

    out = df.copy()
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    # Volume ratios against rolling means
    for window in volume_windows:
        min_periods = max(5, window // 2)
        roll_vol = volume.rolling(window, min_periods=min_periods).mean()
        out[f"volume_ratio_{window}"] = volume / (roll_vol + 1e-9)

    # VWAP over a rolling window
    vwap_min_periods = max(5, vwap_window // 2)
    vol_sum = volume.rolling(vwap_window, min_periods=vwap_min_periods).sum()
    pv_sum = (close * volume).rolling(vwap_window, min_periods=vwap_min_periods).sum()
    vwap = pv_sum / (vol_sum + 1e-9)
    out[f"vwap_{vwap_window}"] = vwap
    out[f"vwap_dev_{vwap_window}"] = close - vwap
    out[f"vwap_pct_{vwap_window}"] = (close - vwap) / (vwap + 1e-9)

    # Price change percentages and rate of change
    for window in price_pct_windows:
        out[f"price_change_pct_{window}"] = close.pct_change(window)
    out[f"roc_{roc_window}"] = close.diff(roc_window) / (close.shift(roc_window) + 1e-9)

    # MACD & signal/histogram
    ema_fast = close.ewm(span=macd_fast, adjust=False, min_periods=macd_fast).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False, min_periods=macd_slow).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False, min_periods=macd_signal).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_gain = gain.ewm(alpha=1 / rsi_window, adjust=False, min_periods=rsi_window).mean()
    roll_loss = loss.ewm(alpha=1 / rsi_window, adjust=False, min_periods=rsi_window).mean()
    rs = roll_gain / (roll_loss + 1e-9)
    out[f"rsi_{rsi_window}"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    min_bb = max(5, bb_window // 2)
    rolling_close = close.rolling(bb_window, min_periods=min_bb)
    mid = rolling_close.mean()
    std = rolling_close.std()
    upper = mid + bb_k * std
    lower = mid - bb_k * std
    out[f"bb_mid_{bb_window}"] = mid
    out[f"bb_upper_{bb_window}"] = upper
    out[f"bb_lower_{bb_window}"] = lower
    out[f"bb_width_{bb_window}"] = (upper - lower) / (mid + 1e-9)
    out[f"bb_percent_{bb_window}"] = (close - lower) / ((upper - lower) + 1e-9)

    # Lag features
    lag_targets: list[str] = ["close", "volume"]
    if "buy_ratio" in out.columns:
        lag_targets.append("buy_ratio")
    if "imbalance" in out.columns:
        lag_targets.append("imbalance")
    for lag in lag_steps:
        for col in lag_targets:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)

    # Rolling statistics
    for window in rolling_windows:
        min_periods = max(5, window // 2)
        rc = close.rolling(window, min_periods=min_periods)
        rv = volume.rolling(window, min_periods=min_periods)
        out[f"roll_close_mean_{window}"] = rc.mean()
        out[f"roll_close_std_{window}"] = rc.std()
        out[f"roll_volume_mean_{window}"] = rv.mean()
        out[f"roll_volume_std_{window}"] = rv.std()
        if "buy_ratio" in out.columns:
            rbr = out["buy_ratio"].rolling(window, min_periods=min_periods)
            out[f"roll_buy_ratio_mean_{window}"] = rbr.mean()
            out[f"roll_buy_ratio_std_{window}"] = rbr.std()

    return out
