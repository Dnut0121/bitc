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

