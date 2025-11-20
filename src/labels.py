from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


LabelValue = Literal[-1, 0, 1]


def make_cost_aware_labels(
    close: pd.Series,
    horizon: int,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    margin_rate: float = 0.0,
) -> pd.Series:
    """수수료/슬리피지를 반영한 단순 3클래스 라벨을 생성합니다.

    정의 (롱/숏 1회 진입+청산 가정):
        - 진입가: price_t = close[t]
        - 청산가: price_T = close[t + horizon]
        - 총 비용(수수료 + 슬리피지, 왕복):
              roundtrip_cost = (fee_rate + slippage_rate) * (price_t + price_T)
        - 총 수익:
              gross_profit = price_T - price_t
        - 순익 (수수료/슬리피지 차감):
              net_profit = gross_profit - roundtrip_cost
        - 진입가 기준 순익률:
              net_ret = net_profit / price_t

        net_ret >  margin_rate  → +1 (롱으로 진입할 가치가 있음)
        net_ret < -margin_rate  → -1 (숏으로 진입할 가치가 있음)
        그 외                     0 (거래 없음 또는 애매)

    마지막 horizon 구간은 미래 가격이 없으므로 0으로 채워집니다.

    Parameters
    ----------
    close:
        종가 시계열 (DatetimeIndex 또는 일반 Index 상관 없음).
    horizon:
        몇 초(또는 몇 스텝) 뒤까지의 움직임을 볼지.
    fee_rate:
        한 번 체결될 때 수수료 비율 (예: 0.0005 = 0.05%).
    slippage_rate:
        한 번 체결될 때 슬리피지 비율.
    margin_rate:
        수수료/슬리피지를 모두 제하고도 이 정도(비율) 이상 남을 때만
        의미 있는 기회로 간주하기 위한 추가 여유 마진.

    Returns
    -------
    pd.Series
        값이 {-1, 0, 1}인 시리즈. close와 동일한 인덱스를 가집니다.
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")
    if len(close) <= horizon:
        # 미래 가격을 볼 수 있는 위치가 하나도 없음
        return pd.Series(data=np.zeros(len(close), dtype=np.int8), index=close.index)

    close = close.astype(float)
    future_close = close.shift(-horizon)

    # 왕복(진입+청산) 기준 수수료/슬리피지 비용을 진입가 기준 수익률로 근사
    price_ratio = future_close / close
    # price_ratio가 NaN인 구간(마지막 horizon 구간)은 이후에서 0으로 떨어지도록 처리
    roundtrip_cost_ret = (fee_rate + slippage_rate) * (1.0 + price_ratio)
    gross_ret = (future_close - close) / close
    net_ret = gross_ret - roundtrip_cost_ret

    labels = pd.Series(data=np.zeros(len(close), dtype=np.int8), index=close.index)

    # NaN (미래 가격 없는 구간)은 그대로 0으로 남겨 둔다.
    valid_mask = net_ret.notna()
    if valid_mask.any():
        net_ret_valid = net_ret[valid_mask]
        long_mask = net_ret_valid > margin_rate
        short_mask = net_ret_valid < -margin_rate
        labels.loc[net_ret_valid.index[long_mask]] = 1
        labels.loc[net_ret_valid.index[short_mask]] = -1

    return labels


def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon: int,
    up_pct: float,
    down_pct: float,
) -> pd.Series:
    """고전적인 Triple-Barrier 라벨링을 단순화해서 구현합니다.

    - 진입 시점 t 의 기준 가격: close[t]
    - 위 배리어 (익절):  close[t] * (1 + up_pct)
    - 아래 배리어 (손절): close[t] * (1 - down_pct)
    - 시간 배리어:       t + horizon

    t 이후 horizon 구간 동안 high/low가 위/아래 배리어에 먼저 닿는지 확인하여:
        위 먼저  → +1
        아래 먼저 → -1
        아무 것도 안 닿고 시간 배리어 도달 → 0

    intrabar에서 위/아래를 동시에 터치했는지까지 구분할 수 없으므로,
    동일 시점에서 둘 다 조건을 만족하면 순서 없이
    (위 배리어를 우선하는 식 등) 단순 규칙으로 라벨링합니다.
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")
    if not isinstance(close, pd.Series) or not isinstance(high, pd.Series) or not isinstance(low, pd.Series):
        raise TypeError("close, high, low must all be pandas Series.")
    if not (len(close) == len(high) == len(low)):
        raise ValueError("close, high, low must have the same length.")

    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    close_values = close.to_numpy(dtype=float)
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)

    for i in range(n):
        p0 = close_values[i]
        if np.isnan(p0):
            continue
        up_barrier = p0 * (1.0 + up_pct)
        down_barrier = p0 * (1.0 - down_pct)
        end = min(n, i + 1 + horizon)
        # 최소 1스텝 이후부터 살펴본다.
        for j in range(i + 1, end):
            hit_up = high_values[j] >= up_barrier
            hit_down = low_values[j] <= down_barrier
            if hit_up and not hit_down:
                labels[i] = 1
                break
            if hit_down and not hit_up:
                labels[i] = -1
                break
            if hit_up and hit_down:
                # 동시에 터치는 드물고 정보가 부족하므로 보수적으로 0으로 둔다.
                labels[i] = 0
                break
        # 루프를 끝까지 돌고 아무 배리어도 안 맞으면 0 유지 (시간 배리어).

    return pd.Series(labels, index=close.index, name="triple_barrier")


def multi_horizon_cost_labels(
    close: pd.Series,
    horizons: list[int],
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    margin_rate: float = 0.0,
) -> pd.DataFrame:
    """여러 horizon 에 대해 cost-aware 3클래스 라벨을 한 번에 생성합니다.

    각 horizon H 에 대해 make_cost_aware_labels(close, H, ...) 를 호출하여
    컬럼 이름을 f\"H_{H}s\" 형태로 붙인 DataFrame을 반환합니다.
    """
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")
    unique_h = sorted({int(h) for h in horizons if h > 0})
    if not unique_h:
        raise ValueError("horizons must contain at least one positive integer.")

    data: dict[str, pd.Series] = {}
    for h in unique_h:
        labels = make_cost_aware_labels(
            close,
            horizon=h,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            margin_rate=margin_rate,
        )
        col_name = f"H_{h}s"
        data[col_name] = labels

    return pd.DataFrame(data, index=close.index)


