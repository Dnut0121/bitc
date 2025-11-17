from __future__ import annotations

"""
간단한 패턴이 섞인 1초 OHLCV 데이터를 생성해서
`dataset/test_ohlcv.csv`로 저장하는 스크립트입니다.

- 구간별로:
  1) 완만한 상승
  2) 박스권(횡보)
  3) 완만한 하락
패턴을 섞어 두어서 LSTM/pressure 피처가 어느 정도 변화를 가지도록 합니다.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_test_ohlcv() -> pd.DataFrame:
    total_seconds = 900  # 15분치 1초 캔들
    index = pd.date_range("2025-01-01 00:00:00", periods=total_seconds, freq="S")

    # 기초 가격 경로: 3만 달러 부근에서 시작
    base_price = 30_000.0

    # 구간 나누기: 상승(0~300), 박스(300~600), 하락(600~900)
    up_len = 300
    box_len = 300
    down_len = total_seconds - up_len - box_len

    # 상승 구간: 초당 +0.5달러 정도의 작은 우상향 + 노이즈
    up_trend = base_price + np.linspace(0, 150, up_len)
    up_noise = np.random.normal(0, 2.0, up_len)
    up_prices = up_trend + up_noise

    # 박스 구간: 가운데 값 근처에서 랜덤 워크
    box_center = up_prices[-1]
    box_steps = np.random.normal(0, 3.0, box_len).cumsum()
    box_prices = box_center + box_steps

    # 하락 구간: 초당 -0.4달러 정도의 완만한 하락 + 노이즈
    down_start = box_prices[-1]
    down_trend = down_start + np.linspace(0, -120, down_len)
    down_noise = np.random.normal(0, 2.0, down_len)
    down_prices = down_trend + down_noise

    close = np.concatenate([up_prices, box_prices, down_prices])
    # open은 한 틱 전 close를 기준으로, 첫 캔들은 close와 동일
    open_ = np.concatenate([[close[0]], close[:-1]])

    # high/low는 open/close 대비 작은 스프레드를 주어 구성
    spread = np.random.uniform(1.0, 4.0, total_seconds)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    # 거래량: 구간에 따라 조금 다르게
    vol_up = np.random.uniform(0.5, 1.5, up_len)
    vol_box = np.random.uniform(0.2, 1.0, box_len)
    vol_down = np.random.uniform(0.4, 1.2, down_len)
    volume = np.concatenate([vol_up, vol_box, vol_down]) * 10_000

    df = pd.DataFrame(
        {
            "timestamp": index,  # ISO 문자열로 저장되지만 load_ohlcv에서 자동 처리
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return df


def main() -> None:
    df = generate_test_ohlcv()
    out_path = Path("dataset") / "test_ohlcv.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"생성 완료: {out_path} (rows={len(df)})")


if __name__ == "__main__":
    main()

