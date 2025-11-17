from __future__ import annotations

"""
GPU / 학습 없이도 Django 대시보드를 테스트할 수 있도록
간단한 예시 결과가 들어 있는 model_result.csv / model_result_meta.json을 생성합니다.
"""

from pathlib import Path
import json

import pandas as pd
import numpy as np


def build_dummy_result() -> pd.DataFrame:
    # 10초 짜리 예시 시계열
    index = pd.date_range("2025-01-01 00:00:00", periods=10, freq="S")

    # 단순한 가격 경로 (조금씩 오른 뒤 살짝 조정)
    close = np.array([30000, 30010, 30020, 30035, 30050, 30045, 30040, 30060, 30080, 30070], dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 5
    low = np.minimum(open_, close) - 5
    volume = np.linspace(0.8, 1.2, len(index)) * 10_000

    price_change = close - open_
    price_range = high - low

    # buy/sell pressure 간단히 구성
    buy_pressure = np.maximum(price_change, 0) * volume
    sell_pressure = np.maximum(-price_change, 0) * volume
    total_pressure = buy_pressure + sell_pressure + 1e-9
    buy_ratio = buy_pressure / total_pressure
    imbalance = buy_pressure - sell_pressure

    # LSTM 결과가 나왔다고 가정하고 예시 확률/시그널
    prob_up = np.array([0.48, 0.51, 0.55, 0.62, 0.58, 0.53, 0.47, 0.60, 0.66, 0.52])
    pred_direction = (prob_up >= 0.5).astype(int)
    signal = np.sign(imbalance)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "price_change": price_change,
            "range": price_range,
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "buy_ratio": buy_ratio,
            "imbalance": imbalance,
            "prob_up": prob_up,
            "pred_direction": pred_direction,
            "signal": signal,
        }
    )
    return df


def main() -> None:
    df = build_dummy_result()
    out_dir = Path("dataset")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "model_result.csv"
    meta_path = out_dir / "model_result_meta.json"

    df.to_csv(csv_path, index=False)

    meta = {
        "loss": 0.42,
        "accuracy": 0.73,
        "window": 60,
        "epochs": 3,
        "batch_size": 32,
        "lr": 0.001,
        "symbol": "BTCUSDT",
        "tail_rows": None,
        "device": "dummy",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"dummy model_result.csv 생성: {csv_path.resolve()}")
    print(f"dummy model_result_meta.json 생성: {meta_path.resolve()}")


if __name__ == "__main__":
    main()

