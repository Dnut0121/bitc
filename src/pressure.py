"""Feature engineering for aggressive buy/sell pressure."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_pressure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_change"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["buy_pressure"] = np.maximum(df["price_change"], 0) * df["volume"]
    df["sell_pressure"] = np.maximum(-df["price_change"], 0) * df["volume"]
    pressure = df["buy_pressure"] + df["sell_pressure"]
    df["buy_ratio"] = df["buy_pressure"] / (pressure + 1e-9)
    df["imbalance"] = df["buy_pressure"] - df["sell_pressure"]
    return df
