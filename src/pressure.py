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
    # order-flow 피처 (가능한 경우에만)
    vol = df["volume"].astype(float)
    if "taker_buy_base_asset_volume" in df.columns:
        tb_base = df["taker_buy_base_asset_volume"].astype(float)
        df["taker_buy_ratio"] = tb_base / (vol + 1e-9)
        # net taker volume (buy - sell)
        df["taker_net_volume"] = tb_base - (vol - tb_base)
        df["taker_imbalance"] = df["taker_net_volume"] / (vol + 1e-9)
    if "number_of_trades" in df.columns:
        trades = df["number_of_trades"].astype(float)
        df["avg_trade_size"] = vol / (trades + 1e-9)
        df["log_trades"] = np.log1p(trades)
        df["volume_per_trade"] = df["avg_trade_size"]
    return df
