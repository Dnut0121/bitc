#!/usr/bin/env python3
"""간단하고 검증된 기술적 전략: 볼린저 밴드 + RSI 조합

prob_up 대신 검증된 기술적 지표를 사용하여 실제 수익을 낼 수 있는 전략을 구현합니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pandas as pd
import numpy as np

from src.data import load_ohlcv
from src.backtest import BacktestResult


def calculate_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """볼린저 밴드 계산"""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, ma, lower


def calculate_rsi(close: pd.Series, window: int = 14):
    """RSI 계산"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def run_technical_backtest(
    df: pd.DataFrame,
    bb_window: int = 20,
    bb_std: float = 2.0,
    rsi_window: int = 14,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    horizon: int = 120,
    risk_reward: float = 2.0,
    fee_rate: float = 0.0006,
    slippage_rate: float = 0.0003,
) -> BacktestResult:
    """볼린저 밴드 + RSI 전략 백테스트
    
    전략:
    - 롱 진입: 가격이 하단 밴드 아래 + RSI < oversold
    - 숏 진입: 가격이 상단 밴드 위 + RSI > overbought
    - TP/SL: ATR 기반으로 설정
    """
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    
    # 지표 계산
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, bb_window, bb_std)
    rsi = calculate_rsi(close, rsi_window)
    
    # ATR 계산
    prev_close = close.shift(1)
    tr_high_low = high - low
    tr_high_close = (high - prev_close).abs()
    tr_low_close = (low - prev_close).abs()
    true_range = pd.concat([tr_high_low, tr_high_close, tr_low_close], axis=1).max(axis=1)
    atr = true_range.rolling(bb_window).mean()
    
    total_cost_rate = fee_rate + slippage_rate
    
    in_position = False
    is_long = True
    entry_price = 0.0
    entry_idx = 0
    target_price = 0.0
    stop_price = 0.0
    
    pnl_per_step = np.zeros(len(df))
    trade_pnls = []
    trade_rs = []
    wins = 0
    losses = 0
    
    for i in range(max(bb_window, rsi_window), len(df)):
        price = close.iloc[i]
        
        if pd.isna(bb_upper.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        
        if not in_position:
            # 롱 진입 조건: 가격이 하단 밴드 터치 + RSI 과매도
            long_signal = (price <= bb_lower.iloc[i] * 1.001) and (rsi.iloc[i] < rsi_oversold)
            
            # 숏 진입 조건: 가격이 상단 밴드 터치 + RSI 과매수
            short_signal = (price >= bb_upper.iloc[i] * 0.999) and (rsi.iloc[i] > rsi_overbought)
            
            atr_val = atr.iloc[i]
            if atr_val <= 0 or pd.isna(atr_val):
                atr_val = price * 0.001  # 최소값
            
            # ATR을 1.5배로 확대하여 노이즈 회피
            atr_val *= 1.5
            
            if long_signal:
                in_position = True
                is_long = True
                entry_price = price * (1 + slippage_rate)
                entry_idx = i
                target_price = entry_price + (atr_val * risk_reward)
                stop_price = entry_price - atr_val
            elif short_signal:
                in_position = True
                is_long = False
                entry_price = price * (1 - slippage_rate)
                entry_idx = i
                target_price = entry_price - (atr_val * risk_reward)
                stop_price = entry_price + atr_val
        
        if in_position:
            holding = i - entry_idx
            exit_reason = None
            
            if is_long:
                if price >= target_price:
                    exit_reason = "tp"
                elif price <= stop_price:
                    exit_reason = "sl"
                elif holding >= horizon:
                    exit_reason = "time"
            else:
                if price <= target_price:
                    exit_reason = "tp"
                elif price >= stop_price:
                    exit_reason = "sl"
                elif holding >= horizon:
                    exit_reason = "time"
            
            if exit_reason:
                fill_price = price
                if exit_reason == "tp":
                    fill_price = target_price if exit_reason == "tp" else fill_price
                    if is_long:
                        fill_price *= (1 - slippage_rate * 0.5)
                    else:
                        fill_price *= (1 + slippage_rate * 0.5)
                elif exit_reason == "sl":
                    fill_price = stop_price
                    if is_long:
                        fill_price *= (1 - slippage_rate)
                    else:
                        fill_price *= (1 + slippage_rate)
                else:  # time
                    if is_long:
                        fill_price *= (1 - slippage_rate)
                    else:
                        fill_price *= (1 + slippage_rate)
                
                if is_long:
                    gross = fill_price - entry_price
                else:
                    gross = entry_price - fill_price
                
                cost = fee_rate * (entry_price + fill_price)
                net = gross - cost
                pnl_per_step[i] = net
                trade_pnls.append(net)
                
                r_multiple = net / (abs(stop_price - entry_price) + 1e-9)
                trade_rs.append(r_multiple)
                
                if net > 0:
                    wins += 1
                else:
                    losses += 1
                
                in_position = False
                entry_price = 0.0
    
    # 성능 계산
    pnl_series = pd.Series(pnl_per_step, index=df.index)
    total_ret = pnl_series.sum()
    
    returns = pnl_series[pnl_series != 0]
    if len(returns) > 0:
        vol = returns.std()
        sharpe = (returns.mean() / vol) * np.sqrt(86_400) if vol > 0 else 0.0
    else:
        sharpe = 0.0
    
    num_trades = len(trade_pnls)
    hit_ratio = wins / num_trades if num_trades > 0 else 0.0
    avg_r = float(np.mean(trade_rs)) if trade_rs else 0.0
    
    profits = [p for p in trade_pnls if p > 0]
    losses_list = [p for p in trade_pnls if p < 0]
    gross_profit = sum(profits)
    gross_loss = -sum(losses_list) if losses_list else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    
    equity = pnl_series.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min() if not drawdown.empty else 0.0
    
    return BacktestResult(
        total_return=total_ret,
        sharpe=sharpe,
        max_drawdown=max_dd,
        hit_ratio=hit_ratio,
        avg_r_multiple=avg_r,
        profit_factor=profit_factor,
        num_trades=num_trades,
    )


def main():
    parser = argparse.ArgumentParser(description="볼린저 밴드 + RSI 전략 백테스트")
    parser.add_argument("--file", type=Path, required=True, help="검증 데이터 파일")
    parser.add_argument("--bb-window", type=int, default=20, help="볼린저 밴드 윈도우")
    parser.add_argument("--bb-std", type=float, default=2.0, help="볼린저 밴드 표준편차 배수")
    parser.add_argument("--rsi-window", type=int, default=14, help="RSI 윈도우")
    parser.add_argument("--rsi-oversold", type=float, default=30, help="RSI 과매도 기준")
    parser.add_argument("--rsi-overbought", type=float, default=70, help="RSI 과매수 기준")
    args = parser.parse_args()
    
    print(f"[기술적 전략] 파일: {args.file.name}")
    print(f"[기술적 전략] 볼린저 밴드: {args.bb_window}기간, {args.bb_std}σ")
    print(f"[기술적 전략] RSI: {args.rsi_window}기간, 과매도<{args.rsi_oversold}, 과매수>{args.rsi_overbought}")
    print()
    
    ohlcv = load_ohlcv(args.file)
    
    results = []
    for horizon in [90, 120, 180, 240]:
        for rr in [1.5, 2.0, 2.5, 3.0]:
            bt = run_technical_backtest(
                ohlcv,
                bb_window=args.bb_window,
                bb_std=args.bb_std,
                rsi_window=args.rsi_window,
                rsi_oversold=args.rsi_oversold,
                rsi_overbought=args.rsi_overbought,
                horizon=horizon,
                risk_reward=rr,
            )
            results.append({
                "horizon": horizon,
                "risk_reward": rr,
                "total_return": bt.total_return,
                "sharpe": bt.sharpe,
                "hit_ratio": bt.hit_ratio,
                "num_trades": bt.num_trades,
                "profit_factor": bt.profit_factor,
            })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("sharpe", ascending=False)
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    
    best = result_df.iloc[0]
    print(f"\n{'='*60}")
    print(f"최고 성능 설정:")
    print(f"  horizon={int(best['horizon'])}s, risk_reward={best['risk_reward']:.1f}")
    print(f"  총 수익: {best['total_return']:.2f}")
    print(f"  Sharpe: {best['sharpe']:.4f}")
    print(f"  승률: {best['hit_ratio']:.2%}")
    print(f"  거래 수: {int(best['num_trades'])}")
    print(f"  Profit Factor: {best['profit_factor']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
