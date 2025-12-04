from __future__ import annotations

"""Simple event-based backtester for LSTM probability outputs."""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    total_return: float
    sharpe: float
    max_drawdown: float
    hit_ratio: float
    avg_r_multiple: float
    profit_factor: float
    num_trades: int
    # 선택적으로 개별 트레이드 기록을 함께 보관 (일부 백테스트에서만 사용)
    trades: "list[TradeRecord] | None" = None


@dataclass
class TradeRecord:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    horizon: int
    holding_period: int
    exit_reason: str
    net_pnl: float
    r_multiple: float


def _compute_performance(pnl_series: pd.Series) -> tuple[float, float]:
    """누적 PnL 시계열에서 총수익률과 Sharpe (초당 기준)을 계산합니다."""
    if pnl_series.empty:
        return 0.0, 0.0
    equity = pnl_series.cumsum()
    total_return = equity.iloc[-1]
    returns = pnl_series  # 1초 단위 PnL 을 수익률처럼 사용 (상대 척도: position 크기 기준)
    vol = returns.std(ddof=0)
    sharpe = (returns.mean() / vol) * np.sqrt(86_400) if vol > 0 else 0.0
    return float(total_return), float(sharpe)


def run_event_backtest(
    df: pd.DataFrame,
    window: int,
    prob_threshold_long: float,
    prob_threshold_short: float,
    horizon: int,
    fee_rate: float,
    slippage_rate: float,
    *,
    record_trades: bool = False,
) -> BacktestResult:
    """LSTM의 prob_up 시그널을 사용한 매우 단순한 이벤트 기반 백테스트.

    전략 개요:
        - 포지션 없음 상태에서:
            prob_up >= prob_threshold_long → 롱 진입
            prob_up <= prob_threshold_short → 숏 진입
        - 진입 시점의 최근 window초 평균 range 를 기준으로
          손익비 2:1 정도의 목표/손절 폭을 설정.
        - 손절/익절/최대 보유시간(horizon) 중 먼저 도달하는 시점에 청산.
        - 포지션 크기는 고정 (1 계약)으로 두고, PnL은 price 차이로 계산.
        - 수수료/슬리피지는 진입/청산 시점 양쪽에 fee_rate + slippage_rate 를
          단순히 차감하는 방식으로 근사.

    Risk rule (간단 버전):
        - 일간 손실이 -3.0 단위를 넘으면 (price 차이 기준) 그날 이후 신규 진입 중단.
        - 연속 손실 5회 이상이면 더 이상 신규 진입하지 않음.
    """
    required_cols = {"close", "prob_up"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"run_event_backtest: DataFrame is missing columns: {sorted(missing)}")
    if len(df) < window + horizon + 5:
        raise ValueError("run_event_backtest: not enough rows for the requested window/horizon.")

    close = df["close"].to_numpy(dtype=float)
    prob_up = df["prob_up"].to_numpy(dtype=float)
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("run_event_backtest: DataFrame index must be a DatetimeIndex.")

    total_cost_rate = max(0.0, fee_rate + slippage_rate)

    in_position = False
    is_long = True
    entry_price = 0.0
    entry_idx = 0
    target_price = 0.0
    stop_price = 0.0

    pnl_per_step = np.zeros(len(df), dtype=float)
    trade_pnls: list[float] = []
    trade_rs: list[float] = []
    trade_records: list[TradeRecord] = []
    wins = 0
    losses = 0
    max_daily_loss = -3.0
    max_consec_losses = 5
    daily_pnl: Dict[pd.Timestamp, float] = {}
    consec_losses = 0

    for i in range(len(df)):
        ts = index[i]
        price = close[i]
        p = prob_up[i]

        day = ts.normalize()
        day_pnl = daily_pnl.get(day, 0.0)

        # 일간 손실/연속 손실 제한으로 신규 진입 차단
        allow_new_trade = (day_pnl > max_daily_loss) and (consec_losses < max_consec_losses)

        if not in_position and allow_new_trade:
            if p >= prob_threshold_long:
                # 롱 진입
                in_position = True
                is_long = True
                entry_price = price
                entry_idx = i
                # 최근 window초 평균 변동폭을 기반으로 목표/손절 폭을 설정
                if "range" in df.columns:
                    avg_range = float(df["range"].iloc[max(0, i - window + 1) : i + 1].mean())
                else:
                    avg_range = float(
                        (df["high"] - df["low"]).iloc[max(0, i - window + 1) : i + 1].mean()
                    )
                tp = avg_range * 2.0
                sl = avg_range
                target_price = entry_price + tp
                stop_price = entry_price - sl
            elif p <= prob_threshold_short:
                # 숏 진입
                in_position = True
                is_long = False
                entry_price = price
                entry_idx = i
                if "range" in df.columns:
                    avg_range = float(df["range"].iloc[max(0, i - window + 1) : i + 1].mean())
                else:
                    avg_range = float(
                        (df["high"] - df["low"]).iloc[max(0, i - window + 1) : i + 1].mean()
                    )
                tp = avg_range * 2.0
                sl = avg_range
                target_price = entry_price - tp
                stop_price = entry_price + sl
            # 진입 없으면 다음 시점으로
            continue

        if in_position:
            # 포지션 유지 중 평가손익
            if is_long:
                unrealized = price - entry_price
            else:
                unrealized = entry_price - price

            # 종료 조건 체크: TP / SL / 최대 보유시간
            holding = i - entry_idx
            exit_reason = None
            if is_long and price >= target_price:
                exit_reason = "tp"
            elif is_long and price <= stop_price:
                exit_reason = "sl"
            elif (not is_long) and price <= target_price:
                exit_reason = "tp"
            elif (not is_long) and price >= stop_price:
                exit_reason = "sl"
            elif holding >= horizon:
                exit_reason = "time"

            if exit_reason is not None:
                # 수수료/슬리피지 차감
                gross = unrealized
                roundtrip_cost = total_cost_rate * (entry_price + price)
                net = gross - roundtrip_cost
                pnl_per_step[i] = net
                trade_pnls.append(net)
                r_multiple = net / (abs(stop_price - entry_price) + 1e-9)
                trade_rs.append(r_multiple)
                if record_trades:
                    trade_records.append(
                        TradeRecord(
                            entry_time=index[entry_idx],
                            exit_time=ts,
                            direction="long" if is_long else "short",
                            entry_price=entry_price,
                            exit_price=price,
                            tp_price=target_price,
                            sl_price=stop_price,
                            horizon=horizon,
                            holding_period=holding,
                            exit_reason=exit_reason,
                            net_pnl=net,
                            r_multiple=r_multiple,
                        )
                    )

                if net > 0:
                    wins += 1
                    consec_losses = 0
                else:
                    losses += 1
                    consec_losses += 1

                # 일간 손익 업데이트
                daily_pnl[day] = daily_pnl.get(day, 0.0) + net

                in_position = False
                entry_price = 0.0
                target_price = 0.0
                stop_price = 0.0
                last_exit_idx = i

    pnl_series = pd.Series(pnl_per_step, index=index)
    total_ret, sharpe = _compute_performance(pnl_series)

    num_trades = len(trade_pnls)
    hit_ratio = wins / num_trades if num_trades > 0 else 0.0
    avg_r = float(np.mean(trade_rs)) if trade_rs else 0.0
    profits = [p for p in trade_pnls if p > 0]
    losses_list = [p for p in trade_pnls if p < 0]
    gross_profit = float(sum(profits))
    gross_loss = -float(sum(losses_list)) if losses_list else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    # Max drawdown: equity 곡선에서 계산
    equity = pnl_series.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    return BacktestResult(
        total_return=float(total_ret),
        sharpe=float(sharpe),
        max_drawdown=max_dd,
        hit_ratio=float(hit_ratio),
        avg_r_multiple=avg_r,
        profit_factor=float(profit_factor),
        num_trades=int(num_trades),
        trades=trade_records if record_trades else None,
    )


def run_quantile_long_short_backtest(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    risk_reward: float,
    fee_rate: float,
    slippage_rate: float,
    q_long: float,
    q_short: float,
    min_range_quantile: float = 0.5,
    min_volume_quantile: float = 0.5,
    consistency_lookback: int = 30,
    max_daily_loss: float | None = None,
    max_consec_losses: int | None = None,
    meta_prob: np.ndarray | pd.Series | None = None,
    meta_prob_long: np.ndarray | pd.Series | None = None,
    meta_prob_short: np.ndarray | pd.Series | None = None,
    meta_threshold: float | None = None,
    min_entry_gap: int = 0,
    ma_window: int | None = None,
    rsi_window: int | None = None,
    rsi_overbought: float | None = None,
    rsi_oversold: float | None = None,
) -> BacktestResult:
    """분위수 기반 long/short 전략을 수행하는 백테스트.

    - prob_up >= q_long 분위수 이상이면 롱 후보
    - prob_up <= q_short 분위수 이하면 숏 후보
    - 필터:
        - range/volume 이 각각 min_range_quantile/min_volume_quantile 분위수 이상
        - 최근 consistency_lookback 초 수익률 방향과 시그널 방향이 일치
    - 진입 시점 ATR(window) 를 기준으로 TP/SL 설정 (risk_reward:1 비율)
    """
    required_cols = {"close", "prob_up"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"run_quantile_long_short_backtest: missing columns: {sorted(missing)}")

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float) if "high" in df.columns else close
    low = df["low"].to_numpy(dtype=float) if "low" in df.columns else close
    prob_up = df["prob_up"].to_numpy(dtype=float)
    rng = df.get("range", (df["high"] - df["low"])).to_numpy(dtype=float)  # type: ignore[index]
    volume = df.get("volume", pd.Series(np.ones_like(close), index=df.index)).to_numpy(dtype=float)
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("run_quantile_long_short_backtest: DataFrame index must be a DatetimeIndex.")
    if len(df) < window + horizon + 5:
        raise ValueError("run_quantile_long_short_backtest: not enough rows for the requested window/horizon.")

    # 분위수 기반 threshold 계산
    thr_long = float(np.quantile(prob_up, q_long))
    thr_short = float(np.quantile(prob_up, q_short))
    meta_thr = 0.5 if meta_threshold is None else float(meta_threshold)

    range_thr = float(np.quantile(rng, min_range_quantile))
    vol_thr = float(np.quantile(volume, min_volume_quantile))
    # ATR (Average True Range)를 window 기준으로 계산하여 SL/TP 폭에 사용
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr_high_low = high - low
    tr_high_close = np.abs(high - prev_close)
    tr_low_close = np.abs(low - prev_close)
    true_range = np.maximum.reduce([tr_high_low, tr_high_close, tr_low_close])
    atr_series = pd.Series(true_range).rolling(window, min_periods=1).mean()
    atr = atr_series.to_numpy(dtype=float)
    # 보조 필터용 MA/RSI 계산 (옵션)
    ma_series = None
    if ma_window is not None and ma_window > 1:
        ma_series = pd.Series(close).rolling(ma_window, min_periods=ma_window).mean().to_numpy(dtype=float)
    rsi_series = None
    if rsi_window is not None and rsi_window > 1:
        diff = pd.Series(close).diff()
        gains = diff.clip(lower=0)
        losses = -diff.clip(upper=0)
        avg_gain = gains.rolling(rsi_window, min_periods=rsi_window).mean()
        avg_loss = losses.rolling(rsi_window, min_periods=rsi_window).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_series = (100.0 - (100.0 / (1.0 + rs))).to_numpy(dtype=float)
    # range 분위수가 0이거나 매우 작을 때 TP/SL 폭이 0이 되는 것을 방지하기 위해
    # 가격 스케일을 기반으로 한 최소 폭을 설정한다.
    median_price = float(np.median(close)) if len(close) > 0 else 1.0
    min_range_floor = max(range_thr, median_price * 1e-6, 1e-9)
    range_thr = max(range_thr, min_range_floor)

    total_cost_rate = max(0.0, fee_rate + slippage_rate)

    in_position = False
    is_long = True
    entry_price = 0.0
    entry_idx = 0
    target_price = 0.0
    stop_price = 0.0

    pnl_per_step = np.zeros(len(df), dtype=float)
    trade_pnls: list[float] = []
    trade_rs: list[float] = []
    trade_records: list[TradeRecord] = []
    wins = 0
    losses = 0
    daily_pnl: Dict[pd.Timestamp, float] = {}
    consec_losses = 0
    # 메타 확률 (롱/숏 별) 정렬
    def _to_arr(arr: np.ndarray | pd.Series | None) -> np.ndarray | None:
        if arr is None:
            return None
        out = np.asarray(arr, dtype=float).reshape(-1)
        if out.shape[0] != len(df):
            raise ValueError(
                f"run_quantile_long_short_backtest: meta_prob length {out.shape[0]} "
                f"does not match df length {len(df)}"
            )
        return out

    meta_long = _to_arr(meta_prob_long)
    if meta_long is None:
        meta_long = _to_arr(meta_prob)
    meta_short = _to_arr(meta_prob_short)
    if meta_short is None:
        meta_short = _to_arr(meta_prob)
    last_exit_idx = -10_000_000

    for i in range(len(df)):
        ts = index[i]
        price = close[i]
        p = prob_up[i]
        r = rng[i]
        v = volume[i]
        mp_long = meta_long[i] if meta_long is not None else None
        mp_short = meta_short[i] if meta_short is not None else None

        day = ts.normalize()
        day_pnl = daily_pnl.get(day, 0.0)

        allow_new_trade = True
        if max_daily_loss is not None:
            allow_new_trade = allow_new_trade and (day_pnl > max_daily_loss)
        if max_consec_losses is not None:
            allow_new_trade = allow_new_trade and (consec_losses < max_consec_losses)
        if min_entry_gap > 0:
            allow_new_trade = allow_new_trade and ((i - last_exit_idx) >= min_entry_gap)

        if not in_position and allow_new_trade:
            # 필터를 단순화하여 진입 빈도를 확보:
            # - 분위수 기반 prob_up 기준만 사용 (롱/숏)
            # - range/volume/방향 일관성 필터는 비활성화
            long_candidate = p >= thr_long
            short_candidate = p <= thr_short
            # 메타 모델 확률 필터가 주어졌다면 롱/숏 각각에 추가 적용
            if meta_long is not None:
                long_candidate = long_candidate and (mp_long is not None and mp_long >= meta_thr)
            if meta_short is not None:
                short_candidate = short_candidate and (mp_short is not None and mp_short >= meta_thr)
            # MA 필터: 롱은 종가가 MA 위, 숏은 MA 아래일 때만
            if ma_series is not None and not np.isnan(ma_series[i]):
                long_candidate = long_candidate and (price >= ma_series[i])
                short_candidate = short_candidate and (price <= ma_series[i])
            # RSI 필터: 롱은 과매도 이하, 숏은 과매수 이상일 때만
            if rsi_series is not None and not np.isnan(rsi_series[i]):
                if rsi_oversold is not None:
                    long_candidate = long_candidate and (rsi_series[i] <= rsi_oversold)
                if rsi_overbought is not None:
                    short_candidate = short_candidate and (rsi_series[i] >= rsi_overbought)

            # TP/SL 폭은 개별 캔들의 range 가 아니라
            # 최근 window 구간 평균 range 를 기준으로 설정하여
            # 지나치게 작은 stop 으로 인해 R 값이 폭주하는 것을 방지한다.
            if long_candidate or short_candidate:
                # ATR 기반 SL/TP 폭 계산 (초기 구간 NaN이면 최근 range 평균으로 보강)
                atr_val = float(atr[i]) if i < len(atr) else float("nan")
                if np.isnan(atr_val) or atr_val <= 0:
                    if "range" in df.columns:
                        atr_val = float(df["range"].iloc[max(0, i - window + 1) : i + 1].mean())
                    else:
                        atr_val = float(
                            (df["high"] - df["low"]).iloc[max(0, i - window + 1) : i + 1].mean()
                        )
                # range 분위수가 0에 가까울 때는 현재 가격 스케일 기반 최소 폭으로 보정
                price_floor = max(range_thr, abs(price) * 1e-6, 1e-9)
                # 거래 비용(수수료+슬리피지)보다 작은 폭으로는 진입하지 않도록 한다.
                # (cost * 2: 진입/청산 왕복 비용, 1.1 배 완충)
                cost_floor = total_cost_rate * abs(price) * 2.0 * 1.1
                eff_range = max(atr_val, price_floor, cost_floor)

            if long_candidate:
                in_position = True
                is_long = True
                entry_price = price
                entry_idx = i
                tp = eff_range * risk_reward
                sl = eff_range
                target_price = entry_price + tp
                stop_price = entry_price - sl
            elif short_candidate:
                in_position = True
                is_long = False
                entry_price = price
                entry_idx = i
                tp = eff_range * risk_reward
                sl = eff_range
                target_price = entry_price - tp
                stop_price = entry_price + sl
            continue

        if in_position:
            unrealized = (price - entry_price) if is_long else (entry_price - price)

            holding = i - entry_idx
            exit_reason = None
            if is_long and price >= target_price:
                exit_reason = "tp"
            elif is_long and price <= stop_price:
                exit_reason = "sl"
            elif (not is_long) and price <= target_price:
                exit_reason = "tp"
            elif (not is_long) and price >= stop_price:
                exit_reason = "sl"
            elif holding >= horizon:
                exit_reason = "time"

            if exit_reason is not None:
                # 슬리피지 없이 TP/SL 가격으로 체결된 것으로 간주해
                # 지나치게 큰 손익(R 배수)을 방지한다.
                fill_price = price
                if exit_reason == "tp":
                    fill_price = target_price
                elif exit_reason == "sl":
                    fill_price = stop_price

                if is_long:
                    gross = fill_price - entry_price
                else:
                    gross = entry_price - fill_price

                roundtrip_cost = total_cost_rate * (entry_price + fill_price)
                net = gross - roundtrip_cost
                pnl_per_step[i] = net
                trade_pnls.append(net)
                r_multiple = net / (abs(stop_price - entry_price) + 1e-9)
                trade_rs.append(r_multiple)

                if net > 0:
                    wins += 1
                    consec_losses = 0
                else:
                    losses += 1
                    consec_losses += 1

                daily_pnl[day] = daily_pnl.get(day, 0.0) + net

                # 트레이드 기록 저장
                trade_records.append(
                    TradeRecord(
                        entry_time=index[entry_idx],
                        exit_time=ts,
                        direction="long" if is_long else "short",
                        entry_price=float(entry_price),
                        exit_price=float(fill_price),
                        tp_price=float(target_price),
                        sl_price=float(stop_price),
                        horizon=int(horizon),
                        holding_period=int(holding),
                        exit_reason=exit_reason,
                        net_pnl=float(net),
                        r_multiple=float(r_multiple),
                    )
                )

                in_position = False
                entry_price = 0.0
                target_price = 0.0
                stop_price = 0.0

    pnl_series = pd.Series(pnl_per_step, index=index)
    total_ret, sharpe = _compute_performance(pnl_series)

    num_trades = len(trade_pnls)
    hit_ratio = wins / num_trades if num_trades > 0 else 0.0
    avg_r = float(np.mean(trade_rs)) if trade_rs else 0.0
    profits = [p for p in trade_pnls if p > 0]
    losses_list = [p for p in trade_pnls if p < 0]
    gross_profit = float(sum(profits))
    gross_loss = -float(sum(losses_list)) if losses_list else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    equity = pnl_series.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    return BacktestResult(
        total_return=float(total_ret),
        sharpe=float(sharpe),
        max_drawdown=max_dd,
        hit_ratio=float(hit_ratio),
        avg_r_multiple=avg_r,
        profit_factor=float(profit_factor),
        num_trades=int(num_trades),
        trades=trade_records,
    )
