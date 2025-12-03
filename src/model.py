from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
from pathlib import Path
import logging
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        self.sequences = torch.from_numpy(sequences.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def build_sequences(
    df: pd.DataFrame,
    lookback: int,
    max_sequences: int | None = None,
    use_cost_labels: bool = False,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    margin_rate: float = 0.0,
    label_horizon: int = 1,
    feature_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    if label_horizon <= 0:
        raise ValueError("label_horizon must be a positive integer.")
    if feature_columns is None:
        # order-flow 피처가 있으면 강화된 조합을 사용하고,
        # 그렇지 않으면 기존 조합으로 폴백한다.
        advanced_features = ["buy_ratio", "price_change", "range", "taker_imbalance", "log_trades", "volume_per_trade"]
        if all(col in df.columns for col in advanced_features):
            feature_columns = advanced_features
        else:
            feature_columns = ["buy_ratio", "volume", "range", "price_change", "imbalance"]
    features: Sequence[str] = list(feature_columns)

    # 메모리 보호: 너무 긴 시계열이면 마지막 max_sequences 구간만 사용
    if max_sequences is None or max_sequences <= 0:
        max_sequences = 500_000
    max_rows_for_sequences = lookback + max_sequences - 1
    total_rows = len(df)
    if total_rows > max_rows_for_sequences:
        logger = logging.getLogger(__name__)
        logger.warning(
            "build_sequences: too many rows (%d); "
            "using only the last %d rows to limit memory.",
            total_rows,
            max_rows_for_sequences,
        )
        df = df.iloc[-max_rows_for_sequences:]

    feature_array = df[features].to_numpy(dtype=np.float32, copy=True)
    if use_cost_labels:
        from .labels import make_cost_aware_labels

        # label_horizon 기준으로, 수수료/슬리피지를 제하고도 롱이 유리한 구간을 1로,
        # 그 외(중립/숏 유리)를 0으로 두고 이진 타깃으로 사용한다.
        cost_labels = make_cost_aware_labels(
            df["close"],
            horizon=label_horizon,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            margin_rate=margin_rate,
        )
        target_series = (cost_labels == 1).astype(int)
    else:
        # label_horizon 초 뒤의 종가가 현재보다 높은지 여부를 타깃으로 사용
        shifted = df["close"].shift(-label_horizon)
        target_series = (shifted > df["close"]).astype(int)

    # 마지막 label_horizon 구간은 타깃을 정의할 수 없으므로 잘라낸다.
    usable_len = max(0, len(df) - label_horizon)
    feature_array = feature_array[:usable_len]
    target_array = target_series.to_numpy()[:usable_len]
    if feature_array.shape[0] < lookback + 1:
        raise ValueError("Not enough samples to construct the requested lookback sequences.")

    sequences: list[np.ndarray] = []
    targets: list[int] = []
    indexes: list[pd.Timestamp] = []
    max_start = feature_array.shape[0] - lookback
    for start in range(max_start + 1):
        end = start + lookback
        sequences.append(feature_array[start:end])
        # 시퀀스의 마지막 시점(end-1)을 기준으로 label_horizon 뒤 타깃을 사용
        targets.append(target_array[end - 1])
        indexes.append(df.index[start + lookback - 1])
    return np.stack(sequences), np.array(targets), pd.DatetimeIndex(indexes)


def normalize_sequences(
    sequences: np.ndarray, train_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flattened = sequences[:train_count].reshape(-1, sequences.shape[-1])
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0, ddof=0)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (sequences - mean) / std
    return normalized, mean, std


class PressureLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        # num_layers=1 이면 LSTM의 dropout 인자는 실제로 적용되지 않으므로 0으로 둔다.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        # LSTM 출력 이후에 별도 Dropout을 한 번 더 적용하여 과적합을 줄인다.
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        h_last = hidden[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits.squeeze(-1)


@dataclass
class ModelResult:
    data: pd.DataFrame
    loss: float
    accuracy: float
    # 실시간/추가 추론을 위해 학습된 모델과 정규화 통계를 선택적으로 함께 보관한다.
    model: "PressureLSTM | None" = None
    feature_mean: "np.ndarray | None" = None
    feature_std: "np.ndarray | None" = None


def train_lstm(
    sequences: np.ndarray,
    targets: np.ndarray,
    indexes: pd.DatetimeIndex,
    df: pd.DataFrame,
    window: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    grad_clip: float,
    model: "PressureLSTM | None" = None,
) -> ModelResult:
    logger = logging.getLogger(__name__)
    total = len(sequences)
    if total < 5:
        raise ValueError("Need at least 5 sequences to train the LSTM.")

    train_count = int(total * 0.7)
    if train_count < 2:
        train_count = max(1, total - 1)
    val_count = total - train_count
    if val_count < 1:
        val_count = 1
        train_count = total - 1

    normalized, mean, std = normalize_sequences(sequences, train_count)
    train_dataset = SequenceDataset(normalized[:train_count], targets[:train_count])
    val_dataset = SequenceDataset(normalized[train_count:], targets[train_count:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_total = len(train_dataset)
    val_total = len(val_dataset)

    # 기존 학습된 모델이 주어지면 그 가중치에서 이어서 학습하고,
    # 아니면 새 모델을 초기화한다.
    if model is None:
        model = PressureLSTM(input_size=normalized.shape[-1]).to(device)
    else:
        model = model.to(device)
    # L2 정규화를 위해 weight_decay를 사용해 과적합을 완화한다.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    val_accuracy = 0.0
    val_loss = float("inf")

    # 간단한 early stopping: 검증 손실이 개선되지 않으면 학습을 조기 종료한다.
    best_val_loss = float("inf")
    best_state: dict | None = None
    patience = 3
    epochs_without_improvement = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = len(train_loader)
        batch_log_every = max(1, n_batches // 10)
        print(
            f"[lstm] Epoch {epoch + 1}/{epochs} start — train={train_total}, val={val_total}, batches={n_batches}"
        )
        logger.info("Starting epoch %d/%d — train samples=%d, val samples=%d, batches=%d", epoch + 1, epochs, train_total, val_total, n_batches)
        for i, (seq_batch, target_batch) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(seq_batch)
            loss = criterion(logits, target_batch)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * seq_batch.size(0)
            if i % batch_log_every == 0 or i == n_batches:
                logger.info("Epoch %d batch %d/%d — batch_loss=%.4f", epoch + 1, i, n_batches, loss.item())
        epoch_loss /= train_total

        model.eval()
        correct = 0
        total_val = 0
        eval_loss = 0.0
        with torch.no_grad():
            for seq_batch, target_batch in val_loader:
                seq_batch = seq_batch.to(device)
                target_batch = target_batch.to(device)
                logits = model(seq_batch)
                loss = criterion(logits, target_batch)
                eval_loss += loss.item() * seq_batch.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += (preds == target_batch).sum().item()
                total_val += target_batch.size(0)
        eval_loss /= val_total
        val_accuracy = correct / total_val
        val_loss = eval_loss
        logger.info(
            "Epoch %d/%d completed: train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            epochs,
            epoch_loss,
            val_loss,
            val_accuracy,
        )
        print(
            f"[lstm] Epoch {epoch + 1}/{epochs} done — train_loss={epoch_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        # best 모델 갱신 및 early stopping 상태 업데이트
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # state_dict를 깊은 복사하여 이후 학습으로부터 분리
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping triggered at epoch %d: no val_loss improvement for %d epochs.",
                    epoch + 1,
                    patience,
                )
                break

    # 가장 좋은 검증 손실을 기록한 모델 파라미터를 복원
    if best_state is not None:
        model.load_state_dict(best_state)
        val_loss = best_val_loss

    model.eval()
    with torch.no_grad():
        # 메모리 폭주를 막기 위해 전체 시퀀스를 한 번에 GPU에 올리지 않고 배치로 추론한다.
        infer_bs = 2048
        logits_chunks: list[torch.Tensor] = []
        total = normalized.shape[0]
        for start in range(0, total, infer_bs):
            end = min(start + infer_bs, total)
            batch_np = normalized[start:end].astype(np.float32, copy=False)
            batch = torch.from_numpy(batch_np).to(device)
            logits_chunks.append(model(batch).detach().cpu())
        all_logits = torch.cat(logits_chunks, dim=0)
        all_probs = torch.sigmoid(all_logits).numpy()
        preds = (all_probs >= 0.5).astype(int)

    enriched = df.reindex(indexes).copy()
    enriched["prob_up"] = all_probs
    enriched["pred_direction"] = preds
    rolling = df["imbalance"].rolling(window, min_periods=1).mean()
    enriched["imbalance_ma"] = rolling.reindex(indexes).to_numpy()
    enriched["signal"] = np.sign(enriched["imbalance_ma"])
    return ModelResult(
        data=enriched,
        loss=val_loss,
        accuracy=val_accuracy,
        model=model,
        feature_mean=mean,
        feature_std=std,
    )


def predict_next_from_df(
    df: pd.DataFrame,
    result: ModelResult,
    lookback: int,
    device: torch.device | None = None,
) -> float:
    """학습된 LSTM과 정규화 통계를 이용해
    df의 마지막 구간(lookback 길이)에 대해 다음 1초 상승 확률을 예측합니다.

    df        : buy/sell pressure가 계산된 전체 시계열 (compute_pressure 결과)
    result    : train_lstm의 반환값 (model/mean/std를 포함)
    lookback  : 학습 때 사용한 시퀀스 길이
    device    : 추론에 사용할 torch.device (None이면 model의 device 사용)
    """
    if result.model is None or result.feature_mean is None or result.feature_std is None:
        raise ValueError("ModelResult에 model / feature_mean / feature_std 정보가 없습니다.")

    model = result.model
    assert model is not None  # 위에서 체크했으므로 type checker용
    model_device = next(model.parameters()).device
    if device is None:
        device = model_device

    feature_names: Sequence[str] = ["buy_ratio", "volume", "range", "price_change", "imbalance"]
    if any(name not in df.columns for name in feature_names):
        missing = [name for name in feature_names if name not in df.columns]
        raise ValueError(f"입력 DataFrame에 필요한 컬럼이 없습니다: {missing}")

    if len(df) < lookback:
        raise ValueError(f"df 길이({len(df)})가 lookback({lookback})보다 짧습니다.")

    recent = df[feature_names].tail(lookback).to_numpy().astype(np.float32)
    # 학습 시점의 정규화 통계 사용
    mean = result.feature_mean.astype(np.float32)
    std = result.feature_std.astype(np.float32)
    std_safe = np.where(std < 1e-6, 1.0, std)
    normalized = (recent - mean) / std_safe

    seq_tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)  # (1, L, F)
    model.eval()
    with torch.no_grad():
        logits = model(seq_tensor)
        prob_up = torch.sigmoid(logits).item()
    return float(prob_up)


def save_model_checkpoint(result: ModelResult, lookback: int, path: str | Path) -> None:
    """학습된 LSTM과 정규화 통계를 디스크에 저장합니다.

    - result  : train_lstm의 반환값 (model/mean/std 포함)
    - lookback: 시퀀스 길이 (build_sequences에서 사용한 값)
    - path    : 저장할 파일 경로 (.pt, .pth 등)
    """
    if result.model is None or result.feature_mean is None or result.feature_std is None:
        raise ValueError("ModelResult에 model / feature_mean / feature_std 정보가 없습니다.")

    model = result.model
    assert model is not None

    # 어떤 시점까지 학습했는지 추적하기 위한 메타데이터
    trained_until: str | None
    num_samples: int
    if result.data.empty:
        trained_until = None
        num_samples = 0
    else:
        trained_until = str(result.data.index[-1])
        num_samples = len(result.data)

    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_mean": result.feature_mean,
        "feature_std": result.feature_std,
        "lookback": int(lookback),
        "input_size": model.lstm.input_size,
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "trained_until": trained_until,
        "num_samples": num_samples,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[PressureLSTM, np.ndarray, np.ndarray, int]:
    """저장된 체크포인트에서 LSTM 모델과 정규화 통계를 복원합니다.

    반환값: (model, feature_mean, feature_std, lookback)
    """
    path = Path(path)
    if device is None:
        map_location: torch.device | str = "cpu"
    else:
        map_location = device

    # PyTorch 2.6부터 torch.load의 기본 weights_only=True로 인해
    # numpy 객체가 섞인 사용자 dict를 불러올 때 오류가 날 수 있어
    # 여기서는 신뢰된 로컬 체크포인트라는 전제 하에 weights_only=False를 명시합니다.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    input_size = int(checkpoint["input_size"])
    hidden_size = int(checkpoint["hidden_size"])
    num_layers = int(checkpoint["num_layers"])
    lookback = int(checkpoint["lookback"])

    model = PressureLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    model.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()

    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)

    return model, feature_mean, feature_std, lookback


def load_checkpoint_metadata(path: str | Path) -> dict:
    """체크포인트 파일에서 학습 메타데이터를 읽어옵니다.

    예: 마지막으로 학습한 시점(trained_until), 사용된 샘플 개수 등.
    """
    path = Path(path)
    # 메타데이터 역시 우리가 저장한 로컬 체크포인트이므로 weights_only=False로 로드
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    meta_keys = ["trained_until", "num_samples", "lookback", "input_size", "hidden_size", "num_layers"]
    return {k: checkpoint.get(k) for k in meta_keys}
