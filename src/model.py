from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import logging

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
    df: pd.DataFrame, lookback: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    features: Sequence[str] = ["buy_ratio", "volume", "range", "price_change", "imbalance"]
    feature_array = df[features].to_numpy()
    target_series = (df["close"].shift(-1) > df["close"]).astype(int)
    feature_array = feature_array[:-1]
    target_array = target_series.to_numpy()[:-1]
    if feature_array.shape[0] < lookback + 1:
        raise ValueError("Not enough samples to construct the requested lookback sequences.")
    sequences: list[np.ndarray] = []
    targets: list[int] = []
    indexes: list[pd.Timestamp] = []
    for start in range(feature_array.shape[0] - lookback + 1):
        end = start + lookback
        sequences.append(feature_array[start:end])
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
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        logits = self.fc(hidden[-1])
        return logits.squeeze(-1)


@dataclass
class ModelResult:
    data: pd.DataFrame
    loss: float
    accuracy: float


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

    normalized, _, _ = normalize_sequences(sequences, train_count)
    train_dataset = SequenceDataset(normalized[:train_count], targets[:train_count])
    val_dataset = SequenceDataset(normalized[train_count:], targets[train_count:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_total = len(train_dataset)
    val_total = len(val_dataset)

    model = PressureLSTM(input_size=normalized.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    val_accuracy = 0.0
    val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = len(train_loader)
        batch_log_every = max(1, n_batches // 10)
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
        logger.info("Epoch %d/%d completed: train_loss=%.4f val_loss=%.4f val_acc=%.4f", epoch + 1, epochs, epoch_loss, val_loss, val_accuracy)

    model.eval()
    with torch.no_grad():
        normalized_tensor = torch.from_numpy(normalized.astype(np.float32)).to(device)
        all_logits = model(normalized_tensor)
        # CUDA 텐서를 바로 numpy로 바꿀 수 없으므로 CPU로 옮긴 뒤 변환
        all_probs = torch.sigmoid(all_logits).cpu().numpy()
        preds = (all_probs >= 0.5).astype(int)

    enriched = df.reindex(indexes).copy()
    enriched["prob_up"] = all_probs
    enriched["pred_direction"] = preds
    rolling = df["imbalance"].rolling(window, min_periods=1).mean()
    enriched["imbalance_ma"] = rolling.reindex(indexes).to_numpy()
    enriched["signal"] = np.sign(enriched["imbalance_ma"])
    return ModelResult(data=enriched, loss=val_loss, accuracy=val_accuracy)
