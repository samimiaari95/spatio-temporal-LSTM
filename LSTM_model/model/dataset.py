from __future__ import annotations

import os
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from LSTM_model.model.config import (
    FEATURES_FILES,
    INPUTPATH,
    LOOKBACK,
    NB_CELLS,
    TARGETVAR_FILE,
    TRAINING_PERIOD,
)


def _load_grid(path: str) -> np.ndarray:
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    if data.ndim != 2 or data.shape[1] != NB_CELLS:
        raise ValueError(
            f"{path} must have shape (timesteps, {NB_CELLS}) or a spatial "
            f"shape that flattens to {NB_CELLS} cells; got {data.shape}."
        )
    return data


def calculate_training_statistics() -> dict[str, float]:
    statistics: dict[str, float] = {}
    for filename in FEATURES_FILES:
        data = _load_grid(os.path.join(INPUTPATH, filename))
        if data.shape[0] < TRAINING_PERIOD:
            raise ValueError(
                f"{filename} has {data.shape[0]} timesteps; "
                f"{TRAINING_PERIOD} are required for training."
            )
        values = np.asarray(data[:TRAINING_PERIOD])
        mean = float(np.mean(values))
        std = float(np.std(values))
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"{filename} contains non-finite training statistics.")
        statistics[f"{filename[:-4]}mean"] = mean
        statistics[f"{filename[:-4]}std"] = std if std > 0 else 1.0

    target = _load_grid(os.path.join(INPUTPATH, TARGETVAR_FILE))
    if target.shape[0] < TRAINING_PERIOD:
        raise ValueError(
            f"{TARGETVAR_FILE} has {target.shape[0]} timesteps; "
            f"{TRAINING_PERIOD} are required for training."
        )
    target_values = np.asarray(target[LOOKBACK:TRAINING_PERIOD]).copy()
    np.nan_to_num(target_values, copy=False)
    target_values[target_values < 0] = 0
    mean = float(np.mean(target_values))
    std = float(np.std(target_values))
    if not np.isfinite(mean) or not np.isfinite(std) or std == 0:
        raise ValueError(f"{TARGETVAR_FILE} has invalid training statistics.")
    statistics[f"{TARGETVAR_FILE[:-4]}mean"] = mean
    statistics[f"{TARGETVAR_FILE[:-4]}std"] = std
    return statistics


class GridSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Create lookback windows lazily so training does not materialize all windows."""

    def __init__(
        self,
        target_start: int,
        target_end: int,
        statistics: Mapping[str, float],
    ) -> None:
        if target_start < LOOKBACK:
            raise ValueError(f"target_start must be at least LOOKBACK ({LOOKBACK}).")
        if target_end <= target_start:
            raise ValueError("target_end must be greater than target_start.")

        self.features = [
            _load_grid(os.path.join(INPUTPATH, filename))
            for filename in FEATURES_FILES
        ]
        self.target = _load_grid(os.path.join(INPUTPATH, TARGETVAR_FILE))
        self.target_start = target_start
        self.target_end = target_end
        self.statistics = statistics

        required_timesteps = target_end
        for filename, data in zip(FEATURES_FILES, self.features):
            if data.shape[0] < required_timesteps:
                raise ValueError(
                    f"{filename} has {data.shape[0]} timesteps; "
                    f"{required_timesteps} are required."
                )
        if self.target.shape[0] < required_timesteps:
            raise ValueError(
                f"{TARGETVAR_FILE} has {self.target.shape[0]} timesteps; "
                f"{required_timesteps} are required."
            )

    def __len__(self) -> int:
        return (self.target_end - self.target_start) * NB_CELLS

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        time_index = self.target_start + index // NB_CELLS
        cell_index = index % NB_CELLS
        feature_window = np.empty((LOOKBACK, len(self.features)), dtype=np.float32)

        for feature_index, (filename, data) in enumerate(
            zip(FEATURES_FILES, self.features)
        ):
            start = time_index - LOOKBACK
            values = np.asarray(data[start:time_index, cell_index])
            mean = self.statistics[f"{filename[:-4]}mean"]
            std = self.statistics[f"{filename[:-4]}std"]
            feature_window[:, feature_index] = (values - mean) / std

        target_mean = self.statistics[f"{TARGETVAR_FILE[:-4]}mean"]
        target_std = self.statistics[f"{TARGETVAR_FILE[:-4]}std"]
        target = np.nan_to_num(self.target[time_index, cell_index])
        if target < 0:
            target = 0.0
        target = (target - target_mean) / target_std

        return (
            torch.from_numpy(feature_window),
            torch.tensor(target, dtype=torch.float32),
        )
