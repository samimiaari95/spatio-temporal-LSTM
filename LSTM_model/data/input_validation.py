from __future__ import annotations

from pathlib import Path

import numpy as np

from LSTM_model.model.config import (
    FEATURES_FILES,
    INPUTPATH,
    NB_CELLS,
    TEST_PERIOD,
    TRAINING_PERIOD,
    TARGETVAR_FILE,
)


def validate_preprocessed_inputs() -> dict[str, tuple[int, ...]]:
    """Validate the prepared NumPy dataset before training or prediction."""
    required_steps = TRAINING_PERIOD + TEST_PERIOD
    expected_files = [*FEATURES_FILES, TARGETVAR_FILE]
    shapes: dict[str, tuple[int, ...]] = {}

    for filename in expected_files:
        path = Path(INPUTPATH) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Required preprocessed input is missing: {path}. Supply the "
                "prepared arrays or set SPATIOTEMPORAL_INPUT_DIR."
            )
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        if data.ndim != 2 or data.shape[1] != NB_CELLS:
            raise ValueError(
                f"{path} must have shape (timesteps, {NB_CELLS}); got {data.shape}."
            )
        if data.shape[0] < required_steps:
            raise ValueError(
                f"{path} has {data.shape[0]} timesteps; {required_steps} "
                "are required by the configured train/test split."
            )
        shapes[filename] = tuple(data.shape)

    return shapes
