import os
from pathlib import Path

import torch
import torch.nn as nn


def get_root_dir():
    return str(Path(__file__).resolve().parents[2])


ROOT_DIR = get_root_dir()
DATASET_NAME = os.environ.get("SPATIOTEMPORAL_DATASET", "ensemble_mean")
TARGET_REGION = os.environ.get("SPATIOTEMPORAL_TARGET_REGION", DATASET_NAME)
SOURCE_REGION = os.environ.get("SPATIOTEMPORAL_SOURCE_REGION", TARGET_REGION)
INPUTPATH = os.environ.get(
    "SPATIOTEMPORAL_INPUT_DIR",
    os.path.join(ROOT_DIR, "inputs", "20yrs_ts", "ensemble_400px", TARGET_REGION),
)
FULL_DATA_PATH = os.environ.get(
    "SPATIOTEMPORAL_FULL_DATA_DIR",
    os.path.dirname(os.path.dirname(INPUTPATH)),
)
GRIDPATH = os.environ.get("SPATIOTEMPORAL_GRID_DIR", FULL_DATA_PATH)
ENSEMBLE_100PX_INPUT_DIR = os.environ.get(
    "SPATIOTEMPORAL_100PX_INPUT_DIR",
    os.path.join(ROOT_DIR, "inputs", "20yrs_ts", "ensemble_100px"),
)
ENSEMBLE_400PX_TRAINING_DIR = os.environ.get(
    "SPATIOTEMPORAL_400PX_TRAINING_DIR",
    os.path.join(ROOT_DIR, "inputs", "20yrs_ts", "ensemble_400px"),
)
OUTPUT_BASE = os.environ.get(
    "SPATIOTEMPORAL_OUTPUT_DIR",
    os.path.join(ROOT_DIR, "outputs", "20yrs_ts", "ensemble_400px"),
)

RAW_DATA_SOURCE = os.environ.get(
    "SPATIOTEMPORAL_RAW_DATA_SOURCE",
    os.path.join(ROOT_DIR, "inputs", "raw_archive"),
)
RAW_STAGING_PATH = os.path.join(ROOT_DIR, "inputs", "raw")

# number of sampled cells in the supplied training subset
X = 10
Y = 10
NB_CELLS = X*Y

member = int(os.environ.get("ENS_MEMBER", 0))

FEATURES_FILES = [
    "TOT_PREC.npy",
    "vpd.npy",
    "soilmoisture.npy",
    "slopex.npy",
    "slopey.npy",
    "soilind.npy",
    "lon2D_ts.npy",
    "lat2D_ts.npy",
]
TARGETVAR_FILE = "wtd.npy"

# time period
TRAINING_PERIOD = 365*15
LOOKBACK = 365
TEST_PERIOD = 365*4+LOOKBACK

# lstm setup
INPUT_SIZE = len(FEATURES_FILES)
HIDDEN_SIZE = 256
NUM_LAYERS = 1
OUTPUT_SIZE = 1
DROPOUT = 0.0
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
LR_SCHEDULER = True
LR_STEP_SIZE = 50
LR_GAMMA = 0.1
BATCH_SIZE = 1000

MODEL_NAME = os.environ.get(
    "SPATIOTEMPORAL_MODEL_NAME",
    (
        f"{NUM_EPOCHS}_{HIDDEN_SIZE}dr{str(DROPOUT).replace('0.', '')}x{NUM_LAYERS}"
        f"lr{str(LR_GAMMA).replace('.', '')}x{LR_STEP_SIZE}_"
        f"{LOOKBACK}x{BATCH_SIZE}_prvpdsmxyind"
        if LR_SCHEDULER
        else f"{NUM_EPOCHS}_{HIDDEN_SIZE}dr{str(DROPOUT).replace('0.', '')}"
        f"x{NUM_LAYERS}_{LOOKBACK}x{BATCH_SIZE}_prvpdsmxyind"
    ),
)
OUTPUTPATH = os.environ.get(
    "SPATIOTEMPORAL_MODEL_OUTPUT_DIR",
    os.path.join(OUTPUT_BASE, TARGET_REGION),
)

class AwesomeLSTM(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT):
        super(AwesomeLSTM, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.NUM_LAYERS = NUM_LAYERS

        # LSTM layer
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        
        # Dropout layer
        self.DROPOUT = nn.Dropout(DROPOUT)
        
        # Fully connected layer
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.NUM_LAYERS, x.size(0), self.HIDDEN_SIZE).to(x.device)
        c0 = torch.zeros(self.NUM_LAYERS, x.size(0), self.HIDDEN_SIZE).to(x.device)

        # Forward propagate the LSTM
        h_out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Manually apply dropout to the LSTM output
        h_out = self.DROPOUT(h_out)

        # Pass through the fully connected layer (take the output of the last time step)
        h_out = self.fc(h_out[:, -1, :])  # out: tensor of shape (batch_size, output_size)
        return h_out