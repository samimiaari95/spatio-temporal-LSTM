import os
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# number of cells
X = 21613
Y = 2
NB_CELLS = X*Y

member = int(os.environ.get("ENS_MEMBER", 0))

# study region
REGION = f"EU_74_-48_69_20"

# directory and inputs
OUTPUTPATH = os.path.join(get_root_dir(), "outputs", REGION)
INPUTPATH = os.path.join(get_root_dir(), "inputs", REGION)
FEATURES_FILES = ["tp_EU.npy", "vpd_EU.npy", "swvl3_EU.npy", "slopex.npy", "slopey.npy", "soilind.npy", "lon2D_ts.npy", "lat2D_ts.npy"]
TARGETVAR_FILE = "wtd.npy"

# time period
TRAINING_PERIOD = 365*15
LOOKBACK = 365
TEST_PERIOD = 365*3+LOOKBACK

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

# model name
MODEL_NAME = f"{NUM_EPOCHS}_{HIDDEN_SIZE}dr{str(DROPOUT).replace('0.','')}x{NUM_LAYERS}lr{str(LR_GAMMA).replace('.','')}x{LR_STEP_SIZE}_{LOOKBACK}x{BATCH_SIZE}_prvpdsmxyind" if LR_SCHEDULER else f"{NUM_EPOCHS}_{HIDDEN_SIZE}dr{str(DROPOUT).replace('0.','')}x{NUM_LAYERS}_{LOOKBACK}x{BATCH_SIZE}_prvpdsmxyind"
logger.warning(f"Check model name: {MODEL_NAME}")


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