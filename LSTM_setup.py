import torch
import torch.nn as nn

#plt.rcParams.update({'font.size': 22})

# number of cells
X = 5
Y = 5
NB_CELLS = X*Y

# study region
TARGET_REGION = "DOURO"
SOURCE_REGION = "DOURO"

# directory and inputs
OUTPUTPATH = f"/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/outputs/{TARGET_REGION}"
INPUTPATH = f"/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/inputs/{TARGET_REGION}"
FEATURES_FILES = ["TOT_PREC.npy", "vpd.npy", "soilmoisture.npy", "slopex.npy", "slopey.npy", "soilind.npy"]
TARGETVAR_FILE = "wtd.npy"

# time period
TRAINING_PERIOD = 365*7
LOOKBACK = 365
TEST_PERIOD = 365*2+LOOKBACK

# lstm setup
INPUT_SIZE = len(FEATURES_FILES) # precip - Tmax - Tmin - soil moisture
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = TRAINING_PERIOD-LOOKBACK

class AwesomeLSTM(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE):
        super(AwesomeLSTM, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.HIDDEN_SIZE).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.HIDDEN_SIZE).to(x.device)

        h_out, _ = self.lstm(x, (h0, c0))

        out = self.fc(h_out[:, -1, :])
        return out