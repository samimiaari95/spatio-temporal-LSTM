import torch
import torch.nn as nn

print("check model name")

# number of cells
X = 10
Y = 10
NB_CELLS = X*Y

# study region
TARGET_REGION = "rand100EU_yravg1mstd_ohe"
SOURCE_REGION = "rand100EU_yravg1mstd_ohe"

# directory and inputs
OUTPUTPATH = f"/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/outputs/20yrs_ts/{TARGET_REGION}"
INPUTPATH = f"/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/inputs/20yrs_ts/{SOURCE_REGION}"
FEATURES_FILES = ["TOT_PREC.npy", "vpd.npy", "soilmoisture.npy", "slopex.npy", "slopey.npy", 
                  "soilind_1.npy", "soilind_2.npy", "soilind_3.npy", "soilind_4.npy", 
                  "soilind_5.npy", "soilind_6.npy"]
#FEATURES_FILES = ["TOT_PREC.npy", "vpd.npy", "TMAX_2M.npy", "TMIN_2M.npy", "soilmoisture.npy", "slopex.npy", "slopey.npy", "soilind.npy", "lon2D_ts.npy", "lat2D_ts.npy"]
TARGETVAR_FILE = "wtd.npy"

# time period
TRAINING_PERIOD = 365*15
LOOKBACK = 365
TEST_PERIOD = 365*4+LOOKBACK

# lstm setup
INPUT_SIZE = len(FEATURES_FILES)
HIDDEN_SIZE = 160
NUM_LAYERS = 1
OUTPUT_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
LR_SCHEDULER = False
LR_STEP_SIZE = 50
LR_GAMMA = 0.1
BATCH_SIZE = TRAINING_PERIOD-LOOKBACK

# model name
MODEL_NAME = f"{X*Y}_{HIDDEN_SIZE}lr{LR_GAMMA.replace('.','')}x{LR_STEP_SIZE}_prvpdsmxyindohe" if LR_SCHEDULER else f"{X*Y}_{HIDDEN_SIZE}_prvpdsmxyindohe"


class AwesomeLSTM(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS):
        super(AwesomeLSTM, self).__init__()
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        #h0 = torch.zeros(1, x.size(0), self.HIDDEN_SIZE).to(x.device)
        #c0 = torch.zeros(1, x.size(0), self.HIDDEN_SIZE).to(x.device)

        h_out, _ = self.lstm(x)#, (h0, c0))

        out = self.fc(h_out[:, -1, :])
        return out