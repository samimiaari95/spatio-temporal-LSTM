import torch
import torch.nn as nn

#plt.rcParams.update({'font.size': 22})

# model name
MODEL_NAME = "wtd_6160_70_32_prvpdTxTnsmxyindlonlat"
#MODEL_NAME = "wtd_100_128_prvpdsmxyindlonlat"

# number of cells
X = 70
Y = 88
NB_CELLS = X*Y

# study region
TARGET_REGION = "EU_px_training"#"EU_px_training"
SOURCE_REGION = "EU_px_training"

# directory and inputs
OUTPUTPATH = f"/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/outputs/{TARGET_REGION}"
INPUTPATH = f"/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/inputs/{SOURCE_REGION}"
FEATURES_FILES = ["TOT_PREC.npy", "vpd.npy", "TMAX_2M.npy", "TMIN_2M.npy", "soilmoisture.npy", "slopex.npy", "slopey.npy", "soilind.npy", "lon2D_ts.npy", "lat2D_ts.npy"]
TARGETVAR_FILE = "wtd.npy"

# time period
TRAINING_PERIOD = 365*7
LOOKBACK = 365
TEST_PERIOD = 365*2+LOOKBACK

# lstm setup
INPUT_SIZE = len(FEATURES_FILES) # precip - Tmax - Tmin - soil moisture
HIDDEN_SIZE = 32
NUM_LAYERS = 1
OUTPUT_SIZE = 1
NUM_EPOCHS = 70
LEARNING_RATE = 0.001
BATCH_SIZE = TRAINING_PERIOD-LOOKBACK

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