import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import postprocess_LSTM_features, postprocess_LSTM_targetvar, plot_mse, plot_wtdmap

plt.rcParams.update({'font.size': 22})

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
    
# number of cells
X = 34
Y = 34
NB_CELLS = X*Y

REGION = "SEINE"
DIRPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs", f"{REGION}")
FEATURES_FILES = ["TOT_PREC.npy", "TMAX_2M.npy", "TMIN_2M.npy", "soilmoisture.npy"]
TARGETVAR_FILE = "wtd.npy"

# training
TRAINING_PERIOD = 200 # days
LOOKBACK = 30
BATCH_SIZE = NB_CELLS
TEST_PERIOD = (365-200) # 365 total number of available time series in the example

# lstm
INPUT_SIZE = len(FEATURES_FILES) # precip - Tmax - Tmin - soil moisture
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
NUM_EPOCHS = 2
LEARNING_RATE = 0.01

def train_LSTM_fromscratch():

    # define mean and std dictionary
    means_stds = {}
    # prepare input data, standardization, lookback and train time series
    train_inputs, means_stds = postprocess_LSTM_features(FEATURES_FILES, DIRPATH, 0, TRAINING_PERIOD, LOOKBACK, INPUT_SIZE, means_stds,train_mode=True)

    # prepare input data of target variable and standardize
    obs_stand_train, means_stds = postprocess_LSTM_targetvar(TARGETVAR_FILE, DIRPATH, LOOKBACK, TRAINING_PERIOD, means_stds, train_mode=True)

    # data loader
    #   Above we build our batch-sampling ourselves. However, since this process
    #   is a standard procedure for training deep learning models pytorch has its
    #   own utilities for handling and sampling data.
    #   They are called a dataset and a dataloader.
    #   We will use them now to train our LSTM
    print("creating dataloader")
    print(f"train features shape: {train_inputs.shape}") # (34*34*timeseries, lookback, features)
    print(f"train target shape: {obs_stand_train.shape}")

    dataset = TensorDataset(torch.tensor(train_inputs).float(), torch.tensor(obs_stand_train).float())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # initialization
    lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

    # training
    print("training")
    epoch_loss = {epoch:[] for epoch in range(NUM_EPOCHS)}
    for epoch in range(NUM_EPOCHS):
        for inputs, y in dataloader:
            optimizer.zero_grad()
            y_hat = lstm_model(inputs)
            loss = criterion(y_hat.flatten(), y)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, loss={loss}")
            epoch_loss[epoch].append(loss.item())

    ################# Testing ######################
    obs_stand_test, means_stds = postprocess_LSTM_targetvar(TARGETVAR_FILE, DIRPATH, TRAINING_PERIOD+LOOKBACK, TRAINING_PERIOD+TEST_PERIOD, means_stds, train_mode=False)
    test_inputs, means_stds = postprocess_LSTM_features(FEATURES_FILES, DIRPATH, TRAINING_PERIOD, TRAINING_PERIOD+TEST_PERIOD, LOOKBACK, INPUT_SIZE, means_stds,train_mode=False)

    print("preparing test dataloader")
    print(f"test inputs shape: {test_inputs.shape}")
    print(f"test obs shape: {obs_stand_test.shape}")
    test_dataset = TensorDataset(torch.tensor(test_inputs).float(), torch.tensor(obs_stand_test).float())
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # no random shuffling for the test

    ## Evaluation
    print("evaluation")
    lstm_model.eval()
    total_loss = 0
    test_s = []
    test_o = []
    with torch.no_grad():
        for inputs, y in test_dataloader:
            y_hat = lstm_model(inputs)
            test_s.append(y_hat.flatten())
            test_o.append(y)
            loss = criterion(y_hat.flatten(), y)
            total_loss += loss.item()
    test_loss = total_loss / len(test_dataloader)
    print(f'Test Loss: {test_loss:.4f}')

    # save the trained model
    torch.save(lstm_model, os.path.join(DIRPATH, f'{REGION}_wtd.pt'))
    print("model saved")

    # plot epochs vs loss
    epoch_vs_loss_plot = {k:np.mean(v) for k, v in epoch_loss.items()}
    plt.plot(list(epoch_vs_loss_plot.keys()), list(epoch_vs_loss_plot.values()))
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig(os.path.join(DIRPATH, "epochs_vs_loss.png"))
    
    test_sim = torch.cat(test_s).numpy().reshape(NB_CELLS, TEST_PERIOD-LOOKBACK)
    test_obs = torch.cat(test_o).numpy().reshape(NB_CELLS, TEST_PERIOD-LOOKBACK)
    cell_mse = [criterion(torch.tensor(test_sim[i,:]).float(), torch.tensor(test_obs[i,:]).float()).item() for i in range(NB_CELLS)]
    cell_mse = np.array(cell_mse)
    cell_mse = cell_mse.reshape(X,Y)
    plot_mse(cell_mse, test_loss, os.path.join(DIRPATH, f"{REGION}_2Dmse.png"))

    return


def load_LSTM():

    # define mean and std dictionary
    means_stds = {}
    # prepare input data, standardization, lookback and train time series
    train_inputs, means_stds = postprocess_LSTM_features(FEATURES_FILES, DIRPATH, 0, TRAINING_PERIOD, LOOKBACK, INPUT_SIZE, means_stds,train_mode=True)

    # prepare input data of target variable and standardize
    #obs_stand_train, means_stds, plott = postprocess_LSTM_targetvar(TARGETVAR_FILE, DIRPATH, LOOKBACK, TRAINING_PERIOD, means_stds, train_mode=True)
    obs_stand_train, means_stds = postprocess_LSTM_targetvar(TARGETVAR_FILE, DIRPATH, LOOKBACK, TRAINING_PERIOD, means_stds, train_mode=True)
    
    # load the model
    lstm_model = torch.load(os.path.join(DIRPATH, f'{REGION}_wtd.pt'))
    criterion = nn.MSELoss()

    ################# Testing ######################
    #obs_stand_test, means_stds, plott = postprocess_LSTM_targetvar(TARGETVAR_FILE, DIRPATH, TRAINING_PERIOD+LOOKBACK, TRAINING_PERIOD+TEST_PERIOD, means_stds, train_mode=False)
    obs_stand_test, means_stds = postprocess_LSTM_targetvar(TARGETVAR_FILE, DIRPATH, TRAINING_PERIOD+LOOKBACK, TRAINING_PERIOD+TEST_PERIOD, means_stds, train_mode=False)
    test_inputs, means_stds = postprocess_LSTM_features(FEATURES_FILES, DIRPATH, TRAINING_PERIOD, TRAINING_PERIOD+TEST_PERIOD, LOOKBACK, INPUT_SIZE, means_stds,train_mode=False)
    #plt.plot(plott)
    #plt.show()
    #return
    
    print("preparing test dataloader")
    print(f"test inputs shape: {test_inputs.shape}")
    print(f"test obs shape: {obs_stand_test.shape}")
    test_dataset = TensorDataset(torch.tensor(test_inputs).float(), torch.tensor(obs_stand_test).float())
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # no random shuffling for the test


    ## Evaluation
    print("evaluation")
    lstm_model.eval()
    total_loss = 0
    test_s = []
    test_o = []
    with torch.no_grad():
        for inputs, y in test_dataloader:
            y_hat = lstm_model(inputs)
            test_s.append(y_hat.flatten())
            test_o.append(y)
            loss = criterion(y_hat.flatten(), y)
            total_loss += loss.item()

    test_loss = total_loss / len(test_dataloader)
    print(f'Test Loss: {test_loss:.4f}')

    test_sim = torch.cat(test_s).numpy().reshape(NB_CELLS, TEST_PERIOD-LOOKBACK)
    test_obs = torch.cat(test_o).numpy().reshape(NB_CELLS, TEST_PERIOD-LOOKBACK)

    # standardization for each cell time series
    # TODO probably no need to destandardize for observations, 
    # compare it with the original values to confirm the standardization process
    obs_destand_test = np.zeros(test_obs.shape)
    for i in range(test_obs.shape[0]):
        obs_destand_test[i,:] = (test_obs[i, :]*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"][i] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"][i]) / means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"][i] if means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"][i] != 0 else 0

    sim_destand_test = np.zeros(test_sim.shape)
    for i in range(test_sim.shape[0]):
        sim_destand_test[i,:] = (test_sim[i, :]*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"][i] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"][i]) / means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"][i] if means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"][i] != 0 else 0

    # average wtd/cell over its time series
    obs_destand_test_avg = [np.mean(obs_destand_test[i,:]) for i in range(obs_destand_test.shape[0])]
    obs_destand_test_avg = np.array(obs_destand_test_avg)
    obs_destand_test_avg = obs_destand_test_avg.reshape(X,Y)
    plot_wtdmap(obs_destand_test_avg, "Observed wtd (mm)", os.path.join(DIRPATH, f"{REGION}_avgobs_wtd.png"))

    sim_destand_test_avg = [np.mean(sim_destand_test[i,:]) for i in range(sim_destand_test.shape[0])]
    sim_destand_test_avg = np.array(sim_destand_test_avg)
    sim_destand_test_avg = sim_destand_test_avg.reshape(X,Y)
    plot_wtdmap(sim_destand_test_avg, "Predicted wtd (mm)", os.path.join(DIRPATH, f"{REGION}_avgpred_wtd.png"))

    diff_destand_test_avg = obs_destand_test_avg - sim_destand_test_avg
    plot_wtdmap(diff_destand_test_avg, "Difference (obs-pred) wtd (mm)", os.path.join(DIRPATH, f"{REGION}_avgdiff_wtd.png"))



train_LSTM_fromscratch()