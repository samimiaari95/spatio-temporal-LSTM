import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import postprocess_LSTM_features, postprocess_LSTM_targetvar

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
dirpath = os.path.dirname(os.path.realpath(__file__))
DIRPATH = f"/p/project1/cslts/miaari1/python_scripts/DailyScriptBox/outputs/LSTM_inputs/{REGION}"
FEATURES_FILES = ["TOT_PREC.npy", "TMAX_2M.npy", "TMIN_2M.npy", "soilmoisture.npy"]
TARGETVAR_FILE = "wtd.npy"

# training
TRAINING_PERIOD = 365*3
LOOKBACK = 365
BATCH_SIZE = NB_CELLS
TEST_PERIOD = 365+LOOKBACK

# lstm
INPUT_SIZE = len(FEATURES_FILES) # precip - Tmax - Tmin - soil moisture
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
NUM_EPOCHS = 1
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

    # NOTE save the trained model
    torch.save(lstm_model, os.path.join(DIRPATH, f'{REGION}_wtd.pt'))
    print("model saved")

    # plot epochs vs loss
    epoch_vs_loss_plot = {k:np.mean(v) for k, v in epoch_loss.items()}
    plt.plot(list(epoch_vs_loss_plot.keys()), list(epoch_vs_loss_plot.values()))
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.savefig(os.path.join(DIRPATH, "epochs_vs_loss.png"))
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

    cell_mse = [criterion(torch.tensor(test_sim[i,:]).float(), torch.tensor(test_obs[i,:]).float()).item() for i in range(NB_CELLS)]
    
    cell_mse = np.array(cell_mse)
    cell_mse = cell_mse.reshape(X,Y)

    ax = plt
    ax.figure(figsize=(16,9))
    #ax.imshow(cell_mse, cmap='hot', interpolation='nearest')
    ax.imshow(cell_mse, interpolation='nearest')
    ax.colorbar().ax.set_ylabel('MSE')
    ax.title(f"avg MSE (mm): {test_loss:.4f}")
    ax.ylabel("y (pixels)")
    ax.xlabel("x (pixels)")
    ax.savefig(os.path.join(DIRPATH, f"{REGION}_2Dmse.png"))

    # average wtd/cell over its time series
    obs_destand_test_avg = [np.mean(obs_destand_test[i,:]) for i in range(obs_destand_test.shape[0])]
    obs_destand_test_avg = np.array(obs_destand_test_avg)
    obs_destand_test_avg = obs_destand_test_avg.reshape(X,Y)

    sim_destand_test_avg = [np.mean(sim_destand_test[i,:]) for i in range(sim_destand_test.shape[0])]
    sim_destand_test_avg = np.array(sim_destand_test_avg)
    sim_destand_test_avg = sim_destand_test_avg.reshape(X,Y)

    diff_destand_test_avg = obs_destand_test_avg - sim_destand_test_avg

    avg_obs = plt
    avg_obs.figure(figsize=(16,9))
    avg_obs.imshow(obs_destand_test_avg, interpolation='nearest')
    avg_obs.colorbar()#.avg_obs.set_ylabel('wtd (m)')
    avg_obs.title(f"Observed wtd (mm)")
    avg_obs.ylabel("y (pixels)")
    avg_obs.xlabel("x (pixels)")
    avg_obs.savefig(os.path.join(DIRPATH, f"{REGION}_avgobs_wtd.png"))
    
    avg_pred = plt
    avg_pred.figure(figsize=(16,9))
    avg_pred.imshow(sim_destand_test_avg, interpolation='nearest')
    avg_pred.colorbar()#.avg_pred.set_ylabel('wtd (m)')
    avg_pred.title(f"Predicted wtd (mm)")
    avg_pred.ylabel("y (pixels)")
    avg_pred.xlabel("x (pixels)")
    avg_pred.savefig(os.path.join(DIRPATH, f"{REGION}_avgpred_wtd.png"))

    avg_diff = plt
    avg_diff.figure(figsize=(16,9))
    avg_diff.imshow(diff_destand_test_avg, interpolation='nearest')
    avg_diff.colorbar()#.avg_diff.set_ylabel('wtd (m)')
    avg_diff.title(f"Difference (obs-pred) wtd (mm)")
    avg_diff.ylabel("y (pixels)")
    avg_diff.xlabel("x (pixels)")
    avg_diff.savefig(os.path.join(DIRPATH, f"{REGION}_avgdiff_wtd.png"))

def transfer_LSTM():
    # load dataset
    SRC_REGION = "ME"
    TRGT_REGION = "FR"
    DIRPATH = "/p/project1/cslts/miaari1/python_scripts/DailyScriptBox"
    src_data = pd.read_csv(os.path.join(DIRPATH, "outputs", f"{SRC_REGION}_precip_Tmax_Tmin_subSurfStor_evap_wtd.csv"))
    trgt_data = pd.read_csv(os.path.join(DIRPATH, "outputs", f"{TRGT_REGION}_precip_Tmax_Tmin_subSurfStor_evap_wtd.csv"))

    # remove index and date column, remove titles raw
    srctrain_df = src_data.iloc[1:TRAINING_PERIOD,2:] # drop date column
    trgttrain_df = trgt_data.iloc[1:TRAINING_PERIOD,2:] # drop date column

    # calculate and store means and stds
    srcmeans_stds = {}
    for col in srctrain_df.columns:
        srcmeans_stds[col] = {
            'mean': srctrain_df[col].mean(),
            'std': srctrain_df[col].std()
        }
    trgtmeans_stds = {}
    for col in trgttrain_df.columns:
        trgtmeans_stds[col] = {
            'mean': trgttrain_df[col].mean(),
            'std': trgttrain_df[col].std()
        }

    # load the model
    src_lstm_model = torch.load(os.path.join(DIRPATH, f'{SRC_REGION}_subSurfStor.pt'))
    trgt_lstm_model = torch.load(os.path.join(DIRPATH, f'{TRGT_REGION}_subSurfStor.pt'))
    criterion = nn.MSELoss()

    ## Testing
    srctest_df = src_data.iloc[TRAINING_PERIOD:,2:]
    trgttest_df = trgt_data.iloc[TRAINING_PERIOD:,2:]

    # use standarizer from training to prevent data-leakage:
    for col in srctest_df.columns:
        srctest_df[col] = (srctest_df[col] - srcmeans_stds[col]['mean']) / srcmeans_stds[col]['std']
    for col in trgttest_df.columns:
        trgttest_df[col] = (trgttest_df[col] - trgtmeans_stds[col]['mean']) / trgtmeans_stds[col]['std']
    
    # source dataloader
    srctest_data = srctest_df.values
    srcraw_inputs = srctest_data[:, :-1]
    srctest_obs = srctest_data[LOOKBACK+1:,-1]
    x = []
    for i in range(LOOKBACK+1, len(srcraw_inputs)):
        input_seq = srcraw_inputs[i-LOOKBACK:i]
        x.append(input_seq)
    srctest_inputs = np.array(x)
    print("preparing test dataloader")
    srctest_dataset = TensorDataset(torch.tensor(srctest_inputs).float(), torch.tensor(srctest_obs).float())
    srctest_dataloader = DataLoader(srctest_dataset, batch_size=BATCH_SIZE, shuffle=False) # no random shuffling for the test
    
    # target dataloader
    trgttest_data = trgttest_df.values
    trgtraw_inputs = trgttest_data[:, :-1]
    trgttest_obs = trgttest_data[LOOKBACK+1:,-1]
    x = []
    for i in range(LOOKBACK+1, len(trgtraw_inputs)):
        input_seq = trgtraw_inputs[i-LOOKBACK:i]
        x.append(input_seq)
    trgttest_inputs = np.array(x)
    print("preparing test dataloader")
    trgttest_dataset = TensorDataset(torch.tensor(trgttest_inputs).float(), torch.tensor(trgttest_obs).float())
    trgttest_dataloader = DataLoader(trgttest_dataset, batch_size=BATCH_SIZE, shuffle=False) # no random shuffling for the test

    ## source Evaluation
    print("evaluation")
    src_lstm_model.eval()
    total_loss = 0
    srctest_s = []
    srctest_o = []
    with torch.no_grad():
        for inputs, y in trgttest_dataloader:
            y_hat = src_lstm_model(inputs)
            srctest_s.append(y_hat.flatten())
            srctest_o.append(y)
            loss = criterion(y_hat.flatten(), y)
            total_loss += loss.item()

    srctest_loss = total_loss / len(srctest_dataloader)
    print(f' source Test Loss: {srctest_loss:.4f}')

    ## target Evaluation
    print("evaluation")
    trgt_lstm_model.eval()
    total_loss = 0
    trgttest_s = []
    trgttest_o = []
    with torch.no_grad():
        for inputs, y in trgttest_dataloader:
            y_hat = trgt_lstm_model(inputs)
            trgttest_s.append(y_hat.flatten())
            trgttest_o.append(y)
            loss = criterion(y_hat.flatten(), y)
            total_loss += loss.item()

    trgttest_loss = total_loss / len(trgttest_dataloader)
    print(f'target Test Loss: {trgttest_loss:.4f}')

    #torch.save(lstm_model, os.path.join(DIRPATH, 'model_pytorch_subSurfStor.pt'))

    srctest_sim = torch.cat(srctest_s).numpy()
    srctest_sim = srctest_sim*trgtmeans_stds['subSurfStor']['std'] + trgtmeans_stds['subSurfStor']['mean']
    srctest_obs = torch.cat(srctest_o).numpy()
    srctest_obs = srctest_obs*trgtmeans_stds['subSurfStor']['std'] + trgtmeans_stds['subSurfStor']['mean']

    trgttest_sim = torch.cat(trgttest_s).numpy()
    trgttest_sim = trgttest_sim*trgtmeans_stds['subSurfStor']['std'] + trgtmeans_stds['subSurfStor']['mean']
    trgttest_obs = torch.cat(trgttest_o).numpy()
    trgttest_obs = trgttest_obs*trgtmeans_stds['subSurfStor']['std'] + trgtmeans_stds['subSurfStor']['mean']

    x_axis = src_data.iloc[TRAINING_PERIOD:,1]
    outputs = {"day": x_axis[LOOKBACK+1:], "Observed": srctest_obs, "srcPredicted": srctest_sim, "trgtPredicted": trgttest_sim}
    df = pd.DataFrame(outputs)
    df.to_csv(os.path.join(DIRPATH, "outputs", f"TL_{SRC_REGION}_{TRGT_REGION}_subSurfStor.csv"), index=False)
    df = pd.read_csv(os.path.join(DIRPATH, "outputs", f"TL_{SRC_REGION}_{TRGT_REGION}_subSurfStor.csv"))

    total_loss = 0
    for i in range(len(trgttest_s)):
        loss = criterion(trgttest_s[i], srctest_s[i])
        total_loss += loss.item()
    TL_loss = total_loss / len(trgttest_s)

    #df = pd.read_csv(os.path.join(DIRPATH, "pred_subSurfStor.csv"))
    plt.plot(df["Observed"].to_list(), c="b", label="FR Observed")
    plt.plot(df["trgtPredicted"].to_list(), c="g", label="FR-FR Predicted")
    plt.plot(df["srcPredicted"].to_list(), c="r", label="ME-FR Predicted")

    plt.ylabel("Subsurface storage (mm)")
    plt.xlabel("2-year daily timestep")
    plt.legend()
    plt.title(f"TL MSE: {TL_loss:.4f} (Benchmark prediction vs TL prediciton)")
    plt.show()

def original_backprob_streamflow():
    rr_data = pd.read_csv("./rr_example.csv")
    # settings

    # training
    TRAINING_PERIOD = 2924
    LOOKBACK = 365
    BATCH_SIZE = 200

    # lstm
    INPUT_SIZE = 4
    HIDDEN_SIZE = 64
    OUTPUT_SIZE = 1
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.01

    # remove date column
    train_df = rr_data.iloc[1:TRAINING_PERIOD,1:] # drop date column

    # calculate and store means and stds
    means_stds = {}
    for col in train_df.columns:
        means_stds[col] = {
            'mean': train_df[col].mean(),
            'std': train_df[col].std()
        }

    # standardization
    for col in train_df.columns:
        train_df[col] = (train_df[col] - means_stds[col]['mean']) / means_stds[col]['std']

    # push data to numpy (assuming the obs are in the last column)
    data = train_df.values
    raw_inputs = data[:, :-1]
    obs = data[LOOKBACK+1:,-1]

    # create input data:
    x = []
    for i in range(LOOKBACK+1, len(raw_inputs)):
        input_seq = raw_inputs[i-LOOKBACK:i]
        x.append(input_seq)
    inputs = np.array(x)

    dataset = TensorDataset(torch.tensor(inputs).float(), torch.tensor(obs).float())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # initialization
    lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    valid_losses = []

    # training
    for epoch in range(NUM_EPOCHS):
        for inputs, y in dataloader:
            optimizer.zero_grad()
            y_hat = lstm_model(inputs)
            loss = criterion(y_hat.flatten(), y)
            loss.backward()
            optimizer.step()

            # write in feedback and validation part to see what is happening

    test_df = rr_data.iloc[TRAINING_PERIOD:,1:]

    # use standarizer from training to prevent data-leakage:
    for col in test_df.columns:
        test_df[col] = (test_df[col] - means_stds[col]['mean']) / means_stds[col]['std']

    test_data = test_df.values
    raw_inputs = test_data[:, :-1]
    test_obs = test_data[LOOKBACK+1:,-1]
    x = []
    for i in range(LOOKBACK+1, len(raw_inputs)):
        input_seq = raw_inputs[i-LOOKBACK:i]
        x.append(input_seq)
    test_inputs = np.array(x)

    test_dataset = TensorDataset(torch.tensor(test_inputs).float(), torch.tensor(test_obs).float())
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # no random shuffling for the test

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

    test_sim = torch.cat(test_s).numpy()
    test_sim = test_sim*means_stds['streamflow']['std'] + means_stds['streamflow']['mean']
    test_obs = torch.cat(test_o).numpy()
    test_obs = test_obs*means_stds['streamflow']['std'] + means_stds['streamflow']['mean']

    plt.plot(test_obs, c="dodgerblue")
    plt.plot(test_sim, c="orange")
    plt.show()




train_LSTM_fromscratch()