import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from LSTM_setup import *


def prepare_inputfeatures(start, end, means_stds):
    all_inputs = np.array([])
    for inputvar in FEATURES_FILES:
        print(inputvar)
        raw_data = np.load(os.path.join(INPUTPATH, inputvar))
        raw_data = raw_data[:,:,:]
        data = raw_data.reshape(raw_data.shape[0], NB_CELLS)
        data = data[start:end, :]
        data = np.moveaxis(data, 0, -1) # (cells, timeseries)
        
        if f"{inputvar.replace('.npy','')}mean" not in means_stds.keys():
            means_stds[f"{inputvar.replace('.npy','')}mean"] = np.mean(data)
            means_stds[f"{inputvar.replace('.npy','')}std"] = np.std(data)

        data = (data - means_stds[f"{inputvar.replace('.npy','')}mean"])/means_stds[f"{inputvar.replace('.npy','')}std"]
        
        lookback_arrays = [data[:, i-LOOKBACK:i] for i in range(LOOKBACK, end-start)]
        lookback_arrays = np.array(lookback_arrays)

        f1 = lookback_arrays.reshape(-1, LOOKBACK)
        f1 = np.array([f1])
        all_inputs = np.concatenate((all_inputs,f1), axis=0) if len(all_inputs)>0 else f1

    all_inputs = np.moveaxis(all_inputs, 0, -1)
    
    #print(np.unique(np.equal(raw_data[LOOKBACK-1], lookback_arrays[0,:,-1].reshape(X,Y))))
    #print(np.unique(np.equal(raw_data[LOOKBACK-1], f1[:,-1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))
    #print(np.unique(np.equal(raw_data[LOOKBACK-1], all_inputs[:,-1, -1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))

    return all_inputs, means_stds

def prepare_targetvar(start, end, means_stds):
    raw_data = np.load(os.path.join(INPUTPATH, TARGETVAR_FILE))
    raw_data = raw_data[:,:,:]
    raw_data = np.nan_to_num(raw_data)
    raw_data[raw_data < 0.0] = 0

    data = raw_data[start:end, :, :]
    
    if f"{TARGETVAR_FILE.replace('.npy','')}mean" not in means_stds.keys():
        means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"] = np.mean(data)
        means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] = np.std(data)

    data = (data - means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"])/means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"]
    raw_data = (raw_data - means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"])/means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"]

    data = data.flatten()

    #print(np.unique(np.equal(raw_data[start, :, :], data.reshape(end-start, X, Y)[0,:,:])))
    return data, means_stds

    


# define mean and std dictionary
means_stds = {}
# prepare input data, standardization, lookback and train time series
train_inputs, means_stds = prepare_inputfeatures(0, TRAINING_PERIOD, means_stds)

# prepare input data of target variable and standardize
obs_stand_train, means_stds = prepare_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds)

print("creating dataloader")
print(f"train features shape: {train_inputs.shape}") # (30*30*timeseries, lookback, features)
print(f"train target shape: {obs_stand_train.shape}")

dataset = TensorDataset(torch.tensor(train_inputs).float(), torch.tensor(obs_stand_train).float())
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# initialization
lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)

# training
print("training")
lstm_model.train()
epoch_loss = {epoch:[] for epoch in range(NUM_EPOCHS)}
for epoch in range(NUM_EPOCHS):
    for inputs, y in dataloader:
        optimizer.zero_grad()
        y_hat = lstm_model(inputs)
        loss = criterion(y_hat.flatten(), y)
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, loss={loss.item()}")
        epoch_loss[epoch].append(loss.item())

# save the trained model
torch.save(lstm_model, os.path.join(OUTPUTPATH, f'{TARGET_REGION}_wtd.pt'))
print("model saved")
# plot epochs vs loss
epoch_vs_loss_plot = {k:np.mean(np.array(v)) for k, v in epoch_loss.items()}
epoch_plot = plt
epoch_plot.plot(list(epoch_vs_loss_plot.keys()), list(epoch_vs_loss_plot.values()))
epoch_plot.xlabel("Epochs")
epoch_plot.ylabel("MSE")
epoch_plot.savefig(os.path.join(OUTPUTPATH, "epochs_vs_loss.png"))
print("epoch loss plotted")
