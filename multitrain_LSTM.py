import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from LSTM_setup import *
from utils import multiregion_inputfeatures, multiregion_targetvar


# define mean and std dictionary
means_stds = {}
# prepare input data, standardization, lookback and train time series
train_inputs, means_stds = multiregion_inputfeatures(0, TRAINING_PERIOD, means_stds)

# prepare input data of target variable and standardize
obs_stand_train, means_stds = multiregion_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds)

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
torch.save(lstm_model, os.path.join(OUTPUTPATH, f'{TARGET_REGION}_wtd_5x5_128_prvpdsmslopexysoilind.pt'))
print("model saved")
# plot epochs vs loss
epoch_vs_loss_plot = {k:np.mean(np.array(v)) for k, v in epoch_loss.items()}
epoch_plot = plt
epoch_plot.plot(list(epoch_vs_loss_plot.keys()), list(epoch_vs_loss_plot.values()))
epoch_plot.xlabel("Epochs")
epoch_plot.ylabel("MSE")
epoch_plot.savefig(os.path.join(OUTPUTPATH, "epochs_vs_loss.png"))
print("epoch loss plotted")
