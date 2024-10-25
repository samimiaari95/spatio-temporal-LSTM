import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from LSTM_model.model.config import *
from LSTM_model.utils.utils import utilities

class train_LSTM_model:

    def train(self):
        utils = utilities()
        # check if GPU available in hardware
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            print("No GPU available. Training will run on CPU.")

        # make sure to have cuda toolkit or PyTorch with GPU support installed before this step
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # check output directory
        if not os.path.exists(os.path.join(OUTPUTPATH)):
            utils.make_dir(os.path.join(OUTPUTPATH))

        # define mean and std dictionary
        means_stds = {}
        # prepare input data, standardization, lookback and train time series
        train_inputs, means_stds = utils.singleregion_inputfeatures(0, TRAINING_PERIOD, means_stds)

        # prepare input data of target variable and standardize
        obs_stand_train, means_stds = utils.singleregion_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds)

        # save training data mean and std
        with open(os.path.join(OUTPUTPATH, f"meanstd_{TARGET_REGION}_{MODEL_NAME}.pkl"), 'wb') as f:
            pickle.dump(means_stds, f)
        f.close()

        print("creating dataloader")
        print(f"train features shape: {train_inputs.shape}") # (30*30*timeseries, lookback, features)
        print(f"train target shape: {obs_stand_train.shape}")

        dataset = TensorDataset(torch.tensor(train_inputs).float().to(device), torch.tensor(obs_stand_train).float().to(device))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # initialization
        lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
        if LR_SCHEDULER: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

        print("training with parallelized GPUs")
        lstm_model = nn.DataParallel(lstm_model) # Wrap the model with DataParallel
        lstm_model = lstm_model.to(device) # Move the model to the GPU

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
            if LR_SCHEDULER: scheduler.step()

            # save the trained model
            torch.save(lstm_model.module.state_dict(), os.path.join(OUTPUTPATH, f'{TARGET_REGION}_{MODEL_NAME}.pt'))
            print("model saved")
            # plot epochs vs loss
            epoch_vs_loss_plot = {k:np.mean(np.array(v)) for k, v in epoch_loss.items()}
            epoch_plot = plt#.figure().clear()
            epoch_plot.plot(list(epoch_vs_loss_plot.keys()), list(epoch_vs_loss_plot.values()))
            epoch_plot.xlabel("Epochs")
            epoch_plot.ylabel("MSE")
            epoch_plot.savefig(os.path.join(OUTPUTPATH, f'{TARGET_REGION}_{MODEL_NAME}.png'))
            print("epoch loss plotted")
