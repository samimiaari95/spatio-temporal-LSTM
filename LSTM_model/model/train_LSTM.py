import os
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from LSTM_model.model.config import *
from LSTM_model.utils.utils import utilities

class MyDataset(Dataset):
    def __init__(self, filename):
        self.datafile = h5py.File(os.path.join(INPUTPATH, filename),'r')
        self.device = torch.device('cuda')
    
    def __len__(self):
        return self.datafile["target_data"].shape[0]

    def __getitem__(self, idx):
        # all input features in one file of shape (datapoints, lookback, # of features)
        inputs = self.datafile["input_data"][idx]
        outputs = self.datafile["target_data"][idx]
        return torch.from_numpy(inputs).float().to(self.device), torch.from_numpy(outputs).float().to(self.device)
        
class train_LSTM_model():

    def train(self):
        def get_local_rank():
            """Return the local rank of this process."""
            return int(os.getenv('LOCAL_RANK'))

        torch.distributed.init_process_group(backend='cpu:gloo,cuda:nccl')

        utils = utilities()
        # check if GPU available in hardware
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            print("No GPU available. Training will run on CPU.")

        # make sure to have cuda toolkit or PyTorch with GPU support installed before this step
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = get_local_rank()

        device = torch.device('cuda', local_rank)

        torch.cuda.set_device(device)

        # check output directory
        if not os.path.exists(os.path.join(OUTPUTPATH)):
            utils.make_dir(os.path.join(OUTPUTPATH))

        # define mean and std dictionary
        #means_stds = {}
        with open(os.path.join(os.path.dirname(INPUTPATH), f"meanstd_ensemble_100px_100_256dr0x1lr01x50_365x1000_prvpdsmxyind.pkl"), 'rb') as f:
            means_stds = pickle.load(f)
        f.close()
        
        # prepare input data, standardization, lookback and train time series
        train_inputs, means_stds = utils.singleregion_inputfeatures(0, TRAINING_PERIOD, means_stds)

        # prepare input data of target variable and standardize
        obs_stand_train, means_stds = utils.singleregion_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds)

        # save training data mean and std
        with open(os.path.join(OUTPUTPATH, f"meanstd_{TARGET_REGION}_{MODEL_NAME}.pkl"), 'wb') as f:
            pickle.dump(means_stds, f)
        f.close()

        #print("creating dataloader")
        dataset = TensorDataset(torch.tensor(train_inputs).float().to(device), torch.tensor(obs_stand_train).float().to(device))
        #dataset = MyDataset("target1d_inputs3d_pointsxlookbackxfeatures.h5")

        #with open(os.path.join(OUTPUTPATH, f"meanstd_{SOURCE_REGION}_{MODEL_NAME}.pkl"), 'rb') as f:
        #    means_stds = pickle.load(f)
        #f.close()


        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, seed=0,)
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        # initialization
        lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
        if LR_SCHEDULER: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

        lstm_model = lstm_model.to(device) # Move the model to the GPU
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        lstm_model = torch.nn.parallel.DistributedDataParallel(
            lstm_model,
            device_ids=[rank],
        )
        # training
        print("training")
        lstm_model.train()
        epoch_loss = {epoch:[] for epoch in range(NUM_EPOCHS)}
        for epoch in range(NUM_EPOCHS):
            train_sampler.set_epoch(epoch)
            for inputs, y in dataloader:
                optimizer.zero_grad()
                y_hat = lstm_model(inputs)
                loss = criterion(y_hat.flatten(), y)
                loss.backward()
                torch.distributed.all_reduce(loss)             
                loss /= world_size
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
        torch.distributed.destroy_process_group()