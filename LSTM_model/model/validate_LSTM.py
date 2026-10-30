import logging
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from LSTM_model.model.config import (
    BATCH_SIZE,
    DROPOUT,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LOOKBACK,
    MODEL_NAME,
    NB_CELLS,
    NUM_LAYERS,
    OUTPUTPATH,
    OUTPUT_SIZE,
    SOURCE_REGION,
    TARGETVAR_FILE,
    TEST_PERIOD,
    TRAINING_PERIOD,
    AwesomeLSTM,
    member,
)
from LSTM_model.model.dataset import GridSequenceDataset


class validate_LSTM_model:
    def eval_LSTM(self) -> tuple[str, str]:
        means_stds_path = os.path.join(
            OUTPUTPATH, f"meanstd_{SOURCE_REGION}_{MODEL_NAME}.pkl"
        )
        with open(means_stds_path, "rb") as stream:
            means_stds = pickle.load(stream)

        device = torch.device("cpu")
        model = AwesomeLSTM(
            INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT
        ).to(device)
        model_path = os.path.join(OUTPUTPATH, f"{SOURCE_REGION}_{MODEL_NAME}.pt")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

        dataset = GridSequenceDataset(
            target_start=TRAINING_PERIOD + LOOKBACK,
            target_end=TRAINING_PERIOD + TEST_PERIOD,
            statistics=means_stds,
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        model.eval()
        predictions = []
        observations = []
        with torch.no_grad():
            for inputs, target in dataloader:
                prediction = model(inputs.to(device)).flatten()
                predictions.append(prediction.cpu())
                observations.append(target)

        target_mean = means_stds[f"{TARGETVAR_FILE[:-4]}mean"]
        target_std = means_stds[f"{TARGETVAR_FILE[:-4]}std"]
        sim_destandardized = (
            torch.cat(predictions).numpy().reshape(-1, NB_CELLS) * target_std
            + target_mean
        )
        obs_destandardized = (
            torch.cat(observations).numpy().reshape(-1, NB_CELLS) * target_std
            + target_mean
        )

        os.makedirs(OUTPUTPATH, exist_ok=True)
        obs_path = os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy")
        sim_path = os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy")
        np.save(obs_path, obs_destandardized)
        np.save(sim_path, sim_destandardized)
        return obs_path, sim_path

class consecutive_validate_LSTM_model:
    def eval_LSTM(self):
        from LSTM_model.utils.utils import utilities

        utils = utilities()
        # import training mean and std (saved during training process)
        with open(os.path.join(OUTPUTPATH, f"meanstd_{SOURCE_REGION}_{MODEL_NAME}.pkl"), 'rb') as f:
            means_stds = pickle.load(f)
        f.close()

        # load the model
        device = torch.device('cpu')
        lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT)
        lstm_model.load_state_dict(torch.load(os.path.join(OUTPUTPATH, f"{SOURCE_REGION}_{MODEL_NAME}.pt"), map_location=device, weights_only=True))
        criterion = nn.MSELoss()

        ################# Testing ######################
        obs_stand_input, means_stds = utils.singleregion_targetvar(TRAINING_PERIOD+LOOKBACK, TRAINING_PERIOD+TEST_PERIOD, means_stds)
        features_stand_inputs, means_stds = utils.singleregion_inputfeatures(TRAINING_PERIOD, TRAINING_PERIOD+TEST_PERIOD, means_stds)
        print(obs_stand_input.shape)
        print(features_stand_inputs.shape)
        
        all_obs_destand = np.zeros((TEST_PERIOD-LOOKBACK, NB_CELLS))
        all_sim_destand = np.zeros((TEST_PERIOD-LOOKBACK, NB_CELLS))

        for cell in range(NB_CELLS):
            # get indexes of timesteps for a specific cell
            cindxs = [i for i in range(cell, len(obs_stand_input), NB_CELLS)]
            # select timeseries of only the relevant cell
            cell_obs_stand_input = obs_stand_input[cindxs]
            cell_features_stand_inputs = features_stand_inputs[cindxs,:,:]
            print(cell_obs_stand_input.shape)
            print(cell_features_stand_inputs.shape)            

            print("preparing dataloader")
            print(f"inputs shape: {cell_features_stand_inputs.shape}")
            print(f"obs shape: {cell_obs_stand_input.shape}")
            test_dataset = TensorDataset(torch.tensor(cell_features_stand_inputs).float(), torch.tensor(cell_obs_stand_input).float())
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
                    print(f"loss={loss.item()}")
                    total_loss += loss.item()

            test_loss = total_loss / len(test_dataloader)
            print(f'Test Loss: {test_loss:.4f}')

            sim_stand = torch.cat(test_s).numpy()
            obs_stand = torch.cat(test_o).numpy()

            # standardization
            # compare it with the original values to confirm the standardization process
            obs_destand = obs_stand*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"]

            sim_destand = sim_stand*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"]

            all_obs_destand[:,cell] = obs_destand
            all_sim_destand[:,cell] = sim_destand

        np.save(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"), all_obs_destand)
        np.save(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy"), all_sim_destand)
        return print("saved destandardized arrays")

class transfer_ensemble_LSTM:
    def transfer_ensemble(self, batchind):
        from LSTM_model.utils.utils import utilities

        def get_local_rank():
            """Return the local rank of this process."""
            return int(os.getenv('LOCAL_RANK'))

        torch.distributed.init_process_group(backend='cpu:gloo,cuda:nccl')

        logger = logging.getLogger(__name__)
        logger.warning(f"Transferring for member {member} and batch index {batchind}")
        print(f"starting eval")
        utils = utilities()

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            print("No GPU available. Training will run on CPU.")

        # make sure to have cuda toolkit or PyTorch with GPU support installed before this step
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = get_local_rank()

        device = torch.device('cuda', local_rank)

        torch.cuda.set_device(device)


        # import training mean and std (saved during training process)
        with open(os.path.join(OUTPUTPATH, f"meanstd_{SOURCE_REGION}_{MODEL_NAME}.pkl"), 'rb') as f:
            means_stds = pickle.load(f)
        f.close()
        
        # load the model
        # device = torch.device('cpu')
        lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT)
        lstm_model.load_state_dict(torch.load(os.path.join(OUTPUTPATH, f"{SOURCE_REGION}_{MODEL_NAME}.pt"), map_location=device, weights_only=True))
        criterion = nn.MSELoss()

        lstm_model = lstm_model.to(device) # Move the model to the GPU
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        lstm_model = torch.nn.parallel.DistributedDataParallel(
            lstm_model,
            device_ids=[rank],
        )

        ################# Testing ######################
        obs_stand_input, means_stds = utils.transferpx_targetvar(TRAINING_PERIOD+LOOKBACK, TRAINING_PERIOD+TEST_PERIOD, means_stds, batchind)
        features_stand_inputs, means_stds = utils.transferpx_inputfeatures(TRAINING_PERIOD, TRAINING_PERIOD+TEST_PERIOD, means_stds, batchind)

        print("preparing dataloader")
        print(f"inputs shape: {features_stand_inputs.shape}")
        print(f"obs shape: {obs_stand_input.shape}")
        test_dataset = TensorDataset(torch.tensor(features_stand_inputs).float().to(device), torch.tensor(obs_stand_input).float().to(device))
        #test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, seed=0,)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)#sampler=test_sampler) # no random shuffling for the test

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
                print(f"loss={loss.item()}")
                total_loss += loss.item()

        test_loss = total_loss / len(test_dataloader)
        print(f'Test Loss: {test_loss:.4f}')

        sim_stand = torch.cat(test_s).cpu().numpy().reshape(TEST_PERIOD-LOOKBACK, int(len(torch.cat(test_s).cpu().numpy())/(TEST_PERIOD-LOOKBACK)))
        obs_stand = torch.cat(test_o).cpu().numpy().reshape(TEST_PERIOD-LOOKBACK, int(len(torch.cat(test_o).cpu().numpy())/(TEST_PERIOD-LOOKBACK)))

        # standardization
        # compare it with the original values to confirm the standardization process
        obs_destand = obs_stand*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"]

        sim_destand = sim_stand*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"]
        print(f"obs destand shape: {obs_destand.shape}")
        print(f"sim destand shape: {sim_destand.shape}")
        np.save(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}_{batchind}.npy"), obs_destand)
        np.save(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}_{batchind}.npy"), sim_destand)
        torch.distributed.destroy_process_group()
        return print("saved destandardized arrays")