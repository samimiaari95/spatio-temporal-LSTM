import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from LSTM_model.model.config import *
from LSTM_model.utils.utils import utilities

class validate_LSTM_model:
    def __init__(self) -> None:
        pass

    def eval_LSTM():
        utils = utilities()
        # import training mean and std (saved during training process)
        with open(os.path.join(OUTPUTPATH, f"meanstd_{MODEL_NAME}.pkl"), 'rb') as f:
            means_stds = pickle.load(f)
        f.close()

        # load the model
        device = torch.device('cpu')
        lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
        lstm_model.load_state_dict(torch.load(os.path.join(OUTPUTPATH, f"{MODEL_NAME}.pt"), map_location=device, weights_only=True))
        criterion = nn.MSELoss()

        ################# Testing ######################
        obs_stand_input, means_stds = utils.singleregion_targetvar(TRAINING_PERIOD+LOOKBACK, TRAINING_PERIOD+TEST_PERIOD, means_stds)
        features_stand_inputs, means_stds = utils.singleregion_inputfeatures(TRAINING_PERIOD, TRAINING_PERIOD+TEST_PERIOD, means_stds)

        print("preparing dataloader")
        print(f"inputs shape: {features_stand_inputs.shape}")
        print(f"obs shape: {obs_stand_input.shape}")
        test_dataset = TensorDataset(torch.tensor(features_stand_inputs).float(), torch.tensor(obs_stand_input).float())
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

        sim_stand = torch.cat(test_s).numpy().reshape(TEST_PERIOD-LOOKBACK, NB_CELLS)
        obs_stand = torch.cat(test_o).numpy().reshape(TEST_PERIOD-LOOKBACK, NB_CELLS)

        # standardization
        # compare it with the original values to confirm the standardization process
        obs_destand = obs_stand*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"]

        sim_destand = sim_stand*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"]

        np.save(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"), obs_destand)
        np.save(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy"), sim_destand)
        return print("saved destandardized arrays")