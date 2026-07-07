import numpy as np
import pickle
import h5py
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

class transfer_ensemble_LSTM:
    def transfer_ensemble(self):
        def get_local_rank():
            """Return the local rank of this process."""
            return int(os.getenv('LOCAL_RANK'))

        torch.distributed.init_process_group(backend='cpu:gloo,cuda:nccl')

        logger = logging.getLogger(__name__)
        logger.warning(f"Transferring for member {member}")
        print(f"starting eval")

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
        with open(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", f"400px_member_{member}", f"meanstd_400px_member_{member}_{MODEL_NAME}.pkl"), 'rb') as f:
            means_stds = pickle.load(f)
        f.close()
        
        # load the model
        # device = torch.device('cpu')
        lstm_model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT)
        lstm_model.load_state_dict(torch.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", f"400px_member_{member}", f"400px_member_{member}_{MODEL_NAME}.pt"), map_location=device, weights_only=True))
        criterion = nn.MSELoss()

        lstm_model = lstm_model.to(device) # Move the model to the GPU
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        lstm_model = torch.nn.parallel.DistributedDataParallel(
            lstm_model,
            device_ids=[rank],
        )

        ################# Testing ######################
        # obs_stand_input, means_stds = utils.transferpx_targetvar(0, TEST_PERIOD-LOOKBACK, means_stds, batchind)
        # usually starting from LOOKBACK to TEST_PERIOD, but here we start from 0 to TEST_PERIOD since we don't provide wtd of TSMP for 1999, 
        # we don't need it because we will later compare to observations not TSMP
        # features_stand_inputs, means_stds = utils.transferpx_inputfeatures(0, TEST_PERIOD, means_stds, batchind)

        print("preparing dataloader")
        # print(f"inputs shape: {features_stand_inputs.shape}")
        # print(f"obs shape: {obs_stand_input.shape}")
        datasetpath = os.path.join("/p/scratch/cesmtst/miaari1/inputhdf5_files", f"standardized3dinput1dtarget_{member}.h5")
        test_dataset = MyDataset(datasetpath)
        # test_dataset = TensorDataset(torch.tensor(features_stand_inputs).float().to(device), torch.tensor(obs_stand_input).float().to(device))
        #test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, seed=0,)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)#sampler=test_sampler) # no random shuffling for the test

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
        # TODO battabombabav probably the means_std for target are not found in previous and so calculated based on this period
        obs_destand = obs_stand*means_stds[f"wtdstd"] + means_stds[f"wtdmean"]
        sim_destand = sim_stand*means_stds[f"wtdstd"] + means_stds[f"wtdmean"]
        print(f"obs destand shape: {obs_destand.shape}")
        print(f"sim destand shape: {sim_destand.shape}")
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", f"400px_member_{member}", f"obs_localobspixels.npy"), obs_destand)
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", f"400px_member_{member}", f"sim_ERA5onlocalobspixels.npy"), sim_destand)
        torch.distributed.destroy_process_group()
        return print("saved destandardized arrays")