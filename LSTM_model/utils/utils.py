import os
import numpy as np
import netCDF4 as nc
from LSTM_model.model.config import *

class utilities:
    def __init__(self) -> None:
        pass

    def make_dir(self, dir_path):
        # iterate through the defined path
        while not os.path.isdir(dir_path):
            # check if the parent directory exists
            if not os.path.isdir(os.path.dirname(dir_path)):
                self.make_dir(os.path.dirname(dir_path))
            else:
                os.mkdir(dir_path)
    
    def write_file(filename, intval):
        with open(filename, 'w') as fp:
            fp.write(str(intval))
    
    def read_file(filename):
        with open(filename) as fp:
            return fp.read()

    def powerlaw_func(self, h, a, b):
            y = a*(h**b)
            return y

    def linear_law(self, x, a, b) :
            return a + x * b

    def read_nc(self, filepath, var):
        ncfile = nc.Dataset(filepath)
        return ncfile[var][:]

    def open_nc(self, filepath):
        ncfile = nc.Dataset(filepath)
        variables = ncfile.variables
        for var in variables:
            print(ncfile[var])

    def delete_files(self, dirpath, key):
        for file in os.listdir(dirpath):
            if key in file:
                os.remove(os.path.join(dirpath, file))

    def singleregion_inputfeatures(self, start, end, means_stds):
        all_inputs = np.array([])
        for inputvar in FEATURES_FILES:
            print(inputvar)
            raw_data = np.load(os.path.join(INPUTPATH, inputvar))
            raw_data = raw_data.reshape(raw_data.shape[0], NB_CELLS) if len(raw_data.shape)>2 else raw_data
            data = raw_data[start:end, :]
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
        f1 = None
        lookback_arrays = None
        data = None
        raw_data = None
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], lookback_arrays[0,:,-1].reshape(X,Y))))
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], f1[:,-1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], all_inputs[:,-1, -1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))

        return all_inputs, means_stds

    def singleregion_targetvar(self, start, end, means_stds):
        raw_data = np.load(os.path.join(INPUTPATH, TARGETVAR_FILE))
        raw_data = np.nan_to_num(raw_data)
        raw_data[raw_data < 0.0] = 0

        data = raw_data[start:end, :, :] if len(raw_data.shape)>2 else raw_data[start:end, :]
        
        if f"{TARGETVAR_FILE.replace('.npy','')}mean" not in means_stds.keys():
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"] = np.mean(data)
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] = np.std(data)

        data = (data - means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"])/means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"]

        data = data.flatten()
        raw_data = None
        #print(np.unique(np.equal(raw_data[start, :, :], data.reshape(end-start, X, Y)[0,:,:])))
        return data, means_stds

    def meanstd_inputfeatures(self, start, end, means_stds, source_path):
        for inputvar in FEATURES_FILES:
            print(inputvar)
            raw_data = np.load(os.path.join(source_path, inputvar))
            raw_data = raw_data.reshape(raw_data.shape[0], NB_CELLS) if len(raw_data.shape)>2 else raw_data
            data = raw_data[start:end, :]
            data = np.moveaxis(data, 0, -1) # (cells, timeseries)
            
            if f"{inputvar.replace('.npy','')}mean" not in means_stds.keys():
                means_stds[f"{inputvar.replace('.npy','')}mean"] = np.mean(data)
                means_stds[f"{inputvar.replace('.npy','')}std"] = np.std(data)
        return means_stds

    def meanstd_targetvar(self, start, end, means_stds, source_path):
        raw_data = np.load(os.path.join(source_path, TARGETVAR_FILE))

        raw_data = np.nan_to_num(raw_data)
        raw_data[raw_data < 0.0] = 0

        data = raw_data[start:end, :, :] if len(raw_data.shape)>2 else raw_data[start:end, :]
        
        if f"{TARGETVAR_FILE.replace('.npy','')}mean" not in means_stds.keys():
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"] = np.mean(data)
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] = np.std(data)
        return means_stds


    def multiregion_inputfeatures(self, start, end, means_stds):
        all_inputs = np.array([])
        for inputvar in FEATURES_FILES:
            var = np.array([])
            meanstd = np.array([])
            #basins = [x for x in os.listdir(INPUTPATH) if os.path.isdir(os.path.join(INPUTPATH, x)) and x in SOURCE_REGION]
            basins = ["SEINE_30x30", "DOURO_30x30"]
            for basin in basins:
                print(inputvar, basin)
                raw_data = np.load(os.path.join(INPUTPATH, basin, inputvar))
                raw_data = raw_data[:,:,:]
                data = raw_data.reshape(raw_data.shape[0], int(raw_data.shape[1]*raw_data.shape[2]))
                data = data[start:end, :]
                data = np.moveaxis(data, 0, -1) # (cells, timeseries)

                # take the original training time series to calculate mean and std
                meanstd = np.concatenate((meanstd,data), axis=0) if len(meanstd)>0 else np.array(data)
                
                # create lookback
                lookback_arrays = [data[:, i-LOOKBACK:i] for i in range(LOOKBACK, end-start)]
                lookback_arrays = np.array(lookback_arrays)

                f1 = lookback_arrays.reshape(-1, LOOKBACK)
                f1 = np.array([f1])
                var = np.concatenate((var,f1), axis=1) if len(var)>0 else f1

            # standardization based on training period of all basins
            if f"{inputvar.replace('.npy','')}mean" not in means_stds.keys():
                means_stds[f"{inputvar.replace('.npy','')}mean"] = np.mean(meanstd)
                means_stds[f"{inputvar.replace('.npy','')}std"] = np.std(meanstd)
            var = (var - means_stds[f"{inputvar.replace('.npy','')}mean"])/means_stds[f"{inputvar.replace('.npy','')}std"]
            
            all_inputs = np.concatenate((all_inputs,var), axis=0) if len(all_inputs)>0 else var

        all_inputs = np.moveaxis(all_inputs, 0, -1)

        #print(np.unique(np.equal(raw_data[LOOKBACK-1], lookback_arrays[0,:,-1].reshape(X,Y))))
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], f1[:,-1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], all_inputs[:,-1, -1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))

        return all_inputs, means_stds

    def multiregion_targetvar(self, start, end, means_stds):
        data = np.array([])
        #basins = [x for x in os.listdir(INPUTPATH) if os.path.isdir(os.path.join(INPUTPATH, x)) and x in SOURCE_REGION]
        basins = ["SEINE_30x30", "DOURO_30x30"]
        # get data for all basins
        for basin in basins:
            raw_data = np.load(os.path.join(INPUTPATH, basin, TARGETVAR_FILE))
            raw_data = raw_data[:,:,:]
            raw_data = np.nan_to_num(raw_data)
            raw_data[raw_data < 0.0] = 0
            raw_data = raw_data[start:end, :, :]

            # concatenate all basins
            data = np.concatenate((data,raw_data), axis=0) if len(data)>0 else raw_data
        
        # standardization
        if f"{TARGETVAR_FILE.replace('.npy','')}mean" not in means_stds.keys():
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"] = np.mean(data)
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] = np.std(data)
        data = (data - means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"])/means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"]

        data = data.flatten()

        #print(np.unique(np.equal(raw_data[start, :, :], data.reshape(end-start, X, Y)[0,:,:])))
        return data, means_stds