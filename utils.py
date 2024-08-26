import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from LSTM_setup import *

plt.rcParams.update({'font.size': 22})


def make_dir(dir_path):
    """
    Creates folders for every directory in the specified path, pass if the directory already exists
    Arguments:
    -   dir_path: path defined as a string
    Returns:
    -   Nothing
    Author: Sami
    """
    # iterate through the defined path
    while not os.path.isdir(dir_path):
        # check if the parent directory exists
        if not os.path.isdir(os.path.dirname(dir_path)):
            make_dir(os.path.dirname(dir_path))
        else:
            os.mkdir(dir_path)

def powerlaw_func(h, a, b):
        y = a*(h**b)
        return y

def linear_law(x, a, b) :
        return a + x * b

def read_nc(filepath, var):
    ncfile = nc.Dataset(filepath)
    return ncfile[var][:]

def open_nc(filepath):
    ncfile = nc.Dataset(filepath)
    variables = ncfile.variables
    for var in variables:
        if var!="lon" and var!="lat" and var!="time" and var!="rlat" and var!="rlon" and var!="rotated_pole" and var!="pressure" and var!="time_bnds":
            print(ncfile[var].long_name)
            #if ncfile[var].long_name == "2m relative humidity":
            #    print(ncfile[var])

def get_S4W_basin(lat2D, lon2D, region):
    """ return a mask

    Return a boolean mask-array (True = masked, False = not masked) based on
    a passed set of longitude and latitude values and the name of the prudence
    region.
    The shape of the mask-array is set equal to the shape of input lat2D.
    Source: http://prudence.dmi.dk/public/publications/PSICC/Christensen&Christensen.pdf p.38

    Input values:
    -------------
    lat2D:    ndarray
        2D latitude information for each pixel
    lon2D:    ndarray
        2D longitude information for each pixel
    prudName: str
        Short name of prudence region

    Return value:
    -------------
    prudMask: ndarray
        Ndarray of dtype boolean of the same shape as lat2D.
        True = masked; False = not masked
    """
    if (region=='SEINE'):
        regionMask = np.where((lat2D < 47.0) | (lat2D > 50.0)  | (lon2D < -2.0) | (lon2D >  3.0), False, True)
    else:
        print(f'Region {region} not found --> EXIT')
    return regionMask

def get_prudenceMask(lat2D, lon2D, prudName):
    """ return a prudance mask

    Return a boolean mask-array (True = masked, False = not masked) based on
    a passed set of longitude and latitude values and the name of the prudence
    region.
    The shape of the mask-array is set equal to the shape of input lat2D.
    Source: http://prudence.dmi.dk/public/publications/PSICC/Christensen&Christensen.pdf p.38

    Input values:
    -------------
    lat2D:    ndarray
        2D latitude information for each pixel
    lon2D:    ndarray
        2D longitude information for each pixel
    prudName: str
        Short name of prudence region

    Return value:
    -------------
    prudMask: ndarray
        Ndarray of dtype boolean of the same shape as lat2D.
        True = masked; False = not masked
    """
    if (prudName=='BI'):
        prudMask = np.where((lat2D < 50.0) | (lat2D > 59.0)  | (lon2D < -10.0) | (lon2D >  2.0), False, True)
    elif (prudName=='IP'):
        prudMask = np.where((lat2D < 36.0) | (lat2D > 44.0)  | (lon2D < -10.0) | (lon2D >  3.0), False, True)
    elif (prudName=='FR'):
        prudMask = np.where((lat2D < 44.0) | (lat2D > 50.0)  | (lon2D < -5.0) | (lon2D >  5.0), False, True)
    elif (prudName=='ME'):
        prudMask = np.where((lat2D < 48.0) | (lat2D > 55.0)  | (lon2D < 2.0) | (lon2D >  16.0), False, True)
    elif (prudName=='SC'):
        prudMask = np.where((lat2D < 55.0) | (lat2D > 70.0)  | (lon2D < 5.0) | (lon2D >  30.0), False, True)
    elif (prudName=='AL'):
        prudMask = np.where((lat2D < 44.0) | (lat2D > 48.0)  | (lon2D < 5.0) | (lon2D >  15.0), False, True)
    elif (prudName=='MD'):
        prudMask = np.where((lat2D < 36.0) | (lat2D > 44.0)  | (lon2D < 3.0) | (lon2D >  25.0), False, True)
    elif (prudName=='EA'):
        prudMask = np.where((lat2D < 44.0) | (lat2D > 55.0)  | (lon2D < 16.0) | (lon2D >  30.0), False, True)
    else:
        print(f'prudance region {prudName} not found --> EXIT')
    return prudMask

def delete_files(dirpath, key):
    for file in os.listdir(dirpath):
        if key in file:
            os.remove(os.path.join(dirpath, file))

def singleregion_inputfeatures(start, end, means_stds):
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

def singleregion_targetvar(start, end, means_stds):
    raw_data = np.load(os.path.join(INPUTPATH, TARGETVAR_FILE))
    raw_data = raw_data[:,:,:]
    raw_data = np.nan_to_num(raw_data)
    raw_data[raw_data < 0.0] = 0

    data = raw_data[start:end, :, :]
    
    if f"{TARGETVAR_FILE.replace('.npy','')}mean" not in means_stds.keys():
        means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"] = np.mean(data)
        means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] = np.std(data)

    data = (data - means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"])/means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"]

    data = data.flatten()

    #print(np.unique(np.equal(raw_data[start, :, :], data.reshape(end-start, X, Y)[0,:,:])))
    return data, means_stds

def multiregion_inputfeatures(start, end, means_stds):
    all_inputs = np.array([])
    for inputvar in FEATURES_FILES:
        var = np.array([])
        meanstd = np.array([])
        basins = [x for x in os.listdir(INPUTPATH) if os.path.isdir(os.path.join(INPUTPATH, x)) and x in SOURCE_REGION]
        for basin in basins:
            print(inputvar, basin)
            raw_data = np.load(os.path.join(INPUTPATH, basin, inputvar))
            raw_data = raw_data[:,:,:]
            data = raw_data.reshape(raw_data.shape[0], NB_CELLS)
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

def multiregion_targetvar(start, end, means_stds):
    data = np.array([])
    basins = [x for x in os.listdir(INPUTPATH) if os.path.isdir(os.path.join(INPUTPATH, x)) and x in SOURCE_REGION]
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