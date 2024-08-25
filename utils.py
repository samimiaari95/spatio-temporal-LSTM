import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

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

def verify_reshape():

    X = 5
    Y = 2
    T = 5
    LOOKBACK = 3
    INPUTS = 2

    a = np.arange(0,50,1)
    #print(a)
    b = a.reshape(T, X ,Y)
    b2 = np.moveaxis(b, 0, -1)

    c = b2.reshape(-1, T)
    
    x = []
    for i in range(LOOKBACK, T+1):
        input_seq = c[:, i-LOOKBACK:i]
        x.append(input_seq)
    inputs = np.array(x)

    d = inputs.reshape(-1, LOOKBACK)

    # next feature
    a = np.arange(50,100,1)
    #print(a)
    b = a.reshape(T, X ,Y)
    #print(b)
    b2 = np.moveaxis(b, 0, -1)
    
    c = b2.reshape(-1, T)
    #print(c)
    x = []
    for i in range(LOOKBACK, T+1):
        input_seq = c[:, i-LOOKBACK:i]
        x.append(input_seq)
    inputs = np.array(x)
    
    #inp = np.moveaxis(inputs, 0,-1)

    d2 = inputs.reshape(-1, LOOKBACK)
    print(d2)
    print(d2.shape)
    
    # concat
    #inputdata = np.append(d,d2, axis=1)
    inputdata = np.concatenate((d,d2), axis=1)
    print(inputdata)
    print(inputdata.shape)
    h = []
    for i in range(inputdata.shape[0]):
        x = inputdata[i].reshape(INPUTS, LOOKBACK)
        x = np.moveaxis(x, 0,1)
        h.append(x)
    h = np.array(h)
    print(h)
    print(h.shape)
    return
    
def postprocess_LSTM_features(filesnames, dirpath, start, end, LOOKBACK, INPUT_SIZE, means_stds,train_mode=True):
    totalinputdata = np.array([])
    for inputvar in filesnames:
        print(inputvar)
        data = np.load(os.path.join(dirpath, inputvar))
        timesteps = data.shape[0]
        data = np.moveaxis(data, 0, -1) # move time to the last axis
        data = data.reshape(-1, timesteps) # outputs shape (34*34, timeseries)

        # select training time series
        train_data = data[:, start:end]
        
        # calculate mean and std only if in train mode
        if train_mode:
            # calculate means and stds for each cell over it's timeseries
            means_stds[f"{inputvar.replace('.npy','')}mean"] = [np.mean(train_data[:, :]) for cell in range(train_data.shape[0])]
            means_stds[f"{inputvar.replace('.npy','')}std"] = [np.std(train_data[:, :]) for cell in range(train_data.shape[0])]

        # standardization for all cells over the time series
        stand_train = np.zeros(train_data.shape)
        for i in range(train_data.shape[0]): #shape[0] is the number of cells
            stand_train[i,:] = (train_data[i, :] - means_stds[f"{inputvar.replace('.npy','')}mean"][i]) / means_stds[f"{inputvar.replace('.npy','')}std"][i] if means_stds[f"{inputvar.replace('.npy','')}std"][i] != 0 else 0

        # generate lookback array
        lookback_arrays = [stand_train[:, i-LOOKBACK:i] for i in range(LOOKBACK, end-start)] 

        inputs = np.array(lookback_arrays) # NOTE (timeseries, nb_cells, lookback)
        # NOTE [t1[c1[lookback1, lookback2,..], c2[lookback], ..], 
        #       t2[c1[lookback], c2[lookback], ..], ..]

        #inputs = inputs.reshape((NB_CELLS*(TRAINING_PERIOD-LOOKBACK), LOOKBACK))
        inputs = inputs.reshape(-1, LOOKBACK) # same as the command in the previous comment
        # NOTE [t1c1[lookback1, lookback2,..], 
        #       t1c2[lookback1, lookback2], .., 
        #       t2c1[lookback1, lookback2], 
        #       t2c2[lookback1, lookback2], ..]

        totalinputdata = np.concatenate((totalinputdata,inputs), axis=1) if len(totalinputdata)>=1 else inputs
        # NOTE [t1c1[lb1f1, lb2f1, lb1f2, lb2f2], 
        #       t1c2[lb1f1, lb2f1, lb1f2, lb2f2], .., 
        #       t2c1[lb1f1, lb2f1, lb1f2, lb2f2]]

    # rearrange columns
    train_inputs = []
    for i in range(totalinputdata.shape[0]):
        x = totalinputdata[i].reshape(INPUT_SIZE, LOOKBACK)
        x = np.moveaxis(x, 0,1)
        train_inputs.append(x)
    train_inputs = np.array(train_inputs)
    # final output: (NB_CELLS*TRAINING_PERIOD, LOOKBACK, NB_FEATURES)​
    #[[t1c1[lb1f1, lb1f2], [lb2f1, lb2f2]], ​
    #[t1c2[lb1f1, lb1f2], [lb2f1, lb2f2]], .., ​
    #[t2c1[lb1f1, lb1f2], [lb2f1, lb2f2]]]
    return train_inputs, means_stds

def postprocess_LSTM_targetvar(targetfilename, dirpath, start, end, means_stds, train_mode=True):
    # load target data
    obs_data = np.load(os.path.join(dirpath, targetfilename))
    timesteps = obs_data.shape[0]
    obs_data = np.moveaxis(obs_data, 0, -1) # move time to the last axis
    obs_data = obs_data.reshape(-1, timesteps) # outputs shape (34*34, timeseries)
    # select required target train data
    obs_train = obs_data[:, start:end]
    # cell 828 is the max
    #plott = obs_train[405,:]
    #print(f"index: {np.where(plott == np.median(plott))[0]}")
    if train_mode:
        # calculate mean and std for each cell
        means_stds[f"{targetfilename.replace('.npy','')}mean"] = [np.mean(obs_train[:, :]) for cell in range(obs_train.shape[0])]
        means_stds[f"{targetfilename.replace('.npy','')}std"] = [np.std(obs_train[:, :]) for cell in range(obs_train.shape[0])]

    # standardization for each cell time series
    obs_stand_train = np.zeros(obs_train.shape)
    for i in range(obs_train.shape[0]):
        obs_stand_train[i,:] = (obs_train[i, :] - means_stds[f"{targetfilename.replace('.npy','')}mean"][i]) / means_stds[f"{targetfilename.replace('.npy','')}std"][i] if means_stds[f"{targetfilename.replace('.npy','')}std"][i] != 0 else 0

    # flatten to enter as input to LSTM model
    obs_stand_train = obs_stand_train.flatten()

    return obs_stand_train, means_stds

def plot_mse(cell_mse, test_loss, filename):
    ax = plt
    ax.figure(figsize=(16,9))
    #ax.imshow(cell_mse, cmap='hot', interpolation='nearest')
    ax.imshow(cell_mse, interpolation='nearest')
    ax.colorbar().ax.set_ylabel('MSE')
    ax.title(f"avg MSE (mm): {test_loss:.4f}")
    ax.ylabel("y (pixels)")
    ax.xlabel("x (pixels)")
    ax.savefig(filename)

def plot_wtdmap(data, title, filepath):
    avg_obs = plt
    avg_obs.figure(figsize=(16,9))
    avg_obs.imshow(data, interpolation='nearest')
    avg_obs.colorbar()
    avg_obs.title(title)
    avg_obs.ylabel("y (pixels)")
    avg_obs.xlabel("x (pixels)")
    avg_obs.savefig(filepath)
    