import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from plot_functions import plot_results, plot_diff, plot_MSE, correlation_map, calc_plot_bias, plot_Europe_avg
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
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


def load_LSTM(model_path):
    # define mean and std dictionary
    means_stds = {}
    # prepare input data, standardization, lookback and train time series
    train_inputs, means_stds = prepare_inputfeatures(0, TRAINING_PERIOD, means_stds)

    # prepare input data of target variable and standardize
    obs_stand_train, means_stds = prepare_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds)

    # load the model
    lstm_model = torch.load(model_path)
    #lstm_model = torch.load(os.path.join(os.path.dirname(DIRPATH), SOURCE_REGION, f'{SOURCE_REGION}_wtd_test.pt'))
    #lstm_model = torch.load(os.path.join(DIRPATH, f'{TARGET_REGION}_wtd_test.pt'))
    #lstm_model = torch.load(os.path.join(DIRPATH, f'wtd_{SOURCE_REGION}_TL_FT{TARGET_REGION}.pt'))
    criterion = nn.MSELoss()

    ################# Testing ######################
    obs_stand_input, means_stds = prepare_targetvar(TRAINING_PERIOD+LOOKBACK, TRAINING_PERIOD+TEST_PERIOD, means_stds)
    features_stand_inputs, means_stds = prepare_inputfeatures(TRAINING_PERIOD, TRAINING_PERIOD+TEST_PERIOD, means_stds)

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
            total_loss += loss.item()

    test_loss = total_loss / len(test_dataloader)
    print(f'Test Loss: {test_loss:.4f}')

    sim_stand = torch.cat(test_s).numpy().reshape(TEST_PERIOD-LOOKBACK, NB_CELLS)
    obs_stand = torch.cat(test_o).numpy().reshape(TEST_PERIOD-LOOKBACK, NB_CELLS)

    # standardization
    # compare it with the original values to confirm the standardization process
    obs_destand = obs_stand*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"]

    sim_destand = sim_stand*means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] + means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"]

    np.save(os.path.join(OUTPUTPATH, "obs_destand.npy"), obs_destand)
    np.save(os.path.join(OUTPUTPATH, "sim_destand.npy"), sim_destand)
    return obs_destand, sim_destand, obs_stand, sim_stand


def finetune_LSTM():
    # define mean and std dictionary
    means_stds = {}
    # prepare input data, standardization, lookback and train time series
    train_inputs, means_stds = prepare_inputfeatures(0, TRAINING_PERIOD, means_stds)

    # prepare input data of target variable and standardize
    obs_stand_train, means_stds = prepare_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds)

    # load the model
    lstm_model = torch.load(os.path.join(os.path.dirname(OUTPUTPATH), SOURCE_REGION, f'{SOURCE_REGION}_wtd.pt'))

    print("creating dataloader")
    print(f"train features shape: {train_inputs.shape}") # (5*5*timeseries, lookback, features)
    print(f"train target shape: {obs_stand_train.shape}")

    dataset = TensorDataset(torch.tensor(train_inputs).float(), torch.tensor(obs_stand_train).float())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # initialization
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
    torch.save(lstm_model, os.path.join(OUTPUTPATH, f'wtd_{SOURCE_REGION}_TL_FT{TARGET_REGION}.pt'))
    print("fine-tuned model saved")
    return
    
def benchmark_vs_TL():
    print("preparing benchmark model")
    # load benchmark model
    obs_destand_BM, sim_destand_BM, obs_stand_BM, sim_stand_BM = load_LSTM(os.path.join(OUTPUTPATH, f'{TARGET_REGION}_wtd.pt'))
    # load TL model
    obs_destand_TL, sim_destand_TL, obs_stand_TL, sim_stand_TL = load_LSTM(os.path.join(os.path.dirname(OUTPUTPATH), SOURCE_REGION, f'{SOURCE_REGION}_wtd.pt'))

    criterion = nn.MSELoss()
    cell_mse = [criterion(torch.tensor(sim_destand_BM[:,i]).float(), torch.tensor(sim_destand_TL[:,i]).float()).item() for i in range(NB_CELLS)]
    cell_mse = np.array(cell_mse)
    cell_mse = cell_mse.reshape(X,Y)

    plot_MSE(cell_mse, TARGET_REGION, f"MSE_{SOURCE_REGION}_TL_FT{TARGET_REGION}_BMvsTL","MSE")

def calc_MSE(obs_destand_test, sim_destand_test):
    criterion = nn.MSELoss()
    cell_mse = [criterion(torch.tensor(obs_destand_test[:,i]).float(), torch.tensor(sim_destand_test[:,i]).float()).item() for i in range(NB_CELLS)]
    cell_mse = np.array(cell_mse)
    cell_mse = cell_mse.reshape(X,Y)
    return cell_mse

def calc_datamap(obs_destand_test):
    #average over timeseries
    obs_destand_test_avg = np.mean(obs_destand_test, axis=0)
    # reshape to map
    obs_destand_test_avg = obs_destand_test_avg.reshape(X,Y)
    # set nan values to 0
    obs_destand_test_avg = np.nan_to_num(obs_destand_test_avg)
    # set all values less than 0.01 to nan (meter)
    obs_destand_test_avg[obs_destand_test_avg < 0.0] = 0
    obs_destand_test_avg[obs_destand_test_avg == 0] = np.nan
    obs_destand_test_avg[obs_destand_test_avg < 0.01] = np.nan
    # set all values less than 0.01 to 0 (meter)
    obs_destand_test_avg = np.nan_to_num(obs_destand_test_avg)
    return obs_destand_test_avg

def calc_diffmap(obs_destand_test_avg, sim_destand_test_avg):
    # calculate difference between observed and simulated
    diff_destand_test_avg = obs_destand_test_avg - sim_destand_test_avg
    # set negative values to positive
    diff_destand_test_avg = np.absolute(diff_destand_test_avg)
    # set all difference values less than 0.01 to 0.01 (meter) 
    diff_destand_test_avg[diff_destand_test_avg == 0] = np.nan
    diff_destand_test_avg[diff_destand_test_avg < 0.01] = 0.01
    # set all nan values to 0
    diff_destand_test_avg = np.nan_to_num(diff_destand_test_avg)
    return diff_destand_test_avg

def load_results():
    lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
    lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

    tl = False
    TL_title = f"{SOURCE_REGION}_TL_FT" if tl else ""
    if tl:
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, "sim_destand.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, "sim_destand_TL.npy"))    
    else:
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, f"sim_destand.npy"))
    
    #convert units to meter
    obs_destand_test = obs_destand_test/1000
    sim_destand_test = sim_destand_test/1000

    obs_destand_test = np.nan_to_num(obs_destand_test)
    sim_destand_test = np.nan_to_num(sim_destand_test)

    obs_destand_test[obs_destand_test < 0.01] = 0.0
    sim_destand_test[sim_destand_test < 0.01] = 0.0


    correlation_map(obs_destand_test, sim_destand_test, f"correlation_{TL_title}{TARGET_REGION}", X, Y, lons, lats)
    calc_plot_bias(obs_destand_test, sim_destand_test, f"bias_{TL_title}{TARGET_REGION}", X, Y, lons, lats)
    #plot_blenaltman(obs_destand_test, sim_destand_test, TARGET_REGION, f"{model}_biasBA_{TL_title}{TARGET_REGION}", X, Y)
    mse_map = calc_MSE(obs_destand_test, sim_destand_test)
    plot_MSE(mse_map, f"MSE_{TL_title}{TARGET_REGION}", "MSE", lons, lats)

    obs_destand_test_avg = calc_datamap(obs_destand_test)
    sim_destand_test_avg = calc_datamap(sim_destand_test)
    diff_destand_test_avg = calc_diffmap(obs_destand_test_avg, sim_destand_test_avg)

    plot_results(obs_destand_test_avg, f"avgwtd_obs_{TL_title}{TARGET_REGION}","Water table depth (m)", lons, lats)

    plot_results(sim_destand_test_avg, f"avgwtd_sim_{TL_title}{TARGET_REGION}", "Water table depth (m)", lons, lats)

    #plot_diff(diff_destand_test_avg, f"avgwtd_diff_{TL_title}{TARGET_REGION}", "Water table depth (m)", lons, lats)

def timeseries_plot(pixel):
    lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
    lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

    tl = False
    TL_title = f"{SOURCE_REGION}_TL_FT" if tl else ""
    if tl:
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, "sim_destand.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, "sim_destand_TL.npy"))    
    else:
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, "obs_destand.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, "sim_destand.npy"))

    #convert units to meter
    obs_destand_test = obs_destand_test/1000
    sim_destand_test = sim_destand_test/1000

    obs_destand_test = obs_destand_test.reshape(obs_destand_test.shape[0],X,Y)
    sim_destand_test = sim_destand_test.reshape(sim_destand_test.shape[0],X,Y)
    
    obs_destand_test[obs_destand_test < 0.01] = 0.0
    sim_destand_test[sim_destand_test < 0.01] = 0.0


    dates = pd.date_range(start='2009-01-01', end='2010-12-31', freq='D')
    dates = dates[(dates.month != 2) | (dates.day != 29)]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    print("plotting timeseries")
    ax.plot(dates, obs_destand_test[:,pixel[0],pixel[1]], "k-", label="Original simulations")
    ax.plot(dates, sim_destand_test[:,pixel[0],pixel[1]], "k--", label="Predicted")
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

    plt.xticks(rotation=45)
    #plt.yscale("log")
    plt.ylabel('Water table depth (m)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTPATH, f"timeseries_pointseine.png"))

def compare_timeseries_plots(pixel):
    lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
    lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

    outputs = {"10yrs_model":[], "topoporo":[], "topoporo128":[], "topoporo256":[], "poro64":[], "topo64":[]}
    labels = {"10yrs_model":"without static (64)", "topoporo":"topography+porosity (64)", "topoporo128":"topography+porosity (128)", "topoporo256":"topography+porosity (256)", "poro64":"porosity (64)", "topo64":"topography (64)"}
    
    for output in outputs.keys():
        outputs[output] = np.load(os.path.join(OUTPUTPATH, output, "sim_destand.npy"))
        outputs[output] = outputs[output]/1000
        outputs[output] = outputs[output].reshape(outputs[output].shape[0],X,Y)
        #outputs[output][outputs[output] < 0.01] = 0.0


    obs_destand_test = np.load(os.path.join(OUTPUTPATH, output, "obs_destand.npy"))

    #convert units to meter
    obs_destand_test = obs_destand_test/1000
    obs_destand_test = obs_destand_test.reshape(obs_destand_test.shape[0],X,Y)
    #obs_destand_test[obs_destand_test < 0.01] = 0.0


    dates = pd.date_range(start='2009-01-01', end='2010-12-31', freq='D')
    dates = dates[(dates.month != 2) | (dates.day != 29)]
    fig, ax = plt.subplots(figsize=(10, 6))

    print("plotting timeseries")
    ax.plot(dates, obs_destand_test[:,pixel[0],pixel[1]], "k-", label="Original simulations")
    for output in outputs.keys():
        ax.plot(dates, outputs[output][:,pixel[0],pixel[1]], "--", label=labels[output])
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

    plt.xticks(rotation=45)
    #plt.yscale("log")
    plt.ylabel('Water table depth (m)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUTPATH, f"timeseries_pointseine.png"))

def bias_timeseries_plot(pixel):
    lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
    lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

    tl = False
    TL_title = f"{SOURCE_REGION}_TL_FT" if tl else ""
    if tl:
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, "sim_destand.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, "sim_destand_TL.npy"))    
    else:
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, "obs_destand.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, "sim_destand.npy"))

    #convert units to meter
    obs_destand_test = obs_destand_test/1000
    sim_destand_test = sim_destand_test/1000

    obs_destand_test = obs_destand_test.reshape(obs_destand_test.shape[0],X,Y)
    sim_destand_test = sim_destand_test.reshape(sim_destand_test.shape[0],X,Y)
    
    obs_destand_test[obs_destand_test < 0.01] = 0.0
    sim_destand_test[sim_destand_test < 0.01] = 0.0

    dates = pd.date_range(start='2009-01-01', end='2010-12-31', freq='D')
    dates = dates[(dates.month != 2) | (dates.day != 29)]
    fig, ax = plt.subplots(figsize=(10, 6))

    print("plotting timeseries bias")
    bias = sim_destand_test[:,pixel[0],pixel[1]] - obs_destand_test[:,pixel[0],pixel[1]]

    ax.plot(dates, bias, "k-")
    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

    plt.xticks(rotation=45)
    plt.ylabel('Bias (m)')
    plt.grid()
    plt.savefig(os.path.join(OUTPUTPATH, f"bias_pointseine.png"))

