import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from plot_functions import plot_results, plot_diff, plot_MSE, correlation_map, calc_plot_bias, plot_Europe_avg, pixel_correlation_map, pixels_biasmap, pixel_plot_MSE, chosenpixels_in_EU, chosenpixels_heatmap
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from LSTM_setup import *
from utils import singleregion_inputfeatures, singleregion_targetvar, multiregion_inputfeatures, multiregion_targetvar, meanstd_inputfeatures, meanstd_targetvar


def load_LSTM(source_path, model_path):
    # define mean and std dictionary
    means_stds = {}
    # prepare input data, standardization, lookback and train time series
    #train_inputs, means_stds = singleregion_inputfeatures(0, TRAINING_PERIOD, means_stds)
    #train_inputs, means_stds = multiregion_inputfeatures(0, TRAINING_PERIOD, means_stds, source_path)
    means_stds = meanstd_inputfeatures(0, TRAINING_PERIOD, means_stds, source_path)
    # prepare input data of target variable and standardize
    #obs_stand_train, means_stds = singleregion_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds)
    #obs_stand_train, means_stds = multiregion_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds, source_path)
    means_stds = meanstd_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds, source_path)

    # load the model
    lstm_model = torch.load(model_path)
    criterion = nn.MSELoss()

    ################# Testing ######################
    obs_stand_input, means_stds = singleregion_targetvar(TRAINING_PERIOD+LOOKBACK, TRAINING_PERIOD+TEST_PERIOD, means_stds)
    features_stand_inputs, means_stds = singleregion_inputfeatures(TRAINING_PERIOD, TRAINING_PERIOD+TEST_PERIOD, means_stds)

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

def calc_2Dheatmap_MSE(obs, sim):
    criterion = nn.MSELoss()
    cell_mse = [criterion(torch.tensor(obs[:,i]).float(), torch.tensor(sim[:,i]).float()).item() for i in range(obs.shape[1])]
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

    obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand.npy"))
    sim_destand_test = np.load(os.path.join(OUTPUTPATH, f"sim_destand.npy"))
    
    #convert units to meter
    obs_destand_test = obs_destand_test/1000
    sim_destand_test = sim_destand_test/1000

    obs_destand_test = np.nan_to_num(obs_destand_test)
    sim_destand_test = np.nan_to_num(sim_destand_test)

    obs_destand_test[obs_destand_test < 0.01] = 0.0
    sim_destand_test[sim_destand_test < 0.01] = 0.0


    correlation_map(obs_destand_test, sim_destand_test, f"correlation_{TARGET_REGION}", X, Y, lons, lats)
    #pixel_correlation_map(obs_destand_test, sim_destand_test, f"correlation_{TARGET_REGION}", X, Y)
    #pixels_biasmap(obs_destand_test, sim_destand_test, f"bias_{TARGET_REGION}", X, Y)
    calc_plot_bias(obs_destand_test, sim_destand_test, f"bias_{TARGET_REGION}", X, Y, lons, lats)
    #plot_blenaltman(obs_destand_test, sim_destand_test, TARGET_REGION, f"{model}_biasBA_{title}{TARGET_REGION}", X, Y)
    mse_map = calc_2Dheatmap_MSE(obs_destand_test, sim_destand_test)
    plot_MSE(mse_map, f"MSE_{TARGET_REGION}", "MSE", lons, lats)
    #pixel_plot_MSE(mse_map, f"MSE_{TARGET_REGION}", "MSE")

    obs_destand_test_avg = calc_datamap(obs_destand_test)
    sim_destand_test_avg = calc_datamap(sim_destand_test)
    #diff_destand_test_avg = calc_diffmap(obs_destand_test_avg, sim_destand_test_avg)

    plot_results(obs_destand_test_avg, f"avgwtd_obs_{TARGET_REGION}","Water table depth (m)", lons, lats)

    plot_results(sim_destand_test_avg, f"avgwtd_sim_{TARGET_REGION}", "Water table depth (m)", lons, lats)

    #plot_diff(diff_destand_test_avg, f"avgwtd_diff_{title}{TARGET_REGION}", "Water table depth (m)", lons, lats)

def timeseries_plot(pixel):
    obs_destand_test = np.load(os.path.join(OUTPUTPATH, "batch_EU_px_training2_wtd_49px_32_prvpdTxTnsmxyindlonlat", "obs_destand.npy"))
    sim_destand_test = np.load(os.path.join(OUTPUTPATH, "batch_EU_px_training2_wtd_49px_32_prvpdTxTnsmxyindlonlat", "sim_destand.npy"))

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
    plt.savefig(os.path.join(OUTPUTPATH, f"timeseries_{pixel[0]}_{pixel[1]}.png"))

def compare_timeseries_plots(pixel):
    lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
    lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

    outputs = {"DOURO_wtd_5x5_100_64_prvpdsmslopexysoilind":[], "SEINE+DOURO_wtd_5x5_100_64_prvpdsmslopexysoilind":[]}
    labels = {"DOURO_wtd_5x5_100_64_prvpdsmslopexysoilind":"Training region: Seine", "SEINE+DOURO_wtd_5x5_100_64_prvpdsmslopexysoilind":"Training region: Seine+Douro"}
    
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
    plt.savefig(os.path.join(OUTPUTPATH, f"timeseries_{pixel[0]}_{pixel[1]}.png"))

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

def calc_2Dheatmap_correlation(obs, sim):
    correlation_map = np.zeros(NB_CELLS)
    # Iterate over each grid cell
    for i in range(obs.shape[1]):
        # Extract the time series for the current grid cell
        time_series1 = obs[:, i]
        time_series2 = sim[:, i]
        # Calculate the Pearson correlation coefficient
        if np.std(time_series1) > 0 and np.std(time_series2) > 0:  # Avoid division by zero
            correlation_matrix = np.corrcoef(time_series1, time_series2)
            r = correlation_matrix[0, 1]
        else:
            r = np.nan  # If there's no variation, set correlation to NaN
        
        # Store the correlation coefficient in the map
        correlation_map[i] = r
    return correlation_map.reshape(X,Y)

def calc_mean_2D_bias(obs, sim):
    # calculate bias
    bias_map = np.mean(sim - obs, axis=0)
    return bias_map

def calc_2D_correlation(obs, sim):
    correlation_map = np.zeros((obs.shape[1],obs.shape[2]))
    # Iterate over each grid cell
    for i in range(obs.shape[1]):
        for j in range(obs.shape[2]):
            # Extract the time series for the current grid cell
            time_series1 = obs[:, i, j]
            time_series2 = sim[:, i, j]
            # Calculate the Pearson correlation coefficient
            if np.std(time_series1) > 0 and np.std(time_series2) > 0:  # Avoid division by zero
                correlation_matrix = np.corrcoef(time_series1, time_series2)
                r = correlation_matrix[0, 1]
            else:
                r = np.nan  # If there's no variation, set correlation to NaN
            
            # Store the correlation coefficient in the map
            correlation_map[i, j] = r
    return correlation_map

def calc_2D_MSE(obs, sim, mapping, choices):
    criterion = nn.MSELoss()
    mse1D = [criterion(torch.tensor(obs[:,i]).float(), torch.tensor(sim[:,i]).float()).item() for i in range(obs.shape[1])]
    mse1D = np.array(mse1D)
    # reshape into 2D
    mse_choicesmap = np.zeros(mapping.shape)
    mse_choicesmap[mse_choicesmap==0] = np.nan
    include = np.where(mapping==1)
    for i, choice in enumerate(choices):
        mse_choicesmap[include[0][choice],include[1][choice]] = mse1D[i]
    return mse_choicesmap


def EUpx_results():
    obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand.npy"))
    sim_destand_test = np.load(os.path.join(OUTPUTPATH, f"sim_destand.npy"))
    
    #convert units to meter
    obs_destand_test = obs_destand_test/1000
    sim_destand_test = sim_destand_test/1000

    obs_destand_test = np.nan_to_num(obs_destand_test)
    sim_destand_test = np.nan_to_num(sim_destand_test)

    obs_destand_test[obs_destand_test < 0.01] = 0.0
    sim_destand_test[sim_destand_test < 0.01] = 0.0

    choices = np.load(os.path.join(INPUTPATH, "choices.npy"))
    mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "mapping_px.npy"))
    ############## for random from scratch ###################
    included = np.load(os.path.join(INPUTPATH, "included_excl_waterbodies.npy"))
    mapping = included
    ############## remove it if not for random from all dataset excluding water bodies ###################    
    obs = np.zeros((obs_destand_test.shape[0], mapping.shape[0], mapping.shape[1]))
    obs[obs==0] = np.nan
    sim = np.zeros((sim_destand_test.shape[0], mapping.shape[0], mapping.shape[1]))
    sim[sim==0] = np.nan

    include = np.where(mapping==1)
    for i, choice in enumerate(choices):
        obs[:,include[0][choice],include[1][choice]] = obs_destand_test[:,i]
        sim[:,include[0][choice],include[1][choice]] = sim_destand_test[:,i]

    # calculate and plot mean bias
    mean_bias_2D_map = calc_mean_2D_bias(obs, sim)
    mean_bias_2D_heatmap = calc_mean_2D_bias(obs_destand_test, sim_destand_test)
    chosenpixels_in_EU(mean_bias_2D_map, False, -1, 1, f"Bias")
    chosenpixels_heatmap(mean_bias_2D_heatmap.reshape(X,Y), False, -1, 1, f"Bias")

    # calculate and plot correlation
    correlation_2D_map = calc_2D_correlation(obs, sim)
    correlation_2D_heatmap = calc_2Dheatmap_correlation(obs_destand_test, sim_destand_test)
    chosenpixels_in_EU(correlation_2D_map, False, -1, 1, f"Correlation")
    chosenpixels_heatmap(correlation_2D_heatmap, False, -1, 1, f"Correlation")

    # calculate and plot MSE
    mse_2D_heatmap = calc_2Dheatmap_MSE(obs_destand_test, sim_destand_test)
    mse_2D_map = calc_2D_MSE(obs_destand_test, sim_destand_test, mapping, choices)
    chosenpixels_in_EU(mse_2D_map, True, 0.001, 10, f"MSE")
    chosenpixels_heatmap(mse_2D_heatmap, True, 0.001, 10, f"MSE")

#load_LSTM(
#    os.path.join(os.path.dirname(INPUTPATH), "maxcorr100_EU"),
#    os.path.join(os.path.dirname(OUTPUTPATH), "maxcorr100_EU", f"maxcorr100_EU_{MODEL_NAME}.pt")
#    )
#load_results()
#timeseries_plot((2,4))
EUpx_results()