import os
import math
import numpy as np
import skill_metrics as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from LSTM_model.utils.plot_functions import plotting_helper
from LSTM_model.utils.utils import utilities
from LSTM_model.model.config import *

class postprocess_calculations:
    def __init__(self) -> None:
        pass

    def calc_2Dheatmap_MSE(self, obs, sim):
        criterion = nn.MSELoss()
        cell_mse = [criterion(torch.tensor(obs[:,i]).float(), torch.tensor(sim[:,i]).float()).item() for i in range(obs.shape[1])]
        cell_mse = np.array(cell_mse)
        cell_mse = cell_mse.reshape(X,Y)
        return cell_mse

    def timeseries_plot(self, i,j):
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy"))

        obs_destand_test = obs_destand_test.reshape(obs_destand_test.shape[0],X,Y)
        sim_destand_test = sim_destand_test.reshape(sim_destand_test.shape[0],X,Y)
        
        obs_destand_test[obs_destand_test < 0.01] = 0.0
        sim_destand_test[sim_destand_test < 0.01] = 0.0

        dates = pd.date_range(start='2017-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))
        
        print(f"plotting timeseries {i}, {j}")
        ax.plot(dates, obs_destand_test[:,i,j], "k-", label="Original simulations")
        ax.plot(dates, sim_destand_test[:,i,j], "k--", label="Predicted")
        
        ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(OUTPUTPATH, f"timeseries_{i}_{j}.png"))

    def compare_timeseries_plots(self, pixel):
        outputs = {"100_256dr0x1lr01x50_365x1000_prvpdsmxyindlonlat":[], "100_256dr0x1lr01x50_365x1000_prvpdsmxyind":[]}
        labels = {"100_256dr0x1lr01x50_365x1000_prvpdsmxyindlonlat":"with Lon & Lat", "100_256dr0x1lr01x50_365x1000_prvpdsmxyind":"without Lon & Lat"}
        
        for output in outputs.keys():
            outputs[output] = np.load(os.path.join(OUTPUTPATH, f"{TARGET_REGION}_{output}", f"sim_destand_{output}.npy"))
            outputs[output] = outputs[output].reshape(outputs[output].shape[0],X,Y)


        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"{TARGET_REGION}_{output}", f"obs_destand_{output}.npy"))

        obs_destand_test = obs_destand_test.reshape(obs_destand_test.shape[0],X,Y)


        dates = pd.date_range(start='2017-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))

        print(f"plotting timeseries {pixel[0]}, {pixel[1]}")
        ax.plot(dates, obs_destand_test[:,pixel[0],pixel[1]], "k-", label="Original simulations")
        for output in outputs.keys():
            ax.plot(dates, outputs[output][:,pixel[0],pixel[1]], "--", label=labels[output])
        
        ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(OUTPUTPATH, f"timeseries_{pixel[0]}_{pixel[1]}.png"))

    def calc_2Dheatmap_correlation(self, obs, sim):
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

    def calc_mean_2D_bias(self, obs, sim):
        # calculate bias
        bias_map = np.mean(sim - obs, axis=0)
        return bias_map

    def calc_2D_correlation(self, obs, sim):
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

    def calc_2D_MSE(self, obs, sim, mapping):
        criterion = nn.MSELoss()
        mse1D = [criterion(torch.tensor(obs[:,i]).float(), torch.tensor(sim[:,i]).float()).item() for i in range(obs.shape[1])]
        mse1D = np.array(mse1D)
        # reshape into 2D
        mse_choicesmap = np.zeros(mapping.shape)
        mse_choicesmap[mse_choicesmap==0] = np.nan
        choices = np.where(mapping==1)
        for i in range(len(choices[0])):
            mse_choicesmap[choices[0][i],choices[1][i]] = mse1D[i]
        return mse_choicesmap

    def calc_rmse(self, y_true, y_pred, mapping):
        rmse = [np.sqrt(np.mean((y_true[:,i] - y_pred[:,i]) ** 2)) for i in range(y_true.shape[1])]
        rmse = np.array(rmse)
        # reshape into 2D
        rmse_choicesmap = np.zeros(mapping.shape)
        rmse_choicesmap[rmse_choicesmap==0] = np.nan
        choices = np.where(mapping==1)
        for i in range(len(choices[0])):
            rmse_choicesmap[choices[0][i],choices[1][i]] = rmse[i]

        rmse = rmse.reshape(X,Y)
        return rmse, rmse_choicesmap
    
    def calc_nse(self, y_true, y_pred, mapping):
        nse = [1 - (np.sum((y_true[:,i] - y_pred[:,i]) ** 2) / np.sum((y_true[:,i] - np.mean(y_true[:,i])) ** 2)) for i in range(y_true.shape[1])]
        nse = np.array(nse)
        # reshape into 2D
        nse_choicesmap = np.zeros(mapping.shape)
        nse_choicesmap[nse_choicesmap==0] = np.nan
        choices = np.where(mapping==1)
        for i in range(len(choices[0])):
            nse_choicesmap[choices[0][i],choices[1][i]] = nse[i]

        nse = nse.reshape(X,Y)
        return nse, nse_choicesmap
    
    def EUpx_results(self):
        plot_functions = plotting_helper()
        util = utilities()
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy"))
        
        obs_destand_test = np.nan_to_num(obs_destand_test)
        sim_destand_test = np.nan_to_num(sim_destand_test)

        #obs_destand_test[obs_destand_test < 0.01] = 0.0
        #sim_destand_test[sim_destand_test < 0.01] = 0.0

        mapping = np.load(os.path.join(INPUTPATH, "choices.npy"))
        
        obs = np.zeros((obs_destand_test.shape[0], mapping.shape[0], mapping.shape[1]))
        obs[obs==0] = np.nan
        sim = np.zeros((sim_destand_test.shape[0], mapping.shape[0], mapping.shape[1]))
        sim[sim==0] = np.nan

        ############### for choices in 2D #################
        mapping_indexes = np.where(mapping==1)
        for i in range(len(mapping_indexes[0])):
            obs[:,mapping_indexes[0][i],mapping_indexes[1][i]] = obs_destand_test[:,i]
            sim[:,mapping_indexes[0][i],mapping_indexes[1][i]] = sim_destand_test[:,i]

        # calculate and plot mean bias
        mean_bias_2D_map = self.calc_mean_2D_bias(obs, sim)
        mean_bias_2D_heatmap = self.calc_mean_2D_bias(obs_destand_test, sim_destand_test)
        plot_functions.chosenpixels_in_EU(mean_bias_2D_map, False, -10, 10, f"Mean bias")
        plot_functions.chosenpixels_heatmap(mean_bias_2D_heatmap.reshape(X,Y), False, -10, 10, f"Mean bias")

        # calculate and plot correlation
        correlation_2D_map = self.calc_2D_correlation(obs, sim)
        correlation_2D_heatmap = self.calc_2Dheatmap_correlation(obs_destand_test, sim_destand_test)
        plot_functions.chosenpixels_in_EU(correlation_2D_map, False, 0, 1, f"Correlation")
        plot_functions.chosenpixels_heatmap(correlation_2D_heatmap, False, 0, 1, f"Correlation")

        # calculate and plot MSE
        mse_2D_heatmap = self.calc_2Dheatmap_MSE(obs_destand_test, sim_destand_test)
        mse_2D_map = self.calc_2D_MSE(obs_destand_test, sim_destand_test, mapping)
        plot_functions.chosenpixels_in_EU(mse_2D_map, True, 0.01, 10, f"MSE")
        plot_functions.chosenpixels_heatmap(mse_2D_heatmap, True, 0.01, 10, f"MSE")

        # calculate and plot RMSE
        rmse_2D_heatmap, rmse_2D_map = self.calc_rmse(obs_destand_test, sim_destand_test, mapping)
        plot_functions.chosenpixels_in_EU(rmse_2D_map, True, 0.01, 10, f"RMSE")
        plot_functions.chosenpixels_heatmap(rmse_2D_heatmap, True, 0.01, 10, f"RMSE")

        # calculate and plot NSE
        nse_2D_heatmap, nse_2D_map = self.calc_nse(obs_destand_test, sim_destand_test, mapping)
        stds = np.std(obs_destand_test, axis=0).reshape(X,Y)
        nse_2D_heatmap[stds<0.1] = np.nan
        plot_functions.chosenpixels_in_EU(nse_2D_map, False, -1, 1, f"NSE")
        plot_functions.chosenpixels_heatmap(nse_2D_heatmap, False, -1, 1, f"NSE")

        # calculate and plot KGE
        kge_2D_heatmap = [util.calculate_kge(obs_destand_test[:,i], sim_destand_test[:,i]) for i in range(obs_destand_test.shape[1])]
        kge_2D_heatmap = np.array(kge_2D_heatmap)
        kge_2D_heatmap = kge_2D_heatmap.reshape(X,Y)
        kge_2D_map = np.zeros(mapping.shape)
        for i in range(obs.shape[1]):
            for j in range(obs.shape[2]):
                kge_2D_map[i,j] = util.calculate_kge(obs[:,i,j], sim[:,i,j])

        plot_functions.chosenpixels_in_EU(kge_2D_map, False, -1, 1, f"KGE")
        plot_functions.chosenpixels_heatmap(kge_2D_heatmap, False, -1, 1, f"KGE")

        #for i in range(X):
        #    for j in range(Y):
        #        pass
        #        self.plot_ensemble_timeseries(i, j, f'{correlation_2D_heatmap[i,j]:.2f}', f'{rmse_2D_heatmap[i,j]:.2f}', f'{kge_2D_heatmap[i,j]:.2f}')

    def calculate_accuracy_parameters(self):
        dirpath = os.path.join(OUTPUTPATH, f"{TARGET_REGION}_{MODEL_NAME}")
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy"))
        
        obs_destand_test = np.nan_to_num(obs_destand_test)
        sim_destand_test = np.nan_to_num(sim_destand_test)

        obs_destand_test[obs_destand_test < 0.01] = 0.0
        sim_destand_test[sim_destand_test < 0.01] = 0.0

        mapping = np.load(os.path.join(INPUTPATH, "choices.npy"))
        
        obs = np.zeros((obs_destand_test.shape[0], mapping.shape[0], mapping.shape[1]))
        obs[obs==0] = np.nan
        sim = np.zeros((sim_destand_test.shape[0], mapping.shape[0], mapping.shape[1]))
        sim[sim==0] = np.nan

        ############### for choices in 2D #################
        mapping_indexes = np.where(mapping==1)
        for i in range(len(mapping_indexes[0])):
            obs[:,mapping_indexes[0][i],mapping_indexes[1][i]] = obs_destand_test[:,i]
            sim[:,mapping_indexes[0][i],mapping_indexes[1][i]] = sim_destand_test[:,i]

        mean_bias_2D_heatmap = self.calc_mean_2D_bias(obs_destand_test, sim_destand_test)
        mean_bias_2D_heatmap = mean_bias_2D_heatmap.reshape(X,Y)
        correlation_2D_heatmap = self.calc_2Dheatmap_correlation(obs_destand_test, sim_destand_test)
        mse_2D_heatmap = self.calc_2Dheatmap_MSE(obs_destand_test, sim_destand_test)

        print(f"avg MSE: {np.mean(mse_2D_heatmap)}")
        print(f"avg corr: {np.mean(correlation_2D_heatmap)}")
        print(f"avg bias: {np.mean(mean_bias_2D_heatmap)}")
        print(f"median MSE: {np.median(mse_2D_heatmap)}")
        print(f"median corr: {np.median(correlation_2D_heatmap)}")
        print(f"90th% MSE: {np.percentile(mse_2D_heatmap, 90)}")
        print(f"10th% corr: {np.percentile(correlation_2D_heatmap, 10)}")

    def plot_ensemble_results(self):
        rmse = np.load(os.path.join(OUTPUTPATH, "rmse.npy"))
        corr = np.load(os.path.join(OUTPUTPATH, "corr.npy"))
        r2 = np.load(os.path.join(OUTPUTPATH, "r2.npy"))

        print("plotting dist")
        plt.figure(figsize=(8, 6))
        plt.hist(rmse, bins=100, color='blue', alpha=0.7, edgecolor='black')  # 50 bins for better resolution
        plt.xlabel('RMSE (m)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(OUTPUTPATH, f"rmse_dist.png"))

        plt.figure(figsize=(8, 6))
        plt.hist(corr, bins=100, color='blue', alpha=0.7, edgecolor='black')  # 50 bins for better resolution
        plt.xlim(-1,1)
        plt.xlabel('Correlation', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(OUTPUTPATH, f"corr_dist.png"))

        plt.figure(figsize=(8, 6))
        plt.hist(r2, bins=100, color='blue', alpha=0.7, edgecolor='black')  # 50 bins for better resolution
        #plt.title('Histogram of Data Distribution', fontsize=14)
        plt.xlim(-1,1)
        plt.xlabel('R2', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(OUTPUTPATH, f"r2_dist.png"))

    def plot_ensemble_timeseries(self, i, j, r, rmse, kge):
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        nb_members = 100
        print(obs_destand_test.shape)
        obs_destand_test[obs_destand_test < 0.0] = 0.0

        dates = pd.date_range(start='2017-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))
        
        print(f"plotting timeseries {i},{j}")
        timeserieslength = obs_destand_test.shape[0]
        obs_destand_test = obs_destand_test.reshape(timeserieslength,X,Y)
        ens_mean = np.zeros((timeserieslength, nb_members))
        for m in range(nb_members):
            sim_destand_test = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"100px_member_{m}", f"sim_destand_{MODEL_NAME}.npy"))
            sim_destand_test[sim_destand_test < 0.0] = 0.0
            sim_destand_test = sim_destand_test.reshape(timeserieslength,X,Y)
            ens_mean[:,m] = sim_destand_test[:,i,j]
            ax.plot(dates, sim_destand_test[:,i,j], color="gray")

        ax.plot(dates, obs_destand_test[:,i,j], "k-", label="Original simulations", linewidth=4.0)
        ax.plot(dates, np.mean(ens_mean, axis=1), "k--", label="Ensemble mean", linewidth=4.0)
        ax.scatter([], [], color="k", label=f"Correlation: {r}, RMSE: {rmse}, KGE: {kge}")

        ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3)
        plt.grid()
        plt.savefig(os.path.join(OUTPUTPATH, "transfer_timeseries", f"ensemble_timeseries_{i}_{j}.png"))

    def ens_mean(self):
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        obs_destand_test[obs_destand_test < 0.0] = 0.0

        sim = np.zeros(obs_destand_test.shape)
        for pixel in range(100):
            ens_mean = np.zeros((obs_destand_test.shape[0], 100))
            for m in range(100):
                sim_destand_test = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"100px_member_{m}", f"sim_destand_{MODEL_NAME}.npy"))
                sim_destand_test[sim_destand_test < 0.0] = 0.0
                ens_mean[:,m] = sim_destand_test[:,pixel]
            sim[:, pixel] = np.mean(ens_mean, axis=1)
        print(obs_destand_test.shape)
        print(sim.shape)
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean", f"obs_destand_{MODEL_NAME}.npy"), obs_destand_test)
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean", f"sim_destand_{MODEL_NAME}.npy"), sim)

    def ens_minMSE(self):
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        print(obs_destand_test.shape)
        obs_destand_test[obs_destand_test < 0.01] = 0.0
        criterion = nn.MSELoss()
        nb_members = 100

        sim = np.zeros(obs_destand_test.shape)
        for pixel in range(100):
            mse = np.zeros((nb_members))
            for m in range(nb_members):
                sim_destand_test = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy"))
                sim_destand_test[sim_destand_test < 0.01] = 0.0
                mse[m] = criterion(torch.tensor(obs_destand_test[:,pixel]).float(), torch.tensor(sim_destand_test[:,pixel]).float()).item()
            
            print(mse)
            print(np.argmin(mse))    
            minsim = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{np.argmin(mse)}", f"sim_destand_{MODEL_NAME}.npy"))
            sim[:, pixel] = minsim[:, pixel]
        
        print(obs_destand_test.shape)
        print(sim.shape)
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_minMSE", f"obs_destand_{MODEL_NAME}.npy"), obs_destand_test)
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_minMSE", f"sim_destand_{MODEL_NAME}.npy"), sim)

    def ens_timeseries_vs(self, i, j):
        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        obs[obs < 0.01] = 0.0

        dates = pd.date_range(start='2017-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))

        timeserieslength = obs.shape[0]
        obs = obs.reshape(timeserieslength,X,Y)
        sim100 = np.load(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy"))
        sim400 = np.load(os.path.join(OUTPUTPATH.replace("ensemble_weightedRMSE", "ensemble_mean"), f"sim_destand_{MODEL_NAME}.npy"))

        sim100[sim100 < 0.01] = 0.0
        sim400[sim400 < 0.01] = 0.0
        sim100 = sim100.reshape(timeserieslength,X,Y)
        sim400 = sim400.reshape(timeserieslength,X,Y)
        
        
        ax.plot(dates, obs[:,i,j], "k-", label="Original simulations")
        ax.plot(dates, sim100[:,i,j], "g--", label="Weighted mean")
        ax.plot(dates, sim400[:,i,j], "b--", label="Mean")

        ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(os.path.dirname(OUTPUTPATH), "timeseries_meanvsweightedRMSE", f"ensemble_timeseries_{i}_{j}.png"))

    def ens_minRMSE(self):
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        print(obs_destand_test.shape)
        obs_destand_test[obs_destand_test < 0.01] = 0.0
        nb_members = 100

        sim = np.zeros(obs_destand_test.shape)
        for pixel in range(100):
            rmse = np.zeros((nb_members))
            for m in range(nb_members):
                sim_destand_test = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy"))
                sim_destand_test[sim_destand_test < 0.01] = 0.0
                rmse[m] = np.sqrt(np.mean((obs_destand_test[:,pixel] - sim_destand_test[:,pixel]) ** 2))
            
            print(rmse)
            print(np.argmin(rmse))    
            minsim = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{np.argmin(rmse)}", f"sim_destand_{MODEL_NAME}.npy"))
            sim[:, pixel] = minsim[:, pixel]
        
        print(obs_destand_test.shape)
        print(sim.shape)
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_minRMSE", f"obs_destand_{MODEL_NAME}.npy"), obs_destand_test)
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_minRMSE", f"sim_destand_{MODEL_NAME}.npy"), sim)

    def ens_weightRMSE(self):
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        obs_destand_test[obs_destand_test < 0.01] = 0.0

        nb_members = 100
        
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        #members_sim = np.moveaxis(members_sim, 0, -1)   # (timeseries, pixels, members)
        members_sim[members_sim < 0.01] = 0.0
        
        sim = np.zeros(obs_destand_test.shape)
        for pixel in range(100):
            rmse_members = [np.sqrt(np.mean((obs_destand_test[:,pixel] - members_sim[m,:,pixel]) ** 2)) for m in range(nb_members)]
            # Inverse error weighting
            weights = 1 / np.array(rmse_members)
            weights /= weights.sum()  # Normalize weights
            #print("Weights assigned to ensemble members:", weights)
            # Weighted ensemble prediction
            sim[:, pixel] = np.dot(weights, members_sim[:,:,pixel])
        
        print(obs_destand_test.shape)
        print(sim.shape)
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_weightedRMSE", f"obs_destand_{MODEL_NAME}.npy"), obs_destand_test)
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_weightedRMSE", f"sim_destand_{MODEL_NAME}.npy"), sim)

    def plot_ensemble_statsvsacc_timeseries(self, i, r, stat):
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        nb_members = 100
        print(obs_destand_test.shape)
        obs_destand_test[obs_destand_test < 0.0] = 0.0

        dates = pd.date_range(start='2017-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))
        
        print(f"plotting timeseries {i} with r: {r} and stat: {stat}")
        #timeserieslength = obs_destand_test.shape[0]
        #obs_destand_test = obs_destand_test.reshape(timeserieslength,X,Y)
        ens_mean = np.zeros(obs_destand_test.shape)
        for m in range(nb_members):
            sim_destand_test = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy"))
            sim_destand_test[sim_destand_test < 0.0] = 0.0
            #sim_destand_test = sim_destand_test.reshape(timeserieslength,X,Y)
            ens_mean[:,m] = sim_destand_test[:,i]
            ax.plot(dates, sim_destand_test[:,i], color="gray")

        ax.plot(dates, obs_destand_test[:,i], "k-", label="Original simulations", linewidth=4.0)
        ax.plot(dates, np.mean(ens_mean, axis=1), "k--", label="Ensemble mean", linewidth=4.0)
        ax.scatter([], [], color="k", label=f"Correlation: {r}, Ensemble variance: {stat}")

        ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3)
        plt.grid()
        plt.savefig(os.path.join(OUTPUTPATH, "transfer_timeseries_statvsacc", f"ensemble_timeseries_{i}.png"))


    def ensemble_statvsacc(self):
        util = utilities()
        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        obs[obs < 0.0] = 0.0
        
        nb_members = 100
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        members_sim[members_sim < 0.0] = 0.0
        
        df_dic = {"Absolute mean bias":[], "Pearson correlation":[], "RMSE":[], "KGE":[], "Beta":[], "Alpha":[], "NSE":[], "Pairwise correlation":[], "Variance":[], "IQR (75-25%)":[], "IQR (100-0%)":[], "std":[], "cv":[],  "(Alpha-1)^2":[], "(Beta-1)^2":[], "(r-1)^2":[]}

        x_axis = []
        y_axis = []
        for pixel in range(100):
            ensemble_predictions = members_sim[:,:,pixel]
            ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
            # Calculate the mean prediction for each time step
            mean_prediction = np.mean(ensemble_predictions, axis=1)
            
            ########### Calculate ensemble statistics ###########
            # Calculate the variance for each time step
            ensemble_variance = np.var(ensemble_predictions, axis=1)
            ensemble_variance = np.mean(ensemble_variance)
            df_dic["Variance"].append(ensemble_variance)

            # Calculate ensemble statistics (e.g., diversity, spread, etc.)
            # Spread interquantile range (IQR) between 75th and 25th percentiles
            ensemble_iqr = np.percentile(ensemble_predictions, 75, axis=1) - np.percentile(ensemble_predictions, 25, axis=1)  # IQR
            iqr_mean = np.mean(ensemble_iqr)
            df_dic["IQR (75-25%)"].append(iqr_mean)

            iqr100 = np.percentile(ensemble_predictions, 100, axis=1) - np.percentile(ensemble_predictions, 0, axis=1)
            iqr100_mean = np.mean(iqr100)
            df_dic["IQR (100-0%)"].append(iqr100_mean)

            # Calculate the std for each time step
            ensemble_std = np.std(ensemble_predictions, axis=1)
            ensemble_std = np.mean(ensemble_std)
            df_dic["std"].append(ensemble_std)

            # Calculate diversity by Pairwise correlation
            correlation_matrix = np.corrcoef(ensemble_predictions.T)  # Transpose to get members on rows
            pairwisecorr = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]) # Compute diversity as 1 - average correlation
            df_dic["Pairwise correlation"].append(pairwisecorr)

            # calculate coefficient of variation
            cv = ensemble_std/np.mean(mean_prediction)
            df_dic["cv"].append(cv)
            
            ########### Calculate accuracy metrics ###########
            # Calculate the correlation between the mean prediction and observation
            correlationobs = np.corrcoef(mean_prediction, obs[:,pixel])[0, 1]
            df_dic["Pearson correlation"].append(correlationobs)

            # Calculate the RMSE between the mean prediction and observation
            rmse = np.sqrt(np.mean((obs[:,pixel] - mean_prediction) ** 2))
            df_dic["RMSE"].append(rmse)

            # Calculate MSE

            # Calculate KGE
            kge = util.calculate_kge(obs[:,pixel], mean_prediction)
            df_dic["KGE"].append(kge)
            
            #### KGE terms analysis ####
            # Compute mean and standard deviation
            mu_o, mu_p = np.mean(obs[:,pixel]), np.mean(mean_prediction)
            sigma_o, sigma_p = np.std(obs[:,pixel]), np.std(mean_prediction)
            
            # Compute bias ratio (β) and variability ratio (γ)
            beta = mu_p / mu_o
            alpha = sigma_p / sigma_o
            
            if np.std(obs[:,pixel]) < 0.1: # if the std is close to zero, exclude pixel kge
                alpha = np.nan
                beta = np.nan
            
            df_dic["Beta"].append(beta)
            df_dic["Alpha"].append(alpha)
            df_dic["(Alpha-1)^2"].append((alpha-1)**2)
            df_dic["(Beta-1)^2"].append((beta-1)**2)
            df_dic["(r-1)^2"].append((correlationobs-1)**2)


            # Calculate NSE
            nse = 1 - (np.sum((obs[:,pixel] - mean_prediction) ** 2) / np.sum((obs[:,pixel] - np.mean(obs[:,pixel])) ** 2))
            if np.std(obs[:,pixel]) < 0.1:
                nse = np.nan
            df_dic["NSE"].append(nse)

            # Calculate mean bias
            bias_mean = np.mean(mean_prediction - obs[:,pixel])
            df_dic["Absolute mean bias"].append(bias_mean)
            
            x_axis.append((correlationobs-1)**2)
            y_axis.append(kge)
            #self.plot_ensemble_statsvsacc_timeseries(pixel, f'{correlationobs:.2f}', f'{math.ceil(ensemble_variance)}')
            #if ensemble_variance>70 and correlationobs>0.85:
            #    print(ensemble_variance)
            #    print(correlationobs)
            #print(f"pixel index is: {pixel}")
            
        ####### save to csv ########
        df = pd.DataFrame(df_dic)
        df.to_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics.csv"), index=False)

        xtitle = "(r-1)^2"
        ytitle = "KGE"

        plt.figure()
        plt.scatter(x_axis, y_axis, marker='o', color='k')
        plt.xlabel(f'{xtitle}')
        plt.ylabel(f'{ytitle}')
        #plt.xlim(min(min(y_axis), min(x_axis)), max(max(y_axis), max(x_axis)))
        #plt.ylim(min(min(y_axis), min(x_axis)), max(max(y_axis), max(x_axis)))
        #plt.xscale("log")
        #plt.yscale("log")
        #plt.yscale("symlog")
        plt.xlim(-1,1)
        plt.ylim(-1, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        print(f"plotting: {ytitle}_{xtitle}")
        #plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"{ytitle}_{xtitle}.png"))

    def ensemble_statvsacc_fitting(self):
        utils = utilities()
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics.csv"))
        xstats = {"std":"exp", "Variance":"exp", "Pairwise correlation":"lin", "IQR (75-25%)":"exp", "IQR (100-0%)":"exp"}
        ystats = {"RMSE":"exp", "Pearson correlation":"lin", "KGE":"lin", "Absolute mean bias":"exp", "NSE":"lin", "Beta":"exp", "Alpha":"exp", "(Alpha-1)^2":"exp", "(Beta-1)^2":"exp", "(r-1)^2":"exp"}
        for xstat in xstats.keys():
            for ystat in ystats.keys():
                print(f"fitting {xstat} with {ystat}")

                logarithmicfit = False
                if xstats[xstat]=="exp" and ystats[ystat]=="exp":
                    fittype = "powerlaw"
                elif xstats[xstat]=="exp" or ystats[ystat]=="exp":
                    fittype = "exponential"
                    if xstats[xstat]=="exp" and ystats[ystat]=="lin":
                        logarithmicfit = True
                else:
                    fittype = "linear"
                
                xval = stat[xstat].values
                yval = stat[ystat].values if not ystat=="Absolute mean bias" else np.absolute(stat[ystat].values)

                if ystat=="Pearson correlation":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                    xval = xval[yval>=0.0]
                    yval = yval[yval>=0.0]
                
                if ystat=="KGE" or ystat=="NSE":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                    xval = xval[yval>=-1]
                    yval = yval[yval>=-1]

                if ystat=="Beta" or ystat=="Alpha" or ystat=="(Alpha-1)^2" or ystat=="(Beta-1)^2" or ystat=="(r-1)^2":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                
                if ystat=="Absolute mean bias":
                    xval = xval[yval>=0.01]
                    yval = yval[yval>=0.01]
                
                if xstat=="Pairwise correlation" and ystat=="Pearson correlation":
                    yval = yval[xval>=0.2]
                    xval = xval[xval>=0.2]
                #if xstat=="IQR (75-25%)" and ystat=="Pearson correlation":
                #    yval = yval[xval<=10]
                #    xval = xval[xval<=10]
                if xstat=="Pairwise correlation" and ystat=="KGE":
                    yval = yval[xval>=0.2]
                    xval = xval[xval>=0.2]
                if xstat=="Pairwise correlation" and ystat=="NSE":
                    yval = yval[xval>=0.6]
                    xval = xval[xval>=0.6]
                        
                
                logx, logy, xminlim, xmaxlim, yminlim, ymaxlim = None, None, None, None, None, None
                if xstat=="std" or xstat=="Variance" or xstat=="IQR (75-25%)" or xstat=="IQR (100-0%)":
                    logx = True
                if ystat=="RMSE" or ystat=="Absolute mean bias" or ystat=="Alpha" or ystat=="Beta" or ystat=="(Alpha-1)^2" or ystat=="(Beta-1)^2" or ystat=="(r-1)^2":
                    logy = True
                if xstat=="Pairwise correlation":
                    xminlim = 0
                    xmaxlim = 1
                if ystat=="Pearson correlation" or ystat=="NSE":
                    yminlim = 0
                    ymaxlim = 1
                if ystat=="KGE":
                    yminlim = -1
                    ymaxlim = 1
                
                if fittype=="linear":
                    # Fit a linear model
                    params, covariance = curve_fit(utils.linear_law, xval, yval)
                    a_fit, b_fit = params
                    # fitting accuracy
                    y_fit = [utils.linear_law(x, a_fit, b_fit) for x in xval]
                    R_square = r2_score(yval, y_fit)
                    #print(f"R2 = {R_square}")
                    #print(f"Fitted a: {a_fit} and b:{b_fit}")
                    lin_eq = f'$y={b_fit:.2f}x+{a_fit:.2f}$' if a_fit>=0 else f'$y={b_fit:.2f}x{a_fit:.2f}$'
                    textstr = '\n'.join((
                        lin_eq,
                        f'$R^2$ = {R_square:.3f}'))
                    x_fit = [min(xval), max(xval)]
                    y_fit = [utils.linear_law(x, a_fit, b_fit) for x in x_fit]


                if fittype=="powerlaw":
                    # Fit power law
                    # linearize
                    y_lin = np.log(yval)
                    x_lin = np.log(xval)
                    # Fit the function
                    params, covariance = curve_fit(utils.linear_law, x_lin, y_lin)
                    a_fit, b_fit = params
                    # fitting accuracy
                    y_fit = [utils.linear_law(x, a_fit, b_fit) for x in x_lin]
                    R_square = r2_score(y_lin, y_fit)
                    #print(f"R2 = {R_square}")
                    # back transform to power law
                    a_fit = np.exp(a_fit)
                    #print(f"Fitted a: {a_fit} and b:{b_fit}")
                    textstr = '\n'.join((
                        f'y={a_fit:.2f}x^{b_fit:.2f}',
                        f'$R^2$ = {R_square:.3f}'))

                    x_fit = [min(xval), max(xval)]
                    y_fit = [utils.powerlaw_func(x, a_fit, b_fit) for x in x_fit]

                if fittype=="exponential":
                    # Define a function to fit (e.g., exponential decay or polynomial)
                    def log_func(x, a, b):
                        return a + b * np.log(x)
                    def exp_func(x, a, b):
                        return a* np.exp(b * x)  # Exponential model without offset

                    # Fit the curve
                    if logarithmicfit:
                        popt, _ = curve_fit(log_func, xval, yval)
                        a_fit, b_fit = popt
                        y_pred = log_func(xval, *popt)
                        r2 = r2_score(yval, y_pred)  # Compute R²
                        
                        # Generate fitted values
                        x_fit = np.linspace(min(xval), max(xval), 100)
                        y_fit = log_func(x_fit, *popt)
                        logeq = f'y={a_fit:.2f}+{b_fit:.3f}log(x)' if b_fit>=0 else f'y={a_fit:.2f}{b_fit:.3f}log(x)'
                        textstr = '\n'.join((
                                logeq,
                                f'$R^2$ = {r2:.3f}'))
                    else:
                        popt, _ = curve_fit(exp_func, xval, yval, p0=(1, -0.1))  # Initial guesses for parameters
                        a_fit, b_fit = popt
                        y_pred = exp_func(xval, *popt)  # Fitted values for the actual x

                        r2 = r2_score(yval, y_pred)  # Compute R²
                        
                        # Generate fitted values
                        x_fit = np.linspace(min(xval), max(xval), 100)
                        y_fit = exp_func(x_fit, *popt)

                        textstr = '\n'.join((
                                f'y={a_fit:.2f}e^{b_fit:.3f}x',
                                f'$R^2$ = {r2:.3f}'))
                    

                plt.figure()
                plt.plot(x_fit, y_fit, color="r", label=textstr, linestyle='--')
                plt.scatter(xval, yval, marker='o', color='k')
                plt.xlabel(xstat)
                plt.ylabel(ystat)
                if logx:
                    plt.xscale("log")
                if logy:
                    plt.yscale("log")
                if xminlim:
                    plt.xlim(xminlim, xmaxlim)
                if yminlim:
                    plt.ylim(yminlim, ymaxlim)
                
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.legend()
                plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"fitted_{ystat}_{xstat}.png"))

    def taylor_diagram(self):

        util = utilities()
        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        
        nb_members = 100
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        
        df_dic = {"Absolute mean bias":[], "Pearson correlation":[], "RMSE":[], "KGE":[], "Beta":[], "Alpha":[], "NSE":[], "Pairwise correlation":[], "Variance":[], "IQR (75-25%)":[], "IQR (100-0%)":[], "std":[]}

        stds = [] # we have 100 obs std, while the plot takes only one obs std
        corr = []
        rmses = []
        for pixel in range(100):
            ensemble_predictions = members_sim[:,:,pixel]
            ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
            # Calculate the mean prediction for each time step
            mean_prediction = np.mean(ensemble_predictions, axis=1)
            
            ########### Calculate ensemble statistics ###########
            # Calculate the variance for each time step
            ensemble_variance = np.var(ensemble_predictions, axis=1)
            ensemble_variance = np.mean(ensemble_variance)
            df_dic["Variance"].append(ensemble_variance)

            # Calculate ensemble statistics (e.g., diversity, spread, etc.)
            # Spread interquantile range (IQR) between 75th and 25th percentiles
            ensemble_iqr = np.percentile(ensemble_predictions, 75, axis=1) - np.percentile(ensemble_predictions, 25, axis=1)  # IQR
            iqr_mean = np.mean(ensemble_iqr)
            df_dic["IQR (75-25%)"].append(iqr_mean)

            iqr100 = np.percentile(ensemble_predictions, 100, axis=1) - np.percentile(ensemble_predictions, 0, axis=1)
            iqr100_mean = np.mean(iqr100)
            df_dic["IQR (100-0%)"].append(iqr100_mean)

            # Calculate the std for each time step
            ensemble_std = np.std(ensemble_predictions, axis=1)
            ensemble_std = np.mean(ensemble_std)
            df_dic["std"].append(ensemble_std)

            # Calculate diversity by Pairwise correlation
            correlation_matrix = np.corrcoef(ensemble_predictions.T)  # Transpose to get members on rows
            pairwisecorr = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]) # Compute diversity as 1 - average correlation
            df_dic["Pairwise correlation"].append(pairwisecorr)
            
            ########### Calculate accuracy metrics ###########
            # Calculate the correlation between the mean prediction and observation
            correlationobs = np.corrcoef(mean_prediction, obs[:,pixel])[0, 1]
            df_dic["Pearson correlation"].append(correlationobs)

            # Calculate the RMSE between the mean prediction and observation
            rmse = np.sqrt(np.mean((obs[:,pixel] - mean_prediction) ** 2))
            df_dic["RMSE"].append(rmse)

            # Calculate MSE

            # Calculate KGE
            kge = util.calculate_kge(obs[:,pixel], mean_prediction)
            df_dic["KGE"].append(kge)
            
            #### KGE terms analysis ####
            # Compute mean and standard deviation
            mu_o, mu_p = np.mean(obs[:,pixel]), np.mean(mean_prediction)
            sigma_o, sigma_p = np.std(obs[:,pixel]), np.std(mean_prediction)
            
            # Compute bias ratio (β) and variability ratio (γ)
            beta = mu_p / mu_o
            alpha = sigma_p / sigma_o
            
            if np.std(obs[:,pixel]) < 0.1: # if the std is close to zero, exclude pixel kge
                alpha = np.nan
                beta = np.nan
            
            df_dic["Beta"].append(beta)
            df_dic["Alpha"].append(alpha)

            # Calculate NSE
            nse = 1 - (np.sum((obs[:,pixel] - mean_prediction) ** 2) / np.sum((obs[:,pixel] - np.mean(obs[:,pixel])) ** 2))
            if np.std(obs[:,pixel]) < 0.1:
                nse = np.nan
            df_dic["NSE"].append(nse)

            # Calculate mean bias
            bias_mean = np.mean(mean_prediction - obs[:,pixel])
            df_dic["Absolute mean bias"].append(bias_mean)
            
            #x_axis.append(ensemble_variance)
            #y_axis.append((beta-1)**2)
      
        # Example data
        # Reference dataset (e.g., observations)
        ref_std = 1.0  # Standard deviation of the reference dataset
        ref_data = np.random.normal(0, ref_std, 100)  # Generate some random data

        # Model datasets (e.g., predictions from different models)
        model1_std = 0.8  # Standard deviation of model 1
        model1_corr = 0.9  # Correlation of model 1 with reference
        model1_data = np.random.normal(0, model1_std, 100)  # Generate some random data

        model2_std = 1.2  # Standard deviation of model 2
        model2_corr = 0.7  # Correlation of model 2 with reference
        model2_data = np.random.normal(0, model2_std, 100)  # Generate some random data

        # Calculate statistics for the Taylor diagram
        # Standard deviation
        stddev = np.array([ref_std, model1_std, model2_std])

        # Correlation
        corr = np.array([1.0, model1_corr, model2_corr])

        # RMS difference (RMSE)
        rmse = np.array([0.0, np.sqrt(np.mean((ref_data - model1_data)**2)), np.sqrt(np.mean((ref_data - model2_data)**2))])

        sm.taylor_diagram(stddev, rmse, corr, markerColor='r', markerSize=10)

        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "taylor_diag.png"))
    
    def pdf_TL_EU(self):
        def calc_correlation(obs, sim):
            correlation_map = []#np.zeros(obs.shape[1])
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
                correlation_map.append(r)
            return correlation_map

        def map_eligible_to_transfer(target_map, transfer_subset):
            flat_target_map = target_map.flatten()
            flat_transfer_subset = transfer_subset.flatten()
            intersecting_indices = np.where((flat_transfer_subset == 1) & (flat_target_map == 1))[0]
            target_indices = np.where(flat_target_map==1)[0]
            indices = np.where(np.isin(target_indices, intersecting_indices))[0]
            return indices

        def load_obs_sim(target):
            obs_destand_test = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            sim_destand_test = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"sim_destand_{MODEL_NAME}_{target}.npy"))
            obs_destand_test = np.nan_to_num(obs_destand_test)
            sim_destand_test = np.nan_to_num(sim_destand_test)
            obs_destand_test[obs_destand_test < 0.0] = 0.0
            sim_destand_test[sim_destand_test < 0.0] = 0.0
            return obs_destand_test, sim_destand_test

        EU_inpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/spatio-temporal-LSTM/inputs/20yrs_ts/ensemble_400px_eu", "ensemble_mean")
        print("starting the pdf calculation")
        #### Transfer exercise subset #####
        corr_TL_ex = []
        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        nb_members = 100
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        for pixel in range(100):
            ensemble_predictions = members_sim[:,:,pixel]
            ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
            # Calculate the mean prediction for each time step
            mean_prediction = np.mean(ensemble_predictions, axis=1)
            corr = np.corrcoef(mean_prediction, obs[:,pixel])[0, 1]
            corr_TL_ex.append(corr)
        print(len(corr_TL_ex))

        ##### Transfer eligible dataset #####
        total_EU_corr = []
        length = 0
        for target in range(100):
            obs_destand_test, sim_destand_test = load_obs_sim(target)
            length = length + obs_destand_test.shape[1]
            corr_EU = calc_correlation(sim_destand_test, obs_destand_test)
            total_EU_corr.extend(corr_EU)
        print(len(total_EU_corr))

        #### Transfer subset ####
        # TODO check why the original mapping file size is 12938 
        # but the total indices here are 12932
        original_transfersubset_size = 12938
        transfer_subset = np.load(os.path.join(os.path.dirname(INPUTPATH), "transfer_subset.npy"))
        transfer_EU_corr = []
        for target in range(100):
            obs_destand_test, sim_destand_test = load_obs_sim(target)
            target_map = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            indices = map_eligible_to_transfer(target_map, transfer_subset)
            obs_destand_test = obs_destand_test[:, indices]
            sim_destand_test = sim_destand_test[:, indices]
            corr_EU = calc_correlation(sim_destand_test, obs_destand_test)
            transfer_EU_corr.extend(corr_EU)
        print(len(transfer_EU_corr))
        
        corr_TL_ex = np.array(corr_TL_ex)
        corr_TL_ex = corr_TL_ex[~np.isnan(corr_TL_ex)]
        total_EU_corr = np.array(total_EU_corr)
        total_EU_corr = total_EU_corr[~np.isnan(total_EU_corr)]
        transfer_EU_corr = np.array(transfer_EU_corr)
        transfer_EU_corr = transfer_EU_corr[~np.isnan(transfer_EU_corr)]
        # Ensure both lists have the same min and max by clipping (if needed)
        min_val = -1
        max_val = 1

        # Create kernel density estimates
        kde1 = gaussian_kde(corr_TL_ex)
        kde2 = gaussian_kde(total_EU_corr)
        kde3 = gaussian_kde(transfer_EU_corr)

        # Create a range of values for plotting
        x = np.linspace(min_val, max_val, length)

        # Plot the PDFs
        plt.figure(figsize=(10, 6))
        plt.plot(x, kde2(x), label=f'Transfer eligible dataset (n={length})', color='red')
        plt.plot(x, kde3(x), label=f'Transfer subset (n={original_transfersubset_size})', color='green')
        plt.plot(x, kde1(x), label=f'Transfer exerciese subset (n={len(corr_TL_ex)})', color='blue')

        # Add plot elements
        plt.xlabel('Pearson correlation')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Show the plot
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "corr_pdf.png"))

    def corr_components(self):
        def pearson_nominator(x, y):
            """Calculate covariance between two arrays (numerator of Pearson correlation)"""
            return np.sum((x - np.mean(x)) * (y - np.mean(y)))

        def pearson_denominator(y, y_hat):
            y_dev = y - np.mean(y)                # Deviations from mean (y)
            y_hat_dev = y_hat - np.mean(y_hat)    # Deviations from mean (ŷ)
            return np.sqrt(np.sum(y_dev**2)) * np.sqrt(np.sum(y_hat_dev**2))

        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        obs[obs < 0.0] = 0.0
        #ens_mean = np.load(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy"))
        
        nb_members = 100
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        members_sim[members_sim < 0.0] = 0.0
        covariances = []
        correlations = []
        for pixel in range(100):
            ensemble_predictions = members_sim[:,:,pixel]
            ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
            # Calculate the mean prediction for each time step
            mean_prediction = np.mean(ensemble_predictions, axis=1)
            cov = pearson_denominator(obs[:,pixel], mean_prediction)
            covariances.append(cov)
            corr = np.corrcoef(mean_prediction, obs[:,pixel])[0, 1]
            correlations.append(corr)
            
        print(min(covariances), max(covariances))
        plt.figure()
        plt.scatter(covariances, correlations, marker='o', color='k')
        plt.xlabel(f'std(obs) * std(sim)')
        plt.ylabel(f'Pearson correlation')
        #plt.xlim(min(min(y_axis), min(x_axis)), max(max(y_axis), max(x_axis)))
        #plt.ylim(min(min(y_axis), min(x_axis)), max(max(y_axis), max(x_axis)))
        plt.xscale("log")
        #plt.yscale("log")
        #plt.xscale("symlog")
        #plt.xlim(0,1)
        plt.ylim(-1, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        #print(f"plotting: {ytitle}_{xtitle}")
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"Pearson correlation_stdobssim.png"))

    def timeseries_exclude_percentiles(self):
        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        obs[obs < 0.0] = 0.0
        
        nb_members = 100
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        members_sim[members_sim < 0.0] = 0.0
        xaxis = []
        yaxis = []
        for pixel in range(100):
            ensemble_predictions = members_sim[:,:,pixel]
            timeseries_data = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
            
            # Compute the 25th and 75th percentiles across all timeseries
            q25 = np.percentile(timeseries_data, 10, axis=1)
            q75 = np.percentile(timeseries_data, 90, axis=1)
            #print(q25)
            #print(q75)

            # Find timeseries within the interquartile range
            valid_mask = [np.all((timeseries_data[:,m] >= q25) & (timeseries_data[:,m] <= q75), axis=0) for m in range(nb_members)]
            valid_mask = np.array(valid_mask)
            print(len(valid_mask[valid_mask==True]))
            
            # Filter timeseries
            filtered_timeseries = timeseries_data[:, valid_mask]

            # Calculate the mean prediction for each time step after percentiles
            mean_prediction = np.mean(filtered_timeseries, axis=1)
        
            ensemble_variance = np.var(filtered_timeseries, axis=1)
            ensemble_variance = np.mean(ensemble_variance)

            corr = np.corrcoef(mean_prediction, obs[:,pixel])[0, 1]

            xaxis.append(ensemble_variance)
            yaxis.append(corr)
            # Plot the remaining timeseries
            dates = pd.date_range(start='2017-01-01', end='2020-12-31', freq='D')
            dates = dates[(dates.month != 2) | (dates.day != 29)]
            fig, ax = plt.subplots(figsize=(16, 10))
            
            for m in range(len(filtered_timeseries[1])):
                ts = filtered_timeseries[:, m]
                ax.plot(dates, ts, color="gray")

            ax.plot(dates, obs[:,pixel], "k-", label="Original simulations", linewidth=4.0)
            ax.plot(dates, mean_prediction, "k--", label="Ensemble mean", linewidth=4.0)
            ax.scatter([], [], color="k", label=f"r: {corr:.2f}, Ensemble variance: {math.ceil(ensemble_variance)}, nb b/w 10-90%: {int(len(filtered_timeseries[1]))}")


            ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

            plt.xticks(rotation=45)
            plt.ylabel('Water table depth (m)')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3)
            plt.grid()
            plt.savefig(os.path.join(OUTPUTPATH, "transfer_timeseries_statvsacc", f"ensemble_timeseries_{pixel}_10-90percentiles.png"))
        
    def fit_stat_vs_acc_excl_percentiles(self):
        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        obs[obs < 0.0] = 0.0
        
        nb_members = 100
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        members_sim[members_sim < 0.0] = 0.0
        xaxis = []
        yaxis = []
        for pixel in range(100):
            ensemble_predictions = members_sim[:,:,pixel]
            timeseries_data = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
            
            # Compute the 25th and 75th percentiles across all timeseries
            q25 = np.percentile(timeseries_data, 10, axis=1)
            q75 = np.percentile(timeseries_data, 90, axis=1)
            
            # Find timeseries within the interquartile range
            valid_mask = [np.all((timeseries_data[:,m] >= q25) & (timeseries_data[:,m] <= q75), axis=0) for m in range(nb_members)]
            valid_mask = np.array(valid_mask)
            print(len(valid_mask[valid_mask==True]))
            
            # Filter timeseries
            filtered_timeseries = timeseries_data[:, valid_mask]

            # Calculate the mean prediction for each time step after percentiles
            mean_prediction = np.mean(filtered_timeseries, axis=1)
        
            ensemble_variance = np.var(filtered_timeseries, axis=1)
            ensemble_variance = np.mean(ensemble_variance)

            corr = np.corrcoef(mean_prediction, obs[:,pixel])[0, 1]

            xaxis.append(ensemble_variance)
            yaxis.append(corr)
            
        ystat = "Pearson correlation"
        xstat = "Variance"
        yval = np.array(yaxis)
        xval = np.array(xaxis)
        fittype = "exponential"
        logarithmicfit = True

        if ystat=="Pearson correlation":
            xval = xval[~np.isnan(yval)]
            yval = yval[~np.isnan(yval)]
            xval = xval[yval>=0.0]
            yval = yval[yval>=0.0]
        
        if ystat=="KGE" or ystat=="NSE":
            xval = xval[~np.isnan(yval)]
            yval = yval[~np.isnan(yval)]
            xval = xval[yval>=-1]
            yval = yval[yval>=-1]

        if ystat=="Beta" or ystat=="Alpha" or ystat=="(Alpha-1)^2" or ystat=="(Beta-1)^2":
            xval = xval[~np.isnan(yval)]
            yval = yval[~np.isnan(yval)]
        
        if ystat=="Absolute mean bias":
            xval = xval[yval>=0.01]
            yval = yval[yval>=0.01]
        
        if xstat=="Pairwise correlation" and ystat=="Pearson correlation":
            yval = yval[xval>=0.2]
            xval = xval[xval>=0.2]
        #if xstat=="IQR (75-25%)" and ystat=="Pearson correlation":
        #    yval = yval[xval<=10]
        #    xval = xval[xval<=10]
        if xstat=="Pairwise correlation" and ystat=="KGE":
            yval = yval[xval>=0.2]
            xval = xval[xval>=0.2]
        if xstat=="Pairwise correlation" and ystat=="NSE":
            yval = yval[xval>=0.6]
            xval = xval[xval>=0.6]
                
        
        logx, logy, xminlim, xmaxlim, yminlim, ymaxlim = None, None, None, None, None, None
        if xstat=="std" or xstat=="Variance" or xstat=="IQR (75-25%)" or xstat=="IQR (100-0%)":
            logx = True
        if ystat=="RMSE" or ystat=="Absolute mean bias" or ystat=="Alpha" or ystat=="Beta" or ystat=="(Alpha-1)^2" or ystat=="(Beta-1)^2":
            logy = True
        if xstat=="Pairwise correlation":
            xminlim = 0
            xmaxlim = 1
        if ystat=="Pearson correlation" or ystat=="NSE":
            yminlim = 0
            ymaxlim = 1
        if ystat=="KGE":
            yminlim = -1
            ymaxlim = 1
        

        if fittype=="exponential":
            # Define a function to fit (e.g., exponential decay or polynomial)
            def log_func(x, a, b):
                return a + b * np.log(x)
            def exp_func(x, a, b):
                return a* np.exp(b * x)  # Exponential model without offset

            # Fit the curve
            if logarithmicfit:
                popt, _ = curve_fit(log_func, xval, yval)
                a_fit, b_fit = popt
                y_pred = log_func(xval, *popt)
                r2 = r2_score(yval, y_pred)  # Compute R²
                
                # Generate fitted values
                x_fit = np.linspace(min(xval), max(xval), 100)
                y_fit = log_func(x_fit, *popt)
                logeq = f'y={a_fit:.2f}+{b_fit:.3f}log(x)' if b_fit>=0 else f'y={a_fit:.2f}{b_fit:.3f}log(x)'
                textstr = '\n'.join((
                        logeq,
                        f'$R^2$ = {r2:.3f}'))
            else:
                popt, _ = curve_fit(exp_func, xval, yval, p0=(1, -0.1))  # Initial guesses for parameters
                a_fit, b_fit = popt
                y_pred = exp_func(xval, *popt)  # Fitted values for the actual x

                r2 = r2_score(yval, y_pred)  # Compute R²
                
                # Generate fitted values
                x_fit = np.linspace(min(xval), max(xval), 100)
                y_fit = exp_func(x_fit, *popt)

                textstr = '\n'.join((
                        f'y={a_fit:.2f}e^{b_fit:.3f}x',
                        f'$R^2$ = {r2:.3f}'))
            

        plt.figure()
        plt.plot(x_fit, y_fit, color="r", label=textstr, linestyle='--')
        plt.scatter(xval, yval, marker='o', color='k')
        plt.xlabel(xstat)
        plt.ylabel(ystat)
        if logx:
            plt.xscale("log")
        if logy:
            plt.yscale("log")
        if xminlim:
            plt.xlim(xminlim, xmaxlim)
        if yminlim:
            plt.ylim(yminlim, ymaxlim)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"fitted_{ystat}_{xstat}_excl%.png"))

    def confirm_choices_mapping(self):
        transfer_subset = np.load(os.path.join(os.path.dirname(INPUTPATH), "transfer_subset.npy"))
        transfer_ex_subset = np.load(os.path.join(INPUTPATH, "choices.npy"))
        ind = np.where(transfer_ex_subset==1)
        
        ##### load exercise #####
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        sim_destand_test = np.load(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy"))
        obs_destand_test = np.nan_to_num(obs_destand_test)
        sim_destand_test = np.nan_to_num(sim_destand_test)
        obs_destand_test[obs_destand_test < 0.0] = 0.0
        sim_destand_test[sim_destand_test < 0.0] = 0.0
        mapping = np.load(os.path.join(INPUTPATH, "choices.npy"))
        obs = np.zeros((obs_destand_test.shape[0], mapping.shape[0], mapping.shape[1]))
        obs[obs==0] = np.nan
        sim = np.zeros((sim_destand_test.shape[0], mapping.shape[0], mapping.shape[1]))
        sim[sim==0] = np.nan
        mapping_indexes = np.where(mapping==1)
        for i in range(len(mapping_indexes[0])):
            obs[:,mapping_indexes[0][i],mapping_indexes[1][i]] = obs_destand_test[:,i]
            sim[:,mapping_indexes[0][i],mapping_indexes[1][i]] = sim_destand_test[:,i]
        corr_ex = np.corrcoef(sim[:,100,119], obs[:,100,119])[0, 1]

        ##### load EU #####
        EU_inpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/spatio-temporal-LSTM/inputs/20yrs_ts/ensemble_400px_eu", "ensemble_mean")
        EU_outpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/spatio-temporal-LSTM/outputs/20yrs_ts/ensemble_400px_eu", "ensemble_mean")
        target_mapping = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_0", f"mappingindices_0.npy"))
        indexes = np.where(target_mapping==1)
        
        xind = np.where(indexes[0]==100)
        yind = np.where(indexes[1]==119)
        pixel = 48
        
        obs_destand_EU = np.load(os.path.join(os.path.dirname(EU_outpath), "400px_member_9", f"obs_destand_{MODEL_NAME}_0.npy"))
        print(obs_destand_EU.shape)
        members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_0.npy")) for m in range(100)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        ensemble_predictions = members_sim[:,:,pixel]
        ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
        # Calculate the mean prediction for each time step
        mean_prediction = np.mean(ensemble_predictions, axis=1)
        corr_EU = np.corrcoef(mean_prediction, obs_destand_EU[:,pixel])[0, 1]

        print(corr_ex)
        print(corr_EU)

    def concat_EU_outputs(self):
        EU_inpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/spatio-temporal-LSTM/inputs/20yrs_ts/ensemble_400px_eu", "ensemble_mean")
        EU_outpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/spatio-temporal-LSTM/outputs/20yrs_ts/ensemble_400px_eu", "ensemble_mean")
        
        for target in range(100):
            obs_destand_EU = np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{target}", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
            mean_prediction = np.mean(members_sim, axis=0)
            print(obs_destand_EU.shape)
            print(mean_prediction.shape)
            np.save(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"obs_destand_{MODEL_NAME}_{target}.npy"), obs_destand_EU)
            np.save(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"sim_destand_{MODEL_NAME}_{target}.npy"), mean_prediction)
            print(f"saved {target} target pixels")
    