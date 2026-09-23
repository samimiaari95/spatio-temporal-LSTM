from fileinput import filename
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import PredictionErrorDisplay, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_predict
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
from LSTM_model.utils.plot_functions import plotting_helper
from LSTM_model.utils.utils import utilities
from LSTM_model.model.config import *


class _CrossValidatedCurveFitRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, fittype="linear", logarithmicfit=False):
        self.fittype = fittype
        self.logarithmicfit = logarithmicfit

    @staticmethod
    def _linear_law(x, a, b):
        return a + b * x

    @staticmethod
    def _powerlaw_func(x, a, b):
        return a * np.power(x, b)

    @staticmethod
    def _log_func(x, a, b):
        return a + b * np.log(x)

    @staticmethod
    def _exp_func(x, a, b):
        return a * np.exp(b * x)

    def fit(self, X, y):
        x = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)

        if self.fittype == "powerlaw":
            params, _ = curve_fit(self._linear_law, np.log(x), np.log(y), maxfev=10000)
            self.a_fit_ = np.exp(params[0])
            self.b_fit_ = params[1]
        elif self.fittype == "linear":
            params, _ = curve_fit(self._linear_law, x, y, maxfev=10000)
            self.a_fit_, self.b_fit_ = params
        elif self.logarithmicfit:
            params, _ = curve_fit(self._log_func, x, y, maxfev=10000)
            self.a_fit_, self.b_fit_ = params
        else:
            params, _ = curve_fit(self._exp_func, x, y, p0=(1, -0.1), maxfev=10000)
            self.a_fit_, self.b_fit_ = params

        return self

    def predict(self, X):
        x = np.asarray(X).reshape(-1)

        if self.fittype == "powerlaw":
            return self._powerlaw_func(x, self.a_fit_, self.b_fit_)
        if self.fittype == "linear":
            return self._linear_law(x, self.a_fit_, self.b_fit_)
        if self.logarithmicfit:
            return self._log_func(x, self.a_fit_, self.b_fit_)
        return self._exp_func(x, self.a_fit_, self.b_fit_)

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
        for target in range(100):
            obs_destand_test = np.load(os.path.join(os.path.dirname(OUTPUTPATH), "400px_member_0", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            ens_mean = np.zeros((obs_destand_test.shape[0], obs_destand_test.shape[1], 100))
            for m in range(100):
                sim_destand_test = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy"))
                ens_mean[:,:,m] = sim_destand_test[:,:]
            sim = np.mean(ens_mean, axis=2)
            print(obs_destand_test.shape)
            print(sim.shape)
            np.save(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"obs_destand_{MODEL_NAME}.npy"), obs_destand_test)
            np.save(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_destand_{MODEL_NAME}.npy"), sim)

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

    def plot_ensemble_statsvsacc_timeseries(self, target, i, members_sim, mean_prediction, obs, kge, stat):
        nb_members = 100

        dates = pd.date_range(start='2017-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))
        
        print(f"plotting timeseries {i} with KGE: {kge} and EV: {stat}")
        for m in range(nb_members):
            sim = members_sim[m,:,i]
            ax.plot(dates, sim, color="gray")

        ax.plot(dates, obs, "k-", label="Observations", linewidth=4.0)
        ax.plot(dates, mean_prediction, "k--", label="Ensemble mean", linewidth=4.0)
        ax.scatter([], [], color="k", label=f"KGE: {kge}, EV: {stat}")

        ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3)
        plt.grid()
        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(OUTPUTPATH)), "validation_400_withcriteria_43226", "timeseries_EV_KGElessthan02", f"ensemble_timeseries_{target}_{i}.png"))

    def split_transfersubset_into_fitandvalidate_subsets(self):
        transfer_subset = np.load(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", "transfer_subset.npy"))
        utils = utilities()
        outpath = os.path.join(os.path.dirname(INPUTPATH), "target_pixels")
        utils.make_dir(outpath)

        transfer_indices = np.where(transfer_subset == 1)
        transfer_pixels = np.column_stack(transfer_indices)
        num_transfer_pixels = transfer_pixels.shape[0]

        rng = np.random.default_rng(0)
        shuffled_indices = rng.permutation(num_transfer_pixels)
        num_fit_pixels = int(np.floor(0.8 * num_transfer_pixels))

        fit_pixels = transfer_pixels[shuffled_indices[:num_fit_pixels]]
        validate_pixels = transfer_pixels[shuffled_indices[num_fit_pixels:]]

        fit_subset = np.full(transfer_subset.shape, np.nan)
        validate_subset = np.full(transfer_subset.shape, np.nan)

        fit_subset[fit_pixels[:, 0], fit_pixels[:, 1]] = 1
        validate_subset[validate_pixels[:, 0], validate_pixels[:, 1]] = 1

        np.save(os.path.join(outpath, "spreadskill_fit_subset.npy"), fit_subset)
        np.save(os.path.join(outpath, "spreadskill_heldout_subset.npy"), validate_subset)

        print(f"transfer pixels: {num_transfer_pixels}")
        print(f"fit pixels: {len(fit_pixels)}")
        print(f"validation pixels: {len(validate_pixels)}")
        print(fit_subset.shape)
        print(validate_subset.shape)
        print(np.sum(fit_subset==1))
        print(np.sum(validate_subset==1))

    def ensemble_statvsacc(self):
        utils = utilities()
        EU_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        EU_outpath = os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean")
        transfer_subset = np.load(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", "spreadskill_fit_subset.npy"))

        #transfer_subset = np.load(os.path.join(os.path.dirname(os.path.dirname(EU_inpath)), "transfer_subset.npy"))
        df_dic = {"mean_membersobs_RMSE":[], "mean_membersobs_correlation":[], "mean_membersobs_KGE":[], "KGE_nobias":[], "stdsim":[], "stdobs":[], "meansim":[], "meanobs":[], "Absolute mean bias":[], "Pearson correlation":[], "RMSE":[], "KGE":[], "KGE'":[], "Beta":[], "Alpha":[], "NSE":[], "Pairwise correlation":[], "Ensemble variance":[], "IQR (75-25%)":[], "std":[], "cv":[], "mad":[],  "(Alpha-1)^2":[], "(Beta-1)^2":[], "(r-1)^2":[]}
        for target in range(100):
            target_map = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            transfer_indices, ind2d = utils.intersect_subsets(target_map, transfer_subset)
            obs = np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_1", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            obs = obs[:, transfer_indices] # keep only transfer pixels
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
            # NOTE we do this acc and statistics relationship only for transfer pixels because
            # 1- this case we plot 3226 points, for all 43226 pixels we will have a black plot with a red fitting line
            # 2- it doesn't make a difference to take only 3226 or 43226 or even a 100 representative pixels, we should get the same fitting line, it should be representative
            members_sim = members_sim[:,:,transfer_indices] # keep only transfer pixels
            obs[obs < 0.0] = 0.0 ############## important to set negatives to zero ##############
            members_sim[members_sim < 0.0] = 0.0 ############## important to set negatives to zero ##############
            print(f"number of pixels: {members_sim.shape[2]} in target: {target}")
            for pixel in range(members_sim.shape[2]):
                ensemble_predictions = members_sim[:,:,pixel]
                ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
                # Calculate the mean prediction for each time step
                mean_prediction = np.mean(ensemble_predictions, axis=1)
                if np.std(obs[:,pixel])==0.0 or np.std(mean_prediction)==0.0 or np.isnan(obs[:,pixel]).all() or np.isnan(mean_prediction).all():
                    continue
                
                ########### Calculate ensemble statistics ###########
                # Calculate the variance for each time step
                ensemble_variance = np.var(ensemble_predictions, axis=1)
                ensemble_variance = np.mean(ensemble_variance)
                df_dic["Ensemble variance"].append(ensemble_variance)

                # Calculate ensemble statistics (e.g., diversity, spread, etc.)
                # Spread interquantile range (IQR) between 75th and 25th percentiles
                ensemble_iqr = np.percentile(ensemble_predictions, 75, axis=1) - np.percentile(ensemble_predictions, 25, axis=1)  # IQR
                iqr_mean = np.mean(ensemble_iqr)
                df_dic["IQR (75-25%)"].append(iqr_mean)

                # calculate MAD
                mad = utils.calculate_ensemble_mad(ensemble_predictions)
                df_dic["mad"].append(mad)

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
                # Calculate mean bias
                bias_mean = np.mean(mean_prediction - obs[:,pixel])
                df_dic["Absolute mean bias"].append(bias_mean)

                ### remove bias from mean prediction ###
                # mean_prediction = mean_prediction - bias_mean

                # Calculate the correlation between the mean prediction and observation
                correlationobs = np.corrcoef(mean_prediction, obs[:,pixel])[0, 1]
                df_dic["Pearson correlation"].append(correlationobs)

                # Calculate the RMSE between the mean prediction and observation
                rmse = np.sqrt(np.mean((obs[:,pixel] - mean_prediction) ** 2))
                df_dic["RMSE"].append(rmse)

                # Calculate KGE
                kge = utils.calculate_kge(obs[:,pixel], mean_prediction)
                df_dic["KGE"].append(kge)
                
                # members correlation
                members_correlation = [np.corrcoef(ensemble_predictions[:,m], obs[:,pixel])[0, 1] for m in range(ensemble_predictions.shape[1])]
                members_correlation_mean = np.mean(members_correlation)
                df_dic["mean_membersobs_correlation"].append(members_correlation_mean)
                if members_correlation_mean==correlationobs:
                    raise ValueError("members mean correlation is the same as correlationobs, check your code")

                # members RMSE
                members_rmse = np.mean(np.sqrt(np.mean((ensemble_predictions - obs[:,pixel][:, None]) ** 2, axis=0)))
                df_dic["mean_membersobs_RMSE"].append(members_rmse)
                if members_rmse==rmse:
                    raise ValueError("members mean RMSE is the same as RMSE, check your code")

                # members KGE
                members_kge = [utils.calculate_kge(ensemble_predictions[:,m], obs[:,pixel]) for m in range(ensemble_predictions.shape[1])]
                members_kge_mean = np.mean(members_kge)
                df_dic["mean_membersobs_KGE"].append(members_kge_mean)
                if members_kge_mean==kge:
                    raise ValueError("members mean KGE is the same as KGE, check your code")
                
                ### KGE terms analysis ####
                # Compute mean and standard deviation
                mu_o, mu_p = np.mean(obs[:,pixel]), np.mean(mean_prediction)
                sigma_o, sigma_p = np.std(obs[:,pixel]), np.std(mean_prediction)

                ### KGE' ###
                kgeprime = utils.kge_prime(obs[:,pixel], mean_prediction)
                df_dic["KGE'"].append(kgeprime)
                
                # Compute bias ratio (β) and variability ratio (γ)
                beta = mu_p / mu_o
                alpha = sigma_p / sigma_o
                kge_nobias = 1 - np.sqrt((correlationobs - 1)**2 + (alpha - 1)**2)

                # Store statistics and accuracy metrics
                df_dic["KGE_nobias"].append(kge_nobias)
                df_dic["stdsim"].append(np.std(mean_prediction))
                df_dic["stdobs"].append(np.std(obs[:,pixel]))
                df_dic["meansim"].append(np.mean(mean_prediction))
                df_dic["meanobs"].append(np.mean(obs[:,pixel]))
                                
                df_dic["Beta"].append(beta)
                df_dic["Alpha"].append(alpha)
                df_dic["(Alpha-1)^2"].append((alpha-1)**2)
                df_dic["(Beta-1)^2"].append((beta-1)**2)
                df_dic["(r-1)^2"].append((correlationobs-1)**2)


                # Calculate NSE
                nse = 1 - (np.sum((obs[:,pixel] - mean_prediction) ** 2) / np.sum((obs[:,pixel] - np.mean(obs[:,pixel])) ** 2))
                df_dic["NSE"].append(nse)
                
                # self.plot_ensemble_statsvsacc_timeseries(target, pixel, members_sim, mean_prediction, obs[:,pixel], f'{kge:.2f}', f'{math.ceil(ensemble_variance)}')

        ####### save to csv ########
        df = pd.DataFrame(df_dic)
        df.to_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics.csv"), index=False)

    def kge_investigation(self):
        df = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics.csv"))
        df = df.dropna()

        x_vals = ["Pearson correlation", "Alpha", "Beta", "stdsim", "stdobs", "meansim", "meanobs"]
        xlabels = {
            "Pearson correlation": "Pearson correlation",
            "Alpha": r"$\alpha=\frac{\sigma_{S}}{\sigma_{O}}$",
            "Beta": r"$\beta=\frac{\bar{S}}{\bar{O}}$",
            "stdsim": r"$\sigma_{S}$",
            "stdobs": r"$\sigma_{O}$",
            "meansim": r"$\bar{S}$",
            "meanobs": r"$\bar{O}$"
        }
        for x in x_vals:
            plt.figure()
            plt.scatter(df[x].values,df["KGE"].values, marker='.', color='k')
            plt.xlabel(xlabels[x])
            
            plt.ylabel(f"KGE")
            #plt.xlim(min(min(y_axis), min(x_axis)), max(max(y_axis), max(x_axis)))
            #plt.ylim(min(min(y_axis), min(x_axis)), max(max(y_axis), max(x_axis)))
            plt.xscale("log")
            #plt.yscale("log")
            plt.yscale("symlog")
            #plt.xlim(-1,1)
            #plt.ylim(-1, 1)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            print(f"plotting: KGE vs {x}")
            plt.savefig(os.path.join(OUTPUTPATH, "statistics", "kgeinv", f"KGE_{x}.png"))
        
    def ensemble_crpsvsstats(self):
        # NOTE we do this acc and statistics relationship only for transfer pixels for consistency with other metrics
        # and also for the same reasons as the other metrics

        # crpstrain = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"crps_99members_onlytraining.npy"))
        crpstransfer = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"crps_100members_onlytransfer.npy"))

        transfersims = [np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_100members_ts_transferpixels_{target}.npy")) for target in range(100)]
        transfersims = np.concatenate(transfersims, axis=2) #(100 members, timeseries, pixels)

        # trainsims = [np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_99members_ts_trainpixels_{target}.npy")) for target in range(100)]
        # trainsims = np.concatenate(trainsims, axis=2) #(99 members, timeseries, pixels)

        # Calculate diversity by Pairwise correlation
        #correlation_matrix = np.corrcoef(ensemble_predictions.T)  # Transpose to get members on rows
        #pairwisecorr = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]) # Compute diversity as 1 - average correlation
        #df_dic["Pairwise correlation"].append(pairwisecorr)

        # trainensemblevariance = np.full((crpstrain.shape[0]), np.nan)
        # trainiqr = np.full((crpstrain.shape[0]), np.nan)
        transferensemblevariance = np.full((crpstransfer.shape[0]), np.nan)
        transferiqr = np.full((crpstransfer.shape[0]), np.nan)

        # Ntrain, Ttrain, Ptrain = trainsims.shape  # Extract dimensions
        # for t in range(Ttrain):
        #     print(f"train t:{t}")
        #     for p in range(Ptrain):
        #         trainensemblevariance[t * Ptrain + p] = np.var(trainsims[:, t, p], axis=0)
        #         trainiqr[t * Ptrain + p] = np.percentile(trainsims[:, t, p], 75, axis=0) - np.percentile(trainsims[:, t, p], 25, axis=0)  # IQR

        Ntransfer, Ttransfer, Ptransfer = transfersims.shape
        data_2d = crpstransfer.reshape(Ttransfer, Ptransfer)
        crps_meants_px = np.mean(data_2d, axis=0)
        transferpairwisecorr = np.full((Ptransfer), np.nan)
        for p in range(Ptransfer):
            sims = transfersims[:,:,p]
            sims = np.moveaxis(sims, 0, -1)   # (timeseries, members)
            # Calculate Pairwise correlation
            correlation_matrix = np.corrcoef(sims.T)  # Transpose to get members on rows
            transferpairwisecorr[p] = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]) # Compute diversity as 1 - average correlation

        for t in range(Ttransfer):
            print(f"transfer t:{t}")
            for p in range(Ptransfer):
                transferensemblevariance[t * Ptransfer + p] = np.var(transfersims[:, t, p], axis=0)
                transferiqr[t * Ptransfer + p] = np.percentile(transfersims[:, t, p], 75, axis=0) - np.percentile(transfersims[:, t, p], 25, axis=0)  # IQR
                # Calculate Pairwise correlation
                #correlation_matrix = np.corrcoef(ensemble_predictions.T)  # Transpose to get members on rows
                #pairwisecorr = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]) # Compute diversity as 1 - average correlation
        
        dirpath = os.path.join(OUTPUTPATH, "statistics")
        # ensemblevariance = np.concatenate((trainensemblevariance, transferensemblevariance), axis=0)
        ensemblevariance = transferensemblevariance
        # iqr = np.concatenate((trainiqr, transferiqr), axis=0)
        iqr = transferiqr
        # crps = np.concatenate((crpstrain, crpstransfer), axis=0)
        crps = crpstransfer
        print(f"crps shape: {crps.shape}")
        print(f"ensemble variance shape: {ensemblevariance.shape}")
        print(f"iqr shape: {iqr.shape}")
        np.save(os.path.join(dirpath, "ensemblevariance_crps_transfer.npy"), ensemblevariance)
        np.save(os.path.join(dirpath, "iqr_crps_transfer.npy"), iqr)
        np.save(os.path.join(dirpath, "crps_transfer.npy"), crps)
        np.save(os.path.join(dirpath, "crps_meants_px.npy"), crps_meants_px)
        np.save(os.path.join(dirpath, "pairwisecorr_crps_transfer.npy"), transferpairwisecorr)

    def ensemble_crpsvsstats_fitting(self, cv_folds=5):
        dirpath = os.path.join(OUTPUTPATH, "statistics")
        stats = ["Ensemble variance", "IQR (75-25%)"]#, "Pairwise correlation"]
        yval_base = np.load(os.path.join(dirpath, "crps_transfer.npy"))

        ncols = 2
        nrows = 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 2.84))
        axes = np.array(axes).flatten()
        labels = ['(A)', '(B)']

        for i, stat in enumerate(stats):
            if stat=="Pairwise correlation":
                filename = "pairwisecorr_crps_transfer.npy"
                yval = np.load(os.path.join(dirpath, "crps_meants_px.npy"))
                xlabel = "Pairwise correlation"
            elif stat=="IQR (75-25%)":
                filename = "iqr_crps_transfer.npy"
                xlabel = r"$\overline{IQR}$"
            elif stat=="Ensemble variance":
                filename = "ensemblevariance_crps_transfer.npy"
                xlabel = r"$\overline{EV}$"

            xval = np.load(os.path.join(dirpath, filename))
            if stat=="Pairwise correlation":
                yval = yval[xval>0.01]
                xval = xval[xval>0.01]
            else:
                yval = yval_base
            
            print(f"fitting {stat} with CRPS")
            mask = ~np.isnan(xval) & ~np.isnan(yval)
            xval = xval[mask]
            yval = yval[mask]

            positive_mask = (xval > 0) & (yval > 0)
            xval = xval[positive_mask]
            yval = yval[positive_mask]

            if len(xval) == 0:
                continue

            model = _CrossValidatedCurveFitRegressor(fittype="powerlaw")
            model.fit(xval.reshape(-1, 1), yval)

            x_fit = np.linspace(min(xval), max(xval), 100)
            y_fit = model.predict(x_fit.reshape(-1, 1))

            fit_params = (model.a_fit_, model.b_fit_)
            legend_text = f'$y={fit_params[0]:.2f}x^{{{fit_params[1]:.2f}}}$'

            r2_cv = np.nan
            n_splits = min(cv_folds, len(xval))
            if n_splits >= 2:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_predictions = cross_val_predict(
                    _CrossValidatedCurveFitRegressor(fittype="powerlaw"),
                    xval.reshape(-1, 1),
                    yval,
                    cv=cv,
                )
                positive_cv = cv_predictions > 0
                if np.any(positive_cv):
                    r2_cv = r2_score(
                        np.log(yval[positive_cv]),
                        np.log(cv_predictions[positive_cv]),
                    )

            legend_text = '\n'.join((
                legend_text,
                f'$R^2_{{CV}}$ = {r2_cv:.3f}' if np.isfinite(r2_cv) else '$R^2_{CV}$ = n/a'))
            #plt.figure()
            axes[i].scatter(xval, yval, color='k', s=5)
            axes[i].plot(x_fit, y_fit, 'r--', label=legend_text)
            # ---- keep only left y-label ----
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel("CRPS (m)")
            if i != 0:
                axes[i].set_ylabel("")
            axes[i].set_xscale("log")
            axes[i].set_yscale("log")
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].legend(fontsize=12, frameon=True)
            # ---- subplot labels ----
            axes[i].text(
                0.02,
                1.11,
                labels[i],
                transform=axes[i].transAxes,
                va='top',
                fontsize=12,
                # zorder=20,
                # bbox=dict(facecolor='white', edgecolor='none', alpha=0.85)
            )
        plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"figure11.eps"), dpi=300)
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"figure11.pdf"), dpi=300)
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"figure11.png"), dpi=300)

    def ensemble_statvsacc_fitting(self):
        ####### old version --- replaced with create_group_figure ##########
        utils = utilities()
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics_mad.csv"))
        xstats = {"std":"exp", "Ensemble variance":"exp", "Pairwise correlation":"lin", "IQR (75-25%)":"exp", "cv":"exp", "mad":"exp"}
        ystats = {"mean_membersobs_correlation": "lin", "mean_membersobs_RMSE": "exp", "mean_membersobs_KGE": "lin", "RMSE":"exp", "Pearson correlation":"lin", "KGE":"lin", "KGE_nobias":"exp", "Absolute mean bias":"exp", "NSE":"lin", "Beta":"exp", "Alpha":"exp", "(Alpha-1)^2":"exp", "(Beta-1)^2":"exp", "(r-1)^2":"exp"}
        plotmapping = {
            "Ensemble variance":r"$\overline{EV}$",
            "IQR (75-25%)":r"$\overline{IQR}$",
            "Beta":r"$\beta$",
            "Alpha":r"$\alpha$",
            "(Alpha-1)^2":r"$(\alpha-1)^2$",
            "(Beta-1)^2":r"$(\beta-1)^2$",
        }
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

                if ystat=="Pearson correlation" or ystat=="mean_membersobs_correlation":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                    #xval = xval[yval>=0.0]
                    #yval = yval[yval>=0.0]
                
                if ystat=="KGE" or ystat=="NSE" or ystat=="KGE_nobias" or ystat=="mean_membersobs_KGE":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                    xval = xval[yval>=0.2]
                    yval = yval[yval>=0.2]

                if ystat=="Beta" or ystat=="Alpha" or ystat=="(Alpha-1)^2" or ystat=="(Beta-1)^2" or ystat=="(r-1)^2":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                
                if ystat=="Absolute mean bias":
                    xval = xval[yval>=0.01]
                    yval = yval[yval>=0.01]
                
                if xstat=="Pairwise correlation":
                    yval = yval[xval>=0.0]
                    xval = xval[xval>=0.0]
                
                logx, logy, xminlim, xmaxlim, yminlim, ymaxlim = None, None, None, None, None, None
                if xstat=="std" or xstat=="Ensemble variance" or xstat=="IQR (75-25%)" or xstat=="cv" or xstat=="mad":
                    logx = True
                if ystat=="RMSE" or ystat=="Absolute mean bias" or ystat=="Alpha" or ystat=="Beta" or ystat=="(Alpha-1)^2" or ystat=="(Beta-1)^2" or ystat=="(r-1)^2" or ystat=="mean_membersobs_RMSE":
                    logy = True
                if xstat=="Pairwise correlation":
                    xminlim = 0
                    xmaxlim = 1
                if ystat=="Pearson correlation":
                    yminlim = -1
                    ymaxlim = 1
                if ystat=="KGE" or ystat=="NSE":
                    yminlim = 0.2
                    ymaxlim = 1
                
                if xstat=="Pairwise correlation" and ystat=="Pearson correlation":
                    fittype = "exponential"
                    logarithmicfit = True
                    logx = True
                
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
                plt.scatter(xval, yval, marker='.', color='k')
                if xstat in plotmapping.keys():
                    plt.xlabel(plotmapping[xstat])
                else:
                    plt.xlabel(xstat)
                if ystat in plotmapping.keys():
                    plt.ylabel(plotmapping[ystat])
                else:
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
                plt.legend(fontsize=18, frameon=True)
                plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc_mad", f"fitted_{ystat}_{xstat}.png"))
                plt.close()
    
    def plot_single_fit(self, ax, xval, yval, xstat, ystat, utils, plotmapping):
        xstats = {
            "std":"exp",
            "Ensemble variance":"exp",
            "Pairwise correlation":"lin",
            "IQR (75-25%)":"exp",
            "cv":"exp",
            "mad":"exp",
            "Alpha":"exp",
            "Beta":"exp",
            "Pearson correlation":"exp",
            "stdsim":"exp",
            "stdobs":"exp",
            "meansim":"exp",
            "meanobs":"exp"
        }

        ystats = {
            "mean_membersobs_correlation": "lin",
            "mean_membersobs_RMSE": "exp",
            "mean_membersobs_KGE": "lin",
            "RMSE":"exp",
            "Pearson correlation":"lin",
            "KGE":"lin",
            "KGE_nobias":"exp",
            "Absolute mean bias":"exp",
            "NSE":"lin",
            "Beta":"exp",
            "Alpha":"exp",
            "(Alpha-1)^2":"exp",
            "(Beta-1)^2":"exp",
            "(r-1)^2":"exp"
        }

        logarithmicfit = False

        if xstats[xstat]=="exp" and ystats[ystat]=="exp":
            fittype = "powerlaw"
        elif xstats[xstat]=="exp" or ystats[ystat]=="exp":
            fittype = "exponential"
            if xstats[xstat]=="exp" and ystats[ystat]=="lin":
                logarithmicfit = True
        else:
            fittype = "linear"

        if xstat=="Pairwise correlation" and ystat=="Pearson correlation":
            fittype = "exponential"
            logarithmicfit = True

        # -------- fitting --------
        if fittype=="linear":
            params, _ = curve_fit(utils.linear_law, xval, yval)
            a_fit, b_fit = params
            x_fit = np.linspace(min(xval), max(xval), 100)
            y_fit = utils.linear_law(x_fit, a_fit, b_fit)

            R2 = r2_score(yval, utils.linear_law(xval, a_fit, b_fit))

            eq = f'$y={b_fit:.2f}x+{a_fit:.2f}$' if a_fit>=0 else f'$y={b_fit:.2f}x{a_fit:.2f}$'
            legend_text = eq + '\n' + f'$R^2$={R2:.3f}'

        elif fittype=="powerlaw":
            params, _ = curve_fit(utils.linear_law, np.log(xval), np.log(yval))
            a_fit, b_fit = params
            a_fit = np.exp(a_fit)

            x_fit = np.linspace(min(xval), max(xval), 100)
            y_fit = utils.powerlaw_func(x_fit, a_fit, b_fit)

            R2 = r2_score(np.log(yval), utils.linear_law(np.log(xval), np.log(a_fit), b_fit))

            legend_text = f'$y={a_fit:.2f}x$^{b_fit:.2f}\n$R^2$={R2:.3f}'

        elif fittype=="exponential":

            def log_func(x, a, b):
                return a + b*np.log(x)

            def exp_func(x, a, b):
                return a*np.exp(b*x)

            if logarithmicfit:
                popt, _ = curve_fit(log_func, xval, yval)
                a_fit, b_fit = popt
                x_fit = np.linspace(min(xval), max(xval), 100)
                y_fit = log_func(x_fit, *popt)
                y_pred = log_func(xval, *popt)
                r2 = r2_score(yval, y_pred)  # Compute R²
                logeq = f'y={a_fit:.2f}+{b_fit:.3f}log(x)' if b_fit>=0 else f'y={a_fit:.2f}{b_fit:.3f}log(x)'
                legend_text = '\n'.join((
                        logeq,
                        f'$R^2$ = {r2:.3f}'))
            else:
                popt, _ = curve_fit(exp_func, xval, yval, p0=(1, -0.1))
                a_fit, b_fit = popt
                x_fit = np.linspace(min(xval), max(xval), 100)
                y_fit = exp_func(x_fit, *popt)
                y_pred = exp_func(xval, *popt)
                r2 = r2_score(yval, y_pred)  # Compute R²
                legend_text = '\n'.join((
                        f'y={a_fit:.2f}e^{b_fit:.3f}x',
                        f'$R^2$ = {r2:.3f}'))

        # -------- plotting --------
        ax.scatter(xval, yval, color='k', s=5)
        # ax.plot(x_fit, y_fit, 'r--', label=legend_text)

        ax.set_xlabel(plotmapping.get(xstat, xstat))
        ax.set_ylabel(plotmapping.get(ystat, ystat))

        if xstat in ["std","Ensemble variance","IQR (75-25%)","cv","mad"]:
            ax.set_xscale("log")

        if ystat in ["RMSE","Absolute mean bias","Alpha","Beta","(Alpha-1)^2","(Beta-1)^2","(r-1)^2","mean_membersobs_RMSE"]:
            ax.set_yscale("log")

        if xstat=="Pairwise correlation":
            ax.set_xlim(0,1)
            if ystat=="Pearson correlation":
                ax.set_xscale("log")
                ax.set_xlim(0.001,1)

        if ystat=="Pearson correlation":
            ax.set_ylim(-1,1)

        if ystat=="KGE": 
            ax.set_ylim(-0.41,1)
            ax.set_yticks([-0.41, 0.0, 0.25, 0.45, 0.7, 1.0])
            
        if ystat=="NSE":
            ax.set_ylim(-1,1)
            ax.set_yticks([-1, -0.5, 0.0, 0.5, 1.0])
        if ystat=="Absolute mean bias":
            ax.set_yscale("log")
            ax.set_yticks([0.01, 0.1, 1, 10])

        ax.grid(True, linestyle='--', alpha=0.7)
        # ax.legend(fontsize=12, frameon=True)


    def create_group_figure(self, stat, combinations, filename, ncols=2):
        utils = utilities()

        plotmapping = {
            "Ensemble variance":r"$\overline{EV}$",
            "IQR (75-25%)":r"$\overline{IQR}$",
            "Beta":r"$\beta$",
            "Alpha":r"$\alpha$"
        }

        n = len(combinations)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 2.5*nrows))
        axes = np.array(axes).flatten()

        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        for i, (ystat, xstat) in enumerate(combinations):

            xval = stat[xstat].values
            yval = stat[ystat].values

            if ystat == "Absolute mean bias":
                yval = np.abs(yval)
                xval = xval[yval>=0.01]
                yval = yval[yval>=0.01]
            if ystat=="KGE":
                xval = xval[~np.isnan(yval)]
                yval = yval[~np.isnan(yval)]
                xval = xval[yval>=-0.41]
                yval = yval[yval>=-0.41]
            if ystat=="NSE":
                xval = xval[~np.isnan(yval)]
                yval = yval[~np.isnan(yval)]
                xval = xval[yval>=-1]
                yval = yval[yval>=-1]

            mask = ~np.isnan(xval) & ~np.isnan(yval)
            xval = xval[mask]
            yval = yval[mask]

            self.plot_single_fit(
                axes[i],
                xval,
                yval,
                xstat,
                ystat,
                utils,
                plotmapping
            )

            row = i // ncols
            col = i % ncols

            # ---- remove repeated labels ----
            if col != 0:
                axes[i].set_ylabel("")

            if row != nrows-1:
                axes[i].set_xlabel("")

            # ---- subplot label position ----
            # ypos = 0.05 if ystat == "Pearson correlation" else 0.95
            # va = 'bottom' if ystat == "Pearson correlation" else 'top'

            axes[i].text(
                0.02,
                1.11,
                f"({labels[i]})",
                transform=axes[i].transAxes,
                va='top',
                fontsize=12,
                #zorder=20,
                #bbox=dict(facecolor='white', edgecolor='none', alpha=0.85)
            )
            plt.subplots_adjust(
            hspace=0.0
            )

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"{filename}.pdf"), dpi=300)
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"{filename}.png"), dpi=300)
        print(f"saved figure: {filename}")
        plt.close()

    def create_group_figure_cv(self, stat, combinations, filename, ncols=2, cv_folds=5):
        utils = utilities()

        plotmapping = {
            "Ensemble variance":r"$\overline{EV}$",
            "IQR (75-25%)":r"$\overline{IQR}$",
            "Beta":r"$\beta$",
            "Alpha":r"$\alpha$"
        }

        n = len(combinations)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 2.5*nrows))
        axes = np.array(axes).flatten()

        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        xstats = {
            "std":"exp",
            "Ensemble variance":"exp",
            "Pairwise correlation":"lin",
            "IQR (75-25%)":"exp",
            "cv":"exp",
            "mad":"exp",
            "Alpha":"exp",
            "Beta":"exp",
            "Pearson correlation":"exp",
            "stdsim":"exp",
            "stdobs":"exp",
            "meansim":"exp",
            "meanobs":"exp"
        }

        ystats = {
            "mean_membersobs_correlation": "lin",
            "mean_membersobs_RMSE": "exp",
            "mean_membersobs_KGE": "lin",
            "RMSE":"exp",
            "Pearson correlation":"lin",
            "KGE":"lin",
            "KGE_nobias":"exp",
            "Absolute mean bias":"exp",
            "NSE":"lin",
            "Beta":"exp",
            "Alpha":"exp",
            "(Alpha-1)^2":"exp",
            "(Beta-1)^2":"exp",
            "(r-1)^2":"exp"
        }

        def filter_values(frame, xstat, ystat):
            xval = frame[xstat].values
            yval = frame[ystat].values

            if ystat == "Absolute mean bias":
                yval = np.abs(yval)
                xval = xval[yval >= 0.01]
                yval = yval[yval >= 0.01]
            if ystat == "KGE" or ystat == "NSE":
                xval = xval[~np.isnan(yval)]
                yval = yval[~np.isnan(yval)]
                xval = xval[yval >= 0.2]
                yval = yval[yval >= 0.2]

            mask = ~np.isnan(xval) & ~np.isnan(yval)
            return xval[mask], yval[mask]

        for i, (ystat, xstat) in enumerate(combinations):
            xval, yval = filter_values(stat, xstat, ystat)

            logarithmicfit = False

            if xstats[xstat] == "exp" and ystats[ystat] == "exp":
                fittype = "powerlaw"
            elif xstats[xstat] == "exp" or ystats[ystat] == "exp":
                fittype = "exponential"
                if xstats[xstat] == "exp" and ystats[ystat] == "lin":
                    logarithmicfit = True
            else:
                fittype = "linear"

            if xstat == "Pairwise correlation" and ystat == "Pearson correlation":
                fittype = "exponential"
                logarithmicfit = True

            if fittype == "powerlaw":
                positive_mask = (xval > 0) & (yval > 0)
                xval = xval[positive_mask]
                yval = yval[positive_mask]
            elif logarithmicfit:
                positive_mask = xval > 0
                xval = xval[positive_mask]
                yval = yval[positive_mask]

            if len(xval) == 0:
                continue

            model = _CrossValidatedCurveFitRegressor(fittype=fittype, logarithmicfit=logarithmicfit)
            model.fit(xval.reshape(-1, 1), yval)

            x_fit = np.linspace(min(xval), max(xval), 100)
            y_fit = model.predict(x_fit.reshape(-1, 1))

            if fittype == "linear":
                fit_params = (model.a_fit_, model.b_fit_)
                eq = f'$y={fit_params[1]:.3f}x+{fit_params[0]:.3f}$' if fit_params[0] >= 0 else f'$y={fit_params[1]:.3f}x{fit_params[0]:.3f}$'
            elif fittype == "powerlaw":
                fit_params = (model.a_fit_, model.b_fit_)
                eq = f'$y={fit_params[0]:.3f}x^{{{fit_params[1]:.3f}}}$'
            elif logarithmicfit:
                fit_params = (model.a_fit_, model.b_fit_, logarithmicfit)
                eq = f'$y={fit_params[0]:.3f}+{fit_params[1]:.3f}\\log(x)$' if fit_params[1] >= 0 else f'$y={fit_params[0]:.3f}{fit_params[1]:.3f}\\log(x)$'
            else:
                fit_params = (model.a_fit_, model.b_fit_, logarithmicfit)
                eq = f'$y={fit_params[0]:.3f}e^{{{fit_params[1]:.3f}x}}$'

            cv_predictions = None
            r2_cv = np.nan
            n_splits = min(cv_folds, len(xval))
            if n_splits >= 2:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_predictions = cross_val_predict(
                    _CrossValidatedCurveFitRegressor(fittype=fittype, logarithmicfit=logarithmicfit),
                    xval.reshape(-1, 1),
                    yval,
                    cv=cv
                )
                if fittype == "powerlaw":
                    r2_cv = r2_score(np.log(yval), np.log(cv_predictions))
                else:
                    r2_cv = r2_score(yval, cv_predictions)

            ax = axes[i]
            ax.scatter(xval, yval, color='k', s=5, alpha=0.6)
            # if cv_predictions is not None:
            #     ax.scatter(xval, cv_predictions, color='orange', s=5, alpha=0.6)#, label=f'$R^2_{{CV}}$ = {r2_cv:.2f}' if np.isfinite(r2_cv) else '$R^2_{CV}$ = n/a')
            eq = eq + '\n' + f'$R^2_{{CV}}$ = {r2_cv:.3f}' if np.isfinite(r2_cv) else '$R^2_{CV}$ = n/a'
            ax.plot(x_fit, y_fit, 'r--', label=eq)

            ax.set_xlabel(plotmapping.get(xstat, xstat))
            ax.set_ylabel(plotmapping.get(ystat, ystat))

            if xstat in ["std", "Ensemble variance", "IQR (75-25%)", "cv", "mad"]:
                ax.set_xscale("log")

            if ystat in ["RMSE", "Absolute mean bias", "Alpha", "Beta", "(Alpha-1)^2", "(Beta-1)^2", "(r-1)^2", "mean_membersobs_RMSE"]:
                ax.set_yscale("log")

            if xstat == "Pairwise correlation":
                ax.set_xlim(0, 1)
                if ystat == "Pearson correlation":
                    ax.set_xscale("log")
                    ax.set_xlim(0.001, 1)

            if ystat == "Pearson correlation":
                ax.set_ylim(-1, 1)

            if ystat == "KGE":
                ax.set_ylim(0.2, 1)

            ax.legend(loc='best', fontsize=9)

            row = i // ncols
            col = i % ncols

            if col != 0:
                ax.set_ylabel("")

            if row != nrows-1:
                ax.set_xlabel("")

            ax.text(
                0.02,
                1.11,
                f"({labels[i]})",
                transform=ax.transAxes,
                va='top',
                fontsize=12,
            )
            plt.subplots_adjust(hspace=0.0)

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"{filename}.pdf"), dpi=300)
        print(f"saved figure: {filename}")
        plt.close()

    def create_group_figure_cv_residuals(self, stat, combinations, filename, ncols=2, cv_folds=5):
        utils = utilities()

        plotmapping = {
            "Ensemble variance":r"$\overline{EV}$",
            "IQR (75-25%)":r"$\overline{IQR}$",
            "Beta":r"$\beta$",
            "Alpha":r"$\alpha$"
        }

        n = len(combinations)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 2.5*nrows))
        axes = np.array(axes).flatten()

        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        xstats = {
            "std":"exp",
            "Ensemble variance":"exp",
            "Pairwise correlation":"lin",
            "IQR (75-25%)":"exp",
            "cv":"exp",
            "mad":"exp",
            "Alpha":"exp",
            "Beta":"exp",
            "Pearson correlation":"exp",
            "stdsim":"exp",
            "stdobs":"exp",
            "meansim":"exp",
            "meanobs":"exp"
        }

        ystats = {
            "mean_membersobs_correlation": "lin",
            "mean_membersobs_RMSE": "exp",
            "mean_membersobs_KGE": "lin",
            "RMSE":"exp",
            "Pearson correlation":"lin",
            "KGE":"lin",
            "KGE_nobias":"exp",
            "Absolute mean bias":"exp",
            "NSE":"lin",
            "Beta":"exp",
            "Alpha":"exp",
            "(Alpha-1)^2":"exp",
            "(Beta-1)^2":"exp",
            "(r-1)^2":"exp"
        }

        def filter_values(frame, xstat, ystat):
            xval = frame[xstat].values
            yval = frame[ystat].values

            if ystat == "Absolute mean bias":
                yval = np.abs(yval)
                xval = xval[yval >= 0.01]
                yval = yval[yval >= 0.01]
            if ystat == "KGE" or ystat == "NSE":
                xval = xval[~np.isnan(yval)]
                yval = yval[~np.isnan(yval)]
                xval = xval[yval >= 0.2]
                yval = yval[yval >= 0.2]

            mask = ~np.isnan(xval) & ~np.isnan(yval)
            return xval[mask], yval[mask]

        for i, (ystat, xstat) in enumerate(combinations):
            xval, yval = filter_values(stat, xstat, ystat)

            logarithmicfit = False

            if xstats[xstat] == "exp" and ystats[ystat] == "exp":
                fittype = "powerlaw"
            elif xstats[xstat] == "exp" or ystats[ystat] == "exp":
                fittype = "exponential"
                if xstats[xstat] == "exp" and ystats[ystat] == "lin":
                    logarithmicfit = True
            else:
                fittype = "linear"

            if xstat == "Pairwise correlation" and ystat == "Pearson correlation":
                fittype = "exponential"
                logarithmicfit = True

            if fittype == "powerlaw":
                positive_mask = (xval > 0) & (yval > 0)
                xval = xval[positive_mask]
                yval = yval[positive_mask]
            elif logarithmicfit:
                positive_mask = xval > 0
                xval = xval[positive_mask]
                yval = yval[positive_mask]

            if len(xval) < 2:
                continue

            model = _CrossValidatedCurveFitRegressor(fittype=fittype, logarithmicfit=logarithmicfit)
            model.fit(xval.reshape(-1, 1), yval)

            n_splits = min(cv_folds, len(xval))
            if n_splits < 2:
                continue

            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_predictions = cross_val_predict(
                _CrossValidatedCurveFitRegressor(fittype=fittype, logarithmicfit=logarithmicfit),
                xval.reshape(-1, 1),
                yval,
                cv=cv
            )

            ax = axes[i]
            display_yval = np.log(yval) if fittype == "powerlaw" else yval
            display_cv_predictions = np.log(cv_predictions) if fittype == "powerlaw" else cv_predictions
            display = PredictionErrorDisplay.from_predictions(
                display_yval,
                display_cv_predictions,
                kind="residual_vs_predicted",
                ax=ax,
                scatter_kwargs={"s": 5, "alpha": 0.7},
            )

            # display.ax_.set_title(f"{ystat} vs {xstat}", fontsize=9)
            display.ax_.set_xlabel(f"Log-scale predicted values from {plotmapping[xstat]}")
            display.ax_.set_ylabel(f"{ystat} residuals")
            # display.ax_.set_xscale("log")
            # display.ax_.set_yscale("log")

            row = i // ncols
            col = i % ncols

            if col != 0:
                display.ax_.set_ylabel("")

            if row != nrows-1:
                display.ax_.set_xlabel("")

            display.ax_.text(
                0.02,
                1.11,
                f"({labels[i]})",
                transform=display.ax_.transAxes,
                va='top',
                fontsize=12,
            )

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"{filename}.pdf"), dpi=300)
        print(f"saved figure: {filename}")
        plt.close()
    
    def create_kge_component_figure(self, stat, filename):
        stat = stat.dropna()
        combinations = [
            ("KGE","Alpha"),
            ("KGE","Beta"),
            ("KGE","Pearson correlation"),
            ("KGE","stdobs"),
            ("KGE","stdsim"),
            ("KGE","meanobs"),
            ("KGE","meansim")
        ]

        plotmapping = {
            "Alpha": r"$\alpha=\frac{\sigma_{S}}{\sigma_{O}}$",
            "Beta": r"$\beta=\frac{\bar{S}}{\bar{O}}$",
            "stdsim": r"$\sigma_{S}$",
            "stdobs": r"$\sigma_{O}$",
            "meansim": r"$\bar{S}$",
            "meanobs": r"$\bar{O}$"
        }

        ncols = 3
        n = len(combinations)
        nrows = int(np.ceil(n / ncols))

        # fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 1.89*nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 2.7*nrows))
        axes = np.array(axes).flatten()

        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        for i, (ystat, xstat) in enumerate(combinations):

            xval = stat[xstat].values
            yval = stat[ystat].values

            mask = ~np.isnan(xval) & ~np.isnan(yval)
            xval = xval[mask]
            yval = yval[mask]

            ax = axes[i]

            ax.scatter(xval, yval, color='k', s=1)
            
            # labels
            col = i % ncols

            if col != 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel("KGE")

            ax.set_xlabel(plotmapping.get(xstat, xstat))

            # KGE limits
            ax.set_yscale("symlog")
            ax.set_xscale("log")
            if "A" in labels[i]:
                ax.set_xticks([1, 10, 100])
                print("done")

            # subplot labels
            ax.text(
                0.02,
                1.11,
                f"({labels[i]})",
                transform=ax.transAxes,
                va='top',
                fontsize=12,
            )
            
            

            ax.grid(True, linestyle='--', alpha=0.7)

        for ax in axes.flat:
            ax.tick_params(axis='both', which='both', length=0)
        # remove empty axes
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        
        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.97,
            bottom=0.12,
            wspace=0.0,
            hspace=-0.20
        )

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc_mad", f"{filename}.pdf"), dpi=300)
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc_mad", f"{filename}.eps"), dpi=300)
        print(f"saved figure: {filename}")
        plt.close()
    
    def create_pairwise_corr_figure(self, stat, filename, cv_folds=5):

        combinations = [
            ("RMSE","Pairwise correlation"),
            ("Pearson correlation","Pairwise correlation"),
            ("Absolute mean bias","Pairwise correlation"),
            # ("Alpha","Pairwise correlation"),
            # ("Beta","Pairwise correlation"),
            ("KGE","Pairwise correlation")
            # ("CRPS","Pairwise correlation"),
            # ("NSE","Pairwise correlation")
        ]

        plotmapping = {
            "Beta": r"$\beta$",
            "Alpha": r"$\alpha$"
        }

        ncols = 2
        n = len(combinations)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(7.09, 2.36*nrows))
        axes = np.array(axes).flatten()

        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        for i, (ystat, xstat) in enumerate(combinations):

            xval = stat[xstat].values
            yval = stat[ystat].values

            if ystat == "Absolute mean bias":
                yval = np.abs(yval)
                xval = xval[yval>=0.01]
                yval = yval[yval>=0.01]
            
            if ystat=="KGE":
                xval = xval[~np.isnan(yval)]
                yval = yval[~np.isnan(yval)]
                xval = xval[yval>=-0.41]
                yval = yval[yval>=-0.41]

            mask = ~np.isnan(xval) & ~np.isnan(yval)
            xval = xval[mask]
            yval = yval[mask]

            logarithmicfit = False

            if xstat=="Pairwise correlation" and ystat=="Pearson correlation":
                fittype = "exponential"
                logarithmicfit = True
            elif xstat in ["Ensemble variance", "IQR (75-25%)", "cv", "mad", "std", "Alpha", "Beta", "Pearson correlation", "stdsim", "stdobs", "meansim", "meanobs"] and ystat in ["RMSE", "Absolute mean bias", "Alpha", "Beta", "(Alpha-1)^2", "(Beta-1)^2", "(r-1)^2", "mean_membersobs_RMSE"]:
                fittype = "powerlaw"
            elif ystat in ["RMSE", "Absolute mean bias", "Alpha", "Beta", "(Alpha-1)^2", "(Beta-1)^2", "(r-1)^2", "mean_membersobs_RMSE"]:
                fittype = "exponential"
            elif ystat in ["Pearson correlation", "KGE", "NSE", "KGE_nobias", "mean_membersobs_correlation", "mean_membersobs_KGE"]:
                fittype = "linear"
            else:
                fittype = "linear"

            if fittype == "powerlaw":
                positive_mask = (xval > 0) & (yval > 0)
                xval = xval[positive_mask]
                yval = yval[positive_mask]
            elif logarithmicfit:
                positive_mask = xval > 0
                xval = xval[positive_mask]
                yval = yval[positive_mask]

            if len(xval) < 2:
                continue

            model = _CrossValidatedCurveFitRegressor(fittype=fittype, logarithmicfit=logarithmicfit)
            model.fit(xval.reshape(-1, 1), yval)

            x_fit = np.linspace(min(xval), max(xval), 100)
            y_fit = model.predict(x_fit.reshape(-1, 1))

            if fittype == "linear":
                fit_params = (model.a_fit_, model.b_fit_)
                eq = f'$y={fit_params[1]:.2f}x+{fit_params[0]:.2f}$' if fit_params[0] >= 0 else f'$y={fit_params[1]:.2f}x{fit_params[0]:.2f}$'
            elif fittype == "powerlaw":
                fit_params = (model.a_fit_, model.b_fit_)
                eq = f'$y={fit_params[0]:.2f}x^{{{fit_params[1]:.2f}}}$'
            elif logarithmicfit:
                fit_params = (model.a_fit_, model.b_fit_, logarithmicfit)
                eq = f'$y={fit_params[0]:.2f}+{fit_params[1]:.2f}\\log(x)$' if fit_params[1] >= 0 else f'$y={fit_params[0]:.2f}{fit_params[1]:.2f}\\log(x)$'
            else:
                fit_params = (model.a_fit_, model.b_fit_, logarithmicfit)
                eq = f'$y={fit_params[0]:.2f}e^{{{fit_params[1]:.2f}x}}$'

            r2_cv = np.nan
            n_splits = min(cv_folds, len(xval))
            if n_splits >= 2:
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_predictions = cross_val_predict(
                    _CrossValidatedCurveFitRegressor(fittype=fittype, logarithmicfit=logarithmicfit),
                    xval.reshape(-1, 1),
                    yval,
                    cv=cv,
                )
                if fittype == "powerlaw":
                    positive_cv = (yval > 0) & (cv_predictions > 0)
                    if np.any(positive_cv):
                        r2_cv = r2_score(
                            np.log(yval[positive_cv]),
                            np.log(cv_predictions[positive_cv]),
                        )
                else:
                    r2_cv = r2_score(yval, cv_predictions)

            legend_text = '\n'.join((
                eq,
                f'$R^2_{{CV}}$ = {r2_cv:.3f}' if np.isfinite(r2_cv) else '$R^2_{CV}$ = n/a'))

            axes[i].scatter(xval, yval, color='k', s=5)
            axes[i].plot(x_fit, y_fit, 'r--', label=legend_text)

            axes[i].set_xlabel(plotmapping.get(xstat, xstat))
            axes[i].set_ylabel(plotmapping.get(ystat, ystat))

            if ystat in ["RMSE", "Absolute mean bias", "Alpha", "Beta", "(Alpha-1)^2", "(Beta-1)^2", "(r-1)^2", "mean_membersobs_RMSE"]:
                axes[i].set_yscale("log")

            if xstat == "Pairwise correlation":
                axes[i].set_xlim(0, 1)
                if ystat == "Pearson correlation":
                    axes[i].set_xscale("log")
                    axes[i].set_xlim(0.001, 1)

            if ystat == "Pearson correlation":
                axes[i].set_ylim(-1, 1)

            if ystat == "KGE":
                axes[i].set_ylim(-0.41, 1)
                axes[i].set_yticks([-0.41, 0.0, 0.25, 0.45, 0.7, 1.0])

            axes[i].legend(fontsize=9, frameon=True)

            # row = i // ncols

            # keep ALL y-labels visible
            # only remove x-labels above bottom row
            # if row != nrows-1:
            #     axes[i].set_xlabel("")

            axes[i].text(
                0.02,
                1.11,
                f"({labels[i]})",
                transform=axes[i].transAxes,
                va='top',
                fontsize=12,
                # zorder=20,
                # bbox=dict(facecolor='white', edgecolor='none', alpha=0.85)
            )

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"{filename}.pdf"), dpi=300)
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"{filename}.png"), dpi=300)
        print(f"saved figure: {filename}")
        plt.close()
    
    def run_statvsacc_fitting(self):
        comb8 = [
            ("RMSE","Ensemble variance"),
            ("RMSE","IQR (75-25%)"),
            ("Absolute mean bias","Ensemble variance"),
            ("Absolute mean bias","IQR (75-25%)"),
            ("Pearson correlation","Ensemble variance"),
            ("Pearson correlation","IQR (75-25%)")
        ]
        comb9 = [
            ("KGE","Ensemble variance"),
            ("KGE","IQR (75-25%)")
        ]
        comb10 = [
            ("Alpha","Ensemble variance"),
            ("Alpha","IQR (75-25%)"),
            ("Beta","Ensemble variance"),
            ("Beta","IQR (75-25%)")
        ]
        combB2 = [
            ("NSE","Ensemble variance"),
            ("NSE","IQR (75-25%)")
        ]
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics.csv"))
        # self.create_group_figure(stat, comb8, "figure8", ncols=2)
        self.create_group_figure(stat, comb9, "figure9", ncols=2)
        # self.create_group_figure(stat, comb10, "figure10", ncols=2)
        # self.ensemble_crpsvsstats_fitting() # figure 11
        # self.create_kge_component_figure(stat, "figureS1")
        self.create_group_figure(stat, combB2, "figureS5", ncols=2)
        # self.create_pairwise_corr_figure(stat, "figureS8")

    def run_statvsacc_fitting_cv(self):
        comb8 = [
            ("RMSE","Ensemble variance"),
            ("RMSE","IQR (75-25%)"),
            ("Absolute mean bias","Ensemble variance"),
            ("Absolute mean bias","IQR (75-25%)"),
            ("Pearson correlation","Ensemble variance"),
            ("Pearson correlation","IQR (75-25%)")
        ]
        comb9 = [
            ("KGE","Ensemble variance"),
            ("KGE","IQR (75-25%)")
        ]
        comb10 = [
            ("Alpha","Ensemble variance"),
            ("Alpha","IQR (75-25%)"),
            ("Beta","Ensemble variance"),
            ("Beta","IQR (75-25%)")
        ]
        combB2 = [
            ("NSE","Ensemble variance"),
            ("NSE","IQR (75-25%)")
        ]
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics_mad.csv"))
        # self.create_group_figure_cv(stat, comb8, "figure8", ncols=2)
        # self.ensemble_crpsvsstats_fitting() # figure 11
        self.create_pairwise_corr_figure(stat, "figureS8")

    def run_statvsacc_fitting_cv_residuals(self):
        comb8 = [
            ("RMSE","Ensemble variance"),
            ("RMSE","IQR (75-25%)"),
            ("Absolute mean bias","Ensemble variance"),
            ("Absolute mean bias","IQR (75-25%)"),
            # ("Pearson correlation","Ensemble variance"),
            # ("Pearson correlation","IQR (75-25%)")
        ]
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics_mad.csv"))
        self.create_group_figure_cv_residuals(stat, comb8, "figure8residuals", ncols=2)

    def count_pearson_higher06(self):
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics_mad.csv"))
        stat = stat.dropna(subset=["Pearson correlation"])
        stat = stat[stat["Ensemble variance"] >= 100]
        count_higher06 = (stat["Pearson correlation"] > 0.6).sum()
        total_count = len(stat)
        percentage_higher06 = (count_higher06 / total_count) * 100
        print(f"Count of Pearson correlation > 0.6: {count_higher06}")
        print(f"Total count: {total_count}")
        print(f"Percentage of Pearson correlation > 0.6: {percentage_higher06:.2f}%")

    def cdf_EU(self):
        def calc_correlation(obs, sim):
            correlation_map = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    correlation_matrix = np.corrcoef(time_series1, time_series2)
                    r = correlation_matrix[0, 1]
                else:
                    r = np.nan
                correlation_map.append(r)
            return correlation_map
        
        def calc_RMSE(obs, sim):
            RMSE_allcells = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    rmse = np.sqrt(np.mean((time_series1 - time_series2) ** 2))
                else:
                    rmse = np.nan
                RMSE_allcells.append(rmse)
            return RMSE_allcells
        
        def calc_KGE_NSE_bias(obs, sim):
            all_kge = []
            all_nse = []
            all_bias = []
            for pixel in range(obs.shape[1]):
                time_series1 = obs[:, pixel]
                time_series2 = sim[:, pixel]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    kge = utils.calculate_kge(time_series1, time_series2)# if np.std(time_series1)>=0.1 else np.nan
                    nse = 1 - (np.sum((time_series1 - time_series2) ** 2) / np.sum((time_series1 - np.mean(time_series1)) ** 2))# if np.std(time_series1)>=0.1 else np.nan
                    bias_mean = np.mean(time_series2 - time_series1)
                else:
                    kge = np.nan
                    nse = np.nan
                    bias_mean = np.nan
                all_kge.append(kge)
                all_nse.append(nse)
                all_bias.append(bias_mean)
            return all_kge, all_nse, all_bias

        def load_obs_sim(dirpath, target, obsname, simname):
            obs = np.load(os.path.join(os.path.dirname(dirpath), f"target_pixels_{target}", obsname))
            sims = np.load(os.path.join(os.path.dirname(dirpath), f"target_pixels_{target}", simname))
            obs = np.nan_to_num(obs)
            sims = np.nan_to_num(sims)
            # obs[obs < 0.0] = 0.0
            # sims[sims < 0.0] = 0.0
            return obs, sims

        utils = utilities()
        print("starting the cdf calculation")
        correlation = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        rmse = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        kge = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        nse = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        bias = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        
        #### Transfer & training filtered subsets ####
        EU400px_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        EU100px_inpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_100_withcriteria_43226/inputs/20yrs_ts/ensemble_100px", "ensemble_mean")
        training_subset = np.load(os.path.join(os.path.dirname(os.path.dirname(EU400px_inpath)), "training_subset.npy"))
        training_subset_100px = np.load(os.path.join(os.path.dirname(EU100px_inpath), "target_pixels", "training_subset_ensemble100px.npy"))
        print("Progress: [" + "." * 100 + "]", flush=True)
        print("          [", end="", flush=True)  # Start progress bar            
        for target in range(100):
            print(".", end="", flush=True)  # Dots without newlines
            ######## 400px ensemble ########
            # calculate transfer metrics for 400px ensemble
            obs_transfer400, sim_transfer400 = load_obs_sim(EU400px_inpath, target, f"obs_{target}.npy", f"sim_transferpixels_{target}.npy")
            obs_test400, sim_test400 = load_obs_sim(EU400px_inpath, target, f"obs_{target}.npy", f"sim_testtrainpixels_{target}.npy")
            target_map = np.load(os.path.join(os.path.dirname(EU400px_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            training400px_indices, ind2d = utils.intersect_subsets(target_map, training_subset)
            corr_EU = calc_correlation(obs_transfer400, sim_transfer400)
            rmse_EU = calc_RMSE(obs_transfer400, sim_transfer400)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_transfer400, sim_transfer400)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from 99 transfer members (training subset) + 100 transfer members (transfer subset)
            correlation["transfer_400px"].extend(corr_EU.tolist())
            rmse["transfer_400px"].extend(rmse_EU.tolist())
            kge["transfer_400px"].extend(kge_EU.tolist())
            nse["transfer_400px"].extend(nse_EU.tolist())
            bias["transfer_400px"].extend(bias_EU.tolist())

            # calculate test metrics for 400px ensemble
            corr_EU = calc_correlation(obs_test400, sim_test400)
            rmse_EU = calc_RMSE(obs_test400, sim_test400)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_test400, sim_test400)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from only 1 training member (training subset)
            correlation["test_400px"].extend(corr_EU[training400px_indices].tolist())
            rmse["test_400px"].extend(rmse_EU[training400px_indices].tolist())
            kge["test_400px"].extend(kge_EU[training400px_indices].tolist())
            nse["test_400px"].extend(nse_EU[training400px_indices].tolist())
            bias["test_400px"].extend(bias_EU[training400px_indices].tolist())

            ######## 100px ensemble ########
            # calculate transfer metrics for 100px ensemble
            obs_transfer100, sim_transfer100 = load_obs_sim(EU100px_inpath, target, f"obs_{target}.npy", f"sim_transferpixels_{target}.npy")
            obs_test100, sim_test100 = load_obs_sim(EU100px_inpath, target, f"obs_{target}.npy", f"sim_testtrainpixels_{target}.npy")
            target_map = np.load(os.path.join(os.path.dirname(EU100px_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            training100px_indices, ind2d = utils.intersect_subsets(target_map, training_subset_100px)
            corr_EU = calc_correlation(obs_transfer100, sim_transfer100)
            rmse_EU = calc_RMSE(obs_transfer100, sim_transfer100)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_transfer100, sim_transfer100)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from 99 transfer members (training subset) + 100 transfer members (transfer subset)
            correlation["transfer_100px"].extend(corr_EU.tolist())
            rmse["transfer_100px"].extend(rmse_EU.tolist())
            kge["transfer_100px"].extend(kge_EU.tolist())
            nse["transfer_100px"].extend(nse_EU.tolist())
            bias["transfer_100px"].extend(bias_EU.tolist())

            # calculate test metrics for 100px ensemble
            corr_EU = calc_correlation(obs_test100, sim_test100)
            rmse_EU = calc_RMSE(obs_test100, sim_test100)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_test100, sim_test100)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from only 1 training member (training subset)
            correlation["test_100px"].extend(corr_EU[training100px_indices].tolist())
            rmse["test_100px"].extend(rmse_EU[training100px_indices].tolist())
            kge["test_100px"].extend(kge_EU[training100px_indices].tolist())
            nse["test_100px"].extend(nse_EU[training100px_indices].tolist())
            bias["test_100px"].extend(bias_EU[training100px_indices].tolist())
        
        print("] Done!", flush=True)
        # cleanup
        for key in correlation.keys():
            correlation[key] = np.array(correlation[key])
            correlation[key] = correlation[key][~np.isnan(correlation[key])]

        for key in rmse.keys():
            rmse[key] = np.array(rmse[key])
            rmse[key] = rmse[key][~np.isnan(rmse[key])]

        for key in kge.keys():
            kge[key] = np.array(kge[key])
            kge[key] = kge[key][~np.isnan(kge[key])]

        for key in nse.keys():
            nse[key] = np.array(nse[key])
            nse[key] = nse[key][~np.isnan(nse[key])]

        for key in bias.keys():
            bias[key] = np.array(bias[key])
            bias[key] = bias[key][~np.isnan(bias[key])]

        print("plotting")
        fig, axes = plt.subplots(
            2, 2,
            figsize=(7.09, 6.5)
        )

        common_colors = ['red', 'blue', 'red', 'blue']
        common_styles = ['--', '--', ':', ':']

        plot_specs = [
            (
                correlation,
                'Pearson correlation',
                '(A)',
                axes[0, 0],
                dict()
            ),
            (
                rmse,
                'RMSE (m)',
                '(B)',
                axes[0, 1],
                dict(logscale=True)
            ),
            (
                bias,
                'Mean bias (m)',
                '(C)',
                axes[1, 0],
                dict()
            ),
            (
                kge,
                'KGE',
                '(D)',
                axes[1, 1],
                dict(xmin=-0.41, xlim=(-0.41, 1), yfloor=True)
            )
        ]

        for metric_dict, xlabel, panel, ax, extra in plot_specs:

            utils.plot_cdfs(
                data_dict={
                    r'Transfer ($n$=400)': metric_dict["transfer_400px"],
                    r'Test ($n$=400)': metric_dict["test_400px"],
                    r'Transfer ($n$=100)': metric_dict["transfer_100px"],
                    r'Test ($n$=100)': metric_dict["test_100px"]
                },
                ax=ax,
                colors=common_colors,
                linestyles=common_styles,
                xlabel=xlabel,
                ylabel='Cumulative probability',
                **extra
            )

            ax.text(
                0.03, 1.08,
                panel,
                transform=ax.transAxes,
                fontsize=12,
                #fontweight='bold',
                va='top'
            )

            ax.tick_params(axis='both', labelsize=12)
            ax.xaxis.label.set_size(12)
            ax.yaxis.label.set_size(12)

        # remove subplot legends
        for ax in axes.flat:
            if ax.get_legend():
                ax.legend().remove()

        # single legend
        handles, labels = axes[0, 0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc='lower center',
            ncol=2,
            fontsize=12,
            frameon=False,
            bbox_to_anchor=(0.5, -0.05)
        )

        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.97,
            bottom=0.12,
            wspace=0.25,
            hspace=0.30
        )

        plt.savefig(
            os.path.join(OUTPUTPATH, "statistics", "figure5.png"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.savefig(
            os.path.join(OUTPUTPATH, "statistics", "figure5.pdf"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()
        
        fig, ax = plt.subplots(figsize=(3.35, 2.51))

        utils.plot_cdfs(
            data_dict={
                r'Transfer ($n$=400)': nse["transfer_400px"],
                r'Test ($n$=400)': nse["test_400px"],
                r'Transfer ($n$=100)': nse["transfer_100px"],
                r'Test ($n$=100)': nse["test_100px"]
            },
            ax=ax,
            colors=common_colors,
            linestyles=common_styles,
            xlabel='NSE',
            ylabel='Cumulative probability',
            xmin=-1,
            xlim=(-1, 1),
            yfloor=True
        )

        ax.tick_params(axis='both', labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)

        plt.savefig(
            os.path.join(OUTPUTPATH, "statistics", "figureB1.png"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.savefig(
            os.path.join(OUTPUTPATH, "statistics", "figureB1.pdf"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

    def crps_seasonal_trainEU(self):
        utils = utilities()

        obs = [np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"obs_ts_trainpixels_{target}.npy")) for target in range(100)]
        sims = [np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_99members_ts_trainpixels_{target}.npy")) for target in range(100)]

        obs = np.concatenate(obs, axis=1) #(timeseries, pixels)
        sims = np.concatenate(sims, axis=2) #(99 members, timeseries, pixels)
        
        crps = utils.compute_mean_seasonal_crps(observations=obs, simulations=sims, plot=True, title="train")
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), f"seasonalcrps_99members_onlytraining_4years4seasonspixel.npy"), crps)

    def crps_seasonal_transferEU(self):
        utils = utilities()

        obs = [np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"obs_ts_transferpixels_{target}.npy")) for target in range(100)]
        sims = [np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_100members_ts_transferpixels_{target}.npy")) for target in range(100)]

        obs = np.concatenate(obs, axis=1) #(timeseries, pixels)
        sims = np.concatenate(sims, axis=2) #(99 members, timeseries, pixels)
        
        crps = utils.compute_mean_seasonal_crps(observations=obs, simulations=sims, plot=True, title="transfer")
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), f"seasonalcrps_100members_onlytransfer_4years4seasonspixel.npy"), crps)

    def crps_transferEU(self):
        utils = utilities()

        obs = [np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"obs_ts_transferpixels_{target}.npy")) for target in range(100)]
        sims = [np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_100members_ts_transferpixels_{target}.npy")) for target in range(100)]

        obs = np.concatenate(obs, axis=1) #(timeseries, pixels)
        sims = np.concatenate(sims, axis=2) #(99 members, timeseries, pixels)

        crps = utils.compute_crps_all_pixels(observations=obs, simulations=sims, plot=False)
        np.save(os.path.join(os.path.dirname(OUTPUTPATH), f"crps_100members_onlytransfer.npy"), crps)

    def cdfplot_crps(self):
        utils = utilities()
        crpstrain = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"crps_99members_onlytraining.npy"))
        crpstransfer = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"crps_100members_onlytransfer.npy"))
        crps = np.concatenate((crpstrain, crpstransfer), axis=0)

        utils.plot_cdfs(data_dict={
            f'Transfer phase': crps,
        }, colors=['red'],
        xlabel='CRPS', logscale=True, title='CRPS')
        return
    
    def boxplot_seasonal_crps(self):
        crpstrain = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"seasonalcrps_99members_onlytraining_4years4seasonspixel.npy"))
        crpstransfer = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"seasonalcrps_100members_onlytransfer_4years4seasonspixel.npy"))
        crps = np.concatenate((crpstrain, crpstransfer), axis=2)
        # Plot boxplot per season per year
        fig, axs = plt.subplots(1, crps.shape[0], figsize=(7.09, 2.13), sharey=True)
        for i, year in enumerate(range(2017, 2021)):
            axs[i].boxplot(crps[i].T, labels=["DJF", "MAM", "JJA", "SON"])
            axs[i].set_title(f"{year}")
            axs[i].set_yscale('log')
            axs[i].grid(True, alpha=0.3)
            if i==0:
                axs[i].set_ylabel("CRPS (m)")
            
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"figure6.eps"), dpi=300)
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"figure6.pdf"), dpi=300)
        return
    
    def mapplot_seasonal_crps(self):
        utils = utilities()
        plotting = plotting_helper()
        crpstrain = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"seasonalcrps_99members_onlytraining_4years4seasonspixel.npy"))
        crpstransfer = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"seasonalcrps_100members_onlytransfer_4years4seasonspixel.npy"))
        training_map = np.load(os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_400_withcriteria_43226/inputs/20yrs_ts", "ensemble_400px", "target_pixels", "training_subset.npy"))
        transfer_map = np.load(os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_400_withcriteria_43226/inputs/20yrs_ts", "ensemble_400px", "target_pixels", "transfer_subset.npy"))
        crps4D = np.zeros((4, 4, training_map.shape[0], training_map.shape[1]))
        crps4D[crps4D==0] = np.nan
        traintarget_indx = 0
        transfertarget_indx = 0
        for target in range(100):
            target_map = np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            trainingpixels_in_target1d, trainingpixels_in_target2d = utils.intersect_subsets(target_map, training_map)
            transferpixels_in_target1d, transferpixels_in_target2d = utils.intersect_subsets(target_map, transfer_map)
            for i, pixel in enumerate(trainingpixels_in_target1d):
                crps4D[:, :, trainingpixels_in_target2d[0][i], trainingpixels_in_target2d[1][i]] = crpstrain[:, :, traintarget_indx+i]
            for i, pixel in enumerate(transferpixels_in_target1d):
                crps4D[:, :, transferpixels_in_target2d[0][i], transferpixels_in_target2d[1][i]] = crpstransfer[:, :, transfertarget_indx+i]
            traintarget_indx += len(trainingpixels_in_target1d)
            transfertarget_indx += len(transferpixels_in_target1d)
        
        figurepath = os.path.join(OUTPUTPATH, "statistics", f"seasonal_crps_map.png")
        plotting.plot_4d_map_logscale(data=crps4D, output_filename=figurepath)

    def preprocess_crps_trainingEU(self):
        utils = utilities()
        EU_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        EU_outpath = os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean")
        EU_traininpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_400_withcriteria_43226/inputs/20yrs_ts", "ensemble_400px")
        trainmembers_mapping = np.load(os.path.join(EU_traininpath, "mapping_memberstrainpixels.npy"))
        training_mapping = np.load(os.path.join(EU_traininpath, "target_pixels", "training_subset.npy"))
        for target in range(100):
            print(target)
            target_mapping = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            obs = np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_1", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
            trainingpixels_in_target1d, trainingpixels_in_target2d = utils.intersect_subsets(target_mapping, training_mapping)
            output = np.zeros((members_sim.shape[0]-1, members_sim.shape[1], len(trainingpixels_in_target1d)))
            out_obs = np.zeros((members_sim.shape[1], len(trainingpixels_in_target1d)))
            for i, pixel in enumerate(trainingpixels_in_target1d):
                member = trainmembers_mapping[trainingpixels_in_target2d[0][i], trainingpixels_in_target2d[1][i]]
                member_mask = np.ones(100, dtype=bool)
                member_mask[int(member)] = False
                pixel_sim = members_sim[member_mask, :, :][..., pixel]
                output[:, :, i] = pixel_sim
                out_obs[:,i] = obs[:, pixel]
            np.save(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_99members_ts_trainpixels_{target}.npy"), output)
            np.save(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"obs_ts_trainpixels_{target}.npy"), out_obs)

    def preprocess_crps_transferEU(self):
        utils = utilities()
        EU_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        EU_outpath = os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean")
        EU_traininpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_400_withcriteria_43226/inputs/20yrs_ts", "ensemble_400px")
        trainmembers_mapping = np.load(os.path.join(EU_traininpath, "mapping_memberstrainpixels.npy"))
        transfer_mapping = np.load(os.path.join(EU_traininpath, "target_pixels", "transfer_subset.npy"))
        print(f"transfer_mapping sum: {np.sum(transfer_mapping)}")
        for target in range(100):
            print(target)
            target_mapping = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            obs = np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_1", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
            transferpixels_in_target1d, transferpixels_in_target2d = utils.intersect_subsets(target_mapping, transfer_mapping)
            output = np.zeros((members_sim.shape[0], members_sim.shape[1], len(transferpixels_in_target1d)))
            out_obs = np.zeros((members_sim.shape[1], len(transferpixels_in_target1d)))
            output[:, :, :] = members_sim[:, :, transferpixels_in_target1d]
            out_obs[:,:] = obs[:, transferpixels_in_target1d]
            np.save(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_100members_ts_transferpixels_{target}.npy"), output)
            np.save(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"obs_ts_transferpixels_{target}.npy"), out_obs)
                
    
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

    def concat_EU_transfer_testtrain_outputs(self):
        utils = utilities()
        EU_inpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_100_withcriteria_43226/inputs/20yrs_ts/ensemble_100px", "ensemble_mean")
        EU_outpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_100_withcriteria_43226/outputs/20yrs_ts/ensemble_100px", "ensemble_mean")
        EU_traininpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_100_withcriteria_43226/inputs/20yrs_ts", "ensemble_100px")
        print(EU_inpath)
        for target in range(100):
            target_mapping = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            obs_destand_EU = np.load(os.path.join(os.path.dirname(EU_outpath), f"100px_member_1", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"100px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
            mean_prediction_transfer = np.zeros((members_sim.shape[1], members_sim.shape[2]))
            # array for test train pixels
            mean_prediction_testtrain = np.zeros((members_sim.shape[1], members_sim.shape[2]))
            mean_prediction_testtrain[mean_prediction_testtrain==0] = np.nan
            
            untrained_pixel_mask = np.ones(members_sim.shape[2], dtype=bool)
            for member in range(100):
                member_mask = np.ones(100, dtype=bool)
                member_mask[member] = False
                pixel_mask = np.zeros(members_sim.shape[2], dtype=bool)
                # get the 2d mapping file of indices of pixels used in training this member
                member_mapping = np.load(os.path.join(EU_traininpath, f"100px_member_{member}", "choices.npy"))
                # call function here to find which pixels of my target chunk were included in the training of this specific member
                indices1d, indices2d = utils.intersect_subsets(target_mapping, member_mapping)
                # chunk pixels included in training are set as True
                pixel_mask[indices1d] = True

                # take only the member and its relative pixels
                mean_prediction_testtrain[:, pixel_mask] = members_sim[member, :, :][..., pixel_mask]

                # keep track of pixels included in training any member
                untrained_pixel_mask[indices1d] = False
                # For pixels where pixel_mask=False (pixels included in training this member) 
                # then get the Mean over SELECTED MEMBERS (member_mask=True) excluding the one False member (the loop member)
                mean_prediction_transfer[:, pixel_mask] = np.mean(members_sim[member_mask, :, :][..., pixel_mask], axis=0)

            # For other pixels (not included in training any member) get the Mean over ALL MEMBERS (axis=0)
            mean_prediction_transfer[:, untrained_pixel_mask] = np.mean(members_sim[:, :, untrained_pixel_mask], axis=0)
            print(obs_destand_EU.shape)
            print(mean_prediction_transfer.shape)
            np.save(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"sim_testtrainpixels_{target}.npy"), mean_prediction_testtrain)
            np.save(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"obs_{target}.npy"), obs_destand_EU)
            np.save(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"sim_transferpixels_{target}.npy"), mean_prediction_transfer)
            print(f"saved {target} target pixels")

    def map_1Dto2D_EU(self):
        def calc_correlation(obs, sim):
            correlation_map = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    correlation_matrix = np.corrcoef(time_series1, time_series2)
                    r = correlation_matrix[0, 1]
                else:
                    r = np.nan
                correlation_map.append(r)
            return correlation_map
        
        def calc_RMSE(obs, sim):
            RMSE_allcells = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    rmse = np.sqrt(np.mean((time_series1 - time_series2) ** 2))
                else:
                    rmse = np.nan
                RMSE_allcells.append(rmse)
            return RMSE_allcells
        
        def calc_KGE_NSE_bias(obs, sim):
            kge = utils.calculate_kge(obs, sim)
            all_kge = []
            nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
            all_nse = []
            bias_mean = np.mean(sim - obs)
            all_bias = []
            for pixel in range(obs.shape[1]):
                time_series1 = obs[:, pixel]
                time_series2 = sim[:, pixel]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    kge = utils.calculate_kge(time_series1, time_series2)# if np.std(time_series1)>=0.1 else np.nan
                    nse = 1 - (np.sum((time_series1 - time_series2) ** 2) / np.sum((time_series1 - np.mean(time_series1)) ** 2))# if np.std(time_series1)>=0.1 else np.nan
                    bias_mean = np.mean(time_series2 - time_series1)
                else:
                    kge = np.nan
                    nse = np.nan
                    bias_mean = np.nan
                all_kge.append(kge)
                all_nse.append(nse)
                all_bias.append(bias_mean)
            return all_kge, all_nse, all_bias

        def load_obs_sim(target):
            obs_destand_test = np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"obs_{target}.npy"))
            sim_destand_test = np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_ensmean.npy"))
            obs_destand_test = np.nan_to_num(obs_destand_test)
            sim_destand_test = np.nan_to_num(sim_destand_test)
            obs_destand_test[obs_destand_test < 0.0] = 0.0
            sim_destand_test[sim_destand_test < 0.0] = 0.0
            return obs_destand_test, sim_destand_test
        
        utils = utilities()
        EU_inpath = os.path.dirname(INPUTPATH)
        print("starting calculations")

        #### Transfer subset ####
        
        # filtered subset
        transfer_subset = np.load(os.path.join(EU_inpath, "target_pixels", "transfer_subset.npy"))
        training_subset = np.load(os.path.join(EU_inpath, "target_pixels", "training_subset.npy"))

        corr2d = np.zeros(transfer_subset.shape)
        corr2d[corr2d==0] = np.nan
        rmse2d = np.zeros(transfer_subset.shape)
        rmse2d[rmse2d==0] = np.nan
        bias2d = np.zeros(transfer_subset.shape)
        bias2d[bias2d==0] = np.nan
        kge2d = np.zeros(transfer_subset.shape)
        kge2d[kge2d==0] = np.nan
        nse2d = np.zeros(transfer_subset.shape)
        nse2d[nse2d==0] = np.nan
        for target in range(100):
            print(f"{target}/99",end="\r", flush=True)
            obs_destand_test, sim_destand_test = load_obs_sim(target)
            target_map = np.load(os.path.join(EU_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            indices, indices_2d = utils.intersect_subsets(target_map, transfer_subset)
            indices_train, indices_2d_train = utils.intersect_subsets(target_map, training_subset)

            obs_transfer = obs_destand_test[:, indices]
            sim_transfer = sim_destand_test[:, indices]
            corr_EU = calc_correlation(sim_transfer, obs_transfer)
            rmse_EU = calc_RMSE(sim_transfer, obs_transfer)
            kge, nse, bias = calc_KGE_NSE_bias(obs_transfer, sim_transfer)
            corr2d[indices_2d] = corr_EU
            rmse2d[indices_2d] = rmse_EU
            bias2d[indices_2d] = bias
            kge2d[indices_2d] = kge
            nse2d[indices_2d] = nse

            obs_train = obs_destand_test[:, indices_train]
            sim_train = sim_destand_test[:, indices_train]
            corr_EU = calc_correlation(sim_train, obs_train)
            rmse_EU = calc_RMSE(sim_train, obs_train)
            kge, nse, bias = calc_KGE_NSE_bias(obs_train, sim_train)
            corr2d[indices_2d_train] = corr_EU
            rmse2d[indices_2d_train] = rmse_EU
            bias2d[indices_2d_train] = bias
            kge2d[indices_2d_train] = kge
            nse2d[indices_2d_train] = nse

        return corr2d, rmse2d, bias2d, kge2d, nse2d

    def map_1Dto2D_EU_onlytransfer(self):
        def calc_correlation(obs, sim):
            correlation_map = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    correlation_matrix = np.corrcoef(time_series1, time_series2)
                    r = correlation_matrix[0, 1]
                else:
                    r = np.nan
                correlation_map.append(r)
            return correlation_map
        
        def calc_RMSE(obs, sim):
            RMSE_allcells = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    rmse = np.sqrt(np.mean((time_series1 - time_series2) ** 2))
                else:
                    rmse = np.nan
                RMSE_allcells.append(rmse)
            return RMSE_allcells
        
        def calc_KGE_NSE_bias(obs, sim):
            kge = utils.calculate_kge(obs, sim)
            all_kge = []
            nse = 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
            all_nse = []
            bias_mean = np.mean(sim - obs)
            all_bias = []
            for pixel in range(obs.shape[1]):
                time_series1 = obs[:, pixel]
                time_series2 = sim[:, pixel]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    kge = utils.calculate_kge(time_series1, time_series2)# if np.std(time_series1)>=0.1 else np.nan
                    nse = 1 - (np.sum((time_series1 - time_series2) ** 2) / np.sum((time_series1 - np.mean(time_series1)) ** 2))# if np.std(time_series1)>=0.1 else np.nan
                    bias_mean = np.mean(time_series2 - time_series1)
                else:
                    kge = np.nan
                    nse = np.nan
                    bias_mean = np.nan
                all_kge.append(kge)
                all_nse.append(nse)
                all_bias.append(bias_mean)
            return all_kge, all_nse, all_bias

        def load_obs_sim(target):
            obs_destand_test = np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"obs_{target}.npy"))
            sim_destand_test = np.load(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{target}", f"sim_ensmean.npy"))
            obs_destand_test = np.nan_to_num(obs_destand_test)
            sim_destand_test = np.nan_to_num(sim_destand_test)
            obs_destand_test[obs_destand_test < 0.0] = 0.0
            sim_destand_test[sim_destand_test < 0.0] = 0.0
            return obs_destand_test, sim_destand_test
        
        utils = utilities()
        EU_inpath = os.path.dirname(INPUTPATH)
        print("starting calculations")

        #### Transfer subset ####
        
        # filtered subset
        transfer_subset = np.load(os.path.join(EU_inpath, "target_pixels", "transfer_subset.npy"))
        training_subset = np.load(os.path.join(EU_inpath, "target_pixels", "training_subset.npy"))

        corr2d = np.zeros(transfer_subset.shape)
        corr2d[corr2d==0] = np.nan
        rmse2d = np.zeros(transfer_subset.shape)
        rmse2d[rmse2d==0] = np.nan
        bias2d = np.zeros(transfer_subset.shape)
        bias2d[bias2d==0] = np.nan
        kge2d = np.zeros(transfer_subset.shape)
        kge2d[kge2d==0] = np.nan
        nse2d = np.zeros(transfer_subset.shape)
        nse2d[nse2d==0] = np.nan
        for target in range(100):
            print(f"{target}/99",end="\r", flush=True)
            obs_destand_test, sim_destand_test = load_obs_sim(target)
            target_map = np.load(os.path.join(EU_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            indices, indices_2d = utils.intersect_subsets(target_map, transfer_subset)
            indices_train, indices_2d_train = utils.intersect_subsets(target_map, training_subset)

            obs_transfer = obs_destand_test[:, indices]
            sim_transfer = sim_destand_test[:, indices]
            corr_EU = calc_correlation(sim_transfer, obs_transfer)
            rmse_EU = calc_RMSE(sim_transfer, obs_transfer)
            kge, nse, bias = calc_KGE_NSE_bias(obs_transfer, sim_transfer)
            corr2d[indices_2d] = corr_EU
            rmse2d[indices_2d] = rmse_EU
            bias2d[indices_2d] = bias
            kge2d[indices_2d] = kge
            nse2d[indices_2d] = nse

        return corr2d, rmse2d, bias2d, kge2d, nse2d
    
    def metrics_2D_EU(self):
        plot_functions = plotting_helper()
        corr2d, rmse2d, bias2d, kge2d, nse2d = self.map_1Dto2D_EU()

        # get indices where kge<-0.41
        # kge2d = np.where(kge2d<-0.41, np.nan, kge2d)
        # nse2d = np.where(np.isnan(kge2d), np.nan, nse2d)
        # corr2d = np.where(np.isnan(kge2d), np.nan, corr2d)
        # rmse2d = np.where(np.isnan(kge2d), np.nan, rmse2d)
        # bias2d = np.where(np.isnan(kge2d), np.nan, bias2d)
        # bias2d = np.where(kge2d<-0.41, np.nan, bias2d)
        
        fig, axes = plt.subplots(
            2, 2,
            figsize=(7.09, 6.5),
            subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                central_longitude=19,
                central_latitude=53
            )}
            )

        plot_functions.combinedfigs_EU_2Dmap(
            data_map=corr2d,
            logscale=False,
            minval=0,
            maxval=1,
            title="Pearson correlation",
            ax=axes[0, 0],
            panel_label="(A)"
        )

        plot_functions.combinedfigs_EU_2Dmap(
            data_map=rmse2d,
            logscale=True,
            minval=0.1,
            maxval=10,
            title="RMSE",
            ax=axes[0, 1],
            panel_label="(B)"
        )

        plot_functions.combinedfigs_EU_2Dmap(
            data_map=bias2d,
            logscale=False,
            minval=-10,
            maxval=10,
            title="Mean bias",
            ax=axes[1, 0],
            panel_label="(C)"
        )

        plot_functions.combinedfigs_EU_2Dmap(
            data_map=kge2d,
            logscale=False,
            minval=-0.41,
            maxval=1,
            title="KGE",
            ax=axes[1, 1],
            panel_label="(D)"
        )

        plt.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.97,
            bottom=0.03,
            wspace=0.08,
            hspace=0.08
        )

        # plt.savefig(
        #     os.path.join(OUTPUTPATH, "figure7.png"),
        #     dpi=300,
        #     bbox_inches='tight'
        # )

        # plt.savefig(
        #     os.path.join(OUTPUTPATH, "figure7.pdf"),
        #     dpi=300,
        #     bbox_inches='tight'
        # )

        plt.close()

        fig, ax = plt.subplots(
            figsize=(3.35, 2.345),
            subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(
                central_longitude=19,
                central_latitude=53
            )}
        )

        plot_functions.combinedfigs_EU_2Dmap(
            data_map=nse2d,
            logscale=False,
            minval=-1,
            maxval=1,
            title="NSE",
            ax=ax
        )

        plt.savefig(
            os.path.join(OUTPUTPATH, "figureB2.png"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.savefig(
            os.path.join(OUTPUTPATH, "figureB2.pdf"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()
    
    def RMSE_vs_05obsstd(self):
        plot_functions = plotting_helper()
        corr2d, rmse2d, bias2d, kge2d, nse2d = self.map_1Dto2D_EU()

        wtdstd_testperiod = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtdstd_testperiod.npy"))
        wtdmax_testperiod = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtdmax_testperiod.npy"))
        wtdmin_testperiod = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtdmin_testperiod.npy"))
        wtdstd_testperiod = np.where(np.isnan(rmse2d), np.nan, wtdstd_testperiod)
        wtdmax_testperiod = np.where(np.isnan(rmse2d), np.nan, wtdmax_testperiod)
        wtdmin_testperiod = np.where(np.isnan(rmse2d), np.nan, wtdmin_testperiod)
        rmse2d_indices = np.where(~np.isnan(rmse2d))
        print(len(rmse2d_indices[0]))
        rmse_stdobs_2d = np.zeros(rmse2d.shape)
        rmse_stdobs_2d[rmse_stdobs_2d==0] = np.nan
 
        # rmse_stdobs_2d = np.where(rmse2d < 0.5*np.nanmean(wtdstd_testperiod), 0.01, 1)
        rmse_stdobs_2d = np.where(rmse2d < 0.5*wtdstd_testperiod, 0.01, 1)
        rmse_stdobs_2d = np.where(np.isnan(rmse2d), np.nan, rmse_stdobs_2d)

        print(rmse_stdobs_2d)
        print(rmse_stdobs_2d.shape)
        print(np.unique(rmse_stdobs_2d, return_counts=True))
        print(np.nanmean(wtdstd_testperiod), np.nanmax(wtdstd_testperiod), np.nanmin(wtdstd_testperiod))
        print(np.nanmean(wtdmax_testperiod), np.nanmax(wtdmax_testperiod), np.nanmin(wtdmax_testperiod))
        print(np.nanmean(wtdmin_testperiod), np.nanmax(wtdmin_testperiod), np.nanmin(wtdmin_testperiod))
        plot_functions.EU_2Dmap_binary(rmse_stdobs_2d, "RMSE/0.5*ObsStd", threshold=0.1)

    def NRMSE_map(self):
        plot_functions = plotting_helper()
        corr2d, rmse2d, bias2d, kge2d, nse2d = self.map_1Dto2D_EU()

        wtdmin_testperiod = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtdmin_testperiod.npy"))
        wtdmax_testperiod = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtdmax_testperiod.npy"))
        rmse2d_indices = np.where(~np.isnan(rmse2d))
        print(len(rmse2d_indices[0]))
        nrmse2d = np.zeros(rmse2d.shape)
        nrmse2d[nrmse2d==0] = np.nan
        for i in range(len(rmse2d_indices[0])):
            print(i)
            y = rmse2d_indices[0][i]
            x = rmse2d_indices[1][i]
            nrmse2d[y,x] = rmse2d[y,x] / (wtdmax_testperiod[y,x] - wtdmin_testperiod[y,x])
        
        print(nrmse2d)
        print(nrmse2d.shape)
        plot_functions.EU_2Dmap_binary(nrmse2d, "NRMSE", threshold=0.1)

    def metrics_vs_topo_combined(self):
        # --- data ---
        corr2d, rmse2d, bias2d, kge2d, nse2d = self.map_1Dto2D_EU_onlytransfer()

        topo = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "topo.npy"))[0, :, :]
        wtd = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtd.npy"))
        wtd = np.mean(wtd, axis=0)

        # mask invalid pixels
        topo = np.where(np.isnan(corr2d), np.nan, topo)
        wtd = np.where(np.isnan(corr2d), np.nan, wtd)
        wtd = np.where(wtd <= 0, np.nan, wtd)  # set non-positive WTD to NaN for log scale

        # flatten & remove NaNs
        def clean_xy(x, y):
            x = x.flatten()
            y = y.flatten()
            mask = ~np.isnan(x) & ~np.isnan(y)
            return x[mask], y[mask]

        wtd_x, corr_wtd = clean_xy(wtd, corr2d)
        topo_x, corr_topo = clean_xy(topo, corr2d)

        # --- figure ---
        fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.95))

        # (a) WTD vs correlation
        axes[0].scatter(wtd_x, corr_wtd, alpha=0.5)
        axes[0].set_xlabel(r'$Mean\ WTD\ (m)$')
        axes[0].set_ylabel('Pearson correlation')
        axes[0].set_xscale('log')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        axes[0].text(
            0.02, 1.08, '(A)',
            transform=axes[0].transAxes,
            va='top',
            fontsize=12,
            # zorder=20,
            # bbox=dict(facecolor='white', edgecolor='none', alpha=0.85)
        )

        # (b) Topography vs correlation
        axes[1].scatter(topo_x, corr_topo, alpha=0.5)
        axes[1].set_xlabel(r'$Topography\ (m)$')
        axes[1].set_ylabel('')  # remove y-label
        axes[1].set_xscale('log')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        axes[1].text(
            0.02, 1.08, '(B)',
            transform=axes[1].transAxes,
            va='top',
            fontsize=12,
            # zorder=20,
            # bbox=dict(facecolor='white', edgecolor='none', alpha=0.85)
        )

        # layout
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUTPATH, "statistics", "figureC1.pdf"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(OUTPUTPATH, "statistics", "figureC1.eps"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def metrics_vs_topo(self):
        ###### comment out the training pixels from self.map_1Dto2D_EU() ######
        # take only the transfer ones
        corr2d, rmse2d, bias2d, kge2d, nse2d = self.map_1Dto2D_EU()
        plot_functions = plotting_helper()
        #### for topography #####
        topo_v1 = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "topo.npy"))
        topo_v1 = topo_v1[0,:,:]
        #### for wtd #####
        wtd = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtd.npy"))
        wtd = np.mean(wtd, axis=0)  # average wtd over time

        topo = topo_v1
        topo[np.isnan(corr2d)] = np.nan
        plot_functions.plot_scatter(topo, corr2d, "Pearson correlation")
        topo = topo_v1
        topo[np.isnan(rmse2d)] = np.nan
        plot_functions.plot_scatter(topo, rmse2d, "RMSE", ylog=True)
        topo = topo_v1
        topo[np.isnan(bias2d)] = np.nan
        plot_functions.plot_scatter(topo, bias2d, "Mean absolute bias (m)", ylog=True)
        topo = topo_v1
        topo[np.isnan(kge2d)] = np.nan
        plot_functions.plot_scatter(topo, kge2d, "KGE", ylog=False, ysymlog=True)
        topo = topo_v1
        topo[np.isnan(nse2d)] = np.nan
        plot_functions.plot_scatter(topo, nse2d, "NSE", ylog=False, ysymlog=True)


    def ensmean_3d(self):
        def load_obs_sim(target):
            obs_destand_test = np.load(os.path.join(os.path.dirname(dirpath), f"target_pixels_{target}", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            sim_destand_test = np.load(os.path.join(os.path.dirname(dirpath), f"target_pixels_{target}", f"sim_destand_{MODEL_NAME}_{target}.npy"))
            obs_destand_test = np.nan_to_num(obs_destand_test)
            sim_destand_test = np.nan_to_num(sim_destand_test)
            obs_destand_test[obs_destand_test < 0.0] = 0.0
            sim_destand_test[sim_destand_test < 0.0] = 0.0
            return obs_destand_test, sim_destand_test
        # TODO review it because it got messy
        utils = utilities()
        print("calculating 2d ensemble mean")
        dirpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        outpath = os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean")
        subset = np.load(os.path.join(os.path.dirname(os.path.dirname(dirpath)), "transfer_subset.npy"))
        print(np.sum(subset))
        obs3d = np.zeros((TEST_PERIOD-LOOKBACK,subset.shape[0], subset.shape[1]))
        obs3d[obs3d==0] = np.nan
        sim3d = np.zeros((TEST_PERIOD-LOOKBACK,subset.shape[0], subset.shape[1]))
        sim3d[sim3d==0] = np.nan
        for target in range(100):
            obs_destand_test, sim_destand_test = load_obs_sim(target)
            target_map = np.load(os.path.join(os.path.dirname(dirpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            indices, indices_2d = utils.intersect_subsets(target_map, subset)
            
            obs_destand_test = obs_destand_test[:, indices]
            sim_destand_test = sim_destand_test[:, indices]
            for i in range(len(indices_2d[0])):
                obs3d[:,indices_2d[0][i],indices_2d[1][i]] = obs_destand_test[:,i]
                sim3d[:,indices_2d[0][i],indices_2d[1][i]] = sim_destand_test[:,i]
        print(obs3d[~np.isnan(obs3d)].shape)
        print(sim3d[~np.isnan(sim3d)].shape)
        print(f"length of timeseries: {sim3d.shape[0]}")
        print(f"calculated length of timeseries: {len(sim3d[~np.isnan(sim3d)])/np.sum(subset)}")
        np.save(os.path.join(outpath, "obs3d_ensmean.npy"), obs3d)
        np.save(os.path.join(outpath, "sim3d_ensmean.npy"), sim3d)
        print("saved ensemble mean")    

    def calculate_mapped_stats(self):
        EU_filtering = "validation_400_withcriteria_43226"
        inpath = os.path.join(f"/p/project1/cslts/miaari1/python_scripts/fork/{EU_filtering}/inputs/20yrs_ts/ensemble_400px", "ensemble_mean")
        outpath = os.path.join(f"/p/project1/cslts/miaari1/python_scripts/fork/{EU_filtering}/outputs/20yrs_ts/ensemble_400px", "ensemble_mean")
        data1_path = os.path.join(outpath, "obs3d_ensmean.npy")
        data2_path = os.path.join(outpath, "sim3d_ensmean.npy")
        mapping_path = os.path.join(os.path.dirname(os.path.dirname(inpath)), "transfer_subset.npy")
        #mapping_path = os.path.join(get_root_dir(), "inputs", "20yrs_ts", "ensemble_400px_org", "transfer_subset_unfiltered.npy")
        # Load all arrays
        data1 = np.load(data1_path)  # Shape (D1, D2, D3)
        data2 = np.load(data2_path)
        mapping = np.load(mapping_path)
        print(data1.shape)
        print(np.sum(mapping))
        # exclude pixels with std=0 or std<0.1
        mapping = np.where(np.std(data1, axis=0)<0.1, 0, mapping)
        mapping = np.where(np.std(data2, axis=0)==0.0, 0, mapping)
        
        # Calculate statistics
        print(data1[:, mapping==1].shape)
        results = {
            'data1': {
                'mean': np.mean(data1[:, mapping==1], axis=0),
                'std': np.std(data1[:, mapping==1], axis=0)
            },
            'data2': {
                'mean': np.mean(data2[:, mapping==1], axis=0),
                'std': np.std(data2[:, mapping==1], axis=0)
            }
        }

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Std vs Std plot
        ax1.scatter(results['data1']['std'], results['data2']['std'], alpha=0.5)
        ax1.plot([0, max(results['data1']['std'].max(), results['data2']['std'].max())], 
                [0, max(results['data1']['std'].max(), results['data2']['std'].max())], 
                'r--')
        ax1.set_xlabel('Observed STD')
        ax1.set_ylabel('Simulated STD')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(min(results['data1']['std'].min(), results['data2']['std'].min()), max(results['data1']['std'].max(), results['data2']['std'].max()))
        ax1.set_ylim(min(results['data1']['std'].min(), results['data2']['std'].min()), max(results['data1']['std'].max(), results['data2']['std'].max()))
        #ax1.set_title('Standard Deviation Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Mean vs Mean plot
        ax2.scatter(results['data1']['mean'], results['data2']['mean'], alpha=0.5)
        ax2.plot([results['data1']['mean'].min(), results['data1']['mean'].max()], 
                [results['data1']['mean'].min(), results['data1']['mean'].max()], 
                'r--')
        ax2.set_xlabel('Observed Mean')
        ax2.set_ylabel('Simulated Mean')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(min(results['data1']['mean'].min(), results['data2']['mean'].min()), max(results['data1']['mean'].max(), results['data2']['mean'].max()))
        ax2.set_ylim(min(results['data1']['mean'].min(), results['data2']['mean'].min()), max(results['data1']['mean'].max(), results['data2']['mean'].max()))
        #ax2.set_xlim(0.000001, max(results['data1']['mean'].max(), results['data2']['mean'].max()))
        #ax2.set_ylim(0.000001, max(results['data1']['mean'].max(), results['data2']['mean'].max()))
        #ax2.set_title('Mean Value Comparison')
        print(min(results['data1']['mean'].min(), results['data2']['mean'].min()), max(results['data1']['mean'].max(), results['data2']['mean'].max()))
        print(min(results['data1']['std'].min(), results['data2']['std'].min()), max(results['data1']['std'].max(), results['data2']['std'].max()))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        plot_path = os.path.join(os.path.dirname(os.path.dirname(OUTPUTPATH)), EU_filtering, "statistics", "std_mean_plots_stdobs01.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {plot_path}")
    
    def estimate_acc_from_stats(self):
        utils = utilities()
        plot_functions = plotting_helper()
        EU_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        EU_outpath = os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean")
        EU_traininpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_400_withcriteria_43226/inputs/20yrs_ts", "ensemble_400px")
        rollsubset = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "mapping_0stdroll6months.npy"))
        iqrmap = np.zeros(rollsubset.shape)
        iqrmap[iqrmap==0] = np.nan
        varmap = np.zeros(rollsubset.shape)
        varmap[varmap==0] = np.nan
        print("Progress: [" + "." * 100 + "]", flush=True)
        print("          [", end="", flush=True)
        for target in range(100):
            print(".", end="", flush=True)  # Dots without newlines
            target_mapping = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            #obs_destand_EU = np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_1", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
            mean_iqr = np.zeros((members_sim.shape[2]))
            mean_var = np.zeros((members_sim.shape[2]))
            untrained_pixel_mask = np.ones(members_sim.shape[2], dtype=bool)
            for member in range(100):
                member_mask = np.ones(100, dtype=bool)
                member_mask[member] = False
                pixel_mask = np.zeros(members_sim.shape[2], dtype=bool)
                # get the 2d mapping file of indices of pixels used in training this member
                member_mapping = np.load(os.path.join(EU_traininpath, f"400px_member_{member}", "choices.npy"))
                # call function here to find which pixels of my target chunk were included in the training of this specific member
                indices1d, indices2d = utils.intersect_subsets(target_mapping, member_mapping)
                # chunk pixels included in training are set as True
                pixel_mask[indices1d] = True
                # keep track of pixels included in training any member
                untrained_pixel_mask[indices1d] = False
                # For pixels where pixel_mask=False (pixels included in training this member) 
                # then get the Mean over SELECTED MEMBERS (member_mask=True) excluding the one False member (the loop member)
                trained_pixels_ts = members_sim[member_mask, :, :][..., pixel_mask] # select 99 members of training pixel
                trained_pixels_iqr = np.percentile(trained_pixels_ts, 75, axis=0) - np.percentile(trained_pixels_ts, 25, axis=0)  # IQR
                trained_pixels_iqr = np.nanmean(trained_pixels_iqr, axis=0)  # Mean IQR over time
                mean_iqr[pixel_mask] = trained_pixels_iqr
                
                trained_pixels_var = np.var(trained_pixels_ts, axis=0)
                trained_pixels_var = np.nanmean(trained_pixels_var, axis=0)
                mean_var[pixel_mask] = trained_pixels_var
                
            # For other pixels (not included in training any member) get the Mean over ALL MEMBERS (axis=0)
            untrained_pixels_ts = members_sim[..., untrained_pixel_mask]

            untrained_pixels_iqr = np.percentile(untrained_pixels_ts, 75, axis=0) - np.percentile(untrained_pixels_ts, 25, axis=0)  # IQR
            untrained_pixels_iqr = np.nanmean(untrained_pixels_iqr, axis=0)  # Mean IQR over time
            untrained_pixels_var = np.var(untrained_pixels_ts, axis=0)
            untrained_pixels_var = np.nanmean(untrained_pixels_var, axis=0)
                
            mean_iqr[untrained_pixel_mask] = untrained_pixels_iqr
            mean_var[untrained_pixel_mask] = untrained_pixels_var
            indices1d, indices2d = utils.intersect_subsets(target_mapping, rollsubset)
            iqrmap[indices2d] = mean_iqr[indices1d]
            varmap[indices2d] = mean_var[indices1d]
            #print(f"saved target {target}")

        print("] Done!", flush=True)
        bias2d_statspred = 0.45*(iqrmap**1.03)
        rmse2d_statspred = 0.67*(varmap**0.45)
        # ----- create combined figure -----
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(7.09, 2.76),
            subplot_kw={'projection': projection}
        )

        # (a) RMSE
        plot_functions.EU_2Dmap_predictedaccfromstats(
            data_map=rmse2d_statspred,
            logscale=True,
            minval=0.1,
            maxval=10,
            title="RMSE",
            ax=axes[0],
            panel_label="(A)"
        )

        # (b) Absolute mean bias
        plot_functions.EU_2Dmap_predictedaccfromstats(
            data_map=bias2d_statspred,
            logscale=True,
            minval=0.1,
            maxval=10,
            title="Absolute mean bias",
            ax=axes[1],
            panel_label="(B)"
        )

        plt.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.95,
            bottom=0.05,
            wspace=0.08
        )

        plt.savefig(
            os.path.join(OUTPUTPATH, "figure12.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(OUTPUTPATH, "figure12.pdf"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()
    
    def estimate_acc_from_stats_posterversion(self):
        utils = utilities()
        plot_functions = plotting_helper()
        EU_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        EU_outpath = os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean")
        EU_traininpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_400_withcriteria_43226/inputs/20yrs_ts", "ensemble_400px")
        rollsubset = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "mapping_0stdroll6months.npy"))
        iqrmap = np.zeros(rollsubset.shape)
        iqrmap[iqrmap==0] = np.nan
        varmap = np.zeros(rollsubset.shape)
        varmap[varmap==0] = np.nan
        print("Progress: [" + "." * 100 + "]", flush=True)
        print("          [", end="", flush=True)
        for target in range(100):
            print(".", end="", flush=True)  # Dots without newlines
            target_mapping = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            #obs_destand_EU = np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_1", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
            mean_iqr = np.zeros((members_sim.shape[2]))
            mean_var = np.zeros((members_sim.shape[2]))
            untrained_pixel_mask = np.ones(members_sim.shape[2], dtype=bool)
            for member in range(100):
                member_mask = np.ones(100, dtype=bool)
                member_mask[member] = False
                pixel_mask = np.zeros(members_sim.shape[2], dtype=bool)
                # get the 2d mapping file of indices of pixels used in training this member
                member_mapping = np.load(os.path.join(EU_traininpath, f"400px_member_{member}", "choices.npy"))
                # call function here to find which pixels of my target chunk were included in the training of this specific member
                indices1d, indices2d = utils.intersect_subsets(target_mapping, member_mapping)
                # chunk pixels included in training are set as True
                pixel_mask[indices1d] = True
                # keep track of pixels included in training any member
                untrained_pixel_mask[indices1d] = False
                # For pixels where pixel_mask=False (pixels included in training this member) 
                # then get the Mean over SELECTED MEMBERS (member_mask=True) excluding the one False member (the loop member)
                trained_pixels_ts = members_sim[member_mask, :, :][..., pixel_mask] # select 99 members of training pixel
                trained_pixels_iqr = np.percentile(trained_pixels_ts, 75, axis=0) - np.percentile(trained_pixels_ts, 25, axis=0)  # IQR
                trained_pixels_iqr = np.nanmean(trained_pixels_iqr, axis=0)  # Mean IQR over time
                mean_iqr[pixel_mask] = trained_pixels_iqr
                
                trained_pixels_var = np.var(trained_pixels_ts, axis=0)
                trained_pixels_var = np.nanmean(trained_pixels_var, axis=0)
                mean_var[pixel_mask] = trained_pixels_var
                
            # For other pixels (not included in training any member) get the Mean over ALL MEMBERS (axis=0)
            untrained_pixels_ts = members_sim[..., untrained_pixel_mask]

            untrained_pixels_iqr = np.percentile(untrained_pixels_ts, 75, axis=0) - np.percentile(untrained_pixels_ts, 25, axis=0)  # IQR
            untrained_pixels_iqr = np.nanmean(untrained_pixels_iqr, axis=0)  # Mean IQR over time
            untrained_pixels_var = np.var(untrained_pixels_ts, axis=0)
            untrained_pixels_var = np.nanmean(untrained_pixels_var, axis=0)
                
            mean_iqr[untrained_pixel_mask] = untrained_pixels_iqr
            mean_var[untrained_pixel_mask] = untrained_pixels_var
            indices1d, indices2d = utils.intersect_subsets(target_mapping, rollsubset)
            iqrmap[indices2d] = mean_iqr[indices1d]
            varmap[indices2d] = mean_var[indices1d]
            #print(f"saved target {target}")

        print("] Done!", flush=True)
        bias2d_statspred = 0.45*(iqrmap**1.03)
        rmse2d_statspred = 0.67*(varmap**0.45)
        # ----- create combined figure -----
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(7.09, 2.76),
            subplot_kw={'projection': projection}
        )

        # (a) RMSE
        plot_functions.EU_2Dmap_predictedaccfromstats(
            data_map=rmse2d_statspred,
            logscale=True,
            minval=0.1,
            maxval=10,
            title="RMSE",
            ax=axes[0],
            panel_label="(A)"
        )

        # (b) RMSE from obs
        rmse2d_predvsobs = self.metrics_2D_EU()
        plot_functions.EU_2Dmap_predictedaccfromstats(
            data_map=rmse2d_predvsobs,
            logscale=True,
            minval=0.1,
            maxval=10,
            title="RMSE",
            ax=axes[1],
            panel_label="(B)"
        )

        plt.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.95,
            bottom=0.05,
            wspace=0.08
        )


        plt.savefig(
            os.path.join(OUTPUTPATH, "figure12posterversion.png"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

    def plot_map_selectedpixels(self):
        plot_functions = plotting_helper()
        EU_traininpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_400_withcriteria_43226/inputs/20yrs_ts", "ensemble_400px")
        mappingfile = os.path.join(EU_traininpath, f"400px_member_0", "choices.npy")
        plot_functions.selectedpixels_in_EU(mappingpath=mappingfile)

    def wtd_vs_acc(self):
        utils = utilities()
        EU_validation = "validation_400_withcriteria_43226"
        EU_trian = "validation_400_withcriteria_43226"
        EU_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        EU_outpath = os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean")
        EU_train_inpath = os.path.join(f"/p/project1/cslts/miaari1/python_scripts/fork/{EU_trian}/inputs/20yrs_ts/ensemble_400px", "target_pixels")
        transfer_subset = np.load(os.path.join(EU_train_inpath, "transfer_subset.npy"))
        topo = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "topo.npy"))

        #transfer_subset = np.load(os.path.join(os.path.dirname(os.path.dirname(EU_inpath)), "transfer_subset.npy"))
        df_dic = {"stdsim":[], "stdobs":[], "meansim":[], "meanobs":[], "Absolute mean bias":[], "Pearson correlation":[], "RMSE":[], "KGE":[], "KGE'":[], "Beta":[], "Alpha":[], "NSE":[], "Pairwise correlation":[], "Ensemble variance":[], "IQR (75-25%)":[], "std":[], "cv":[],  "(Alpha-1)^2":[], "(Beta-1)^2":[], "(r-1)^2":[]}
        x_axis = []
        y_axis = []
        for target in range(100):
            target_map = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            transfer_indices, ind2d = utils.intersect_subsets(target_map, transfer_subset)
            obs = np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_1", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            obs = obs[:, transfer_indices] # keep only transfer pixels
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
            # NOTE we do this acc and statistics relationship only for transfer pixels because
            # 1- this case we plot 3226 points, for all 43226 pixels we will have a black box with a red fitting line
            # 2- it doesn't make a difference to take only 3226 or 43226 or even a 100 representative pixels, we should get the same fitting line
            members_sim = members_sim[:,:,transfer_indices] # keep only transfer pixels
            obs[obs < 0.0] = 0.0
            members_sim[members_sim < 0.0] = 0.0
            print(f"number of pixels: {members_sim.shape[2]} in target: {target}")
            for pixel in range(members_sim.shape[2]):
                ensemble_predictions = members_sim[:,:,pixel]
                ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
                # Calculate the mean prediction for each time step
                mean_prediction = np.mean(ensemble_predictions, axis=1)
                if np.std(obs[:,pixel])==0.0 or np.std(mean_prediction)==0.0 or np.isnan(obs[:,pixel]).all() or np.isnan(mean_prediction).all():
                    continue
                
                # Calculate the variance for each time step
                ensemble_variance = np.var(ensemble_predictions, axis=1)
                ensemble_variance = np.mean(ensemble_variance)
                df_dic["Ensemble variance"].append(ensemble_variance)

                ########### Calculate accuracy metrics ###########
                # Calculate the correlation between the mean prediction and observation
                correlationobs = np.corrcoef(mean_prediction, obs[:,pixel])[0, 1]
                df_dic["Pearson correlation"].append(correlationobs)

                # Calculate the RMSE between the mean prediction and observation
                rmse = np.sqrt(np.mean((obs[:,pixel] - mean_prediction) ** 2))
                df_dic["RMSE"].append(rmse)

                # Calculate KGE
                kge = utils.calculate_kge(obs[:,pixel], mean_prediction)
                df_dic["KGE"].append(kge)
                
                ### KGE terms analysis ####
                # Compute mean and standard deviation
                mu_o, mu_p = np.mean(obs[:,pixel]), np.mean(mean_prediction)
                sigma_o, sigma_p = np.std(obs[:,pixel]), np.std(mean_prediction)

                ### KGE' ###
                kgeprime = utils.kge_prime(obs[:,pixel], mean_prediction)
                df_dic["KGE'"].append(kgeprime)
                
                # Compute bias ratio (β) and variability ratio (γ)
                beta = mu_p / mu_o
                alpha = sigma_p / sigma_o
                gamma = (np.std(mean_prediction) / np.mean(mean_prediction)) / (np.std(obs[:,pixel]) / np.mean(obs[:,pixel]))  # variability ratio
                df_dic["stdsim"].append(np.std(mean_prediction))
                df_dic["stdobs"].append(np.std(obs[:,pixel]))
                df_dic["meansim"].append(np.mean(mean_prediction))
                df_dic["meanobs"].append(np.mean(obs[:,pixel]))
                
                #if np.std(obs[:,pixel]) < 0.1: # if the std is close to zero, exclude pixel kge
                #alpha = np.nan
                #beta = np.nan
                
                df_dic["Beta"].append(beta)
                df_dic["Alpha"].append(alpha)
                df_dic["(Alpha-1)^2"].append((alpha-1)**2)
                df_dic["(Beta-1)^2"].append((beta-1)**2)
                df_dic["(r-1)^2"].append((correlationobs-1)**2)


                # Calculate NSE
                nse = 1 - (np.sum((obs[:,pixel] - mean_prediction) ** 2) / np.sum((obs[:,pixel] - np.mean(obs[:,pixel])) ** 2))
                #nsecomp = np.sum((obs[:,pixel] - np.mean(obs[:,pixel])) ** 2)
                #if np.std(obs[:,pixel]) < 0.1:
                #    nse = np.nan
                df_dic["NSE"].append(nse)

                # Calculate mean bias
                bias_mean = np.mean(mean_prediction - obs[:,pixel])
                df_dic["Absolute mean bias"].append(bias_mean)
                
                x_axis.append(np.mean(obs[:,pixel]))
                y_axis.append(np.absolute(bias_mean))

        plt.figure()
        plt.scatter(x_axis, y_axis, alpha=0.5)
        plt.xlabel(r'$Mean WTD_O (m)$')
        plt.ylabel(r'Mean Absolute Bias (m)')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(OUTPUTPATH)), EU_validation, "statistics", "mean_vs_MAB.png"), dpi=300, bbox_inches='tight')

    def plot_avg_wtd(self):
        plot_functions = plotting_helper()
        # wtd = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "topo.npy"))
        # print(wtd.shape)
        wtd = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtd.npy"))
        wtd = np.mean(wtd[-365:,:,:], axis=0)

        # exclude sides
        wtd[:100,:] = 0
        wtd[432-10:,:] = 0
        wtd[:,444-10:] = 0
        wtd[:,:10] = 0

        # set negative wtd to 0
        wtd[wtd<0] = 0

        #wtd = wtd[0,:,:]
        print(wtd.shape)
        maxwtd = np.nanmax(wtd)
        print(maxwtd)
        wtd[wtd==0] = np.nan
        plot_functions.EU_2Dmap(wtd, logscale=True, minval=0.01, maxval=maxwtd, title="Mean water table depth")
    
    def plot_wtd_topo_combined(self):
        plot_functions = plotting_helper()

        base_path = os.path.dirname(os.path.dirname(INPUTPATH))

        # --- Load Topography ---
        topo = np.load(os.path.join(base_path, "topo.npy"))
        topo = topo[0, :, :]
        topo[:100,:] = 0
        topo[432-10:,:] = 0
        topo[:, 444-10:] = 0
        topo[:, :10] = 0

        topo[topo <= 0] = np.nan
        max_topo = np.nanmax(topo)

        # --- Load WTD ---
        wtd = np.load(os.path.join(base_path, "wtd.npy"))
        print(wtd.shape)
        wtd = np.mean(wtd[-365:, :, :], axis=0)
        print(wtd.shape)
        # exclude edges
        wtd[:100,:] = 0
        wtd[432-10:,:] = 0
        wtd[:, 444-10:] = 0
        wtd[:, :10] = 0

        wtd[wtd < 0] = 0
        wtd[wtd == 0] = np.nan
        max_wtd = np.nanmax(wtd)

        # --- Figure with 2 subplots ---
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(7.09, 2.76),
            subplot_kw={'projection': projection}
        )

        # (a) WTD
        plot_functions.EU_2Dmap_wtdtopo(
            data_map=wtd,
            logscale=True,
            minval=0.01,
            maxval=max_wtd,
            title="Mean water table depth",
            ax=axes[0],
            panel_label="(A)"
        )

        # (b) Topography
        plot_functions.EU_2Dmap_wtdtopo(
            data_map=topo,
            logscale=False,
            minval=0.01,
            maxval=max_topo,
            title="Topography",
            ax=axes[1],
            panel_label="(B)"
        )

        plt.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.95,
            bottom=0.05,
            wspace=0.08
        )

        plt.savefig(
            os.path.join(OUTPUTPATH, "figureS7.eps"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(OUTPUTPATH, "figureS7.pdf"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.savefig(
            os.path.join(OUTPUTPATH, "figureS7.png"),
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

    def plot_wtd_variance(self):
        plot_functions = plotting_helper()
        wtd = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtd.npy"))
        mapping = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "mapping_0stdroll6months.npy"))
        print(wtd.shape)
        print(np.mean(wtd))
        print(np.sum(mapping))
        # exclude sides
        wtd[:, :100,:] = 0
        wtd[:, 432-10:,:] = 0
        wtd[:, :,444-10:] = 0
        wtd[:, :,:10] = 0

        # set negative wtd to 0
        wtd[wtd<0] = 0

        # calculate variance over time
        var = np.std(wtd, axis=0)
        # set pixels not in mapping to nan
        var[mapping==0] = np.nan
        # wtd = wtd[0,:,:]
        print(var.shape)
        maxvar = np.nanmax(var)
        print(maxvar)
        print(np.nanmin(var))
        # var[var==0] = np.nan
        plot_functions.EU_2Dmap(var, logscale=True, minval=0.1, maxval=maxvar, title=r"$\sigma_{WTD}$")

    def histogram_kgense(self):
        plot_functions = plotting_helper()
        EU_filtering = "validation_400_withcriteria_43226"
        stat = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(OUTPUTPATH)), EU_filtering, "statistics", "ensemble_statistics.csv"))
        
        ev = stat["Ensemble variance"].values
        kge = stat["KGE"].values
        nse = stat["NSE"].values
        print(ev.shape)
        print(kge.shape)
        print(nse.shape)
        ev_kge = ev[kge<0.2]
        kge = kge[kge<0.2]
        
        print(ev_kge.shape)
        print(kge.shape)
        ev_nse = ev[nse<0.2]
        nse = nse[nse<0.2]
        print(ev_nse.shape)
        print(nse.shape)

        plot_functions.logscales_histogram(ev_kge, "Ensemble variance", "KGE < 0.2", nbbins=30, title="Ensemble variance vs KGElessthan02")
        plot_functions.logscales_histogram(ev_nse, "Ensemble variance", "NSE < 0.2", nbbins=30, title="Ensemble variance vs NSElessthan02")

    def kgecomponents_vs_EV(self):
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics_mad.csv"))
        
        ev = stat["Ensemble variance"].values
        kge = stat["KGE"].values
        alpha = stat["Alpha"].values
        beta = stat["Beta"].values
        r = stat["Pearson correlation"].values
        alpha_comp = stat["(Alpha-1)^2"].values
        beta_comp = stat["(Beta-1)^2"].values
        r_comp = stat["(r-1)^2"].values

        ev = ev[kge<0.2]
        alpha_comp = alpha_comp[kge<0.2]
        beta_comp = beta_comp[kge<0.2]
        r_comp = r_comp[kge<0.2]

        plt.figure(figsize=(3.35, 2.51))
        plt.scatter(ev, alpha_comp, color='#0072B2', alpha=0.4, label=r'$(\alpha-1)^2$')
        plt.scatter(ev, beta_comp, color='#009E73', alpha=0.4, label=r'$(\beta-1)^2$')
        plt.scatter(ev, r_comp, color='#D55E00', alpha=0.4, label=r'$(r-1)^2$')

        # Decorations
        print("plotting it")
        plt.xlabel(r"$\overline{EV}$")
        plt.ylabel("KGE component")
        plt.legend(fontsize=18)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "figureA2.eps"), dpi=300)
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "figureA2.pdf"), dpi=300)
    
    def inputvars_plot(self):
        plot_functions = plotting_helper()
        mapping = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "mapping_0stdroll6months.npy"))
        vars = FEATURES_FILES
        vars.append(TARGETVAR_FILE)
        for var in vars:
            print(var)
            data = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), var))
            print(data.shape)
            data = np.mean(data, axis=0)
            print(data.shape)
            data[mapping==0] = np.nan
            data = data[100:-10,10:-10]
            print(data.shape)
            logscale = True if var in ["wtd.npy", "vpd.npy"] else False
            minval = 0.01 if logscale else np.nanmin(data)
            plot_functions.EU_2Dmap_inputvars(data_map=data, logscale=logscale, minval=minval, maxval=np.nanmax(data), title=var.replace(".npy", ""))

    def wtd_timeseries_plot(self):
        mapping = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "mapping_0stdroll6months.npy"))
        data = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtd.npy"))
        data = data[365:,mapping==1]
        data = data[:, 234] # select one pixel with mapping=1
        
        dates = pd.date_range(start='2002-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]

        # Aggregate daily values to monthly means before plotting.
        monthly_mean = pd.Series(data, index=dates).resample('MS').mean()

        fig, ax = plt.subplots()
        ax.plot(monthly_mean.index, monthly_mean.values, color='k')
        ax.xaxis.set_major_locator(mdates.MonthLocator([1]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_ylabel("Water table depth (m)")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(OUTPUTPATH, "wtd_timeseries.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def check_equal_files(self):
        dir1 = "/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/inputs/20yrs_ts/ensemble_100px"
        dir2 = "/p/project1/cslts/miaari1/python_scripts/fork/train_100_withcriteria_43226/inputs/20yrs_ts/ensemble_100px"
        file1 = np.load(os.path.join(dir1, "target_pixels_75", "sim_testtrainpixels_75.npy"))
        file2 = np.load(os.path.join(dir2, "target_pixels_75", "sim_testtrainpixels_75.npy"))
        file3 = np.load(os.path.join(dir2, "target_pixels_75", "sim_testtrainpixels_75_old.npy"))
        file1 = np.nan_to_num(file1)
        file2 = np.nan_to_num(file2)
        file3 = np.nan_to_num(file3)
        print(np.unique(np.equal(file1, file2), return_counts=True))
        print(np.unique(np.equal(file1, file3), return_counts=True))
        print(np.unique(np.equal(file2, file3), return_counts=True))
        print(file1)
        print(file2)

    def delete_old_files(self):
        dirpath = "/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/inputs/20yrs_ts/ensemble_400px"
        for i in range(100):
            print(i)
            targetpath = os.path.join(dirpath, f"target_pixels_{i}")
            if os.path.exists(os.path.join(targetpath, f"sim_destand_{MODEL_NAME}_{i}.npy")):
                os.remove(os.path.join(targetpath, f"sim_destand_{MODEL_NAME}_{i}.npy"))
            if os.path.exists(os.path.join(targetpath, f"obs_destand_{MODEL_NAME}_{i}.npy")):
                os.remove(os.path.join(targetpath, f"obs_destand_{MODEL_NAME}_{i}.npy"))
            if os.path.exists(os.path.join(targetpath, f"obs_testtrainpixels_{i}.npy")):
                os.remove(os.path.join(targetpath, f"obs_testtrainpixels_{i}.npy"))
    
    def move_old_files(self):
        import shutil
        inpath = os.path.dirname(OUTPUTPATH)
        outpath = "/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/outputs/20yrs_ts/ensemble_400px"
        for i in range(100):
            print(i)
            ####### from /outputs/20yrs_ts/ensemble_400px ####
            if os.path.exists(os.path.join(inpath, f"crps_100members_onlytransfer.npy")):
                shutil.move(os.path.join(inpath, f"crps_100members_onlytransfer.npy"), os.path.join(outpath, f"crps_100members_onlytransfer.npy"))
                print("moved crps_100members_onlytransfer")
            ####### from /inputs/20yrs_ts/ensemble_400px ####
            intargetpath = os.path.join(inpath, f"target_pixels_{i}")
            if os.path.exists(os.path.join(intargetpath, f"obs_{i}.npy")):
                shutil.move(os.path.join(intargetpath, f"obs_{i}.npy"), os.path.join(outpath, f"target_pixels_{i}", f"obs_{i}.npy"))
                print("moved obs")
            if os.path.exists(os.path.join(intargetpath, f"obs_ts_trainpixels_{i}.npy")):
                shutil.move(os.path.join(intargetpath, f"obs_ts_trainpixels_{i}.npy"), os.path.join(outpath, f"target_pixels_{i}", f"obs_ts_trainpixels_{i}.npy"))
                print("moved obs_ts_trainpixels")
            if os.path.exists(os.path.join(intargetpath, f"obs_ts_transferpixels_{i}.npy")):
                shutil.move(os.path.join(intargetpath, f"obs_ts_transferpixels_{i}.npy"), os.path.join(outpath, f"target_pixels_{i}", f"obs_ts_transferpixels_{i}.npy"))
                print("moved obs_ts_transferpixels")

            if os.path.exists(os.path.join(intargetpath, f"sim_100members_ts_transferpixels_{i}.npy")):
                shutil.move(os.path.join(intargetpath, f"sim_100members_ts_transferpixels_{i}.npy"), os.path.join(outpath, f"target_pixels_{i}", f"sim_100members_ts_transferpixels_{i}.npy"))
                print("moved sim_100members_ts_transferpixels")
            if os.path.exists(os.path.join(intargetpath, f"sim_99members_ts_trainpixels_{i}.npy")):
                shutil.move(os.path.join(intargetpath, f"sim_99members_ts_trainpixels_{i}.npy"), os.path.join(outpath, f"target_pixels_{i}", f"sim_99members_ts_trainpixels_{i}.npy"))
                print("moved sim_99members_ts_trainpixels")

            if os.path.exists(os.path.join(intargetpath, f"sim_testtrainpixels_{i}.npy")):
                shutil.move(os.path.join(intargetpath, f"sim_testtrainpixels_{i}.npy"), os.path.join(outpath, f"target_pixels_{i}", f"sim_testtrainpixels_{i}.npy"))
                print("moved sim_testtrainpixels")

            if os.path.exists(os.path.join(intargetpath, f"sim_transferpixels_{i}.npy")):
                shutil.move(os.path.join(intargetpath, f"sim_transferpixels_{i}.npy"), os.path.join(outpath, f"target_pixels_{i}", f"sim_transferpixels_{i}.npy"))
                print("moved sim_transferpixels")

    def relative_metrics_difference_transfer_local(self):
        def calc_correlation(obs, sim):
            correlation_map = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    correlation_matrix = np.corrcoef(time_series1, time_series2)
                    r = correlation_matrix[0, 1]
                else:
                    r = np.nan
                correlation_map.append(r)
            return correlation_map
        
        def calc_RMSE(obs, sim):
            RMSE_allcells = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    rmse = np.sqrt(np.mean((time_series1 - time_series2) ** 2))
                else:
                    rmse = np.nan
                RMSE_allcells.append(rmse)
            return RMSE_allcells
        
        def calc_KGE_NSE_bias(obs, sim):
            all_kge = []
            all_nse = []
            all_bias = []
            for pixel in range(obs.shape[1]):
                time_series1 = obs[:, pixel]
                time_series2 = sim[:, pixel]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    kge = utils.calculate_kge(time_series1, time_series2)# if np.std(time_series1)>=0.1 else np.nan
                    nse = 1 - (np.sum((time_series1 - time_series2) ** 2) / np.sum((time_series1 - np.mean(time_series1)) ** 2))# if np.std(time_series1)>=0.1 else np.nan
                    bias_mean = np.mean(time_series2 - time_series1)
                else:
                    kge = np.nan
                    nse = np.nan
                    bias_mean = np.nan
                all_kge.append(kge)
                all_nse.append(nse)
                all_bias.append(bias_mean)
            return all_kge, all_nse, all_bias

        def load_obs_sim(dirpath, target, obsname, simname):
            obs = np.load(os.path.join(os.path.dirname(dirpath), f"target_pixels_{target}", obsname))
            sims = np.load(os.path.join(os.path.dirname(dirpath), f"target_pixels_{target}", simname))
            obs = np.nan_to_num(obs)
            sims = np.nan_to_num(sims)
            obs[obs < 0.0] = 0.0
            sims[sims < 0.0] = 0.0
            return obs, sims

        utils = utilities()
        print("starting the cdf calculation")
        correlation = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        rmse = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        kge = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        nse = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        bias = {"transfer_400px":[], "test_400px":[], "transfer_100px":[], "test_100px":[]}
        
        #### Transfer & training filtered subsets ####
        EU400px_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        EU100px_inpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_100_withcriteria_43226/inputs/20yrs_ts/ensemble_100px", "ensemble_mean")
        training_subset = np.load(os.path.join(os.path.dirname(os.path.dirname(EU400px_inpath)), "ensemble_400px", "target_pixels", "training_subset.npy"))
        training_subset_100px = np.load(os.path.join(os.path.dirname(EU100px_inpath), "target_pixels", "training_subset_ensemble100px.npy"))
        print("Progress: [" + "." * 100 + "]", flush=True)
        print("          [", end="", flush=True)  # Start progress bar            
        for target in range(100):
            print(".", end="", flush=True)  # Dots without newlines
            ######## 400px ensemble ########
            # calculate transfer metrics for 400px ensemble
            obs_transfer400, sim_transfer400 = load_obs_sim(EU400px_inpath, target, f"obs_{target}.npy", f"sim_transferpixels_{target}.npy")
            obs_test400, sim_test400 = load_obs_sim(EU400px_inpath, target, f"obs_{target}.npy", f"sim_testtrainpixels_{target}.npy")
            target_map = np.load(os.path.join(os.path.dirname(EU400px_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            training400px_indices, ind2d = utils.intersect_subsets(target_map, training_subset)
            corr_EU = calc_correlation(obs_transfer400, sim_transfer400)
            rmse_EU = calc_RMSE(obs_transfer400, sim_transfer400)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_transfer400, sim_transfer400)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from 99 transfer members (training subset) + 100 transfer members (transfer subset)
            correlation["transfer_400px"].extend(corr_EU.tolist())
            rmse["transfer_400px"].extend(rmse_EU.tolist())
            kge["transfer_400px"].extend(kge_EU.tolist())
            nse["transfer_400px"].extend(nse_EU.tolist())
            bias["transfer_400px"].extend(bias_EU.tolist())

            # calculate test metrics for 400px ensemble
            corr_EU = calc_correlation(obs_test400, sim_test400)
            rmse_EU = calc_RMSE(obs_test400, sim_test400)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_test400, sim_test400)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from only 1 training member (training subset)
            correlation["test_400px"].extend(corr_EU[training400px_indices].tolist())
            rmse["test_400px"].extend(rmse_EU[training400px_indices].tolist())
            kge["test_400px"].extend(kge_EU[training400px_indices].tolist())
            nse["test_400px"].extend(nse_EU[training400px_indices].tolist())
            bias["test_400px"].extend(bias_EU[training400px_indices].tolist())

            ######## 100px ensemble ########
            # calculate transfer metrics for 100px ensemble
            obs_transfer100, sim_transfer100 = load_obs_sim(EU100px_inpath, target, f"obs_{target}.npy", f"sim_transferpixels_{target}.npy")
            obs_test100, sim_test100 = load_obs_sim(EU100px_inpath, target, f"obs_{target}.npy", f"sim_testtrainpixels_{target}.npy")
            target_map = np.load(os.path.join(os.path.dirname(EU100px_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            training100px_indices, ind2d = utils.intersect_subsets(target_map, training_subset_100px)
            corr_EU = calc_correlation(obs_transfer100, sim_transfer100)
            rmse_EU = calc_RMSE(obs_transfer100, sim_transfer100)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_transfer100, sim_transfer100)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from 99 transfer members (training subset) + 100 transfer members (transfer subset)
            correlation["transfer_100px"].extend(corr_EU.tolist())
            rmse["transfer_100px"].extend(rmse_EU.tolist())
            kge["transfer_100px"].extend(kge_EU.tolist())
            nse["transfer_100px"].extend(nse_EU.tolist())
            bias["transfer_100px"].extend(bias_EU.tolist())

            # calculate test metrics for 100px ensemble
            corr_EU = calc_correlation(obs_test100, sim_test100)
            rmse_EU = calc_RMSE(obs_test100, sim_test100)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_test100, sim_test100)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from only 1 training member (training subset)
            correlation["test_100px"].extend(corr_EU[training100px_indices].tolist())
            rmse["test_100px"].extend(rmse_EU[training100px_indices].tolist())
            kge["test_100px"].extend(kge_EU[training100px_indices].tolist())
            nse["test_100px"].extend(nse_EU[training100px_indices].tolist())
            bias["test_100px"].extend(bias_EU[training100px_indices].tolist())
        
        print("] Done!", flush=True)
        # cleanup
        for key in correlation.keys():
            correlation[key] = np.array(correlation[key])
            correlation[key] = correlation[key][~np.isnan(correlation[key])]

        for key in rmse.keys():
            rmse[key] = np.array(rmse[key])
            rmse[key] = rmse[key][~np.isnan(rmse[key])]

        for key in kge.keys():
            kge[key] = np.array(kge[key])
            kge[key] = kge[key][~np.isnan(kge[key])]

        for key in nse.keys():
            nse[key] = np.array(nse[key])
            nse[key] = nse[key][~np.isnan(nse[key])]

        for key in bias.keys():
            bias[key] = np.array(bias[key])
            bias[key] = bias[key][~np.isnan(bias[key])]

        print("calculating relative difference")
        # calculate relative difference 400px test vs transfer
        relative_difference = {}
        for metric_dict, metric_name in zip([correlation, rmse, kge, nse, bias],
                            ["correlation", "rmse", "kge", "nse", "bias"]):
            relative_difference[metric_name] = {}
            transfer_400px_mean = np.nanmean(metric_dict["transfer_400px"])
            test_400px_mean = np.nanmean(metric_dict["test_400px"])
            transfer_100px_mean = np.nanmean(metric_dict["transfer_100px"])
            test_100px_mean = np.nanmean(metric_dict["test_100px"])
            relative_difference[metric_name]["transfer_vs_test_400px"] = (test_400px_mean - transfer_400px_mean) / transfer_400px_mean * 100
            relative_difference[metric_name]["transfer_vs_test_100px"] = (test_100px_mean - transfer_100px_mean) / transfer_100px_mean * 100
            relative_difference[metric_name]["transfer_400px_vs_100px"] = (transfer_100px_mean - transfer_400px_mean) / transfer_100px_mean * 100
            relative_difference[metric_name]["test_400px_vs_100px"] = (test_100px_mean - test_400px_mean) / test_100px_mean * 100
            print(f"{metric_name} relative difference test vs transfer 400px: {relative_difference[metric_name]['transfer_vs_test_400px']:.2f} %")
            print(f"{metric_name} relative difference test vs transfer 100px: {relative_difference[metric_name]['transfer_vs_test_100px']:.2f} %")
            print(f"{metric_name} relative difference transfer 400px vs 100px: {relative_difference[metric_name]['transfer_400px_vs_100px']:.2f} %")
            print(f"{metric_name} relative difference test 400px vs 100px: {relative_difference[metric_name]['test_400px_vs_100px']:.2f} %")

    def plot2dmap_trainingpixels(self):
        plot_functions = plotting_helper()
        dirpath = os.path.join(os.path.dirname(get_root_dir()), "fork", "train_400_withcriteria_43226", "inputs", "20yrs_ts", "ensemble_400px")
        for i in range(100):
            print(i)
            choicespath = os.path.join(dirpath, f"400px_member_{i}", "choices.npy")
            plot_functions.selectedpixels_in_EU(choicespath, i)
    
    def ensemble_statvsacc_anomalies(self):
        utils = utilities()
        EU_validation = "validation_400_withcriteria_43226"
        EU_trian = "validation_400_withcriteria_43226"
        EU_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
        EU_outpath = os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean")
        EU_train_inpath = os.path.join(f"/p/project1/cslts/miaari1/python_scripts/fork/{EU_trian}/inputs/20yrs_ts/ensemble_400px", "target_pixels")
        transfer_subset = np.load(os.path.join(EU_train_inpath, "transfer_subset.npy"))

        #transfer_subset = np.load(os.path.join(os.path.dirname(os.path.dirname(EU_inpath)), "transfer_subset.npy"))
        df_dic = {"mean_membersobs_RMSE":[], "mean_membersobs_correlation":[], "mean_membersobs_KGE":[], "KGE_nobias":[], "stdsim":[], "stdobs":[], "meansim":[], "meanobs":[], "Absolute mean bias":[], "Pearson correlation":[], "RMSE":[], "KGE":[], "KGE'":[], "Beta":[], "Alpha":[], "NSE":[], "Pairwise correlation":[], "Ensemble variance":[], "IQR (75-25%)":[], "std":[], "cv":[], "mad":[],  "(Alpha-1)^2":[], "(Beta-1)^2":[], "(r-1)^2":[]}
        x_axis = []
        y_axis = []
        for target in range(100):
            target_map = np.load(os.path.join(os.path.dirname(EU_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            transfer_indices, ind2d = utils.intersect_subsets(target_map, transfer_subset)
            obs = np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_1", f"obs_destand_{MODEL_NAME}_{target}.npy"))
            obs = obs[:, transfer_indices] # keep only transfer pixels
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
            # NOTE we do this acc and statistics relationship only for transfer pixels because
            # 1- this case we plot 3226 points, for all 43226 pixels we will have a black plot with a red fitting line
            # 2- it doesn't make a difference to take only 3226 or 43226 or even a 100 representative pixels, we should get the same fitting line, it should be representative
            members_sim = members_sim[:,:,transfer_indices] # keep only transfer pixels
            obs[obs < 0.0] = 0.0 ############## important to set negatives to zero ##############
            members_sim[members_sim < 0.0] = 0.0 ############## important to set negatives to zero ##############

            print(f"number of pixels: {members_sim.shape[2]} in target: {target}")
            for pixel in range(members_sim.shape[2]):
                ensemble_predictions = members_sim[:,:,pixel]
                ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
                # Calculate the mean prediction for each time step
                mean_prediction = np.mean(ensemble_predictions, axis=1)
                obs_pixel = obs[:, pixel]

                def calculate_monthly_anomaly(series):
                    anomaly = np.zeros_like(series, dtype=float)
                    for month in range(12):
                        month_values = series[month::12]
                        month_mean = np.mean(month_values)
                        month_std = np.std(month_values)
                        anomaly[month::12] = np.divide(
                            month_values - month_mean,
                            month_std,
                            out=np.zeros_like(month_values, dtype=float),
                            where=month_std != 0,
                        )
                    return anomaly

                obs_anom = calculate_monthly_anomaly(obs_pixel)
                mean_prediction_anom = calculate_monthly_anomaly(mean_prediction)
                ensemble_predictions_anom = np.array(
                    [calculate_monthly_anomaly(ensemble_predictions[:, m]) for m in range(ensemble_predictions.shape[1])]
                ).T

                if np.std(obs_anom)==0.0 or np.std(mean_prediction_anom)==0.0 or np.isnan(obs_anom).all() or np.isnan(mean_prediction_anom).all():
                    continue

                obs_pixel = obs_anom
                mean_prediction = mean_prediction_anom
                ensemble_predictions = ensemble_predictions_anom
                
                ########### Calculate ensemble statistics ###########
                # Calculate the variance for each time step
                ensemble_variance = np.var(ensemble_predictions, axis=1)
                ensemble_variance = np.mean(ensemble_variance)
                df_dic["Ensemble variance"].append(ensemble_variance)

                # Calculate ensemble statistics (e.g., diversity, spread, etc.)
                # Spread interquantile range (IQR) between 75th and 25th percentiles
                ensemble_iqr = np.percentile(ensemble_predictions, 75, axis=1) - np.percentile(ensemble_predictions, 25, axis=1)  # IQR
                iqr_mean = np.mean(ensemble_iqr)
                df_dic["IQR (75-25%)"].append(iqr_mean)

                # calculate MAD
                mad = utils.calculate_ensemble_mad(ensemble_predictions)
                df_dic["mad"].append(mad)

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
                # Calculate mean bias
                bias_mean = np.mean(mean_prediction - obs_pixel)
                df_dic["Absolute mean bias"].append(bias_mean)

                ### remove bias from mean prediction ###
                # mean_prediction = mean_prediction - bias_mean

                # Calculate the correlation between the mean prediction and observation
                correlationobs = np.corrcoef(mean_prediction, obs_pixel)[0, 1]
                df_dic["Pearson correlation"].append(correlationobs)

                # Calculate the RMSE between the mean prediction and observation
                rmse = np.sqrt(np.mean((obs_pixel - mean_prediction) ** 2))
                df_dic["RMSE"].append(rmse)

                # Calculate KGE
                kge = utils.calculate_kge(obs_pixel, mean_prediction)
                df_dic["KGE"].append(kge)
                
                # members correlation
                members_correlation = [np.corrcoef(ensemble_predictions[:,m], obs_pixel)[0, 1] for m in range(ensemble_predictions.shape[1])]
                members_correlation_mean = np.mean(members_correlation)
                df_dic["mean_membersobs_correlation"].append(members_correlation_mean)
                if members_correlation_mean==correlationobs:
                    raise ValueError("members mean correlation is the same as correlationobs, check your code")

                # members RMSE
                members_rmse = np.mean(np.sqrt(np.mean((ensemble_predictions - obs_pixel[:, None]) ** 2, axis=0)))
                df_dic["mean_membersobs_RMSE"].append(members_rmse)
                if members_rmse==rmse:
                    raise ValueError("members mean RMSE is the same as RMSE, check your code")

                # members KGE
                members_kge = [utils.calculate_kge(ensemble_predictions[:,m], obs_pixel) for m in range(ensemble_predictions.shape[1])]
                members_kge_mean = np.mean(members_kge)
                df_dic["mean_membersobs_KGE"].append(members_kge_mean)
                if members_kge_mean==kge:
                    raise ValueError("members mean KGE is the same as KGE, check your code")
                
                ### KGE terms analysis ####
                # Compute mean and standard deviation
                mu_o, mu_p = np.mean(obs_pixel), np.mean(mean_prediction)
                sigma_o, sigma_p = np.std(obs_pixel), np.std(mean_prediction)

                ### KGE' ###
                kgeprime = utils.kge_prime(obs_pixel, mean_prediction)
                df_dic["KGE'"].append(kgeprime)
                
                # Compute bias ratio (β) and variability ratio (γ)
                beta = mu_p / mu_o
                alpha = sigma_p / sigma_o
                kge_nobias = 1 - np.sqrt((correlationobs - 1)**2 + (alpha - 1)**2)

                # Store statistics and accuracy metrics
                df_dic["KGE_nobias"].append(kge_nobias)
                df_dic["stdsim"].append(np.std(mean_prediction))
                df_dic["stdobs"].append(np.std(obs_pixel))
                df_dic["meansim"].append(np.mean(mean_prediction))
                df_dic["meanobs"].append(np.mean(obs_pixel))
                                
                df_dic["Beta"].append(beta)
                df_dic["Alpha"].append(alpha)
                df_dic["(Alpha-1)^2"].append((alpha-1)**2)
                df_dic["(Beta-1)^2"].append((beta-1)**2)
                df_dic["(r-1)^2"].append((correlationobs-1)**2)


                # Calculate NSE
                nse = 1 - (np.sum((obs_pixel - mean_prediction) ** 2) / np.sum((obs_pixel - np.mean(obs_pixel)) ** 2))
                df_dic["NSE"].append(nse)
                
        ####### save to csv ########
        df = pd.DataFrame(df_dic)
        df.to_csv(os.path.join(OUTPUTPATH, "statistics_anom", "ensemble_statistics.csv"), index=False)

    def ensemble_statvsacc_fitting_anom_colored_by_wtd(self):
        utils = utilities()
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics_anom", "ensemble_statistics.csv"))
        xstats = {"std":"exp", "Ensemble variance":"exp", "Pairwise correlation":"lin", "IQR (75-25%)":"exp", "cv":"exp", "mad":"exp"}
        ystats = {"mean_membersobs_correlation": "lin", "mean_membersobs_RMSE": "exp", "mean_membersobs_KGE": "lin", "RMSE":"exp", "Pearson correlation":"lin", "KGE":"lin", "KGE_nobias":"exp", "Absolute mean bias":"exp", "NSE":"lin", "Beta":"exp", "Alpha":"exp", "(Alpha-1)^2":"exp", "(Beta-1)^2":"exp", "(r-1)^2":"exp"}
        plotmapping = {
            "Ensemble variance":r"$\overline{EV}$",
            "IQR (75-25%)":r"$\overline{IQR}$",
            "Beta":r"$\beta$",
            "Alpha":r"$\alpha$",
            "(Alpha-1)^2":r"$(\alpha-1)^2$",
            "(Beta-1)^2":r"$(\beta-1)^2$",
        }
        meanobswtd = stat["Pearson correlation"].values
        
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
                colorval = meanobswtd.copy()

                if ystat=="Pearson correlation" or ystat=="mean_membersobs_correlation":
                    mask = ~np.isnan(yval)
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                    colorval = colorval[mask]
                    #xval = xval[yval>=0.0]
                    #yval = yval[yval>=0.0]
                
                if ystat=="KGE" or ystat=="KGE_nobias" or ystat=="mean_membersobs_KGE":
                    mask = ~np.isnan(yval)
                    xval = xval[mask]
                    yval = yval[mask]
                    colorval = colorval[mask]
                    mask = yval >= -0.4
                    xval = xval[mask]
                    yval = yval[mask]
                    colorval = colorval[mask]
                
                if ystat=="NSE":
                    mask = ~np.isnan(yval)
                    xval = xval[mask]
                    yval = yval[mask]
                    colorval = colorval[mask]
                    mask = yval >= -1
                    xval = xval[mask]
                    yval = yval[mask]
                    colorval = colorval[mask]

                if ystat=="Beta" or ystat=="Alpha" or ystat=="(Alpha-1)^2" or ystat=="(Beta-1)^2" or ystat=="(r-1)^2":
                    mask = ~np.isnan(yval)
                    xval = xval[mask]
                    yval = yval[mask]
                    colorval = colorval[mask]
                
                if ystat=="Absolute mean bias":
                    mask = yval >= 0.01
                    xval = xval[mask]
                    yval = yval[mask]
                    colorval = colorval[mask]
                
                if xstat=="Pairwise correlation":
                    mask = xval >= 0.0
                    yval = yval[mask]
                    xval = xval[mask]
                    colorval = colorval[mask]

                
                logx, logy, xminlim, xmaxlim, yminlim, ymaxlim = None, None, None, None, None, None
                if xstat=="std" or xstat=="Ensemble variance" or xstat=="IQR (75-25%)" or xstat=="cv" or xstat=="mad":
                    logx = True
                if ystat=="RMSE" or ystat=="Absolute mean bias" or ystat=="Alpha" or ystat=="Beta" or ystat=="(Alpha-1)^2" or ystat=="(Beta-1)^2" or ystat=="(r-1)^2" or ystat=="mean_membersobs_RMSE":
                    logy = True
                if xstat=="Pairwise correlation":
                    xminlim = 0
                    xmaxlim = 1
                if ystat=="Pearson correlation":
                    yminlim = -1
                    ymaxlim = 1
                if ystat=="KGE":
                    yminlim = -0.4
                    ymaxlim = 1
                if ystat=="NSE":
                    yminlim = -1
                    ymaxlim = 1                

                if xstat=="Pairwise correlation" and ystat=="Pearson correlation":
                    fittype = "exponential"
                    logarithmicfit = True
                    logx = True

                try:
                    if fittype == "linear":
                        params, covariance = curve_fit(
                            utils.linear_law, xval, yval)
                        a_fit, b_fit = params
                        y_fit = [utils.linear_law(x, a_fit, b_fit)
                                 for x in xval]
                        R_square = r2_score(yval, y_fit)
                        lin_eq = f'$y={b_fit:.2f}x+{a_fit:.2f}$' if a_fit >= 0 else f'$y={b_fit:.2f}x{a_fit:.2f}$'
                        textstr = '\n'.join((
                            lin_eq,
                            f'$R^2$ = {R_square:.3f}'))
                        x_fit = [min(xval), max(xval)]
                        y_fit = [utils.linear_law(x, a_fit, b_fit)
                                 for x in x_fit]

                    if fittype == "powerlaw":
                        y_lin = np.log(yval)
                        x_lin = np.log(xval)
                        params, covariance = curve_fit(
                            utils.linear_law, x_lin, y_lin)
                        a_fit, b_fit = params
                        y_fit = [utils.linear_law(x, a_fit, b_fit)
                                 for x in x_lin]
                        R_square = r2_score(y_lin, y_fit)
                        a_fit = np.exp(a_fit)
                        textstr = '\n'.join((
                            f'y={a_fit:.2f}x^{b_fit:.2f}',
                            f'$R^2$ = {R_square:.3f}'))

                        x_fit = [min(xval), max(xval)]
                        y_fit = [utils.powerlaw_func(
                            x, a_fit, b_fit) for x in x_fit]

                    if fittype == "exponential":
                        def log_func(x, a, b):
                            return a + b * np.log(x)

                        def exp_func(x, a, b):
                            return a * np.exp(b * x)

                        if logarithmicfit:
                            popt, _ = curve_fit(log_func, xval, yval)
                            a_fit, b_fit = popt
                            y_pred = log_func(xval, *popt)
                            r2 = r2_score(yval, y_pred)

                            x_fit = np.linspace(min(xval), max(xval), 100)
                            y_fit = log_func(x_fit, *popt)
                            logeq = f'y={a_fit:.2f}+{b_fit:.3f}log(x)' if b_fit >= 0 else f'y={a_fit:.2f}{b_fit:.3f}log(x)'
                            textstr = '\n'.join((
                                logeq,
                                f'$R^2$ = {r2:.3f}'))
                        else:
                            popt, _ = curve_fit(exp_func, xval, yval, p0=(
                                1, -0.1))
                            a_fit, b_fit = popt
                            y_pred = exp_func(xval, *popt)

                            r2 = r2_score(yval, y_pred)

                            x_fit = np.linspace(min(xval), max(xval), 100)
                            y_fit = exp_func(x_fit, *popt)

                            textstr = '\n'.join((
                                f'y={a_fit:.2f}e^{b_fit:.3f}x',
                                f'$R^2$ = {r2:.3f}'))
                except:
                    print(
                        f"fitting failed for {xstat} with {ystat}, skipping...")
                    continue

                plt.figure()
                plt.plot(x_fit, y_fit, color="r",
                         label=textstr, linestyle='--')
                scatter = plt.scatter(
                    xval,
                    yval,
                    # c=colorval,
                    # cmap="viridis",
                    marker='.',
                    edgecolors='none',
                    # vmin=-1,
                    # vmax=1,
                )
                # plt.colorbar(scatter, label="Pearson correlation")
                if xstat in plotmapping.keys():
                    plt.xlabel(plotmapping[xstat])
                else:
                    plt.xlabel(xstat)
                if ystat in plotmapping.keys():
                    plt.ylabel(plotmapping[ystat])
                else:
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
                plt.legend(frameon=True)
                plt.savefig(os.path.join(OUTPUTPATH, "statistics_anom", f"fitted_{ystat}_{xstat}.png"))
                plt.close()
