import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
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
        plot_functions.chosenpixels_in_EU(mean_bias_2D_map, False, -1, 1, f"Bias")
        plot_functions.chosenpixels_heatmap(mean_bias_2D_heatmap.reshape(X,Y), False, -10, 10, f"Bias")

        # calculate and plot correlation
        correlation_2D_map = self.calc_2D_correlation(obs, sim)
        correlation_2D_heatmap = self.calc_2Dheatmap_correlation(obs_destand_test, sim_destand_test)
        plot_functions.chosenpixels_in_EU(correlation_2D_map, False, 0, 1, f"Correlation")
        plot_functions.chosenpixels_heatmap(correlation_2D_heatmap, False, 0, 1, f"Correlation")

        # calculate and plot MSE
        mse_2D_heatmap = self.calc_2Dheatmap_MSE(obs_destand_test, sim_destand_test)
        mse_2D_map = self.calc_2D_MSE(obs_destand_test, sim_destand_test, mapping)
        plot_functions.chosenpixels_in_EU(mse_2D_map, True, 0.001, 10, f"MSE")
        plot_functions.chosenpixels_heatmap(mse_2D_heatmap, True, 0.001, 10, f"MSE")

        # calculate and plot RMSE
        rmse_2D_heatmap, rmse_2D_map = self.calc_rmse(obs_destand_test, sim_destand_test, mapping)
        plot_functions.chosenpixels_in_EU(rmse_2D_map, True, 0.001, 10, f"RMSE")
        plot_functions.chosenpixels_heatmap(rmse_2D_heatmap, True, 0.001, 10, f"RMSE")

        # calculate and plot NSE
        nse_2D_heatmap, nse_2D_map = self.calc_nse(obs_destand_test, sim_destand_test, mapping)
        nse_2D_heatmap[nse_2D_heatmap<-1] = np.nan
        nse_2D_map[nse_2D_map<-1] = np.nan
        plot_functions.chosenpixels_in_EU(nse_2D_map, False, -1, 1, f"NSE")
        plot_functions.chosenpixels_heatmap(nse_2D_heatmap, False, -1, 1, f"NSE")
        #plot_functions.plot_predr2(obs_destand_test[:,412], sim_destand_test[:,412])

        # calculate and plot KGE
        kge_2D_heatmap = [util.calculate_kge(obs_destand_test[:,i], sim_destand_test[:,i]) for i in range(obs_destand_test.shape[1])]
        kge_2D_heatmap = np.array(kge_2D_heatmap)
        kge_2D_heatmap = kge_2D_heatmap.reshape(X,Y)
        kge_2D_map = np.zeros(mapping.shape)
        kge_2D_heatmap[kge_2D_heatmap<-1] = np.nan
        for i in range(obs.shape[1]):
            for j in range(obs.shape[2]):
                if np.isnan(obs[:,i,j]).all() or np.isnan(sim[:,i,j]).all():
                    kge_2D_map[i,j] = np.nan
                else:
                    kge = util.calculate_kge(obs[:,i,j], sim[:,i,j])
                    kge_2D_map[i,j] = kge if kge>=-1 else np.nan
        plot_functions.chosenpixels_in_EU(kge_2D_map, False, -1, 1, f"KGE")
        plot_functions.chosenpixels_heatmap(kge_2D_heatmap, False, -1, 1, f"KGE")

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

    def plot_ensemble_timeseries(self, i, j):
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        nb_members = 100
        print(obs_destand_test.shape)
        obs_destand_test[obs_destand_test < 0.01] = 0.0

        dates = pd.date_range(start='2017-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))
        
        print(f"plotting timeseries {i},{j}")
        timeserieslength = obs_destand_test.shape[0]
        obs_destand_test = obs_destand_test.reshape(timeserieslength,X,Y)
        ens_mean = np.zeros((timeserieslength, nb_members))
        for m in range(nb_members):
            sim_destand_test = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy"))
            sim_destand_test[sim_destand_test < 0.01] = 0.0
            sim_destand_test = sim_destand_test.reshape(timeserieslength,X,Y)
            ens_mean[:,m] = sim_destand_test[:,i,j]
            ax.plot(dates, sim_destand_test[:,i,j])

        ax.plot(dates, obs_destand_test[:,i,j], "k-", label="Original simulations", linewidth=4.0)
        ax.plot(dates, np.mean(ens_mean, axis=1), "k--", label="Ensemble mean", linewidth=4.0)

        ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(os.path.dirname(OUTPUTPATH), "transfer_timeseries", f"ensemble_timeseries_{i}_{j}.png"))

    def ens_mean(self):
        obs_destand_test = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        #obs_destand_test[obs_destand_test < 0.01] = 0.0

        sim = np.zeros(obs_destand_test.shape)
        for pixel in range(100):
            ens_mean = np.zeros((obs_destand_test.shape[0], 100))
            for m in range(100):
                sim_destand_test = np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"100px_member_{m}", f"sim_destand_{MODEL_NAME}.npy"))
                #sim_destand_test[sim_destand_test < 0.01] = 0.0
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

    def ensemble_statvsacc(self):
        util = utilities()
        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))
        meansim = np.load(os.path.join(OUTPUTPATH, f"sim_destand_{MODEL_NAME}.npy"))
        nb_members = 100
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        #members_sim = np.moveaxis(members_sim, 0, -1)   # (timeseries, pixels, members)
        #members_sim[members_sim < 0.01] = 0.0

        x_axis = []
        y_axis = []
        ind = 0
        for pixel in range(100):
            ensemble_predictions = members_sim[:,:,pixel]
            ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
            # Calculate the mean prediction for each time step
            mean_prediction = np.mean(ensemble_predictions, axis=1)
            
            ########### Calculate ensemble statistics ###########
            # Calculate the variance for each time step
            ensemble_variance = np.var(ensemble_predictions, axis=1)
            ensemble_variance = np.mean(ensemble_variance)

            # Calculate ensemble statistics (e.g., diversity, spread, etc.)
            # Spread interquantile range (IQR) between 75th and 25th percentiles
            ensemble_iqr = np.percentile(ensemble_predictions, 75, axis=1) - np.percentile(ensemble_predictions, 25, axis=1)  # IQR
            ensemble_iqr_mean = np.mean(ensemble_iqr)
            # Calculate the std for each time step
            ensemble_std = np.std(ensemble_predictions, axis=1)
            ensemble_std = np.mean(ensemble_std)

            # Calculate diversity by Pairwise correlation
            correlation_matrix = np.corrcoef(ensemble_predictions.T)  # Transpose to get members on rows
            ensemble_diversity = 1 - np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]) # Compute diversity as 1 - average correlation

            
            ########### Calculate accuracy metrics ###########
            # Calculate the correlation between the mean prediction and observation
            correlationobs = np.corrcoef(mean_prediction, obs[:,pixel])[0, 1]

            # Calculate the RMSE between the mean prediction and observation
            rmse = np.sqrt(np.mean((obs[:,pixel] - mean_prediction) ** 2))

            # Calculate MSE

            # Calculate kingupta
            kge = util.calculate_kge(obs[:,pixel], mean_prediction)
            
            #### KGE terms analysis ####
            # Compute mean and standard deviation
            mu_o, mu_p = np.mean(obs[:,pixel]), np.mean(mean_prediction)
            sigma_o, sigma_p = np.std(obs[:,pixel]), np.std(mean_prediction)


            # Compute bias ratio (β) and variability ratio (γ)
            beta = mu_p / mu_o
            gamma = sigma_p / sigma_o

            # Calculate NSE
            nse = 1 - (np.sum((obs[:,pixel] - mean_prediction) ** 2) / np.sum((obs[:,pixel] - np.mean(obs[:,pixel])) ** 2))
            
            # Calculate mean bias
            bias_mean = np.mean(mean_prediction - obs[:,pixel])
            
            x_axis.append(ensemble_variance)
            y_axis.append(abs(bias_mean))
            

        xtitle = "Variance"
        ytitle = "Bias"


        plt.figure()
        plt.scatter(x_axis, y_axis, marker='o', color='k')
        #plt.title(f'{ytitle} vs {xtitle}')
        plt.xlabel(f'{xtitle}')
        plt.ylabel(f'{ytitle}')
        #plt.xlim(0, 1)
        #plt.ylim(-1, 1)
        plt.xscale("log")
        plt.yscale("log")
        #plt.plot([min(y_axis), max(y_axis)], [min(y_axis), max(y_axis)], color='red', linestyle='--', label='Ideal Fit (y=x)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"{ytitle}_{xtitle}.png"))
    
    def ensemble_linreg(self):
        util = utilities()
        # load data
        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))

        nb_members = 100
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        #members_sim = np.moveaxis(members_sim, 0, -1)   # (timeseries, pixels, members)
        #members_sim[members_sim < 0.01] = 0.0 # some predictions have full timeseries less than 0.01 

        # Calculate ensemble mean, variance, and diversity
        variance = []
        diversity = []
        acc_matrix = []
        
        for pixel in range(100):
            ensemble_predictions = members_sim[:,:,pixel]
            ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
            # Calculate the mean prediction for each time step
            ensemble_mean = np.mean(ensemble_predictions, axis=1)
            
            ########### Calculate ensemble statistics ###########
            # Calculate ensemble spread
            # Calculate the variance for each time step
            ensemble_variance = np.var(ensemble_predictions, axis=1)
            ensemble_variance = np.mean(ensemble_variance)
            # Spread interquantile range (IQR) between 75th and 25th percentiles
            ensemble_iqr = np.percentile(ensemble_predictions, 75, axis=1) - np.percentile(ensemble_predictions, 25, axis=1)  # IQR
            ensemble_iqr_mean = np.mean(ensemble_iqr)
            # Calculate the std for each time step
            ensemble_std = np.std(ensemble_predictions, axis=1)
            ensemble_std = np.mean(ensemble_std)

            # Calculate diversity by Pairwise correlation
            correlation_matrix = np.corrcoef(ensemble_predictions.T)  # Transpose to get members on rows
            ensemble_diversity = 1 - np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]) # Compute diversity as 1 - average correlation
            
            ########### Calculate accuracy metrics ###########
            # Calculate the correlation between the mean prediction and observation
            correlationobs = np.corrcoef(ensemble_mean, obs[:,pixel])[0, 1]

            # Calculate the RMSE between the mean prediction and observation
            pixel_rmse = np.sqrt(np.mean((obs[:,pixel] - ensemble_mean) ** 2))

            # Calculate kingupta
            kge = util.calculate_kge(obs[:,pixel], ensemble_mean)
            #if kge<-1:
            #    kge = -1

            # Calculate mean bias
            bias_mean = np.mean(ensemble_mean - obs[:,pixel])
            if not np.isnan(correlationobs):# and kge >= -1:
                variance.append(ensemble_variance)
                diversity.append(ensemble_diversity)
                acc_matrix.append(correlationobs)
        
        # Prepare features and target variable
        X = np.column_stack((variance, diversity))  # Independent variables
        y = acc_matrix  # Dependent variable (accuracy metric)

        # Fit a linear regression model
        # accuracy_matrix = a * variance + b * diversity + c
        model = LinearRegression()
        model.fit(X, y)

        # Predictions
        y_pred = model.predict(X)

        # Model evaluation
        r2 = r2_score(y, y_pred)
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        print("Mean Squared Error:", mean_squared_error(y, y_pred))
        print("R² Score:", r2)

        # Plot results
        acc = "correlation"
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, color='blue', alpha=0.7, label='Predicted vs. Actual')
        plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Fit (y=x)')
        plt.xlabel(f"Actual {acc}")
        plt.ylabel(f"Predicted {acc}")
        plt.title(f"Linear Regression: {acc} vs Ensemble variance & diversity")
        #plt.xscale("log")
        #plt.yscale("log")
        plt.legend([f"R²={round(r2, 2)}"])
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"linreg_{acc}.png"))

    def ensemble_nonlinreg(self):
        util = utilities()
        # load data
        obs = np.load(os.path.join(OUTPUTPATH, f"obs_destand_{MODEL_NAME}.npy"))

        nb_members = 100
        members_sim = [np.load(os.path.join(os.path.dirname(OUTPUTPATH), f"400px_member_{m}", f"sim_destand_{MODEL_NAME}.npy")) for m in range(nb_members)]
        members_sim = np.array(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0) #(members, timeseries, pixels)
        #members_sim = np.moveaxis(members_sim, 0, -1)   # (timeseries, pixels, members)
        #members_sim[members_sim < 0.01] = 0.0 # some predictions have full timeseries less than 0.01 

        # Calculate ensemble mean, variance, and diversity
        variance = []
        diversity = []
        acc_matrix = []
        
        for pixel in range(100):
            ensemble_predictions = members_sim[:,:,pixel]
            ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)   # (timeseries, members)
            # Calculate the mean prediction for each time step
            ensemble_mean = np.mean(ensemble_predictions, axis=1)
            
            ########### Calculate ensemble statistics ###########
            # Calculate ensemble spread
            # Calculate the variance for each time step
            ensemble_variance = np.var(ensemble_predictions, axis=1)
            ensemble_variance = np.mean(ensemble_variance)
            # Spread interquantile range (IQR) between 75th and 25th percentiles
            ensemble_iqr = np.percentile(ensemble_predictions, 75, axis=1) - np.percentile(ensemble_predictions, 25, axis=1)  # IQR
            ensemble_iqr_mean = np.mean(ensemble_iqr)
            # Calculate the std for each time step
            ensemble_std = np.std(ensemble_predictions, axis=1)
            ensemble_std = np.mean(ensemble_std)

            # Calculate diversity by Pairwise correlation
            correlation_matrix = np.corrcoef(ensemble_predictions.T)  # Transpose to get members on rows
            ensemble_diversity = 1 - np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]) # Compute diversity as 1 - average correlation
            
            ########### Calculate accuracy metrics ###########
            # Calculate the correlation between the mean prediction and observation
            correlationobs = np.corrcoef(ensemble_mean, obs[:,pixel])[0, 1]

            # Calculate the RMSE between the mean prediction and observation
            pixel_rmse = np.sqrt(np.mean((obs[:,pixel] - ensemble_mean) ** 2))

            # Calculate kingupta
            kge = util.calculate_kge(obs[:,pixel], ensemble_mean)
            
            # Calculate mean bias
            bias_mean = np.mean(ensemble_mean - obs[:,pixel])

            if not np.isnan(correlationobs) and kge >= -1:
                variance.append(ensemble_variance)
                diversity.append(ensemble_diversity)
                acc_matrix.append(kge)
        
        # Prepare features and target variable
        X = np.column_stack((variance, diversity))  # Independent variables
        y = acc_matrix  # Dependent variable (accuracy metric)

        # Create a polynomial regression model (degree=2 for non-linearity)
        # accuracy_matrix = 1 + a * variance + b * diversity + c * variance^2 + e * variance * diversity + d * diversity^2 + f
        poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        poly_model.fit(X, y)
        #print("Polynomial Model Parameters:", poly_model.get_params())
        #print("Polynomial Model Intercept:", poly_model.named_steps['linearregression'].intercept_)
        #print("Polynomial Model Coefficients:", poly_model.named_steps['linearregression'].coef_)
        #print("Polynomial Model Degree:", poly_model.named_steps['polynomialfeatures'].degree)
        #print("Polynomial Model Score:", poly_model.score(X, y))
        #print("Polynomial Model R² Score:", r2_score(y, poly_model.predict(X)))
        #print("Polynomial Model Mean Squared Error:", mean_squared_error(y, poly_model.predict(X)))
        #print(len(poly_model.named_steps['linearregression'].coef_))
        
        # Predictions
        y_poly_pred = poly_model.predict(X)

        # Model evaluation
        r2 = r2_score(y, y_poly_pred)
        print("Polynomial Model Mean Squared Error:", mean_squared_error(y, y_poly_pred))
        print("Polynomial Model R² Score:", r2)

        # Plot results
        acc = "KGE"
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_poly_pred, color='green', alpha=0.7, label='Predicted vs. Actual (Polynomial)')
        plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Fit (y=x)')
        plt.xlabel(f"Actual {acc}")
        plt.ylabel(f"Predicted {acc}")
        plt.title(f"Polynomial Regression: {acc} vs Ensemble Statistics")
        plt.legend([f"R²={round(r2, 2)}"])
        #plt.xscale("log")
        #plt.yscale("log")
        plt.grid(True)
        #plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"nonlinreg_{acc}.png"))

