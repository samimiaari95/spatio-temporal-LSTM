import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from LSTM_model.utils.plot_functions import plotting_helper
from LSTM_model.utils.utils import utilities
from LSTM_model.model.config import *

class postprocess_calculations:
    def __init__(self) -> None:
        pass

    def plot_ensemble_statsvsacc_timeseries(self, target, i, members_sim, mean_prediction, obs, kge, rmse, r):
        nb_members = 100

        dates = pd.date_range(start='2017-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))

        print(f"plotting timeseries {i} with KGE: {kge} , r: {r} and RMSE: {rmse}")
        for m in range(nb_members):
            sim = members_sim[m,:,i]
            ax.plot(dates, sim, color="gray", alpha=0.5)

        ax.plot(dates, obs, "k-", label="Observations", linewidth=4.0)
        ax.plot(dates, mean_prediction, "k--", label="Ensemble mean", linewidth=4.0)
        ax.scatter([], [], color="k", label=f"KGE: {kge}, r: {r}, RMSE: {rmse}")

        ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3)
        plt.grid()
        plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "timeseries_rmseKGEr_lessthan02", f"timeseries_{target}_{i}.png"))


    def ensemble_statvsacc(self):
        utils = utilities()
        EU_inpath = os.path.join(INPUTPATH, "validation_ERA5")
        EU_outpath = os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px")
        transfer_subset = np.load(os.path.join(INPUTPATH, "transfer_subset.npy"))

        df_dic = {"stdsim":[], "stdobs":[], "meansim":[], "meanobs":[], "Absolute mean bias":[], "Pearson correlation":[], "RMSE":[], "KGE":[], "KGE'":[], "Beta":[], "Alpha":[], "NSE":[], "Pairwise correlation":[], "Ensemble variance":[], "IQR (75-25%)":[], "std":[], "cv":[],  "(Alpha-1)^2":[], "(Beta-1)^2":[], "(r-1)^2":[]}
        x_axis = []
        y_axis = []
        for target in range(100):
            target_map = np.load(os.path.join(EU_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            transfer_indices, ind2d = utils.intersect_subsets(target_map, transfer_subset)
            obs = np.load(os.path.join(EU_outpath, f"400px_member_1", f"obs_{target}.npy"))
            obs = obs[:, transfer_indices] # keep only transfer pixels
            members_sim = [np.load(os.path.join(EU_outpath, f"400px_member_{m}", f"sim_{target}.npy")) for m in range(100)]
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
                                
                df_dic["Beta"].append(beta)
                df_dic["Alpha"].append(alpha)
                df_dic["(Alpha-1)^2"].append((alpha-1)**2)
                df_dic["(Beta-1)^2"].append((beta-1)**2)
                df_dic["(r-1)^2"].append((correlationobs-1)**2)


                # Calculate NSE
                nse = 1 - (np.sum((obs[:,pixel] - mean_prediction) ** 2) / np.sum((obs[:,pixel] - np.mean(obs[:,pixel])) ** 2))
                #nsecomp = np.sum((obs[:,pixel] - np.mean(obs[:,pixel])) ** 2)
                # #if np.std(obs[:,pixel]) < 0.1:
                # #    nse = np.nan
                df_dic["NSE"].append(nse)

                # Calculate mean bias
                bias_mean = np.mean(mean_prediction - obs[:,pixel])
                df_dic["Absolute mean bias"].append(bias_mean)
                
                # x_axis.append(ensemble_variance)
                # y_axis.append(kgeprime)
                if kge < 0.2:
                    self.plot_ensemble_statsvsacc_timeseries(target, pixel, members_sim, mean_prediction, obs[:,pixel], f'{kge:.2f}', f'{rmse:.2f}', f'{correlationobs:.2f}')

        ####### save to csv ########
        # df = pd.DataFrame(df_dic)
        # df.to_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics.csv"), index=False)

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

    def ensemble_crpsvsstats_fitting(self):
        utils = utilities()
        dirpath = os.path.join(OUTPUTPATH, "statistics")
        stats = ["IQR (75-25%)", "Ensemble variance", "Pairwise correlation"]
        yval = np.load(os.path.join(dirpath, "crps_transfer.npy"))
        for stat in stats:
            if stat=="Pairwise correlation":
                filename = "pairwisecorr_crps_transfer.npy"
                yval = np.load(os.path.join(dirpath, "crps_meants_px.npy"))
            elif stat=="IQR (75-25%)":
                filename = "iqr_crps_transfer.npy"
            elif stat=="Ensemble variance":
                filename = "ensemblevariance_crps_transfer.npy"

            xval = np.load(os.path.join(dirpath, filename))
            if stat=="Pairwise correlation":
                yval = yval[xval>0.01]
                xval = xval[xval>0.01]
            print(f"fitting {stat} with CRPS")
            # Fit a powerlaw
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
            plt.figure()
            plt.plot(x_fit, y_fit, color="r", label=textstr, linestyle='--')
            plt.scatter(xval, yval, marker='.', color='k')
            plt.xlabel(stat)
            plt.ylabel("CRPS")
            plt.xscale("log")
            plt.yscale("log")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.legend(fontsize=18, frameon=True)
            plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"fitted_CRPS_{stat}.png"))


    def ensemble_statvsacc_fitting(self):
        utils = utilities()
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics.csv"))
        xstats = {"std":"exp", "Ensemble variance":"exp", "Pairwise correlation":"lin", "IQR (75-25%)":"exp"}
        ystats = {"RMSE":"exp", "Pearson correlation":"lin", "KGE":"lin", "Absolute mean bias":"exp", "NSE":"lin", "Beta":"exp", "Alpha":"exp", "(Alpha-1)^2":"exp", "(Beta-1)^2":"exp", "(r-1)^2":"exp"}
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

                if ystat=="Pearson correlation":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                    #xval = xval[yval>=0.0]
                    #yval = yval[yval>=0.0]
                
                if ystat=="KGE" or ystat=="NSE":
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
                if xstat=="std" or xstat=="Ensemble variance" or xstat=="IQR (75-25%)":
                    logx = True
                if ystat=="RMSE" or ystat=="Absolute mean bias" or ystat=="Alpha" or ystat=="Beta" or ystat=="(Alpha-1)^2" or ystat=="(Beta-1)^2" or ystat=="(r-1)^2":
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
                plt.savefig(os.path.join(OUTPUTPATH, "statistics", "fitted_statsvsacc", f"fitted_{ystat}_{xstat}.png"))

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
            obs = np.load(os.path.join(dirpath, f"target_pixels_{target}", obsname))
            sims = np.load(os.path.join(dirpath, f"target_pixels_{target}", simname))
            obs = np.nan_to_num(obs)
            sims = np.nan_to_num(sims)
            obs[obs < 0.0] = 0.0
            sims[sims < 0.0] = 0.0
            return obs, sims

        utils = utilities()
        print("starting the cdf calculation")
        correlation = {"transfer_400px":[], "test_400px":[], "transfer_ERA5":[], "test_ERA5":[]}
        rmse = {"transfer_400px":[], "test_400px":[], "transfer_ERA5":[], "test_ERA5":[]}
        kge = {"transfer_400px":[], "test_400px":[], "transfer_ERA5":[], "test_ERA5":[]}
        nse = {"transfer_400px":[], "test_400px":[], "transfer_ERA5":[], "test_ERA5":[]}
        bias = {"transfer_400px":[], "test_400px":[], "transfer_ERA5":[], "test_ERA5":[]}
        
        # TODO plot cdf of ERA5 test and transfer evaluation compared with observations
        # TODO plot with it the TSMP test and transfer evaluation compared with observations

        # TODO select timeseries only for january 2020

        #### Transfer & training filtered subsets ####
        ERA5_inpath = os.path.join(INPUTPATH, "validation_ERA5")
        EU400px_inpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/inputs/20yrs_ts", "ensemble_400px")
        training_subset = np.load(os.path.join(INPUTPATH, "training_subset.npy"))
        print("Progress: [" + "." * 100 + "]", flush=True)
        print("          [", end="", flush=True)  # Start progress bar            
        for target in range(100):
            print(".", end="", flush=True)  # Dots without newlines
            ######## ERA5 ensemble ########
            # calculate transfer metrics for 400px ensemble
            obs_transferERA5, sim_transferERA5 = load_obs_sim(ERA5_inpath, target, f"obs_{target}.npy", f"sim_transferpixels_{target}.npy")
            obs_testERA5, sim_testERA5 = load_obs_sim(ERA5_inpath, target, f"obs_{target}.npy", f"sim_testtrainpixels_{target}.npy")
            target_map = np.load(os.path.join(ERA5_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            trainingERA5_indices, ind2d = utils.intersect_subsets(target_map, training_subset)
            corr_EU = calc_correlation(obs_transferERA5, sim_transferERA5)
            rmse_EU = calc_RMSE(obs_transferERA5, sim_transferERA5)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_transferERA5, sim_transferERA5)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from 99 transfer members (training subset) + 100 transfer members (transfer subset)
            correlation["transfer_ERA5"].extend(corr_EU.tolist())
            rmse["transfer_ERA5"].extend(rmse_EU.tolist())
            kge["transfer_ERA5"].extend(kge_EU.tolist())
            nse["transfer_ERA5"].extend(nse_EU.tolist())
            bias["transfer_ERA5"].extend(bias_EU.tolist())

            # calculate test metrics for ERA5 ensemble
            corr_EU = calc_correlation(obs_testERA5, sim_testERA5)
            rmse_EU = calc_RMSE(obs_testERA5, sim_testERA5)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(obs_testERA5, sim_testERA5)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from only 1 training member (training subset)
            correlation["test_ERA5"].extend(corr_EU[trainingERA5_indices].tolist())
            rmse["test_ERA5"].extend(rmse_EU[trainingERA5_indices].tolist())
            kge["test_ERA5"].extend(kge_EU[trainingERA5_indices].tolist())
            nse["test_ERA5"].extend(nse_EU[trainingERA5_indices].tolist())
            bias["test_ERA5"].extend(bias_EU[trainingERA5_indices].tolist())

            ######## 400px ensemble ########
            # calculate transfer metrics for 400px ensemble
            obs_transfer400, sim_transfer400 = load_obs_sim(EU400px_inpath, target, f"obs_{target}.npy", f"sim_transferpixels_{target}.npy")
            obs_test400, sim_test400 = load_obs_sim(EU400px_inpath, target, f"obs_{target}.npy", f"sim_testtrainpixels_{target}.npy")
            target_map = np.load(os.path.join(EU400px_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
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
        
        if not sim_transferERA5.shape[0]==TEST_PERIOD-LOOKBACK or not sim_testERA5.shape[0]==TEST_PERIOD-LOOKBACK or not sim_transfer400.shape[0]==TEST_PERIOD-LOOKBACK or not sim_test400.shape[0]==TEST_PERIOD-LOOKBACK:
            print(f"WARNING: unexpected timeseries length for target {target}")

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
        #### plot Pearson correlation ####
        utils.plot_cdfs(data_dict={
            f'Transfer TSMP': correlation["transfer_400px"],
            f'Test TSMP': correlation["test_400px"],
            f'Transfer ERA5': correlation["transfer_ERA5"],
            f'Test ERA5': correlation["test_ERA5"]
        }, colors=['red', 'blue', 'red', 'blue'], linestyles=['--', '--', ':', ':'],
        xlabel='Pearson correlation', title='testtransfersets')

        #### plot RMSE ####
        utils.plot_cdfs(data_dict={
            f'Transfer TSMP': rmse["transfer_400px"],
            f'Test TSMP': rmse["test_400px"],
            f'Transfer ERA5': rmse["transfer_ERA5"],
            f'Test ERA5': rmse["test_ERA5"]
        }, colors=['red', 'blue', 'red', 'blue'], linestyles=['--', '--', ':', ':'],
        xlabel='RMSE (m)', logscale=True, title='testtransfersets')

        # #### plot KGE ####
        utils.plot_cdfs(data_dict={
            f'Transfer TSMP': kge["transfer_400px"],
            f'Test TSMP': kge["test_400px"],
            f'Transfer ERA5': kge["transfer_ERA5"],
            f'Test ERA5': kge["test_ERA5"]
        }, xmin=0.2, colors=['red', 'blue', 'red', 'blue'], linestyles=['--', '--', ':', ':'],
        xlabel='KGE', xlim=(0.2, 1), yfloor=True, title='testtransfersets')

        # #### plot NSE ####
        utils.plot_cdfs(data_dict={
            f'Transfer TSMP': nse["transfer_400px"],
            f'Test TSMP': nse["test_400px"],
            f'Transfer ERA5': nse["transfer_ERA5"],
            f'Test ERA5': nse["test_ERA5"]
        }, xmin=0.2, colors=['red', 'blue', 'red', 'blue'], linestyles=['--', '--', ':', ':'],
        xlabel='NSE', xlim=(0.2, 1), yfloor=True, title='testtransfersets')

        #### plot Bias ####
        utils.plot_cdfs(data_dict={
            f'Transfer TSMP': bias["transfer_400px"],
            f'Test TSMP': bias["test_400px"],
            f'Transfer ERA5': bias["transfer_ERA5"],
            f'Test ERA5': bias["test_ERA5"]
        }, colors=['red', 'blue', 'red', 'blue'], linestyles=['--', '--', ':', ':'],
        xlabel='Mean bias (m)', title='testtransfersets')

        return

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
        fig, axs = plt.subplots(1, crps.shape[0], figsize=(20, 6), sharey=True)
        for i, year in enumerate(range(2017, 2021)):
            axs[i].boxplot(crps[i].T, labels=["DJF", "MAM", "JJA", "SON"])
            axs[i].set_title(f"CRPS {year}")
            #axs[i].set_ylabel("CRPS")
            axs[i].set_yscale('log')
            axs[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"seasonal_crps.png"))
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
        EU_inpath = os.path.join(INPUTPATH, "validation_ERA5")
        EU_outpath = os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px")
        EU_traininpath = os.path.join(INPUTPATH, "validation_ERA5")
        print(EU_inpath)
        for target in range(100):
            target_mapping = np.load(os.path.join(EU_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            obs_destand_EU = np.load(os.path.join(EU_outpath, f"400px_member_1", f"obs_{target}.npy"))
            members_sim = [np.load(os.path.join(EU_outpath, f"400px_member_{m}", f"sim_{target}.npy")) for m in range(100)]
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
                member_mapping = np.load(os.path.join(EU_traininpath, f"400px_member_{member}", "choices.npy"))
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
            np.save(os.path.join(EU_inpath, f"target_pixels_{target}", f"sim_testtrainpixels_{target}.npy"), mean_prediction_testtrain)
            np.save(os.path.join(EU_inpath, f"target_pixels_{target}", f"obs_{target}.npy"), obs_destand_EU)
            np.save(os.path.join(EU_inpath, f"target_pixels_{target}", f"sim_transferpixels_{target}.npy"), mean_prediction_transfer)
            print(f"saved {target} target pixels")

    def map_1Dto2D_EU(self, dirpath):
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
            obs_destand_test = np.load(os.path.join(dirpath, f"target_pixels_{target}", f"obs_ensmean.npy"))
            sim_destand_test = np.load(os.path.join(dirpath, f"target_pixels_{target}", f"sim_ensmean.npy"))
            obs_destand_test = np.nan_to_num(obs_destand_test)
            sim_destand_test = np.nan_to_num(sim_destand_test)
            obs_destand_test[obs_destand_test < 0.0] = 0.0
            sim_destand_test[sim_destand_test < 0.0] = 0.0
            return obs_destand_test, sim_destand_test
        
        utils = utilities()

        #### Transfer subset ####
        
        # filtered subset
        transfer_subset = np.load(os.path.join(INPUTPATH, "transfer_subset.npy"))
        training_subset = np.load(os.path.join(INPUTPATH, "training_subset.npy"))

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
            obs_destand_test, sim_destand_test = load_obs_sim(target)
            target_map = np.load(os.path.join(dirpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
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
    
    def metrics_2D_EU(self):
        plot_functions = plotting_helper()
        print("mapping 1D metrics to 2D EU map")
        era5dirpath = os.path.join(INPUTPATH, "validation_ERA5")
        tsmpdirpath = os.path.join(os.path.dirname(os.path.dirname(get_root_dir())), "spatio-temporal-LSTM", "inputs", "20yrs_ts", "ensemble_400px")
        era5corr2d, era5rmse2d, era5bias2d, era5kge2d, era5nse2d = self.map_1Dto2D_EU(era5dirpath)
        tsmpcorr2d, tsmprmse2d, tsmpbias2d, tsmpkge2d, tsmpnse2d = self.map_1Dto2D_EU(tsmpdirpath)

        # TODO plot for KGE<0.2 from the tsmp not era5
        # get indices where kge<0.2
        era5kge2d = np.where(tsmpkge2d<0.2, np.nan, era5kge2d)
        print(f"number of pixels with KGE>=0.2: {np.sum(~np.isnan(era5kge2d))}")
        era5nse2d = np.where(np.isnan(era5kge2d), np.nan, era5nse2d)
        era5corr2d = np.where(np.isnan(era5kge2d), np.nan, era5corr2d)
        era5rmse2d = np.where(np.isnan(era5kge2d), np.nan, era5rmse2d)
        era5bias2d = np.where(np.isnan(era5kge2d), np.nan, era5bias2d)
        # bias2d = np.where(kge2d<0.2, np.nan, bias2d)

        plot_functions.EU_2Dmap(data_map=era5corr2d, logscale=False, minval=0, maxval=1, title="Pearson correlation")
        plot_functions.EU_2Dmap(data_map=era5rmse2d, logscale=True, minval=0.01, maxval=10, title="RMSE")
        plot_functions.EU_2Dmap(data_map=era5bias2d, logscale=False, minval=-10, maxval=10, title="Mean bias")
        plot_functions.EU_2Dmap(data_map=era5kge2d, logscale=False, minval=0.2, maxval=1, title="KGE")
        plot_functions.EU_2Dmap(data_map=era5nse2d, logscale=False, minval=-1, maxval=1, title="NSE")

    def metrics_vs_topo(self):
        ###### comment out the training pixels from self.map_1Dto2D_EU() ######
        # take only the transfer ones
        # change topo for wtd and select the average wtd for year 2020
        corr2d, rmse2d, bias2d, kge2d, nse2d = self.map_1Dto2D_EU()
        plot_functions = plotting_helper()
        topo_v1 = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "topo.npy"))
        topo_v1 = topo_v1[0,:,:]
        # topo_v1 = np.mean(topo_v1[-365:,:,:], axis=0)  # average wtd for year 2020

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
        EU_inpath = os.path.join(INPUTPATH, "validation_ERA5")
        EU_outpath = os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean")
        EU_traininpath = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_400_withcriteria_43226/inputs/20yrs_ts", "ensemble_400px")
        rollsubset = np.load(os.path.join(INPUTPATH, "mapping_0stdroll6months.npy"))
        iqrmap = np.zeros(rollsubset.shape)
        iqrmap[iqrmap==0] = np.nan
        varmap = np.zeros(rollsubset.shape)
        varmap[varmap==0] = np.nan
        print("Progress: [" + "." * 100 + "]", flush=True)
        print("          [", end="", flush=True)
        for target in range(100):
            print(".", end="", flush=True)  # Dots without newlines
            target_mapping = np.load(os.path.join(EU_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            members_sim = [np.load(os.path.join(os.path.dirname(EU_outpath), f"400px_member_{m}", f"sim_{target}.npy")) for m in range(100)]
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
        # keep only pixels that we have observations for
        obsmask = np.zeros(rollsubset.shape)

        proj_mapping = pd.read_csv(os.path.join(INPUTPATH, "localobservations", "mapping_localobs_TSMPproj.csv"))
        proj_mapping = proj_mapping.dropna(subset=["sim_1darrayindex"])
        proj_mapping.reset_index(drop=True, inplace=True)
        tsmpx = proj_mapping["tsmp_xindex"].values
        tsmpy = proj_mapping["tsmp_yindex"].values
        obsmask[tsmpy, tsmpx] = 1.0

        iqrmap = np.where(obsmask==0.0, np.nan, iqrmap)
        varmap = np.where(obsmask==0.0, np.nan, varmap)

        bias2d_statspred = 0.45*(iqrmap**1.03)
        rmse2d_statspred = 0.67*(varmap**0.45)
        plot_functions.doublefig_EU_2Dmap(data_map=bias2d_statspred, logscale=True, minval=0.01, maxval=10, title="Absolute mean bias")
        plot_functions.doublefig_EU_2Dmap(data_map=rmse2d_statspred, logscale=True, minval=0.01, maxval=10, title="RMSE")
    
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
        wtd = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "topo.npy"))
        print(wtd.shape)
        # exclude sides
        wtd[:, :100,:] = 0
        wtd[:, 432-10:,:] = 0
        wtd[:, :,444-10:] = 0
        wtd[:, :,:10] = 0

        # set negative wtd to 0
        wtd[wtd<0] = 0

        wtd = wtd[0,:,:]
        print(wtd.shape)
        maxwtd = np.nanmax(wtd)
        print(maxwtd)
        wtd[wtd==0] = np.nan
        plot_functions.EU_2Dmap(wtd, logscale=False, minval=0.01, maxval=maxwtd, title="Topography")

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
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics.csv"))
        
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

        plt.figure(figsize=(8, 6))
        plt.scatter(ev, alpha_comp, color='blue', alpha=0.4, label=r'$(\alpha-1)^2$')
        plt.scatter(ev, beta_comp, color='green', alpha=0.4, label=r'$(\beta-1)^2$')
        plt.scatter(ev, r_comp, color='red', alpha=0.4, label=r'$(r-1)^2$')

        # Decorations
        print("plotting it")
        plt.xlabel(r"$\overline{EV}$")
        plt.ylabel("KGE component")
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", "KGEcomp_EV_kgelessthan02.png"), dpi=300)
    
    def era5wtdensemble_localobs_timeseries(self, obslocation, sim_era5wtd, mean_prediction, tsmpobs, localobs, kge, rmse, r, bias, nse):
        dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))

        kge = round(kge, 2)
        r = round(r, 2)
        rmse = round(rmse, 2)
        bias = round(bias, 2)
        nse = round(nse, 2)

        print(f"plotting timeseries {obslocation} with KGE: {kge} , r: {r}, RMSE: {rmse}, Bias: {bias}, NSE: {nse}")
        for m in range(100):
            sim = sim_era5wtd[m,:]
            ax.plot(dates, sim, color="gray", alpha=0.5)

        ax.plot(dates, localobs, "r-", label="Observations", linewidth=3.0)
        ax.plot(dates, mean_prediction, "k-", label="ERA5 ensemble mean", linewidth=3.0)
        ax.plot(dates, tsmpobs, "b-", label="TSMP observations", linewidth=3.0)
        ax.scatter([], [], color="k", label=f"KGE:{kge} ,r:{r}, RMSE:{rmse},\nBias:{bias}, NSE: {nse}")

        ax.xaxis.set_major_locator(mdates.MonthLocator([1,7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3)
        plt.grid()
        plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "timeseries", f"timeseries_{obslocation.replace('.', 'p')}.png"))

    def get_era5wtd(self, target, sim_1darrayindex):
        sim_allmembers = np.array([np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", f"400px_member_{m}", f"sim_{target}.npy")) for m in range(100)])
        sim_allmembers = sim_allmembers[:,:,sim_1darrayindex]  # (members, timeseries)
        return sim_allmembers

    def get_tsmpwtd(self, target, sim_1darrayindex):
        tsmpobs = np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", f"400px_member_0", f"obs_{target}.npy"))
        tsmpobs = tsmpobs[:, sim_1darrayindex]
        return tsmpobs

    def eval_era5wtd_obswtd_tsmpwtd(self):
        utils = utilities()
        plot_functions = plotting_helper()
        map2d = np.load(os.path.join(INPUTPATH, "mapping_0stdroll6months.npy"))
        map2d_metrics = {"KGE": np.zeros(map2d.shape), "r": np.zeros(map2d.shape), "RMSE": np.zeros(map2d.shape), "Bias": np.zeros(map2d.shape), "NSE": np.zeros(map2d.shape)}
        map2d_metrics["KGE"][map2d_metrics["KGE"]==0] = np.nan
        map2d_metrics["r"][map2d_metrics["r"]==0] = np.nan
        map2d_metrics["RMSE"][map2d_metrics["RMSE"]==0] = np.nan
        map2d_metrics["Bias"][map2d_metrics["Bias"]==0] = np.nan
        map2d_metrics["NSE"][map2d_metrics["NSE"]==0] = np.nan
        obswtd = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "obs_EUwtd_DPSF.parquet"))
        obswtd = obswtd.dropna(axis=1)  # drop columns with all NaN values
        print(len(obswtd.columns))
        proj_mapping = pd.read_csv(os.path.join(INPUTPATH, "localobservations", "mapping_DPSFobs_TSMPproj.csv"))
        proj_mapping = proj_mapping.dropna(subset=["sim_1darrayindex"])
        proj_mapping.reset_index(drop=True, inplace=True)
        cdf_data = {
            'KGE': [],
            'Pearson correlation': [],
            'RMSE': [],
            'Bias': [],
            'NSE': []
        }
        ev = []
        iqr = []
        rmse_list = []
        ambias_list = []
        for pixel in range(len(proj_mapping)):
            print(f"Processing pixel {pixel+1} of {len(proj_mapping)}", end='\r')
            countrylocation = proj_mapping.iloc[pixel]["countylocation"]
            if not countrylocation in obswtd.columns:
                continue
            tsmplat = proj_mapping.iloc[pixel]["tsmp_lat"]
            tsmplon = proj_mapping.iloc[pixel]["tsmp_lon"]
            tsmp_xindex = proj_mapping.iloc[pixel]["tsmp_xindex"]
            tsmp_yindex = proj_mapping.iloc[pixel]["tsmp_yindex"]
            target = int(proj_mapping.iloc[pixel]["target_chunk"])
            sim_1darrayindex = int(proj_mapping.iloc[pixel]["sim_1darrayindex"])
            sim_era5wtd = self.get_era5wtd(target, sim_1darrayindex)
            sim_era5wtd = sim_era5wtd[:, :-365]  # remove year 2020 to match localobs
            ens_mean_era5wtd = np.mean(sim_era5wtd, axis=0)
            tsmpobs = self.get_tsmpwtd(target, sim_1darrayindex)
            tsmpobs = tsmpobs[:-365]  # remove year 2020 to match localobs
            localobs = np.array(obswtd[f"{countrylocation}"].to_list())
            # localobs = localobs[730:]  # remove year 2016 to match sim_era5wtd and tsmpobs
            # check lengths
            if len(localobs) != sim_era5wtd.shape[1] or len(localobs) != tsmpobs.shape[0]:
                print(f"lengths do not match for pixel {countrylocation}, skipping...")
                return
            kge = utils.calculate_kge(localobs, ens_mean_era5wtd)
            rmse = np.sqrt(np.mean((localobs - ens_mean_era5wtd) ** 2))
            r = np.corrcoef(localobs, ens_mean_era5wtd)[0, 1]
            bias = np.mean(ens_mean_era5wtd - localobs)
            nse = 1 - (np.sum((localobs - ens_mean_era5wtd) ** 2) / np.sum((localobs - np.mean(localobs)) ** 2))
            cdf_data['KGE'].append(kge)
            cdf_data['Pearson correlation'].append(r)
            cdf_data['RMSE'].append(rmse)
            cdf_data['Bias'].append(bias)
            cdf_data['NSE'].append(nse)
            ############## plot timeseries ##############
            self.era5wtdensemble_localobs_timeseries(countrylocation, sim_era5wtd, ens_mean_era5wtd, tsmpobs, localobs,
                                               kge=kge,
                                               rmse=rmse,
                                               r=r,
                                               bias=bias,
                                               nse=nse
                                               )
            
            ############## collect data for EV vs metrics ##############
            ensemble_variance = np.var(sim_era5wtd, axis=0)
            ensemble_variance = np.mean(ensemble_variance)
            ev.append(ensemble_variance)
            ensemble_iqr = np.percentile(sim_era5wtd, 75, axis=0) - np.percentile(sim_era5wtd, 25, axis=0)
            ensemble_iqr = np.mean(ensemble_iqr)
            iqr.append(ensemble_iqr)
            rmse_list.append(rmse)
            ambias_list.append(np.abs(bias))

            ############## plot 2D map of pixels with colored accuracy ##############
            map2d_metrics["r"][int(tsmp_yindex), int(tsmp_xindex)] = r
            map2d_metrics["KGE"][int(tsmp_yindex), int(tsmp_xindex)] = kge
            map2d_metrics["RMSE"][int(tsmp_yindex), int(tsmp_xindex)] = rmse
            map2d_metrics["Bias"][int(tsmp_yindex), int(tsmp_xindex)] = bias
            map2d_metrics["NSE"][int(tsmp_yindex), int(tsmp_xindex)] = nse
        
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "r_map.npy"), map2d_metrics["r"])
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "KGE_map.npy"), map2d_metrics["KGE"])
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "RMSE_map.npy"), map2d_metrics["RMSE"])
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "Bias_map.npy"), map2d_metrics["Bias"])
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "NSE_map.npy"), map2d_metrics["NSE"])
        
        
        ############## plot CDF ##############
        # absolute mean bias instead of mean bias
        # cdf_data["Bias"] = np.abs(cdf_data["Bias"])
        # utils.plot_cdfs(data_dict={ "KGE": cdf_data["KGE"] }, colors=['k'], linestyles=['-'], xlabel='KGE', title='ERA5ensemblevsobs', logscale=False, xlim=(0.2,1))
        # utils.plot_cdfs(data_dict={ "Pearson correlation": cdf_data["Pearson correlation"] }, colors=['k'], linestyles=['-'], xlabel='Pearson correlation', title='ERA5ensemblevsobs')
        # utils.plot_cdfs(data_dict={ "RMSE": cdf_data["RMSE"] }, colors=['k'], linestyles=['-'], xlabel='RMSE (m)', title='ERA5ensemblevsobs', logscale=True)
        # utils.plot_cdfs(data_dict={ "Bias": cdf_data["Bias"] }, colors=['k'], linestyles=['-'], xlabel='Absolute Mean Bias (m)', title='ERA5ensemblevsobs', logscale=True)
        # utils.plot_cdfs(data_dict={ "NSE": cdf_data["NSE"] }, colors=['k'], linestyles=['-'], xlabel='NSE', title='ERA5ensemblevsobs', logscale=False, xlim=(0,1))

        ############### plot EV vs metrics ##############
        # plot also the fitting line from before, plot it as a line not scatter
        # for EV vs RMSE: rmse2d_statspred = 0.67*(varmap**0.45)
        # for IQR vs Bias: bias2d_statspred = 0.45*(iqrmap**1.03)
        line = np.array([np.min(iqr), np.max(iqr)])
        fitted_bias = 0.45 * (line ** 1.03)
        # calculate R2
        ev = np.array(iqr)
        # rmse_list = np.array(rmse_list)
        # log_ev = np.log(ev)
        # log_rmse = np.log(rmse_list)
        # slope, intercept, r_value, p_value, std_err = stats.linregress(log_ev, log_rmse)
        # r_squared = r_value**2
        # print(f"EV vs RMSE fitting line R2: {r_squared}")
        ambias_list = np.array(ambias_list)
        log_iqr = np.log(iqr)
        log_ambias = np.log(ambias_list)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_iqr, log_ambias)
        r_squared = r_value**2
        print(f"IQR vs Mean Absolute Bias fitting line R2: {r_squared}")

        plt.figure()
        plt.scatter(iqr, ambias_list, alpha=0.5)
        # plot fitting line
        plt.plot(line, fitted_bias, 'r-', label=f'R2={r_squared:.2f}\nMAB=0.45*(IQR^1.03)')
        plt.legend(loc='lower right')
        plt.xlabel(r'$\overline{IQR}$')
        plt.ylabel(r'Mean Absolute Bias (m)')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "statistics", "IQR_vs_MAB.png"), dpi=300, bbox_inches='tight')
    
    def plot2Dmaps(self):
        plot_functions = plotting_helper()
        map2d_metrics = {"KGE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "KGE_map.npy")),
                         "r": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "r_map.npy")),
                         "RMSE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "RMSE_map.npy")),
                         "Bias": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "Bias_map.npy")),
                         "NSE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "NSE_map.npy"))}
        # plot absolute mean bias instead of mean bias
        map2d_metrics["Bias"] = np.abs(map2d_metrics["Bias"])
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["KGE"], logscale=False, minval=-1, maxval=1, title="KGE")
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["r"], logscale=False, minval=0, maxval=1, title="Pearson correlation")
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["RMSE"], logscale=True, minval=0.01, maxval=10, title="RMSE")
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["Bias"], logscale=True, minval=0.01, maxval=10, title="Absolute Mean Bias")
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["NSE"], logscale=False, minval=-1, maxval=1, title="NSE")
    
    def get_country_mask(self, country):
        mappingfile = pd.read_csv(os.path.join(INPUTPATH, "localobservations", "mapping_DPSFobs_TSMPproj.csv"))
        mappingfile = mappingfile.dropna(subset=["sim_1darrayindex"])
        mappingfile.reset_index(drop=True, inplace=True)
        country_mask = np.zeros((432, 444), dtype=bool)
        # TODO do the same without a for loop
        for pixel in range(len(mappingfile)):
            countrylocation = mappingfile.iloc[pixel]["countylocation"]
            if countrylocation.startswith(country):
                tsmpxindex = int(mappingfile.iloc[pixel]["tsmp_xindex"])
                tsmpyindex = int(mappingfile.iloc[pixel]["tsmp_yindex"])
                country_mask[tsmpyindex, tsmpxindex] = True
        return country_mask

    def plot2Dmaps_byregion(self, country):
        plot_functions = plotting_helper()
        map2d_metrics = {"KGE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "KGE_map.npy")),
                         "r": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "r_map.npy")),
                         "RMSE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "RMSE_map.npy")),
                         "Bias": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "Bias_map.npy")),
                         "NSE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "NSE_map.npy"))}
        # plot absolute mean bias instead of mean bias
        map2d_metrics["Bias"] = np.abs(map2d_metrics["Bias"])

        # apply region mask
        country_mask = self.get_country_mask(country)
        for key in map2d_metrics.keys():
            map2d_metrics[key] = np.where(country_mask, map2d_metrics[key], np.nan)
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["KGE"], logscale=False, minval=-1, maxval=1, title="KGE", country=country)
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["r"], logscale=False, minval=0, maxval=1, title="Pearson correlation", country=country)
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["RMSE"], logscale=True, minval=0.01, maxval=10, title="RMSE", country=country)
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["Bias"], logscale=True, minval=0.01, maxval=10, title="Absolute Mean Bias", country=country)
        plot_functions.doublefig_EU_2Dmap(data_map=map2d_metrics["NSE"], logscale=False, minval=-1, maxval=1, title="NSE", country=country)
    
    def plot2Dmaps_multiregion(self):
        regions = ["France", "Sweden"]
        for country in regions:
            self.plot2Dmaps_byregion(country)

        
    def plot_cdfs(self):
        map2d_metrics = {"KGE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "KGE_map.npy")),
                         "r": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "r_map.npy")),
                         "RMSE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "RMSE_map.npy")),
                         "Bias": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "Bias_map.npy")),
                         "NSE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "NSE_map.npy"))}
        utils = utilities()
        map2d_metrics["KGE"] = map2d_metrics["KGE"][~np.isnan(map2d_metrics["KGE"])]
        print(map2d_metrics["KGE"].shape)
        map2d_metrics["r"] = map2d_metrics["r"][~np.isnan(map2d_metrics["r"])]
        print(map2d_metrics["r"].shape)
        map2d_metrics["RMSE"] = map2d_metrics["RMSE"][~np.isnan(map2d_metrics["RMSE"])]
        print(map2d_metrics["RMSE"].shape)
        map2d_metrics["Bias"] = np.abs(map2d_metrics["Bias"])
        map2d_metrics["Bias"] = map2d_metrics["Bias"][~np.isnan(map2d_metrics["Bias"])]
        print(map2d_metrics["Bias"].shape)
        map2d_metrics["NSE"] = map2d_metrics["NSE"][~np.isnan(map2d_metrics["NSE"])]
        print(map2d_metrics["NSE"].shape)

        utils.plot_cdfs(data_dict={"KGE": map2d_metrics["KGE"]}, colors=['k'], linestyles=['-'], xlabel='KGE', xmin=0.2, xlim=(0.2, 1), yfloor=True, title='ERA5ensemblevsobs')
        utils.plot_cdfs(data_dict={"Pearson correlation": map2d_metrics["r"]}, colors=['k'], linestyles=['-'], xlabel='Pearson correlation', title='ERA5ensemblevsobs')
        utils.plot_cdfs(data_dict={"RMSE": map2d_metrics["RMSE"]}, colors=['k'], linestyles=['-'], logscale=True, xlabel='RMSE (m)', title='ERA5ensemblevsobs')
        utils.plot_cdfs(data_dict={"Bias": map2d_metrics["Bias"]}, colors=['k'], linestyles=['-'], xlabel='Mean Absolute Bias (m)', logscale=True, title='ERA5ensemblevsobs')
        utils.plot_cdfs(data_dict={"NSE": map2d_metrics["NSE"]}, colors=['k'], linestyles=['-'], xlabel='NSE', xmin=0.2, xlim=(0.2, 1), yfloor=True, title='ERA5ensemblevsobs')

    def location_of_localobs(self):
        plot_functions = plotting_helper()
        proj_mapping = pd.read_csv(os.path.join(INPUTPATH, "localobservations", "mapping_DPSFobs_TSMPproj.csv"))
        topo = np.load(os.path.join(INPUTPATH, "topo.npy"))
        topo = topo[0,:,:]  # remove time dimension if present
        # topo[topo==0] = np.nan  # set ocean to nan for better visualization
        for i in range(len(proj_mapping)):
            obslocation = proj_mapping.iloc[i]["countylocation"]
            tsmp_xindex = proj_mapping.iloc[i]["tsmp_xindex"]
            tsmp_yindex = proj_mapping.iloc[i]["tsmp_yindex"]
            df = proj_mapping
            # select all rows with the same country in the column "countylocation" that has a format "country_lonlat"
            # create a column with only the country name
            df["country"] = df["countylocation"].apply(lambda x: x.split("_")[0])
            country = obslocation.split("_")[0]
            df = df[df["country"] == country]
            tsmplat = df["tsmp_lat"].to_list()
            tsmplon = df["tsmp_lon"].to_list()

            plot_functions.onepixel_in_doublefig_EU_2Dmap(x=tsmp_xindex, y=tsmp_yindex, location=obslocation, tsmp_lon=tsmplon, tsmp_lat=tsmplat, topo=topo)