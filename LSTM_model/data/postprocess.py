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

    def ensemble_statistics_scipy(self, ensemble, observations=None, quantiles=(5, 50, 95)):
        """
        Compute statistics for an ensemble of time series.

        Parameters
        ----------
        ensemble : np.ndarray
            Shape (n_members, n_time)
        observations : np.ndarray, optional
            Shape (n_time,) for verification metrics
        quantiles : tuple
            Quantiles to compute (in percent)

        Returns
        -------
        dict containing:
            mean, median, std, var, skewness, kurtosis,
            quantiles, spread, and optional verification metrics
        """

        ensemble = np.asarray(ensemble)

        # Pointwise statistics
        mean = np.mean(ensemble, axis=0)
        median = np.median(ensemble, axis=0)
        std = np.std(ensemble, axis=0)
        var = np.var(ensemble, axis=0)
        skewness = stats.skew(ensemble, axis=0, bias=False)
        kurtosis = stats.kurtosis(ensemble, axis=0, bias=False)

        q = np.percentile(ensemble, quantiles, axis=0)
        quantilesdic = [f"Mean {x}th quantile" for x in quantiles]
        spread = std.mean()  # mean spread over time

        results = {
            "Ensemble mean": mean,
            "Ensemble median": median,
            "std": std,
            "variance": var,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            quantilesdic[0]: q[0],
            quantilesdic[1]: q[1],
            quantilesdic[2]: q[2],
            "mean_spread": spread
        }

        # Optional verification against observations
        if observations is not None:
            observations = np.asarray(observations)

            rmse = np.sqrt(np.mean((mean - observations) ** 2))
            mae = np.mean(np.abs(mean - observations))
            bias = np.mean(mean - observations)

            results.update({
                "RMSE": rmse,
                "MAE": mae,
                "Bias": bias
            })
        return results

    def ensemble_statvsacc(self):
        utils = utilities()
        topo = np.load(os.path.join(INPUTPATH, "topo.npy"))
        topo = topo[0, :, :]
        soilmoisture = np.load(os.path.join(INPUTPATH, "swvl3_EU.npy"))
        precip = np.load(os.path.join(INPUTPATH, "tp_EU.npy"))
        vpd = np.load(os.path.join(INPUTPATH, "vpd_EU.npy"))

        obswtd = pd.read_parquet(os.path.join(os.path.join(
            INPUTPATH, "localobservations"), "wtd_obsEU_monthly_avgdup.parquet"))
        obswtd = obswtd.dropna(axis=1)  # drop columns with all NaN values
        proj_mapping = pd.read_csv(os.path.join(
            INPUTPATH, "localobservations", "mapping_wtdobsEU_TSMP_avgdup.csv"))
        proj_mapping = proj_mapping.dropna(subset=["sim_1darrayindex"])
        proj_mapping.reset_index(drop=True, inplace=True)
        # metrics according to https://climpred.readthedocs.io/en/stable/metrics.html
        df_metricdict = {"Spearman correlation": [], "Kendall's tau": [], "Pearson p-value": [], "CCC": [], "Median AE": [],
                         "MAPE": [], "SMAPE": [], "Murphy Skill Score": [], "Bias Slope": [], "Conditional Bias": [], "Unconditional Bias": [],
                         "MSE Skill Score": [], "NMSE": [], "NRMSE": [], "NMAE": [], "Unbiased Anomaly Correlation Coefficient": [],
                         "Multiplicative Bias": [], "LESS": [], "Bias_std": [],
                         "Members RMSE": [], "Members correlation": [], "Members KGE": [],
                         "Members correlation std": [], "Members RMSE std": [], "Members KGE std": [],
                         "KGE_nobias": [], "stdobs": [], "meanobs": [], "Absolute mean bias": [], "MAE": [],
                         "Pearson correlation": [], "RMSE": [], "KGE": [], "KGE'": [], "Beta": [], "Alpha": [], "NSE": [],
                         "(Alpha-1)^2": [], "(Beta-1)^2": [], "(r-1)^2": []}
        df_statdict = {"Pairwise correlation": [], "Ensemble variance": [], "IQR (75-25%)": [], "std": [], "cv": [], "mad": [], "meansim": [], "stdsim": [], "VPD variance": [], "Precipitation variance": [
        ], "Soil moisture variance": [], "Ensemble mean": [], "Ensemble median": [], "Skewness": [], "Kurtosis": [], "Mean 5th quantile": [], "Mean 95th quantile": [], "Mean 50th quantile": [], "Topography": []}
        for pixel in range(len(proj_mapping)):
            print(
                f"Processing pixel {pixel+1} of {len(proj_mapping)}", end='\r')
            # countrylocation = proj_mapping.iloc[pixel]["countrylocation"]
            country = proj_mapping.iloc[pixel]["country"]
            tsmpx = int(proj_mapping.iloc[pixel]["tsmp_xindex"])
            tsmpy = int(proj_mapping.iloc[pixel]["tsmp_yindex"])
            lon = proj_mapping.iloc[pixel]["lon"].replace("_", "")
            lat = proj_mapping.iloc[pixel]["lat"].replace("_", "")
            countrylocation = f"{country}_LON{lon}LAT{lat}"
            if not countrylocation in obswtd.columns:
                print(
                    f"countrylocation {countrylocation} not in obswtd columns, skipping...")
                continue
            target = int(proj_mapping.iloc[pixel]["target_chunk"])
            sim_1darrayindex = int(
                proj_mapping.iloc[pixel]["sim_1darrayindex"])
            sim_era5wtd = self.get_era5wtd(target, sim_1darrayindex)

            topography = topo[tsmpy, tsmpx]
            df_statdict["Topography"].append(topography)
            sm_pixel = soilmoisture[:, tsmpy, tsmpx]
            df_statdict["Soil moisture variance"].append(np.var(sm_pixel))
            precip_pixel = precip[:, tsmpy, tsmpx]
            df_statdict["Precipitation variance"].append(np.var(precip_pixel))
            vpd_pixel = vpd[:, tsmpy, tsmpx]
            df_statdict["VPD variance"].append(np.var(vpd_pixel))

            # rescale sims from daily to monthly timestep
            time = pd.date_range(start="2000-01-01", periods=5840, freq="D")
            df = pd.DataFrame(sim_era5wtd.T, index=time)   # shape (5840, 100)
            sim_era5wtd = df.resample(
                "MS").mean().T.to_numpy()  # shape (100, 192)

            # sim_era5wtd = sim_era5wtd[:, :-365]  # remove year 2020 to match localobs
            ens_mean_era5wtd = np.mean(sim_era5wtd, axis=0)
            localobs = np.array(obswtd[f"{countrylocation}"].to_list())
            # localobs = localobs[730:]  # remove year 2016 to match sim_era5wtd and tsmpobs

            # call the statistics function here:
            stats_dict = self.ensemble_statistics_scipy(
                sim_era5wtd, observations=localobs)
            stats_mean = {k: np.mean(v) if isinstance(
                v, np.ndarray) else v for k, v in stats_dict.items()}

            df_statdict["Ensemble mean"].append(stats_mean["Ensemble mean"])
            df_statdict["Ensemble median"].append(
                stats_mean["Ensemble median"])
            df_statdict["Skewness"].append(stats_mean["Skewness"])
            df_statdict["Kurtosis"].append(stats_mean["Kurtosis"])
            df_statdict["Mean 5th quantile"].append(
                stats_mean["Mean 5th quantile"])
            df_statdict["Mean 95th quantile"].append(
                stats_mean["Mean 95th quantile"])
            df_statdict["Mean 50th quantile"].append(
                stats_mean["Mean 50th quantile"])

            ensemble_predictions = sim_era5wtd  # shape (members, timeseries)
            # move axis to (timeseries, members) for easier calculation
            ensemble_predictions = np.moveaxis(
                ensemble_predictions, 0, 1)  # shape (timeseries, members)
            # important to set negatives to zero ##############
            ensemble_predictions[ensemble_predictions < 0.0] = 0.0
            # check lengths
            if len(localobs) != sim_era5wtd.shape[1]:
                raise ValueError(
                    f"lengths do not match for pixel {countrylocation}, len localobs: {len(localobs)}, len sim_era5wtd: {sim_era5wtd.shape[1]}, skipping...")

            ########### Calculate ensemble statistics ###########
            # Calculate the variance for each time step
            ensemble_variance = np.var(ensemble_predictions, axis=1)
            ensemble_variance = np.mean(ensemble_variance)
            df_statdict["Ensemble variance"].append(ensemble_variance)

            # Calculate ensemble statistics (e.g., diversity, spread, etc.)
            # Spread interquantile range (IQR) between 75th and 25th percentiles
            ensemble_iqr = np.percentile(
                ensemble_predictions, 75, axis=1) - np.percentile(ensemble_predictions, 25, axis=1)  # IQR
            iqr_mean = np.mean(ensemble_iqr)
            df_statdict["IQR (75-25%)"].append(iqr_mean)

            # calculate MAD
            mad = utils.calculate_ensemble_mad(ensemble_predictions)
            df_statdict["mad"].append(mad)

            # Calculate the std for each time step
            ensemble_std = np.std(ensemble_predictions, axis=1)
            ensemble_std = np.mean(ensemble_std)
            df_statdict["std"].append(ensemble_std)

            # Calculate diversity by Pairwise correlation
            # Transpose to get members on rows
            correlation_matrix = np.corrcoef(ensemble_predictions.T)
            pairwisecorr = np.mean(correlation_matrix[np.triu_indices_from(
                correlation_matrix, k=1)])  # Compute diversity as 1 - average correlation
            df_statdict["Pairwise correlation"].append(pairwisecorr)

            # calculate coefficient of variation
            cv = ensemble_std/np.mean(ens_mean_era5wtd)
            df_statdict["cv"].append(cv)

            ########### Calculate accuracy metrics ###########
            # Calculate mean bias
            bias_mean = np.mean(ens_mean_era5wtd - localobs)
            df_metricdict["Absolute mean bias"].append(bias_mean)

            # Calculate bias/std
            bias_std = np.abs(
                np.mean(ens_mean_era5wtd - localobs))/ensemble_std
            df_metricdict["Bias_std"].append(bias_std)

            # Calculate mean absolute error
            mae = np.mean(np.abs(ens_mean_era5wtd - localobs))
            df_metricdict["MAE"].append(mae)

            ### remove bias from mean prediction ###
            # mean_prediction = mean_prediction - bias_mean

            # Calculate the correlation between the mean prediction and observation
            correlationobs = np.corrcoef(ens_mean_era5wtd, localobs)[0, 1]
            df_metricdict["Pearson correlation"].append(correlationobs)

            # Calculate Spearman's rank correlation
            spearman_corr, _ = stats.spearmanr(ens_mean_era5wtd, localobs)
            df_metricdict["Spearman correlation"].append(spearman_corr)

            # Calculate Kendall's tau
            kendall_tau, _ = stats.kendalltau(ens_mean_era5wtd, localobs)
            df_metricdict["Kendall's tau"].append(kendall_tau)

            # Calculate the concordance correlation coefficient (CCC)
            ccc_numerator = 2 * correlationobs * \
                np.std(ens_mean_era5wtd) * np.std(localobs)
            ccc_denominator = np.var(ens_mean_era5wtd) + np.var(localobs) + \
                (np.mean(ens_mean_era5wtd) - np.mean(localobs)) ** 2
            ccc = ccc_numerator / ccc_denominator
            df_metricdict["CCC"].append(ccc)

            # Calculate Pearson Correlation p value
            _, p_value = stats.pearsonr(ens_mean_era5wtd, localobs)
            df_metricdict["Pearson p-value"].append(p_value)

            # Calculate the Median Absolute Error
            median_ae = np.median(np.abs(localobs - ens_mean_era5wtd))
            df_metricdict["Median AE"].append(median_ae)

            # Calculate Mean Absolute Percentage Error
            mape = np.mean(
                np.abs((localobs - ens_mean_era5wtd) / localobs)) * 100
            df_metricdict["MAPE"].append(mape)

            # Calculate Symmetric Mean Absolute Percentage Error
            smape = 100/len(localobs) * np.sum(2 * np.abs(ens_mean_era5wtd -
                                                          localobs) / (np.abs(localobs) + np.abs(ens_mean_era5wtd)))
            df_metricdict["SMAPE"].append(smape)

            # Calculate Murphy’s Mean Square Error Skill Score
            mse_model = np.mean((localobs - ens_mean_era5wtd) ** 2)
            mse_climatology = np.mean((localobs - np.mean(localobs)) ** 2)
            skill_score = 1 - (mse_model / mse_climatology)
            df_metricdict["Murphy Skill Score"].append(skill_score)

            # Calculate Bias Slope
            bias_slope, _, _, _, _ = stats.linregress(
                localobs, ens_mean_era5wtd)
            df_metricdict["Bias Slope"].append(bias_slope)

            # Calculate Conditional Bias
            conditional_bias = np.mean(ens_mean_era5wtd - localobs)
            df_metricdict["Conditional Bias"].append(conditional_bias)

            # Calculate Unconditional Bias
            unconditional_bias = np.mean(ens_mean_era5wtd) - np.mean(localobs)
            df_metricdict["Unconditional Bias"].append(unconditional_bias)

            # Calculate Mean Square Error Skill Score
            mse = np.mean((localobs - ens_mean_era5wtd) ** 2)
            mse_clim = np.mean((localobs - np.mean(localobs)) ** 2)
            mse_skill_score = 1 - (mse / mse_clim)
            df_metricdict["MSE Skill Score"].append(mse_skill_score)

            # Calculate Normalized Mean Square Error
            nmse = mse / ensemble_variance
            df_metricdict["NMSE"].append(nmse)

            # Normalized Root Mean Square Error
            nrmse = np.sqrt(mse) / np.sqrt(ensemble_variance)
            df_metricdict["NRMSE"].append(nrmse)

            # Normalized Mean Absolute Error
            nmae = mae / np.sqrt(ensemble_variance)
            df_metricdict["NMAE"].append(nmae)

            # Unbiased Anomaly Correlation Coefficient
            localobs_anom = np.sqrt(skill_score)
            df_metricdict["Unbiased Anomaly Correlation Coefficient"].append(
                localobs_anom)

            # Multiplicative bias
            multiplicative_bias = np.mean(ens_mean_era5wtd/localobs)
            df_metricdict["Multiplicative Bias"].append(multiplicative_bias)

            # Logarithmic Ensemble Spread Score
            less = np.log(ensemble_variance / mse)
            df_metricdict["LESS"].append(less)

            # Calculate the RMSE between the mean prediction and observation
            rmse = np.sqrt(np.mean((localobs - ens_mean_era5wtd) ** 2))
            df_metricdict["RMSE"].append(rmse)

            # Calculate KGE
            kge = utils.calculate_kge(localobs, ens_mean_era5wtd)
            df_metricdict["KGE"].append(kge)

            # members correlation
            members_correlation = [np.corrcoef(ensemble_predictions[:, m], localobs)[
                0, 1] for m in range(ensemble_predictions.shape[1])]
            members_correlation_mean = np.mean(members_correlation)
            df_metricdict["Members correlation"].append(
                members_correlation_mean)
            members_correlation_std = np.std(members_correlation)
            df_metricdict["Members correlation std"].append(
                members_correlation_std)
            if members_correlation_mean == correlationobs:
                raise ValueError(
                    "members mean correlation is the same as correlationobs, check your code")

            # members RMSE
            members_rmse = np.mean(
                np.sqrt(np.mean((ensemble_predictions - localobs[:, None]) ** 2, axis=0)))
            df_metricdict["Members RMSE"].append(members_rmse)
            members_rmse_std = np.std(
                np.sqrt(np.mean((ensemble_predictions - localobs[:, None]) ** 2, axis=0)))
            df_metricdict["Members RMSE std"].append(members_rmse_std)
            if members_rmse == rmse:
                raise ValueError(
                    "members mean RMSE is the same as RMSE, check your code")

            # members KGE
            members_kge = [utils.calculate_kge(
                ensemble_predictions[:, m], localobs) for m in range(ensemble_predictions.shape[1])]
            members_kge_mean = np.mean(members_kge)
            df_metricdict["Members KGE"].append(members_kge_mean)
            members_kge_std = np.std(members_kge)
            df_metricdict["Members KGE std"].append(members_kge_std)
            if members_kge_mean == kge:
                raise ValueError(
                    "members mean KGE is the same as KGE, check your code")

            ### KGE terms analysis ####
            # Compute mean and standard deviation
            mu_o, mu_p = np.mean(localobs), np.mean(ens_mean_era5wtd)
            sigma_o, sigma_p = np.std(localobs), np.std(ens_mean_era5wtd)

            ### KGE' ###
            kgeprime = utils.kge_prime(localobs, ens_mean_era5wtd)
            df_metricdict["KGE'"].append(kgeprime)

            # Compute bias ratio (β) and variability ratio (γ)
            beta = mu_p / mu_o
            alpha = sigma_p / sigma_o
            kge_nobias = 1 - np.sqrt((correlationobs - 1)**2 + (alpha - 1)**2)

            # Store statistics and accuracy metrics
            df_metricdict["KGE_nobias"].append(kge_nobias)
            df_statdict["stdsim"].append(np.std(ens_mean_era5wtd))
            df_metricdict["stdobs"].append(np.std(localobs))
            df_statdict["meansim"].append(np.mean(ens_mean_era5wtd))
            df_metricdict["meanobs"].append(np.mean(localobs))

            df_metricdict["Beta"].append(beta)
            df_metricdict["Alpha"].append(alpha)
            df_metricdict["(Alpha-1)^2"].append((alpha-1)**2)
            df_metricdict["(Beta-1)^2"].append((beta-1)**2)
            df_metricdict["(r-1)^2"].append((correlationobs-1)**2)

            # Calculate NSE
            nse = 1 - (np.sum((localobs - ens_mean_era5wtd) ** 2) /
                       np.sum((localobs - np.mean(localobs)) ** 2))
            df_metricdict["NSE"].append(nse)

        ####### save to csv ########
        df_metric = pd.DataFrame(df_metricdict)
        df_stat = pd.DataFrame(df_statdict)
        df_metric.to_csv(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean",
                         "era5wtd_vs_localobs", "statistics", "performance_metrics.csv"), index=False)
        df_stat.to_csv(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean",
                       "era5wtd_vs_localobs", "statistics", "ensemble_statistics.csv"), index=False)

    def ensemble_statvsacc_fitting(self):
        utils = utilities()
        stat = pd.read_csv(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px",
                           "ensemble_mean", "era5wtd_vs_localobs", "statistics", "ensemble_statistics.csv"))
        metric = pd.read_csv(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px",
                             "ensemble_mean", "era5wtd_vs_localobs", "statistics", "performance_metrics.csv"))
        xstats = {"VPD variance": "lin", "Precipitation variance": "lin", "Soil moisture variance": "lin", "Ensemble mean": "lin", "Ensemble median": "lin", "Skewness": "lin", "Kurtosis": "lin",
                  "Mean 5th quantile": "lin", "Mean 95th quantile": "lin", "Mean 50th quantile": "lin", "Topography": "lin", "std": "exp", "Ensemble variance": "exp", "Pairwise correlation": "lin", "IQR (75-25%)": "exp", "cv": "exp", "mad": "exp"}
        ymetrics = {"Spearman correlation": "lin", "Kendall's tau": "lin", "Pearson p-value": "lin", "CCC": "exp", "Median AE": "exp",
                    "MAPE": "lin", "SMAPE": "lin", "Murphy Skill Score": "lin", "Bias Slope": "lin", "Conditional Bias": "lin", "Unconditional Bias": "lin",
                    "MSE Skill Score": "lin", "NMSE": "exp", "NRMSE": "exp", "NMAE": "exp", "Unbiased Anomaly Correlation Coefficient": "lin",
                    "Multiplicative Bias": "lin", "LESS": "lin", "Bias_std": "exp", "Members correlation": "lin", "Members RMSE": "exp", "Members KGE": "lin",
                    "Members correlation std": "exp", "Members RMSE std": "exp", "Members KGE std": "exp",
                    "RMSE": "exp", "Pearson correlation": "lin", "KGE": "lin", "KGE_nobias": "exp", "Absolute mean bias": "exp", "NSE": "lin", "Beta": "exp",
                    "Alpha": "exp", "(Alpha-1)^2": "exp", "(Beta-1)^2": "exp", "(r-1)^2": "exp"}
        plotmapping = {
            "Ensemble variance": r"$\overline{EV}$",
            "IQR (75-25%)": r"$\overline{IQR}$",
            "Beta": r"$\beta$",
            "Alpha": r"$\alpha$",
            "(Alpha-1)^2": r"$(\alpha-1)^2$",
            "(Beta-1)^2": r"$(\beta-1)^2$",
        }
        for xstat in xstats.keys():
            for ymetric in ymetrics.keys():
                print(f"fitting {xstat} with {ymetric}")

                logarithmicfit = False
                if xstats[xstat] == "exp" and ymetrics[ymetric] == "exp":
                    fittype = "powerlaw"
                elif xstats[xstat] == "exp" or ymetrics[ymetric] == "exp":
                    fittype = "exponential"
                    if xstats[xstat] == "exp" and ymetrics[ymetric] == "lin":
                        logarithmicfit = True
                else:
                    fittype = "linear"

                xval = stat[xstat].values
                yval = metric[ymetric].values if not ymetric == "Absolute mean bias" else np.absolute(
                    metric[ymetric].values)

                if ymetric == "Pearson correlation" or ymetric == "Members correlation":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                    # xval = xval[yval>=0.0]
                    # yval = yval[yval>=0.0]

                if ymetric == "KGE" or ymetric == "NSE" or ymetric == "KGE_nobias" or ymetric == "Members KGE":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]
                    xval = xval[yval >= 0.2]
                    yval = yval[yval >= 0.2]

                if ymetric == "Beta" or ymetric == "Alpha" or ymetric == "(Alpha-1)^2" or ymetric == "(Beta-1)^2" or ymetric == "(r-1)^2":
                    xval = xval[~np.isnan(yval)]
                    yval = yval[~np.isnan(yval)]

                if ymetric == "Absolute mean bias":
                    xval = xval[yval >= 0.01]
                    yval = yval[yval >= 0.01]

                if xstat == "Pairwise correlation":
                    yval = yval[xval >= 0.0]
                    xval = xval[xval >= 0.0]

                logx, logy, xminlim, xmaxlim, yminlim, ymaxlim = None, None, None, None, None, None
                if xstat == "std" or xstat == "Ensemble variance" or xstat == "IQR (75-25%)" or xstat == "cv" or xstat == "mad":
                    logx = True
                if ymetric == "Members correlation std" or ymetric == "Members RMSE std" or ymetric == "Members KGE std" or ymetric == "RMSE" or ymetric == "Absolute mean bias" or ymetric == "Alpha" or ymetric == "Beta" or ymetric == "(Alpha-1)^2" or ymetric == "(Beta-1)^2" or ymetric == "(r-1)^2" or ymetric == "Members RMSE" or ymetric == "Bias_std":
                    logy = True
                if ymetrics[ymetric] == "exp":
                    logy = True
                if xstats[xstat] == "exp":
                    logx = True
                if xstat == "Pairwise correlation":
                    xminlim = 0
                    xmaxlim = 1
                if ymetric == "Pearson correlation":
                    yminlim = -1
                    ymaxlim = 1
                if ymetric == "KGE" or ymetric == "NSE":
                    yminlim = 0.2
                    ymaxlim = 1

                if xstat == "Pairwise correlation" and ymetric == "Members correlation std":
                    fittype = "powerlaw"
                    logx = True
                    logy = True
                try:
                    if fittype == "linear":
                        # Fit a linear model
                        params, covariance = curve_fit(
                            utils.linear_law, xval, yval)
                        a_fit, b_fit = params
                        # fitting accuracy
                        y_fit = [utils.linear_law(x, a_fit, b_fit)
                                 for x in xval]
                        R_square = r2_score(yval, y_fit)
                        # print(f"R2 = {R_square}")
                        # print(f"Fitted a: {a_fit} and b:{b_fit}")
                        lin_eq = f'$y={b_fit:.2f}x+{a_fit:.2f}$' if a_fit >= 0 else f'$y={b_fit:.2f}x{a_fit:.2f}$'
                        textstr = '\n'.join((
                            lin_eq,
                            f'$R^2$ = {R_square:.3f}'))
                        x_fit = [min(xval), max(xval)]
                        y_fit = [utils.linear_law(x, a_fit, b_fit)
                                 for x in x_fit]

                    if fittype == "powerlaw":
                        # Fit power law
                        # linearize
                        y_lin = np.log(yval)
                        x_lin = np.log(xval)
                        # Fit the function
                        params, covariance = curve_fit(
                            utils.linear_law, x_lin, y_lin)
                        a_fit, b_fit = params
                        # fitting accuracy
                        y_fit = [utils.linear_law(x, a_fit, b_fit)
                                 for x in x_lin]
                        R_square = r2_score(y_lin, y_fit)
                        # print(f"R2 = {R_square}")
                        # back transform to power law
                        a_fit = np.exp(a_fit)
                        # print(f"Fitted a: {a_fit} and b:{b_fit}")
                        textstr = '\n'.join((
                            f'y={a_fit:.2f}x^{b_fit:.2f}',
                            f'$R^2$ = {R_square:.3f}'))

                        x_fit = [min(xval), max(xval)]
                        y_fit = [utils.powerlaw_func(
                            x, a_fit, b_fit) for x in x_fit]

                    if fittype == "exponential":
                        # Define a function to fit (e.g., exponential decay or polynomial)
                        def log_func(x, a, b):
                            return a + b * np.log(x)

                        def exp_func(x, a, b):
                            # Exponential model without offset
                            return a * np.exp(b * x)

                        # Fit the curve
                        if logarithmicfit:
                            popt, _ = curve_fit(log_func, xval, yval)
                            a_fit, b_fit = popt
                            y_pred = log_func(xval, *popt)
                            r2 = r2_score(yval, y_pred)  # Compute R²

                            # Generate fitted values
                            x_fit = np.linspace(min(xval), max(xval), 100)
                            y_fit = log_func(x_fit, *popt)
                            logeq = f'y={a_fit:.2f}+{b_fit:.3f}log(x)' if b_fit >= 0 else f'y={a_fit:.2f}{b_fit:.3f}log(x)'
                            textstr = '\n'.join((
                                logeq,
                                f'$R^2$ = {r2:.3f}'))
                        else:
                            popt, _ = curve_fit(exp_func, xval, yval, p0=(
                                1, -0.1))  # Initial guesses for parameters
                            a_fit, b_fit = popt
                            # Fitted values for the actual x
                            y_pred = exp_func(xval, *popt)

                            r2 = r2_score(yval, y_pred)  # Compute R²

                            # Generate fitted values
                            x_fit = np.linspace(min(xval), max(xval), 100)
                            y_fit = exp_func(x_fit, *popt)

                            textstr = '\n'.join((
                                f'y={a_fit:.2f}e^{b_fit:.3f}x',
                                f'$R^2$ = {r2:.3f}'))
                except:
                    print(
                        f"fitting failed for {xstat} with {ymetric}, skipping...")
                    continue

                plt.figure()
                plt.plot(x_fit, y_fit, color="r",
                         label=textstr, linestyle='--')
                plt.scatter(xval, yval, marker='.', color='k')
                if xstat in plotmapping.keys():
                    plt.xlabel(plotmapping[xstat])
                else:
                    plt.xlabel(xstat)
                if ymetric in plotmapping.keys():
                    plt.ylabel(plotmapping[ymetric])
                else:
                    plt.ylabel(ymetric)
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
                plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean",
                            "era5wtd_vs_localobs", "statistics", f"fitted_{ymetric}_{xstat}.png"))
                plt.close()

    def onepoint_distribution(self):
        utils = utilities()
        topo = np.load(os.path.join(INPUTPATH, "topo.npy"))
        topo = topo[0, :, :]
        soilmoisture = np.load(os.path.join(INPUTPATH, "swvl3_EU.npy"))
        precip = np.load(os.path.join(INPUTPATH, "tp_EU.npy"))
        vpd = np.load(os.path.join(INPUTPATH, "vpd_EU.npy"))

        obswtd = pd.read_parquet(os.path.join(os.path.join(
            INPUTPATH, "localobservations"), "wtd_obsEU_monthly_avgdup.parquet"))
        obswtd = obswtd.dropna(axis=1)  # drop columns with all NaN values
        proj_mapping = pd.read_csv(os.path.join(
            INPUTPATH, "localobservations", "mapping_wtdobsEU_TSMP_avgdup.csv"))
        proj_mapping = proj_mapping.dropna(subset=["sim_1darrayindex"])
        proj_mapping.reset_index(drop=True, inplace=True)
        df_dic = {"Bias_std": [], "VPD variance": [], "Precipitation variance": [], "Soil moisture variance": [], "Ensemble mean": [], "Ensemble median": [], "Skewness": [], "Kurtosis": [], "Mean 5th quantile": [], "Mean 95th quantile": [], "Mean 50th quantile": [], "Topography": [], "Members RMSE": [], "Members correlation": [], "Members KGE": [], "KGE_nobias": [
        ], "stdsim": [], "stdobs": [], "meansim": [], "meanobs": [], "Absolute mean bias": [], "Pearson correlation": [], "RMSE": [], "KGE": [], "KGE'": [], "Beta": [], "Alpha": [], "NSE": [], "Pairwise correlation": [], "Ensemble variance": [], "IQR (75-25%)": [], "std": [], "cv": [], "mad": [],  "(Alpha-1)^2": [], "(Beta-1)^2": [], "(r-1)^2": []}
        for pixel in range(len(proj_mapping)):
            print(
                f"Processing pixel {pixel+1} of {len(proj_mapping)}", end='\r')
            # countrylocation = proj_mapping.iloc[pixel]["countrylocation"]
            country = proj_mapping.iloc[pixel]["country"]
            tsmpx = int(proj_mapping.iloc[pixel]["tsmp_xindex"])
            tsmpy = int(proj_mapping.iloc[pixel]["tsmp_yindex"])
            lon = proj_mapping.iloc[pixel]["lon"].replace("_", "")
            lat = proj_mapping.iloc[pixel]["lat"].replace("_", "")
            countrylocation = f"{country}_LON{lon}LAT{lat}"
            if not countrylocation == "Germany_LON11.452182142638508LAT52.33930908026621":
                continue
            if not countrylocation in obswtd.columns:
                print(
                    f"countrylocation {countrylocation} not in obswtd columns, skipping...")
                continue
            target = int(proj_mapping.iloc[pixel]["target_chunk"])
            sim_1darrayindex = int(
                proj_mapping.iloc[pixel]["sim_1darrayindex"])
            sim_era5wtd = self.get_era5wtd(target, sim_1darrayindex)

            topography = topo[tsmpy, tsmpx]
            df_dic["Topography"].append(topography)
            sm_pixel = soilmoisture[:, tsmpy, tsmpx]
            df_dic["Soil moisture variance"].append(np.var(sm_pixel))
            precip_pixel = precip[:, tsmpy, tsmpx]
            df_dic["Precipitation variance"].append(np.var(precip_pixel))
            vpd_pixel = vpd[:, tsmpy, tsmpx]
            df_dic["VPD variance"].append(np.var(vpd_pixel))

            # rescale sims from daily to monthly timestep
            time = pd.date_range(start="2000-01-01", periods=5840, freq="D")
            df = pd.DataFrame(sim_era5wtd.T, index=time)   # shape (5840, 100)
            sim_era5wtd = df.resample(
                "MS").mean().T.to_numpy()  # shape (100, 192)

            # sim_era5wtd = sim_era5wtd[:, :-365]  # remove year 2020 to match localobs
            ens_mean_era5wtd = np.mean(sim_era5wtd, axis=0)
            localobs = np.array(obswtd[f"{countrylocation}"].to_list())
            # localobs = localobs[730:]  # remove year 2016 to match sim_era5wtd and tsmpobs

            # call the statistics function here:
            stats_dict = self.ensemble_statistics(
                sim_era5wtd, observations=localobs)
            stats_mean = {k: np.mean(v) if isinstance(
                v, np.ndarray) else v for k, v in stats_dict.items()}

            df_dic["Ensemble mean"].append(stats_mean["Ensemble mean"])
            df_dic["Ensemble median"].append(stats_mean["Ensemble median"])
            df_dic["Skewness"].append(stats_mean["Skewness"])
            df_dic["Kurtosis"].append(stats_mean["Kurtosis"])
            df_dic["Mean 5th quantile"].append(stats_mean["Mean 5th quantile"])
            df_dic["Mean 95th quantile"].append(
                stats_mean["Mean 95th quantile"])
            df_dic["Mean 50th quantile"].append(
                stats_mean["Mean 50th quantile"])

            ensemble_predictions = sim_era5wtd  # shape (members, timeseries)
            # move axis to (timeseries, members) for easier calculation
            ensemble_predictions = np.moveaxis(
                ensemble_predictions, 0, 1)  # shape (timeseries, members)
            # important to set negatives to zero ##############
            ensemble_predictions[ensemble_predictions < 0.0] = 0.0
            # check lengths
            if len(localobs) != sim_era5wtd.shape[1]:
                raise ValueError(
                    f"lengths do not match for pixel {countrylocation}, len localobs: {len(localobs)}, len sim_era5wtd: {sim_era5wtd.shape[1]}, skipping...")

            ########### Calculate ensemble statistics ###########
            # Calculate the variance for each time step
            ensemble_variance = np.var(ensemble_predictions, axis=1)
            ensemble_variance = np.mean(ensemble_variance)
            df_dic["Ensemble variance"].append(ensemble_variance)

            # Calculate ensemble statistics (e.g., diversity, spread, etc.)
            # Spread interquantile range (IQR) between 75th and 25th percentiles
            ensemble_iqr = np.percentile(
                ensemble_predictions, 75, axis=1) - np.percentile(ensemble_predictions, 25, axis=1)  # IQR
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
            # Transpose to get members on rows
            correlation_matrix = np.corrcoef(ensemble_predictions.T)
            pairwisecorr = np.mean(correlation_matrix[np.triu_indices_from(
                correlation_matrix, k=1)])  # Compute diversity as 1 - average correlation
            df_dic["Pairwise correlation"].append(pairwisecorr)

            # calculate coefficient of variation
            cv = ensemble_std/np.mean(ens_mean_era5wtd)
            df_dic["cv"].append(cv)

            # Calculate bias/std
            bias_std = np.abs(
                np.mean(ens_mean_era5wtd - localobs))/ensemble_std
            df_dic["Bias_std"].append(bias_std)

            ########### Calculate accuracy metrics ###########
            # Calculate mean bias
            bias_mean = np.mean(ens_mean_era5wtd - localobs)
            df_dic["Absolute mean bias"].append(bias_mean)

            ### remove bias from mean prediction ###
            # mean_prediction = mean_prediction - bias_mean

            # Calculate the correlation between the mean prediction and observation
            correlationobs = np.corrcoef(ens_mean_era5wtd, localobs)[0, 1]
            df_dic["Pearson correlation"].append(correlationobs)

            # Calculate the RMSE between the mean prediction and observation
            rmse = np.sqrt(np.mean((localobs - ens_mean_era5wtd) ** 2))
            df_dic["RMSE"].append(rmse)

            # Calculate KGE
            kge = utils.calculate_kge(localobs, ens_mean_era5wtd)
            df_dic["KGE"].append(kge)

            # members correlation
            members_correlation = [np.corrcoef(ensemble_predictions[:, m], localobs)[
                0, 1] for m in range(ensemble_predictions.shape[1])]
            members_correlation_mean = np.mean(members_correlation)
            df_dic["Members correlation"].append(members_correlation_mean)
            if members_correlation_mean == correlationobs:
                raise ValueError(
                    "members mean correlation is the same as correlationobs, check your code")

            # members RMSE
            members_rmse = np.mean(
                np.sqrt(np.mean((ensemble_predictions - localobs[:, None]) ** 2, axis=0)))
            df_dic["Members RMSE"].append(members_rmse)
            if members_rmse == rmse:
                raise ValueError(
                    "members mean RMSE is the same as RMSE, check your code")

            # members KGE
            members_kge = [utils.calculate_kge(
                ensemble_predictions[:, m], localobs) for m in range(ensemble_predictions.shape[1])]
            members_kge_mean = np.mean(members_kge)
            df_dic["Members KGE"].append(members_kge_mean)
            if members_kge_mean == kge:
                raise ValueError(
                    "members mean KGE is the same as KGE, check your code")

            ### KGE terms analysis ####
            # Compute mean and standard deviation
            mu_o, mu_p = np.mean(localobs), np.mean(ens_mean_era5wtd)
            sigma_o, sigma_p = np.std(localobs), np.std(ens_mean_era5wtd)

            ### KGE' ###
            kgeprime = utils.kge_prime(localobs, ens_mean_era5wtd)
            df_dic["KGE'"].append(kgeprime)

            # Compute bias ratio (β) and variability ratio (γ)
            beta = mu_p / mu_o
            alpha = sigma_p / sigma_o
            kge_nobias = 1 - np.sqrt((correlationobs - 1)**2 + (alpha - 1)**2)

            # Store statistics and accuracy metrics
            df_dic["KGE_nobias"].append(kge_nobias)
            df_dic["stdsim"].append(np.std(ens_mean_era5wtd))
            df_dic["stdobs"].append(np.std(localobs))
            df_dic["meansim"].append(np.mean(ens_mean_era5wtd))
            df_dic["meanobs"].append(np.mean(localobs))

            df_dic["Beta"].append(beta)
            df_dic["Alpha"].append(alpha)
            df_dic["(Alpha-1)^2"].append((alpha-1)**2)
            df_dic["(Beta-1)^2"].append((beta-1)**2)
            df_dic["(r-1)^2"].append((correlationobs-1)**2)

            # Calculate NSE
            nse = 1 - (np.sum((localobs - ens_mean_era5wtd) ** 2) /
                       np.sum((localobs - np.mean(localobs)) ** 2))
            df_dic["NSE"].append(nse)

            # plot the ensemble member RMSE distribution
            plt.figure(figsize=(10, 5))
            plt.hist(np.sqrt(
                np.mean((ensemble_predictions - localobs[:, None]) ** 2, axis=0)), bins=50)
            # plt.hist(np.var(ensemble_predictions, axis=1), bins=50, alpha=0.5)
            plt.xlabel("RMSE")
            plt.ylabel("Frequency")
            plt.title("Distribution of Ensemble Member RMSEs")
            plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean",
                        "era5wtd_vs_localobs", "distribution_analysis", "ensemble_member_rmse_distribution.png"))
            plt.close()

    def cdf_EU(self):
        def calc_correlation(obs, sim):
            correlation_map = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    correlation_matrix = np.corrcoef(
                        time_series1, time_series2)
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
                    # if np.std(time_series1)>=0.1 else np.nan
                    kge = utils.calculate_kge(time_series1, time_series2)
                    # if np.std(time_series1)>=0.1 else np.nan
                    nse = 1 - (np.sum((time_series1 - time_series2) ** 2) /
                               np.sum((time_series1 - np.mean(time_series1)) ** 2))
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
            obs = np.load(os.path.join(
                dirpath, f"target_pixels_{target}", obsname))
            sims = np.load(os.path.join(
                dirpath, f"target_pixels_{target}", simname))
            obs = np.nan_to_num(obs)
            sims = np.nan_to_num(sims)
            obs[obs < 0.0] = 0.0
            sims[sims < 0.0] = 0.0
            return obs, sims

        utils = utilities()
        print("starting the cdf calculation")
        correlation = {"transfer_400px": [], "test_400px": [],
                       "transfer_ERA5": [], "test_ERA5": []}
        rmse = {"transfer_400px": [], "test_400px": [],
                "transfer_ERA5": [], "test_ERA5": []}
        kge = {"transfer_400px": [], "test_400px": [],
               "transfer_ERA5": [], "test_ERA5": []}
        nse = {"transfer_400px": [], "test_400px": [],
               "transfer_ERA5": [], "test_ERA5": []}
        bias = {"transfer_400px": [], "test_400px": [],
                "transfer_ERA5": [], "test_ERA5": []}

        # TODO plot cdf of ERA5 test and transfer evaluation compared with observations
        # TODO plot with it the TSMP test and transfer evaluation compared with observations

        # TODO select timeseries only for january 2020

        #### Transfer & training filtered subsets ####
        ERA5_inpath = os.path.join(INPUTPATH, "validation_ERA5")
        EU400px_inpath = os.path.join(
            "/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/inputs/20yrs_ts", "ensemble_400px")
        training_subset = np.load(os.path.join(
            INPUTPATH, "training_subset.npy"))
        print("Progress: [" + "." * 100 + "]", flush=True)
        print("          [", end="", flush=True)  # Start progress bar
        for target in range(100):
            print(".", end="", flush=True)  # Dots without newlines
            ######## ERA5 ensemble ########
            # calculate transfer metrics for 400px ensemble
            obs_transferERA5, sim_transferERA5 = load_obs_sim(
                ERA5_inpath, target, f"obs_{target}.npy", f"sim_transferpixels_{target}.npy")
            obs_testERA5, sim_testERA5 = load_obs_sim(
                ERA5_inpath, target, f"obs_{target}.npy", f"sim_testtrainpixels_{target}.npy")
            target_map = np.load(os.path.join(
                ERA5_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            trainingERA5_indices, ind2d = utils.intersect_subsets(
                target_map, training_subset)
            corr_EU = calc_correlation(obs_transferERA5, sim_transferERA5)
            rmse_EU = calc_RMSE(obs_transferERA5, sim_transferERA5)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(
                obs_transferERA5, sim_transferERA5)
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
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(
                obs_testERA5, sim_testERA5)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from only 1 training member (training subset)
            correlation["test_ERA5"].extend(
                corr_EU[trainingERA5_indices].tolist())
            rmse["test_ERA5"].extend(rmse_EU[trainingERA5_indices].tolist())
            kge["test_ERA5"].extend(kge_EU[trainingERA5_indices].tolist())
            nse["test_ERA5"].extend(nse_EU[trainingERA5_indices].tolist())
            bias["test_ERA5"].extend(bias_EU[trainingERA5_indices].tolist())

            ######## 400px ensemble ########
            # calculate transfer metrics for 400px ensemble
            obs_transfer400, sim_transfer400 = load_obs_sim(
                EU400px_inpath, target, f"obs_{target}.npy", f"sim_transferpixels_{target}.npy")
            obs_test400, sim_test400 = load_obs_sim(
                EU400px_inpath, target, f"obs_{target}.npy", f"sim_testtrainpixels_{target}.npy")
            target_map = np.load(os.path.join(
                EU400px_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            training400px_indices, ind2d = utils.intersect_subsets(
                target_map, training_subset)
            corr_EU = calc_correlation(obs_transfer400, sim_transfer400)
            rmse_EU = calc_RMSE(obs_transfer400, sim_transfer400)
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(
                obs_transfer400, sim_transfer400)
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
            kge_EU, nse_EU, bias_EU = calc_KGE_NSE_bias(
                obs_test400, sim_test400)
            corr_EU = np.array(corr_EU)
            rmse_EU = np.array(rmse_EU)
            kge_EU = np.array(kge_EU)
            nse_EU = np.array(nse_EU)
            bias_EU = np.array(bias_EU)
            # append values from only 1 training member (training subset)
            correlation["test_400px"].extend(
                corr_EU[training400px_indices].tolist())
            rmse["test_400px"].extend(rmse_EU[training400px_indices].tolist())
            kge["test_400px"].extend(kge_EU[training400px_indices].tolist())
            nse["test_400px"].extend(nse_EU[training400px_indices].tolist())
            bias["test_400px"].extend(bias_EU[training400px_indices].tolist())

        if not sim_transferERA5.shape[0] == TEST_PERIOD-LOOKBACK or not sim_testERA5.shape[0] == TEST_PERIOD-LOOKBACK or not sim_transfer400.shape[0] == TEST_PERIOD-LOOKBACK or not sim_test400.shape[0] == TEST_PERIOD-LOOKBACK:
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

    def concat_EU_transfer_testtrain_outputs(self):
        utils = utilities()
        EU_inpath = os.path.join(INPUTPATH, "validation_ERA5")
        EU_outpath = os.path.join(
            OUTPUTPATH, "validation_ERA5", "ensemble_400px")
        for target in range(100):
            target_mapping = np.load(os.path.join(
                EU_inpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            obs_destand_EU = np.load(os.path.join(
                EU_outpath, f"400px_member_1", f"obs_{target}.npy"))
            members_sim = [np.load(os.path.join(
                EU_outpath, f"400px_member_{m}", f"sim_{target}.npy")) for m in range(100)]
            members_sim = np.array(members_sim)
            members_sim = np.expand_dims(members_sim, axis=0)
            # (members, timeseries, pixels)
            members_sim = np.concatenate((members_sim), axis=0)
            mean_prediction_transfer = np.zeros(
                (members_sim.shape[1], members_sim.shape[2]))
            # array for test train pixels
            mean_prediction_testtrain = np.zeros(
                (members_sim.shape[1], members_sim.shape[2]))
            mean_prediction_testtrain[mean_prediction_testtrain == 0] = np.nan

            untrained_pixel_mask = np.ones(members_sim.shape[2], dtype=bool)
            for member in range(100):
                member_mask = np.ones(100, dtype=bool)
                member_mask[member] = False
                pixel_mask = np.zeros(members_sim.shape[2], dtype=bool)
                # get the 2d mapping file of indices of pixels used in training this member
                member_mapping = np.load(os.path.join(
                    EU_inpath, f"400px_member_{member}", "choices.npy"))
                # call function here to find which pixels of my target chunk were included in the training of this specific member
                indices1d, indices2d = utils.intersect_subsets(
                    target_mapping, member_mapping)
                # chunk pixels included in training are set as True
                pixel_mask[indices1d] = True

                # take only the member and its relative pixels
                mean_prediction_testtrain[:,
                                          pixel_mask] = members_sim[member, :, :][..., pixel_mask]

                # keep track of pixels included in training any member
                untrained_pixel_mask[indices1d] = False
                # For pixels where pixel_mask=False (pixels included in training this member)
                # then get the Mean over SELECTED MEMBERS (member_mask=True) excluding the one False member (the loop member)
                mean_prediction_transfer[:, pixel_mask] = np.mean(
                    members_sim[member_mask, :, :][..., pixel_mask], axis=0)

            # For other pixels (not included in training any member) get the Mean over ALL MEMBERS (axis=0)
            mean_prediction_transfer[:, untrained_pixel_mask] = np.mean(
                members_sim[:, :, untrained_pixel_mask], axis=0)
            print(obs_destand_EU.shape)
            print(mean_prediction_transfer.shape)
            np.save(os.path.join(
                EU_inpath, f"target_pixels_{target}", f"sim_testtrainpixels_{target}.npy"), mean_prediction_testtrain)
            np.save(os.path.join(
                EU_inpath, f"target_pixels_{target}", f"obs_{target}.npy"), obs_destand_EU)
            np.save(os.path.join(
                EU_inpath, f"target_pixels_{target}", f"sim_transferpixels_{target}.npy"), mean_prediction_transfer)
            print(f"saved {target} target pixels")

    def map_1Dto2D_EU(self, dirpath):
        def calc_correlation(obs, sim):
            correlation_map = []
            for i in range(obs.shape[1]):
                time_series1 = obs[:, i]
                time_series2 = sim[:, i]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    correlation_matrix = np.corrcoef(
                        time_series1, time_series2)
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
            nse = 1 - (np.sum((obs - sim) ** 2) /
                       np.sum((obs - np.mean(obs)) ** 2))
            all_nse = []
            bias_mean = np.mean(sim - obs)
            all_bias = []
            for pixel in range(obs.shape[1]):
                time_series1 = obs[:, pixel]
                time_series2 = sim[:, pixel]
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:
                    # if np.std(time_series1)>=0.1 else np.nan
                    kge = utils.calculate_kge(time_series1, time_series2)
                    # if np.std(time_series1)>=0.1 else np.nan
                    nse = 1 - (np.sum((time_series1 - time_series2) ** 2) /
                               np.sum((time_series1 - np.mean(time_series1)) ** 2))
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
            obs_destand_test = np.load(os.path.join(
                dirpath, f"target_pixels_{target}", f"obs_ensmean.npy"))
            sim_destand_test = np.load(os.path.join(
                dirpath, f"target_pixels_{target}", f"sim_ensmean.npy"))
            obs_destand_test = np.nan_to_num(obs_destand_test)
            sim_destand_test = np.nan_to_num(sim_destand_test)
            obs_destand_test[obs_destand_test < 0.0] = 0.0
            sim_destand_test[sim_destand_test < 0.0] = 0.0
            return obs_destand_test, sim_destand_test

        utils = utilities()

        #### Transfer subset ####

        # filtered subset
        transfer_subset = np.load(os.path.join(
            INPUTPATH, "transfer_subset.npy"))
        training_subset = np.load(os.path.join(
            INPUTPATH, "training_subset.npy"))

        corr2d = np.zeros(transfer_subset.shape)
        corr2d[corr2d == 0] = np.nan
        rmse2d = np.zeros(transfer_subset.shape)
        rmse2d[rmse2d == 0] = np.nan
        bias2d = np.zeros(transfer_subset.shape)
        bias2d[bias2d == 0] = np.nan
        kge2d = np.zeros(transfer_subset.shape)
        kge2d[kge2d == 0] = np.nan
        nse2d = np.zeros(transfer_subset.shape)
        nse2d[nse2d == 0] = np.nan
        for target in range(100):
            obs_destand_test, sim_destand_test = load_obs_sim(target)
            target_map = np.load(os.path.join(
                dirpath, f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            indices, indices_2d = utils.intersect_subsets(
                target_map, transfer_subset)
            indices_train, indices_2d_train = utils.intersect_subsets(
                target_map, training_subset)

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
        tsmpdirpath = os.path.join(os.path.dirname(os.path.dirname(
            get_root_dir())), "spatio-temporal-LSTM", "inputs", "20yrs_ts", "ensemble_400px")
        era5corr2d, era5rmse2d, era5bias2d, era5kge2d, era5nse2d = self.map_1Dto2D_EU(
            era5dirpath)
        tsmpcorr2d, tsmprmse2d, tsmpbias2d, tsmpkge2d, tsmpnse2d = self.map_1Dto2D_EU(
            tsmpdirpath)

        # TODO plot for KGE<0.2 from the tsmp not era5
        # get indices where kge<0.2
        era5kge2d = np.where(tsmpkge2d < 0.2, np.nan, era5kge2d)
        print(
            f"number of pixels with KGE>=0.2: {np.sum(~np.isnan(era5kge2d))}")
        era5nse2d = np.where(np.isnan(era5kge2d), np.nan, era5nse2d)
        era5corr2d = np.where(np.isnan(era5kge2d), np.nan, era5corr2d)
        era5rmse2d = np.where(np.isnan(era5kge2d), np.nan, era5rmse2d)
        era5bias2d = np.where(np.isnan(era5kge2d), np.nan, era5bias2d)
        # bias2d = np.where(kge2d<0.2, np.nan, bias2d)

        plot_functions.EU_2Dmap(data_map=era5corr2d, logscale=False,
                                minval=0, maxval=1, title="Pearson correlation")
        plot_functions.EU_2Dmap(
            data_map=era5rmse2d, logscale=True, minval=0.01, maxval=10, title="RMSE")
        plot_functions.EU_2Dmap(
            data_map=era5bias2d, logscale=False, minval=-10, maxval=10, title="Mean bias")
        plot_functions.EU_2Dmap(
            data_map=era5kge2d, logscale=False, minval=0.2, maxval=1, title="KGE")
        plot_functions.EU_2Dmap(
            data_map=era5nse2d, logscale=False, minval=-1, maxval=1, title="NSE")

    def metrics_vs_topo(self):
        ###### comment out the training pixels from self.map_1Dto2D_EU() ######
        # take only the transfer ones
        # change topo for wtd and select the average wtd for year 2020
        corr2d, rmse2d, bias2d, kge2d, nse2d = self.map_1Dto2D_EU()
        plot_functions = plotting_helper()
        topo_v1 = np.load(os.path.join(os.path.dirname(
            os.path.dirname(INPUTPATH)), "topo.npy"))
        topo_v1 = topo_v1[0, :, :]
        # topo_v1 = np.mean(topo_v1[-365:,:,:], axis=0)  # average wtd for year 2020

        topo = topo_v1
        topo[np.isnan(corr2d)] = np.nan
        plot_functions.plot_scatter(topo, corr2d, "Pearson correlation")
        topo = topo_v1
        topo[np.isnan(rmse2d)] = np.nan
        plot_functions.plot_scatter(topo, rmse2d, "RMSE", ylog=True)
        topo = topo_v1
        topo[np.isnan(bias2d)] = np.nan
        plot_functions.plot_scatter(
            topo, bias2d, "Mean absolute bias (m)", ylog=True)
        topo = topo_v1
        topo[np.isnan(kge2d)] = np.nan
        plot_functions.plot_scatter(
            topo, kge2d, "KGE", ylog=False, ysymlog=True)
        topo = topo_v1
        topo[np.isnan(nse2d)] = np.nan
        plot_functions.plot_scatter(
            topo, nse2d, "NSE", ylog=False, ysymlog=True)

    def plot_topo(self):
        plot_functions = plotting_helper()
        wtd = np.load(os.path.join(os.path.dirname(
            os.path.dirname(INPUTPATH)), "topo.npy"))
        print(wtd.shape)
        # exclude sides
        wtd[:, :100, :] = 0
        wtd[:, 432-10:, :] = 0
        wtd[:, :, 444-10:] = 0
        wtd[:, :, :10] = 0

        # set negative wtd to 0
        wtd[wtd < 0] = 0

        wtd = wtd[0, :, :]
        print(wtd.shape)
        maxwtd = np.nanmax(wtd)
        print(maxwtd)
        wtd[wtd == 0] = np.nan
        plot_functions.EU_2Dmap(wtd, logscale=False,
                                minval=0.01, maxval=maxwtd, title="Topography")

    def avgwtd_2020(self):
        plots = plotting_helper()
        wtd = np.load(os.path.join(os.path.dirname(
            os.path.dirname(INPUTPATH)), "wtd.npy"))
        mapping = np.load(os.path.join(os.path.dirname(os.path.dirname(
            INPUTPATH)), "ensemble_400px_org", "mapping_0stdroll6months.npy"))
        print(np.sum(mapping))
        print(wtd.shape)
        wtd = wtd[-365:, :, :]
        wtd = np.mean(wtd, axis=0)
        print(wtd.shape)
        wtd = np.where(mapping == 1, wtd, np.nan)
        print(wtd.shape)
        # plot it in 2d map
        plots.EU_2Dmap(data_map=wtd, logscale=False, minval=0,
                       maxval=50, title="Water table depth")

    ###################### obs processing #########################

    def simulations_to_repeat(self):
        # run script chechmoddate.sh first to get the outdated files
        logfile = open(os.path.join(get_root_dir(), "outdated_files.log"), "r")
        lines = logfile.readlines()
        logfile.close()
        orgscript = open(os.path.join(
            get_root_dir(), "onejobbooster_ddp_eval.sh"), "r")
        orglines = orgscript.readlines()
        orgscript.close()
        members_from = {}
        for line in lines:
            member = line.split("400px_member_")[1].split("//")[0]
            batchind = line.split("sim_")[1].replace(".npy", "")
            if member not in members_from.keys():
                members_from[member] = []
            members_from[member].append(int(batchind))
        print(members_from)
        for member in members_from.keys():
            newscript = open(os.path.join(
                get_root_dir(), "repeats", f"m{member}onejobbooster_ddp_eval.sh"), "w")
            for orgline in orglines:
                if "srun" in orgline:
                    for batchind in members_from[member]:
                        newscript.write(orgline.replace("82", f"{batchind}"))
                        newscript.write("\n")
                else:
                    newscript.write(orgline)
            newscript.close()

    def era5wtdensemble_localobs_timeseries(self, obslocation, sim_era5wtd, mean_prediction, tsmpobs, localobs, kge, rmse, r, bias, nse):
        dates = pd.date_range(start='2000-01-01', end='2015-12-31', freq='MS')
        fig, ax = plt.subplots(figsize=(16, 10))

        kge = round(kge, 2)
        r = round(r, 2)
        rmse = round(rmse, 2)
        bias = round(bias, 2)
        nse = round(nse, 2)

        # print(f"plotting timeseries {obslocation} with KGE: {kge} , r: {r}, RMSE: {rmse}, Bias: {bias}, NSE: {nse}")
        for m in range(100):
            sim = sim_era5wtd[m, :]
            ax.plot(dates, sim, color="gray", alpha=0.5)

        ax.plot(dates, localobs, "r-", label="Observations", linewidth=3.0)
        ax.plot(dates, mean_prediction, "k-",
                label="ERA5-LSTM ensemble mean", linewidth=3.0)
        # ax.plot(dates, tsmpobs, "b-", label="TSMP observations", linewidth=3.0)
        ax.scatter([], [], color="k",
                   label=f"KGE:{kge}, r:{r}, RMSE:{rmse},\nBias:{bias}, NSE: {nse}")

        ax.xaxis.set_major_locator(mdates.MonthLocator([1]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.xticks(rotation=45)
        plt.ylabel('Water table depth (m)')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3)
        plt.grid()
        plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean",
                    "era5wtd_vs_localobs", "timeseries", f"timeseries_{obslocation.replace('.', 'p')}.png"))

    def get_era5wtd(self, target, sim_1darrayindex):
        sim_allmembers = np.array([np.load(os.path.join(OUTPUTPATH, "validation_ERA5",
                                  "ensemble_400px", f"400px_member_{m}", f"sim_{target}.npy")) for m in range(100)])
        # (members, timeseries)
        sim_allmembers = sim_allmembers[:, :, sim_1darrayindex]
        return sim_allmembers

    def get_tsmpwtd(self, target, sim_1darrayindex):
        tsmpobs = np.load(os.path.join(OUTPUTPATH, "validation_ERA5",
                          "ensemble_400px", f"400px_member_0", f"obs_{target}.npy"))
        tsmpobs = tsmpobs[:, sim_1darrayindex]
        return tsmpobs

    def eval_era5wtd_obswtd_tsmpwtd(self):
        utils = utilities()
        map2d = np.load(os.path.join(INPUTPATH, "mapping_0stdroll6months.npy"))
        map2d_metrics = {"KGE": np.zeros(map2d.shape), "r": np.zeros(map2d.shape), "RMSE": np.zeros(
            map2d.shape), "Bias": np.zeros(map2d.shape), "NSE": np.zeros(map2d.shape)}
        map2d_metrics["KGE"][map2d_metrics["KGE"] == 0] = np.nan
        map2d_metrics["r"][map2d_metrics["r"] == 0] = np.nan
        map2d_metrics["RMSE"][map2d_metrics["RMSE"] == 0] = np.nan
        map2d_metrics["Bias"][map2d_metrics["Bias"] == 0] = np.nan
        map2d_metrics["NSE"][map2d_metrics["NSE"] == 0] = np.nan
        obswtd = pd.read_parquet(os.path.join(os.path.join(
            INPUTPATH, "localobservations"), "wtd_obsEU_monthly_avgdup.parquet"))
        collength = len(obswtd.columns)
        obswtd = obswtd.dropna(axis=1)  # drop columns with all NaN values
        if len(obswtd.columns) < collength:
            print(
                f"Dropped {collength - len(obswtd.columns)} columns with all NaN values from obswtd")
            return
        proj_mapping = pd.read_csv(os.path.join(
            INPUTPATH, "localobservations", "mapping_wtdobsEU_TSMP_avgdup.csv"))
        proj_mapping = proj_mapping.dropna(subset=["sim_1darrayindex"])
        proj_mapping.reset_index(drop=True, inplace=True)
        cdf_data = {
            'KGE': [],
            'Pearson correlation': [],
            'RMSE': [],
            'Bias': [],
            'NSE': []
        }
        for pixel in range(len(proj_mapping)):
            print(
                f"Processing pixel {pixel+1} of {len(proj_mapping)}", end='\r')
            # countrylocation = proj_mapping.iloc[pixel]["countrylocation"]
            country = proj_mapping.iloc[pixel]["country"]
            lon = proj_mapping.iloc[pixel]["lon"].replace("_", "")
            lat = proj_mapping.iloc[pixel]["lat"].replace("_", "")
            countrylocation = f"{country}_LON{lon}LAT{lat}"
            if not countrylocation in obswtd.columns:
                print(
                    f"countrylocation {countrylocation} not in obswtd columns, stopping...")
                return
            tsmp_xindex = proj_mapping.iloc[pixel]["tsmp_xindex"]
            tsmp_yindex = proj_mapping.iloc[pixel]["tsmp_yindex"]
            target = int(proj_mapping.iloc[pixel]["target_chunk"])
            sim_1darrayindex = int(
                proj_mapping.iloc[pixel]["sim_1darrayindex"])
            # (members, timeseries) (100, 5840)
            sim_era5wtd = self.get_era5wtd(target, sim_1darrayindex)
            sim_era5wtd[sim_era5wtd < 0.0] = 0.0
            # rescale sims from daily to monthly timestep
            time = pd.date_range(start="2000-01-01", periods=5840, freq="D")
            df = pd.DataFrame(sim_era5wtd.T, index=time)   # shape (5840, 100)
            sim_era5wtd = df.resample(
                "MS").mean().T.to_numpy()  # shape (100, 192)

            # sim_era5wtd = sim_era5wtd[:, :-365]  # remove year 2020 to match localobs
            ens_mean_era5wtd = np.mean(sim_era5wtd, axis=0)
            # tsmpobs = self.get_tsmpwtd(target, sim_1darrayindex)
            # tsmpobs = tsmpobs[:-365]  # remove year 2020 to match localobs
            localobs = np.array(obswtd[f"{countrylocation}"].to_list())
            if np.mean(localobs) > 50:
                print(
                    f"mean localobs > 50m for pixel {countrylocation}, skipping...")
                continue
            # localobs = localobs[730:]  # remove year 2016 to match sim_era5wtd and tsmpobs
            # check lengths
            if len(localobs) != sim_era5wtd.shape[1]:
                print(
                    f"lengths do not match for pixel {countrylocation}, skipping...")
                print(
                    f"len localobs: {len(localobs)}, len sim_era5wtd: {sim_era5wtd.shape[1]}")
                raise ValueError("Lengths do not match")
            kge = utils.calculate_kge(localobs, ens_mean_era5wtd)
            rmse = np.sqrt(np.mean((localobs - ens_mean_era5wtd) ** 2))
            r = np.corrcoef(localobs, ens_mean_era5wtd)[0, 1]
            bias = np.mean(ens_mean_era5wtd - localobs)
            nse = 1 - (np.sum((localobs - ens_mean_era5wtd) ** 2) /
                       np.sum((localobs - np.mean(localobs)) ** 2))
            cdf_data['KGE'].append(kge)
            cdf_data['Pearson correlation'].append(r)
            cdf_data['RMSE'].append(rmse)
            cdf_data['Bias'].append(bias)
            cdf_data['NSE'].append(nse)
            ############## plot timeseries ##############
            self.era5wtdensemble_localobs_timeseries(countrylocation, sim_era5wtd, ens_mean_era5wtd, None, localobs,
                                                     kge=kge,
                                                     rmse=rmse,
                                                     r=r,
                                                     bias=bias,
                                                     nse=nse
                                                     )

            ############## plot 2D map of pixels with colored accuracy ##############
            map2d_metrics["r"][int(tsmp_yindex), int(tsmp_xindex)] = r
            map2d_metrics["KGE"][int(tsmp_yindex), int(tsmp_xindex)] = kge
            map2d_metrics["RMSE"][int(tsmp_yindex), int(tsmp_xindex)] = rmse
            map2d_metrics["Bias"][int(tsmp_yindex), int(tsmp_xindex)] = bias
            map2d_metrics["NSE"][int(tsmp_yindex), int(tsmp_xindex)] = nse

        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px",
                "ensemble_mean", "era5wtd_vs_localobs", "r_map.npy"), map2d_metrics["r"])
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px",
                "ensemble_mean", "era5wtd_vs_localobs", "KGE_map.npy"), map2d_metrics["KGE"])
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px",
                "ensemble_mean", "era5wtd_vs_localobs", "RMSE_map.npy"), map2d_metrics["RMSE"])
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px",
                "ensemble_mean", "era5wtd_vs_localobs", "Bias_map.npy"), map2d_metrics["Bias"])
        np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px",
                "ensemble_mean", "era5wtd_vs_localobs", "NSE_map.npy"), map2d_metrics["NSE"])

    def plot2Dmaps(self):
        plot_functions = plotting_helper()
        map2d_metrics = {"KGE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "KGE_map.npy")),
                         "r": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "r_map.npy")),
                         "RMSE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "RMSE_map.npy")),
                         "Bias": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "Bias_map.npy")),
                         "NSE": np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "NSE_map.npy"))}
        # plot absolute mean bias instead of mean bias
        map2d_metrics["Bias"] = np.abs(map2d_metrics["Bias"])
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["KGE"], logscale=False, minval=-1, maxval=1, title="KGE", country="EU")
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["r"], logscale=False, minval=0, maxval=1, title="Pearson correlation", country="EU")
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["RMSE"], logscale=True, minval=0.01, maxval=10, title="RMSE", country="EU")
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["Bias"], logscale=True, minval=0.01, maxval=10, title="Absolute Mean Bias", country="EU")
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["NSE"], logscale=False, minval=-1, maxval=1, title="NSE", country="EU")

    def get_country_mask(self, country):
        mappingfile = pd.read_csv(os.path.join(
            INPUTPATH, "localobservations", "mapping_obs_TSMP_noduplicates.csv"))
        mappingfile = mappingfile.dropna(subset=["sim_1darrayindex"])
        mappingfile.reset_index(drop=True, inplace=True)
        country_mask = np.zeros((432, 444), dtype=bool)
        # TODO do the same without a for loop
        for pixel in range(len(mappingfile)):
            countrylocation = mappingfile.iloc[pixel]["countrylocation"]
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
            map2d_metrics[key] = np.where(
                country_mask, map2d_metrics[key], np.nan)
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["KGE"], logscale=False, minval=-1, maxval=1, title="KGE", country=country)
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["r"], logscale=False, minval=0, maxval=1, title="Pearson correlation", country=country)
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["RMSE"], logscale=True, minval=0.01, maxval=10, title="RMSE", country=country)
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["Bias"], logscale=True, minval=0.01, maxval=10, title="Absolute Mean Bias", country=country)
        plot_functions.doublefig_EU_2Dmap(
            data_map=map2d_metrics["NSE"], logscale=False, minval=-1, maxval=1, title="NSE", country=country)

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
        map2d_metrics["KGE"] = map2d_metrics["KGE"][~np.isnan(
            map2d_metrics["KGE"])]
        print(map2d_metrics["KGE"].shape)
        map2d_metrics["r"] = map2d_metrics["r"][~np.isnan(map2d_metrics["r"])]
        print(map2d_metrics["r"].shape)
        map2d_metrics["RMSE"] = map2d_metrics["RMSE"][~np.isnan(
            map2d_metrics["RMSE"])]
        print(map2d_metrics["RMSE"].shape)
        map2d_metrics["Bias"] = np.abs(map2d_metrics["Bias"])
        map2d_metrics["Bias"] = map2d_metrics["Bias"][~np.isnan(
            map2d_metrics["Bias"])]
        print(map2d_metrics["Bias"].shape)
        map2d_metrics["NSE"] = map2d_metrics["NSE"][~np.isnan(
            map2d_metrics["NSE"])]
        print(map2d_metrics["NSE"].shape)

        utils.plot_cdfs(data_dict={"KGE": map2d_metrics["KGE"]}, colors=['k'], linestyles=[
                        '-'], xlabel='KGE', xmin=0.2, xlim=(0.2, 1), yfloor=True, title='ERA5ensemblevsobs')
        utils.plot_cdfs(data_dict={"Pearson correlation": map2d_metrics["r"]}, colors=[
                        'k'], linestyles=['-'], xlabel='Pearson correlation', title='ERA5ensemblevsobs')
        utils.plot_cdfs(data_dict={"RMSE": map2d_metrics["RMSE"]}, colors=['k'], linestyles=[
                        '-'], logscale=True, xlabel='RMSE (m)', title='ERA5ensemblevsobs')
        utils.plot_cdfs(data_dict={"Bias": map2d_metrics["Bias"]}, colors=['k'], linestyles=[
                        '-'], xlabel='Mean Absolute Bias (m)', logscale=True, title='ERA5ensemblevsobs')
        utils.plot_cdfs(data_dict={"NSE": map2d_metrics["NSE"]}, colors=['k'], linestyles=[
                        '-'], xlabel='NSE', xmin=0.2, xlim=(0.2, 1), yfloor=True, title='ERA5ensemblevsobs')

    def location_of_localobs(self):
        plot_functions = plotting_helper()
        proj_mapping = pd.read_csv(os.path.join(
            INPUTPATH, "localobservations", "mapping_wtdobsEU_TSMP_avgdup.csv"))
        topo = np.load(os.path.join(INPUTPATH, "topo.npy"))
        topo = topo[0, :, :]  # remove time dimension if present
        # topo[topo==0] = np.nan  # set ocean to nan for better visualization
        countries_lonlat = {x: [proj_mapping[proj_mapping["country"] == x]["tsmp_lat"].to_list(
        ), proj_mapping[proj_mapping["country"] == x]["tsmp_lon"].to_list()] for x in proj_mapping["country"].unique()}
        for i in range(len(proj_mapping)):
            print(f"Plotting location {i+1} of {len(proj_mapping)}", end='\r')
            country = proj_mapping.iloc[i]["country"]
            lon = proj_mapping.iloc[i]["lon"].replace("_", "")
            lat = proj_mapping.iloc[i]["lat"].replace("_", "")
            obslocation = f"{country}_LON{lon}LAT{lat}"
            tsmp_xindex = proj_mapping.iloc[i]["tsmp_xindex"]
            tsmp_yindex = proj_mapping.iloc[i]["tsmp_yindex"]
            tsmplat = countries_lonlat[country][0]
            tsmplon = countries_lonlat[country][1]

            plot_functions.onepixel_in_doublefig_EU_2Dmap(
                x=tsmp_xindex, y=tsmp_yindex, location=obslocation, tsmp_lon=tsmplon, tsmp_lat=tsmplat, topo=topo)

    def calc_ev_iqr_rmse_bias(self):
        obswtd = pd.read_parquet(os.path.join(os.path.join(
            INPUTPATH, "localobservations"), "wtd_obsEU_monthly_avgdup.parquet"))
        obswtd = obswtd.dropna(axis=1)  # drop columns with all NaN values
        proj_mapping = pd.read_csv(os.path.join(
            INPUTPATH, "localobservations", "mapping_wtdobsEU_TSMP_avgdup.csv"))
        proj_mapping = proj_mapping.dropna(subset=["sim_1darrayindex"])
        proj_mapping.reset_index(drop=True, inplace=True)
        ev = []
        iqr = []
        rmse_list = []
        ambias_list = []
        # for storing obs and pred for each pixel
        obs_pred_pixels = np.zeros((len(proj_mapping), len(obswtd), 2))
        for pixel in range(len(proj_mapping)):
            print(
                f"Processing pixel {pixel+1} of {len(proj_mapping)}", end='\r')
            # countrylocation = proj_mapping.iloc[pixel]["countrylocation"]
            country = proj_mapping.iloc[pixel]["country"]
            lon = proj_mapping.iloc[pixel]["lon"].replace("_", "")
            lat = proj_mapping.iloc[pixel]["lat"].replace("_", "")
            countrylocation = f"{country}_LON{lon}LAT{lat}"
            if not countrylocation in obswtd.columns:
                print(
                    f"countrylocation {countrylocation} not in obswtd columns, skipping...")
                continue
            target = int(proj_mapping.iloc[pixel]["target_chunk"])
            sim_1darrayindex = int(
                proj_mapping.iloc[pixel]["sim_1darrayindex"])
            sim_era5wtd = self.get_era5wtd(target, sim_1darrayindex)
            # rescale sims from daily to monthly timestep
            time = pd.date_range(start="2000-01-01", periods=5840, freq="D")
            df = pd.DataFrame(sim_era5wtd.T, index=time)   # shape (5840, 100)
            sim_era5wtd = df.resample(
                "MS").mean().T.to_numpy()  # shape (100, 192)

            # sim_era5wtd = sim_era5wtd[:, :-365]  # remove year 2020 to match localobs
            ens_mean_era5wtd = np.mean(sim_era5wtd, axis=0)
            localobs = np.array(obswtd[f"{countrylocation}"].to_list())
            # localobs = localobs[730:]  # remove year 2016 to match sim_era5wtd and tsmpobs
            # check lengths
            if len(localobs) != sim_era5wtd.shape[1]:
                print(
                    f"lengths do not match for pixel {countrylocation}, skipping...")
                print(
                    f"len localobs: {len(localobs)}, len sim_era5wtd: {sim_era5wtd.shape[1]}")
                return

            obs_pred_pixels[pixel, :, 0] = localobs
            obs_pred_pixels[pixel, :, 1] = ens_mean_era5wtd

            ############## calculate metrics ##############
            rmse = np.sqrt(np.mean((localobs - ens_mean_era5wtd) ** 2))
            bias = np.mean(ens_mean_era5wtd - localobs)
            # members RMSE
            ensemble_predictions = sim_era5wtd  # shape (members, timeseries)
            # move axis to (timeseries, members) for easier calculation
            ensemble_predictions = np.moveaxis(
                ensemble_predictions, 0, 1)  # shape (timeseries, members)
            members_rmse = np.mean(
                np.sqrt(np.mean((ensemble_predictions - localobs[:, None]) ** 2, axis=0)))
            # members correlation
            members_correlation = [np.corrcoef(ensemble_predictions[:, m], localobs)[
                0, 1] for m in range(ensemble_predictions.shape[1])]
            members_correlation_mean = np.mean(members_correlation)

            # Calculate diversity by Pairwise correlation
            # Transpose to get members on rows
            correlation_matrix = np.corrcoef(ensemble_predictions.T)
            pairwisecorr = np.mean(correlation_matrix[np.triu_indices_from(
                correlation_matrix, k=1)])  # Compute diversity as 1 - average correlation

            ############## collect data for EV vs metrics ##############
            ensemble_variance = np.var(sim_era5wtd, axis=0)
            ensemble_variance = np.mean(ensemble_variance)
            ev.append(pairwisecorr)
            ensemble_iqr = np.percentile(
                sim_era5wtd, 75, axis=0) - np.percentile(sim_era5wtd, 25, axis=0)
            ensemble_iqr = np.mean(ensemble_iqr)
            iqr.append(ensemble_iqr)
            # rmse_list.append(rmse)
            rmse_list.append(members_correlation_mean)
            ambias_list.append(np.abs(bias))
        # np.save(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "pixels_ts_obs_pred.npy"), obs_pred_pixels)
        return np.array(ev), np.array(iqr), np.array(rmse_list), np.array(ambias_list), obs_pred_pixels

    def fitted_ev_rmse(self):
        ev, iqr, rmse_list, ambias_list, obs_pred_pixels = self.calc_ev_iqr_rmse_bias()
        # for EV vs RMSE: rmse2d_statspred = 0.67*(varmap**0.45)
        # for IQR vs Bias: bias2d_statspred = 0.45*(iqrmap**1.03)
        line = np.array([np.min(ev), np.max(ev)])
        # fitted_bias = 1.01 * (line ** 0.47)
        fitted_bias = 1.04 * line - 0.01
        ev = np.array(ev)
        rmse_list = np.array(rmse_list)
        # log_ev = np.log(ev)
        # log_rmse = np.log(rmse_list)
        # slope, intercept, r_value, p_value, std_err = stats.linregress(log_ev, log_rmse)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            ev, rmse_list)
        r_squared = r_value**2
        print(f"EV vs RMSE fitting line R2: {r_squared}")

        plt.figure()
        plt.scatter(ev, rmse_list, alpha=0.5)
        plt.plot(line, fitted_bias, 'r-',
                 label=f'R2={r_squared:.2f}\nRMSE=1.04*EV-0.01')
        plt.legend(loc='lower right')
        plt.xlabel(r'$\overline{EV}$')
        plt.ylabel(r'Members Correlation')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean",
                    "era5wtd_vs_localobs", "statistics", "EV_vs_mRMSE.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def fitted_iqr_mab(self):
        ev, iqr, rmse_list, ambias_list, obs_pred_pixels = self.calc_ev_iqr_rmse_bias()
        # for EV vs RMSE: rmse2d_statspred = 0.67*(varmap**0.45)
        # for IQR vs Bias: bias2d_statspred = 0.45*(iqrmap**1.03)
        line = np.array([np.min(iqr), np.max(iqr)])
        fitted_bias = 0.45 * (line ** 1.03)
        iqr = np.array(iqr)
        ambias_list = np.array(ambias_list)
        log_iqr = np.log(iqr)
        log_ambias = np.log(ambias_list)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_iqr, log_ambias)
        r_squared = r_value**2
        print(f"IQR vs Mean Absolute Bias fitting line R2: {r_squared}")

        plt.figure()
        plt.scatter(iqr, ambias_list, alpha=0.5)
        plt.plot(line, fitted_bias, 'r-',
                 label=f'R2={r_squared:.2f}\nMAB=0.45*(IQR^1.03)')
        plt.legend(loc='lower right')
        plt.xlabel(r'$\overline{IQR}$')
        plt.ylabel(r'Mean Absolute Bias (m)')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean",
                    "era5wtd_vs_localobs", "statistics", "IQR_vs_MAB.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap_scatter(self, observed, predicted, bins=50, cmap='viridis', i=0):
        # Calculate R²
        ss_res = np.sum((np.array(observed) - np.array(predicted)) ** 2)
        ss_tot = np.sum((np.array(observed) - np.mean(observed)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # # Create 2D histogram for heatmap
        # hist, xedges, yedges = np.histogram2d(observed, predicted, bins=bins)

        # # Assign each point a color based on density
        # x_idx = np.digitize(observed, xedges) - 1
        # y_idx = np.digitize(predicted, yedges) - 1
        # x_idx = np.clip(x_idx, 0, bins - 1)
        # y_idx = np.clip(y_idx, 0, bins - 1)
        # colors = hist[x_idx, y_idx]

        # Calculate density based on nearest neighbors
        observed = np.array(observed)
        predicted = np.array(predicted)
        points = np.column_stack((observed, predicted))

        # Use KDE (Kernel Density Estimation) for smooth density calculation
        kde = gaussian_kde(points.T)
        colors = kde(points.T)

        # Create plot
        plt.figure(figsize=(10, 8))

        # Plot scatter with density-based colors
        scatter = plt.scatter(observed, predicted, c=colors, cmap=cmap,
                              alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, label='Density')

        # Add y=x line
        min_val = min(min(observed), min(predicted))
        max_val = max(max(observed), max(predicted))
        plt.plot([min_val, max_val], [min_val, max_val],
                 'r-', linewidth=2, label='y=x')

        plt.xlabel('Observed WTD (m)')
        plt.ylabel('Predicted WTD (m)')
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'R² = {r2:.3f}')
        plt.legend()
        plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", "era5wtd_vs_localobs", "r2plots", f"obsvspred_{i}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean",
                    "era5wtd_vs_localobs", f"obsvspred_r2_3d.png"), dpi=300, bbox_inches='tight')
        return r2

    def yx_accuracy_plot(self):
        # ev, iqr, rmse_list, ambias_list, obs_pred_pixels = self.calc_ev_iqr_rmse_bias()
        obs_pred_pixels = np.load(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px",
                                  "ensemble_mean", "era5wtd_vs_localobs", "pixels_ts_obs_pred.npy"))
        self.plot_heatmap_scatter(obs_pred_pixels[:, :, 0].flatten(
        ), obs_pred_pixels[:, :, 1].flatten(), bins=30, cmap='viridis', i=0)

    def postprocess_era5wtd_vs_obs(self):
        pass
        # self.ensemble_statvsacc()
        # self.ensemble_statvsacc_fitting()
        # self.fit_ensvar_rmse_r()
        # self.onepoint_distribution()
        # self.eval_era5wtd_obswtd_tsmpwtd()

        # self.plot2Dmaps()
        # self.fitted_ev_rmse()
        # self.fitted_iqr_mab()
        # self.plot_cdfs()
        # self.yx_accuracy_plot()
        # self.location_of_localobs()
