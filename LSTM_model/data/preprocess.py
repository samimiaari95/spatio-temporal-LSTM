import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import OneHotEncoder
from LSTM_model.model.config import *
from LSTM_model.utils.plot_functions import plotting_helper

class preprocessing_data:
    def __init__(self) -> None:
        pass

    def conc_vars():
        dirpath = "/p/scratch/cslts/miaari1/raw"
        outpath = "/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/inputs/20yrs_ts"
        output_dic = {"soilmoisture": np.array([])}
        #output_dic = {"QFLX_EVAP_TOT": np.array([]), "soilmoisture": np.array([]), "subSurfStor": np.array([]),
        #              "TMAX_2M": np.array([]), "TMIN_2M": np.array([]), "TOT_PREC": np.array([]), "wtd": np.array([])}

        months = [x for x in os.listdir(dirpath)]
        months.sort()
        for month in months:
            print(month)
            vars = [x for x in os.listdir(os.path.join(dirpath, month)) if x.endswith(".npy") and "soilmoisture" in x]
            for var in vars:
                varname = var.replace(f"_{month}.npy","")
                data = np.load(os.path.join(dirpath, month, var))
                #print(data.shape)
                if len(output_dic[f"{varname}"])>=1:
                    output_dic[f"{varname}"] = np.concatenate((output_dic[f"{varname}"], data), axis=0)
                else:
                    output_dic[f"{varname}"] = data

        for key in output_dic.keys():
            print(key)
            print(output_dic[key].shape)
            np.save(os.path.join(outpath, f"{key}.npy"), output_dic[key])

    def crop_vars(varname):
        var = np.load(os.path.join(os.path.dirname(INPUTPATH), varname))
        print(varname)
        print(var.shape)
        # douro
        i = 150
        j = 95
        grid_size = 30
        var = var[:, i:i+grid_size, j:j+grid_size]

        print(var.shape)
        np.save(os.path.join(INPUTPATH, "DOURO_30x30", varname), var)

    def features_correlation():
        plot_functions = plotting_helper()
        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))
        
        features = ["TOT_PREC.npy", "TMAX_2M.npy", "TMIN_2M.npy", "soilmoisture.npy", "QFLX_EVAP_TOT.npy"]
        targetvar = np.load(os.path.join(INPUTPATH, "wtd.npy"))
        targetvar = targetvar[:365*7,:,:]
        for f in features:
            feature = np.load(os.path.join(INPUTPATH, f))
            feature = feature[:365*7,:,:]
            print(feature.shape)
            plot_functions.correlation_map(targetvar, feature, f"10yrscorrelation_wtd_{f.replace('.npy','')}", 5, 5, lons, lats)

    def conc_basins():
        basins = ["SEINE_30x30", "DOURO_30x30"]
        foldername = "SEINE+DOURO_30x30"
        vars = ["TOT_PREC.npy", "vpd.npy", "TMAX_2M.npy", "TMIN_2M.npy", "soilmoisture.npy", "slopex.npy", "slopey.npy", "soilind.npy", "topo.npy", "porosity_1mdepth.npy", "QFLX_EVAP_TOT.npy", "lon2D_ts.npy", "lat2D_ts.npy"]
        vars = ["wtd.npy"]
        for var in vars:
            var_basins = {basin: np.load(os.path.join(INPUTPATH, basin, var)) for basin in basins}
            data = np.array([])
            for basin in var_basins.keys():
                print(var, basin)
                data = np.concatenate((data, var_basins[basin]), axis=0) if len(data)>0 else var_basins[basin]
            np.save(os.path.join(INPUTPATH, var), data)

    def static_timeseries_EU(varname):
        filepath = os.path.join(os.path.dirname(INPUTPATH), varname)
        soilind = np.load(filepath)
        print(varname)

        soilind = np.array(soilind)
        soilind_1yr = np.array([])
        soilind = np.expand_dims(soilind, axis=0)
        for i in range(365):
            soilind_1yr = np.concatenate((soilind_1yr,soilind), axis=0) if len(soilind_1yr)>0 else soilind
        soilind_timeseries = np.array([])
        for i in range(10):
            soilind_timeseries = np.concatenate((soilind_timeseries,soilind_1yr), axis=0) if len(soilind_timeseries)>0 else soilind_1yr
        print(soilind_timeseries.shape)
        np.save(os.path.join(os.path.dirname(INPUTPATH),"EU_px_training", varname), soilind_timeseries)

    def subset_excl_waterbodies():
        wtd = np.load(os.path.join(os.path.dirname(INPUTPATH), "wtd.npy"))

        # exclude sides
        wtd[:, :100,:] = 0
        wtd[:, 432-10:,:] = 0
        wtd[:, :,444-10:] = 0
        wtd[:, :,:10] = 0

        # set negative wtd to 0
        wtd[wtd<0] = 0

        # exclude waterbodies
        std = np.std(wtd, axis=0)
        include = np.where(std==0, 0, 1)
        np.save(os.path.join(os.path.dirname(INPUTPATH), "included_excl_waterbodies.npy"), include)
        return include

    def remove_shallow_wtd():
        mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "included_excl_waterbodies.npy"))
        wtdorg = np.load(os.path.join(os.path.dirname(INPUTPATH), "wtd.npy"))
        wtd = wtdorg[:, mapping==1]

        meanwtd = np.mean(wtd, axis=0)
        deepind = np.where(meanwtd>1)
        print(deepind)
        print(len(deepind[0]))
        map_deepind = np.zeros(mapping.shape)
        include = np.where(mapping==1)
        for i, ind in enumerate(deepind[0]):
            map_deepind[include[0][ind],include[1][ind]] = 1
        print(map_deepind.shape)
        print(np.sum(map_deepind))
        np.save(os.path.join(os.path.dirname(INPUTPATH), "mapping_1mstd.npy"), include)

    def select_1mstd_wtd():
        mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "included_excl_waterbodies.npy"))
        wtdorg = np.load(os.path.join(os.path.dirname(INPUTPATH), "wtd.npy"))
        wtd = wtdorg[:, mapping==1]

        std = np.std(wtd, axis=0)
        #include = np.where(std<1, 0, 1)
        include = np.where(std>=1)
        print(include)

        map_1mstd = np.zeros(mapping.shape)
        incmap = np.where(mapping==1)
        for i, ind in enumerate(include[0]):
            map_1mstd[incmap[0][ind],incmap[1][ind]] = 1
        
        print(map_1mstd.shape)
        print(np.sum(map_1mstd))
        np.save(os.path.join(os.path.dirname(INPUTPATH), "mapping_1mstd.npy"), map_1mstd)

    def select_yearlyavg_1mstd_wtd():
        mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "included_excl_waterbodies.npy"))
        wtdorg = np.load(os.path.join(os.path.dirname(INPUTPATH), "wtd.npy"))
        wtd = wtdorg[:, mapping==1]

        map_1mstd = np.zeros(mapping.shape)
        incmap = np.where(mapping==1)
        
        for i in range(wtd.shape[1]):
            print(i)
            yearly_std = []
            for y in range(0, wtd.shape[0], 365):
                oneyear_std = np.std(wtd[y:y+365, i])
                yearly_std.append(oneyear_std)

            if np.mean(yearly_std)>1:
                map_1mstd[incmap[0][i], incmap[1][i]] = 1

        print(np.sum(map_1mstd))
        np.save(os.path.join(os.path.dirname(INPUTPATH), "mapping_yearlyavg1mstd.npy"), map_1mstd)

        # testing no errors in output
        wtd = wtdorg[:, map_1mstd==1]
        print(wtd.shape)
        test = []
        for i in range(wtd.shape[1]):
            yearly_std = []
            for y in range(0, wtd.shape[0], 365):
                oneyear_std = np.std(wtd[y:y+365, i])
                yearly_std.append(oneyear_std)
            print(np.mean(yearly_std))
            if np.mean(yearly_std)>1:
                test.append(True)
        test = np.array(test)
        print(np.unique(test))
        print(yearly_std)
        print(len(yearly_std))

    def create_choices():
        mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "mapping_yearlyavg1mstd.npy"))
        map1d = np.where(mapping==1)
        nb_samples = 100
        samples = np.random.choice(len(map1d[0]), nb_samples, False)
        samples.sort()
        map_choices = np.zeros(mapping.shape)
        for i, sample in enumerate(samples):
            map_choices[map1d[0][sample],map1d[1][sample]] = 1

        print(map_choices.shape)
        print(np.sum(map_choices))
        np.save(os.path.join(INPUTPATH, "choices.npy"), map_choices)
        
    def crop_vars_mapping(varname):
        mapping = np.load(os.path.join(INPUTPATH, "choices.npy"))
        var = np.load(os.path.join(os.path.dirname(INPUTPATH), varname))
        var = var[:,mapping==1]
        print(varname)
        print(var.shape)
        np.save(os.path.join(INPUTPATH, varname), var)

    def distribute_onehotencoding(vardata, varname):
        for i in range(vardata.shape[1]):
            print(i+1)
            onedimdata = vardata[:,i]
            onedimdata = onedimdata.reshape(-1,vardata.shape[0])
            onedimdata = np.repeat(onedimdata, 365*20, axis=0)
            np.save(os.path.join(INPUTPATH, f"{varname.replace('.npy',f'_{i+1}.npy')}"), onedimdata)

    def apply_onehotencoding(self):
        # Load the numpy file (adjust file path as needed)
        file_path = os.path.join(INPUTPATH,"soilind.npy")  # Replace this with the actual path
        soilind = np.load(file_path)
        # Since it's a static feature, select the first timestep (axis=0)
        static_features = soilind[0, :]  # Taking the first timestep
        # Reshape to make it compatible with OneHotEncoder
        static_features_reshaped = static_features.reshape(-1, 1)
        # Initialize OneHotEncoder
        encoder = OneHotEncoder(sparse=False, categories='auto')
        # Fit and transform the data to get one-hot encoded values
        one_hot_encoded = encoder.fit_transform(static_features_reshaped)
        # Display the shape of the result and preview the encoded array
        print(f"Original shape: {static_features.shape}")
        print(f"One-hot encoded shape: {one_hot_encoded.shape}")
        print(f"One-hot encoded preview:\n{one_hot_encoded[:5]}")
        # save a separate file for each hot-encoded dimension
        self.distribute_onehotencoding(one_hot_encoded, "soilind.npy")

    def var_timeseries(pixel, varname):
        vardata = np.load(os.path.join(INPUTPATH, varname))

        dates = pd.date_range(start='2001-01-01', end='2020-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        fig, ax = plt.subplots(figsize=(16, 10))
        
        ax.plot(dates, vardata[:,pixel], "k-")
        
        ax.xaxis.set_major_locator(mdates.MonthLocator([1]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

        plt.xticks(rotation=45)
        #plt.yscale("log")
        plt.ylabel('Water table depth (m)')
        plt.grid()
        plt.savefig(os.path.join(INPUTPATH, "timeseries", f"timeseries_{varname.replace('.npy','')}_{pixel}.png"))

