import os
import pickle
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import OneHotEncoder
from LSTM_model.model.config import *
from LSTM_model.utils.plot_functions import plotting_helper
from LSTM_model.utils.utils import utilities

class preprocessing_data:
    def __init__(self) -> None:
        pass

    def conc_vars(self):
        dirpath = "/p/scratch/cslts/miaari1/raw"
        outpath = os.path.join(get_root_dir(), "inputs", "20yrs_ts")
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

    def crop_vars(self, varname):
        var = np.load(os.path.join(os.path.dirname(INPUTPATH), varname))
        print(varname)
        print(var.shape)
        # seine
        i = 195
        j = 164
        grid_size = 30
        var = var[:, i:i+grid_size, j:j+grid_size]

        print(var.shape)
        np.save(os.path.join(os.path.dirname(INPUTPATH), "SEINE_30x30", varname), var)

    def features_correlation(self):
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

    def conc_basins(self):
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

    def static_timeseries_EU(self, varname):
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

    def subset_excl_waterbodies(self):
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

    def remove_shallow_wtd(self):
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

    def select_1mstd_wtd(self):
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

    def select_yearlyavg_1mstd_wtd(self):
        mapping = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "included_excl_waterbodies.npy"))
        wtdorg = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtd.npy"))
        window_size = 182
        wtd = wtdorg[:, mapping==1]

        map_1mstd = np.zeros(mapping.shape)
        incmap = np.where(mapping==1)
        
        for i in range(wtd.shape[1]):
            print(i)
            # Compute the rolling standard deviation using a moving window
            rolling_std = np.array([np.std(wtd[t:t + window_size, i]) for t in range(wtd.shape[0] - window_size + 1)])

            # Check if there is at least one window where std == 0
            std_is_zero = np.any(rolling_std == 0)

            if not std_is_zero:
                map_1mstd[incmap[0][i], incmap[1][i]] = 1

        print(np.sum(map_1mstd))
        np.save(os.path.join(os.path.dirname(INPUTPATH), f"mapping_0stdroll6months.npy"), map_1mstd)

    def create_choices(self):
        mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "transfer_subset.npy"))
        map1d = np.where(mapping==1)
        nb_samples = 100
        samples = np.random.choice(len(map1d[0]), nb_samples, False)
        samples.sort()
        map_choices = np.zeros(mapping.shape)
        for i, sample in enumerate(samples):
            map_choices[map1d[0][sample],map1d[1][sample]] = 1

        print(map_choices.shape)
        print(np.sum(map_choices))
        np.save(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", "choices.npy"), map_choices)
        
    def crop_vars_mapping(self, varname):
        vardata = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), varname))
        for i in range(100):
            print(f"var: {varname}, member: {i}")
            var = vardata
            mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", f"choices.npy"))
            var = var[:,mapping==1]
            np.save(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", varname), var)

    def distribute_onehotencoding(self, vardata, varname):
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

    def var_timeseries(self, pixel, varname):
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


    def pixelscriteriainRB(self):
        mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "mapping_yearlyavg1mstd.npy"))
        # seine
        i = 195
        j = 164
        grid_size = 30
        mapping = mapping[i:i+grid_size, j:j+grid_size]
        print(np.sum(mapping))
        selected = np.where(mapping==1)
        print(selected)

    def ensemble_choices(self):
        utils = utilities()
        nb_samples = 100
        mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "ensemble_allchoices.npy"))
        for m in range(100):
            print(np.sum(mapping))
            map1d = np.where(mapping==1)
            utils.make_dir(os.path.join(os.path.dirname(INPUTPATH), f"400px_member_{m}"))
            samples = np.random.choice(len(map1d[0]), nb_samples, False)
            samples.sort()
            map_choices = np.zeros(mapping.shape)
            for i, sample in enumerate(samples):
                map_choices[map1d[0][sample],map1d[1][sample]] = 1

            np.save(os.path.join(os.path.dirname(INPUTPATH), f"400px_member_{m}", "choices.npy"), map_choices)
            mapping = np.where(map_choices==1, 0, mapping)
        np.save(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", "unchosen_pixels.npy"), mapping)
    
    def ensemble_cropvars(self):
        varnames = [x for x in FEATURES_FILES]
        varnames.append("wtd.npy")
        for varname in varnames:
            vardata = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), varname))
            for m in range(100):
                print(f"100px_member_{m}")
                mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), f"100px_member_{m}", "choices.npy"))
                var = vardata
                var = var[:,mapping==1]
                np.save(os.path.join(os.path.dirname(INPUTPATH), f"100px_member_{m}", varname), var)
    
    def jobs_scripts(self):
        filepath = os.path.join(get_root_dir(), "pythonjob_juwels_2nodes.sh")
        f = open(filepath, "r")
        sbatch = f.readlines()
        f.close()
        sbatch = sbatch[:19]
        #jobname = f"100x100_256dr0x1lr1x50_365x1000_prvpdsmxyindlonlat_member{i}"
        run_command_2 = "tclsh exfiltration.tcl"    
        for i in range(0, 100):
            case_path = f"cd /p/scratch/cslts/miaari1/infexfcases/test_case{i}"
            sbatch.append(case_path)
            #sbatch.append(jobname)
            sbatch.append(run_command_2)
        
        with open("/p/project/cslts/miaari1/parflow_simulations.sh", 'w') as sbatchfile:
            sbatchfile.write('\n'.join(sbatch))
        sbatchfile.close()

    def stand_npy_hdf5(self):
        utils = utilities()

        if os.path.exists(os.path.join(OUTPUTPATH, f"meanstd_{TARGET_REGION}_{MODEL_NAME}.pkl")):
            with open(os.path.join(OUTPUTPATH, f"meanstd_{SOURCE_REGION}_{MODEL_NAME}.pkl"), 'rb') as f:
                means_stds = pickle.load(f)
            f.close()
        else:
            means_stds = {}
        # prepare input data, standardization, lookback and train time series
        train_inputs, means_stds = utils.singleregion_inputfeatures(0, TRAINING_PERIOD, means_stds)

        # prepare input data of target variable and standardize
        obs_stand_train, means_stds = utils.singleregion_targetvar(LOOKBACK, TRAINING_PERIOD, means_stds)

        filename = "target1d_inputs3d_pointsxlookbackxfeatures.h5"
        # Save the array in HDF5 format
        with h5py.File(os.path.join(INPUTPATH, filename), "w") as h5_file:
            # Create a dataset and store the 3D array
            h5_file.create_dataset("input_data", data=train_inputs)
            h5_file.create_dataset("target_data", data=obs_stand_train)
        
        print("successfully saved")

        train_inputs = None
        obs_stand_train = None
        with h5py.File(os.path.join(INPUTPATH, filename), "r") as h5_file:
            loaded_array = h5_file["input_data"][:]
            print(loaded_array.shape)  # Should match the original array shape
            loaded_array = h5_file["target_data"][:]
            print(loaded_array.shape)  # Should match the original array shape


        print(means_stds)
        # save training data mean and std
        with open(os.path.join(INPUTPATH, f"meanstd_{TARGET_REGION}_{MODEL_NAME}.pkl"), 'wb') as f:
            pickle.dump(means_stds, f)
        f.close()
        print("saved pickle")
    
    def ensemble_pixels(self):
        choices = np.load(os.path.join(INPUTPATH, "choices.npy"))
        all_choices = np.zeros(choices.shape)
        for m in range(100):
            choices = np.load(os.path.join(os.path.dirname(INPUTPATH), f"400px_member_{m}", "choices.npy"))
            all_choices = np.where(choices==1, 1, all_choices)
        np.save(os.path.join(os.path.dirname(INPUTPATH), "training_subsets.npy"), all_choices)
        print(np.sum(all_choices))
      
    def ensemble_transfer_pixels(self):
        mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "mapping_0stdroll6months.npy"))
        choices = np.load(os.path.join(os.path.dirname(INPUTPATH), "training_subsets.npy"))
        print(np.sum(mapping))
        print(np.sum(choices))
        print(np.sum(mapping)-np.sum(choices))

        mapping = np.where(choices==1, 0, mapping)
        np.save(os.path.join(os.path.dirname(INPUTPATH), "transfer_subset.npy"), mapping)
        print(np.sum(mapping))

    def split_mapping(self):
        utils = utilities()
        mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", "included_excl_waterbodies.npy"))
        indices = np.where(mapping==1)
        print(len(indices[0]))
        indlength = len(indices[0])
        num_splits = 100
        split_indices0 = [indices[0][int((i)*indlength/num_splits):int((i+1)*indlength/num_splits)] for i in range(num_splits)]
        split_indices1 = [indices[1][int((i)*indlength/num_splits):int((i+1)*indlength/num_splits)] for i in range(num_splits)]

        split_arrays = [np.zeros(mapping.shape) for _ in range(num_splits)]
        for i in range(num_splits):
            split_arrays[i][(split_indices0[i], split_indices1[i])] = 1 

        testsummapping = split_arrays[0]

        for i in range(num_splits):
            print(np.sum(split_arrays[i]))
            utils.make_dir(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{i}"))
            np.save(os.path.join(os.path.dirname(INPUTPATH), f"target_pixels_{i}", f"mappingindices_{i}.npy"), split_arrays[i])
            testsummapping = np.where(testsummapping==1, 1, split_arrays[i])
        
        print("total sum")
        print(np.sum(testsummapping))
        print(np.unique(np.equal(testsummapping, mapping), return_counts=True))

    def exclude_iceland(self):
        filepath = os.path.join(os.path.dirname(get_root_dir()), "fork/validation_400_withcriteria_43200/inputs/20yrs_ts/ensemble_400px/target_pixels", "transfer_subset.npy")

        mappingfile = np.load(filepath)
        lon = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "lon2D.npy"))
        lat = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "lat2D.npy"))
        iceland_indx = np.where((lon< 0) & (lat> 60))
        print(np.sum(mappingfile))
        print(np.sum(mappingfile[iceland_indx]))
        mappingfile[iceland_indx] = 0
        print(np.sum(mappingfile))
        np.save(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "transfer_subset_0stdroll6months_43226exICELAND.npy"), mappingfile)