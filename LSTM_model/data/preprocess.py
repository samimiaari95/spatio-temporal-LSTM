import os
import pickle
import h5py
import shutil
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
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
        dirpath = os.path.join(INPUTPATH, "raw", "ERA5-Land", "npy_regridded_to_EUR11")
        output_dic = {"tp": np.array([]), "swvl3": np.array([]), "vpd": np.array([])}
        months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
        months.sort()
        for var in output_dic.keys():
            print(var)
            for year in range(2016, 2021):
                for month in months:
                    print(f"year: {year}, month: {month}")
                    data = np.load(os.path.join(dirpath, f"{var}_{year}{month}_EU.npy"))
                    if len(output_dic[f"{var}"])>=1:
                        output_dic[f"{var}"] = np.concatenate((output_dic[f"{var}"], data), axis=0)
                    else:
                        output_dic[f"{var}"] = data

        for key in output_dic.keys():
            print(key)
            print(output_dic[key].shape)
            np.save(os.path.join(INPUTPATH, f"{key}_EU.npy"), output_dic[key])

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

    def exclude_boundaries(self):
        dirpath = os.path.dirname(os.path.dirname(INPUTPATH))
        np_files = [x for x in os.listdir(os.path.join(dirpath, "orgsize_variables")) if x.endswith(".npy")]
        for np_file in np_files:
            print(np_file)
            var = np.load(os.path.join(dirpath, "orgsize_variables", np_file))
            print(var.shape)
            # exclude sides
            if len(var.shape)==2:
                var = var[100:432-10,10:444-10]
            else:
                var = var[:, 100:432-10,10:444-10]
            print(var.shape)
            np.save(os.path.join(dirpath, np_file.replace('_org.npy','.npy')), var)

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
        varnames = FEATURES_FILES
        varnames.append("wtd.npy")
        for varname in varnames:
            vardata = np.load(os.path.join(INPUTPATH, varname))
            print(varname)
            print(vardata.shape)
            for m in range(100):
                mapping = np.load(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{m}", f"mappingindices_{m}.npy"))
                var = vardata
                var = var[:,mapping==1]
                np.save(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{m}", varname), var)
    
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
        mapping = np.load(os.path.join(INPUTPATH, "mapping_0stdroll6months.npy"))
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
            utils.make_dir(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{i}"))
            np.save(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{i}", f"mappingindices_{i}.npy"), split_arrays[i])
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
    
    def map_members(self):
        train_in_path = os.path.join("/p/project1/cslts/miaari1/python_scripts/fork/train_100_withcriteria_43226/inputs/20yrs_ts", "ensemble_100px")
        mapping = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "mapping_0stdroll6months_43226.npy"))
        mapping = np.zeros(mapping.shape)
        mapping[mapping==0] = np.nan
        for i in range(100):
            print(f"member {i}")
            choices = np.load(os.path.join(train_in_path, f"100px_member_{i}", "choices.npy"))
            mapping = np.where(choices==1, i, mapping)
        np.save(os.path.join(train_in_path, "mapping_memberstrainpixels100.npy"), mapping)
        print(mapping[~np.isnan(mapping)].shape)
    
    def deleteoldmodels(self):
        rootdir = os.path.join(INPUTPATH, "validation_ERA5")
        for i in range(100):
            print(f"member {i}")
            targetdir = os.path.join(rootdir, f"target_pixels_{i}")
            files = [x for x in os.listdir(targetdir)]
            for file in files:
                os.remove(os.path.join(targetdir, file))
    
    def get_preselected_targetpixels(self):
        utils = utilities()
        indir = "/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/inputs/20yrs_ts/ensemble_400px"
        outdir = os.path.join(INPUTPATH, "validation_ERA5")
        for i in range(100):
            print(f"member {i}")
            utils.make_dir(os.path.join(outdir, f"400px_member_{i}"))
            # shutil.copy2(os.path.join(indir, f"target_pixels_{i}", f"mappingindices_{i}.npy"), os.path.join(outdir, f"target_pixels_{i}", f"mappingindices_{i}.npy"))
            shutil.copy2(os.path.join(indir, f"400px_member_{i}", f"choices.npy"), os.path.join(outdir, f"400px_member_{i}", f"choices.npy"))

    def get_pretrained400pxmodels(self):
        utils = utilities()
        indir = "/p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/outputs/20yrs_ts/ensemble_400px"
        outdir = os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px")
        for i in range(100):
            print(f"member {i}")
            utils.make_dir(os.path.join(outdir, f"400px_member_{i}"))
            shutil.copy2(os.path.join(indir, f"400px_member_{i}", f"400px_member_{i}_100_256dr0x1lr01x50_365x1000_prvpdsmxyind.png"), os.path.join(outdir, f"400px_member_{i}", f"400px_member_{i}_100_256dr0x1lr01x50_365x1000_prvpdsmxyind.png"))
            shutil.copy2(os.path.join(indir, f"400px_member_{i}", f"400px_member_{i}_100_256dr0x1lr01x50_365x1000_prvpdsmxyind.pt"), os.path.join(outdir, f"400px_member_{i}", f"400px_member_{i}_100_256dr0x1lr01x50_365x1000_prvpdsmxyind.pt"))
            shutil.copy2(os.path.join(indir, f"400px_member_{i}", f"meanstd_400px_member_{i}_100_256dr0x1lr01x50_365x1000_prvpdsmxyind.pkl"), os.path.join(outdir, f"400px_member_{i}", f"meanstd_400px_member_{i}_100_256dr0x1lr01x50_365x1000_prvpdsmxyind.pkl"))
    
    def find_leastdistance_TSMPpixel_to_EUobs(self, lon, lat):
        # load TSMP lonlat
        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

        # find nearest TSMP grid point
        # explain how it is done
        # calculate the distance between the observation location and all TSMP grid points
        # use numpy broadcasting to calculate the distance efficiently
        # formula: distance = sqrt((lon2D - lon_obs)^2 + (lat2D - lat_obs)^2)
        # then find the index of the minimum distance
        # unit is in degrees
        dist = np.sqrt((lons - lon)**2 + (lats - lat)**2)

        # calculate haversine distance instead?
        # compute haversine distance (in kilometers) between obs point and all grid points
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        lons_rad = np.radians(lons)
        lats_rad = np.radians(lats)

        dlon = lons_rad - lon_rad
        dlat = lats_rad - lat_rad
        a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad) * np.cos(lats_rad) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0  # Earth radius in kilometers
        dist = R * c
        
        yindex, xindex = np.unravel_index(np.argmin(dist), dist.shape)
        tsmp_lon = lons[yindex, xindex]
        tsmp_lat = lats[yindex, xindex]
        return yindex, xindex, tsmp_lon, tsmp_lat
    
    def find_chunk_flatidx_from_2Dindices(self, yindex, xindex):
        chunksmap = np.load(os.path.join(INPUTPATH, "chunksmap.npy"))
        if np.isnan(chunksmap[yindex, xindex]):
            print(f"Not in TSMP dataset")
            return None, None
        target = int(chunksmap[yindex, xindex])
        target_map = np.load(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{target}", f"mappingindices_{target}.npy"))
        # find the 1d index within this target
        rows, cols = np.where(target_map == 1)
        flat_idx = np.flatnonzero((rows == yindex) & (cols == xindex))[0]
        return target, flat_idx

    def plot_inputvars(self):
        import cartopy.crs as ccrs
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        dirpath = os.path.join(os.path.dirname(os.path.dirname(get_root_dir())), "spatio-temporal-LSTM", "inputs", "20yrs_ts")
        mapping = np.load(os.path.join(INPUTPATH, "mapping_0stdroll6months.npy"))
        vars = {"tp_EU.npy": "TOT_PREC.npy", "vpd_EU.npy": "vpd.npy", "swvl3_EU.npy": "soilmoisture.npy"}
        unit = {"tp": "Total precipitation: (TSMP - ERA5)/ERA5", "vpd": "Vapor pressure deficit: (TSMP - ERA5)/ERA5", "swvl3": "Soil moisture: (TSMP - ERA5)/ERA5"}
        for var in vars.keys():
            print(var)
            eradata = np.load(os.path.join(INPUTPATH, var))[-365:,:,:]
            tsmpdata = np.load(os.path.join(dirpath, vars[var]))[-365:,:,:]
            eradata = np.where(mapping==1, eradata, np.nan)
            tsmpdata = np.where(mapping==1, tsmpdata, np.nan)
            # build a 365-day date index (non-leap year) and iterate months Jan..Dec
            dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
            # exclude february 29th if present
            dates = dates[(dates.month != 2) | (dates.day != 29)]
            lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
            lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))
            base = var.replace('_EU.npy', '')
            diffs = []
            for m in range(1, 13):
                mask_m = dates.month == m
                if "tp" in var:
                    era_sum_m = np.sum(eradata[mask_m, ...], axis=0)
                    tsmp_sum_m = np.sum(tsmpdata[mask_m, ...], axis=0)
                    # make the difference relative to ERA5
                    # avoid division by zero
                    era_sum_m = np.where(era_sum_m==0, 1e-6, era_sum_m)
                    diff = (tsmp_sum_m - era_sum_m) / era_sum_m
                    diffs.append(diff)
                else:
                    era_avg_m = np.mean(eradata[mask_m, ...], axis=0)
                    tsmp_avg_m = np.mean(tsmpdata[mask_m, ...], axis=0)
                    # avoid division by zero
                    era_avg_m = np.where(era_avg_m==0, 1e-6, era_avg_m)
                    diffs_m = (tsmp_avg_m - era_avg_m) / era_avg_m
                    diffs.append(diffs_m)

            # global symmetric color scale
            global_vmax = max(np.max(np.abs(d)) for d in diffs)
            if global_vmax == 0:
                global_vmax = 1e-6
            # if "tp" in var:
            #     global_vmax = 100

            # create subplots (3 rows x 4 cols)
            fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True, subplot_kw={'projection': projection})

            axes_flat = axes.flatten()
            cmap = plt.get_cmap('RdBu_r') #"viridis"

            # plot each month's diff in its subplot
            for idx, d in enumerate(diffs):
                ax = axes_flat[idx]
                im = ax.pcolormesh(lons, lats, d, cmap=cmap, vmin=-global_vmax, vmax=global_vmax, shading='auto', transform=ccrs.PlateCarree())
                ax.set_title(f"2020{idx+1:02d}")
                ax.coastlines()
                ax.gridlines()
    

            # single colorbar for all subplots
            # plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label=colorbar_label, pad=0.08)
            cbar = fig.colorbar(im, ax=axes_flat.tolist(), orientation='horizontal', fraction=0.05, pad=0.02)
            cbar.set_label(f"{unit[base]}")

            # save and close
            plt.savefig(os.path.join(INPUTPATH, "checkinputs", f"2Dmap_{base}_ERA5vsTSMP_monthly_diff.png"))
            plt.close(fig)

    def obs_seine_BDDADES(self):
        dirpath = os.path.join(os.path.dirname(get_root_dir()), "dataset_basins", "Seine", "BDD_ADES", "saves")
        meta = pd.read_parquet(os.path.join(os.path.dirname(dirpath), "meta.parquet"))
        wtd_df = pd.DataFrame()
        files = [x for x in os.listdir(dirpath) if x.startswith("WTDobs_") and x.endswith(".parquet")]
        for i, file in enumerate(files):
            print(f"{file} of ({i+1}/{len(files)})")
            data = pd.read_parquet(os.path.join(dirpath, file))
            if not data.empty:
                # select dates between 2015-01-01 and 2019-12-31
                data = data[(data["date"] >= '2015-01-01') & (data["date"] <= '2019-12-31')]
                # exclude february 29th
                data = data[~((data['date'].dt.month == 2) & (data['date'].dt.day == 29))]
                # exclude p column
                data = data[['date','wtd']]
                data.reset_index(drop=True, inplace=True)
                # get longitude and latitude from metadata
                code = file.replace("WTDobs_","").replace(".parquet","")
                lon = meta[meta['Code INSEE']==code]['lon'].values[0]
                lat = meta[meta['Code INSEE']==code]['lat'].values[0]

                # rename wtd column to location in the format LONxxxLATyyy
                loc_name = f"LON{lon}LAT{lat}"
                data.rename(columns={'wtd': loc_name}, inplace=True)
                
                # exclude points with missing dates
                all_dates = pd.date_range(start='2015-01-01', end='2019-12-31', freq='D')
                all_dates = all_dates[(all_dates.month != 2) | (all_dates.day != 29)]
                if len(data) != len(all_dates):
                    continue
                    
                # add the new column to the dataframe
                if wtd_df.empty:
                    wtd_df = data
                else:
                    wtd_df = pd.merge(wtd_df, data, on='date', how='outer')

        # remove columns with Nan values in more than 50% of the rows
        # wtd_df.dropna(axis=1, thresh=len(wtd_df)*0.5, inplace=True)
        wtd_df.dropna(axis=1, inplace=True)
        print(wtd_df)
        print(len(wtd_df))
        print(len(wtd_df.columns))
        wtd_df.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "obs_wtd_seine_BDDADES.parquet"))
        return

    def mapping_localobs_TSMPproj(self):
        # load local observations
        obs_wtd = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "obs_wtd_seine_BDDADES.parquet"))
        obs_wtd.set_index('date', inplace=True)
        print(obs_wtd)
        print(obs_wtd.shape)

        # load TSMP projected wtd
        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

        dates = pd.date_range(start='2015-01-01', end='2019-12-31', freq='D')
        dates = dates[(dates.month != 2) | (dates.day != 29)]
        tsmp_dates = pd.DataFrame(dates, columns=['date'])
        tsmp_dates.set_index('date', inplace=True)

        # create a mapping dataframe
        mapping_df = pd.DataFrame(columns=['obs_location', 'tsmp_lon', 'tsmp_lat', 'tsmp_xindex', 'tsmp_yindex'])

        for col in obs_wtd.columns:
            print(col)
            lon = float(col.split("LAT")[0].replace("LON",""))
            lat = float(col.split("LAT")[1])
            # find nearest TSMP grid point
            # explain how it is done
            # calculate the distance between the observation location and all TSMP grid points
            # use numpy broadcasting to calculate the distance efficiently
            # formula: distance = sqrt((lon2D - lon_obs)^2 + (lat2D - lat_obs)^2)
            # then find the index of the minimum distance
            # unit is in degrees
            dist = np.sqrt((lons - lon)**2 + (lats - lat)**2)

            # calculate haversine distance instead?
            # compute haversine distance (in kilometers) between obs point and all grid points
            lon_rad = np.radians(lon)
            lat_rad = np.radians(lat)
            lons_rad = np.radians(lons)
            lats_rad = np.radians(lats)

            dlon = lons_rad - lon_rad
            dlat = lats_rad - lat_rad
            a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad) * np.cos(lats_rad) * np.sin(dlon / 2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            R = 6371.0  # Earth radius in kilometers
            dist = R * c
            
            yindex, xindex = np.unravel_index(np.argmin(dist), dist.shape)
            print(dist[yindex, xindex])
            tsmp_lon = lons[yindex, xindex]
            tsmp_lat = lats[yindex, xindex]
            # concatenate the obtained location to the mapping dataframe
            mapping_df = pd.concat([mapping_df, pd.DataFrame({'obs_location': [col], 'tsmp_lon': [tsmp_lon], 'tsmp_lat': [tsmp_lat], 'tsmp_xindex': [xindex], 'tsmp_yindex': [yindex]})], ignore_index=True)

            # mapping_df = mapping_df.append({'obs_location': col, 'tsmp_lon': tsmp_lon, 'tsmp_lat': tsmp_lat, 'tsmp_xindex': xindex, 'tsmp_yindex': yindex}, ignore_index=True)

        print(mapping_df)
        mapping_df.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_localobs_TSMPproj.csv"), index=False)
        return
    
    def chunks_mapping(self):
        allchunksmapping = np.load(os.path.join(INPUTPATH, "mapping_0stdroll6months.npy"))
        allchunksmapping = np.zeros(allchunksmapping.shape)
        allchunksmapping[allchunksmapping==0] = np.nan
        for i in range(100):
            targetmap = np.load(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{i}", f"mappingindices_{i}.npy"))
            allchunksmapping = np.where(targetmap==1, i, allchunksmapping)
        np.save(os.path.join(INPUTPATH, "chunksmap.npy"), allchunksmapping)
        print(np.unique(allchunksmapping, return_counts=True))
        print(np.count_nonzero(~np.isnan(allchunksmapping)))
    
    def getlocalpixel_simindex(self):
        org_mapping = np.load(os.path.join(INPUTPATH, "mapping_0stdroll6months.npy"))
        chunksmap = np.load(os.path.join(INPUTPATH, "chunksmap.npy"))
        obs = pd.read_csv(os.path.join(INPUTPATH, "localobservations", "mapping_localobs_TSMPproj.csv"))
        obs["target_chunk"] = None
        obs["sim_1darrayindex"] = None
        print(obs.head())
        for i in range(len(obs)):
            y = obs.iloc[i]['tsmp_yindex']
            x = obs.iloc[i]['tsmp_xindex']
            # check that this pixel is within the dataset
            if not org_mapping[y, x]:
                print(f"obs point {i} at x:{x}, y:{y} is not in the dataset")
                continue
            # find which target it belongs to
            target = int(chunksmap[y, x])
            target_map = np.load(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{target}", f"mappingindices_{target}.npy"))
            # find the 1d index within this target
            rows, cols = np.where(target_map == 1)
            flat_idx = np.flatnonzero((rows == y) & (cols == x))[0]
            obs.at[i, 'target_chunk'] = target
            obs.at[i, 'sim_1darrayindex'] = flat_idx
            print(f"obs point {i} at x:{x}, y:{y} is in target {target} at sim index {flat_idx}")
        obs.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_localobs_TSMPproj.csv"), index=False)
    
    def check_timeperiod_timestep_EUobsDenmark(self):
        globalrootdir = os.path.dirname(get_root_dir())
        EUdir = os.path.join(globalrootdir, "obsWTD_point_measurements_in_EU")
        denmarkdir = os.path.join(EUdir, "obsWTD_point_measurements_in_Denmark_GEUS", "o.data", "WTD_measurements_Denmark")

        wells_daily = []
        files = [x for x in os.listdir(denmarkdir) if (x.endswith(".csv") and "jupiter" not in x.lower())]
        files.sort()
        # filename is the well ID
        # measured time is Pejletidspunkt
        # measured WTD m underground is  Vandstand m. under terr√¶n
        # adapt to pandas datetime and sort by date
        for file in files:
            # print(file)
            data = pd.read_csv(os.path.join(denmarkdir, file), sep=';', encoding='latin1', engine="python")
            data = data[['Pejletidspunkt', ' Vandstand m. under terrÃ¦n']]
            data = data.rename(columns={' Vandstand m. under terrÃ¦n': 'WTD_m_under_terrain'})
            data['Pejletidspunkt'] = pd.to_datetime(data['Pejletidspunkt'], errors='coerce', dayfirst=True)
            data = data.dropna(subset=['Pejletidspunkt'])
            data = data.sort_values(by='Pejletidspunkt')

            mask = (data['Pejletidspunkt'] >= '2017-01-01') & (data['Pejletidspunkt'] < '2020-01-01')
            data = data.loc[mask]
            # check timestep
            data = data.set_index('Pejletidspunkt')
            all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
            if len(data) < len(all_dates):
                pass
                # print(f"not daily timestep expected {len(all_dates)} days, got {len(data)} days")
            elif len(data) == len(all_dates):
                print(f"added file {file} daily timestep from 2017-01-01 to 2019-12-31")
                wells_daily.append(file)
            elif len(data) > len(all_dates):
                # check if more than one measurement per day then calculate daily average
                data = data.resample('D').mean()
                if len(data) == len(all_dates):
                    print(f"daily timestep from 2017-01-01 to 2019-12-31 after resampling")
                    wells_daily.append(file)
                elif len(data) < len(all_dates):
                    print("missing dates after resampling")
                    if len(data)==1094:
                        # find missing day
                        missing_dates = all_dates.difference(data.index)
                        print(f"missing dates: {missing_dates}")
                        if missing_dates == pd.DatetimeIndex(['2019-12-31']):
                            # append last day as previous day's value
                            last_value = data.iloc[-1]['WTD_m_under_terrain']
                            new_row = pd.DataFrame({'WTD_m_under_terrain': [last_value]}, index=missing_dates)
                            data = pd.concat([data, new_row])
                            data = data.sort_index()
                            print(f"added missing last day value {last_value}")
                            wells_daily.append(file)
                else:
                    print(f"wtf still not daily timestep after resampling expected {len(all_dates)} days, got {len(data)} days")
        print("Denmark wells with daily timestep from 2017-01-01 to 2019-12-31:")
        # print(wells_daily)
        return wells_daily

    def check_timeperiod_timestep_EUobsPortugal(self):
        globalrootdir = os.path.dirname(get_root_dir())
        EUdir = os.path.join(globalrootdir, "obsWTD_point_measurements_in_EU")
        portugaldir = os.path.join(EUdir, "obsWTD_point_measurements_in_Portugal_SNIRH", "o.data", "WTD_measurements_Portugal_1996_2020")

        # read all csv files in the directory that have the name: serie_<number>.csv
        # remove the first 2 rows and also 4th row (units)
        wells_daily = []
        files = [x for x in os.listdir(portugaldir) if (x.endswith(".csv") and x.startswith("serie_"))]
        # remove the following from the list as it doesnt have daily timestep
        files.remove("serie_27072020172741.csv")
        files.remove("serie_27072020172921.csv")
        files.remove("serie_27072020174207.csv")
        files.sort()
        for file in files:
            # print(file)
            data = pd.read_csv(os.path.join(portugaldir, file), skiprows=[0,1,3], encoding='latin1', engine="python")
            # rename DATA to Date, filter between 2017-01-01 and 2019-12-31, remove empty columns
            data = data.rename(columns={'DATA': 'Date'})
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)
            # filter dates
            mask = (data['Date'] >= '2017-01-01') & (data['Date'] < '2020-01-01')
            data = data.loc[mask]
            data = data.dropna(axis=1, how='all')
            all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
            if len(data) < len(all_dates):
                # print(f"not daily timestep expected {len(all_dates)} days, got {len(data)} days")
                pass
            elif (len(data['Date']) == len(all_dates)) and (data['Date'] == all_dates).all():
                # drop columns with at least one NaN value
                data = data.dropna(axis=1)
                # check if at least one column remains
                if data.empty:
                    # print("all columns have NaN values after filtering")
                    continue
                else:
                    # get names of the remaining columns and append them to wells_daily
                    colnames = data.columns.tolist()
                    colnames.remove('Date')
                    cols = [file + "_" + col for col in colnames]
                    wells_daily.extend(cols)
            elif len(data) > len(all_dates):
                # check if more than one measurement per day then calculate daily average
                data = data.set_index('Date')
                data = data.resample('D').mean(numeric_only=True)
                if (len(data.index) == len(all_dates)) and (data.index == all_dates).all():
                    data = data.dropna(axis=1)
                    # check if at least one column remains
                    if data.empty:
                        # print("all columns have NaN values after filtering")
                        continue
                    else:
                        # get names of the remaining columns and append them to wells_daily
                        colnames = data.columns.tolist()
                        # colnames.remove('Date')
                        cols = [file + "_" + col for col in colnames]
                        wells_daily.extend(cols)
                
        print("Wells with daily timestep from 2017-01-01 to 2019-12-31:")
        # print(wells_daily)
        return wells_daily
    
    def check_timeperiod_timestep_EUobsSpain(self):
        globalrootdir = os.path.dirname(get_root_dir())
        EUdir = os.path.join(globalrootdir, "obsWTD_point_measurements_in_EU")
        spaindir = os.path.join(EUdir, "obsWTD_point_measurements_in_Spain_MITECO", "o.data", "WTD_measurements_Spain", "csv_files")

        wells_daily = []
        files = [x for x in os.listdir(spaindir) if (x.endswith(".csv"))]
        files.sort()

        for file in files:
            print(file)
            data = pd.read_csv(os.path.join(spaindir, file), skiprows=[0], encoding='latin1', engine="python")
            data['Fecha'] = pd.to_datetime(data['Fecha'], errors='coerce', dayfirst=True)
            # filter dates
            mask = (data['Fecha'] >= '2017-01-01') & (data['Fecha'] < '2020-01-01')
            data = data.loc[mask]
            data = data.dropna(axis=1, how='all')
            all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
            if len(data) < len(all_dates):
                # print(f"not enough timestep expected {len(all_dates)} days, got {len(data)} days")
                pass
            else:
                print("theres a chance")

    def check_timeperiod_timestep_EUobsSweden(self):
        globalrootdir = os.path.dirname(get_root_dir())
        EUdir = os.path.join(globalrootdir, "obsWTD_point_measurements_in_EU")
        swedendir = os.path.join(EUdir, "obsWTD_point_measurements_in_Sweden_SGU", "o.data", "WTD_measurements_Sweden")

        sweddirs = [os.path.join(swedendir, d) for d in os.listdir(swedendir) if os.path.isdir(os.path.join(swedendir, d))]
        wells_daily = []
        for subdir in sweddirs:
            files = [x for x in os.listdir(subdir) if (x.endswith(".csv"))]
            for file in files:
                data = pd.read_csv(os.path.join(subdir, file), encoding='iso-8859-1',sep=';',usecols=[2,5,17,18],skiprows=1, names=['time','WTD [m]','N','E'])
                # print(data)
                appendname = os.path.basename(subdir) + "_" + file
                data['time'] = pd.to_datetime(data['time'], errors='coerce', dayfirst=False)
                # filter dates
                mask = (data['time'] >= '2017-01-01') & (data['time'] < '2020-01-01')
                data = data.loc[mask]
                data = data.dropna(axis=1, how='all')
                all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
                if len(data) < len(all_dates):
                    # print(f"not enough timestep expected {len(all_dates)} days, got {len(data)} days")
                    pass
                elif (len(data) == len(all_dates)):
                    # print(f"added file {file} daily timestep from 2017-01-01 to 2019-12-31")
                    wells_daily.append(appendname)
                elif len(data) > len(all_dates):
                    # check if more than one measurement per day then calculate daily average
                    data = data.set_index('time')
                    data = data.resample('D').mean()
                    if len(data) == len(all_dates):
                        wells_daily.append(appendname)
                    elif len(data) > len(all_dates):
                        print(f"still not daily timestep after resampling expected {len(all_dates)} days, got {len(data)} days")
        print("Sweden wells with daily timestep from 2017-01-01 to 2019-12-31:")
        # print(wells_daily)
        # print(len(wells_daily))
        return wells_daily

    def mapping_EUobs_TSMPproj(self):
        globalrootdir = os.path.dirname(get_root_dir())
        EUdir = os.path.join(globalrootdir, "obsWTD_point_measurements_in_EU")
            
        eudict = {
            "Denmark": self.check_timeperiod_timestep_EUobsDenmark(),
            "Portugal": self.check_timeperiod_timestep_EUobsPortugal(),
            "Sweden": self.check_timeperiod_timestep_EUobsSweden()
        }
        locations = pd.DataFrame(columns=['wellID', 'country', 'lon', 'lat', 'tsmp_lon', 'tsmp_lat', 'tsmp_xindex', 'tsmp_yindex', 'target_chunk', 'sim_1darrayindex'])
        denmarkcoordinate_dataset = pd.read_csv(os.path.join(EUdir, "obsWTD_point_measurements_in_Denmark_GEUS", "o.data", "WTD_measurements_Denmark", "jupiter.csv"),encoding='cp865',sep=',',usecols=[0,31,32])
        # get Denmark locations
        for well in eudict["Denmark"]:
            wellid = f"{well}".replace(".csv","")
            X = denmarkcoordinate_dataset[denmarkcoordinate_dataset['DGUNR']==wellid].iloc[0,1]
            Y = denmarkcoordinate_dataset[denmarkcoordinate_dataset['DGUNR']==wellid].iloc[0,2]
                    
            #Transform coordinates in the ETRS89 / UTM zone 32N (EPSG: 25832) to WGS84
            transformer = Transformer.from_crs(25832,4326)
            (lat, lon) = transformer.transform(X,Y)
            yindex, xindex, tsmp_lon, tsmp_lat = self.find_leastdistance_TSMPpixel_to_EUobs(lon, lat)

            # find which target chunk it belongs to
            target, flat_idx = self.find_chunk_flatidx_from_2Dindices(yindex, xindex)
            if target is None:
                print(f"Denmark well {wellid} at lon: {lon}, lat: {lat} is not in TSMP dataset")
                continue
            
            # append to locations dataframe
            locations = pd.concat([locations, pd.DataFrame({'wellID': [wellid], 'country': ['Denmark'], 'lon': [lon], 'lat': [lat], 'tsmp_lon': [tsmp_lon], 'tsmp_lat': [tsmp_lat], 'tsmp_xindex': [xindex], 'tsmp_yindex': [yindex], 'target_chunk': [target], 'sim_1darrayindex': [flat_idx]})], ignore_index=True)

        # get Portugal locations
        portugalcoordinate_dataset = pd.read_csv(os.path.join(EUdir, "obsWTD_point_measurements_in_Portugal_SNIRH", "o.data", "WTD_measurements_Portugal_1996_2020", "rede_seleccao_Piezometria.csv"),skiprows=4,encoding='cp1252',index_col=False)
        portugalcoordinate_dataset.drop(portugalcoordinate_dataset.index[-1],inplace=True)
        for well in eudict["Portugal"]:
            wellid = f"{well}".split("_")[-1]
            x = portugalcoordinate_dataset.loc[portugalcoordinate_dataset['CÓDIGO']==wellid,'COORD_X (M)'].values[0]
            y = portugalcoordinate_dataset.loc[portugalcoordinate_dataset['CÓDIGO']==wellid,'COORD_Y (M)'].values[0]
            
            #Transform coordinates in the Hayford Gauss Militar Datum Lisboa (EPSG: 20790) to WGS84
            transformer = Transformer.from_crs(20790,4326)
            (lat, lon) = transformer.transform(x,y)
            yindex, xindex, tsmp_lon, tsmp_lat = self.find_leastdistance_TSMPpixel_to_EUobs(lon, lat)

            # find which target chunk it belongs to
            target, flat_idx = self.find_chunk_flatidx_from_2Dindices(yindex, xindex)
            if target is None:
                print(f"Portugal well {wellid} at lon: {lon}, lat: {lat} is not in TSMP dataset")
                continue
            
            # append to locations dataframe
            locations = pd.concat([locations, pd.DataFrame({'wellID': [wellid], 'country': ['Portugal'], 'lon': [lon], 'lat': [lat], 'tsmp_lon': [tsmp_lon], 'tsmp_lat': [tsmp_lat], 'tsmp_xindex': [xindex], 'tsmp_yindex': [yindex], 'target_chunk': [target], 'sim_1darrayindex': [flat_idx]})], ignore_index=True)

        # get Sweden locations
        swedendir = os.path.join(EUdir, "obsWTD_point_measurements_in_Sweden_SGU", "o.data", "WTD_measurements_Sweden")
        sweddirs = [f"{d}".split("]")[0]+"]" for d in eudict["Sweden"]]
        for subdir in sweddirs:
            dirpath = os.path.join(swedendir, subdir)
            wellid = subdir
            files = [x for x in os.listdir(dirpath) if (x.endswith(".csv"))]
            for file in files:
                data = pd.read_csv(os.path.join(dirpath, file), encoding='iso-8859-1',sep=';',usecols=[2,5,17,18],skiprows=1, names=['time','WTD [m]','N','E'])
                N = data.iloc[0,2]
                E = data.iloc[0,3]
                #Transform coordinates in the SWEREF99TM (EPSG: 3006) to WGS84
                transformer = Transformer.from_crs(3006,4326)
                (lat, lon) = transformer.transform(N,E)
                yindex, xindex, tsmp_lon, tsmp_lat = self.find_leastdistance_TSMPpixel_to_EUobs(lon, lat)

                # find which target chunk it belongs to
                target, flat_idx = self.find_chunk_flatidx_from_2Dindices(yindex, xindex)
                if target is None:
                    print(f"Sweden well {wellid} at lon: {lon}, lat: {lat} is not in TSMP dataset")
                    continue
                
                # append to locations dataframe
                locations = pd.concat([locations, pd.DataFrame({'wellID': [wellid], 'country': ['Sweden'], 'lon': [lon], 'lat': [lat], 'tsmp_lon': [tsmp_lon], 'tsmp_lat': [tsmp_lat], 'tsmp_xindex': [xindex], 'tsmp_yindex': [yindex], 'target_chunk': [target], 'sim_1darrayindex': [flat_idx]})], ignore_index=True)
        print(locations)
        locations.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_EUobs_TSMPproj.csv"), index=False)

    def merge_EUobs_TSMPproj(self):
        dps_proj = pd.read_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_EUobs_TSMPproj.csv"))
        f_proj = pd.read_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_localobs_TSMPproj.csv"))
        finalcols = ['countylocation', 'wellID', 'country', 'lon', 'lat', 'tsmp_lon', 'tsmp_lat', 'tsmp_xindex', 'tsmp_yindex', 'target_chunk', 'sim_1darrayindex']
        df = pd.DataFrame(columns=finalcols)
        dps_proj['countylocation'] = dps_proj['country'] + "_LON" + dps_proj['lon'].astype(str) + "LAT" + dps_proj['lat'].astype(str)
        f_proj['countylocation'] = 'France_' + f_proj['obs_location']
        f_proj['lon'] = f_proj['obs_location'].apply(lambda x: float(x.split("LAT")[0].replace("LON","")))
        f_proj['lat'] = f_proj['obs_location'].apply(lambda x: float(x.split("LAT")[1]))
        f_proj = f_proj.rename(columns={'obs_location': 'wellID'})
        f_proj['country'] = 'France'
        print(dps_proj.columns)
        print(f_proj.columns)
        df = pd.concat([df, dps_proj[finalcols]], ignore_index=True)
        df = pd.concat([df, f_proj[finalcols]], ignore_index=True)
        df.dropna(subset=['sim_1darrayindex'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(df)
        df.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_DPSFobs_TSMPproj.csv"), index=False)

    def EUobs_to_parquet(self):
        df = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "obs_wtd_seine_BDDADES.parquet"))
        df.set_index('date', inplace=True)
        # rename columns to match the format Country_LONxxxLATyyy
        newcols = {}
        for col in df.columns:
            newcol = f"France_{col}"
            newcols[col] = newcol
        df.rename(columns=newcols, inplace=True)
        
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
        # filter df to only have dates between 2017-01-01 and 2019-12-31
        df = df[(df.index >= '2017-01-01') & (df.index <= '2019-12-31')]
        df = df.reindex(all_dates)
        df.dropna(axis=1, inplace=True)

        mapping = pd.read_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_EUobs_TSMPproj.csv"))
        for i in range(len(mapping)):
            wellid = mapping.iloc[i]['wellID']
            country = mapping.iloc[i]['country']
            lon = mapping.iloc[i]['lon']
            lat = mapping.iloc[i]['lat']
            
            if country == "Denmark":
                dirpath = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_EU", "obsWTD_point_measurements_in_Denmark_GEUS", "o.data", "WTD_measurements_Denmark")
                file = f"{wellid}.csv"
                print(file)
                data = pd.read_csv(os.path.join(dirpath, file), encoding='cp865', sep=';', usecols=[1,3,6], skiprows=1, names=['Well_index','time','WTD [m]'], engine="python")
                # drop well index column
                data = data.drop(columns=['Well_index'])
                # TODO check from here after
                # data = data[['time', ' WTD [m]']]
                # data = data.rename(columns={' WTD [m]': wellid})
                data['time'] = pd.to_datetime(data['time'], errors='coerce', dayfirst=True)
                data = data.dropna(subset=['time'])
                data = data.sort_values(by='time')
                data.set_index('time', inplace=True)
                # filter between 2017-01-01 and 2019-12-31
                mask = (data.index >= '2017-01-01') & (data.index <= '2019-12-31')
                data = data.loc[mask]
                # check the length of data is equal to all_dates
                if len(data) == len(all_dates):
                    df[f"Denmark_LON{lon}LAT{lat}"] = data['WTD [m]']
                elif len(data) > len(all_dates):
                    data = data.resample('D').mean()
                    if len(data) == len(all_dates):
                        df[f"Denmark_LON{lon}LAT{lat}"] = data['WTD [m]']
            elif country == "Portugal":
                dirpath = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_EU", "obsWTD_point_measurements_in_Portugal_SNIRH", "o.data", "WTD_measurements_Portugal_1996_2020")
                print(wellid)
                portugalfiles = {f"serie_27072020174714.csv_528/16", "serie_27072020182243.csv_600/50", "serie_27072020182735.csv_608/470"}
                filename = [f for f in portugalfiles if wellid in f][0].split(".csv_")[0]
                file = f"{filename}.csv"
                data = pd.read_csv(os.path.join(dirpath, file),skiprows=2,encoding='cp1252')
    
                #Drop the columns having "FLAG"
                data.drop([col for col in data.columns if 'Unnamed:' in col],axis=1,inplace=True)
                
                #Drop the first row
                data.drop(data.index[0],inplace=True)
                
                #Drop the last row containing website location
                data.drop(data.index[-1],inplace=True)
                
                #Rename the time column name
                data.rename(columns = {'DATA':'time'},inplace=True)
                
                #print(data.head)
                # drop all columns except time and the wellid column
                data = data[['time', wellid]]
                #filter dates
                data['time'] = pd.to_datetime(data['time'], errors='coerce', dayfirst=True)
                mask = (data['time'] >= '2017-01-01') & (data['time'] <= '2019-12-31')
                data = data.loc[mask]
                data = data.dropna(subset=['time', wellid])
                data = data.sort_values(by='time')
                # drop rows with NaN values in the wellid column
                data = data.dropna(subset=[wellid])
                # check the length of data is equal to all_dates
                if len(data) == len(df):
                    df[f"Portugal_LON{lon}LAT{lat}"] = data[wellid].values
                elif len(data) > len(df):
                    data = data.set_index('time')
                    data = data.resample('D').mean()
                    if len(data) == len(df):
                        df[f"Portugal_LON{lon}LAT{lat}"] = data[wellid].values
            elif country == "Sweden":
                swedendir = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_EU", "obsWTD_point_measurements_in_Sweden_SGU", "o.data", "WTD_measurements_Sweden")
                filename = [f for f in os.listdir(os.path.join(swedendir, wellid)) if f.endswith(".csv")][0]
                data = pd.read_csv(os.path.join(swedendir, wellid, filename), encoding='iso-8859-1',sep=';',usecols=[2,5,17,18],skiprows=1, names=['time','WTD [m]','N','E'])
                data['time'] = pd.to_datetime(data['time'], errors='coerce', dayfirst=False)
                # filter dates
                mask = (data['time'] >= '2017-01-01') & (data['time'] < '2020-01-01')
                data = data.loc[mask]
                data = data.dropna(axis=1, how='all')
                all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
                if (len(data) == len(all_dates)):
                    df[f"Sweden_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]']
                elif len(data) > len(all_dates):
                    # check if more than one measurement per day then calculate daily average
                    data = data.set_index('time')
                    data = data.resample('D').mean()
                    if len(data) == len(all_dates):
                        df[f"Sweden_LON{lon}LAT{lat}"] = data['WTD [m]']

        print(df)
        df.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "obs_EUwtd_DPSF.parquet"))