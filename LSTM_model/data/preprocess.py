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
            for year in range(1999, 2016):
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

    def static_timeseries(self):
        staticfiles = ["slopex.npy", "slopey.npy", "soilind.npy", "lon2D_ts.npy", "lat2D_ts.npy"]
        for varname in staticfiles:
            filepath = os.path.join(INPUTPATH, varname)
            vardata = np.load(filepath)
            print(varname)
            print(vardata.shape)
            if len(vardata.shape)>2:
                vardata = vardata[0,:,:]
            arr3d = np.repeat(vardata[np.newaxis, :, :], 6205, axis=0)
            print(arr3d.shape)
            np.save(os.path.join(INPUTPATH, varname), arr3d)

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
        varnames.append("wtd_2000-2015.npy")
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

    def check_tp_shapes(self):
        utils = utilities()
        dirpath = os.path.join(INPUTPATH, "raw", "ERA5-Land", "daily_23h_tp")
        files = os.listdir(dirpath)
        files.sort()
        i = 0
        for file in files:
            print(f"{i}/{len(files)}")
            i += 1
            if file.endswith(".nc"):
                data = utils.read_nc(os.path.join(dirpath, file), "tp")
                datashape = data.shape
                if datashape != (28, 541, 1171) and datashape != (29, 541, 1171) and datashape != (30, 541, 1171) and datashape != (31, 541, 1171):
                    raise ValueError(f"Unexpected shape {datashape} in file {file}")
                # rename files not matching text pattern
                if "at23h_" not in file:
                    newfile = file.replace("ERA5land", "at23h_ERA5land")
                    os.rename(os.path.join(dirpath, file), os.path.join(dirpath, newfile))
        
    def check_t2m_shapes(self):
        utils = utilities()
        dirpath = os.path.join(INPUTPATH, "raw", "ERA5-Land", "daily_mean_d2m_t2m_swvl3")
        files = [x for x in os.listdir(dirpath) if "swvl3" in x]
        files.sort()
        i = 0
        for file in files:
            print(f"{i}/{len(files)}")
            i += 1
            data = utils.read_nc(os.path.join(dirpath, file), "swvl3")
            datashape = data.shape
            if datashape != (28, 541, 1171) and datashape != (29, 541, 1171) and datashape != (30, 541, 1171) and datashape != (31, 541, 1171):
                raise ValueError(f"Unexpected shape {datashape} in file {file}")
            print(f"max: {np.max(data)}, min: {np.min(data)}, mean : {np.mean(data)}")
