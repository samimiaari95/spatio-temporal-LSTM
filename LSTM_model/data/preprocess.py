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
        vars = ["tp", "swvl3", "vpd"]
        months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
        months.sort()
        period = (1995, 2016) #both years included
        for var in vars:
            print(var)
            dataarray = np.array([])
            for year in range(period[0], period[1]+1):
                for month in months:
                    print(f"year: {year}, month: {month}")
                    data = np.load(os.path.join(dirpath, f"{var}_{year}{month}_EU.npy"))
                    if len(dataarray)>=1:
                        dataarray = np.concatenate((dataarray, data), axis=0)
                    else:
                        dataarray = data
            np.save(os.path.join(INPUTPATH, f"{var}_{period[0]}_{period[1]}.npy"), dataarray)

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
            arr3d = np.repeat(vardata[np.newaxis, :, :], 365*22, axis=0)
            print(arr3d.shape)
            np.save(os.path.join(INPUTPATH, varname), arr3d)
    
    def repeat_wtd_timeseries(self):
        # Extend the daily WTD timeseries from 2000-2015 to 1996-2016 by repeating the
        # first available year for 1996-1999 and the last available year for 2016.
        wtd = np.load(os.path.join(INPUTPATH, TARGETVAR_FILE))
        print("Loaded WTD shape:", wtd.shape)

        if wtd.ndim != 3:
            raise ValueError(f"Expected a 3D WTD array, got shape {wtd.shape}")

        if wtd.shape[0] % 365 != 0:
            raise ValueError(f"Expected daily data in yearly chunks of 365 days, got {wtd.shape[0]} rows")

        n_years = wtd.shape[0] // 365
        if n_years < 1:
            raise ValueError("Input WTD array does not contain a full year of data")

        first_year = wtd[:365]
        last_year = wtd[(n_years - 1) * 365:n_years * 365]

        extended_wtd = np.concatenate(
            [np.repeat(first_year[None, :, :, :], 4, axis=0).reshape(-1, *first_year.shape[1:]),
             wtd,
             np.repeat(last_year[None, :, :, :], 1, axis=0).reshape(-1, *last_year.shape[1:])],
            axis=0,
        )

        out_path = os.path.join(INPUTPATH, "wtd_1996-2016.npy")
        np.save(out_path, extended_wtd)
        print("Saved extended WTD shape:", extended_wtd.shape)
        print("Saved to:", out_path)
        return extended_wtd
    
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
                    diff = (tsmp_sum_m - era_sum_m) / tsmp_sum_m
                    diffs.append(diff)
                else:
                    era_avg_m = np.mean(eradata[mask_m, ...], axis=0)
                    tsmp_avg_m = np.mean(tsmpdata[mask_m, ...], axis=0)
                    # avoid division by zero
                    era_avg_m = np.where(era_avg_m==0, 1e-6, era_avg_m)
                    diffs_m = (tsmp_avg_m - era_avg_m) / tsmp_avg_m
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
    
    def compare_input_datasets(self):
        # compare the 3 variables for the ERA5 and TSMP datasets (tp, swvl3, vpd) in pearson correlation
        # load the ERA5 dataset
        tp = np.load(os.path.join(INPUTPATH, "tp_EU.npy"))
        swvl3 = np.load(os.path.join(INPUTPATH, "swvl3_EU.npy"))
        vpd = np.load(os.path.join(INPUTPATH, "vpd_EU.npy"))
        tp = tp[365*2:,:,:]
        swvl3 = swvl3[365*2:,:,:]
        vpd = vpd[365*2:,:,:]
        print("tp shape:", tp.shape)
        print("swvl3 shape:", swvl3.shape)
        print("vpd shape:", vpd.shape)
        # start='2000-01-01', end='2015-12-31'
        # load the TSMP dataset
        dirpath = os.path.join(os.path.dirname(os.path.dirname(get_root_dir())), "spatio-temporal-LSTM", "inputs", "20yrs_ts")
        tp_tsmp = np.load(os.path.join(dirpath, "TOT_PREC.npy"))[:365*15,:,:]
        swvl3_tsmp = np.load(os.path.join(dirpath, "soilmoisture.npy"))[:365*15,:,:]
        vpd_tsmp = np.load(os.path.join(dirpath, "vpd.npy"))[:365*15,:,:]
        print("tp_tsmp shape:", tp_tsmp.shape)
        print("swvl3_tsmp shape:", swvl3_tsmp.shape)
        print("vpd_tsmp shape:", vpd_tsmp.shape)
        # map to obs
        mapping = pd.read_csv(os.path.join(INPUTPATH, "localobservations", "mapping_wtdobsEU_TSMP_avgdup.csv"))
        print(mapping)
        map2d = np.load(os.path.join(INPUTPATH, "mapping_0stdroll6months.npy"))
        map2d = np.zeros(map2d.shape) # ignore mapping for now, just compare the full datasets
        map2d[map2d==0] = np.nan
        for i in range(len(mapping)):
            map2d[mapping.loc[i, "tsmp_yindex"], mapping.loc[i, "tsmp_xindex"]] = 1
        # calculate the pearson correlation between the two datasets for each variable
        from scipy.stats import pearsonr
        plot_functions = plotting_helper()
        for var, var_tsmp, varname in zip([tp, swvl3, vpd], [tp_tsmp, swvl3_tsmp, vpd_tsmp], ["Precipitation", "soil moisture", "vpd"]):
            corr_map = np.zeros(var.shape[1:])
            for i in range(var.shape[1]):
                for j in range(var.shape[2]):
                    if np.isnan(map2d[i,j]):
                        corr_map[i,j] = np.nan
                        continue
                    corr, _ = pearsonr(var[:,i,j], var_tsmp[:,i,j])
                    corr_map[i,j] = corr
            plot_functions.doublefig_EU_2Dmap(corr_map, False, 0,1,f"correlation_{varname}_ERA5vsTSMP","EU")
            # save the correlation npy
            np.save(os.path.join(INPUTPATH, "checkinputs", f"correlation_ERA5vsTSMP_{varname}.npy"), corr_map)
    
        # highlight the points with high correlation and low correlation in the mapping file to later highlight in the statfitting

    def relativecompare_input_datasets(self):
        # TODO ask the agent to use the same code as in plot_inputvars to calculate the relative difference between the two datasets (TSMP-ERA5)/TSMP
        pass

    def mapping_localobs(self):
        obsfilepath = os.path.join(INPUTPATH, "wtd_obsEU_YMA_1996_2016.npy")
        obswtda = np.load(obsfilepath)
        obswtda = np.mean(obswtda, axis=0)
        # mapping = np.zeros(obswtda.shape)
        mapping = np.where(np.isnan(obswtda), 0, 1)
        print(mapping.shape)
        print(np.sum(mapping))
        print(np.unique(mapping, return_counts=True))
        np.save(os.path.join(INPUTPATH, "mapping_wtdobs_YMA.npy"), mapping)


    def save_to_hdf5(self, input_data, target_data, filepath, chunk_size=None):
        """
        input_data: 3D numpy array (datapoints, lookback, num_features)
        target_data: 1D numpy array (datapoints,)
        """
        assert input_data.shape[0] == target_data.shape[0], "input_data and target_data must have same number of datapoints"

        n_samples, lookback, n_features = input_data.shape

        # Reasonable default chunking: one chunk = one sample's window.
        # This makes single-item __getitem__ reads fast since each read
        # pulls exactly one contiguous chunk from disk.
        if chunk_size is None:
            chunk_size = (1, lookback, n_features)

        with h5py.File(filepath, 'w') as f:
            f.create_dataset(
                'input_data', data=input_data.astype(np.float32),
                chunks=chunk_size
            )
            f.create_dataset(
                'target_data', data=target_data.astype(np.float32),
                chunks=(1,)
            )
        f.close()

        print(f"Saved {n_samples} samples to {filepath}")
        print(f"input_data shape: {input_data.shape}, target_data shape: {target_data.shape}")
    
    # call here the function to prepare input array and also target array
    def prepare_input_target_arrays_in_hdf5(self):
        mapping = np.load(os.path.join(INPUTPATH, "mapping_wtdobs_YMA.npy"))
        # import training mean and std (saved during training process)
        for member in range(94,100):
            print(f"Preparing input and target arrays for member {member}")
            with open(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", f"400px_member_{member}", f"meanstd_400px_member_{member}_{MODEL_NAME}.pkl"), 'rb') as f:
                means_stds = pickle.load(f)
            f.close()

            featuresnamesintraining = ['TOT_PREC', 'vpd', 'soilmoisture', 'slopex', 'slopey', 'soilind', 'lon2D_ts', 'lat2D_ts', 'wtd']
            featuresnamesintransfer = FEATURES_FILES + [TARGETVAR_FILE]
            means_std_mapping = {featuresnamesintransfer[i]: featuresnamesintraining[i] for i in range(len(featuresnamesintransfer))}

            ####### target var #######
            raw_data = np.load(os.path.join(INPUTPATH, TARGETVAR_FILE)).astype(np.float32)
            raw_data = np.nan_to_num(raw_data)
            raw_data[raw_data < 0.0] = 0
            raw_data = raw_data[:,mapping==1]
            data = raw_data[:TEST_PERIOD-LOOKBACK, :, :] if len(raw_data.shape)>2 else raw_data[:TEST_PERIOD-LOOKBACK, :]
            data = (data - means_stds[f"{means_std_mapping[TARGETVAR_FILE]}mean"])/means_stds[f"{means_std_mapping[TARGETVAR_FILE]}std"]
            data = data.flatten()
            raw_data = None
            target_data = data.astype(np.float32)

            ####### input vars #######
            all_inputs = np.array([])
            for inputvar in FEATURES_FILES:
                print(inputvar)
                raw_data = np.load(os.path.join(INPUTPATH, inputvar)).astype(np.float32)
                raw_data = raw_data[:,mapping==1]
                raw_data = raw_data.reshape(raw_data.shape[0], NB_CELLS) if len(raw_data.shape)>2 else raw_data
                data = raw_data[:TEST_PERIOD, :]
                data = np.moveaxis(data, 0, -1) # (cells, timeseries)
                data = (data - means_stds[f"{means_std_mapping[inputvar]}mean"])/means_stds[f"{means_std_mapping[inputvar]}std"]

                lookback_arrays = [data[:, i-LOOKBACK:i] for i in range(LOOKBACK, TEST_PERIOD)]
                lookback_arrays = np.array(lookback_arrays)

                f1 = lookback_arrays.reshape(-1, LOOKBACK)
                f1 = np.array([f1])
                all_inputs = np.concatenate((all_inputs,f1), axis=0) if len(all_inputs)>0 else f1

            all_inputs = np.moveaxis(all_inputs, 0, -1)
            all_inputs = all_inputs.astype(np.float32)
            f1 = None
            lookback_arrays = None
            data = None
            raw_data = None

            # save to hdf5
            self.save_to_hdf5(all_inputs, target_data, os.path.join(INPUTPATH, "validation_ERA5", f"400px_member_{member}", f"standardized3dinput1dtarget_{member}.h5"))
            target_data = None
            all_inputs = None
    
    def test_hdf5(self):
        with h5py.File(os.path.join(INPUTPATH, "raw", "ERA5-Land", "ERA5land_t2m_199604.nc"), "r") as f:
            inputs = f["t2m"][()]
        h5py.File.close(f)
        print(inputs.shape)
