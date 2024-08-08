import shutil
import tarfile
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import read_nc, get_prudenceMask, delete_files, get_S4W_basin
from volumetric_soilmoisture import calculate_soilmoisture_diag
import matplotlib.pyplot as plt


def copy_files():
    source_path = "/p/largedata2/detectdata/CentralDB/projects/d02/working_directory/sim/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline/postpro/ProductionV1"
    dst_path = "/p/scratch/cslts/miaari1/detect/raw"
    months = [month for month in os.listdir(source_path) if int(month.split(".")[0])>1999120100 and int(month.split(".")[0])<2011010100 and not month.endswith(".tar")]
    months.sort()
    model = "clm"
    var = "QFLX_EVAP_TOT.nc"
    for month in months:
        shutil.copy(os.path.join(source_path, month, model, var), os.path.join(dst_path, month, var))
        print(month, var)

def extract_tar(file_timesteps, month):
    
    source_path = "/p/largedata2/detectdata/CentralDB/projects/d02/working_directory/sim/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline/postpro/ProductionV1"
    dst_path = "/p/scratch/cslts/miaari1/detect/raw"
    files = [file for file in os.listdir(source_path) if f"{month}.tar" in file]#if int(file.split(".")[0])>2001013000 and int(file.split(".")[0])<2001030100 and file.endswith(".tar")]
    files.sort() #2011010100

    for file in files:
        print(file)
        #month = file.split(".tar")[0]
        if not os.path.exists(os.path.join(dst_path, month)):
            os.mkdir(os.path.join(dst_path, month))

        with tarfile.open(os.path.join(source_path, file), 'r:') as tar:
            for file_timestep in file_timesteps:
                if ("040100" in month or "060100" in month or "090100" in month or "110100" in month) and ("2977" in file_timestep):
                    continue
                if ("2785" in file_timestep or "2881" in file_timestep or "2977" in file_timestep) and ("020100" in month):
                    continue

                img_file = tar.extractfile(f'{month}/parflow/ParFlow_EU11_{month}.out.{file_timestep}')
                
                # --------------------- save file ---------------------------
                with open (os.path.join(dst_path, month, f"ParFlow_EU11_{month}.out.{file_timestep}"), "wb") as outfile:
                    outfile.write(img_file.read())
                outfile.close()
        tar.close()
        # calculate soil moisture
        calculate_soilmoisture_diag(month)
        # delete saturation files
        delete_files(os.path.join(dst_path, month), "saturation")

def extract_calc_soilmoisture(month):
    exdir = "/p/scratch/cslts/miaari1/detect/2000010100/parflow/"
    filenames = [x.split(".out.")[-1] for x in os.listdir(exdir) if "_saturation.nc" in x]
    filenames.sort()
    extract_tar(file_timesteps=filenames, month=month)

def mask_ME(lat2D, lon2D, data):
    prudName = 'ME'
    prudenceMask = get_prudenceMask(lat2D, lon2D, prudName)
    for i in range(data.shape[0]):
        masked_values = np.ma.masked_where(prudenceMask==False, data[i][:,:])
        data[i,:,:] = masked_values
    #print(np.ma.count(masked_values))
    return data

def mask_FR(lat2D, lon2D, data):
    prudName = 'FR'
    prudenceMask = get_prudenceMask(lat2D, lon2D, prudName)
    for i in range(data.shape[0]):
        masked_values = np.ma.masked_where(prudenceMask==False, data[i][:,:])
        data[i,:,:] = masked_values
    return data

def mask_matrix(lat2D, lon2D, data, region):
    prudenceMask = get_S4W_basin(lat2D, lon2D, region)
    for i in range(data.shape[0]):
        masked_values = np.ma.masked_where(prudenceMask==False, data[i][:,:])
        data[i,:,:] = masked_values
    print(np.ma.count(data[1,:,:]))
    return data
    
def temporalAgg_matrix(data, timestep, agg):
    daily_data = []
    for i in range(0, data.shape[0], int(60*24/timestep)):
        daily_var = data[i:i+int(60*24/timestep), :, :]
        
        if agg == "sum":
            daily_var = np.sum(daily_var, axis=0)
        elif agg == "mean":
            daily_var = np.mean(daily_var, axis=0)
        elif agg == "max":
            daily_var = np.max(daily_var, axis=0)
        elif agg == "min":
            daily_var = np.min(daily_var, axis=0)
        
        daily_data.append(daily_var)
    return np.array(daily_data)

def temporalAgg(df, timestep, agg, var):
    daily_data = {"day": [], f"{var}": []}
    nb_iter = 1
    for i in range(0, len(df), int(60*24/timestep)):
        daily_data["day"].append(nb_iter)
        daily_var = df[f"{var}"].iloc[i:i+int(60*24/timestep)]

        if agg == "sum":
            daily_var = np.sum(daily_var.to_list())
        elif agg == "mean":
            daily_var = np.mean(daily_var.to_list())
        elif agg == "max":
            daily_var = np.max(daily_var.to_list())
        elif agg == "min":
            daily_var = np.min(daily_var.to_list())
        
        daily_data[f"{var}"].append(daily_var)
        nb_iter += 1
    return pd.DataFrame(daily_data)

def spatialAgg(data, agg, var):
    dic_data = {"time": [], f"{var}": []}
    for t in range(data.shape[0]):
        dic_data["time"].append(t)
        var_timestep = data[t,:,:]
        if agg == "sum":
            var_timestep = np.sum(var_timestep)
        elif agg == "mean":
            var_timestep = np.mean(var_timestep)
        elif agg == "max":
            var_timestep = np.max(var_timestep)
        elif agg == "min":
            var_timestep = np.min(var_timestep)

        dic_data[f"{var}"].append(var_timestep)
    return pd.DataFrame(dic_data)

def detectdata_preprocessing():
    # TODO subsurfstorage should be only in fully saturated zone
    subsurfstor = "subSurfStor.nc"
    tasmax = "TMAX_2M_ts.nc"
    tasmin = "TMIN_2M_ts.nc"
    precip = "TOT_PREC_ts.nc"
    wtd = "wtd.nc"
    evap = "QFLX_EVAP_TOT.nc"
    # variables to consider from dataset
    vars = ["subSurfStor.nc", "TMAX_2M_ts.nc", "TMIN_2M_ts.nc", "TOT_PREC_ts.nc", "QFLX_EVAP_TOT.nc", "wtd.nc"]
    # define timesteps of the models
    timesteps = {"parflow": 15, "clm": 60, "cosmo": 60}
    # prepare postprocessed dataframe dictionary
    preprocessed_df = {"date": [], "TOT_PREC": [], "TMAX_2M": [], "TMIN_2M": [], "QFLX_EVAP_TOT": [], "wtd": [], "subSurfStor": []}
    # raw data path
    dirpath = "/p/scratch/cslts/miaari1/detect/raw"
    # iterate through the folders of months in the dataset
    for month in tqdm(sorted(os.listdir(dirpath)), desc = 'months'):
        print(month)
        # iterate for every variable (.nc file) in each month
        for var in vars:
            varname = var.replace("_ts.nc", ".nc").split(".nc")[0]
            var_data = read_nc(filepath=os.path.join(dirpath, month, var), var=varname)
            lat2D = read_nc(filepath=os.path.join(dirpath, month, var), var="lat")
            lon2D = read_nc(filepath=os.path.join(dirpath, month, var), var="lon")
            # set the timestep based on the model produced output
            if var == subsurfstor:
                # remove the first and last timestep from parflow output
                var_data = var_data[1:-1,:,:]
                timestep = timesteps["parflow"]
                Sagg = "sum"
                Tagg = "mean"
            elif var == wtd:
                # remove the first and last timestep from parflow output
                var_data = var_data[1:-1,:,:]
                timestep = timesteps["parflow"]
                Sagg = "mean"
                Tagg = "mean"
            elif var == evap:
                timestep = timesteps["clm"]
                Sagg = "sum"
                Tagg = "sum"
            elif var == precip:
                # remove the last timestep from cosmo output
                var_data = var_data[:-1,:,:]
                timestep = timesteps["cosmo"]
                Sagg = "sum"
                Tagg = "sum"
            elif var == tasmax:
                # remove the last timestep from cosmo output
                var_data = var_data[:-1,:,:]
                timestep = timesteps["cosmo"]
                Sagg = "max"
                Tagg = "max"
            elif var == tasmin:
                # remove the last timestep from cosmo output
                var_data = var_data[:-1,:,:]
                timestep = timesteps["cosmo"]
                Sagg = "min"
                Tagg = "min"

            # TODO mask to study region here (currently on prudence ME)
            #var_data = mask_ME(lat2D=lat2D, lon2D=lon2D, data=var_data)
            var_data = mask_FR(lat2D=lat2D, lon2D=lon2D, data=var_data)
            # call aggregation function for every variable
            df = spatialAgg(data=var_data, agg=Sagg, var=varname)
            df = temporalAgg(df, timestep, Tagg, varname)
            dates = [f"{month[:6]}{str(day).zfill(2)}" for day in df["day"].to_list()]
            # append data
            preprocessed_df[varname].extend(df[varname].to_list())
        preprocessed_df["date"].extend(dates)
    preprocessed_df = pd.DataFrame(preprocessed_df)
    preprocessed_df.sort_values(by=["date"], inplace=True)
    preprocessed_df.reset_index(inplace=True)
    preprocessed_df.to_csv("/p/project/cslts/miaari1/python_scripts/DailyScriptBox/FR_precip_Tmax_Tmin_subsurfstor_evap_wtd.csv", index=False)


def preprocess_matrix_sm(month):
    dirpath = "/p/scratch/cslts/miaari1/detect/soilmoisture"
    outpath = "/p/scratch/cslts/miaari1/detect/raw"
    #months = os.listdir("/p/scratch/cslts/miaari1/detect/raw")
    #for month in months:
    monthfiles = [x for x in os.listdir(dirpath) if f"{month}" in x]
    monthfiles.sort()
    monthly_data = []
    for dailyfile in monthfiles:
        dailydata = read_nc(filepath=os.path.join(dirpath, dailyfile), var="volsm")
        if dailydata.shape[0]==1:
            continue
        day = dailyfile.replace(f"{month}.out.","").replace("_volsm.nc", "")
        day = (int(day)-1)/96 + 1
        print(day)
        dailydata = dailydata[:, 6, :, :] # NOTE soil moisture only at layer 6
        dailytimestep = np.mean(dailydata, axis=0)
        monthly_data.append(dailytimestep)
    monthly_data = np.array(monthly_data)
    outfile = os.path.join(outpath, month, f"soilmoisture_{month}.npy")
    np.save(outfile, monthly_data)


def preprocess_matrix(month):
    outpath = "/p/scratch/cslts/miaari1/detect/raw"
    # define variables filenames
    subsurfstor = "subSurfStor.nc"
    tasmax = "TMAX_2M_ts.nc"
    tasmin = "TMIN_2M_ts.nc"
    precip = "TOT_PREC_ts.nc"
    wtd = "wtd.nc"
    evap = "QFLX_EVAP_TOT.nc"
    # variables to consider from dataset
    vars = ["subSurfStor.nc", "TMAX_2M_ts.nc", "TMIN_2M_ts.nc", "TOT_PREC_ts.nc", "QFLX_EVAP_TOT.nc", "wtd.nc"]
    # define timesteps of the models
    timesteps = {"parflow": 15, "clm": 60, "cosmo": 60}
    # iterate through the variables in one month
    for var in vars:
        print(var)
        varname = var.replace("_ts.nc", ".nc").split(".nc")[0]
        var_data = read_nc(filepath=os.path.join(outpath, month, var), var=varname)
        lat2D = read_nc(filepath=os.path.join(outpath, month, var), var="lat")
        lon2D = read_nc(filepath=os.path.join(outpath, month, var), var="lon")
        # set the timestep based on the model produced output
        if var == subsurfstor:
            # remove the first and last timestep from parflow output
            var_data = var_data[1:-1,:,:]
            timestep = timesteps["parflow"]
            Tagg = "mean"
        elif var == wtd:
            # remove the first and last timestep from parflow output
            var_data = var_data[1:-1,:,:]
            timestep = timesteps["parflow"]
            Tagg = "mean"
        elif var == evap:
            timestep = timesteps["clm"]
            Tagg = "sum"
        elif var == precip:
            # remove the last timestep from cosmo output
            var_data = var_data[:-1,:,:]
            timestep = timesteps["cosmo"]
            Tagg = "sum"
        elif var == tasmax:
            # remove the last timestep from cosmo output
            var_data = var_data[:-1,:,:]
            timestep = timesteps["cosmo"]
            Tagg = "max"
        elif var == tasmin:
            # remove the last timestep from cosmo output
            var_data = var_data[:-1,:,:]
            timestep = timesteps["cosmo"]
            Tagg = "min"
        
        data = temporalAgg_matrix(var_data, timestep, Tagg)
        print("final")
        print(data.shape)
        outfile = os.path.join(outpath, month, f"{varname}_{month}.npy")
        np.save(outfile, data)


# NOTE the parse arguments were only used to parallel compute the soil moisture
#parser = argparse.ArgumentParser(description='Tell me what this script can do!.')
#parser.add_argument('--month', type=str, required=True,
#                    help='yyyymmdd00') # month of simulation
#args = parser.parse_args()
#month = args.month
#preprocess_matrix(f"{month}")
#preprocess_matrix_sm(month=month)

def conc_vars():
    dirpath = "/p/scratch/cslts/miaari1/detect/raw"
    outpath = "/p/project/cslts/miaari1/python_scripts/DailyScriptBox/outputs/LSTM_inputs/ME"
    output_dic = {"TOT_PREC": np.array([]), "TMAX_2M": np.array([]), "TMIN_2M": np.array([]), "QFLX_EVAP_TOT": np.array([]), "wtd": np.array([]), "subSurfStor": np.array([]), "soilmoisture": np.array([])}

    months = [x for x in os.listdir(dirpath)]
    months.sort()
    for month in months:
        print(month)
        vars = [x for x in os.listdir(os.path.join(dirpath, month)) if x.endswith(".npy")]
        for var in vars:
            varname = var.replace(f"_{month}.npy","")
            data = np.load(os.path.join(dirpath, month, var))
            #print(data.shape)
            if len(output_dic[f"{varname}"])>=1:
                output_dic[f"{varname}"] = np.concatenate((output_dic[f"{varname}"], data), axis=0)
            else:
                output_dic[f"{varname}"] = data

    for key in output_dic.keys():
        print(output_dic[key].shape)
        np.save(os.path.join(outpath, f"{key}.npy"), output_dic[key])

def savelonlat():
    outdir = "/p/project1/cslts/miaari1/python_scripts/DailyScriptBox/outputs/LSTM_inputs"
    lon2D = read_nc(filepath="/p/oldscratch/cslts/miaari1/detect/raw/2001010100/wtd.nc", var="lon")
    lat2D = read_nc(filepath="/p/oldscratch/cslts/miaari1/detect/raw/2001010100/wtd.nc", var="lat")
    np.save(os.path.join(outdir, "lon2D.npy"), np.array(lon2D))
    np.save(os.path.join(outdir, "lat2D.npy"), np.array(lat2D))

def get_min_rowcol(data, region):
    lon2D = np.load("/p/project1/cslts/miaari1/python_scripts/DailyScriptBox/outputs/LSTM_inputs/lon2D.npy")
    lat2D = np.load("/p/project1/cslts/miaari1/python_scripts/DailyScriptBox/outputs/LSTM_inputs/lat2D.npy")
    maskeddata = mask_matrix(lat2D=lat2D, lon2D=lon2D, data=data, region=region)
    min_i = 500
    min_j = 500
    max_i = 0
    max_j = 0
    for i in range(maskeddata.shape[1]):
        for j in range(len(maskeddata[1,i,:])):
            if not np.ma.is_masked(maskeddata[1,i,j]):
                if i<min_i:
                    min_i = i
                #if i>max_i:
                #    max_i = i
                if j<min_j:
                    min_j = j
                #if j>max_j:
                #    max_j = j
    return min_i, min_j

def mask_vars():
    dirpath = "/p/project/cslts/miaari1/python_scripts/DailyScriptBox/outputs/LSTM_inputs"
    vars = [x for x in os.listdir(dirpath) if x.endswith(".npy")]
    region = "DANUBE_DOWNSTREAM"
    for var in vars:
        print(var)
        data = np.load(os.path.join(dirpath, var))
        data = np.ma.array(data)
        #min_i, min_j = get_min_rowcol(data, region)
        min_i = 160
        min_j = 290
        if len(data.shape)==3:
            data = data[:, min_i:min_i+30, min_j:min_j+30]
        elif len(data.shape)==2:
            data = data[min_i:min_i+30, min_j:min_j+30]
        data = np.array(data)
        print(data.shape)

        np.save(os.path.join(dirpath, region, var), data)
    return

def latlon_plot():
    data = read_nc(filepath="/p/oldscratch/cslts/miaari1/detect/raw/2001010100/wtd.nc", var="wtd")
    avg_diff = plt
    avg_diff.figure(figsize=(16,9))
    avg_diff.imshow(data[0,:,:], interpolation='nearest')
    avg_diff.colorbar()#.avg_diff.set_ylabel('wtd (m)')
    avg_diff.gca().invert_yaxis()
    avg_diff.savefig("/p/project1/cslts/miaari1/python_scripts/DailyScriptBox/wtd.png")

def preprocess_topography():
    from LSTM_setup import INPUTPATH
    # topography height in m
    topo_file = read_nc(filepath="/p/scratch/cslts/miaari1/static/topodata_CLM_EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432.nc", var="TOPO")
    topo_file = np.array(topo_file)
    topo_1yr = np.array([])
    topo_file = np.expand_dims(topo_file, axis=0)
    print(topo_file.shape)
    for i in range(365):
        print(f"topo {i}")
        topo_1yr = np.concatenate((topo_1yr,topo_file), axis=0) if len(topo_1yr)>0 else topo_file

    topo_timeseries = np.array([])
    for i in range(10):
        topo_timeseries = np.concatenate((topo_timeseries,topo_1yr), axis=0) if len(topo_timeseries)>0 else topo_1yr
    print(topo_timeseries.shape)
    np.save(os.path.join(os.path.dirname(INPUTPATH), "topo.npy"), topo_timeseries)

def preprocess_porosity():
    from LSTM_setup import INPUTPATH
    # porosity
    #open_nc("/p/scratch/cslts/miaari1/static/porosity.nc")
    poro_file = read_nc(filepath="/p/scratch/cslts/miaari1/static/porosity.nc", var="porosity")
    poro_file = poro_file[0, 6, :, :] # porosity at layer 6, same layer as soil moisture at 1m depth
    poro_file = np.array(poro_file)
    poro_1yr = np.array([])
    poro_file = np.expand_dims(poro_file, axis=0)
    print(poro_file.shape)
    for i in range(365):
        print(i)
        poro_1yr = np.concatenate((poro_1yr,poro_file), axis=0) if len(poro_1yr)>0 else poro_file
    poro_timeseries = np.array([])
    for i in range(10):
        poro_timeseries = np.concatenate((poro_timeseries,poro_1yr), axis=0) if len(poro_timeseries)>0 else poro_1yr
    print(poro_timeseries.shape)
    np.save(os.path.join(os.path.dirname(INPUTPATH), "porosity_1mdepth.npy"), poro_timeseries)

def crop_vars():
    from LSTM_setup import INPUTPATH
    topo = np.load(os.path.join(os.path.dirname(INPUTPATH), "topo.npy"))
    poro = np.load(os.path.join(os.path.dirname(INPUTPATH), "porosity_1mdepth.npy"))
    topo = topo[:, 211:211+5, 177:177+5]
    poro = poro[:, 211:211+5, 177:177+5]
    print(topo.shape)
    print(poro.shape)
    np.save(os.path.join(INPUTPATH, "topo.npy"), topo)
    np.save(os.path.join(INPUTPATH, "porosity_1mdepth.npy"), poro)

def features_correlation():
    from LSTM_setup import INPUTPATH
    from plot_functions import correlation_map

    lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
    lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))
    
    features = ["TOT_PREC.npy", "TMAX_2M.npy", "TMIN_2M.npy", "soilmoisture.npy", "QFLX_EVAP_TOT.npy"]
    targetvar = np.load(os.path.join(INPUTPATH, "wtd.npy"))
    targetvar = targetvar[:365*7,:,:]
    for f in features:
        feature = np.load(os.path.join(INPUTPATH, f))
        feature = feature[:365*7,:,:]
        print(feature.shape)
        correlation_map(targetvar, feature, f"10yrscorrelation_wtd_{f.replace('.npy','')}", 5, 5, lons, lats)

