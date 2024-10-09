import tarfile
import os
import numpy as np
import argparse
from utils import read_nc, delete_files, make_dir
from volumetric_soilmoisture import calculate_soilmoisture_diag
from sklearn.preprocessing import OneHotEncoder
from SLOTH.sloth.IO import readSa
from LSTM_setup import INPUTPATH


def extract_file_from_tar(filename, month):
    # filename example: "parflow/wtd.nc"
    source_path = "/p/largedata2/detectdata/CentralDB/projects/d02/working_directory/sim/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline/postpro/ProductionV1"
    dst_path = "/p/scratch/cslts/miaari1/raw"
    files = [file for file in os.listdir(source_path) if f"{month}.tar" in file]
    files.sort()
    tarname = f"{month}.tar"
    make_dir(os.path.join(dst_path, month))
    
    with tarfile.open(os.path.join(source_path, tarname), 'r:') as tar:
        filedata = tar.extractfile(f'{month}/{filename}')

        # --------------------- save file ---------------------------
        with open(os.path.join(dst_path, month, f"{filename.split('/')[-1]}"), "wb") as outfile:
            outfile.write(filedata.read())
        outfile.close()
    tar.close()


def extract_tar(file_timesteps, month):
    
    source_path = "/p/largedata2/detectdata/CentralDB/projects/d02/working_directory/sim/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline/postpro/ProductionV1"
    dst_path = "/p/scratch/cslts/miaari1/raw"
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
        # use heat environment from diagnostics, without any modification. just comment in this script the not installed libraries 
        calculate_soilmoisture_diag(month)
        # delete saturation files
        delete_files(os.path.join(dst_path, month), "saturation")

def extract_calc_soilmoisture(month):
    exdir = "/p/scratch/cslts/miaari1/2001010100/parflow/"
    filenames = [x.split(".out.")[-1] for x in os.listdir(exdir) if "_saturation.nc" in x]
    filenames.sort()
    extract_tar(file_timesteps=filenames, month=month)
    
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

def preprocess_matrix_sm(month):
    dirpath = "/p/scratch/cslts/miaari1/soilmoisture"
    outpath = "/p/scratch/cslts/miaari1/raw"
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
        print(month)
        print(day)
        dailydata = dailydata[:, 6, :, :] # NOTE soil moisture only at layer 6
        dailytimestep = np.mean(dailydata, axis=0)
        monthly_data.append(dailytimestep)
    monthly_data = np.array(monthly_data)
    outfile = os.path.join(outpath, month, f"soilmoisture_{month}.npy")
    np.save(outfile, monthly_data)


def preprocess_matrix(month):
    outpath = "/p/scratch/cslts/miaari1/raw"
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
        #print(var)
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
        #print(data.shape)
        outfile = os.path.join(outpath, month, f"{varname}_{month}.npy")
        np.save(outfile, data)


# NOTE the parse arguments were only used to parallel compute the soil moisture
#parser = argparse.ArgumentParser(description='insert the preprocessing month')
#parser.add_argument('--month', type=str, required=True,
#                    help='yyyymmdd00') # month of simulation
#args = parser.parse_args()
#month = args.month
#extract_calc_soilmoisture(month=month)
#extract_file_from_tar(filename,month)
#preprocess_matrix(f"{month}")
#preprocess_matrix_sm(month=month)

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

def savelonlat():
    outdir = "/p/project1/cslts/miaari1/python_scripts/DailyScriptBox/outputs/LSTM_inputs"
    lon2D = read_nc(filepath="/p/oldscratch/cslts/miaari1/detect/raw/2001010100/wtd.nc", var="lon")
    lat2D = read_nc(filepath="/p/oldscratch/cslts/miaari1/detect/raw/2001010100/wtd.nc", var="lat")
    np.save(os.path.join(outdir, "lon2D.npy"), np.array(lon2D))
    np.save(os.path.join(outdir, "lat2D.npy"), np.array(lat2D))

def crop_vars(varname):
    from LSTM_setup import INPUTPATH
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

def sa_to_npy():
    filepath = os.path.join(os.path.dirname(INPUTPATH), "EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432_YSLOPE_TPS_HydroRIVER_sea_streams_corr.sa")
    ind = readSa(filepath)
    print(ind.shape)

    ind = ind[0,:,:]
    print(ind.shape)
    np.save(os.path.join(os.path.dirname(INPUTPATH), "slopey.npy"), ind)

def preprocess_raw_vpd():
    outpath = "/p/scratch/cslts/miaari1/raw"
    rh = "RELHUM_2M_ts.nc"
    t = "T_2M_ts.nc"

    # define timestep
    timestep = 60

    # iterate through the months
    for month in os.listdir(outpath):
        print(month)
        rh_data = read_nc(filepath=os.path.join(outpath, month, rh), var=rh.replace("_ts.nc",""))
        t_data = read_nc(filepath=os.path.join(outpath, month, t), var=t.replace("_ts.nc",""))
        print(rh_data.shape)

        # remove the last timestep from cosmo output
        rh_data = np.array(rh_data[:-1,:,:])
        t_data = np.array(t_data[:-1,:,:])

        ##### calculate vapor pressure deficit #####
        # saturated vapor pressure
        svp = 0.611*np.exp((17.27*t_data)/(t_data+237.3))
        # vapor pressure deficit
        vpd = svp*(1-(rh_data/100))

        Tagg = "mean"
        data = temporalAgg_matrix(vpd, timestep, Tagg)
        print("final")
        print(data.shape)
        outfile = os.path.join(outpath, month, f"vpd_{month}.npy")
        np.save(outfile, data)

def conc_basins():
    from LSTM_setup import FEATURES_FILES, TARGETVAR_FILE
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


def create_choices():
    mapping = np.load(os.path.join(os.path.dirname(INPUTPATH), "mapping_1mstd.npy"))
    map1d = np.where(mapping==1)
    nb_samples = 225
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

def apply_onehotencoding():
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
    distribute_onehotencoding(one_hot_encoded, "soilind.npy")
