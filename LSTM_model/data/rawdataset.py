import tarfile
import os
import numpy as np
import argparse
from SLOTH.sloth.IO import readSa
from LSTM_model.utils.utils import utilities
from LSTM_model.utils.volumetric_soilmoisture import calculate_soilmoisture
from LSTM_model.model.config import *


class preprocess_rawdata:
    def __init__(self) -> None:
        pass

    def extract_file_from_tar(self, filename, month):
        utils = utilities()
        # filename example: "parflow/wtd.nc"
        source_path = "/p/largedata2/detectdata/CentralDB/projects/d02/working_directory/sim/DETECT_EUR-11_ECMWF-ERA5_evaluation_r1i1p1_FZJ-COSMO5-01-CLM3-5-0-ParFlow3-12-0_v1Baseline/postpro/ProductionV1"
        dst_path = "/p/scratch/cslts/miaari1/raw"
        files = [file for file in os.listdir(source_path) if f"{month}.tar" in file]
        files.sort()
        tarname = f"{month}.tar"
        utils.make_dir(os.path.join(dst_path, month))
        
        with tarfile.open(os.path.join(source_path, tarname), 'r:') as tar:
            filedata = tar.extractfile(f'{month}/{filename}')

            # --------------------- save file ---------------------------
            with open(os.path.join(dst_path, month, f"{filename.split('/')[-1]}"), "wb") as outfile:
                outfile.write(filedata.read())
            outfile.close()
        tar.close()


    def extract_tar(self, file_timesteps, month):
        utils = utilities()
        volumetric_soilmoisture = calculate_soilmoisture()
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
            volumetric_soilmoisture.calculate_soilmoisture_diag(month)
            # delete saturation files
            utils.delete_files(os.path.join(dst_path, month), "saturation")

    def extract_calc_soilmoisture(self, month):
        exdir = "/p/scratch/cslts/miaari1/2001010100/parflow/"
        filenames = [x.split(".out.")[-1] for x in os.listdir(exdir) if "_saturation.nc" in x]
        filenames.sort()
        self.extract_tar(file_timesteps=filenames, month=month)
        
    def temporalAgg_matrix(self, data, timestep, agg):
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

    def preprocess_matrix_sm(self, month):
        utils = utilities()
        dirpath = "/p/scratch/cslts/miaari1/soilmoisture"
        outpath = "/p/scratch/cslts/miaari1/raw"
        #months = os.listdir("/p/scratch/cslts/miaari1/detect/raw")
        #for month in months:
        monthfiles = [x for x in os.listdir(dirpath) if f"{month}" in x]
        monthfiles.sort()
        monthly_data = []
        for dailyfile in monthfiles:
            dailydata = utils.read_nc(filepath=os.path.join(dirpath, dailyfile), var="volsm")
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


    def preprocess_matrix(self, month):
        utils = utilities()
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
            var_data = utils.read_nc(filepath=os.path.join(outpath, month, var), var=varname)
            lat2D = utils.read_nc(filepath=os.path.join(outpath, month, var), var="lat")
            lon2D = utils.read_nc(filepath=os.path.join(outpath, month, var), var="lon")
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
            
            data = self.temporalAgg_matrix(var_data, timestep, Tagg)
            #print(data.shape)
            outfile = os.path.join(outpath, month, f"{varname}_{month}.npy")
            np.save(outfile, data)

    def savelonlat(self):
        utils = utilities()
        outdir = os.path.join(get_root_dir(), "inputs")
        lon2D = utils.read_nc(filepath="/p/oldscratch/cslts/miaari1/detect/raw/2001010100/wtd.nc", var="lon")
        lat2D = utils.read_nc(filepath="/p/oldscratch/cslts/miaari1/detect/raw/2001010100/wtd.nc", var="lat")
        np.save(os.path.join(outdir, "lon2D.npy"), np.array(lon2D))
        np.save(os.path.join(outdir, "lat2D.npy"), np.array(lat2D))

    def sa_to_npy(self):
        filepath = os.path.join(os.path.dirname(INPUTPATH), "EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432_YSLOPE_TPS_HydroRIVER_sea_streams_corr.sa")
        ind = readSa(filepath)
        ind = ind[0,:,:]
        np.save(os.path.join(os.path.dirname(INPUTPATH), "slopey.npy"), ind)

    def preprocess_raw_vpd(self):
        utils = utilities()
        outpath = "/p/scratch/cslts/miaari1/raw"
        rh = "RELHUM_2M_ts.nc"
        t = "T_2M_ts.nc"

        # define timestep
        timestep = 60

        # iterate through the months
        for month in os.listdir(outpath):
            print(month)
            rh_data = utils.read_nc(filepath=os.path.join(outpath, month, rh), var=rh.replace("_ts.nc",""))
            t_data = utils.read_nc(filepath=os.path.join(outpath, month, t), var=t.replace("_ts.nc",""))
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
            data = self.temporalAgg_matrix(vpd, timestep, Tagg)
            print("final")
            print(data.shape)
            outfile = os.path.join(outpath, month, f"vpd_{month}.npy")
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