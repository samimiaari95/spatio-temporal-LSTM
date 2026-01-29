import tarfile
import os
import numpy as np
import argparse
from SLOTH.sloth.IO import readSa
from LSTM_model.utils.utils import utilities
from LSTM_model.utils.plot_functions import plotting_helper
#from LSTM_model.utils.volumetric_soilmoisture import calculate_soilmoisture
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
            print(f"month: {month}, day: {day}")
            dailydata = dailydata[:, 6:, :, :] # NOTE soil moisture only at layer 6
            # soil layers thickness in https://icg4geo.icg.kfa-juelich.de/Configurations/TSMP_statfiles_IBG3/TSMP_EUR-11/-/blob/main/static.resource/06_Texture_Indicator/01_fitRescaleAnyClassify_SoilGridsv2017.py?ref_type=heads
            # also found in py file in scratch/static
            dailydata = np.sum(dailydata, axis=1) # sum over the layers
            dailytimestep = np.mean(dailydata, axis=0)
            monthly_data.append(dailytimestep)
        monthly_data = np.array(monthly_data)
        outfile = os.path.join(outpath, month, f"sm_1mcum_{month}.npy")
        np.save(outfile, monthly_data)

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

            # remove the last timestep from cosmo output
            rh_data = np.array(rh_data[:-1,:,:])
            t_data = np.array(t_data[:-1,:,:])
            t_data = t_data - 273.15 # convert temperature to degree celsius

            ##### calculate vapor pressure deficit #####
            # saturated vapor pressure
            svp = 0.611*np.exp((17.27*t_data)/(t_data+237.3))
            # vapor pressure deficit
            vpd = svp*(1-(rh_data/100))

            Tagg = "mean"
            data = self.temporalAgg_matrix(vpd, timestep, Tagg)
            outfile = os.path.join(outpath, month, f"vpd_{month}.npy")
            np.save(outfile, data)
    
    def calculate_vpd_fromTandTd(self, month, year):
        utils = utilities()
        dirpath = os.path.join(get_root_dir(), 'inputs', 'EU_74_-48_69_20', "raw", "ERA5-Land")
        t2m = utils.read_nc(os.path.join(dirpath, "regridded_to_EUR11", f'BonA_nn_ERA5land_t2m_{year}{month}.nc'), 't2m') - 273.15
        d2m = utils.read_nc(os.path.join(dirpath, "regridded_to_EUR11", f'BonA_nn_ERA5land_d2m_{year}{month}.nc'), 'd2m') - 273.15
        es = 0.611 * np.exp((17.27 * t2m) / (t2m + 237.3))
        ea = 0.611 * np.exp((17.27 * d2m) / (d2m + 237.3))
        vpd = es - ea
        vpd = vpd.filled(0)
        if month == "02" and vpd.shape[0]==29:
            vpd = vpd[:-1,:,:]  # remove last day for february in leap years

        print(vpd.shape)
        np.save(os.path.join(dirpath, "npy_regridded_to_EUR11", f'vpd_{year}{month}_EU.npy'), vpd)
        return

    def extract_nc_to_npy(self, infile, outfile, varname, month):
        """
        Extract a variable from a netCDF file and save it as a numpy array.

        Parameters:
        infile (str): Path to the input netCDF file.
        outfile (str): Path to the output numpy file.
        varname (str): Name of the variable to extract.
        """
        utils = utilities()
        var_data = utils.read_nc(filepath=infile, var=varname)
        var_data = var_data.filled(0)
        if month == "02" and var_data.shape[0]==29:
            var_data = var_data[:-1,:,:]  # remove last day for february in leap years
        # convert total precipitation from m to mm for ERA5 data
        if varname == "tp":
            var_data = var_data * 1000
        print(var_data.shape)
        np.save(outfile, var_data)
        return
    
    def extract_vars(self):
        dirpath = os.path.join(INPUTPATH, "raw", "ERA5-Land")
        months = [f"{i:02d}" for i in range(1,13)]
        years = [f"{x}" for x in range (1999, 2021)]
        varnames = ["swvl3", "tp"]
        for year in years:
            for month in months:
                print(f"Processing year: {year}, month: {month}")
                self.calculate_vpd_fromTandTd(month=month, year=year)
                for varname in varnames:
                    if varname == "tp":
                        infile = os.path.join(dirpath, "regridded_to_EUR11", f'BonA_conservative_at23h_ERA5land_{varname}_{year}{month}.nc')
                        outfile = os.path.join(dirpath, "npy_regridded_to_EUR11", f'{varname}_{year}{month}_EU.npy')
                    elif varname == "swvl3":
                        infile = os.path.join(dirpath, "regridded_to_EUR11", f'BonA_nn_ERA5land_{varname}_{year}{month}.nc')
                        outfile = os.path.join(dirpath, "npy_regridded_to_EUR11", f'{varname}_{year}{month}_EU.npy')
                    self.extract_nc_to_npy(infile, outfile, varname, month)
        return
    
    def rawdata_temporal_agg(self, infile, outfile, varname, method='mean'):
        """
        Perform temporal aggregation on a netCDF file and save the result to a new file.

        Parameters:
        infile (str): Path to the input netCDF file.
        outfile (str): Path to the output netCDF file.
        varname (str): Name of the variable to aggregate.
        method (str): Aggregation method ('mean', 'sum', etc.). Default is 'mean'.
        """
        utils = utilities()
        # Open the input dataset
        var_data = utils.read_nc(filepath=infile, var=varname)
        print(var_data.shape)
        dailydata = self.temporalAgg_matrix(var_data, timestep=60, agg=method)
        print(dailydata.shape)
        np.save(outfile, dailydata)
        return

    def check_ERA5_vs_TSMP_units(self):
        utils = utilities()
        dirpath = os.path.join(INPUTPATH, "raw", "2019")
        tsmpdirpath = "/p/scratch/cslts/miaari1/soilmoisture"
        print("#################### ERA5 ####################")
        era5_org = utils.read_nc(os.path.join(dirpath, f'volumetric_soil_water_layer_3_201901.nc'), "swvl3")
        print("#################### regridded ERA5 ####################")
        era5_regrid = utils.read_nc(os.path.join(dirpath, f'BonA_volumetric_soil_water_layer_3_201901.nc'), "swvl3")
        print("#################### TSMP ####################")
        tsmp = utils.read_nc(os.path.join(tsmpdirpath, f'2019010100.out.00000_volsm.nc'), "volsm")
        tsmp = tsmp[0,6:,:,:] # soil moisture only at layer 6
        print(tsmp.shape)
        print(tsmp)
        print(f"ERA5 original mean: {np.mean(era5_org)}, max: {np.max(era5_org)}, min: {np.min(era5_org)}")
        print(f"ERA5 regridded mean: {np.mean(era5_regrid)}, max: {np.max(era5_regrid)}, min: {np.min(era5_regrid)}")
        print(f"TSMP mean: {np.nanmean(tsmp)}, max: {np.nanmax(tsmp)}, min: {np.nanmin(tsmp)}")
        return
    
    def test_precipnc(self):
        utils = utilities()
        lvar = utils.read_nc(os.path.join(INPUTPATH, "raw", "grib", "ERA5_precip_201501.nc"), "var228")
        lvar = lvar*1000 # convert to mm
        print("###########################################")
        eravar = utils.read_nc(os.path.join(INPUTPATH, "raw", "grib", "era5orgnc_precip_201501.nc"), "tp")
        eravarhourly = eravar*1000 # convert to mm
        eravardaily = eravarhourly.reshape(31, 24, *eravarhourly.shape[1:]).sum(axis=1)

        # list of 23:00 of each day for January 2015 (31 days)
        hours_23 = [23 + i*24 for i in range(31)]
        eralhour = eravarhourly[hours_23,:,:]
        print("###########################################")
        print(eralhour.shape)
        print(f"eralhour mean: {np.mean(eralhour)}, max: {np.max(eralhour)}, min: {np.min(eralhour)}")
        print(lvar.shape)
        print(f"lvar mean: {np.mean(lvar)}, max: {np.max(lvar)}, min: {np.min(lvar)}")
        print(eravardaily.shape)
        print(f"eravar mean: {np.mean(eravardaily)}, max: {np.max(eravardaily)}, min: {np.min(eravardaily)}")

        tsmpdirpath = "/p/scratch/cslts/miaari1/2001010100/cosmo"
        tsmp = utils.read_nc(os.path.join(tsmpdirpath, f'TOT_PREC_ts.nc'), "TOT_PREC")
        # get daily precip by summing every 24 hours at 23:00
        # remove last timestep
        tsmp = tsmp[:-1,:,:]
        tsmp_daily = tsmp.reshape(31, 24, *tsmp.shape[1:]).sum(axis=1)
        print(tsmp_daily.shape)
        print(f"tsmp_daily_201501 mean: {np.mean(tsmp_daily)}, max: {np.max(tsmp_daily)}, min: {np.min(tsmp_daily)}")
        tsmp_23hour = tsmp[hours_23,:,:]
        print(tsmp_23hour.shape)
        print(f"tsmp_23hour mean: {np.mean(tsmp_23hour)}, max: {np.max(tsmp_23hour)}, min: {np.min(tsmp_23hour)}")

        utils.open_nc(os.path.join(INPUTPATH, "raw", "regrid_2015.nc"))

    def test_tp_TSMPvsERA5(self):
        # NOTE keep this function as a proof of why only the timestep at 23:00 is used for ERA5 while the daily sum is used for TSMP
        import matplotlib.pyplot as plt
        utils = utilities()
        tsmppath = "/p/scratch/cslts/miaari1/2020120100"
        tsmp = utils.read_nc(os.path.join(tsmppath, f'TOT_PREC_ts.nc'), 'TOT_PREC')
        # remove last timestep
        tsmp = tsmp[:-1,:,:]
        print(tsmp.shape)

        # load ERA5 regridded
        erapath = os.path.join(INPUTPATH, "raw", "testingERA5")
        era5 = utils.read_nc(os.path.join(erapath, f'BonAERA5land_tp_202012.nc'), 'tp')
        # convert from m to mm
        era5 = era5 * 1000
        print(era5.shape)
        print(f"tsmp mean: {np.mean(tsmp)}, max: {np.max(tsmp)}, min: {np.min(tsmp)}")
        print(f"era5 mean: {np.mean(era5)}, max: {np.max(era5)}, min: {np.min(era5)}")

        # choose a pixel to plot cumulative time series
        lat_idx = 200
        lon_idx = 150
        tsmp_pixel = tsmp[:, lat_idx, lon_idx]
        era5_pixel = era5[:, lat_idx, lon_idx]

        # calculate daily sum for both datasets
        tsmp_daily = tsmp_pixel.reshape(31, 24).sum(axis=1)
        era5_daily = era5_pixel.reshape(31, 24).sum(axis=1)
        # get timestep at 23:00 for era5
        era5_23 = era5_pixel[23::24]

        print(tsmp_daily.shape)
        print(era5_daily.shape)
        print(era5_23.shape)
        print(f"tsmp_daily mean: {np.mean(tsmp_daily)}, max: {np.max(tsmp_daily)}, min: {np.min(tsmp_daily)}")
        print(f"era5_daily mean: {np.mean(era5_daily)}, max: {np.max(era5_daily)}, min: {np.min(era5_daily)}")
        print(f"era5_23 mean: {np.mean(era5_23)}, max: {np.max(era5_23)}, min: {np.min(era5_23)}")

        plt.figure(figsize=(10,5))
        plt.plot(tsmp_daily.cumsum(), label='TSMP')
        plt.plot(era5_23.cumsum(), label='ERA5')
        plt.plot(era5_daily.cumsum(), label='ERA5_dailysum')
        plt.title(f'Cumulative Time Series of Total Precipitation at pixel ({lat_idx}, {lon_idx}) - December 2020')
        plt.xlabel('Time (hours)')
        plt.ylabel('Total Precipitation (mm)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(INPUTPATH, "raw", "testingERA5", f"tsmp_vs_era5_pixel_{lat_idx}_{lon_idx}.png"))
        plt.close()

    def test_selected_23h_timestep_precipnc(self):
        utils = utilities()
        # fresh from era5-land download
        era5 = utils.read_nc(os.path.join(INPUTPATH, "raw", "ERA5-Land", "ERA5land_tp_201603.nc"), "tp")
        # convert to mm
        era5 = era5*1000
        # select only the 23:00 timestep of each day
        hours_23 = [23 + i*24 for i in range(31)]
        era5_23 = era5[hours_23,:,:]

        # from cdo selected 23h timestep
        era5_cdo = utils.read_nc(os.path.join(INPUTPATH, "raw", "ERA5-Land", "daily_23h", "at23h_ERA5land_tp_201603.nc"), "tp")
        era5_cdo = era5_cdo*1000
        print(era5_23.shape)
        print(f"era5_23 mean: {np.mean(era5_23)}, max: {np.max(era5_23)}, min: {np.min(era5_23)}")
        print(era5_cdo.shape)
        print(f"era5_cdo mean: {np.mean(era5_cdo)}, max: {np.max(era5_cdo)}, min: {np.min(era5_cdo)}")

    def test(self):
        utils = utilities()
        utils.open_nc(os.path.join(INPUTPATH, "raw", "ERA5-Land", "ERA5land_t2m_201501.nc"))
        utils.open_nc(os.path.join(INPUTPATH, "raw", "ERA5-Land", "ERA5land_d2m_201501.nc"))
        utils.open_nc(os.path.join(INPUTPATH, "raw", "ERA5-Land", "ERA5land_swvl3_201501.nc"))
        t2m = utils.read_nc(os.path.join(INPUTPATH, "raw", "ERA5-Land", "ERA5land_t2m_201501.nc"), "t2m")
        d2m = utils.read_nc(os.path.join(INPUTPATH, "raw", "ERA5-Land", "ERA5land_d2m_201501.nc"), "d2m")
        swvl3 = utils.read_nc(os.path.join(INPUTPATH, "raw", "ERA5-Land", "ERA5land_swvl3_201501.nc"), "swvl3")
        print(t2m.shape)
        print(d2m.shape)
        print(swvl3.shape)
        print(f"t2m mean: {np.mean(t2m)}, max: {np.max(t2m)}, min: {np.min(t2m)}")
        print(f"d2m mean: {np.mean(d2m)}, max: {np.max(d2m)}, min: {np.min(d2m)}")
        print(f"swvl3 mean: {np.mean(swvl3)}, max: {np.max(swvl3)}, min: {np.min(swvl3)}")

    def check_cdo_remapping(self):
        utils = utilities()
        regridpath = os.path.join(INPUTPATH, "raw", "regridchecks_ERA5", "dataoutput")
        data1 = utils.read_nc(os.path.join(regridpath, f'BonA_conservative_at23h_ERA5land_tp_202012.nc'), 'tp')
        data2 = utils.read_nc(os.path.join(regridpath, f'at23h_ERA5land_tp_202012_EUR11_con.nc'), 'tp')
        data1 = np.array(data1)
        data2 = np.array(data2)
        print(data1.shape)
        print(data2.shape)
        print(data1[0, :5, :5])
        print(data2[0, :5, :5])
        print(np.unique(np.equal(data1, data2), return_counts=True))
        # print only the differences
        diff = data1 - data2
        print("Differences between data1 and data2 except nan:")
        diff = diff[~np.isnan(diff)]
        print("difference without 0")
        diff_nozero = diff[diff != 0]
        print(diff_nozero)
        print("difference more than 1")
        diff_small = diff_nozero[np.abs(diff_nozero) > 0.001] #mm for precip
        print(diff_small)
        print(f"Number of different values: {len(diff_small)}")
