import os
import shutil
import tarfile
import numpy as np
from pathlib import Path, PurePosixPath
#from LSTM_model.utils.volumetric_soilmoisture import calculate_soilmoisture
from LSTM_model.model.config import *


class preprocess_rawdata:
    def stage_files(self, months, filenames):
        if not months or not filenames:
            raise ValueError("At least one archive month and one member path are required.")

        staged_files = []
        for month in months:
            if not month.isdigit():
                raise ValueError(f"Archive month must contain digits only: {month!r}")
            archive_path = Path(RAW_DATA_SOURCE) / f"{month}.tar"
            if not archive_path.is_file():
                raise FileNotFoundError(f"Raw archive not found: {archive_path}")
            destination_dir = Path(RAW_STAGING_PATH) / month
            destination_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(archive_path, "r:*") as archive:
                used_destinations = set()
                for filename in filenames:
                    member_path = PurePosixPath(filename)
                    if (
                        member_path.is_absolute()
                        or not member_path.parts
                        or any(part in {"", ".", ".."} for part in member_path.parts)
                    ):
                        raise ValueError(f"Invalid archive member path: {filename!r}")

                    archive_member = PurePosixPath(month, *member_path.parts).as_posix()
                    source = archive.extractfile(archive_member)
                    if source is None:
                        raise FileNotFoundError(
                            f"Archive member {archive_member!r} is missing from "
                            f"{archive_path}."
                        )
                    destination = destination_dir / member_path.name
                    if destination in used_destinations:
                        raise ValueError(
                            f"Multiple archive members would overwrite {destination}."
                        )
                    used_destinations.add(destination)
                    with source, destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    staged_files.append(str(destination))

        return staged_files

    def extract_file_from_tar(self, filename, month):
        return self.stage_files([month], [filename])[0]

    def extract_tar(self, file_timesteps, month):
        from LSTM_model.utils.volumetric_soilmoisture import calculate_soilmoisture

        if not month.isdigit():
            raise ValueError(f"Archive month must contain digits only: {month!r}")
        destination = Path(RAW_STAGING_PATH) / month
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(Path(RAW_DATA_SOURCE) / f"{month}.tar", "r:*") as archive:
            for timestep in file_timesteps:
                if (
                    any(
                        marker in month
                        for marker in ("040100", "060100", "090100", "110100")
                    )
                    and "2977" in timestep
                ):
                    continue
                if "020100" in month and any(
                    marker in timestep for marker in ("2785", "2881", "2977")
                ):
                    continue
                member_path = (
                    PurePosixPath(month)
                    / "parflow"
                    / f"ParFlow_EU11_{month}.out.{timestep}"
                )
                source = archive.extractfile(member_path.as_posix())
                if source is None:
                    raise FileNotFoundError(
                        f"Archive member {member_path} is missing for {month}."
                    )
                with source, (destination / member_path.name).open("wb") as output:
                    output.write(source.read())

        calculate_soilmoisture().calculate_soilmoisture_diag(month)

    def extract_calc_soilmoisture(self, month):
        exdir = os.path.join(RAW_STAGING_PATH, month)
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
        from LSTM_model.utils.utils import utilities

        utils = utilities()
        dirpath = os.path.join(RAW_STAGING_PATH, "soilmoisture")
        outpath = RAW_STAGING_PATH
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


    def preprocess_matrix(self, month):
        from LSTM_model.utils.utils import utilities

        utils = utilities()
        outpath = RAW_STAGING_PATH
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
        from LSTM_model.utils.utils import utilities

        utils = utilities()
        outdir = os.path.join(get_root_dir(), "inputs")
        wtd_path = os.path.join(RAW_STAGING_PATH, "2001010100", "wtd.nc")
        lon2D = utils.read_nc(filepath=wtd_path, var="lon")
        lat2D = utils.read_nc(filepath=wtd_path, var="lat")
        np.save(os.path.join(outdir, "lon2D.npy"), np.array(lon2D))
        np.save(os.path.join(outdir, "lat2D.npy"), np.array(lat2D))

    def sa_to_npy(self):
        from SLOTH.sloth.IO import readSa

        filepath = os.path.join(os.path.dirname(INPUTPATH), "EUR-11_TSMP_FZJ-IBG3_CLMPFLDomain_444x432_YSLOPE_TPS_HydroRIVER_sea_streams_corr.sa")
        ind = readSa(filepath)
        ind = ind[0,:,:]
        np.save(os.path.join(os.path.dirname(INPUTPATH), "slopey.npy"), ind)

    def preprocess_raw_vpd(self):
        from LSTM_model.utils.utils import utilities

        utils = utilities()
        outpath = RAW_STAGING_PATH
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
    
    def avgwtd_2020(self):
        from LSTM_model.utils.plot_functions import plotting_helper

        plots = plotting_helper()
        wtd = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "wtd.npy"))
        mapping = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "ensemble_400px_org", "mapping_0stdroll6months.npy"))
        print(np.sum(mapping))
        print(wtd.shape)
        wtd = wtd[-365:,:,:]
        wtd = np.mean(wtd, axis=0)
        print(wtd.shape)
        wtd = np.where(mapping==1, wtd, np.nan)
        print(wtd.shape)
        # plot it in 2d map
        plots.EU_2Dmap(data_map=wtd,logscale=False, minval=0, maxval=50, title="Water table depth")

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