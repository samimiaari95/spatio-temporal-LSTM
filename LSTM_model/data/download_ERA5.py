import cdsapi
import os
import calendar
import threading
import zipfile
from queue import Queue
from LSTM_model.model.config import *


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class DownloadWorker(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            year, month = self.queue.get()
            try:
                download_era5_t2m_d2m_swvl3_derived_era5_land_daily_statistics(download_dir=os.path.join(INPUTPATH, "raw", "ERA5-Land"), year=str(year), month=f"{month:02d}")
            except Exception as e:
                print(f"Error downloading {year}-{month:02d}: {e}")
            finally:
                self.queue.task_done()

def download_era5_tp(download_dir, year, month):
    dataset = "reanalysis-era5-land"
    request = {
        "variable": ["total_precipitation"],
        "year": year,
        "month": month,
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": ["23:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [74, -48, 20, 69]
    }

    filename = f"at23h_ERA5land_tp_{year}{month}.nc"
    filepath = os.path.join(download_dir, filename)
    print(f"Downloading {filename}...")
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(filepath)
    print(f"{filename} downloaded successfully.")

def download_era5_t2m_d2m_swvl3_reanalysis_era5_land(download_dir, year, month):
    dataset = "reanalysis-era5-land"
    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "volumetric_soil_water_layer_3"
        ],
        "year": year,
        "month": month,
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [74, -48, 20, 69]
    }
    filename = f"ERA5land_t2m_d2m_swvl3_{year}{month}.zip"
    filepath = os.path.join(download_dir, filename)
    print(f"Downloading {filename}...")
    client = cdsapi.Client()
    client.retrieve(dataset, request).download(filepath)
    print(f"{filename} downloaded successfully.")

def download_era5_t2m_d2m_swvl3_derived_era5_land_daily_statistics(download_dir, year, month):
    dataset  = "derived-era5-land-daily-statistics"
    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "volumetric_soil_water_layer_3"
        ],
        "year": year,
        "month": month,
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
    "daily_statistic": "daily_mean",
    "time_zone": "utc+00:00",
    "frequency": "1_hourly",
    "area": [74, -48, 20, 69]
    }
    filename = f"ERA5land_t2m_d2m_swvl3_{year}{month}.zip"
    filepath = os.path.join(download_dir, filename)
    
    print(f"Downloading {filename}...")
    client = cdsapi.Client() #cdsapi.Client(timeout=600)
    client.retrieve(dataset, request).download(filepath)
    print(f"{filename} downloaded successfully.")

def extract_downloaded_zips(download_dir, year, month):
    zip_path = os.path.join(download_dir, f"ERA5land_t2m_d2m_swvl3_{year}{month}.zip")
    extract_to = download_dir

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # if only one file in zip, extract then split by the following:
        # cdo selname,var1 input.nc var1.nc
        # cdo selname,var2 input.nc var2.nc
        # cdo selname,var3 input.nc var3.nc
        zip_ref.extractall(extract_to)
        print(f"Extracted ERA5land_t2m_d2m_swvl3_{year}{month}.zip")
        filenames = zip_ref.namelist()
        for filename in filenames:
            filepath = os.path.join(extract_to, filename)
            # rename file to the format ERA5land_variable_yyyymm.nc
            if "2m_temperature" in filename:
                variable = "t2m"
            elif "2m_dewpoint_temperature" in filename:
                variable = "d2m"
            elif "volumetric_soil_water_layer_3" in filename:
                variable = "swvl3"
            else:
                variable = f"ERA5land_t2m_d2m_swvl3_{year}{month}"

            new_filename = f"ERA5land_{variable}_{year}{month}.nc"
            new_filepath = os.path.join(extract_to, new_filename)
            os.rename(filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")

def request_ERA5_t2m_d2m_swvl3():
    for year in range(1999, 2000):
        for month in range(1, 13):
            download_era5_t2m_d2m_swvl3_derived_era5_land_daily_statistics(download_dir=os.path.join(INPUTPATH, "raw", "ERA5-Land"), year=str(year), month=f"{month:02d}")

def request_ERA5_tp():
    for year in range(2000, 2001):
        for month in range(7, 13):
            download_era5_tp(download_dir=os.path.join(INPUTPATH, "raw", "ERA5-Land"), year=str(year), month=f"{month:02d}")

def extract_allzipfiles():
    download_dir=os.path.join(INPUTPATH, "raw", "ERA5-Land")
    zip_files = [f for f in os.listdir(download_dir) if f.endswith('.zip')]
    for zip_file in zip_files:
        zip_path = os.path.join(download_dir, zip_file)
        year_month = zip_file.split('_')[-1].replace('.zip', '')
        year = year_month[:4]
        month = year_month[4:6]
        extract_to = download_dir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # if only one file in zip, extract then split by the following:
            # cdo selname,var1 input.nc var1.nc
            # cdo selname,var2 input.nc var2.nc
            # cdo selname,var3 input.nc var3.nc
            zip_ref.extractall(extract_to)
            print(f"Extracted ERA5land_t2m_d2m_swvl3_{year}{month}.zip")
            filenames = zip_ref.namelist()
            for filename in filenames:
                # rename file to the format ERA5land_variable_yyyymm.nc
                filepath = os.path.join(extract_to, filename)
                if "2m_temperature" in filename:
                    variable = "t2m"
                elif "2m_dewpoint_temperature" in filename:
                    variable = "d2m"
                elif "volumetric_soil_water_layer_3" in filename:
                    variable = "swvl3"
                else:
                    variable = f"t2m_d2m_swvl3"

                new_filename = f"ERA5land_{variable}_{year}{month}.nc"
                new_filepath = os.path.join(extract_to, new_filename)
                os.rename(filepath, new_filepath)
                print(f"Renamed {filename} to {new_filename}")

# NOTE the download on HPC is from the julich copernicus account while on the mac is from my gmail account
