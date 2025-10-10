import os
import cdsapi

def request_era5(variable, year, month, target):
    dataset = "reanalysis-era5-land"
    request = {
        "variable": [variable],
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
        "download_format": "unarchived",
        "area": [74, -48, 20, 69]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(target)

if __name__ == "__main__":
    variables = ["volumetric_soil_water_layer_3", "total_precipitation", "2m_temperature", "2m_dewpoint_temperature"]
    years = ["2019"]
    months = ["04", "06", "09", "11"]

    for variable in variables:
        for year in years:
            for month in months:
                target_file = f"{variable}_{year}{month}.nc"
                print(f"Requesting {variable} for {year}-{month}")
                request_era5(variable, year, month, target_file)
                print(f"Saved to {target_file}")