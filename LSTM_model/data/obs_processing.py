import os
import math
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from LSTM_model.utils.utils import utilities
from LSTM_model.utils.plot_functions import plotting_helper
from pyproj import Transformer
from LSTM_model.model.config import *

##### required data for local observations #####
# 'wellID', 'country', 'lon', 'lat', 'tsmp_lon', 'tsmp_lat', 'tsmp_xindex', 'tsmp_yindex', 'target_chunk', 'sim_1darrayindex'
# save timeseries in parquet file with columns named as Country_LONxxxLATyyy

import warnings
warnings.filterwarnings("ignore")


class obs_processing:
    def __init__(self) -> None:
        pass

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
    
    def find_leastdistance_TSMPpixel_to_EUobs(self, lon, lat):
        # load TSMP lonlat
        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

        # find nearest TSMP grid point
        # explain how it is done
        # calculate the distance between the observation location and all TSMP grid points
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
            # print(f"Not in TSMP dataset")
            return None, None
        target = int(chunksmap[yindex, xindex])
        target_map = np.load(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{target}", f"mappingindices_{target}.npy"))
        # find the 1d index within this target
        rows, cols = np.where(target_map == 1)
        flat_idx = np.flatnonzero((rows == yindex) & (cols == xindex))[0]
        return target, flat_idx
    
    def find_flatidx_from_2Dindices(self, yindex, xindex):
        obs_map = np.load(os.path.join(INPUTPATH, "mapping_wtdobs_YMA.npy"))
        # find the 1d index within this target
        rows, cols = np.where(obs_map == 1)
        flat_idx = np.flatnonzero((rows == yindex) & (cols == xindex))[0]
        return flat_idx

    def mapping_localobs_TSMPproj_v1(self):
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
 
    def mapping_EUobs_TSMPproj_old_v2(self):
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

    def check_duplicates(self, df):
        # check for duplicates
        df['dup'] = df['tsmp_xindex'].astype(str) + "_" + df['tsmp_yindex'].astype(str)
        duplicates = df[df.duplicated(subset=['dup'], keep=False)]
        if not duplicates.empty:
            print("Duplicates found:")
            return duplicates, df
        return None, df
        df.drop_duplicates(subset=['dup'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_obs_TSMP_noduplicates.csv"), index=False)
        print(df)
        # TODO process duplicates by averaging their observations

    def mapping_obs_TSMP(self):
        # keep all columns except well id
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        locations = pd.DataFrame(columns=['country', 'lon', 'lat', 'tsmp_lon', 'tsmp_lat', 'tsmp_xindex', 'tsmp_yindex', 'target_chunk', 'sim_1darrayindex'])
        for column in df_eu.columns:
            country = column.split("_")[0]
            strlon = f'_{column.split("LON")[1].split("LAT")[0]}'
            strlat = f'_{column.split("LAT")[1]}'
            lon = float(column.split("LON")[1].split("LAT")[0])
            lat = float(column.split("LAT")[1])
            yindex, xindex, tsmp_lon, tsmp_lat = self.find_leastdistance_TSMPpixel_to_EUobs(lon, lat)
            # find which target chunk it belongs to
            target, flat_idx = self.find_chunk_flatidx_from_2Dindices(yindex, xindex)
            if target is None:
                continue
            locations = pd.concat([locations, pd.DataFrame({'country': [country], 'lon': [strlon], 'lat': [strlat], 'tsmp_lon': [tsmp_lon], 'tsmp_lat': [tsmp_lat], 'tsmp_xindex': [xindex], 'tsmp_yindex': [yindex], 'target_chunk': [target], 'sim_1darrayindex': [flat_idx]})], ignore_index=True)
        print(locations)
        # locations.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_obs_TSMP.csv"), index=False)
        # locations = pd.read_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_obs_TSMP.csv"))
        self.check_duplicates(locations, df_eu)


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
        df.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
    
    def wtdobs_Denmark(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "obs_wtd_seine_BDDADES.parquet"))
        df_eu.dropna(axis=1, inplace=True)
        df_eu.reset_index(inplace=True, drop=True)
        # rename columns to match the format Country_LONxxxLATyyy
        newcols = {}
        for col in df_eu.columns:
            newcol = f"France_{col}"
            newcols[col] = newcol
        df_eu.rename(columns=newcols, inplace=True)
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_EU", "obsWTD_point_measurements_in_Denmark_GEUS", "o.data", "WTD_measurements_Denmark")
        format = '%d-%m-%Y  %H:%M:%S'

        for file in [x for x in os.listdir(src) if x.endswith('.csv') and x!='jupiter.csv']:
            welldata = pd.read_csv(os.path.join(src, file),encoding='cp865',sep=';',usecols=[1,3,6],skiprows=1, names=['Well_index','time','WTD [m]'])
            welldata['WTD [m]'] = welldata['WTD [m]'].astype(str).str.replace(",", ".", regex=False).astype(float)
            try:
                welldata['time'] = pd.to_datetime(welldata['time'],format=format)
            except:
                continue
            welldata.reset_index(drop=True, inplace=True)
            if welldata.empty:
                print(f"No data for well {file}, skipping")
                continue

            coordinate_dataset = pd.read_csv(os.path.join(src, 'jupiter.csv'),encoding='cp865',sep=',',usecols=[0,31,32])
            X = coordinate_dataset[coordinate_dataset['DGUNR']==welldata['Well_index'].iloc[0].replace(" ", "")].iloc[0,1]
            Y = coordinate_dataset[coordinate_dataset['DGUNR']==welldata['Well_index'].iloc[0].replace(" ", "")].iloc[0,2]        
            #Transform coordinates in the ETRS89 / UTM zone 32N (EPSG: 25832) to WGS84
            transformer = Transformer.from_crs(25832,4326)
            (lat, lon) = transformer.transform(X,Y)

            # filter dates
            mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
            data = welldata.loc[mask]
            data.drop(columns=['Well_index'], inplace=True)
            data = data.dropna(axis=1, how='all')
            if data.empty or 'WTD [m]' not in data.columns:
                print(f"No data for LON{lon}LAT{lat}, skipping")
                continue
            data = data.dropna(axis=0, how='any', subset=['WTD [m]'])
            data.reset_index(drop=True, inplace=True)
            data = data.dropna(subset=['WTD [m]'])
            data.reset_index(drop=True, inplace=True)
            if (len(data) == len(all_dates)):
                df_eu[f"Denmark_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]'].to_list()
                print(f"added well LON{lon}LAT{lat}")
            elif len(data) > len(all_dates):
                # check if more than one measurement per day then calculate daily average
                data = data.set_index('time')
                data = data.resample('D').mean()
                data.dropna(inplace=True, how='any', subset=['WTD [m]'])
                data.reset_index(drop=True, inplace=True)
                if len(data) == len(all_dates):
                    df_eu[f"Denmark_LON{lon}LAT{lat}"] = data['WTD [m]'].to_list()
                    print(f"added well LON{lon}LAT{lat}")
            else:
                pass
                print(len(data)," not enough data points for well expected ", len(all_dates))
        print(df_eu)
        print(df_eu.columns)
        precols = df_eu.columns
        df_eu.reset_index(inplace=True, drop=True)
        df_eu.dropna(axis=1, inplace=True)
        postcols = df_eu.columns
        if len(precols) != len(postcols):
            print("Total wells in EU obs dataset after adding Denmark: ", precols)
            print("Total wells in EU obs dataset after dropping NaN columns: ", postcols)
            print("Dropped ", len(precols) - len(postcols), " columns with NaN values")
            diffcols = set(precols) - set(postcols)
            print("Dropped columns: ", diffcols)
            raise ValueError("Dropped columns with NaN values after adding Denmark wells, please check the data!")
        else:
            print("Total wells in EU obs dataset after adding Denmark: ", precols)
            print("No columns were dropped after adding Denmark wells.")
        # None found
    
    def wtdobs_portugal(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "obs_wtd_seine_BDDADES.parquet"))
        df_eu.dropna(axis=1, inplace=True)
        df_eu.reset_index(inplace=True, drop=True)
        df_eu.set_index('date', inplace=True)
        df_eu = df_eu[(df_eu.index >= '2017-01-01') & (df_eu.index <= '2019-12-31')]
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
        df_eu = df_eu.reindex(all_dates)
        df_eu.dropna(axis=1, inplace=True)

        # rename columns to match the format Country_LONxxxLATyyy
        newcols = {}
        for col in df_eu.columns:
            newcol = f"France_{col}"
            newcols[col] = newcol
        df_eu.rename(columns=newcols, inplace=True)
        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_EU", "obsWTD_point_measurements_in_Portugal_SNIRH", "o.data", "WTD_measurements_Portugal_1996_2020")
        format0 = '%d/%m/%Y  %H:%M'
        format1 = '%Y-%m-%d'
        format2 = '%Y-%m-%d %H:%M:%S'

        for file in [x for x in os.listdir(src) if x.startswith('serie_') and x.endswith('.csv')]:
            alldata = pd.read_csv(os.path.join(src, file),skiprows=2,encoding='cp1252')
            alldata.drop([col for col in alldata.columns if 'Unnamed:' in col],axis=1,inplace=True)
            alldata.drop(alldata.index[0],inplace=True)
            alldata.drop(alldata.index[-1],inplace=True)
            alldata.rename(columns = {'DATA':'time'},inplace=True)
            for col in alldata.columns[1:]:
                welldata = pd.DataFrame(columns=['time'])
                flag1 = True
                while flag1:
                    try:
                        welldata['time'] = pd.to_datetime(alldata['time'],format=format0)
                        flag1 = False
                    except:
                        alldata.drop(alldata.index[-1],inplace=True)
                    
                welldata[col] = alldata[col].astype('float64')
                try:
                    welldata['time'] = pd.to_datetime(welldata['time'],format=format1)
                except:
                    welldata['time'] = pd.to_datetime(welldata['time'],format=format2)
            welldata.reset_index(drop=True, inplace=True)
            
            coordinate_dataset = pd.read_csv(os.path.join(src, 'rede_seleccao_Piezometria.csv'),skiprows=4,encoding='cp1252',index_col=False)
            coordinate_dataset.drop(coordinate_dataset.index[-1],inplace=True)
            x = coordinate_dataset.loc[coordinate_dataset['CÓDIGO']==alldata.columns[2],'COORD_X (M)'].values[0]
            y = coordinate_dataset.loc[coordinate_dataset['CÓDIGO']==alldata.columns[2],'COORD_Y (M)'].values[0]
            #Transform coordinates in the Hayford Gauss Militar Datum Lisboa (EPSG: 20790) to WGS84
            transformer = Transformer.from_crs(20790,4326)
            (lat, lon) = transformer.transform(x,y)
            
            # filter dates
            mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
            data = welldata.loc[mask]
            data = data.dropna(axis=1, how='all')
            if data.empty or 'WTD [m]' not in data.columns:
                print(f"No data for LON{lon}LAT{lat}, skipping")
                continue
            data = data.dropna(axis=0, how='any', subset=['WTD [m]'])
            data.reset_index(drop=True, inplace=True)
            data = data.dropna(subset=['WTD [m]'])
            data.reset_index(drop=True, inplace=True)
            if (len(data) == len(all_dates)):
                df_eu[f"Portugal_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]'].to_list()
                print(f"added well LON{lon}LAT{lat}")
            elif len(data) > len(all_dates):
                # check if more than one measurement per day then calculate daily average
                data = data.set_index('time')
                data = data.resample('D').mean()
                data.dropna(inplace=True, how='any', subset=['WTD [m]'])
                data.reset_index(drop=True, inplace=True)
                if len(data) == len(all_dates):
                    df_eu[f"Portugal_LON{lon}LAT{lat}"] = data['WTD [m]'].to_list()
                    print(f"added well LON{lon}LAT{lat}")
            else:
                pass
                print(len(data)," not enough data points for well expected ", len(all_dates))
        print(df_eu)
        print(df_eu.columns)
        precols = df_eu.columns
        df_eu.reset_index(inplace=True, drop=True)
        df_eu.dropna(axis=1, inplace=True)
        postcols = df_eu.columns
        if len(precols) != len(postcols):
            print("Total wells in EU obs dataset after adding Portugal: ", precols)
            print("Total wells in EU obs dataset after dropping NaN columns: ", postcols)
            print("Dropped ", len(precols) - len(postcols), " columns with NaN values")
            diffcols = set(precols) - set(postcols)
            print("Dropped columns: ", diffcols)
            raise ValueError("Dropped columns with NaN values after adding Portugal wells, please check the data!")
        else:
            print("Total wells in EU obs dataset after adding Portugal: ", precols)
            print("No columns were dropped after adding Portugal wells.")
        # saving df with modified column name to france, though none were found in portugal
        df_eu.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))

    def wtdobs_sweden(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        df_eu.dropna(axis=1, inplace=True)
        df_eu.reset_index(inplace=True, drop=True)

    def wtdobs_netherlands(self):
        from rijksdriehoek import rijksdriehoek
        rd = rijksdriehoek.Rijksdriehoek()

        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        df_eu.dropna(axis=1, inplace=True)
        df_eu.reset_index(inplace=True, drop=True)
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')

        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_EU", "obsWTD_point_measurements_in_the_Netherlands_DINOloket", "o.data", "WTD_measurements_NL_zip")
        format = '%d-%m-%Y'

        for subdir in [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]:
            src_subdir =  os.path.join(src, subdir)
            for file in [x for x in os.listdir(src_subdir) if x.endswith('1_1.csv')]:
                skiprows = 15
                flag1 = True
                while flag1:
                    try:
                        welldata = pd.read_csv(os.path.join(src_subdir, file),skiprows=skiprows,usecols=[0,2,4],names=['location','time','wtd(cm)'])
                        welldata['time'] = pd.to_datetime(welldata['time'],format=format)
                        flag1 = False
                    except:
                        skiprows += 1
                welldata.reset_index(drop=True, inplace=True)
                
                #Load coordinates
                try:
                    coordinates_dataset = pd.read_csv(os.path.join(src_subdir, file),skiprows=11,nrows=1)
            
                    X_coord = coordinates_dataset.iloc[0,3]
                    Y_coord = coordinates_dataset.iloc[0,4]
                
                    #Convert RD coordinate to WGS'84 coordinate
                    rd.rd_x = int(X_coord)
                    rd.rd_y = int(Y_coord)
                    lat, lon = rd.to_wgs()
                except:
                    continue

                # filter dates
                mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
                data = welldata.loc[mask]
                if data['wtd(cm)'].isna().all():
                    print(f"all values are NaN for well LON{lon}LAT{lat}, skipping")
                    continue
                data = data.dropna(axis=1, how='all')
                data['WTD [m]'] = pd.to_numeric(data['wtd(cm)'], errors='coerce') / 100.0
                data.drop(columns=['wtd(cm)','location'], inplace=True)
                data = data.dropna(axis=0, how='any', subset=['WTD [m]'])
                data.reset_index(drop=True, inplace=True)
                if (len(data) == len(all_dates)):
                    df_eu[f"Netherlands_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]'].to_list()
                    print(f"added well LON{lon}LAT{lat}")
                    # print index of nan values
                    # print(df_eu[df_eu[f"Netherlands_LON{lon}LAT{lat}"].isna()].index)
                    # print(data[data['WTD [m]'].isna()].index)
                    # print(df_eu[f"Netherlands_LON{lon}LAT{lat}"].iloc[df_eu[df_eu[f"Netherlands_LON{lon}LAT{lat}"].isna()].index])
                    prevlen = len(df_eu.columns)                
                    df_eu.dropna(axis=1, inplace=True)
                    if len(df_eu.columns) != prevlen:
                        print(f"Dropped NaN values for well LON{lon}LAT{lat}, new length {len(df_eu.columns)} from previous {prevlen}")
                        print(f"column name: Netherlands_LON{lon}LAT{lat}")
                        raise ValueError("Dropped NaN values, please check the data!")
                    
                elif len(data) > len(all_dates):
                    # check if more than one measurement per day then calculate daily average
                    data = data.set_index('time')
                    data = data.resample('D').mean()
                    if len(data) == len(all_dates):
                        df_eu[f"Netherlands_LON{lon}LAT{lat}"] = data['WTD [m]'].to_list()
                        print(f"added well LON{lon}LAT{lat}")
                        prevlen = len(df_eu.columns)                
                        df_eu.dropna(axis=1, inplace=True)
                        if len(df_eu.columns) != prevlen:
                            print(f"Dropped resampled NaN values for well LON{lon}LAT{lat}, new length {len(df_eu.columns)} from previous {prevlen}")
                            print(f"column name: Netherlands_LON{lon}LAT{lat}")
                            raise ValueError("Dropped NaN values, please check the data!")
                else:
                    pass
                    print(len(data)," not enough data points for well expected ", len(all_dates))
        print(df_eu)
        print(df_eu.columns)
        precols = len(df_eu.columns)
        df_eu.reset_index(inplace=True, drop=True)
        df_eu.dropna(axis=1, inplace=True)
        postcols = len(df_eu.columns)
        if precols != postcols:
            print("Total wells in EU obs dataset after adding Netherlands: ", precols)
            print("Total wells in EU obs dataset after dropping NaN columns: ", postcols)
            print("Dropped ", precols - postcols, " columns with NaN values")
            raise ValueError("Dropped columns with NaN values after adding Netherlands wells, please check the data!")
        else:
            print("Total wells in EU obs dataset after adding Netherlands: ", precols)
            print("No columns were dropped after adding Netherlands wells.")
        # found a lot, save them later
        df_eu.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        
    
    def wtdobs_germany_Baden_Wuerttemberg(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        precols = len(df_eu.columns)
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')

        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_Germany", "obsWTD_point_measurements_in_Baden_Wuerttemberg_LUBW", "o.data", "WTD_measurements_Baden_Wuerttemberg")
        format = '%d.%m.%Y %H:%M'
            
        for file in [x for x in os.listdir(src) if x.endswith('.csv')]:
            skiprows = 7
            flag1 = True
            while flag1:
                try:
                    #Load data files
                    df = pd.read_csv(os.path.join(src, file), encoding='iso-8859-1', sep=',', usecols=[0,2,3,6,7], skiprows=skiprows)
                    flag1 = False
                except:
                    skiprows += 1
                    
            df.rename(columns={'Grundwassernummer':'Well index','Ost':'E','Nord':'N','Datum / Uhrzeit':'time','Messwert':'WTD [m]'},inplace=True)
                    
            well_indexes = pd.unique(df['Well index'])
            
            for well_index in well_indexes:
                data_well = df[df['Well index']==well_index]
                data_well.reset_index(drop=True, inplace=True)
                data_well['WTD [m]'] = data_well['WTD [m]'].astype(str).str.replace(",", ".", regex=False).astype(float)
                E = data_well.iloc[0,1]
                N = data_well.iloc[0,2]
                #Transform coordinates in the ETRS89 / UTM (EPSG: 25832) to WGS84
                transformer = Transformer.from_crs(25832,4326)
                (lat, lon) = transformer.transform(E,N)

                data_well['time'] = pd.to_datetime(data_well['time'],format=format)
                # check we have daily timestep data from 2017 to 2019
                # filter dates
                mask = (data_well['time'] >= '2017-01-01') & (data_well['time'] < '2020-01-01')
                data = data_well.loc[mask]
                data = data.dropna(axis=1, how='all')
                if (len(data) == len(all_dates)):
                    df_eu[f"Germany(wuerttemberg)_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]']
                    print(f"added well {well_index} LON{lon}LAT{lat}")
                elif len(data) > len(all_dates):
                    # check if more than one measurement per day then calculate daily average
                    data = data.set_index('time')
                    data = data.resample('D').mean()
                    if len(data) == len(all_dates):
                        df_eu[f"Germany(wuerttemberg)_LON{lon}LAT{lat}"] = data['WTD [m]']
                        print(f"added well {well_index} LON{lon}LAT{lat}")
                else:
                    pass
                    print(len(data)," not enough data points for well ", well_index," expected ", len(all_dates))
        print(df_eu.columns)
        if len(df_eu.columns) == precols:
            print("No wells added for Germany Baden Wuerttemberg")
        # not saving because no wells found with daily data from 2017-2019
    
    def wtdobs_germany_Bayern(self):
        # TODO all good but check excel file to calculate the wtd from elevation
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')

        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_Germany", "obsWTD_point_measurements_in_Bayern_GKD_Bayern", "o.data", "WTD_measurements_Bayern")
        format = '%Y-%m-%d'

        subdirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
        for subdir in subdirs:
            src_subdir =  os.path.join(src, subdir, "grundwasser-gwo")
            for file in [x for x in os.listdir(src_subdir) if x.endswith('_beginn_bis_31.12.2019_tmw.csv')]:
                welldata = pd.read_csv(os.path.join(src_subdir, file),encoding='iso-8859-1',sep=';',usecols=[0,1],skiprows=8, names=['time','WTD [m]'])
                welldata.reset_index(drop=True, inplace=True)

                #Load well ID
                well_ID_dataset = pd.read_csv(os.path.join(src_subdir, file),sep=';',header=None,skiprows=4,nrows=1)
                well_ID = well_ID_dataset.iloc[0,1]
                
                #Load coordinates
                coordinates_dataset = pd.read_csv(os.path.join(src_subdir, file),sep=';',header=None,skiprows=5,nrows=1)
                E = coordinates_dataset.iloc[0,1]
                N = coordinates_dataset.iloc[0,3]
                #Transform coordinates in the ETRS89 / UTM (EPSG: 25832) to WGS84
                transformer = Transformer.from_crs(25832,4326)
                (lat, lon) = transformer.transform(E,N)
        
                welldata['time'] = pd.to_datetime(welldata['time'],format=format)
                # check we have daily timestep data from 2017 to 2019
                # filter dates
                mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
                data = welldata.loc[mask]
                data = data.dropna(axis=1, how='all')
                if (len(data) == len(all_dates)):
                    df_eu[f"Germany(bayern)_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]']
                    print(f"added well {well_ID} LON{lon}LAT{lat}")
                elif len(data) > len(all_dates):
                    # check if more than one measurement per day then calculate daily average
                    data = data.set_index('time')
                    data = data.resample('D').mean()
                    if len(data) == len(all_dates):
                        df_eu[f"Germany(bayern)_LON{lon}LAT{lat}"] = data['WTD [m]']
                        print(f"added well {well_ID} LON{lon}LAT{lat}")
                else:
                    pass
                    print(len(data)," not enough data points for well ", well_ID," expected ", len(all_dates))
        print(df_eu.columns)
        # must save after correcting the wtd calculation from elevation
        # TODO get elevation from source data website and calculate wtd
        # complicated because we have to go for each well to the website and get the elevation manually
        # we have 246 wells added here
    
        # df_eu.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
    
    def wtdobs_germany_Hessen(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        df_eu.dropna(axis=1, inplace=True)
        df_eu.reset_index(inplace=True, drop=True)
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')

        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_Germany", "obsWTD_point_measurements_in_Hessen_LGD", "o.data", "WTD_measurements_Hessen")
        format = '%b %d, %Y'
            
        for file in [x for x in os.listdir(src) if x.endswith('.csv') and 'Stamm_MST.csv' not in x]:
            allwellsdata = pd.read_csv(os.path.join(src, file),encoding='iso-8859-1',sep=';',usecols=[0,3,5],skiprows=1, names=['Well index','time','WTD [m]'])
            well_indexes = pd.unique(allwellsdata['Well index'])
            for well_index in well_indexes:
                welldata = allwellsdata[allwellsdata['Well index']==well_index]
                welldata.reset_index(drop=True, inplace=True)

                coordintates_dataset = pd.read_csv(src+'/Stamm_MST.csv',encoding='iso-8859-1',sep=';',usecols=[0,4,5,21],skiprows=1, names=['Well index','E','N', 'Elevation [m]'])
                E = coordintates_dataset[coordintates_dataset['Well index']==well_index].iloc[0,1]
                N = coordintates_dataset[coordintates_dataset['Well index']==well_index].iloc[0,2]
                #Transform coordinates in the ETRS89 / UTM (EPSG: 25832) to WGS84
                transformer = Transformer.from_crs(25832,4326)
                (lat, lon) = transformer.transform(E,N)

                # calculate WTD from elevation if needed
                welldata["WTD [m]"] = (
                    welldata["WTD [m]"]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )
                elevation = coordintates_dataset[coordintates_dataset['Well index']==well_index].iloc[0,3]
                welldata['elevation'] = float(elevation.replace(',', '.'))
                welldata['WTD [m]'] = pd.to_numeric(welldata['WTD [m]'], errors='coerce')
                welldata['WTD [m]'] = welldata['elevation'] - welldata['WTD [m]']

                welldata['time'] = pd.to_datetime(welldata['time'],format=format)
                # check we have daily timestep data from 2017 to 2019
                # filter dates
                mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
                data = welldata.loc[mask]
                # check for NaN values
                data = data.dropna(axis=0, how='any', subset=['WTD [m]'])
                if data.empty:
                    print(f"No data for well {well_index} LON{lon}LAT{lat}, skipping")
                    continue
                data = data.dropna(subset=['WTD [m]'])
                if (len(data) == len(all_dates)):
                    df_eu[f"Germany(hessen)_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]'].to_list()
                    print(f"added well {well_index} LON{lon}LAT{lat}")
                elif len(data) > len(all_dates):
                    # check if more than one measurement per day then calculate daily average
                    data = data.set_index('time')
                    data = data.resample('D').mean()
                    if len(data) == len(all_dates):
                        df_eu[f"Germany(hessen)_LON{lon}LAT{lat}"] = data['WTD [m]'].to_list()
                        print(f"added well {well_index} LON{lon}LAT{lat}")
                else:
                    pass
                    print(len(data)," not enough data points for well ", well_index," expected ", len(all_dates))
        print(df_eu.columns)
        precols = df_eu.columns
        df_eu.reset_index(inplace=True, drop=True)
        df_eu.dropna(axis=1, inplace=True)
        postcols = df_eu.columns
        if len(precols) != len(postcols):
            print("Total wells in EU obs dataset after adding Hessen: ", precols)
            print("Total wells in EU obs dataset after dropping NaN columns: ", postcols)
            print("Dropped ", len(precols) - len(postcols), " columns with NaN values")
            diffcols = set(postcols) - set(precols)
            print("Dropped columns: ", diffcols)
            raise ValueError("Dropped columns with NaN values after adding Hessen wells, please check the data!")
        else:
            print("Total wells in EU obs dataset after adding Hessen: ", precols)
            print("No columns were dropped after adding Hessen wells.")
        # ready, save it
        df_eu.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
    
    def wtdobs_germany_Niedersachsen(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        df_eu.dropna(axis=1, inplace=True)
        df_eu.reset_index(inplace=True, drop=True)
        precols = len(df_eu.columns)
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')

        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_Germany", "obsWTD_point_measurements_in_Niedersachsen_NLWKH", "o.data", "WTD_measurements_Niedersachsen")
        format = '%Y-%m-%d'

        coordintates_dataset = pd.read_excel(src+'/table04082020122025329.xls',usecols=[0,11,12])

        subdirs = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
        for subdir in subdirs:
            src_subdir =  os.path.join(src, subdir)
            for file in [x for x in os.listdir(src_subdir) if x.endswith('.xls')]:
                welldata = pd.read_excel(os.path.join(src_subdir, file),usecols=[0,2,4],names=['Well_index','time','WTD [m]'])
                welldata.reset_index(drop=True, inplace=True)
                welldata['WTD [m]'] = welldata['WTD [m]'].astype(str).str.replace(",", ".", regex=False).astype(float)
                well_ID = welldata.iloc[0,0]
                E = coordintates_dataset[coordintates_dataset['Messstelle Nr.']==well_ID].iloc[0,1]
                N = coordintates_dataset[coordintates_dataset['Messstelle Nr.']==well_ID].iloc[0,2]
                #Transform coordinates in the ETRS89 / UTM zone 32N (zE-N) to WGS84
                transformer = Transformer.from_crs(4647,4326)
                (lat, lon) = transformer.transform(E,N)
                welldata['time'] = pd.to_datetime(welldata['time'],format=format)
                # check we have daily timestep data from 2017 to 2019
                # filter dates
                mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
                data = welldata.loc[mask]
                data = data.dropna(axis=1, how='all')
                if (len(data) == len(all_dates)):
                    df_eu[f"Germany(niedersachsen)_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]']
                    print(f"added well {well_ID} LON{lon}LAT{lat}")
                elif len(data) > len(all_dates):
                    # check if more than one measurement per day then calculate daily average
                    data = data.set_index('time')
                    data = data.resample('D').mean()
                    if len(data) == len(all_dates):
                        df_eu[f"Germany(niedersachsen)_LON{lon}LAT{lat}"] = data['WTD [m]']
                        print(f"added well {well_ID} LON{lon}LAT{lat}")
                else:
                    pass
                    print(len(data)," not enough data points for well ", well_ID," expected ", len(all_dates))
        print(df_eu)
        print(df_eu.columns)
        if len(df_eu.columns) == precols:
            print("No wells added for Germany Niedersachsen")
        else:
            raise ValueError("Wells were added for Germany Niedersachsen, please check the data!")
        # NOTE none found with daily data from 2017-2019
    
    def wtdobs_germany_Rheinland(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')
        precols = len(df_eu.columns)

        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_Germany", "obsWTD_point_measurements_in_Rheinland_Pfalz_GDA_Wasser_rlp", "o.data", "WTD_measurements_Rheinland_Pfalz")
        format = '%d.%m.%Y'

        for file in [x for x in os.listdir(src) if x.endswith('.xls')]:
            welldata = pd.read_excel(os.path.join(src, file),usecols=[0,2,5],names=['Well_index','time','WTD [m]'])
            welldata.drop(welldata[welldata['WTD [m]']=='-'].index,inplace=True)
            welldata.reset_index(drop=True, inplace=True)
            welldata['WTD [m]'] = welldata['WTD [m]'].astype(str).str.replace(",", ".", regex=False).astype(float)
            well_ID = welldata.iloc[0,0]
            
            coordintates_dataset = pd.read_csv(src+'/Coordinates.csv',encoding='iso-8859-1',sep=',')            
            E = coordintates_dataset[coordintates_dataset['Messstellennummer']==well_ID].iloc[0,1]
            N = coordintates_dataset[coordintates_dataset['Messstellennummer']==well_ID].iloc[0,2]
            #Transform coordinates in the ETRS89 / UTM (EPSG: 25832) to WGS84
            transformer = Transformer.from_crs(25832,4326)
            (lat, lon) = transformer.transform(E,N)

            welldata['time'] = pd.to_datetime(welldata['time'],format=format)
            # check we have daily timestep data from 2017 to 2019
            # filter dates
            mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
            data = welldata.loc[mask]
            data = data.dropna(axis=1, how='all')
            if (len(data) == len(all_dates)):
                df_eu[f"Germany(rheinland)_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]']
                print(f"added well {well_ID} LON{lon}LAT{lat}")
            elif len(data) > len(all_dates):
                # check if more than one measurement per day then calculate daily average
                data = data.set_index('time')
                data = data.resample('D').mean()
                if len(data) == len(all_dates):
                    df_eu[f"Germany(rheinland)_LON{lon}LAT{lat}"] = data['WTD [m]']
                    print(f"added well {well_ID} LON{lon}LAT{lat}")
            else:
                pass
                print(len(data)," not enough data points for well ", well_ID," expected ", len(all_dates))
        print(df_eu)
        print(df_eu.columns)
        if len(df_eu.columns) == precols:
            print("No wells added for Germany Rheinland")
        else:
            raise ValueError("Wells were added for Germany Rheinland, please check the data!")
        # NOTE none found with daily data from 2017-2019
    
    def wtdobs_germany_Anhalt(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        df_eu.dropna(axis=1, inplace=True)
        df_eu.reset_index(inplace=True, drop=True)
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')

        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_Germany", "obsWTD_point_measurements_in_Sachsen_Anhalt_GLD", "o.data", "WTD_measurements_Sachsen_Anhalt")
        format = '%d.%m.%Y'

        coordintates_dataset = pd.read_csv(os.path.join(src,'data.csv'),encoding='iso-8859-1',sep=';',usecols=[0,12,13],skiprows=1, names=['Well_index','E','N'])

        df1 = pd.read_csv(os.path.join(src,'Grundwassermessstellen_20200831_112408.csv'),encoding='iso-8859-1',sep=';',usecols=[1,2,4],skiprows=1, names=['Well_index','time','WTD [m]']) 
        df1['WTD [m]'] = df1['WTD [m]'] / 100
        df2 = pd.read_csv(os.path.join(src,'Grundwassermessstellen_20200831_112734.csv'),encoding='iso-8859-1',sep=';',usecols=[1,2,4],skiprows=1, names=['Well_index','time','WTD [m]']) 
        df2['WTD [m]'] = df2['WTD [m]'] / 100
        df3 = pd.read_csv(os.path.join(src,'Grundwassermessstellen_20200831_112915.csv'),encoding='iso-8859-1',sep=';',usecols=[1,2,4],skiprows=1, names=['Well_index','time','WTD [m]']) 
        df3['WTD [m]'] = df3['WTD [m]'] / 100
        df4 = pd.concat([df1,df2,df3])
        #Find unique well indexes
        well_indexes = pd.unique(df4['Well_index'])
            
        #Save separated file for each well to the destination folder
        for well_index in well_indexes:
            welldata = df4[df4['Well_index']==well_index]
            # welldata['WTD [m]'] = welldata['WTD [m]'].str.replace(',','.').astype('float64')
            welldata.reset_index(drop=True, inplace=True)
            
            E = coordintates_dataset[coordintates_dataset['Well_index']==str(well_index)].iloc[0,1]
            N = coordintates_dataset[coordintates_dataset['Well_index']==str(well_index)].iloc[0,2]            
            #Transform coordinates in the ETRS89 / UTM (EPSG: 25832) to WGS84
            transformer = Transformer.from_crs(25832,4326)
            (lat, lon) = transformer.transform(E,N)
            
            welldata['time'] = pd.to_datetime(welldata['time'],format=format)
            # check we have daily timestep data from 2017 to 2019
            # filter dates
            mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
            data = welldata.loc[mask]
            data = data.dropna(axis=1, how='all')
            if (len(data) == len(all_dates)):
                df_eu[f"Germany(anhalt)_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]'].to_list()
                print(f"added well {well_index} LON{lon}LAT{lat}")
            elif len(data) > len(all_dates):
                # check if more than one measurement per day then calculate daily average
                data = data.set_index('time')
                data = data.resample('D').mean()
                if len(data) == len(all_dates):
                    df_eu[f"Germany(anhalt)_LON{lon}LAT{lat}"] = data['WTD [m]'].to_list()
                    print(f"added well {well_index} LON{lon}LAT{lat}")
            else:
                pass
                print(len(data)," not enough data points for well ", well_index," expected ", len(all_dates))
        print(df_eu)
        print(df_eu.columns)
        precols = df_eu.columns
        df_eu.reset_index(inplace=True, drop=True)
        df_eu.dropna(axis=1, inplace=True)
        postcols = df_eu.columns
        if len(precols) != len(postcols):
            print("Total wells in EU obs dataset after adding Anhalt: ", precols)
            print("Total wells in EU obs dataset after dropping NaN columns: ", postcols)
            print("Dropped ", len(precols) - len(postcols), " columns with NaN values")
            diffcols = set(precols) - set(postcols)
            print("Dropped columns: ", diffcols)
            raise ValueError("Dropped columns with NaN values after adding Anhalt wells, please check the data!")
        else:
            print("Total wells in EU obs dataset after adding Anhalt: ", precols)
            print("No columns were dropped after adding Anhalt wells.")
        # found a lot, save them later
        df_eu.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))

    def wtdobs_germany_Sachsen(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        df_eu.dropna(axis=1, inplace=True)
        df_eu.reset_index(inplace=True, drop=True)
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')

        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_Germany", "obsWTD_point_measurements_in_Sachsen_iDA", "o.data", "WTD_measurements_Sachsen")
        format = '%Y-%m-%d'

        for file in [x for x in os.listdir(src) if x.endswith('.xlsx') and 'table06082020135204377.xlsx' not in x]:
            allwells = pd.read_excel(os.path.join(src, file),usecols=[0,2,3,4,7],names=['Well index','E','N','time','WTD [m]'])
            well_indexes = pd.unique(allwells['Well index'])
            for well_index in well_indexes:
                welldata = allwells[allwells['Well index']==well_index]
                welldata['WTD [m]'] = welldata['WTD [m]'].astype(str).str.replace(",", ".", regex=False).astype(float)
                welldata['WTD [m]'] = welldata['WTD [m]'] / 100.0
                welldata.reset_index(drop=True, inplace=True)

                E = welldata.iloc[0,1]
                N = welldata.iloc[0,2]
                #Transform coordinates in the ETRS89 / UTM 33 (EPSG: 25833) to WGS84
                transformer = Transformer.from_crs(25833,4326)
                (lat, lon) = transformer.transform(E,N)
                
                welldata['time'] = pd.to_datetime(welldata['time'],format=format)
                # check we have daily timestep data from 2017 to 2019
                # filter dates
                mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
                data = welldata.loc[mask]
                data = data.dropna(axis=1, how='all')
                if data.empty:
                    print(f"No data for well {well_index} LON{lon}LAT{lat}, skipping")
                    continue
                data = data.dropna(axis=0, how='any', subset=['WTD [m]'])
                data.reset_index(drop=True, inplace=True)
                data = data.dropna(subset=['WTD [m]'])
                data.reset_index(drop=True, inplace=True)
                if (len(data) == len(all_dates)):
                    df_eu[f"Germany(sachsen)_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]'].to_list()
                    print(f"added well {well_index} LON{lon}LAT{lat}")
                elif len(data) > len(all_dates):
                    # check if more than one measurement per day then calculate daily average
                    data = data.set_index('time')
                    data = data.resample('D').mean()
                    data.dropna(inplace=True, how='any', subset=['WTD [m]'])
                    data.reset_index(drop=True, inplace=True)
                    if len(data) == len(all_dates):
                        df_eu[f"Germany(sachsen)_LON{lon}LAT{lat}"] = data['WTD [m]'].to_list()
                        print(f"added well {well_index} LON{lon}LAT{lat}")
                else:
                    pass
                    print(len(data)," not enough data points for well ", well_index," expected ", len(all_dates))
        print(df_eu)
        print(df_eu.columns)
        precols = df_eu.columns
        df_eu.reset_index(inplace=True, drop=True)
        df_eu.dropna(axis=1, inplace=True)
        postcols = df_eu.columns
        if len(precols) != len(postcols):
            print("Total wells in EU obs dataset after adding Sachsen: ", precols)
            print("Total wells in EU obs dataset after dropping NaN columns: ", postcols)
            print("Dropped ", len(precols) - len(postcols), " columns with NaN values")
            diffcols = set(precols) - set(postcols)
            print("Dropped columns: ", diffcols)
            raise ValueError("Dropped columns with NaN values after adding Sachsen wells, please check the data!")
        else:
            print("Total wells in EU obs dataset after adding Sachsen: ", precols)
            print("No columns were dropped after adding Sachsen wells.")
        # found a lot, save them later
        df_eu.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))

    def wtdobs_germany_SchleswigHolstein(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        df_eu.dropna(axis=1, inplace=True)
        df_eu.reset_index(inplace=True, drop=True)
        all_dates = pd.date_range(start='2017-01-01', end='2019-12-31', freq='D')

        src = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_Germany", "obsWTD_point_measurements_in_Schleswig_Holstein_LUSH", "o.data", "WTD_measurements_Schleswig_Holstein")
        format = '%d.%m.%Y %H:%M:%S'

        coordintates_dataset = pd.read_csv(os.path.join(src,'Coordinates_SH.csv'),sep=',',skiprows=1,names=['Well_index','E','N'])

        for file in [x for x in os.listdir(src) if x.endswith('.csv') and 'Coordinates_SH.csv' not in x]:
            welldata = pd.read_csv(os.path.join(src, file),encoding='iso-8859-1',sep=';',skiprows=1,names=['time','WTD [m]'])
            welldata['WTD [m]'] = welldata['WTD [m]'].str.replace(',','.').astype('float64')
            well_index = int(file[-13:-4])
            welldata['Well_index'] = well_index

            welldata_well = welldata[welldata['Well_index']==well_index]
            welldata_well.reset_index(drop=True, inplace=True)

            E = coordintates_dataset[coordintates_dataset['Well_index']==well_index].iloc[0,1]
            N = coordintates_dataset[coordintates_dataset['Well_index']==well_index].iloc[0,2]
            #Transform coordinates in the ETRS89 / UTM (EPSG: 25832) to WGS84
            transformer = Transformer.from_crs(4647,4326)
            (lat, lon) = transformer.transform(E,N)
            
            welldata['time'] = pd.to_datetime(welldata['time'],format=format)
            # check we have daily timestep data from 2017 to 2019
            # filter dates
            mask = (welldata['time'] >= '2017-01-01') & (welldata['time'] < '2020-01-01')
            data = welldata.loc[mask]
            data = data.dropna(axis=1, how='all')
            data = data.dropna(axis=0, how='any', subset=['WTD [m]'])
            data.reset_index(drop=True, inplace=True)
            if data.empty:
                print(f"No data for well {well_index} LON{lon}LAT{lat}, skipping")
                continue
            data = data.dropna(subset=['WTD [m]'])
            data.reset_index(drop=True, inplace=True)
            if (len(data) == len(all_dates)):
                df_eu[f"Germany(schleswigholstein)_LON{lon}LAT{lat}"] = data.set_index('time')['WTD [m]'].to_list()
                print(f"added well {well_index} LON{lon}LAT{lat}")
            elif len(data) > len(all_dates):
                # check if more than one measurement per day then calculate daily average
                data = data.set_index('time')
                data = data.resample('D').mean()
                data.dropna(inplace=True, how='any', subset=['WTD [m]'])
                data.reset_index(drop=True, inplace=True)
                if len(data) == len(all_dates):
                    df_eu[f"Germany(schleswigholstein)_LON{lon}LAT{lat}"] = data['WTD [m]'].to_list()
                    print(f"added well {well_index} LON{lon}LAT{lat}")
            else:
                pass
                print(len(data)," not enough data points for well ", well_index," expected ", len(all_dates))
        print(df_eu)
        print(df_eu.columns)
        precols = df_eu.columns
        df_eu.reset_index(inplace=True, drop=True)
        df_eu.dropna(axis=1, inplace=True)
        postcols = df_eu.columns
        if len(precols) != len(postcols):
            print("Total wells in EU obs dataset after adding Schleswig-Holstein: ", precols)
            print("Total wells in EU obs dataset after dropping NaN columns: ", postcols)
            print("Dropped ", len(precols) - len(postcols), " columns with NaN values")
            diffcols = set(precols) - set(postcols)
            print("Dropped columns: ", diffcols)
            raise ValueError("Dropped columns with NaN values after adding Schleswig-Holstein wells, please check the data!")
        else:
            print("Total wells in EU obs dataset after adding Schleswig-Holstein: ", precols)
            print("No columns were dropped after adding Schleswig-Holstein wells.")
        # found a lot, save them later
        df_eu.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU.parquet"))
        
        ############################### the following functions are for a monthly timestep ###############################
    
    def read_csv_monthlyobs_from_yueling(self):
        # define dataframe of monthly timestep from 2000 to 2015
        obsdf = pd.DataFrame()
        all_dates = pd.date_range(start='2000-01-01', end='2015-12-31', freq='MS')
        obsdf['time'] = all_dates
        obsdf.set_index('time', inplace=True)

        dirobseu = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_EU")
        dirobsde = os.path.join(os.path.dirname(get_root_dir()), "obsWTD_point_measurements_in_Germany")
        
        # get eu obs
        for country in os.listdir(dirobseu):
            dircountry = os.path.join(dirobseu, country)
            print(country)
            if os.path.isdir(dircountry) and 'Denmark' not in country and 'Portugal' not in country: # no monthly data for these countries
                datapath = os.path.join(dircountry, "p.data.monthly_averaged_WTD_measured_data", os.listdir(os.path.join(dircountry, "p.data.monthly_averaged_WTD_measured_data"))[0], "2000_2015_monthly")
                # read txt as a dataframe
                well_coords_df = pd.read_csv(os.path.join(datapath, "well_coordinates.txt"), delim_whitespace=True)
                well_coords_df['well_index'] = well_coords_df['well_index'].astype(str)

                for file in os.listdir(datapath):
                    if file.endswith('.csv'):
                        welldata = pd.read_csv(os.path.join(datapath, file))
                        countryname = country.split('_')[-2]
                        wellindex = file.split('_')[-1].replace('.csv','')
                        colname = f"{countryname}_LON{well_coords_df[well_coords_df['well_index']==wellindex]['lon'].values[0]}LAT{well_coords_df[well_coords_df['well_index']==wellindex]['lat'].values[0]}"
                        wtddata = welldata['WTD [m]'].to_list()
                        obsdf[colname] = wtddata
                        print(f"added well {wellindex} for country {countryname}")
        for country in os.listdir(dirobsde):
            dircountry = os.path.join(dirobsde, country)
            print(country)
            # bayern and wuerttemberg are included when working with anomalies
            # if "Bayern" in country or "Wuerttemberg" in country:
            #     continue # should correct elevations manually
            if os.path.isdir(dircountry):
                datapath = os.path.join(dircountry, "p.data.monthly_averaged_WTD_measured_data", os.listdir(os.path.join(dircountry, "p.data.monthly_averaged_WTD_measured_data"))[0], "2000_2015_monthly")
                # read txt as a dataframe
                well_coords_df = pd.read_csv(os.path.join(datapath, "well_coordinates.txt"), delim_whitespace=True)
                well_coords_df['well_index'] = well_coords_df['well_index'].astype(str)

                for file in os.listdir(datapath):
                    if file.endswith('.csv'):
                        # Hessen, Sachsen_iDA and Rheinland_Pfalz are all included with corrected and wrong elevations when working with anomalies
                        # check modified date of elevation corrected files, should be after 2026-01-01:
                        if ("Hessen" in country or "Sachsen_iDA" in country or "Rheinland_Pfalz" in country) and os.path.getmtime(os.path.join(datapath, file)) < datetime(2026,1,1).timestamp():
                            continue
                        welldata = pd.read_csv(os.path.join(datapath, file))
                        wellindex = file.split('_')[-1].replace('.csv','')
                        colname = f"Germany_LON{well_coords_df[well_coords_df['well_index']==wellindex]['lon'].values[0]}LAT{well_coords_df[well_coords_df['well_index']==wellindex]['lat'].values[0]}"
                        wtddata = welldata['WTD [m]'].to_list()
                        obsdf[colname] = wtddata
                        print(f"added well {wellindex} for country Germany")
        print(obsdf)
        obsdf.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU_monthly_incorrect_elevations.parquet"), index=False)

    def find_mindistance_pointobs_to_grid(self, lon, lat, lons=None, lats=None):
        """
        Calculating minimum distance from a point with given lon lat to the nearest grid point on a 2D lon lat grid
        2D grid is loaded from TSMP lon2D.npy and lat2D.npy files
        Haversine formula is used to calculate the distance between two points on the Earth's surface given their
        longitude and latitude.
        This function is following the script from Bibi Naz found here:
        https://icg4geo.icg.kfa-juelich.de/SoftwareTools/postpro_clm_validation_with_stations_data/-/tree/master
        
        :param lon: float longitude of the observation point
        :param lat: float latitude of the observation point
        :return: yindex, xindex, tsmp_lon, tsmp_lat of the nearest grid point
        """

        # Load the TSMP grid unless it was supplied by a caller processing
        # many stations at once.
        if lons is None and lats is None:
            lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
            lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))
        elif lons is None or lats is None:
            raise ValueError("Both longitude and latitude grids must be provided together.")
        if lons.shape != lats.shape or lons.ndim != 2:
            raise ValueError("TSMP longitude and latitude grids must be matching 2D arrays.")

        # find nearest grid point
        # explain how it is done
        # calculate the distance between the observation location and all TSMP grid points
        # compute haversine distance (in kilometers) between obs point and all grid points
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        lons_rad = np.radians(lons)
        lats_rad = np.radians(lats)

        dlon = lons_rad - lon_rad
        dlat = lats_rad - lat_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat_rad) * np.cos(lats_rad) * np.sin(dlon / 2)**2
        # c = 2 * np.arcsin(np.sqrt(a))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6373.0  # Earth radius in kilometers
        dist = R * c
        
        yindex, xindex = np.unravel_index(np.argmin(dist), dist.shape)
        tsmp_lon = lons[yindex, xindex]
        tsmp_lat = lats[yindex, xindex]
        return yindex, xindex, tsmp_lon, tsmp_lat

    def maptoTSMP_averageduplicates(self):
        # keep all columns except well id
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU_monthly_incorrect_elevations.parquet"))
        locations = pd.DataFrame(columns=['country', 'lon', 'lat', 'tsmp_lon', 'tsmp_lat', 'tsmp_xindex', 'tsmp_yindex', 'target_chunk', 'sim_1darrayindex'])
        for column in df_eu.columns:
            country = column.split("_")[0]
            strlon = f'_{column.split("LON")[1].split("LAT")[0]}'
            strlat = f'_{column.split("LAT")[1]}'
            lon = float(column.split("LON")[1].split("LAT")[0])
            lat = float(column.split("LAT")[1])
            yindex, xindex, tsmp_lon, tsmp_lat = self.find_mindistance_pointobs_to_grid(lon, lat)
            # find which target chunk the observation point belongs to in the predicted dataset grid
            target, flat_idx = self.find_chunk_flatidx_from_2Dindices(yindex, xindex)
            if target is None:
                continue
            locations = pd.concat([locations, pd.DataFrame({'country': [country], 'lon': [strlon], 'lat': [strlat], 'tsmp_lon': [tsmp_lon], 'tsmp_lat': [tsmp_lat], 'tsmp_xindex': [xindex], 'tsmp_yindex': [yindex], 'target_chunk': [target], 'sim_1darrayindex': [flat_idx]})], ignore_index=True)
        print(locations)
        # remove columns in df_eu not found in TSMP dataset (not in locations)
        cols_to_keep = [f"{row['country']}_LON{row['lon'].replace('_', '')}LAT{row['lat'].replace('_', '')}" for index, row in locations.iterrows()]
        df_eu = df_eu[cols_to_keep]
        # locations.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_obs_TSMP.csv"), index=False)
        # locations = pd.read_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_obs_TSMP.csv"))
        duplicates, locations = self.check_duplicates(locations)
        locations.reset_index(inplace=True, drop=True)
        if duplicates is not None:
            print("Duplicates found in the mapping of observation points to TSMP grid points:")
            print(duplicates)
            dup_list = duplicates['dup'].unique().tolist()
            # iterate over each duplicate group
            for dup in dup_list:
                dup_subset = duplicates[duplicates['dup'] == dup]
                dup_subset.reset_index(inplace=True, drop=True)
                print(f"Duplicate group {dup} has {len(dup_subset)} entries")
                # create list of columns in df_eu corresponding to this duplicate group
                dupcols_list = [f"{row['country']}_LON{row['lon'].replace('_', '')}LAT{row['lat'].replace('_', '')}" for index, row in dup_subset.iterrows()]
                # average these columns in df_eu and save the result in the first column of the list
                df_eu[dupcols_list[0]] = df_eu[dupcols_list].mean(axis=1)
                # drop the other columns
                df_eu.drop(columns=dupcols_list[1:], inplace=True)
                df_eu.reset_index(inplace=True, drop=True)
                # drop duplicate from locations dataframe based on dup column of dup_subset except the first one
                # get indices of rows in locations mapping where dup column matches except the first one
                indices = locations[locations['dup']==dup].index.tolist()
                # drop these indices from locations dataframe and keep first
                locations.drop(indices[1:], inplace=True)
                locations.reset_index(inplace=True, drop=True)
            print("Duplicates have been averaged and removed.")

        locations.drop(columns=['dup'], inplace=True, errors='ignore')
        print("Final mapped dataframe columns:")
        print(df_eu.columns)
        print(df_eu)
        df_eu.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU_monthly_avgdup_incorrect_elevations.parquet"), index=False)
        print(locations)
        locations.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_wtdobsEU_TSMP_avgdup_incorrect_elevations.csv"), index=False)
    
    def tsmpindicesofallobs(self):
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU_monthly.parquet"))
        locations = pd.DataFrame(columns=['tsmp_xindex', 'tsmp_yindex'])
        for column in df_eu.columns:
            lon = float(column.split("LON")[1].split("LAT")[0])
            lat = float(column.split("LAT")[1])
            yindex, xindex, tsmp_lon, tsmp_lat = self.find_mindistance_pointobs_to_grid(lon, lat)
            locations = pd.concat([locations, pd.DataFrame([{
                'tsmp_xindex': xindex,
                'tsmp_yindex': yindex
            }])], ignore_index=True)
        print(locations)
        locations["xy"] = locations.apply(lambda row: f"{row['tsmp_xindex']}_{row['tsmp_yindex']}", axis=1)
        print(locations)
        locations.drop_duplicates(subset=['xy'], inplace=True)
        locations.reset_index(inplace=True, drop=True)
        print(locations)
        # locations.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "tsmp_indices_of_all_obs.csv"), index=False)
    
    def additional_obs(self):
        obsfilepath = os.path.join(os.path.dirname(get_root_dir()), "grid_measured_wtd_a_data_1996_2016_EUR11_YM_v_20210407.nc")
        with h5py.File(obsfilepath, "r") as f:
            obswtda = f["wtd_a"][()]
            h5py.File.close(f)
        print(obswtda.shape)
        indices = np.where(~np.isnan(np.nanmean(obswtda, axis=0)))
        print(obswtda[0, 351, 258])
        print(obswtda[0, 154, 161])

        # expand dimensions to match map2d shape
        output = np.full((252, 432, 444), np.nan, dtype=obswtda.dtype)
        # Compute offsets to center the data
        row_offset = (432 - 412) // 2   # 10
        col_offset = (444 - 424) // 2   # 10
        output[
            :,
            row_offset:row_offset+412,
            col_offset:col_offset+424
        ] = obswtda
        print(output.shape)
        np.save(os.path.join(INPUTPATH, "wtd_obsEU_YMA_1996_2016.npy"), output)
    
    def obs_lonlat_area(self):
        plots = plotting_helper()
        obsfilepath = os.path.join(os.path.dirname(get_root_dir()), "grid_measured_wtd_a_data_1996_2016_EUR11_YM_v_20210407.nc")
        with h5py.File(obsfilepath, "r") as f:
            lon = f["lon"][()]
            lat = f["lat"][()]
            obswtda = f["wtd_a"][()]
            h5py.File.close(f)
        print(lon.shape)
        print(lat.shape)
        # expand dimensions to match map2d shape
        output = np.full((252, 432, 444), np.nan, dtype=obswtda.dtype)
        # Compute offsets to center the data
        row_offset = (432 - 412) // 2   # 10
        col_offset = (444 - 424) // 2   # 10
        output[
            :,
            row_offset:row_offset+412,
            col_offset:col_offset+424
        ] = obswtda
        obswtda = np.mean(output, axis=0)

        plots.zoomed2Dmap_adaptedtoma(obswtda, False, np.min(obswtda), np.max(obswtda), "wtda obs")

    @staticmethod
    def _england_station_id(value):
        numeric_id = pd.to_numeric(value, errors="raise")
        if not float(numeric_id).is_integer():
            raise ValueError(f"Expected an integer England station ID, got {value!r}")
        return str(int(numeric_id))

    def _read_england_station_locations(self, shapefile_path):
        try:
            import shapefile
        except ImportError as exc:
            raise ImportError(
                "Reading the England station shapefile requires the 'pyshp' package."
            ) from exc

        reader = shapefile.Reader(shapefile_path, encoding="utf-8")
        try:
            field_names = [field[0] for field in reader.fields[1:]]
            required_fields = {"No", "Lat", "Long"}
            missing_fields = required_fields.difference(field_names)
            if missing_fields:
                raise ValueError(
                    f"England station shapefile is missing fields: {sorted(missing_fields)}"
                )
            locations = pd.DataFrame(reader.records(), columns=field_names)
        finally:
            reader.close()

        locations["station_id"] = locations["No"].map(self._england_station_id)
        locations["lat"] = pd.to_numeric(locations["Lat"], errors="coerce")
        locations["lon"] = pd.to_numeric(locations["Long"], errors="coerce")
        locations = locations.dropna(subset=["station_id", "lat", "lon"]).copy()
        if locations["station_id"].duplicated().any():
            duplicates = locations.loc[
                locations["station_id"].duplicated(keep=False), "station_id"
            ].unique()
            raise ValueError(
                f"Station IDs are not unique in the England shapefile: {duplicates.tolist()}"
            )
        if (
            ~locations["lon"].between(-180, 180)
            | ~locations["lat"].between(-90, 90)
        ).any():
            raise ValueError("The England shapefile contains coordinates outside WGS84 bounds.")
        return locations.set_index("station_id", drop=False)

    def _read_continuous_england_timeseries(self, timeseries_dir, locations):
        expected_months = pd.period_range("1996-01", "2016-12", freq="M")
        series_by_station = []
        station_metadata = []
        seen_ids = set()
        csv_files = sorted(
            filename
            for filename in os.listdir(timeseries_dir)
            if filename.endswith("_timeseries.csv")
        )
        if not csv_files:
            raise FileNotFoundError(
                f"No '*_timeseries.csv' files were found in {timeseries_dir}"
            )

        for filename in csv_files:
            station_id = self._england_station_id(
                filename[: -len("_timeseries.csv")]
            )
            if station_id in seen_ids:
                raise ValueError(f"Duplicate timeseries files found for station {station_id}.")
            seen_ids.add(station_id)
            if station_id not in locations.index:
                continue

            data = pd.read_csv(os.path.join(timeseries_dir, filename))
            if not {"date", "GD"}.issubset(data.columns):
                raise ValueError(
                    f"{filename} must contain 'date' and 'GD' columns; "
                    f"found {data.columns.tolist()}"
                )
            dates = pd.to_datetime(data["date"], errors="coerce")
            values = pd.to_numeric(data["GD"], errors="coerce")
            valid = (
                dates.notna()
                & values.notna()
                & dates.ge(pd.Timestamp("1996-01-01"))
                & dates.lt(pd.Timestamp("2017-01-01"))
                & np.isfinite(values)
            )
            data = pd.DataFrame(
                {"date": dates[valid], "value": values[valid]}
            ).sort_values("date")
            covered_months = pd.PeriodIndex(data["date"], freq="M").unique()
            if not expected_months.isin(covered_months).all():
                continue

            station = locations.loc[station_id]
            station_column = (
                f"England_station{station_id}_"
                f"LON{station['lon']:.6f}LAT{station['lat']:.6f}"
            )
            series_by_station.append(
                pd.Series(
                    data["value"].to_numpy(),
                    index=pd.DatetimeIndex(data["date"]),
                    name=station_column,
                )
            )
            station_metadata.append(
                {
                    "station_id": station_id,
                    "station_column": station_column,
                    "lon": float(station["lon"]),
                    "lat": float(station["lat"]),
                }
            )

        if not series_by_station:
            raise ValueError(
                "No England stations have at least one valid reading in every "
                "calendar month from 1996-01 through 2016-12."
            )
        observations = pd.concat(series_by_station, axis=1).sort_index()
        observations.index.name = "date"
        return observations, station_metadata

    def plot_england_station_locations(self, mapping, output_path):
        import cartopy.crs as ccrs
        from cartopy.feature import ShapelyFeature
        from cartopy.io.shapereader import Reader
        import matplotlib.pyplot as plt

        if mapping.empty:
            raise ValueError("Cannot plot England station locations: mapping is empty.")
        required_columns = {
            "lon",
            "lat",
            "tsmp_lon",
            "tsmp_lat",
            "tsmp_xindex",
            "tsmp_yindex",
        }
        missing_columns = required_columns.difference(mapping.columns)
        if missing_columns:
            raise ValueError(
                f"England station mapping is missing columns: {sorted(missing_columns)}"
            )

        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))
        if lons.ndim != 2 or lons.shape != lats.shape:
            raise ValueError("TSMP longitude and latitude grids must be matching 2D arrays.")
        xindices = pd.to_numeric(
            mapping["tsmp_xindex"], errors="raise"
        ).to_numpy(dtype=float)
        yindices = pd.to_numeric(
            mapping["tsmp_yindex"], errors="raise"
        ).to_numpy(dtype=float)
        if (
            not np.isfinite(xindices).all()
            or not np.isfinite(yindices).all()
            or not np.equal(xindices, np.floor(xindices)).all()
            or not np.equal(yindices, np.floor(yindices)).all()
        ):
            raise ValueError("England mapping contains invalid TSMP grid indices.")
        xindices = xindices.astype(int)
        yindices = yindices.astype(int)
        if (
            (xindices < 0).any()
            or (xindices >= lons.shape[1]).any()
            or (yindices < 0).any()
            or (yindices >= lons.shape[0]).any()
        ):
            raise ValueError("England mapping contains TSMP indices outside the grid.")

        cell_counts = np.zeros(lons.shape, dtype=np.int32)
        np.add.at(cell_counts, (yindices, xindices), 1)
        plotted_cells = np.ma.masked_where(cell_counts == 0, cell_counts)
        max_count = int(cell_counts.max())

        countries_path = os.path.join(
            os.path.dirname(get_root_dir()),
            "ne_10m_admin_0_countries",
            "ne_10m_admin_0_countries.shp",
        )
        if not os.path.isfile(countries_path):
            raise FileNotFoundError(
                f"Country boundaries shapefile not found: {countries_path}"
            )
        country_boundaries = ShapelyFeature(
            Reader(countries_path).geometries(),
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        grid_cells = mapping.drop_duplicates(
            subset=["tsmp_yindex", "tsmp_xindex"]
        )
        projection = ccrs.LambertAzimuthalEqualArea(
            central_longitude=-2,
            central_latitude=54,
        )
        figure, axis = plt.subplots(
            figsize=(13, 9),
            subplot_kw={"projection": projection},
        )
        mesh = axis.pcolormesh(
            lons,
            lats,
            plotted_cells,
            cmap="YlOrRd",
            vmin=0.5,
            vmax=max_count + 0.5,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        colorbar = figure.colorbar(mesh, ax=axis, shrink=0.75, pad=0.04)
        colorbar.set_label("Stations mapped to TSMP cell")
        axis.add_feature(country_boundaries, linewidth=0.6, zorder=2)
        axis.scatter(
            mapping["lon"],
            mapping["lat"],
            s=12,
            alpha=0.65,
            label="Stations",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        axis.scatter(
            grid_cells["tsmp_lon"],
            grid_cells["tsmp_lat"],
            s=20,
            marker="x",
            label="Mapped TSMP cells",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        lon_buffer = max(
            0.25,
            0.05 * (mapping["lon"].max() - mapping["lon"].min()),
        )
        lat_buffer = max(
            0.25,
            0.05 * (mapping["lat"].max() - mapping["lat"].min()),
        )
        lon_min = min(-7.5, mapping["lon"].min() - lon_buffer)
        lon_max = max(2.5, mapping["lon"].max() + lon_buffer)
        lat_min = min(49.0, mapping["lat"].min() - lat_buffer)
        lat_max = max(57.5, mapping["lat"].max() + lat_buffer)
        axis.set_extent(
            [
                lon_min,
                lon_max,
                lat_min,
                lat_max,
            ],
            crs=ccrs.PlateCarree(),
        )
        axis.set_title("England groundwater observation stations and TSMP cells")
        axis.gridlines(draw_labels=True, linestyle=":", alpha=0.5)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        return output_path

    def _merge_england_observations_into_netcdf(
        self, monthly_observations, mapping, source_path, output_path
    ):
        import netCDF4

        if os.path.abspath(source_path) == os.path.abspath(output_path):
            raise ValueError("The England merge output must not overwrite its source NetCDF.")
        tsmp_lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        tsmp_lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))
        if tsmp_lons.shape != tsmp_lats.shape or tsmp_lons.ndim != 2:
            raise ValueError("TSMP longitude and latitude grids must be matching 2D arrays.")

        with netCDF4.Dataset(source_path, "r") as source:
            if "wtd_a" not in source.variables:
                raise KeyError(f"The source NetCDF does not contain a 'wtd_a' variable: {source_path}")
            source_wtd = source.variables["wtd_a"]
            if len(source_wtd.dimensions) != 3 or source_wtd.dimensions[0] not in source.dimensions:
                raise ValueError("'wtd_a' must have dimensions ordered as time, y, x.")
            if source_wtd.dimensions[0].lower().find("time") < 0:
                raise ValueError("'wtd_a' must have its time dimension first.")
            if source_wtd.shape[0] != len(monthly_observations):
                raise ValueError(
                    f"Expected {len(monthly_observations)} monthly NetCDF records, "
                    f"found {source_wtd.shape[0]}."
                )
            if (
                source_wtd.shape[1] > tsmp_lons.shape[0]
                or source_wtd.shape[2] > tsmp_lons.shape[1]
            ):
                raise ValueError(
                    f"NetCDF spatial shape {source_wtd.shape[1:]} exceeds "
                    f"the TSMP grid shape {tsmp_lons.shape}."
                )

            time_name = source_wtd.dimensions[0]
            if time_name not in source.variables:
                raise KeyError(f"The NetCDF time coordinate '{time_name}' is missing.")
            time_variable = source.variables[time_name]
            if "units" not in time_variable.ncattrs():
                raise ValueError(f"NetCDF time coordinate '{time_name}' has no units attribute.")
            decoded_time = netCDF4.num2date(
                time_variable[:],
                units=time_variable.units,
                calendar=getattr(time_variable, "calendar", "standard"),
                only_use_cftime_datetimes=False,
            )
            source_months = [(date.year, date.month) for date in decoded_time]
            expected_months = [
                (date.year, date.month) for date in monthly_observations.index
            ]
            if source_months != expected_months:
                raise ValueError(
                    "NetCDF time coordinate does not match the England monthly period "
                    "1996-01 through 2016-12."
                )
            if not np.issubdtype(source_wtd.dtype, np.floating):
                raise TypeError("'wtd_a' must use a floating-point datatype to merge observations.")

            row_offset = (tsmp_lons.shape[0] - source_wtd.shape[1]) // 2
            col_offset = (tsmp_lons.shape[1] - source_wtd.shape[2]) // 2
            cells = []
            for _, cell in mapping.drop_duplicates(
                subset=["tsmp_yindex", "tsmp_xindex"]
            ).iterrows():
                source_y = int(cell["tsmp_yindex"]) - row_offset
                source_x = int(cell["tsmp_xindex"]) - col_offset
                if not (
                    0 <= source_y < source_wtd.shape[1]
                    and 0 <= source_x < source_wtd.shape[2]
                ):
                    raise ValueError(
                        f"Mapped TSMP cell ({cell['tsmp_yindex']}, {cell['tsmp_xindex']}) "
                        "falls outside the source NetCDF grid."
                    )
                cells.append((source_y, source_x, cell["grid_column"]))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with netCDF4.Dataset(output_path, "w", format=source.data_model) as output:
                for attribute in source.ncattrs():
                    output.setncattr(attribute, source.getncattr(attribute))
                for name, dimension in source.dimensions.items():
                    output.createDimension(
                        name, None if dimension.isunlimited() else len(dimension)
                    )
                for name, variable in source.variables.items():
                    creation_options = {}
                    if "_FillValue" in variable.ncattrs():
                        creation_options["fill_value"] = variable.getncattr("_FillValue")
                    copied_variable = output.createVariable(
                        name,
                        variable.datatype,
                        variable.dimensions,
                        **creation_options,
                    )
                    for attribute in variable.ncattrs():
                        if attribute != "_FillValue":
                            copied_variable.setncattr(
                                attribute, variable.getncattr(attribute)
                            )

                    if name != "wtd_a":
                        copied_variable[...] = variable[...]
                        continue

                    for month_index in range(source_wtd.shape[0]):
                        field = np.ma.asarray(variable[month_index, :, :])
                        field_values = np.asarray(
                            field.filled(np.nan), dtype=np.float64
                        )
                        for source_y, source_x, grid_column in cells:
                            value = monthly_observations.iloc[month_index][grid_column]
                            if (
                                np.isfinite(value)
                                and not np.isfinite(field_values[source_y, source_x])
                            ):
                                field_values[source_y, source_x] = value
                        copied_variable[month_index, :, :] = np.ma.masked_invalid(
                            field_values
                        )
        return output_path

    def process_england_observations(self, source_root=None, output_dir=None):
        """Process complete England station records and create mapped monthly products."""
        project_root = get_root_dir()
        if source_root is None:
            source_root = os.path.join(
                os.path.dirname(project_root),
                "Groundwater_Depth_Time_Series_for_2902_wells_across_England",
            )
        if output_dir is None:
            output_dir = os.path.join(
                INPUTPATH, "localobservations", "England"
            )
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        shapefile_path = os.path.join(source_root, "Stations location.shp")
        timeseries_dir = os.path.join(source_root, "All_stations_timeseries")
        locations = self._read_england_station_locations(shapefile_path)
        observations, station_metadata = self._read_continuous_england_timeseries(
            timeseries_dir, locations
        )
        raw_path = os.path.join(
            output_dir, "wtd_obsEngland_continuous_1996_2016.parquet"
        )
        observations.reset_index().to_parquet(raw_path, index=False)
        print(f"Saved continuous England observations to {raw_path}")
        mapping_rows = []
        for station in station_metadata:
            yindex, xindex, tsmp_lon, tsmp_lat = (
                self.find_mindistance_pointobs_to_grid(
                    station["lon"], station["lat"]
                )
            )
            mapping_rows.append(
                {
                    **station,
                    "tsmp_xindex": int(xindex),
                    "tsmp_yindex": int(yindex),
                    "tsmp_lon": float(tsmp_lon),
                    "tsmp_lat": float(tsmp_lat),
                }
            )
        mapping = pd.DataFrame(mapping_rows)
        monthly_index = pd.date_range("1996-01-01", "2016-12-01", freq="MS")
        station_monthly = (
            observations.resample("MS")
            .mean()
            .reindex(monthly_index)
        )
        mapping["grid_column"] = ""
        monthly_grid = pd.DataFrame(index=monthly_index)
        print("Aggregating England station observations to TSMP grid cells...")
        for (yindex, xindex), row_indices in mapping.groupby(
            ["tsmp_yindex", "tsmp_xindex"], sort=True
        ).groups.items():
            cell_rows = mapping.loc[row_indices]
            first = cell_rows.iloc[0]
            grid_column = (
                f"England_LON{first['tsmp_lon']}LAT{first['tsmp_lat']}"
            )
            if grid_column in monthly_grid.columns:
                raise ValueError(
                    f"Multiple TSMP cells produce the same output column {grid_column!r}."
                )
            mapping.loc[row_indices, "grid_column"] = grid_column
            monthly_grid[grid_column] = station_monthly[
                cell_rows["station_column"].tolist()
            ].mean(axis=1)
        if not monthly_grid.notna().to_numpy().any():
            raise ValueError("No monthly England values were available after grid aggregation.")
        print("Finished aggregating England station observations to TSMP grid cells.")
        mapping_path = os.path.join(
            output_dir, "mapping_wtdobsEngland_TSMP.csv"
        )
        mapping.to_csv(mapping_path, index=False)
        monthly_path = os.path.join(
            output_dir, "wtd_obsEngland_monthly_avgdup.parquet"
        )
        print(f"Saving monthly England observations to {monthly_path}")
        monthly_grid.rename_axis("date").reset_index().to_parquet(
            monthly_path, index=False
        )

        climatology_mean = monthly_grid.groupby(
            monthly_grid.index.month
        ).transform("mean")
        climatology_std = monthly_grid.groupby(
            monthly_grid.index.month
        ).transform(lambda values: values.std(ddof=0))
        anomalies = (monthly_grid - climatology_mean).div(
            climatology_std.where(climatology_std.ne(0))
        )
        anomaly_path = os.path.join(
            output_dir, "wtd_obsEngland_monthly_anomalies.parquet"
        )
        print(f"Saving monthly England anomalies to {anomaly_path}")
        anomalies.rename_axis("date").reset_index().to_parquet(
            anomaly_path, index=False
        )

        plot_path = os.path.join(
            output_dir, "wtd_obsEngland_station_locations.png"
        )
        self.plot_england_station_locations(mapping, plot_path)

        source_netcdf = os.path.join(
            os.path.dirname(project_root),
            "grid_measured_wtd_a_data_1996_2016_EUR11_YM_v_20210407.nc",
        )
        output_netcdf = os.path.join(
            output_dir,
            "grid_measured_wtd_a_data_1996_2016_EUR11_YM_v_20210407_with_England.nc",
        )
        self._merge_england_observations_into_netcdf(
            monthly_grid, mapping, source_netcdf, output_netcdf
        )
        timeseries_plot_path = self.plot_timeseries_englandobs(
            output_dir=output_dir
        )
        return {
            "continuous_observations": raw_path,
            "mapping": mapping_path,
            "monthly_observations": monthly_path,
            "monthly_anomalies": anomaly_path,
            "station_plot": plot_path,
            "merged_netcdf": output_netcdf,
            "anomaly_timeseries_plot": timeseries_plot_path,
            "station_count": len(station_metadata),
            "grid_cell_count": len(monthly_grid.columns),
        }

    def plot_timeseries_englandobs(self, output_dir=None, output_path=None):
        """Plot England monthly standardized anomalies, one panel per TSMP cell."""
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import netCDF4

        if output_dir is None:
            output_dir = os.path.join(
                INPUTPATH, "localobservations", "England"
            )
        output_dir = os.path.abspath(output_dir)
        if output_path is None:
            output_path = os.path.join(
                output_dir,
                "wtd_obsEngland_monthly_anomalies_timeseries.png",
            )

        mapping_path = os.path.join(
            output_dir, "mapping_wtdobsEngland_TSMP.csv"
        )
        anomalies_path = os.path.join(
            output_dir, "wtd_obsEngland_monthly_anomalies.parquet"
        )
        merged_netcdf_path = os.path.join(
            output_dir,
            "grid_measured_wtd_a_data_1996_2016_EUR11_YM_v_20210407_with_England.nc",
        )

        mapping = pd.read_csv(mapping_path)
        required_mapping_columns = {
            "grid_column",
            "tsmp_xindex",
            "tsmp_yindex",
        }
        missing_mapping_columns = required_mapping_columns.difference(mapping.columns)
        if missing_mapping_columns:
            raise ValueError(
                f"England mapping is missing columns: {sorted(missing_mapping_columns)}"
            )
        grid_columns = mapping["grid_column"].dropna().astype(str).unique().tolist()
        if not grid_columns:
            raise ValueError("The England mapping contains no TSMP grid cells.")

        anomalies = pd.read_parquet(anomalies_path)
        if "date" not in anomalies.columns:
            raise ValueError(
                f"England anomaly data must contain a 'date' column: {anomalies_path}"
            )
        missing_grid_columns = sorted(set(grid_columns).difference(anomalies.columns))
        if missing_grid_columns:
            raise ValueError(
                "England anomaly data is missing mapped grid-cell columns: "
                f"{missing_grid_columns}"
            )
        anomalies["date"] = pd.to_datetime(anomalies["date"], errors="raise")
        anomalies = anomalies.sort_values("date").reset_index(drop=True)
        if anomalies.empty or anomalies["date"].duplicated().any():
            raise ValueError("England anomaly data must have unique, non-empty dates.")

        with netCDF4.Dataset(merged_netcdf_path, "r") as merged:
            if "wtd_a" not in merged.variables:
                raise KeyError(
                    f"Merged NetCDF does not contain a 'wtd_a' variable: "
                    f"{merged_netcdf_path}"
                )
            wtd_variable = merged.variables["wtd_a"]
            if len(wtd_variable.dimensions) != 3:
                raise ValueError(
                    "Merged NetCDF 'wtd_a' must have (time, y, x) dimensions."
                )
            time_name = wtd_variable.dimensions[0]
            if time_name not in merged.variables:
                raise KeyError(
                    f"Merged NetCDF time coordinate '{time_name}' is missing."
                )
            time_variable = merged.variables[time_name]
            if "units" not in time_variable.ncattrs():
                raise ValueError(
                    f"Merged NetCDF time coordinate '{time_name}' has no units."
                )
            netcdf_dates = netCDF4.num2date(
                time_variable[:],
                units=time_variable.units,
                calendar=getattr(time_variable, "calendar", "standard"),
                only_use_cftime_datetimes=False,
            )
            netcdf_months = [(date.year, date.month) for date in netcdf_dates]
            anomaly_months = [
                (date.year, date.month) for date in anomalies["date"]
            ]
            if netcdf_months != anomaly_months:
                raise ValueError(
                    "Merged NetCDF time coordinate does not match the England "
                    "monthly anomaly dates."
                )

            tsmp_lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
            tsmp_lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))
            if tsmp_lons.shape != tsmp_lats.shape or tsmp_lons.ndim != 2:
                raise ValueError(
                    "TSMP longitude and latitude grids must be matching 2D arrays."
                )
            if (
                wtd_variable.shape[1] > tsmp_lons.shape[0]
                or wtd_variable.shape[2] > tsmp_lons.shape[1]
            ):
                raise ValueError(
                    "Merged NetCDF spatial dimensions exceed the TSMP grid."
                )
            row_offset = (tsmp_lons.shape[0] - wtd_variable.shape[1]) // 2
            col_offset = (tsmp_lons.shape[1] - wtd_variable.shape[2]) // 2
            for _, cell in mapping.drop_duplicates(
                subset=["tsmp_yindex", "tsmp_xindex"]
            ).iterrows():
                netcdf_y = int(cell["tsmp_yindex"]) - row_offset
                netcdf_x = int(cell["tsmp_xindex"]) - col_offset
                if not (
                    0 <= netcdf_y < wtd_variable.shape[1]
                    and 0 <= netcdf_x < wtd_variable.shape[2]
                ):
                    raise ValueError(
                        f"Mapped TSMP cell ({cell['tsmp_yindex']}, "
                        f"{cell['tsmp_xindex']}) is outside the merged NetCDF grid."
                    )

        column_count = min(3, len(grid_columns))
        row_count = (len(grid_columns) + column_count - 1) // column_count
        figure, axes = plt.subplots(
            row_count,
            column_count,
            figsize=(6 * column_count, 3.5 * row_count),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        for axis, grid_column in zip(axes.flat, grid_columns):
            values = pd.to_numeric(anomalies[grid_column], errors="coerce")
            axis.plot(anomalies["date"], values, linewidth=0.9)
            axis.axhline(0, color="black", linewidth=0.7, alpha=0.65)
            axis.set_title(grid_column.replace("England_", ""), fontsize=9)
            axis.set_ylabel("Standardized anomaly")
            axis.grid(True, linestyle=":", alpha=0.5)
            axis.xaxis.set_major_locator(mdates.YearLocator(5))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for axis in axes.flat[len(grid_columns):]:
            axis.set_visible(False)
        figure.suptitle("England groundwater monthly anomalies by TSMP cell")
        for axis in axes[-1, :]:
            if axis.get_visible():
                axis.set_xlabel("Year")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        figure.savefig(output_path, dpi=200)
        plt.close(figure)
        return output_path

    @staticmethod
    def find_longest_monthly_anomaly_period(
        station_dirs,
        min_years=4,
        date_column=None,
        anomaly_column="wtd_a",
    ):
        """Find the longest gap-free period with maximum station coverage.

        Each CSV in ``station_dirs`` is treated as one station and must contain
        monthly records, a date/month column, and an anomaly column. The
        station count is maximized over all periods at least ``min_years``
        long; ties are resolved by choosing the longest period, then the
        earliest start. A month is covered if it has at least one finite
        anomaly value.

        Returns the inclusive start and end months, period length, station
        count, and station IDs for the selected period.
        """
        # TODO this function is currently not working, analysis was done with copilot, but the concept in this function is the same
        if (
            isinstance(min_years, bool)
            or not isinstance(min_years, (int, np.integer))
            or min_years < 1
        ):
            raise ValueError("min_years must be a positive integer.")
        if not anomaly_column:
            raise ValueError("anomaly_column must be a non-empty column name.")

        if isinstance(station_dirs, (str, os.PathLike)):
            station_dirs = [station_dirs]
        else:
            station_dirs = list(station_dirs)
        if not station_dirs:
            raise ValueError("At least one station directory must be provided.")

        csv_files = []
        seen_directories = set()
        for directory in station_dirs:
            directory = os.path.abspath(os.fspath(directory))
            if not os.path.isdir(directory):
                raise NotADirectoryError(f"Station directory does not exist: {directory}")
            if directory in seen_directories:
                continue
            seen_directories.add(directory)
            csv_files.extend(
                os.path.join(directory, filename)
                for filename in sorted(os.listdir(directory))
                if filename.lower().endswith(".csv")
                and not filename.startswith("_")
                and os.path.isfile(os.path.join(directory, filename))
            )
        if not csv_files:
            raise FileNotFoundError("No station CSV files were found in the supplied directories.")

        min_months = min_years * 12
        runs_by_station = {}
        events = {}
        for csv_file in csv_files:
            station_id = os.path.splitext(os.path.basename(csv_file))[0]
            if station_id in runs_by_station:
                raise ValueError(
                    f"Station ID {station_id!r} occurs in more than one input directory."
                )

            columns = pd.read_csv(csv_file, nrows=0).columns
            if date_column is None:
                selected_date_column = next(
                    (column for column in ("month", "date") if column in columns),
                    None,
                )
            else:
                selected_date_column = date_column
            missing_columns = [
                column
                for column in (selected_date_column, anomaly_column)
                if column is None or column not in columns
            ]
            if missing_columns:
                raise ValueError(
                    f"{csv_file} is missing required columns: {missing_columns}"
                )

            data = pd.read_csv(
                csv_file,
                usecols=[selected_date_column, anomaly_column],
            )
            months = pd.to_datetime(
                data[selected_date_column], errors="coerce"
            ).dt.to_period("M")
            anomalies = pd.to_numeric(data[anomaly_column], errors="coerce")
            valid_rows = months.notna() & np.isfinite(anomalies)
            valid_months = sorted(
                set(months[valid_rows].astype("int64").tolist())
            )

            station_runs = []
            if valid_months:
                run_start = previous_month = valid_months[0]
                for month in valid_months[1:]:
                    if month != previous_month + 1:
                        station_runs.append((run_start, previous_month))
                        run_start = month
                    previous_month = month
                station_runs.append((run_start, previous_month))
            runs_by_station[station_id] = station_runs

            for run_start, run_end in station_runs:
                if run_end - run_start + 1 < min_months:
                    continue
                events.setdefault(run_start, []).append((1, run_end))
                events.setdefault(run_end - min_months + 2, []).append(
                    (-1, run_end)
                )

        if not events:
            raise ValueError(
                f"No station has {min_years} consecutive years of valid anomalies."
            )

        active_ends = {}
        max_station_count = 0
        longest_month_count = 0
        best_start = None
        best_end = None
        first_month = min(events)
        last_start = max(events) - 1
        for start_month in range(first_month, last_start + 1):
            for change, run_end in events.get(start_month, ()):
                count = active_ends.get(run_end, 0) + change
                if count:
                    active_ends[run_end] = count
                else:
                    del active_ends[run_end]

            station_count = sum(active_ends.values())
            if station_count < max_station_count:
                continue

            if station_count > max_station_count:
                max_station_count = station_count
                longest_month_count = 0

            end_month = min(active_ends)
            month_count = end_month - start_month + 1
            if month_count > longest_month_count:
                longest_month_count = month_count
                best_start = start_month
                best_end = end_month

        selected_stations = sorted(
            station_id
            for station_id, station_runs in runs_by_station.items()
            if any(
                run_start <= best_start and run_end >= best_end
                for run_start, run_end in station_runs
            )
        )
        output = {
            "start_month": pd.Period(ordinal=best_start, freq="M"),
            "end_month": pd.Period(ordinal=best_end, freq="M"),
            "n_months": longest_month_count,
            "station_count": max_station_count,
            "station_ids": selected_stations,
        }
        print(output)
        return output

    def _read_continuous_station_anomalies(
        self, source_dirs, expected_months, grid_lons, grid_lats
    ):
        """Read complete station anomalies and aggregate them by TSMP cell."""
        cell_station_values = {}
        mapping_rows = []
        source_counts = {}

        for source_name, source_dir in source_dirs.items():
            if not os.path.isdir(source_dir):
                raise NotADirectoryError(
                    f"{source_name} station directory does not exist: {source_dir}"
                )
            station_files = sorted(
                os.path.join(source_dir, filename)
                for filename in os.listdir(source_dir)
                if filename.lower().endswith(".csv")
                and not filename.startswith("_")
                and os.path.isfile(os.path.join(source_dir, filename))
            )
            if not station_files:
                raise FileNotFoundError(
                    f"No station CSV files were found for {source_name}: {source_dir}"
                )

            source_counts[source_name] = {
                "files": len(station_files),
                "complete_stations": 0,
                "incomplete_stations": 0,
            }
            for station_file in station_files:
                columns = pd.read_csv(station_file, nrows=0).columns
                date_column = next(
                    (column for column in ("month", "date") if column in columns),
                    None,
                )
                required = {"lat", "lon", "wtd_a"}
                missing = required.difference(columns)
                if date_column is None:
                    missing.add("month or date")
                if missing:
                    raise ValueError(
                        f"{station_file} is missing required columns: {sorted(missing)}"
                    )

                data = pd.read_csv(
                    station_file,
                    usecols=[date_column, "lat", "lon", "wtd_a"],
                )
                months = pd.to_datetime(
                    data[date_column], errors="coerce"
                ).dt.to_period("M")
                anomaly = pd.to_numeric(data["wtd_a"], errors="coerce")
                anomaly = anomaly.where(np.isfinite(anomaly))
                in_period = months.isin(expected_months)
                monthly_values = pd.Series(
                    anomaly[in_period].to_numpy(dtype=float),
                    index=months[in_period],
                )
                monthly_values = (
                    monthly_values.groupby(level=0)
                    .mean()
                    .reindex(expected_months)
                )
                if not monthly_values.notna().all():
                    source_counts[source_name]["incomplete_stations"] += 1
                    continue

                coordinates = data.loc[in_period, ["lat", "lon"]].apply(
                    pd.to_numeric, errors="coerce"
                )
                coordinates = coordinates[
                    np.isfinite(coordinates["lat"])
                    & np.isfinite(coordinates["lon"])
                ].drop_duplicates()
                if len(coordinates) != 1:
                    raise ValueError(
                        f"{station_file} must have one stable, valid latitude/"
                        f"longitude pair during {expected_months[0]} to "
                        f"{expected_months[-1]}."
                    )
                lat = float(coordinates.iloc[0]["lat"])
                lon = float(coordinates.iloc[0]["lon"])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError(
                        f"{station_file} has coordinates outside WGS84 bounds: "
                        f"latitude={lat}, longitude={lon}."
                    )

                yindex, xindex, grid_lon, grid_lat = (
                    self.find_mindistance_pointobs_to_grid(
                        lon, lat, grid_lons, grid_lats
                    )
                )
                cell = (int(yindex), int(xindex))
                cell_station_values.setdefault(cell, []).append(
                    monthly_values.to_numpy(dtype=np.float64)
                )
                mapping_rows.append(
                    {
                        "source": source_name,
                        "station_id": os.path.splitext(
                            os.path.basename(station_file)
                        )[0],
                        "lat": lat,
                        "lon": lon,
                        "tsmp_yindex": cell[0],
                        "tsmp_xindex": cell[1],
                        "tsmp_lat": float(grid_lat),
                        "tsmp_lon": float(grid_lon),
                    }
                )
                source_counts[source_name]["complete_stations"] += 1

        if not mapping_rows:
            raise ValueError(
                "No stations have finite, gap-free monthly anomalies for the "
                f"full period {expected_months[0]} to {expected_months[-1]}."
            )

        cell_anomalies = {
            cell: np.mean(np.stack(values, axis=0), axis=0)
            for cell, values in cell_station_values.items()
        }
        mapping = pd.DataFrame(mapping_rows)
        station_counts = mapping.groupby(
            ["tsmp_yindex", "tsmp_xindex"]
        )["station_id"].transform("size")
        mapping["stations_in_cell"] = station_counts.astype(int)
        return cell_anomalies, mapping, source_counts

    def _write_station_anomaly_netcdf(
        self,
        cell_anomalies,
        source_netcdf,
        output_netcdf,
        expected_months,
        grid_shape,
    ):
        """Copy the selected monthly window and fill missing cells with stations."""
        import netCDF4
        import tempfile

        source_netcdf = os.path.abspath(source_netcdf)
        output_netcdf = os.path.abspath(output_netcdf)
        if source_netcdf == output_netcdf:
            raise ValueError("The station anomaly output must not overwrite its source NetCDF.")
        if not os.path.isfile(source_netcdf):
            raise FileNotFoundError(f"Source NetCDF does not exist: {source_netcdf}")

        expected_pairs = [(month.year, month.month) for month in expected_months]
        grid_height, grid_width = grid_shape
        output_dir = os.path.dirname(output_netcdf)
        os.makedirs(output_dir, exist_ok=True)
        temporary_file = tempfile.NamedTemporaryFile(
            prefix=os.path.basename(output_netcdf) + ".",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        )
        temporary_file.close()
        temporary_path = temporary_file.name

        try:
            with netCDF4.Dataset(source_netcdf, "r") as source:
                if "wtd_a" not in source.variables:
                    raise KeyError(
                        f"The source NetCDF does not contain 'wtd_a': {source_netcdf}"
                    )
                source_wtd = source.variables["wtd_a"]
                if len(source_wtd.dimensions) != 3:
                    raise ValueError("'wtd_a' must have (time, y, x) dimensions.")
                time_name = source_wtd.dimensions[0]
                if time_name.lower().find("time") < 0:
                    raise ValueError("'wtd_a' must have its time dimension first.")
                if not np.issubdtype(source_wtd.dtype, np.floating):
                    raise TypeError("'wtd_a' must use a floating-point datatype.")
                if time_name not in source.variables:
                    raise KeyError(f"The NetCDF time coordinate '{time_name}' is missing.")

                time_variable = source.variables[time_name]
                if "units" not in time_variable.ncattrs():
                    raise ValueError(
                        f"NetCDF time coordinate '{time_name}' has no units attribute."
                    )
                decoded_time = netCDF4.num2date(
                    time_variable[:],
                    units=time_variable.units,
                    calendar=getattr(time_variable, "calendar", "standard"),
                    only_use_cftime_datetimes=False,
                )
                source_months = [(date.year, date.month) for date in decoded_time]
                if len(source_months) != len(set(source_months)):
                    raise ValueError("The source NetCDF has duplicate monthly time records.")
                try:
                    start_index = source_months.index(expected_pairs[0])
                except ValueError as exc:
                    raise ValueError(
                        f"The source NetCDF does not contain {expected_months[0]}."
                    ) from exc
                end_index = start_index + len(expected_pairs)
                if source_months[start_index:end_index] != expected_pairs:
                    raise ValueError(
                        "The source NetCDF does not contain the complete, consecutive "
                        f"period {expected_months[0]} to {expected_months[-1]}."
                    )
                if source_wtd.shape[0] != len(source_months):
                    raise ValueError(
                        "The source NetCDF time coordinate does not match 'wtd_a'."
                    )

                if (
                    source_wtd.shape[1] > grid_height
                    or source_wtd.shape[2] > grid_width
                ):
                    raise ValueError(
                        f"NetCDF spatial shape {source_wtd.shape[1:]} exceeds "
                        f"the TSMP grid shape {grid_shape}."
                    )
                y_offset = (grid_height - source_wtd.shape[1]) // 2
                x_offset = (grid_width - source_wtd.shape[2]) // 2
                if (
                    2 * y_offset != grid_height - source_wtd.shape[1]
                    or 2 * x_offset != grid_width - source_wtd.shape[2]
                ):
                    raise ValueError(
                        "The NetCDF grid must be centered on the TSMP grid with "
                        "even-sized offsets."
                    )

                cells = np.asarray(list(cell_anomalies), dtype=int)
                values_by_cell = np.stack(
                    [cell_anomalies[tuple(cell)] for cell in cells], axis=0
                )
                source_y = cells[:, 0] - y_offset
                source_x = cells[:, 1] - x_offset
                if (
                    (source_y < 0).any()
                    or (source_y >= source_wtd.shape[1]).any()
                    or (source_x < 0).any()
                    or (source_x >= source_wtd.shape[2]).any()
                ):
                    raise ValueError(
                        "At least one station maps outside the source NetCDF grid."
                    )

                with netCDF4.Dataset(
                    temporary_path, "w", format=source.data_model
                ) as output:
                    for attribute in source.ncattrs():
                        output.setncattr(attribute, source.getncattr(attribute))
                    for name, dimension in source.dimensions.items():
                        dimension_size = (
                            len(expected_months)
                            if name == time_name
                            else len(dimension)
                        )
                        output.createDimension(
                            name,
                            None if dimension.isunlimited() else dimension_size,
                        )
                    for name, variable in source.variables.items():
                        creation_options = {}
                        if "_FillValue" in variable.ncattrs():
                            creation_options["fill_value"] = variable.getncattr(
                                "_FillValue"
                            )
                        output_variable = output.createVariable(
                            name,
                            variable.datatype,
                            variable.dimensions,
                            **creation_options,
                        )
                        for attribute in variable.ncattrs():
                            if attribute != "_FillValue":
                                output_variable.setncattr(
                                    attribute, variable.getncattr(attribute)
                                )

                        if name != "wtd_a":
                            if time_name in variable.dimensions:
                                time_axis = variable.dimensions.index(time_name)
                                selection = [slice(None)] * variable.ndim
                                selection[time_axis] = slice(start_index, end_index)
                                variable_data = variable[tuple(selection)]
                            else:
                                variable_data = variable[...]
                            output_variable[...] = variable_data
                            continue

                        for output_index, input_index in enumerate(
                            range(start_index, end_index)
                        ):
                            field = np.ma.asarray(
                                source_wtd[input_index, :, :]
                            )
                            field_values = np.asarray(
                                field.filled(np.nan), dtype=np.float64
                            )
                            existing = field_values[source_y, source_x]
                            replace = ~np.isfinite(existing)
                            if replace.any():
                                field_values[
                                    source_y[replace], source_x[replace]
                                ] = values_by_cell[replace, output_index]
                            output_variable[output_index, :, :] = (
                                np.ma.masked_invalid(field_values)
                            )

                    output.setncattr(
                        "station_anomaly_period",
                        f"{expected_months[0]} to {expected_months[-1]}",
                    )
                    output.setncattr(
                        "station_anomaly_merge_method",
                        "Finite station wtd_a values averaged by TSMP grid cell; "
                        "existing finite grid values preserved.",
                    )
            os.replace(temporary_path, output_netcdf)
        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            raise

        return output_netcdf

    def process_groundwater_station_anomalies(
        self,
        groundwater_root=None,
        source_netcdf=None,
        output_netcdf=None,
        start_month="1996-01",
        end_month="2012-12",
    ):
        """Map continuous monthly station anomalies into a period-specific NetCDF.

        Station CSVs must contain ``wtd_a`` and a ``month`` or ``date`` column.
        Only stations with a finite anomaly in every requested month are used.
        Stations sharing a TSMP cell are averaged, and those averages fill
        missing values in the original gridded ``wtd_a`` data.
        """
        project_root = get_root_dir()
        data_root = os.path.dirname(project_root)
        if groundwater_root is None:
            groundwater_root = os.path.join(data_root, "Groundwater_Observations")
        if source_netcdf is None:
            source_netcdf = os.path.join(
                data_root,
                "grid_measured_wtd_a_data_1996_2016_EUR11_YM_v_20210407.nc",
            )
        if output_netcdf is None:
            output_netcdf = os.path.join(
                data_root,
                "grid_measured_wtd_a_data_1996_2012_EUR11_YM_v_20210407_"
                "with_station_anomalies.nc",
            )

        try:
            first_month = pd.Period(start_month, freq="M")
            last_month = pd.Period(end_month, freq="M")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "start_month and end_month must be valid monthly dates."
            ) from exc
        if first_month > last_month:
            raise ValueError("start_month must not be later than end_month.")
        expected_months = pd.period_range(first_month, last_month, freq="M")

        source_dirs = {
            "England": os.path.join(
                groundwater_root,
                "England_GW_data",
                "england_gw_monthly_wells",
            ),
            "GEMS-GER": os.path.join(
                groundwater_root,
                "GEMS-GER_data",
                "gems_ger_monthly_wells",
            ),
            "GROW Europe": os.path.join(
                groundwater_root,
                "GROW_europe_monthly_wells",
            ),
            "Switzerland": os.path.join(
                groundwater_root,
                "Swiss_GW_data",
                "swiss_gw_monthly_wells",
            ),
        }
        grid_lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        grid_lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))
        if grid_lons.shape != grid_lats.shape or grid_lons.ndim != 2:
            raise ValueError(
                "TSMP longitude and latitude grids must be matching 2D arrays."
            )
        cell_anomalies, mapping, source_counts = (
            self._read_continuous_station_anomalies(
                source_dirs, expected_months, grid_lons, grid_lats
            )
        )
        output_netcdf = self._write_station_anomaly_netcdf(
            cell_anomalies,
            source_netcdf,
            output_netcdf,
            expected_months,
            grid_lons.shape,
        )

        mapping_path = os.path.splitext(output_netcdf)[0] + "_station_mapping.csv"
        mapping.to_csv(mapping_path, index=False)
        return {
            "netcdf": output_netcdf,
            "station_mapping": mapping_path,
            "start_month": str(first_month),
            "end_month": str(last_month),
            "n_months": len(expected_months),
            "station_count": len(mapping),
            "stations_by_source": {
                source: counts["complete_stations"]
                for source, counts in source_counts.items()
            },
            "excluded_incomplete_by_source": {
                source: counts["incomplete_stations"]
                for source, counts in source_counts.items()
            },
            "grid_cell_count": len(cell_anomalies),
            "stations_in_shared_cells": int(
                (mapping["stations_in_cell"] > 1).sum()
            ),
            "extra_stations_in_shared_cells": (
                len(mapping) - len(cell_anomalies)
            ),
        }

    # The longest gap-free window maximizing station coverage is January 1996–December 2012 inclusive
    # (204 monthly values; 17 years), with all 4,339 stations complete: England 41, GEMS-GER 3,207, GROW Europe 810, Switzerland 281.