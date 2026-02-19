import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
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
            if "Bayern" in country or "Wuerttemberg" in country:
                continue # should correct elevations manually
            if os.path.isdir(dircountry):
                datapath = os.path.join(dircountry, "p.data.monthly_averaged_WTD_measured_data", os.listdir(os.path.join(dircountry, "p.data.monthly_averaged_WTD_measured_data"))[0], "2000_2015_monthly")
                # read txt as a dataframe
                well_coords_df = pd.read_csv(os.path.join(datapath, "well_coordinates.txt"), delim_whitespace=True)
                well_coords_df['well_index'] = well_coords_df['well_index'].astype(str)

                for file in os.listdir(datapath):
                    if file.endswith('.csv'):
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
        obsdf.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU_monthly.parquet"), index=False)

    def find_mindistance_pointobs_to_grid(self, lon, lat):
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

        # load TSMP lonlat
        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

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
        df_eu = pd.read_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU_monthly.parquet"))
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
        df_eu.to_parquet(os.path.join(os.path.join(INPUTPATH, "localobservations"), "wtd_obsEU_monthly_avgdup.parquet"), index=False)
        print(locations)
        locations.to_csv(os.path.join(os.path.join(INPUTPATH, "localobservations"), "mapping_wtdobsEU_TSMP_avgdup.csv"), index=False)