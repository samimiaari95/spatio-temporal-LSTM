import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import r2_score
from LSTM_model.model.config import *

class plotting_helper:
    def __init__(self) -> None:
        pass

    def plot_results(self, data, title, label, lons, lats):
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
        
        # Plot the data
        data = np.nan_to_num(data)
        norm = mcolors.LogNorm(vmin=0.01, vmax=2)
        data[data == 0.0] = np.nan

        cla = ax.pcolormesh(lons, lats, data, norm=norm, cmap='viridis', transform=ccrs.PlateCarree())

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.08)
        cbar.set_label(f'{label}')

        # Add coastlines, gridlines, etc.
        ax.gridlines(draw_labels=True)
        print(f"saving {title}")
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

    def plot_diff(self, data, title, label, lons, lats):
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
        
        # Plot the data
        data = np.nan_to_num(data)
        norm = mcolors.LogNorm(vmin=0.01, vmax=55)

        cla = ax.pcolormesh(lons, lats, data, norm=norm, cmap='viridis', transform=ccrs.PlateCarree())

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label(f'{label}')

        # Add coastlines, gridlines, etc.
        ax.gridlines(draw_labels=True)
        print(f"saving {title}")
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

    def plot_MSE(self, data, title, label, lons, lats):
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
        
        # Plot the data
        data = np.nan_to_num(data)
        #norm = mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))
        #norm = mcolors.LogNorm(vmin=0.001, vmax=1.1)
        norm = mcolors.LogNorm(vmin=0.001, vmax=10.0)

        cla = ax.pcolormesh(lons, lats, data,norm=norm, cmap='viridis', transform=ccrs.PlateCarree())

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.08)
        cbar.set_label(f'{label}')

        # Add coastlines, gridlines, etc.
        #ax.coastlines()
        ax.gridlines(draw_labels=True)
        print(f"saving {title}")
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

    def plot_Europe_avg(self, TRAINING_PERIOD, LOOKBACK, TEST_PERIOD, lons, lats):
        # TODO convert data from mm to m
        data = np.load(os.path.join(os.path.dirname(INPUTPATH), "wtd.npy"))[TRAINING_PERIOD+LOOKBACK:TRAINING_PERIOD+TEST_PERIOD,:,:]
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
        # NOTE axis 0 is the time step
        data = np.mean(data, axis=0)

        # Plot the data
        # TODO set max limit
        norm = mcolors.LogNorm(vmin=0.01, vmax=np.max(data))
        data[data < 0.01] = np.nan

        cla = ax.pcolormesh(lons, lats, data, cmap='viridis', norm=norm, transform=ccrs.PlateCarree())

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.08)
        cbar.set_label('Water table depth (m)')

        # Add coastlines, gridlines, etc.
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        # Path to your shapefile
        shapefile_path = os.path.join(os.path.dirname(INPUTPATH), "DANUBE", "catchment_shp", "danube.shp")
        shape_feature_danube = ShapelyFeature(Reader(shapefile_path).geometries(), ccrs.PlateCarree(), edgecolor='red')
        #ax.add_feature(shape_feature_danube, facecolor='none', edgecolor='red', linewidth=1)

        # Path to your shapefile
        shapefile_path = os.path.join(os.path.dirname(INPUTPATH), "SEINE", "catchment_shp", "seine.shp")
        shape_feature_seine = ShapelyFeature(Reader(shapefile_path).geometries(), ccrs.PlateCarree(), edgecolor='red')
        ax.add_feature(shape_feature_seine, facecolor='none', edgecolor='red', linewidth=1)

        # Path to your shapefile
        shapefile_path = os.path.join(os.path.dirname(INPUTPATH), "DOURO", "catchment_shp", "douro.shp")
        shape_feature_seine = ShapelyFeature(Reader(shapefile_path).geometries(), ccrs.PlateCarree(), edgecolor='red')
        ax.add_feature(shape_feature_seine, facecolor='none', edgecolor='red', linewidth=1)

        # danube region starting lat lon
        i = 174#160
        j = 302#290
        grid_size = 5
        points = [[lons[i,j], lats[i,j]], [lons[i,j+grid_size], lats[i,j+grid_size]], [lons[i+grid_size,j+grid_size], lats[i+grid_size,j+grid_size]], [lons[i+grid_size,j], lats[i+grid_size,j]]]
        study_area_polygon = Polygon(points)
        study_area_feature_danube = ShapelyFeature([study_area_polygon], ccrs.PlateCarree(), edgecolor='blue', facecolor='none')
        #ax.add_feature(study_area_feature_danube, edgecolor='blue', linewidth=2)

        # seine region starting lat lon
        i = 211 #195+16
        j = 177 #164+13
        grid_size = 5    
        points = [[lons[i,j], lats[i,j]], [lons[i,j+grid_size], lats[i,j+grid_size]], [lons[i+grid_size,j+grid_size], lats[i+grid_size,j+grid_size]], [lons[i+grid_size,j], lats[i+grid_size,j]]]
        study_area_polygon = Polygon(points)
        study_area_feature_seine = ShapelyFeature([study_area_polygon], ccrs.PlateCarree(), edgecolor='blue', facecolor='none')
        ax.add_feature(study_area_feature_seine, edgecolor='blue', linewidth=2)

        # DOURO region starting lat lon
        i = 163# 150
        j = 100#95
        grid_size = 5
        points = [[lons[i,j], lats[i,j]], [lons[i,j+grid_size], lats[i,j+grid_size]], [lons[i+grid_size,j+grid_size], lats[i+grid_size,j+grid_size]], [lons[i+grid_size,j], lats[i+grid_size,j]]]
        study_area_polygon = Polygon(points)
        study_area_feature_seine = ShapelyFeature([study_area_polygon], ccrs.PlateCarree(), edgecolor='blue', facecolor='none')
        ax.add_feature(study_area_feature_seine, edgecolor='blue', linewidth=2)

        print("saving europe")
        fig.savefig(os.path.join(os.path.dirname(OUTPUTPATH), "Europe_avgwtd_5x5.png"))

    def correlation_map(self, obs, sim, title, X, Y, lons, lats):
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})

        # TODO calculate correlation in original script
        # reshape input data
        obs = obs.reshape(obs.shape[0], X,Y)
        sim = sim.reshape(sim.shape[0], X,Y)
        # calculate correlation
        correlation_map = np.zeros((X,Y))
        # Iterate over each grid cell
        for i in range(X):
            for j in range(Y):
                # Extract the time series for the current grid cell
                time_series1 = obs[:, i, j]
                time_series2 = sim[:, i, j]
                
                # Calculate the Pearson correlation coefficient
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:  # Avoid division by zero
                    correlation_matrix = np.corrcoef(time_series1, time_series2)
                    r = correlation_matrix[0, 1]
                else:
                    r = np.nan  # If there's no variation, set correlation to NaN
                
                # Store the correlation coefficient in the map
                correlation_map[i, j] = r
        
        # Plot the data
        correlation_map = np.nan_to_num(correlation_map)
        correlation_map[correlation_map == 0] = np.nan

        cla = ax.pcolormesh(lons, lats, correlation_map, cmap='coolwarm', transform=ccrs.PlateCarree(), vmin=-1, vmax=1)

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.08)
        cbar.set_label("Correlation coefficient")

        # Add coastlines, gridlines, etc.
        #ax.coastlines()
        ax.gridlines(draw_labels=True)
        print(f"saving correlation")
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

    def calc_plot_bias(self, obs, sim, title, X, Y, lons, lats):
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})

        # TODO calculate bias in original script
        # reshape input data
        obs = obs.reshape(obs.shape[0], X,Y)
        sim = sim.reshape(sim.shape[0], X,Y)
        
        # calculate bias
        bias_map = np.mean(sim - obs, axis=0)
        
        # Plot the data
        bias_map = np.nan_to_num(bias_map)
        bias_map[bias_map == 0] = np.nan
        cmap = plt.get_cmap('coolwarm')  # 'coolwarm' is a commonly used diverging colormap
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


        #cla = ax.pcolormesh(lons, lats, bias_map, cmap='viridis', transform=ccrs.PlateCarree(), vmin=np.min(bias_map), vmax=np.max(bias_map))
        cla = ax.pcolormesh(lons, lats, bias_map, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        #cla = ax.pcolormesh(lons, lats, bias_map, cmap='viridis', transform=ccrs.PlateCarree(), vmin=-0.5, vmax=0.9)

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.08)
        cbar.set_label("Bias (m)")

        # Add coastlines, gridlines, etc.
        #ax.coastlines()
        ax.gridlines(draw_labels=True)
        print(f"saving bias")
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

    def pixels_biasmap(self, obs, sim, title, X, Y):
        fig, ax = plt.subplots(figsize=(16, 9))

        # TODO calculate bias in original script
        # reshape input data
        obs = obs.reshape(obs.shape[0], X,Y)
        sim = sim.reshape(sim.shape[0], X,Y)
        
        # calculate bias
        bias_map = np.mean(sim - obs, axis=0)
        
        # Plot the data
        bias_map = np.nan_to_num(bias_map)
        bias_map[bias_map == 0] = np.nan
        cmap = plt.get_cmap('coolwarm')  # 'coolwarm' is a commonly used diverging colormap
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


        #cla = ax.pcolormesh(lons, lats, bias_map, cmap='viridis', transform=ccrs.PlateCarree(), vmin=np.min(bias_map), vmax=np.max(bias_map))
        cla = ax.pcolormesh(bias_map, cmap=cmap, norm=norm)
        #cla = ax.pcolormesh(lons, lats, bias_map, cmap='viridis', transform=ccrs.PlateCarree(), vmin=-0.5, vmax=0.9)

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label("Bias (m)")

        # Add coastlines, gridlines, etc.
        #ax.gridlines(draw_labels=True)
        print(f"saving bias")
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

    def pixel_correlation_map(self, obs, sim, title, X, Y):
        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9))

        # TODO calculate correlation in original script
        # reshape input data
        obs = obs.reshape(obs.shape[0], X,Y)
        sim = sim.reshape(sim.shape[0], X,Y)
        # calculate correlation
        correlation_map = np.zeros((X,Y))
        # Iterate over each grid cell
        for i in range(X):
            for j in range(Y):
                # Extract the time series for the current grid cell
                time_series1 = obs[:, i, j]
                time_series2 = sim[:, i, j]
                
                # Calculate the Pearson correlation coefficient
                if np.std(time_series1) > 0 and np.std(time_series2) > 0:  # Avoid division by zero
                    correlation_matrix = np.corrcoef(time_series1, time_series2)
                    r = correlation_matrix[0, 1]
                else:
                    r = np.nan  # If there's no variation, set correlation to NaN
                
                # Store the correlation coefficient in the map
                correlation_map[i, j] = r
        
        # Plot the data
        correlation_map = np.nan_to_num(correlation_map)
        correlation_map[correlation_map == 0] = np.nan

        cla = ax.pcolormesh(correlation_map, cmap='viridis', vmin=-1, vmax=1)
        
        # Add the value for each pixel
        for i in range(correlation_map.shape[0]):
            for j in range(correlation_map.shape[1]):
                plt.text(j, i, f'{correlation_map[i, j]:.2f}', ha='left', va='bottom', color='white')

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label("Correlation coefficient")

        # Add coastlines, gridlines, etc.
        #ax.gridlines(draw_labels=True)
        print(f"saving correlation")
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

    def pixel_plot_MSE(self, data, title, label):
        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Plot the data
        data = np.nan_to_num(data)
        #norm = mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))
        norm = mcolors.LogNorm(vmin=0.001, vmax=1.0)

        cla = ax.pcolormesh(data,norm=norm, cmap='viridis')

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label(f'{label}')

        # Add coastlines, gridlines, etc.
        #ax.gridlines(draw_labels=True)
        print(f"saving {title}")
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

    def chosenpixels_heatmap_seinetransfer(self, data, logscale, minval, maxval, title):
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
        
        # get EU lon lat
        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))
        # define limits and normalization
        norm  = mcolors.LogNorm(vmin=minval, vmax=maxval) if logscale else Normalize(vmin=minval, vmax=maxval)

        # define colorscale
        cmap_colors = "viridis" if "MSE" in title else "coolwarm"
        cmap = plt.get_cmap(cmap_colors)

        cla = ax.pcolormesh(lons, lats, data, norm=norm, cmap=cmap, transform=ccrs.PlateCarree())
        
        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
        colorbar_label = f"{title}" if title=="Correlation" else f"{title} (m)"
        cbar.set_label(colorbar_label)

        ax.gridlines(draw_labels=True)
        # Add the value for each pixel
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if (i==0 and j==8) or (i==0 and j==22) or (i==13 and j==22) or (i==14 and j==25) or (i==15 and j==25):
                    ax.plot(lons[i,j], lats[i,j], marker='*', color="lime", markersize=15, 
                            transform=ccrs.PlateCarree(), label='Special Point')

        print(f"saving {title} heatmap")
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}_{MODEL_NAME}_heatmap.png"))

    def chosenpixels_heatmap(self, data, logscale, minval, maxval, title):
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # define limits and normalization
        norm  = mcolors.LogNorm(vmin=minval, vmax=maxval) if logscale else Normalize(vmin=minval, vmax=maxval)

        # define colorscale
        cmap_colors = "viridis" if "MSE" in title else "coolwarm"
        cmap = plt.get_cmap(cmap_colors)

        cla = ax.pcolormesh(data, norm=norm, cmap=cmap)

        # Add a colorbar
        cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
        colorbar_label = f"{title}" if title=="Correlation" or title=="NSE" or title=="KGE" else f"{title} (m)"
        cbar.set_label(colorbar_label)

        # Add the value for each pixel
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f'{data[i, j]:.2f}', ha='left', va='bottom', color='white')


        print(f"saving {title} heatmap")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}_{MODEL_NAME}_heatmap.png"))


    def chosenpixels_in_EU(self, data_map, logscale, minval, maxval, title):
        indices = np.where(~np.isnan(data_map))
        indices_list = list(zip(indices[0], indices[1]))

        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # get EU lon lat
        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
        cmap_colors = "viridis" if "MSE" in title else "coolwarm"
        cmap = plt.get_cmap(cmap_colors)

        # define limits and normalization
        norm  = mcolors.LogNorm(vmin=minval, vmax=maxval) if logscale else Normalize(vmin=minval, vmax=maxval)

        # Plot the 2D EU map
        cla = ax.pcolormesh(lons, lats, data_map, cmap='viridis', norm=norm, transform=ccrs.PlateCarree())

        for index in indices_list:
            pixel_value = data_map[index]  # Get the data value at the specific pixel
            pixel_color = cmap(norm(pixel_value))  # Get the color from the colormap
            ax.plot(lons[index], lats[index], marker='*', color=pixel_color, markersize=15, 
                    transform=ccrs.PlateCarree(), label='Special Point')

        # Add a colorbar
        colorbar_label = f"{title}" if title=="Correlation" or title=="NSE" or title=="KGE" else f"{title} (m)"
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label=colorbar_label, pad=0.08)

        # Add coastlines, gridlines, etc.
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        print(f"saving {title}")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUTPATH, f"{title}_{MODEL_NAME}.png"))

    def onechosenpixel_in_EU(self, indexes):
        choices = np.load(os.path.join(INPUTPATH, "choices.npy"))    
        include = np.where(choices==1)
        map_choices = np.zeros(choices.shape)
        for ind in indexes:
            map_choices[include[0][ind], include[1][ind]] = 1
        map_choices[map_choices == 0] = np.nan
        indices = np.where(~np.isnan(map_choices))
        indices_list = list(zip(indices[0], indices[1]))

        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # get EU lon lat
        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
        
        # define limits and normalization
        norm  = mcolors.LogNorm(vmin=0.001, vmax=1)

        # Plot the 2D EU map
        cla = ax.pcolormesh(lons, lats, map_choices, cmap='viridis', norm=norm, transform=ccrs.PlateCarree())

        for index in indices_list:
            ax.plot(lons[index], lats[index], marker='*', color="r", markersize=15, 
                    transform=ccrs.PlateCarree(), label='Special Point')

        # Add coastlines, gridlines, etc.
        ax.coastlines(color='black')
        ax.gridlines(draw_labels=True)
        
        indexes = [str(x) for x in indexes]
        indexes = "".join(indexes)
        print(f"saving one pixel at {indexes}")
        fig.savefig(os.path.join(OUTPUTPATH, f"{indexes}_{TARGET_REGION}.png"))

    def selectedpixels_in_EU(self, mappingpath):
        mapping = np.load(mappingpath)
        mapping[mapping == 0] = np.nan
        indices = np.where(~np.isnan(mapping))
        indices_list = list(zip(indices[0], indices[1]))

        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # get EU lon lat
        lons = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "lon2D.npy"))
        lats = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "lat2D.npy"))

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
        
        # Plot the 2D EU map
        cla = ax.pcolormesh(lons, lats, mapping, transform=ccrs.PlateCarree())

        for index in indices_list:
            ax.plot(lons[index], lats[index], marker='*', color="k", markersize=15, 
                    transform=ccrs.PlateCarree(), label='Special Point')

        # Add coastlines, gridlines, etc.
        ax.coastlines(color='black')
        ax.gridlines(draw_labels=True)
        
        print(f"saving selected pixels at {TARGET_REGION}")
        filename = os.path.basename(mappingpath)
        filename = filename.split(".")[0]
        fig.savefig(os.path.join(os.path.dirname(OUTPUTPATH), f"{filename}.png"))

    def plot_RB(self):
        data = np.zeros((30,30))
        print(data.shape)

        projection = ccrs.LambertAzimuthalEqualArea()

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})

        i = 195 #195+16
        j = 164 #164+13
        grid_size = 30
        

        lons = np.load(os.path.join(os.path.dirname(INPUTPATH), "lon2D.npy"))[i:i+grid_size, j:j+grid_size]
        lats = np.load(os.path.join(os.path.dirname(INPUTPATH), "lat2D.npy"))[i:i+grid_size, j:j+grid_size]
        print(lats.shape)

        data[data < 0.01] = np.nan

        cla = ax.pcolormesh(lons, lats, data, cmap='viridis', transform=ccrs.PlateCarree())

        # Add coastlines, gridlines, etc.
        #ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        # Path to your shapefile
        shapefile_path = os.path.join(os.path.dirname(INPUTPATH), "SEINE_10x10", "catchment_shp", "seine.shp")
        shape_feature_seine = ShapelyFeature(Reader(shapefile_path).geometries(), ccrs.PlateCarree(), edgecolor='red')
        ax.add_feature(shape_feature_seine, facecolor='none', edgecolor='red', linewidth=1)

        # seine region starting lat lon
        i = 195 #195+16
        j = 164 #164+13
        grid_size = 30
        lons = np.load(os.path.join(os.path.dirname(INPUTPATH), "lon2D.npy"))
        lats = np.load(os.path.join(os.path.dirname(INPUTPATH), "lat2D.npy"))
        points = [[lons[i,j], lats[i,j]], [lons[i,j+grid_size], lats[i,j+grid_size]], [lons[i+grid_size,j+grid_size], lats[i+grid_size,j+grid_size]], [lons[i+grid_size,j], lats[i+grid_size,j]]]
        study_area_polygon = Polygon(points)
        study_area_feature_seine = ShapelyFeature([study_area_polygon], ccrs.PlateCarree(), edgecolor='blue', facecolor='none')
        ax.add_feature(study_area_feature_seine, edgecolor='blue', linewidth=2)

        print("saving seine")
        fig.savefig(os.path.join(os.path.dirname(OUTPUTPATH), "Seine_30x30.png"))
    
    def plot_predr2(self, obs, sim):
        # Calculate R2
        r2 = r2_score(obs, sim)
        print(r2)
        # Scatter plot
        plt.figure(figsize=(16, 9))
        plt.scatter(obs, sim, color='k', alpha=0.6, label=f'$R^2$: {r2:.3f}')

        # Plot identity line (y = x)
        plt.plot([obs.min(), obs.max()],
                [obs.min(), obs.max()],
                color='red', linestyle='--', label='Identity Line')

        # Add labels, title, and legend
        plt.xlabel('Original Simulations')
        plt.ylabel('Predictions')

        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Show the plot
        print("saving r2")
        plt.savefig(os.path.join(OUTPUTPATH, f"R2_{MODEL_NAME}.png"))

    def EU_2Dmap(self, data_map, logscale, minval, maxval, title):
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

        # get EU lon lat
        lons = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lats = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

        # Create a figure and an axis with a Cartopy projection
        fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
        cmap_colors = "viridis" if ("MSE" in title or "ias" in title or "KGE" in title) else "coolwarm"
        # cmap_colors = "terrain"
        cmap = plt.get_cmap(cmap_colors)

        # define limits and normalization
        norm = mcolors.LogNorm(vmin=minval, vmax=maxval) if logscale else Normalize(vmin=minval, vmax=maxval)

        # Plot the 2D EU map
        cla = ax.pcolormesh(lons, lats, data_map, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

        # Add a colorbar
        colorbar_label = f"{title}" if title=="Pearson correlation" or title=="NSE" or title=="KGE" else f"{title} (m)"
        plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label=colorbar_label, pad=0.08)

        # Add coastlines, gridlines, etc.
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        print(f"saving {title}")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", "ensemble_mean", f"2Dmap_{title}.png"))
    
    def plot_4d_map_logscale(self, data, cmap='viridis', vmin=None, vmax=None, output_filename="map_4d_logscale.png"):
        """
        Plot a 4D array (year, season, lon, lat) into 16 subplots with a log color scale and save as high-res PNG.

        Args:
            data: np.ndarray of shape (4 years, 4 seasons, lon, lat)
            lon: optional longitude array (2D)
            lat: optional latitude array (2D)
            cmap: colormap
            vmin, vmax: minimum and maximum values for the color normalization (must be >0 for log scale)
            output_filename: filename to save the figure
        """
        projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)
        # get EU lon lat
        lon = np.load(os.path.join(INPUTPATH, "lon2D.npy"))
        lat = np.load(os.path.join(INPUTPATH, "lat2D.npy"))

        years = [2017, 2018, 2019, 2020]
        seasons = ["DJF", "MAM", "JJA", "SON"]
        
        # Check vmin and vmax
        if vmin is None:
            vmin = np.nanmin(data[data > 0])  # log scale needs positive values
        if vmax is None:
            vmax = np.nanmax(data)
            
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        fig, axs = plt.subplots(4, 4, figsize=(24, 20), constrained_layout=True, subplot_kw={'projection': projection})

        for i in range(4):  # years
            for j in range(4):  # seasons
                ax = axs[i, j]
                
                if lon is not None and lat is not None:
                    # pcolormesh with log norm
                    # im = ax.pcolormesh(lon, lat, data[i, j, :, :], cmap=cmap, norm=norm, shading='auto')
                    im = ax.pcolormesh(lon, lat, data[i, j, :, :], cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')
                    # ax.set_xlabel('Longitude')
                    # ax.set_ylabel('Latitude')
                    ax.set_aspect('auto')
                    ax.gridlines(draw_labels=False)
        
                else:
                    # imshow with log norm
                    im = ax.imshow(data[i, j, :, :], origin='lower', cmap=cmap, norm=norm, aspect='auto')
                    ax.set_xlabel('Pixel-X')
                    ax.set_ylabel('Pixel-Y')
                
                #ax.set_title(f"{years[i]} - {seasons[j]}")
                #ax.set_aspect('auto')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
                # Major ticks formatting
                ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

                # Only add season titles to top row
                if i == 0:
                    ax.set_title(seasons[j], fontsize=26)

                # Only add year labels to rightmost column
                if j == 3:  # last column (rightmost)
                    ax.annotate(f"{years[i]}", xy=(1.05, 0.5), xycoords='axes fraction', rotation=0, ha='left', va='center', fontsize=26)


        # Add shared colorbar
        cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('CRPS', rotation=270, labelpad=15)

        #fig.subplots_adjust(left=0.12)  # <-- added, to make space for left labels

        # Save the figure
        #os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)


    def plot_scatter(self, x, y, title, ylog=False, ysymlog=False):
        plt.figure()
        plt.scatter(x, y, alpha=0.5)
        plt.xlabel(r'$Topography (m)$')
        # plt.xlabel(r'$Mean WTD_O (m)$')
        plt.ylabel(f"{title}")
        plt.xscale('log')
        if ylog:
            plt.yscale('log')
        if ysymlog:
            plt.yscale('symlog')
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"topo_vs_{title}.png"), dpi=300, bbox_inches='tight')

    def logscales_histogram(self, data, xlabel, ylabel, nbbins=50, title="Histogram"):
        plt.figure(figsize=(10, 6))
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), num=nbbins)
        plt.hist(data, bins=bins, color='blue', edgecolor='black', alpha=0.7)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(OUTPUTPATH)), "validation_400_withcriteria_43226", "statistics", f"{title}.png"), dpi=300, bbox_inches='tight')
        plt.close()