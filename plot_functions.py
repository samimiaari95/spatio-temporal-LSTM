import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from shapely.geometry import Polygon
import os
from LSTM_setup import *


def plot_results(data, title, label, lons, lats):
    projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

    # Create a figure and an axis with a Cartopy projection
    fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
    
    # Plot the data
    data = np.nan_to_num(data)
    norm = mcolors.LogNorm(vmin=0.01, vmax=2)
    data[data == 0.0] = np.nan

    cla = ax.pcolormesh(lons, lats, data, norm=norm, cmap='viridis', transform=ccrs.PlateCarree())

    # Add a colorbar
    cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label(f'{label}')

    # Add coastlines, gridlines, etc.
    ax.gridlines(draw_labels=True)
    print(f"saving {title}")
    fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

def plot_diff(data, title, label, lons, lats):
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

def plot_MSE(data, title, label, lons, lats):
    projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

    # Create a figure and an axis with a Cartopy projection
    fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
    
    # Plot the data
    data = np.nan_to_num(data)
    #norm = mcolors.LogNorm(vmin=np.min(data), vmax=np.max(data))
    #norm = mcolors.LogNorm(vmin=0.001, vmax=1.1)
    norm = mcolors.LogNorm(vmin=0.001, vmax=1.0)

    cla = ax.pcolormesh(lons, lats, data,norm=norm, cmap='viridis', transform=ccrs.PlateCarree())

    # Add a colorbar
    cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label(f'{label}')

    # Add coastlines, gridlines, etc.
    ax.gridlines(draw_labels=True)
    print(f"saving {title}")
    fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

def plot_Europe_avg(TRAINING_PERIOD, LOOKBACK, TEST_PERIOD, lons, lats):
    # TODO load data from original file
    data = np.load(os.path.join(os.path.dirname(INPUTPATH), "wtd.npy"))[TRAINING_PERIOD+LOOKBACK:TRAINING_PERIOD+TEST_PERIOD,:,:]
    projection = ccrs.LambertAzimuthalEqualArea(central_longitude=19, central_latitude=53)

    # Create a figure and an axis with a Cartopy projection
    fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})
    # NOTE axis 0 is the time step
    data = np.mean(data, axis=0)

    # Plot the data
    norm = mcolors.LogNorm(vmin=0.01, vmax=np.max(data))
    data[data == 0] = np.nan

    cla = ax.pcolormesh(lons, lats, data, cmap='viridis', norm=norm, transform=ccrs.PlateCarree())

    # Add a colorbar
    cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Water table depth (mm)')

    # Add coastlines, gridlines, etc.
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    # Path to your shapefile
    #shapefile_path = os.path.join(os.path.dirname(INPUTPATH), "DANUBE_DOWNSTREAM", "catchment_shp", "danube.shp")
    #shape_feature_danube = ShapelyFeature(Reader(shapefile_path).geometries(), ccrs.PlateCarree(), edgecolor='red')
    #ax.add_feature(shape_feature_danube, facecolor='none', edgecolor='red', linewidth=1)

    # Path to your shapefile
    shapefile_path = os.path.join(os.path.dirname(INPUTPATH), "SEINE", "catchment_shp", "seine.shp")
    shape_feature_seine = ShapelyFeature(Reader(shapefile_path).geometries(), ccrs.PlateCarree(), edgecolor='red')
    ax.add_feature(shape_feature_seine, facecolor='none', edgecolor='red', linewidth=1)


    # danube region starting lat lon
    #i = 160
    #j = 290
    #grid_size = 5
    #points = [[lons[i,j], lats[i,j]], [lons[i,j+grid_size], lats[i,j+grid_size]], [lons[i+grid_size,j+grid_size], lats[i+grid_size,j+grid_size]], [lons[i+grid_size,j], lats[i+grid_size,j]]]
    #study_area_polygon = Polygon(points)
    #study_area_feature_danube = ShapelyFeature([study_area_polygon], ccrs.PlateCarree(), edgecolor='blue', facecolor='none')
    #ax.add_feature(study_area_feature_danube, edgecolor='blue', linewidth=2)

    # seine region starting lat lon
    i = 211 #195+16
    j = 177 #164+13
    grid_size = 5    
    points = [[lons[i,j], lats[i,j]], [lons[i,j+grid_size], lats[i,j+grid_size]], [lons[i+grid_size,j+grid_size], lats[i+grid_size,j+grid_size]], [lons[i+grid_size,j], lats[i+grid_size,j]]]
    study_area_polygon = Polygon(points)
    study_area_feature_seine = ShapelyFeature([study_area_polygon], ccrs.PlateCarree(), edgecolor='blue', facecolor='none')
    ax.add_feature(study_area_feature_seine, edgecolor='blue', linewidth=2)

    print("saving europe")
    fig.savefig(os.path.join(os.path.dirname(OUTPUTPATH), "Europe_avgwtd.png"))

def correlation_map(obs, sim, title, X, Y, lons, lats):
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

    cla = ax.pcolormesh(lons, lats, correlation_map, cmap='viridis', transform=ccrs.PlateCarree(), vmin=-1, vmax=1)

    # Add a colorbar
    cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label("Correlation coefficient")

    # Add coastlines, gridlines, etc.
    ax.gridlines(draw_labels=True)
    print(f"saving correlation")
    fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

def calc_plot_bias(obs, sim, title, X, Y, lons, lats):
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

    #cla = ax.pcolormesh(lons, lats, bias_map, cmap='viridis', transform=ccrs.PlateCarree(), vmin=np.min(bias_map), vmax=np.max(bias_map))
    cla = ax.pcolormesh(lons, lats, bias_map, cmap='viridis', transform=ccrs.PlateCarree(), vmin=-0.7, vmax=1.1)
    #cla = ax.pcolormesh(lons, lats, bias_map, cmap='viridis', transform=ccrs.PlateCarree(), vmin=-0.5, vmax=0.9)

    # Add a colorbar
    cbar = plt.colorbar(cla, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label("Bias (m)")

    # Add coastlines, gridlines, etc.
    ax.gridlines(draw_labels=True)
    print(f"saving bias")
    fig.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

def plot_blendaltman(obs, sim, title):
    predictions = sim.flatten()
    observations = obs.flatten()

    # Calculate the mean and difference between predictions and observations
    mean_values = np.mean([predictions, observations], axis=0)
    differences = predictions - observations

    # Calculate the mean difference (bias) and the limits of agreement
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # Create the Bland-Altman plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_values, differences, alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference (Bias)')
    plt.axhline(loa_upper, color='blue', linestyle='--', label='Upper Limit of Agreement (Mean + 1.96 SD)')
    plt.axhline(loa_lower, color='blue', linestyle='--', label='Lower Limit of Agreement (Mean - 1.96 SD)')
    plt.xlabel('Mean of Predictions and Observations (m)')
    plt.ylabel('Difference between Predictions and Observations (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTPATH, f"{title}.png"))

