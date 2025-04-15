import os
import numpy as np
import netCDF4 as nc
from typing import Dict, Union, List, Optional
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from LSTM_model.model.config import *

class utilities:
    def __init__(self) -> None:
        pass

    def make_dir(self, dir_path):
        # iterate through the defined path
        while not os.path.isdir(dir_path):
            # check if the parent directory exists
            if not os.path.isdir(os.path.dirname(dir_path)):
                self.make_dir(os.path.dirname(dir_path))
            else:
                os.mkdir(dir_path)
    
    def write_file(filename, intval):
        with open(filename, 'w') as fp:
            fp.write(str(intval))
    
    def read_file(filename):
        with open(filename) as fp:
            return fp.read()

    def powerlaw_func(self, h, a, b):
            y = a*(h**b)
            return y

    def linear_law(self, x, a, b) :
            return a + x * b

    def exponential_func(self, x, a, b):
        return a * np.exp(b * x)

    def read_nc(self, filepath, var):
        ncfile = nc.Dataset(filepath)
        return ncfile[var][:]

    def open_nc(self, filepath):
        ncfile = nc.Dataset(filepath)
        variables = ncfile.variables
        for var in variables:
            print(ncfile[var])

    def delete_files(self, dirpath, key):
        for file in os.listdir(dirpath):
            if key in file:
                os.remove(os.path.join(dirpath, file))
    
    def calculate_kge(self, observed, predicted):
        """
        Compute Kling-Gupta Efficiency (KGE) between observed and predicted time series.

        Parameters:
        observed (numpy.ndarray): 1D array of observed values.
        predicted (numpy.ndarray): 1D array of predicted values.

        Returns:
        float: KGE value
        """
        # Ensure both arrays have the same length
        assert len(observed) == len(predicted), "Observed and predicted arrays must have the same length."
        
        if np.isnan(observed).all() or np.isnan(predicted).all():
            return np.nan
        if np.std(observed) < 0.1: # if the std is close to zero, exclude pixel kge
            return np.nan
            
        # Compute correlation coefficient (r)
        r = np.corrcoef(observed, predicted)[0, 1]

        # Compute mean and standard deviation
        mu_o, mu_p = np.mean(observed), np.mean(predicted)
        sigma_o, sigma_p = np.std(observed), np.std(predicted)

        # Compute bias ratio (β) and variability ratio (γ)
        beta = mu_p / mu_o
        gamma = sigma_p / sigma_o

        # Compute KGE
        kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

        return kge

    def singleregion_inputfeatures(self, start, end, means_stds):
        all_inputs = np.array([])
        for inputvar in FEATURES_FILES:
            print(inputvar)
            raw_data = np.load(os.path.join(INPUTPATH, inputvar))
            raw_data = raw_data.reshape(raw_data.shape[0], NB_CELLS) if len(raw_data.shape)>2 else raw_data
            data = raw_data[start:end, :]
            data = np.moveaxis(data, 0, -1) # (cells, timeseries)
            
            if f"{inputvar.replace('.npy','')}mean" not in means_stds.keys():
                means_stds[f"{inputvar.replace('.npy','')}mean"] = np.mean(data)
                means_stds[f"{inputvar.replace('.npy','')}std"] = np.std(data)

            data = (data - means_stds[f"{inputvar.replace('.npy','')}mean"])/means_stds[f"{inputvar.replace('.npy','')}std"]
            
            lookback_arrays = [data[:, i-LOOKBACK:i] for i in range(LOOKBACK, end-start)]
            lookback_arrays = np.array(lookback_arrays)

            f1 = lookback_arrays.reshape(-1, LOOKBACK)
            f1 = np.array([f1])
            all_inputs = np.concatenate((all_inputs,f1), axis=0) if len(all_inputs)>0 else f1

        all_inputs = np.moveaxis(all_inputs, 0, -1)
        f1 = None
        lookback_arrays = None
        data = None
        raw_data = None
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], lookback_arrays[0,:,-1].reshape(X,Y))))
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], f1[:,-1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], all_inputs[:,-1, -1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))

        return all_inputs, means_stds

    def singleregion_targetvar(self, start, end, means_stds):
        raw_data = np.load(os.path.join(INPUTPATH, TARGETVAR_FILE))
        raw_data = np.nan_to_num(raw_data)
        raw_data[raw_data < 0.0] = 0

        data = raw_data[start:end, :, :] if len(raw_data.shape)>2 else raw_data[start:end, :]
        
        if f"{TARGETVAR_FILE.replace('.npy','')}mean" not in means_stds.keys():
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"] = np.mean(data)
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] = np.std(data)

        data = (data - means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"])/means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"]

        data = data.flatten()
        raw_data = None
        #print(np.unique(np.equal(raw_data[start, :, :], data.reshape(end-start, X, Y)[0,:,:])))
        return data, means_stds

    def meanstd_inputfeatures(self, start, end, means_stds, source_path):
        for inputvar in FEATURES_FILES:
            print(inputvar)
            raw_data = np.load(os.path.join(source_path, inputvar))
            raw_data = raw_data.reshape(raw_data.shape[0], NB_CELLS) if len(raw_data.shape)>2 else raw_data
            data = raw_data[start:end, :]
            data = np.moveaxis(data, 0, -1) # (cells, timeseries)
            
            if f"{inputvar.replace('.npy','')}mean" not in means_stds.keys():
                means_stds[f"{inputvar.replace('.npy','')}mean"] = np.mean(data)
                means_stds[f"{inputvar.replace('.npy','')}std"] = np.std(data)
        return means_stds

    def meanstd_targetvar(self, start, end, means_stds, source_path):
        raw_data = np.load(os.path.join(source_path, TARGETVAR_FILE))

        raw_data = np.nan_to_num(raw_data)
        raw_data[raw_data < 0.0] = 0

        data = raw_data[start:end, :, :] if len(raw_data.shape)>2 else raw_data[start:end, :]
        
        if f"{TARGETVAR_FILE.replace('.npy','')}mean" not in means_stds.keys():
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"] = np.mean(data)
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] = np.std(data)
        return means_stds


    def multiregion_inputfeatures(self, start, end, means_stds):
        all_inputs = np.array([])
        for inputvar in FEATURES_FILES:
            var = np.array([])
            meanstd = np.array([])
            #basins = [x for x in os.listdir(INPUTPATH) if os.path.isdir(os.path.join(INPUTPATH, x)) and x in SOURCE_REGION]
            basins = ["SEINE_30x30", "DOURO_30x30"]
            for basin in basins:
                print(inputvar, basin)
                raw_data = np.load(os.path.join(INPUTPATH, basin, inputvar))
                raw_data = raw_data[:,:,:]
                data = raw_data.reshape(raw_data.shape[0], int(raw_data.shape[1]*raw_data.shape[2]))
                data = data[start:end, :]
                data = np.moveaxis(data, 0, -1) # (cells, timeseries)

                # take the original training time series to calculate mean and std
                meanstd = np.concatenate((meanstd,data), axis=0) if len(meanstd)>0 else np.array(data)
                
                # create lookback
                lookback_arrays = [data[:, i-LOOKBACK:i] for i in range(LOOKBACK, end-start)]
                lookback_arrays = np.array(lookback_arrays)

                f1 = lookback_arrays.reshape(-1, LOOKBACK)
                f1 = np.array([f1])
                var = np.concatenate((var,f1), axis=1) if len(var)>0 else f1

            # standardization based on training period of all basins
            if f"{inputvar.replace('.npy','')}mean" not in means_stds.keys():
                means_stds[f"{inputvar.replace('.npy','')}mean"] = np.mean(meanstd)
                means_stds[f"{inputvar.replace('.npy','')}std"] = np.std(meanstd)
            var = (var - means_stds[f"{inputvar.replace('.npy','')}mean"])/means_stds[f"{inputvar.replace('.npy','')}std"]
            
            all_inputs = np.concatenate((all_inputs,var), axis=0) if len(all_inputs)>0 else var

        all_inputs = np.moveaxis(all_inputs, 0, -1)

        #print(np.unique(np.equal(raw_data[LOOKBACK-1], lookback_arrays[0,:,-1].reshape(X,Y))))
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], f1[:,-1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))
        #print(np.unique(np.equal(raw_data[LOOKBACK-1], all_inputs[:,-1, -1].reshape(TRAINING_PERIOD-LOOKBACK,X,Y)[0,:,:])))

        return all_inputs, means_stds

    def multiregion_targetvar(self, start, end, means_stds):
        data = np.array([])
        #basins = [x for x in os.listdir(INPUTPATH) if os.path.isdir(os.path.join(INPUTPATH, x)) and x in SOURCE_REGION]
        basins = ["SEINE_30x30", "DOURO_30x30"]
        # get data for all basins
        for basin in basins:
            raw_data = np.load(os.path.join(INPUTPATH, basin, TARGETVAR_FILE))
            raw_data = raw_data[:,:,:]
            raw_data = np.nan_to_num(raw_data)
            raw_data[raw_data < 0.0] = 0
            raw_data = raw_data[start:end, :, :]

            # concatenate all basins
            data = np.concatenate((data,raw_data), axis=0) if len(data)>0 else raw_data
        
        # standardization
        if f"{TARGETVAR_FILE.replace('.npy','')}mean" not in means_stds.keys():
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"] = np.mean(data)
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] = np.std(data)
        data = (data - means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"])/means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"]

        data = data.flatten()

        #print(np.unique(np.equal(raw_data[start, :, :], data.reshape(end-start, X, Y)[0,:,:])))
        return data, means_stds
    
    def transferpx_targetvar(self, start, end, means_stds):
        raw_data = np.load(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", TARGETVAR_FILE))
        raw_data = np.nan_to_num(raw_data)
        raw_data[raw_data < 0.0] = 0

        data = raw_data[start:end, :, :] if len(raw_data.shape)>2 else raw_data[start:end, :]
        
        if f"{TARGETVAR_FILE.replace('.npy','')}mean" not in means_stds.keys():
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"] = np.mean(data)
            means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"] = np.std(data)

        data = (data - means_stds[f"{TARGETVAR_FILE.replace('.npy','')}mean"])/means_stds[f"{TARGETVAR_FILE.replace('.npy','')}std"]

        data = data.flatten()
        raw_data = None
        return data, means_stds

    def transferpx_inputfeatures(self, start, end, means_stds):
        all_inputs = np.array([])
        for inputvar in FEATURES_FILES:
            print(inputvar)
            raw_data = np.load(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", inputvar))
            raw_data = raw_data.reshape(raw_data.shape[0], NB_CELLS) if len(raw_data.shape)>2 else raw_data
            data = raw_data[start:end, :]
            data = np.moveaxis(data, 0, -1) # (cells, timeseries)
            if f"{inputvar.replace('.npy','')}mean" not in means_stds.keys() and f"{inputvar.replace('.npy','')}std" not in means_stds.keys():
                means_stds[f"{inputvar.replace('.npy','')}mean"] = np.mean(data)
                means_stds[f"{inputvar.replace('.npy','')}std"] = np.std(data)

            data = (data - means_stds[f"{inputvar.replace('.npy','')}mean"])/means_stds[f"{inputvar.replace('.npy','')}std"]

            lookback_arrays = [data[:, i-LOOKBACK:i] for i in range(LOOKBACK, end-start)]
            lookback_arrays = np.array(lookback_arrays)

            f1 = lookback_arrays.reshape(-1, LOOKBACK)
            f1 = np.array([f1])
            all_inputs = np.concatenate((all_inputs,f1), axis=0) if len(all_inputs)>0 else f1

        all_inputs = np.moveaxis(all_inputs, 0, -1)
        f1 = None
        lookback_arrays = None
        data = None
        raw_data = None
        #timeseries = [i for i in range(10, 365*4*100, 100)]
        #plt.plot(all_inputs[timeseries,-1,2])
        #plt.savefig("sm1mcum.png")
        return all_inputs, means_stds

    def plot_cdfs(self,
        data_dict: Dict[str, Union[list, np.ndarray]],
        colors: Optional[List[str]] = None,
        linestyles: Optional[List[str]] = None,
        title: str = "",
        xlabel: str = "Values",
        ylabel: str = "Cumulative Probability",
        logscale: bool = False,
        symlog: bool = False,
        grid: bool = True,
        figsize: tuple = (10, 6)
    ) -> plt.Figure:
        """
        Plot CDFs for multiple datasets using dictionary input.
        
        Args:
            data_dict: Dictionary where keys are labels and values are data lists
            colors: Optional list of line colors (matches dictionary order)
            linestyles: Optional list of line styles (matches dictionary order)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
            figsize: Figure size
        
        Returns:
            matplotlib Figure object
        """
        # Extract labels and data from dictionary
        labels = list(data_dict.keys())
        data_lists = list(data_dict.values())
        n = len(data_dict)
        
        # Set default styles if not provided
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, n))  # Use colormap
        if linestyles is None:
            linestyles = ['-'] * n  # Solid lines by default
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each dataset
        for (label, data), color, ls in zip(data_dict.items(), colors, linestyles):
            arr = np.array(data)
            sorted_data = np.sort(arr)
            cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            ax.plot(sorted_data, cdf, label=label, color=color, linestyle=ls, linewidth=2)
        
        # Add plot decorations
        #ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if logscale:
            ax.set_xscale('log')
        if symlog:
            ax.set_xscale('symlog')
        ax.legend()
        if grid:
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"cdf_{xlabel}_{title}.png"))
        return print(f"plotted cdfs of {xlabel}")

    def plot_pdfs(self,
        data_dict: Dict[str, Union[list, np.ndarray]],
        colors: Optional[List[str]] = None,
        linestyles: Optional[List[str]] = None,
        title: str = "",
        xlabel: str = "Values",
        ylabel: str = "Density",
        logscale: bool = False,
        symlog: bool = False,
        grid: bool = True,
        figsize: tuple = (10, 6),
        bandwidth: Optional[float] = None,
        alpha: float = 0.7,
        show_hist: bool = False,
        bins: Union[int, str] = 'auto'
    ) -> plt.Figure:
        """
        Plot PDFs for multiple datasets using dictionary input.
        
        Args:
            data_dict: Dictionary where keys are labels and values are data lists
            colors: Optional list of line colors
            linestyles: Optional list of line styles
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
            figsize: Figure size
            bandwidth: Bandwidth for KDE (None for automatic)
            alpha: Transparency for histogram (if shown)
            show_hist: Overlay histograms
            bins: Number of bins for histogram (if shown)
        
        Returns:
            matplotlib Figure object
        """
        # Extract labels and data
        labels = list(data_dict.keys())
        data_lists = list(data_dict.values())
        n = len(data_dict)
        
        # Set default styles
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, n))
        if linestyles is None:
            linestyles = ['-'] * n
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate global min/max for x-axis
        all_data = np.concatenate(list(data_dict.values()))
        x_min, x_max = np.min(all_data), np.max(all_data)
        x_vals = np.linspace(x_min, x_max, 1000)
        
        # Plot each dataset
        for (label, data), color, ls in zip(data_dict.items(), colors, linestyles):
            arr = np.array(data)
            
            # Kernel Density Estimation
            kde = gaussian_kde(arr, bw_method=bandwidth)
            ax.plot(x_vals, kde(x_vals), label=label, color=color, 
                linestyle=ls, linewidth=2)
            
            # Optional histogram
            if show_hist:
                ax.hist(arr, bins=bins, density=True, alpha=alpha, 
                    color=color, histtype='stepfilled')
        
        # Add plot decorations
        #ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if logscale:
            ax.set_xscale('log')
        if symlog:
            ax.set_xscale('symlog')
        ax.legend()
        if grid:
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTPATH, "statistics", f"pdf_{xlabel}_{title}.png"))
        return print(f"plotted pdfs of {xlabel}")

    def intersect_subsets(self, target_map, transfer_subset):
        flat_target_map = target_map.flatten()
        flat_transfer_subset = transfer_subset.flatten()
        intersecting_indices = np.where((flat_transfer_subset == 1) & (flat_target_map == 1))[0]
        target_indices = np.where(flat_target_map==1)[0]
        # get 1D list of indices of pixels in the flattened target map=1 and at the same time included in the transfer subset
        indices = np.where(np.isin(target_indices, intersecting_indices))[0]
        # get tuple of x and y indices of the pixels in the target map=1 and at the same time included in the transfer subset
        indices_2d = np.where(target_map==1)
        indices_2d = (indices_2d[0][indices], indices_2d[1][indices])
        return indices, indices_2d