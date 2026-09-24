from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

from LSTM_model.model.config import INPUTPATH, OUTPUTPATH, MODEL_NAME
from LSTM_model.utils.utils import utilities


@dataclass(frozen=True)
class SpatiallyBlockedCVInfo:
    groups: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    residuals: np.ndarray
    variogram_range_m: float
    block_size_m: float
    variogram_plot_path: str


def _project_latlon_to_meters(latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    lat0 = np.deg2rad(np.mean(latitudes))
    earth_radius_m = 6371000.0
    x = earth_radius_m * lon_rad * np.cos(lat0)
    y = earth_radius_m * lat_rad
    return x, y


def _spherical_variogram(h: np.ndarray, nugget: float, sill: float, range_m: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    gamma = np.full_like(h, sill, dtype=float)
    if range_m <= 0:
        return gamma

    within_range = h <= range_m
    ratio = h[within_range] / range_m
    gamma[within_range] = nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio**3)
    return gamma


def _haversine_distances(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(latitudes)
    lon_rad = np.deg2rad(longitudes)
    coords = np.column_stack((lat_rad, lon_rad))

    def haversine(u: np.ndarray, v: np.ndarray) -> float:
        dlat = v[0] - u[0]
        dlon = v[1] - u[1]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(u[0]) * np.cos(v[0]) * np.sin(dlon / 2.0) ** 2
        return 2.0 * 6371000.0 * np.arcsin(np.sqrt(a))

    return pdist(coords, metric=haversine)


def _load_transfer_pixel_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    utils = utilities()
    statistics_path = os.path.join(OUTPUTPATH, "statistics", "ensemble_statistics_mad.csv")
    statistics = pd.read_csv(statistics_path)
    residuals = np.abs(statistics["Absolute mean bias"].to_numpy(dtype=float))

    transfer_subset = np.load(os.path.join(os.path.dirname(INPUTPATH), "target_pixels", "transfer_subset.npy"))
    lat2d = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "lat2D.npy"))
    lon2d = np.load(os.path.join(os.path.dirname(os.path.dirname(INPUTPATH)), "lon2D.npy"))

    eu_inpath = os.path.join(os.path.dirname(INPUTPATH), "ensemble_mean")
    eu_outpath = os.path.join(os.path.dirname(OUTPUTPATH), "ensemble_mean")

    latitudes = []
    longitudes = []
    residual_index = 0

    for target in range(100):
        target_map = np.load(os.path.join(os.path.dirname(eu_inpath), f"target_pixels_{target}", f"mappingindices_{target}.npy"))
        transfer_indices, indices_2d = utils.intersect_subsets(target_map, transfer_subset)

        obs = np.load(os.path.join(os.path.dirname(eu_outpath), "400px_member_1", f"obs_destand_{MODEL_NAME}_{target}.npy"))
        obs = obs[:, transfer_indices]
        members_sim = [np.load(os.path.join(os.path.dirname(eu_outpath), f"400px_member_{member}", f"sim_destand_{MODEL_NAME}_{target}.npy")) for member in range(100)]
        members_sim = np.asarray(members_sim)
        members_sim = np.expand_dims(members_sim, axis=0)
        members_sim = np.concatenate((members_sim), axis=0)
        members_sim = members_sim[:, :, transfer_indices]

        obs[obs < 0.0] = 0.0
        members_sim[members_sim < 0.0] = 0.0

        for pixel in range(members_sim.shape[2]):
            ensemble_predictions = members_sim[:, :, pixel]
            ensemble_predictions = np.moveaxis(ensemble_predictions, 0, -1)
            mean_prediction = np.mean(ensemble_predictions, axis=1)

            if np.std(obs[:, pixel]) == 0.0 or np.std(mean_prediction) == 0.0 or np.isnan(obs[:, pixel]).all() or np.isnan(mean_prediction).all():
                continue

            if residual_index >= len(residuals):
                raise ValueError("Residual table ended before the pixel reconstruction loop completed.")

            row, col = indices_2d[0][pixel], indices_2d[1][pixel]
            latitudes.append(lat2d[row, col])
            longitudes.append(lon2d[row, col])
            residual_index += 1

    if residual_index != len(residuals):
        raise ValueError(
            f"Residual count mismatch: reconstructed {residual_index} pixels, but CSV contains {len(residuals)} rows."
        )

    return np.asarray(latitudes), np.asarray(longitudes), residuals


def _fit_variogram_and_group_blocks(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    residuals: np.ndarray,
    block_size_multiplier: float,
    cv_folds: int,
) -> SpatiallyBlockedCVInfo:
    x_m, y_m = _project_latlon_to_meters(latitudes, longitudes)
    pairwise_distances = _haversine_distances(latitudes, longitudes)
    pairwise_semivariance = 0.5 * pdist(residuals.reshape(-1, 1), metric="sqeuclidean")

    if len(pairwise_distances) == 0:
        raise ValueError("Not enough pixels to estimate a variogram.")

    max_distance = (1/3) * float(np.max(pairwise_distances)) # calculating the max lag
    if not np.isfinite(max_distance) or max_distance <= 0:
        max_distance = float(np.max(pairwise_distances))
    print(f"Max distance for variogram estimation: {max_distance:.2f} m")

    n_bins = 200
    bin_edges = np.linspace(0.0, max_distance, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    semivariance_bins = np.full(n_bins, np.nan)
    pair_counts = np.zeros(n_bins, dtype=int)

    for index in range(n_bins):
        if index < n_bins - 1:
            bin_mask = (pairwise_distances >= bin_edges[index]) & (pairwise_distances < bin_edges[index + 1])
        else:
            bin_mask = (pairwise_distances >= bin_edges[index]) & (pairwise_distances <= bin_edges[index + 1])

        pair_counts[index] = int(np.sum(bin_mask))
        if pair_counts[index] > 0:
            print(f"Bin {index + 1}: {pair_counts[index]} pairs, distance range [{bin_edges[index]:.2f}, {bin_edges[index + 1]:.2f}] m")
            semivariance_bins[index] = float(np.mean(pairwise_semivariance[bin_mask]))
    print(f"pair_counts: {pair_counts}")
    print(f"semivariance_bins: {semivariance_bins}")
    valid_bins = np.isfinite(semivariance_bins) & (pair_counts >= 30)
    if np.count_nonzero(valid_bins) >= 3:
        x_fit = bin_centers[valid_bins]
        y_fit = semivariance_bins[valid_bins]

        nugget_guess = float(np.nanmin(y_fit))
        sill_guess = float(np.nanmax(y_fit))
        ########## here provide the visually inspected range in m
        range_guess = float(np.nanmedian(x_fit)) if np.nanmedian(x_fit) > 0 else float(np.nanmax(x_fit))
        print(f"Initial variogram parameter guesses: nugget={nugget_guess:.4f}, sill={sill_guess:.4f}, range={range_guess:.2f} m")

        try:
            params, _ = curve_fit(
                _spherical_variogram,
                x_fit,
                y_fit,
                p0=(nugget_guess, sill_guess, range_guess),
                bounds=((0.0, 0.0, 1.0), (np.inf, np.inf, np.inf)),
                maxfev=20000,
            )
            nugget, sill, variogram_range_m = map(float, params)
        except Exception:
            nugget = nugget_guess
            sill = sill_guess
            variogram_range_m = range_guess
    else:
        nugget = float(np.nanmin(semivariance_bins[np.isfinite(semivariance_bins)]))
        sill = float(np.nanmax(semivariance_bins[np.isfinite(semivariance_bins)]))
        variogram_range_m = float(np.nanmax(bin_centers))
        print("Not enough valid bins for variogram fitting. Using default parameters.#############################")
        print(f"Default variogram parameters: nugget={nugget:.4f}, sill={sill:.4f}, range={variogram_range_m:.2f} m")

    if not np.isfinite(variogram_range_m) or variogram_range_m <= 0:
        variogram_range_m = float(np.nanmax(pairwise_distances))

    print(f"Final variogram parameters: nugget={nugget:.4f}, sill={sill:.4f}, range={variogram_range_m:.2f} m")
    block_size_m = max(variogram_range_m * float(block_size_multiplier), 1.0)

    x_blocks = np.floor((x_m - np.nanmin(x_m)) / block_size_m).astype(int)
    y_blocks = np.floor((y_m - np.nanmin(y_m)) / block_size_m).astype(int)
    block_pairs = np.column_stack((x_blocks, y_blocks))
    _, groups = np.unique(block_pairs, axis=0, return_inverse=True)

    unique_groups = np.unique(groups)
    while len(unique_groups) < cv_folds and block_size_m > 1.0:
        block_size_m *= 0.5
        x_blocks = np.floor((x_m - np.nanmin(x_m)) / block_size_m).astype(int)
        y_blocks = np.floor((y_m - np.nanmin(y_m)) / block_size_m).astype(int)
        block_pairs = np.column_stack((x_blocks, y_blocks))
        _, groups = np.unique(block_pairs, axis=0, return_inverse=True)
        unique_groups = np.unique(groups)

    plot_directory = os.path.join(OUTPUTPATH, "statistics")
    os.makedirs(plot_directory, exist_ok=True)
    plot_path = os.path.join(plot_directory, "spatially_blocked_cv_variogram.png")

    plt.figure(figsize=(6.0, 4.0))
    plt.scatter(bin_centers, semivariance_bins, s=25, color="k", label="Empirical variogram")
    print(f"Fitted variogram parameters: nugget={nugget:.4f}, sill={sill:.4f}, range={variogram_range_m:.2f} m")
    print(f"Block size for spatially blocked CV: {block_size_m:.2f} m")
    print(f"Number of unique spatial blocks: {len(unique_groups)}")
    print(bin_centers)
    print(semivariance_bins)
    if np.isfinite(variogram_range_m):
        model_x = np.linspace(0.0, max(bin_centers[-1], variogram_range_m), 200)
        model_y = _spherical_variogram(model_x, nugget, sill, variogram_range_m)
        #plt.plot(model_x, model_y, "r--", label=f"Spherical fit (range={variogram_range_m/1000.0:.1f} km)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Semivariance")
    plt.title("Spatially blocked CV variogram")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return SpatiallyBlockedCVInfo(
        groups=groups,
        latitudes=latitudes,
        longitudes=longitudes,
        residuals=residuals,
        variogram_range_m=variogram_range_m,
        block_size_m=block_size_m,
        variogram_plot_path=plot_path,
    )


@lru_cache(maxsize=1)
def get_spatially_blocked_cv_info(block_size_multiplier: float = 1.0, cv_folds: int = 5) -> SpatiallyBlockedCVInfo:
    latitudes, longitudes, residuals = _load_transfer_pixel_table()
    return _fit_variogram_and_group_blocks(latitudes, longitudes, residuals, block_size_multiplier, cv_folds)


@lru_cache(maxsize=1)
def get_spatially_blocked_groups(block_size_multiplier: float = 1.0, cv_folds: int = 5) -> np.ndarray:
    return get_spatially_blocked_cv_info(block_size_multiplier=block_size_multiplier, cv_folds=cv_folds).groups