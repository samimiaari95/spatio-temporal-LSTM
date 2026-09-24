from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from LSTM_model.model.config import INPUTPATH, OUTPUTPATH, MODEL_NAME
from LSTM_model.utils.utils import utilities


@dataclass(frozen=True)
class EmpiricalCorrelogramInfo:
    bin_centers_m: np.ndarray
    correlogram: np.ndarray
    pair_counts: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    residuals: np.ndarray
    plot_path: str


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


def compute_empirical_correlogram(max_distance_m: float, n_bins: int = 30) -> EmpiricalCorrelogramInfo:
    if max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    latitudes, longitudes, residuals = _load_transfer_pixel_table()

    distances = _haversine_distances(latitudes, longitudes)
    centered = residuals - np.mean(residuals)
    std = np.std(centered)
    if std == 0.0:
        raise ValueError("Residuals have zero variance, correlogram cannot be computed.")
    standardized = centered / std
    pairwise_products = pdist(standardized.reshape(-1, 1), metric=lambda u, v: float(u[0] * v[0]))

    bin_edges = np.linspace(0.0, max_distance_m, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    correlogram = np.full(n_bins, np.nan)
    pair_counts = np.zeros(n_bins, dtype=int)

    for index in range(n_bins):
        if index < n_bins - 1:
            bin_mask = (distances >= bin_edges[index]) & (distances < bin_edges[index + 1])
        else:
            bin_mask = (distances >= bin_edges[index]) & (distances <= bin_edges[index + 1])

        pair_counts[index] = int(np.sum(bin_mask))
        if pair_counts[index] > 0:
            correlogram[index] = float(np.mean(pairwise_products[bin_mask]))

    plot_directory = os.path.join(OUTPUTPATH, "statistics")
    os.makedirs(plot_directory, exist_ok=True)
    plot_path = os.path.join(plot_directory, "spatially_blocked_cv_correlogram.png")

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.axhline(0.0, color="0.45", linestyle="--", linewidth=1)
    ax1.plot(bin_centers, correlogram, marker="o", color="black", linewidth=1.5, label="Empirical correlogram")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Correlogram")
    ax1.set_title("Spatial correlogram for manual range inspection")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_ylim(-1.05, 1.05)

    ax2 = ax1.twinx()
    ax2.bar(bin_centers, pair_counts, width=(bin_edges[1] - bin_edges[0]) * 0.9, color="tab:blue", alpha=0.12, label="Pair count")
    ax2.set_ylabel("Pair count")
    ax2.set_ylim(0, max(1, int(np.nanmax(pair_counts) * 1.1)))

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    return EmpiricalCorrelogramInfo(
        bin_centers_m=bin_centers,
        correlogram=correlogram,
        pair_counts=pair_counts,
        latitudes=latitudes,
        longitudes=longitudes,
        residuals=residuals,
        plot_path=plot_path,
    )
