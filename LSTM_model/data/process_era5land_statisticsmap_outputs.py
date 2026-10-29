import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from LSTM_model.data.predict_era5land_statisticsmap import (
    FIRST_CONTEXT_YEAR,
    INPUTPATH,
    OUTPUTPATH,
    PREDICTION_YEARS,
    all_dates,
    write_final_outputs,
)


LOGGER = logging.getLogger("statisticsmap")


def configure_logging(run_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=(
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "logs" / "postprocess.log", mode="a"),
        ),
        force=True,
    )


def create_plots(run_dir: Path, dates: list[dt.date]) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = run_dir / "maps"

    for year in PREDICTION_YEARS:
        mean_map = np.load(
            maps_dir / f"wtd_prediction_map_daily_{year}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        spread_map = np.load(
            maps_dir / f"wtd_prediction_map_spread_{year}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        mean_count = np.sum(np.isfinite(mean_map), axis=0)
        spread_count = np.sum(np.isfinite(spread_map), axis=0)
        annual_mean = np.divide(
            np.nansum(mean_map, axis=0),
            mean_count,
            out=np.full(mean_map.shape[1:], np.nan, dtype=np.float32),
            where=mean_count > 0,
        )
        annual_spread = np.divide(
            np.nansum(spread_map, axis=0),
            spread_count,
            out=np.full(spread_map.shape[1:], np.nan, dtype=np.float32),
            where=spread_count > 0,
        )
        for data, title, filename, colorbar_label in (
            (
                annual_mean,
                f"Mean predicted WTD, {year}",
                f"annual_mean_wtd_{year}.png",
                "WTD",
            ),
            (
                annual_spread,
                f"Mean ensemble spread, {year}",
                f"annual_ensemble_spread_{year}.png",
                "Ensemble standard deviation",
            ),
        ):
            LOGGER.info("Creating plot %s", filename)
            figure, axis = plt.subplots(figsize=(10, 7))
            image = axis.imshow(data, origin="lower", aspect="auto")
            axis.set(
                title=title,
                xlabel="Grid x index",
                ylabel="Grid y index",
            )
            figure.colorbar(image, ax=axis, label=colorbar_label)
            figure.tight_layout()
            figure.savefig(plots_dir / filename, dpi=150)
            plt.close(figure)
        del mean_map, spread_map

    mean_predictions = np.load(
        run_dir / "predictions" / "daily_wtd_ensemble_mean.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    std_predictions = np.load(
        run_dir / "predictions" / "daily_wtd_ensemble_std.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    LOGGER.info("Creating domain-average daily prediction plot")
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(dates, np.mean(mean_predictions, axis=1), label="Mean WTD")
    axis.plot(dates, np.mean(std_predictions, axis=1), label="Mean ensemble spread")
    axis.set(
        title="Daily statistics-map prediction across selected pixels",
        xlabel="Date",
        ylabel="WTD / ensemble standard deviation",
    )
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(plots_dir / "daily_prediction_summary.png", dpi=150)
    plt.close(figure)
    LOGGER.info("Plots saved under %s", plots_dir)


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    allowed_roots = {
        (Path(OUTPUTPATH) / "statisticsEUmap").resolve(),
        (Path(INPUTPATH) / "statisticsEUmap").resolve(),
    }
    if output_root not in allowed_roots:
        raise ValueError(
            "The output root must be either the inputs or outputs "
            "EU_74_-48_69_20/statisticsEUmap directory"
        )
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        raise ValueError("--run-name must be a single directory name")
    run_dir = output_root / args.run_name
    metadata_dir = run_dir / "metadata"
    preprocessing_path = metadata_dir / "preprocessing.json"
    prediction_config_path = metadata_dir / "prediction_config.json"
    if not preprocessing_path.is_file() or not prediction_config_path.is_file():
        raise FileNotFoundError(
            "CPU preprocessing and GPU prediction must both complete before "
            f"postprocessing; expected metadata in {metadata_dir}"
        )
    existing_outputs = (
        run_dir / "maps",
        run_dir / "summary",
        run_dir / "plots",
        run_dir / "predictions" / "daily_wtd_ensemble_mean.npy",
        run_dir / "predictions" / "daily_wtd_ensemble_std.npy",
    )
    existing = [path for path in existing_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            f"Postprocessing outputs already exist ({', '.join(map(str, existing))}); "
            "use a new run directory rather than overwriting them"
        )
    configure_logging(run_dir)
    LOGGER.info(
        "CPU output processing started; CUDA_VISIBLE_DEVICES=%s",
        os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"),
    )
    with preprocessing_path.open(encoding="utf-8") as file:
        preprocessing = json.load(file)
    with prediction_config_path.open(encoding="utf-8") as file:
        prediction_config = json.load(file)

    mask = np.load(preprocessing["mask_path"], allow_pickle=False)
    if list(mask.shape) != preprocessing["mask_shape"]:
        raise ValueError("Mask shape does not match preprocessing metadata")
    y_indices, x_indices = np.where(mask == 1)
    flat_indices = np.ravel_multi_index((y_indices, x_indices), mask.shape)
    selected_pixels = np.load(
        metadata_dir / "selected_pixels.npy", allow_pickle=False
    )
    if not np.array_equal(selected_pixels, np.column_stack((y_indices, x_indices))):
        raise ValueError("Selected pixels do not match the preprocessing mask")

    dates = [
        dt.date.fromisoformat(str(value))
        for value in np.load(metadata_dir / "dates.npy", allow_pickle=False)
        .astype("datetime64[D]")
        .astype(str)
    ]
    expected_dates = all_dates(FIRST_CONTEXT_YEAR, PREDICTION_YEARS[-1])
    prediction_dates = [
        date for date in dates if date.year in PREDICTION_YEARS
    ]
    if dates != expected_dates:
        raise ValueError("Prepared date sequence does not match the configured years")
    if prediction_config["selected_pixel_count"] != len(flat_indices):
        raise ValueError("Prediction metadata pixel count does not match the mask")

    member_predictions_path = Path(prediction_config["prediction_output"])
    expected_prediction_shape = (
        len(prediction_config["members"]),
        len(prediction_dates),
        len(flat_indices),
    )
    member_predictions = np.load(
        member_predictions_path, mmap_mode="r", allow_pickle=False
    )
    if member_predictions.shape != expected_prediction_shape:
        raise ValueError(
            f"Prediction array has shape {member_predictions.shape}; expected "
            f"{expected_prediction_shape}"
        )

    args.mask_path = preprocessing["mask_path"]
    args.era5_dir = preprocessing["era5_directory"]
    args.input_dir = preprocessing["input_directory"]
    args.members = prediction_config["members"]
    args.world_size = prediction_config["world_size"]
    args.pixel_batch_size = prediction_config["pixel_batch_size"]
    args.time_batch_size = prediction_config["time_batch_size"]
    devices = prediction_config["devices"]
    device = devices.get("0", {}).get("device", "unknown")

    LOGGER.info("Combining %d ensemble members", len(args.members))
    write_final_outputs(
        args,
        run_dir,
        mask,
        y_indices,
        x_indices,
        flat_indices,
        prediction_dates,
        member_predictions_path,
        device,
    )
    create_plots(run_dir, prediction_dates)
    LOGGER.info("CPU output processing and plotting completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate statistics-map predictions and create CPU plots."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception:
        LOGGER.exception("Statistics-map output processing failed")
        raise


if __name__ == "__main__":
    main()
