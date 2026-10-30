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
    ENSEMBLE_MEMBERS,
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
    if not preprocessing_path.is_file():
        raise FileNotFoundError(
            f"CPU preprocessing metadata is missing: {preprocessing_path}"
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
    member_predictions_path = None
    member_prediction_paths = None
    if prediction_config_path.is_file():
        with prediction_config_path.open(encoding="utf-8") as file:
            prediction_config = json.load(file)
        member_prediction_outputs = prediction_config.get(
            "member_prediction_outputs"
        )
        if member_prediction_outputs is None:
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
                    f"Prediction array has shape {member_predictions.shape}; "
                    f"expected {expected_prediction_shape}"
                )
        else:
            member_predictions_path = None
            members = prediction_config["members"]
            if sorted(map(int, member_prediction_outputs)) != members:
                raise ValueError(
                    "Per-member prediction files do not match the configured "
                    "ensemble member list"
                )
            member_prediction_paths = [
                Path(member_prediction_outputs[str(member)]) for member in members
            ]
    else:
        members = list(ENSEMBLE_MEMBERS)
        member_prediction_outputs = {}
        member_manifests = []
        missing_members = []
        for member in members:
            manifest_path = metadata_dir / f"member_{member:03d}_prediction.json"
            if not manifest_path.is_file():
                missing_members.append(member)
                continue
            with manifest_path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            expected_output_path = (
                run_dir / "predictions" / f"daily_wtd_member_{member:03d}.npy"
            ).resolve()
            output_path = Path(manifest["prediction_output"]).resolve()
            if manifest.get("run_name") != args.run_name or manifest.get(
                "member"
            ) != member:
                raise ValueError(
                    f"Prediction manifest does not match run/member {args.run_name}/{member}: "
                    f"{manifest_path}"
                )
            if output_path != expected_output_path or not output_path.is_file():
                raise FileNotFoundError(
                    f"Prediction output is missing or unexpected for member "
                    f"{member}: {output_path}"
                )
            if (
                manifest.get("prediction_date_count") != len(prediction_dates)
                or manifest.get("selected_pixel_count") != len(flat_indices)
            ):
                raise ValueError(
                    f"Prediction dimensions in manifest do not match the "
                    f"prepared run for member {member}"
                )
            member_array = np.load(
                output_path, mmap_mode="r", allow_pickle=False
            )
            expected_member_shape = (
                1,
                len(prediction_dates),
                len(flat_indices),
            )
            if member_array.shape != expected_member_shape:
                raise ValueError(
                    f"Member {member} prediction array has shape "
                    f"{member_array.shape}; expected {expected_member_shape}"
                )
            if member_array.dtype != np.float32:
                raise ValueError(
                    f"Member {member} prediction array has dtype "
                    f"{member_array.dtype}; expected float32"
                )
            member_prediction_outputs[str(member)] = str(output_path)
            member_manifests.append(manifest)
        if missing_members:
            raise FileNotFoundError(
                "Prediction jobs have not completed for ensemble members: "
                + ", ".join(map(str, missing_members))
            )
        first_manifest = member_manifests[0]
        for manifest in member_manifests:
            if any(
                manifest.get(field) != first_manifest.get(field)
                for field in (
                    "model_name",
                    "pixel_batch_size",
                    "time_batch_size",
                    "world_size",
                )
            ):
                raise ValueError(
                    "Per-member prediction settings are inconsistent across "
                    "the ensemble"
                )
        prediction_config = {
            "run_name": args.run_name,
            "model_name": first_manifest["model_name"],
            "members": members,
            "prediction_date_count": len(prediction_dates),
            "selected_pixel_count": len(flat_indices),
            "world_size": 1,
            "devices": {
                "0": {
                    "device": first_manifest["device"],
                    "name": first_manifest["device_name"],
                }
            },
            "member_devices": {
                str(manifest["member"]): manifest["device_name"]
                for manifest in member_manifests
            },
            "pixel_batch_size": first_manifest["pixel_batch_size"],
            "time_batch_size": first_manifest["time_batch_size"],
            "prediction_output": None,
            "member_prediction_outputs": member_prediction_outputs,
        }
        temporary_config_path = prediction_config_path.with_suffix(
            prediction_config_path.suffix + f".partial-{os.getpid()}"
        )
        with temporary_config_path.open("w", encoding="utf-8") as file:
            json.dump(prediction_config, file, indent=2)
        os.replace(temporary_config_path, prediction_config_path)
        member_prediction_paths = [
            Path(member_prediction_outputs[str(member)]) for member in members
        ]

    if prediction_config["selected_pixel_count"] != len(flat_indices):
        raise ValueError("Prediction metadata pixel count does not match the mask")

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
        member_prediction_paths,
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
