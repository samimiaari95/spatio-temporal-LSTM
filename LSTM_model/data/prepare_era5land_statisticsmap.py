import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from LSTM_model.data.predict_era5land_statisticsmap import (
    ENSEMBLE_MEMBERS,
    FEATURES_FILES,
    FIRST_CONTEXT_YEAR,
    INPUTPATH,
    LOOKBACK,
    OUTPUTPATH,
    PREDICTION_YEARS,
    all_dates,
    build_lookback_features,
    stage_features,
)


LOGGER = logging.getLogger("statisticsmap")


def configure_logging(run_dir: Path) -> None:
    handlers = (
        logging.StreamHandler(),
        logging.FileHandler(run_dir / "logs" / "preprocess.log", mode="a"),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


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
    if not run_dir.is_dir() or not (run_dir / "logs").is_dir():
        raise FileNotFoundError(
            f"Run directory and logs directory must be created before submission: "
            f"{run_dir}"
        )
    feature_path = run_dir / "feature_inputs" / "era5land_features.npy"
    if feature_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite prepared features: {feature_path}"
        )
    temporal_feature_path = (
        run_dir / "feature_inputs" / "era5land_temporal_features.tmp.npy"
    )
    if temporal_feature_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporary temporal features: "
            f"{temporal_feature_path}"
        )
    metadata_dir = run_dir / "metadata"
    if (metadata_dir / "preprocessing.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite preprocessing metadata in {metadata_dir}"
        )

    configure_logging(run_dir)
    LOGGER.info(
        "CPU preprocessing started on host %s; visible GPUs: %s",
        os.uname().nodename,
        os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"),
    )
    mask = np.load(args.mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0:
        raise ValueError(f"No pixels equal to 1 in mask {args.mask_path}")
    flat_indices = np.ravel_multi_index((y_indices, x_indices), mask.shape)
    dates = all_dates(FIRST_CONTEXT_YEAR, PREDICTION_YEARS[-1])
    prediction_dates = [
        date for date in dates if date.year in PREDICTION_YEARS
    ]
    if dates[LOOKBACK:] != prediction_dates:
        raise ValueError(
            "The prepared date sequence must contain exactly LOOKBACK context "
            "days before the configured prediction dates"
        )

    LOGGER.info(
        "Preparing %d dates, %d of %d grid cells, %d predictors",
        len(dates),
        len(flat_indices),
        mask.size,
        len(FEATURES_FILES),
    )
    expected_feature_shape = (
        (len(dates) - LOOKBACK) * len(flat_indices),
        LOOKBACK,
        len(FEATURES_FILES),
    )
    windowed_bytes = (
        int(np.prod(expected_feature_shape, dtype=np.int64))
        * np.dtype(np.float32).itemsize
    )
    temporal_bytes = (
        len(FEATURES_FILES)
        * len(dates)
        * len(flat_indices)
        * np.dtype(np.float32).itemsize
    )
    prediction_bytes = (
        len(ENSEMBLE_MEMBERS)
        * len(prediction_dates)
        * len(flat_indices)
        * np.dtype(np.float32).itemsize
    )
    required_bytes = windowed_bytes + temporal_bytes + prediction_bytes
    available_bytes = shutil.disk_usage(run_dir).free
    LOGGER.info(
        "Windowed features require %.2f GiB and per-member predictions "
        "require %.2f GiB; %.2f GiB available before staging",
        windowed_bytes / (1024**3),
        prediction_bytes / (1024**3),
        available_bytes / (1024**3),
    )
    if available_bytes < required_bytes:
        raise OSError(
            "Insufficient free filesystem space for windowed features and "
            "per-member predictions plus temporary staging: need "
            f"{required_bytes / (1024**3):.2f} GiB, "
            f"have {available_bytes / (1024**3):.2f} GiB"
        )
    stage_features(
        temporal_feature_path,
        dates,
        Path(args.era5_dir),
        Path(args.input_dir),
        y_indices,
        x_indices,
        flat_indices,
        mask.shape,
        args.pixel_batch_size,
    )
    feature_shape = build_lookback_features(
        temporal_feature_path,
        feature_path,
        len(dates),
        len(flat_indices),
    )
    temporal_feature_path.unlink()
    LOGGER.info(
        "Removed temporary temporal feature stack after constructing model "
        "inputs: %s",
        temporal_feature_path,
    )

    metadata_dir.mkdir(parents=True, exist_ok=True)
    np.save(metadata_dir / "dates.npy", np.asarray(dates, dtype="datetime64[D]"))
    np.save(metadata_dir / "selected_pixels.npy", np.column_stack((y_indices, x_indices)))
    preprocessing_config = {
        "run_name": args.run_name,
        "mask_path": str(Path(args.mask_path).resolve()),
        "era5_directory": str(Path(args.era5_dir).resolve()),
        "input_directory": str(Path(args.input_dir).resolve()),
        "feature_path": str(feature_path.resolve()),
        "feature_files": FEATURES_FILES,
        "first_context_year": FIRST_CONTEXT_YEAR,
        "prediction_years": list(PREDICTION_YEARS),
        "excluded_calendar_dates": ["02-29"],
        "lookback_days": LOOKBACK,
        "mask_shape": list(mask.shape),
        "selected_pixel_count": len(flat_indices),
        "date_count": len(dates),
        "feature_layout": "time_pixel_lookback_feature",
        "feature_shape": list(feature_shape),
        "prediction_sample_count": feature_shape[0],
        "feature_sample_start_date": prediction_dates[0].isoformat(),
        "feature_sample_end_date": prediction_dates[-1].isoformat(),
        "dtype": "float32",
        "preprocessing_device": "cpu",
        "pixel_batch_size": args.pixel_batch_size,
    }
    with (metadata_dir / "preprocessing.json").open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(preprocessing_config, config_file, indent=2)
    LOGGER.info("CPU preprocessing complete; staged features at %s", feature_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare masked ERA5-Land features for statistics-map inference."
    )
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--input-dir", default=INPUTPATH)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pixel-batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pixel_batch_size < 1:
        raise ValueError("--pixel-batch-size must be positive")
    try:
        run(args)
    except Exception:
        LOGGER.exception("Statistics-map preprocessing failed")
        raise


if __name__ == "__main__":
    main()
