import argparse
import calendar
import csv
import datetime as dt
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from LSTM_model.model.config import (
    BATCH_SIZE,
    DROPOUT,
    FEATURES_FILES,
    HIDDEN_SIZE,
    INPUTPATH,
    INPUT_SIZE,
    LOOKBACK,
    MODEL_NAME,
    NUM_LAYERS,
    OUTPUTPATH,
    OUTPUT_SIZE,
    AwesomeLSTM,
)


LOGGER = logging.getLogger("statisticsmap")
FIRST_CONTEXT_YEAR = 2016
PREDICTION_YEARS = (2017, 2018, 2019, 2020)
FEATURE_SCALER_PREFIXES = (
    "TOT_PREC",
    "vpd",
    "soilmoisture",
    "slopex",
    "slopey",
    "soilind",
    "lon2D_ts",
    "lat2D_ts",
)
DYNAMIC_VARIABLES = ("tp", "vpd", "swvl3")


class NetCDFVariable(Protocol):
    ndim: int
    shape: Tuple[int, ...]

    def __getitem__(self, key: Tuple[object, ...]) -> np.ndarray:
        ...


def parse_members(value: str) -> List[int]:
    members = []
    for part in value.split(","):
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(
                    f"Invalid member range {part!r}: end must not precede start"
                )
            members.extend(range(start, end + 1))
        else:
            members.append(int(part))

    if not members or min(members) < 0 or len(set(members)) != len(members):
        raise argparse.ArgumentTypeError(
            "Members must be a non-empty list of unique, non-negative integers"
        )
    return members


def month_dates(year: int, month: int) -> List[dt.date]:
    days_in_month = calendar.monthrange(year, month)[1]
    return [dt.date(year, month, day) for day in range(1, days_in_month + 1)]


def all_dates(start_year: int, end_year: int) -> List[dt.date]:
    return [
        dt.date(year, month, day)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
        if not (month == 2 and day == 29)
    ]


def _select_from_array(
    array: np.ndarray,
    flat_indices: np.ndarray,
    mask_shape: Tuple[int, int],
    variable: str,
) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError(
            f"{variable} must have a time dimension and spatial cells; got {array.shape}"
        )
    if array.ndim == 2:
        if array.shape[1] != int(np.prod(mask_shape)):
            raise ValueError(
                f"{variable} has {array.shape[1]} cells, but the mask has "
                f"{int(np.prod(mask_shape))}"
            )
        selected = array[:, flat_indices]
    else:
        if tuple(array.shape[1:]) != mask_shape:
            raise ValueError(
                f"{variable} spatial shape {array.shape[1:]} does not match "
                f"mask shape {mask_shape}"
            )
        selected = array.reshape(array.shape[0], -1)[:, flat_indices]
    return np.nan_to_num(np.asarray(selected, dtype=np.float32), copy=True)


def _npy_month(
    era5_dir: Path,
    variable: str,
    year: int,
    month: int,
    flat_indices: np.ndarray,
    mask_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    filename = era5_dir / "npy_regridded_to_EUR11" / f"{variable}_{year}{month:02d}_EU.npy"
    if not filename.is_file():
        return None

    array = np.load(filename, mmap_mode="r")
    expected_days = calendar.monthrange(year, month)[1]
    if array.shape[0] != expected_days:
        LOGGER.info(
            "%s has %d days; reading %04d-%02d from the regridded NetCDF source",
            filename,
            array.shape[0],
            year,
            month,
        )
        return None
    return _select_from_array(array, flat_indices, mask_shape, variable)


def _select_from_netcdf(
    variable: NetCDFVariable,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    flat_indices: np.ndarray,
    mask_shape: Tuple[int, int],
    name: str,
) -> np.ndarray:
    if variable.ndim == 2:
        if variable.shape[1] != int(np.prod(mask_shape)):
            raise ValueError(
                f"{name} has {variable.shape[1]} cells, but the mask has "
                f"{int(np.prod(mask_shape))}"
            )
        selected = variable[:, flat_indices]
    elif variable.ndim == 3:
        if tuple(variable.shape[1:]) != mask_shape:
            raise ValueError(
                f"{name} spatial shape {variable.shape[1:]} does not match "
                f"mask shape {mask_shape}"
            )
        selected = np.ma.stack(
            [
                variable[:, int(y_index), int(x_index)]
                for y_index, x_index in zip(y_indices, x_indices)
            ],
            axis=1,
        )
    else:
        raise ValueError(f"{name} must be 2D or 3D; got {variable.shape}")
    return np.nan_to_num(
        np.asarray(np.ma.filled(selected, 0), dtype=np.float32), copy=True
    )


def _netcdf_month(
    era5_dir: Path,
    variable: str,
    year: int,
    month: int,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    flat_indices: np.ndarray,
    mask_shape: Tuple[int, int],
) -> np.ndarray:
    try:
        import netCDF4
    except ImportError as error:
        raise RuntimeError(
            "netCDF4 is required when the cached NPY month is missing "
            "calendar days"
        ) from error

    month_text = f"{month:02d}"
    if variable == "tp":
        path = era5_dir / "regridded_to_EUR11" / (
            f"BonA_conservative_at23h_ERA5land_tp_{year}{month_text}.nc"
        )
        with netCDF4.Dataset(path) as dataset:
            selected = _select_from_netcdf(
                dataset.variables["tp"],
                y_indices,
                x_indices,
                flat_indices,
                mask_shape,
                "tp",
            )
        return selected * np.float32(1000.0)

    if variable == "swvl3":
        path = era5_dir / "regridded_to_EUR11" / (
            f"BonA_nn_ERA5land_swvl3_{year}{month_text}.nc"
        )
        with netCDF4.Dataset(path) as dataset:
            return _select_from_netcdf(
                dataset.variables["swvl3"],
                y_indices,
                x_indices,
                flat_indices,
                mask_shape,
                "swvl3",
            )

    if variable == "vpd":
        directory = era5_dir / "regridded_to_EUR11"
        t_path = directory / f"BonA_nn_ERA5land_t2m_{year}{month_text}.nc"
        d_path = directory / f"BonA_nn_ERA5land_d2m_{year}{month_text}.nc"
        with netCDF4.Dataset(t_path) as t_dataset, netCDF4.Dataset(d_path) as d_dataset:
            t2m = _select_from_netcdf(
                t_dataset.variables["t2m"],
                y_indices,
                x_indices,
                flat_indices,
                mask_shape,
                "t2m",
            ) - np.float32(273.15)
            d2m = _select_from_netcdf(
                d_dataset.variables["d2m"],
                y_indices,
                x_indices,
                flat_indices,
                mask_shape,
                "d2m",
            ) - np.float32(273.15)
        es = np.float32(0.611) * np.exp(
            (np.float32(17.27) * t2m) / (t2m + np.float32(237.3))
        )
        ea = np.float32(0.611) * np.exp(
            (np.float32(17.27) * d2m) / (d2m + np.float32(237.3))
        )
        return np.nan_to_num(es - ea, copy=False)

    raise ValueError(f"Unsupported ERA5-Land variable: {variable}")


def load_dynamic_month(
    era5_dir: Path,
    variable: str,
    year: int,
    month: int,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    flat_indices: np.ndarray,
    mask_shape: Tuple[int, int],
) -> np.ndarray:
    data = _npy_month(
        era5_dir, variable, year, month, flat_indices, mask_shape
    )
    if data is None:
        data = _netcdf_month(
            era5_dir,
            variable,
            year,
            month,
            y_indices,
            x_indices,
            flat_indices,
            mask_shape,
        )

    expected_days = calendar.monthrange(year, month)[1]
    if data.shape[0] != expected_days:
        raise ValueError(
            f"{variable} {year}-{month:02d} has {data.shape[0]} days; "
            f"expected {expected_days}"
        )
    return data


def load_static_feature(
    input_dir: Path,
    filename: str,
    flat_indices: np.ndarray,
    mask_shape: Tuple[int, int],
) -> np.ndarray:
    path = input_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required static predictor is missing: {path}")

    data = np.load(path, mmap_mode="r")
    if data.ndim == 1 and data.size == int(np.prod(mask_shape)):
        selected = data[flat_indices]
    elif data.ndim == 2 and tuple(data.shape) == mask_shape:
        selected = data.reshape(-1)[flat_indices]
    elif data.ndim > 2:
        if tuple(data.shape[1:]) != mask_shape:
            raise ValueError(
                f"{path} spatial shape {data.shape[1:]} does not match mask "
                f"shape {mask_shape}"
            )
        selected = data[0].reshape(-1)[flat_indices]
    else:
        raise ValueError(
            f"{path} must contain one value per mask cell; got shape {data.shape}"
        )
    return np.nan_to_num(np.asarray(selected, dtype=np.float32), copy=True)


def stage_features(
    feature_path: Path,
    dates: Sequence[dt.date],
    era5_dir: Path,
    input_dir: Path,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    flat_indices: np.ndarray,
    mask_shape: Tuple[int, int],
    pixel_batch_size: int,
) -> np.ndarray:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_path.exists():
        raise FileExistsError(f"Refusing to overwrite staged features: {feature_path}")

    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(FEATURES_FILES), len(dates), len(flat_indices)),
    )
    date_offsets = {date: index for index, date in enumerate(dates)}
    for variable in DYNAMIC_VARIABLES:
        feature_index = DYNAMIC_VARIABLES.index(variable)
        for year in range(FIRST_CONTEXT_YEAR, PREDICTION_YEARS[-1] + 1):
            for month in range(1, 13):
                month_days = month_dates(year, month)
                kept_day_indices = [
                    day_index
                    for day_index, date in enumerate(month_days)
                    if date in date_offsets
                ]
                start = date_offsets[month_days[kept_day_indices[0]]]
                end = start + len(kept_day_indices)
                for pixel_start in range(0, len(flat_indices), pixel_batch_size):
                    pixel_end = min(pixel_start + pixel_batch_size, len(flat_indices))
                    monthly_data = load_dynamic_month(
                        era5_dir,
                        variable,
                        year,
                        month,
                        y_indices[pixel_start:pixel_end],
                        x_indices[pixel_start:pixel_end],
                        flat_indices[pixel_start:pixel_end],
                        mask_shape,
                    )
                    features[
                        feature_index, start:end, pixel_start:pixel_end
                    ] = monthly_data[kept_day_indices]

    for feature_index, filename in enumerate(FEATURES_FILES[3:], start=3):
        for pixel_start in range(0, len(flat_indices), pixel_batch_size):
            pixel_end = min(pixel_start + pixel_batch_size, len(flat_indices))
            selected = load_static_feature(
                input_dir,
                filename,
                flat_indices[pixel_start:pixel_end],
                mask_shape,
            )
            features[feature_index, :, pixel_start:pixel_end] = selected[None, :]

    features.flush()
    return features


def load_member_scalers(member: int) -> Dict[str, object]:
    member_dir = Path(OUTPUTPATH) / "validation_ERA5" / "ensemble_400px" / f"400px_member_{member}"
    scaler_path = member_dir / f"meanstd_400px_member_{member}_{MODEL_NAME}.pkl"
    with scaler_path.open("rb") as scaler_file:
        scalers = pickle.load(scaler_file)

    for prefix in (*FEATURE_SCALER_PREFIXES, "wtd"):
        mean_key = f"{prefix}mean"
        std_key = f"{prefix}std"
        if mean_key not in scalers or std_key not in scalers:
            raise KeyError(
                f"{scaler_path} is missing required scaler keys "
                f"{mean_key!r} and/or {std_key!r}"
            )
        mean = float(np.asarray(scalers[mean_key]))
        std = float(np.asarray(scalers[std_key]))
        if not np.isfinite(mean):
            raise ValueError(f"Invalid mean {mean_key} in {scaler_path}")
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Invalid standard deviation {std_key} in {scaler_path}")
    return scalers


def load_member_model(member: int, device: torch.device) -> torch.nn.Module:
    member_dir = Path(OUTPUTPATH) / "validation_ERA5" / "ensemble_400px" / f"400px_member_{member}"
    model_path = member_dir / f"400px_member_{member}_{MODEL_NAME}.pt"
    model = AwesomeLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, DROPOUT)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_year_batch(
    feature_block: np.ndarray,
    scaler: Dict[str, object],
    model: torch.nn.Module,
    device: torch.device,
    prediction_days: int,
    context_days: int,
    time_batch_size: int,
) -> np.ndarray:
    pixel_count = feature_block.shape[2]
    prediction_start = context_days - LOOKBACK
    if prediction_start < 0:
        raise ValueError(
            f"Context has {context_days} days, fewer than LOOKBACK={LOOKBACK}"
        )
    windows_by_feature = []
    for feature_index, prefix in enumerate(FEATURE_SCALER_PREFIXES):
        feature_data = feature_block[feature_index]
        mean = np.float32(np.asarray(scaler[f"{prefix}mean"]))
        std = np.float32(np.asarray(scaler[f"{prefix}std"]))
        standardized = (feature_data - mean) / std
        windows_by_feature.append(
            np.lib.stride_tricks.sliding_window_view(
                standardized, LOOKBACK, axis=0
            )
        )

    result = np.empty((prediction_days, pixel_count), dtype=np.float32)
    for day_start in range(0, prediction_days, time_batch_size):
        day_end = min(day_start + time_batch_size, prediction_days)
        window_start = prediction_start + day_start
        window_end = prediction_start + day_end
        input_block = np.stack(
            [windows[window_start:window_end] for windows in windows_by_feature],
            axis=-1,
        )
        input_block = input_block.reshape(
            -1, LOOKBACK, len(FEATURES_FILES)
        )
        predictions = []
        with torch.inference_mode():
            for sample_start in range(0, len(input_block), BATCH_SIZE):
                input_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        input_block[sample_start : sample_start + BATCH_SIZE]
                    )
                ).to(device)
                predictions.append(model(input_tensor).reshape(-1).cpu().numpy())

        standardized_predictions = np.concatenate(predictions).reshape(
            day_end - day_start, pixel_count
        )
        if not np.isfinite(standardized_predictions).all():
            raise ValueError("The ensemble member produced non-finite predictions")
        mean = np.float32(np.asarray(scaler["wtdmean"]))
        std = np.float32(np.asarray(scaler["wtdstd"]))
        result[day_start:day_end] = standardized_predictions * std + mean
    return result


def validate_run_directory(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_names = ("feature_inputs", "predictions", "maps", "summary", "metadata")
    existing = [name for name in output_names if (run_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing statistics-map outputs in {run_dir}: "
            + ", ".join(existing)
        )
    (run_dir / "logs").mkdir(exist_ok=True)


def save_metadata(
    metadata_dir: Path,
    args: argparse.Namespace,
    run_dir: Path,
    dates: Sequence[dt.date],
    mask_shape: Tuple[int, int],
    selected_count: int,
    members: Sequence[int],
    device: torch.device,
) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        metadata_dir / "dates.npy",
        np.asarray(dates, dtype="datetime64[D]"),
    )
    np.savez(
        metadata_dir / "year_dates.npz",
        **{
            str(year): np.asarray(
                [date for date in dates if date.year == year],
                dtype="datetime64[D]",
            )
            for year in PREDICTION_YEARS
        },
    )
    config = {
        "run_name": args.run_name,
        "run_directory": str(run_dir),
        "mask_path": str(Path(args.mask_path).resolve()),
        "era5_directory": str(Path(args.era5_dir).resolve()),
        "input_directory": str(Path(args.input_dir).resolve()),
        "model_name": MODEL_NAME,
        "feature_files": FEATURES_FILES,
        "feature_scaler_prefixes": FEATURE_SCALER_PREFIXES,
        "lookback_days": LOOKBACK,
        "mask_shape": list(mask_shape),
        "selected_pixel_count": selected_count,
        "prediction_years": list(PREDICTION_YEARS),
        "excluded_calendar_dates": ["02-29"],
        "context_years": {
            str(year): year - 1 for year in PREDICTION_YEARS
        },
        "prediction_date_start": dates[0].isoformat(),
        "prediction_date_end": dates[-1].isoformat(),
        "prediction_day_count": len(dates),
        "ensemble_members": list(members),
        "device": str(device),
        "pixel_batch_size": args.pixel_batch_size,
        "time_batch_size": args.time_batch_size,
    }
    with (metadata_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def write_summaries(
    summary_path: Path,
    mean_predictions: np.ndarray,
    std_predictions: np.ndarray,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
    pixel_batch_size: int,
) -> None:
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ("y", "x", "mean_wtd", "std_wtd", "min_wtd", "max_wtd")
        )
        for start in range(0, mean_predictions.shape[1], pixel_batch_size):
            end = min(start + pixel_batch_size, mean_predictions.shape[1])
            mean_block = np.asarray(mean_predictions[:, start:end])
            spread_block = np.asarray(std_predictions[:, start:end])
            for offset in range(end - start):
                values = mean_block[:, offset]
                writer.writerow(
                    (
                        int(y_indices[start + offset]),
                        int(x_indices[start + offset]),
                        float(np.mean(values)),
                        float(np.mean(spread_block[:, offset])),
                        float(np.min(values)),
                        float(np.max(values)),
                    )
                )


def run(args: argparse.Namespace) -> None:
    mask_path = Path(args.mask_path)
    era5_dir = Path(args.era5_dir)
    input_dir = Path(args.input_dir)
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
    validate_run_directory(run_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "logs" / "run.log", mode="a"),
        ],
    )

    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0:
        raise ValueError(f"No pixels equal to 1 in mask {mask_path}")
    flat_indices = np.ravel_multi_index((y_indices, x_indices), mask.shape)
    LOGGER.info("Selected %d of %d grid cells", len(flat_indices), mask.size)

    missing_members = []
    for member in args.members:
        member_dir = Path(OUTPUTPATH) / "validation_ERA5" / "ensemble_400px" / f"400px_member_{member}"
        model_path = member_dir / f"400px_member_{member}_{MODEL_NAME}.pt"
        scaler_path = member_dir / f"meanstd_400px_member_{member}_{MODEL_NAME}.pkl"
        if not model_path.is_file() or not scaler_path.is_file():
            missing_members.append(member)
    if missing_members:
        raise FileNotFoundError(
            f"Missing pre-trained model or scaler files for ensemble members: "
            f"{missing_members}"
        )

    source_dates = all_dates(FIRST_CONTEXT_YEAR, PREDICTION_YEARS[-1])
    prediction_dates = [
        date for date in source_dates if date.year in PREDICTION_YEARS
    ]
    date_offsets = {date: index for index, date in enumerate(source_dates)}
    dates_dir = run_dir / "metadata"
    predictions_dir = run_dir / "predictions"
    maps_dir = run_dir / "maps"
    feature_path = run_dir / "feature_inputs" / "era5land_features.npy"

    feature_data = stage_features(
        feature_path,
        source_dates,
        era5_dir,
        input_dir,
        y_indices,
        x_indices,
        flat_indices,
        mask.shape,
        args.pixel_batch_size,
    )

    member_predictions_path = predictions_dir / "daily_wtd_members.npy"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    member_predictions = np.lib.format.open_memmap(
        member_predictions_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(args.members), len(prediction_dates), len(flat_indices)),
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    LOGGER.info("Using device %s", device)

    for member_index, member in enumerate(args.members):
        LOGGER.info("Loading ensemble member %d", member)
        scalers = load_member_scalers(member)
        model = load_member_model(member, device)
        for year in PREDICTION_YEARS:
            context_year = year - 1
            context_start = date_offsets[dt.date(context_year, 1, 1)]
            prediction_start = date_offsets[dt.date(year, 1, 1)]
            prediction_end = date_offsets[dt.date(year + 1, 1, 1)] if year < PREDICTION_YEARS[-1] else len(source_dates)
            context_days = prediction_start - context_start
            year_dates = [date for date in prediction_dates if date.year == year]
            output_start = prediction_dates.index(year_dates[0])
            output_end = output_start + len(year_dates)
            LOGGER.info(
                "Predicting %d using %d context days from %d",
                year,
                context_days,
                context_year,
            )
            for pixel_start in range(
                0, len(flat_indices), args.pixel_batch_size
            ):
                pixel_end = min(
                    pixel_start + args.pixel_batch_size, len(flat_indices)
                )
                feature_block = np.asarray(
                    feature_data[
                        :, context_start:prediction_end, pixel_start:pixel_end
                    ],
                    dtype=np.float32,
                )
                batch_predictions = predict_year_batch(
                    feature_block,
                    scalers,
                    model,
                    device,
                    len(year_dates),
                    context_days,
                    args.time_batch_size,
                )
                member_predictions[
                    member_index, output_start:output_end, pixel_start:pixel_end
                ] = batch_predictions
        member_predictions.flush()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del member_predictions
    member_predictions = np.load(member_predictions_path, mmap_mode="r")
    mean_path = predictions_dir / "daily_wtd_ensemble_mean.npy"
    std_path = predictions_dir / "daily_wtd_ensemble_std.npy"
    mean_predictions = np.lib.format.open_memmap(
        mean_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(prediction_dates), len(flat_indices)),
    )
    std_predictions = np.lib.format.open_memmap(
        std_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(prediction_dates), len(flat_indices)),
    )
    for pixel_start in range(0, len(flat_indices), args.pixel_batch_size):
        pixel_end = min(pixel_start + args.pixel_batch_size, len(flat_indices))
        member_block = np.asarray(
            member_predictions[:, :, pixel_start:pixel_end], dtype=np.float32
        )
        mean_predictions[:, pixel_start:pixel_end] = np.mean(member_block, axis=0)
        std_predictions[:, pixel_start:pixel_end] = np.std(member_block, axis=0)
    mean_predictions.flush()
    std_predictions.flush()

    maps_dir.mkdir(parents=True, exist_ok=True)
    full_mean = np.lib.format.open_memmap(
        maps_dir / "wtd_prediction_map_ensemble_mean.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(prediction_dates), *mask.shape),
    )
    full_std = np.lib.format.open_memmap(
        maps_dir / "wtd_prediction_map_ensemble_std.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(prediction_dates), *mask.shape),
    )
    full_mean[:] = np.nan
    full_std[:] = np.nan

    output_offset = 0
    for year in PREDICTION_YEARS:
        year_dates = [date for date in prediction_dates if date.year == year]
        count = len(year_dates)
        year_mean = np.lib.format.open_memmap(
            maps_dir / f"wtd_prediction_map_daily_{year}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, *mask.shape),
        )
        year_std = np.lib.format.open_memmap(
            maps_dir / f"wtd_prediction_map_spread_{year}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, *mask.shape),
        )
        year_mean[:] = np.nan
        year_std[:] = np.nan
        year_mean[:, y_indices, x_indices] = mean_predictions[
            output_offset : output_offset + count
        ]
        year_std[:, y_indices, x_indices] = std_predictions[
            output_offset : output_offset + count
        ]
        full_mean[output_offset : output_offset + count] = year_mean
        full_std[output_offset : output_offset + count] = year_std
        year_mean.flush()
        year_std.flush()
        output_offset += count
    full_mean.flush()
    full_std.flush()

    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        metadata_dir / "selected_pixels.npy",
        np.column_stack((y_indices, x_indices)),
    )
    save_metadata(
        metadata_dir,
        args,
        run_dir,
        prediction_dates,
        mask.shape,
        len(flat_indices),
        args.members,
        device,
    )
    (run_dir / "summary").mkdir(parents=True, exist_ok=True)
    write_summaries(
        run_dir / "summary" / "predictions_summary.csv",
        mean_predictions,
        std_predictions,
        y_indices,
        x_indices,
        args.pixel_batch_size,
    )
    LOGGER.info(
        "Saved %d daily predictions for %d selected pixels to %s",
        len(prediction_dates),
        len(flat_indices),
        run_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict masked EU daily WTD using the trained ERA5-Land LSTM ensemble."
    )
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--input-dir", default=INPUTPATH)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--members", type=parse_members, default=list(range(100)))
    parser.add_argument("--pixel-batch-size", type=int, default=64)
    parser.add_argument("--time-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pixel_batch_size < 1 or args.time_batch_size < 1:
        raise ValueError("Batch sizes must be positive integers")
    run(args)


if __name__ == "__main__":
    main()
