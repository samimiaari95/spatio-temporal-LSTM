import argparse
import logging
import os


def _validate_local_write_paths(*paths):
    from LSTM_model.model.config import ROOT_DIR

    root = os.path.realpath(ROOT_DIR)
    for path in paths:
        resolved = os.path.realpath(path)
        if os.path.commonpath((root, resolved)) != root:
            raise ValueError(
                f"Refusing to write outside the repository: {resolved}"
            )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the raw-data, preparation, model, and analysis stages."
    )
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help="Stage files from the configured local archive.",
    )
    stages.add_argument(
        "-p",
        "--prep",
        "--prepare",
        action="store_true",
        help="Validate the prepared training inputs.",
    )
    stages.add_argument("-t", "--train", action="store_true", help="Train the LSTM model.")
    stages.add_argument(
        "-e",
        "--eval",
        "--predict",
        action="store_true",
        help="Predict the configured test period.",
    )
    stages.add_argument(
        "-c",
        "--calc",
        action="store_true",
        help="Calculate single-model evaluation metrics and plots.",
    )
    stages.add_argument(
        "--figures",
        action="store_true",
        help="Run the available manuscript figure generators.",
    )
    stages.add_argument(
        "--all",
        action="store_true",
        help="Run prepare, train, predict, and analysis stages.",
    )
    parser.add_argument(
        "--raw-month",
        action="append",
        default=[],
        help="Archive month identifier (repeat once per archive; raw stage only).",
    )
    parser.add_argument(
        "--raw-file",
        action="append",
        default=[],
        help="Archive member path relative to the month directory (repeatable; raw stage only).",
    )
    return parser


def _prepare_inputs():
    from LSTM_model.data.input_validation import validate_preprocessed_inputs

    shapes = validate_preprocessed_inputs()
    logging.info("Prepared inputs validated:")
    for filename, shape in shapes.items():
        logging.info("  %s: %s", filename, shape)


def _stage_raw_files(months, filenames):
    if not months or not filenames:
        raise ValueError(
            "The raw stage requires at least one --raw-month and one --raw-file."
        )
    from LSTM_model.model.config import RAW_DATA_SOURCE, RAW_STAGING_PATH

    _validate_local_write_paths(RAW_DATA_SOURCE, RAW_STAGING_PATH)
    from LSTM_model.data.rawdataset import preprocess_rawdata

    staged_files = preprocess_rawdata().stage_files(months, filenames)
    for filepath in staged_files:
        logging.info("Staged %s", filepath)


def _train():
    from LSTM_model.model.config import OUTPUTPATH

    _validate_local_write_paths(OUTPUTPATH)
    from LSTM_model.model.train_LSTM import train_LSTM_model

    train_LSTM_model().train()


def _predict():
    from LSTM_model.model.config import OUTPUTPATH

    _validate_local_write_paths(OUTPUTPATH)
    from LSTM_model.model.validate_LSTM import validate_LSTM_model

    observation_path, prediction_path = validate_LSTM_model().eval_LSTM()
    logging.info("Saved observations to %s", observation_path)
    logging.info("Saved predictions to %s", prediction_path)


def _calculate_results():
    from LSTM_model.model.config import OUTPUTPATH

    _validate_local_write_paths(OUTPUTPATH)
    from LSTM_model.data.postprocess import postprocess_calculations

    calculations = postprocess_calculations()
    calculations.calculate_accuracy_parameters()
    calculations.EUpx_results()


def _generate_figures():
    from LSTM_model.model.config import INPUTPATH, OUTPUTPATH

    _validate_local_write_paths(os.path.dirname(INPUTPATH), OUTPUTPATH)
    from LSTM_model.data.postprocess import postprocess_calculations

    postprocess_calculations().generate_figures()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.raw:
        _stage_raw_files(args.raw_month, args.raw_file)
    elif args.prep:
        _prepare_inputs()
    elif args.train:
        _prepare_inputs()
        _train()
    elif args.eval:
        _prepare_inputs()
        _predict()
    elif args.calc:
        _calculate_results()
    elif args.figures:
        _generate_figures()
    elif args.all:
        _prepare_inputs()
        _train()
        _predict()
        _calculate_results()


if __name__ == "__main__":
    main()
