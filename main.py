import argparse
from LSTM_model.model.config import *
from LSTM_model.model.train_LSTM import train_LSTM_model
from LSTM_model.model.validate_LSTM import validate_LSTM_model
from LSTM_model.data.postprocess import postprocess_calculations
from LSTM_model.data.preprocessing import preprocessing_data
from LSTM_model.data.rawdataset import preprocess_rawdata
from LSTM_model.utils.plot_functions import plotting_helper
from LSTM_model.utils.utils import utilities
from LSTM_model.utils.volumetric_soilmoisture import calculate_soilmoisture


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true', required=False, help='start model traning')
args = parser.parse_args()

if args.train:
    train_LSTM = train_LSTM_model()
    train_LSTM.train()
