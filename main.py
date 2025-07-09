import argparse
import matplotlib.pyplot as plt
from LSTM_model.model.config import *
from LSTM_model.model.train_LSTM import train_LSTM_model
from LSTM_model.model.validate_LSTM import validate_LSTM_model, transfer_ensemble_LSTM
from LSTM_model.data.postprocess import postprocess_calculations
from LSTM_model.data.preprocess import preprocessing_data
from LSTM_model.data.rawdataset import preprocess_rawdata
from LSTM_model.utils.plot_functions import plotting_helper
from LSTM_model.utils.utils import utilities
#from LSTM_model.utils.volumetric_soilmoisture import calculate_soilmoisture


parser = argparse.ArgumentParser()

parser.add_argument('-r', '--raw', action='store_true', required=False, help='Rawdata preprocessing')

parser.add_argument('-p', '--prep', action='store_true', required=False, help='Preprocessing data')

parser.add_argument('-t', '--train', action='store_true', required=False, help='Training mode')

parser.add_argument('-e', '--eval', action='store_true', required=False, help='Evaluation mode')

parser.add_argument('-c', '--calc', action='store_true', required=False, help='Results calculation')

args = parser.parse_args()
plt.rcParams.update({'font.size': 20})

if args.raw:
    rawdataset = preprocess_rawdata()
    # call function here

if args.prep:
    preprocessing = preprocessing_data()
    # call function here

if args.train:
    train_LSTM = train_LSTM_model()
    train_LSTM.train()

if args.eval:
    validate_LSTM = transfer_ensemble_LSTM()
    validate_LSTM.transfer_ensemble()

if args.calc:
    postprocess = postprocess_calculations()
    # call function here