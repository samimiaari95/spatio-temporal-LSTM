# spatio-temporal-LSTM
This is the script for spatially training an LSTM model with an example dataset
## LSTM_setup.py
This is the settings file, where the model parameters, time period, input features, and directories are defined. It includes all the global variables
## train_LSTM.py
train_LSTM.py is the script for only training the model, it saves the trained model in outputs/REGION
## postprocess_LSTMoutput.py
This is the script for model evaluation (testing phase), and calculation of error metrices.
## plot_functions.py
This is the script for all functions for plots and figures.
## preprocessing_rawdataset.py
This is the script for preprocessing the raw data from the source dataset to prepare it as inputs for the model training
## volumetric_soilmoisture.py
The soil moisture calculation is done in this script per month
## utils.py
utils.py is a script that includes generalized functions to be used upon needs
## basic examples
The inputs folder contains the input dataset for the SEINE 5x5 domain in numpy arrays 
The outputs folder contains the results of the evaluation and figures.

