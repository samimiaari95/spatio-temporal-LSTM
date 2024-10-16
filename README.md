# spatio-temporal-LSTM
This is the script for spatially training an LSTM model with an example dataset
## Folder structure
spatio-temporal-LSTM/<br>
├── LSTM_model/<br>
│   ├── data/<br>
│   |   ├── postprocess.py<br>
│   |   ├── preprocess.py<br>
│   |   └── rawdataset.py<br>
│   ├── model/<br>
│   |   ├── config.py<br>
│   |   ├── train_LSTM.py<br>
│   |   └── validate_LSTM.py<br>
│   └── utils/<br>
│       ├── plot_functions.py<br>
│       ├── utils.py<br>
│       └── volumetric_soilmoisture.py<br>
├── inputs<br>
├── outputs<br>
├── requirements.txt<br>
├── README.md<br>
└── main.py<br>

## Example dataset
In the input directory exists the input variables for training and evaluating the model. <br>
The output directory includes the trained model for the respective inputs.<br>
The config.py is adjusted for the provided example. <br>
### Run the example
To run the example install requirements and run the main.py file in addition to the argument for the needed mode as follow:
```
python main.py -train # training mode
python main.py -eval  # validation mode
python main.py -calc  # calculation  
```
The provided input data are already preprocessed and ready for training and validation.
