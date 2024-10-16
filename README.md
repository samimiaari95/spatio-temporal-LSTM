# spatio-temporal-LSTM
This is the script for spatially training an LSTM model with an example dataset
## Folder structure
spatio-temporal-LSTM/<br>
├── LSTM_model/<br>
│&emsp;&emsp;├── data/<br>
│&emsp;&emsp;&nbsp;|&emsp;&emsp;&emsp;├── postprocess.py<br>
│&emsp;&emsp;&nbsp;|&emsp;&emsp;&emsp;├── preprocess.py<br>
│&emsp;&emsp;&nbsp;|&emsp;&emsp;&emsp;└── rawdataset.py<br>
│&emsp;&emsp;├── model/<br>
│&emsp;&emsp;&nbsp;|&emsp;&emsp;&emsp;├── config.py<br>
│&emsp;&emsp;&nbsp;|&emsp;&emsp;&emsp;├── train_LSTM.py<br>
│&emsp;&emsp;&nbsp;|&emsp;&emsp;&emsp;└── validate_LSTM.py<br>
│&emsp;&emsp;└── utils/<br>
│&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;├── plot_functions.py<br>
│&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;├── utils.py<br>
│&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;└── volumetric_soilmoisture.py<br>
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
