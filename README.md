# spatio-temporal-LSTM
This is the script for spatially training an LSTM model with an example dataset
## Folder structure
spatio-temporal-LSTM/
├── LSTM_model/
│   ├── data/
│   |   ├── postprocess.py
│   |   ├── preprocess.py
│   |   └── rawdataset.py
│   ├── model/
│   |   ├── config.py
│   |   ├── train_LSTM.py
│   |   └── validate_LSTM.py
│   └── utils/
│       ├── plot_functions.py
│       ├── utils.py
│       └── volumetric_soilmoisture.py
├── inputs
├── outputs
├── requirements.txt
├── README.md
└── main.py

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
