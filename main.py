import argparse
import matplotlib.pyplot as plt
# from LSTM_model.data.ensemble_analysis import ensemble_analysis
from LSTM_model.model.config import *
from LSTM_model.model.train_LSTM import train_LSTM_model
from LSTM_model.model.validate_LSTM import transfer_ensemble_LSTM
from LSTM_model.data.postprocess import postprocess_calculations
from LSTM_model.data.preprocess import preprocessing_data
from LSTM_model.data.rawdataset import preprocess_rawdata
from LSTM_model.data.obs_processing import obs_processing
from LSTM_model.utils.plot_functions import plotting_helper
from LSTM_model.utils.utils import utilities
import LSTM_model.data.download_ERA5 as download_ERA5
# from LSTM_model.data.metplotpy import MetplotpyEnsembleAnalysis
#from LSTM_model.utils.volumetric_soilmoisture import calculate_soilmoisture


parser = argparse.ArgumentParser()
# parser.add_argument('--batchind', type=str, required=True, help='yyyymmdd00') # month of simulation

parser.add_argument('-d', '--dera', action='store_true', required=False, help='ERA5 data download')

parser.add_argument('-r', '--raw', action='store_true', required=False, help='Rawdata preprocessing')

parser.add_argument('-p', '--prep', action='store_true', required=False, help='Preprocessing data')

parser.add_argument('-t', '--train', action='store_true', required=False, help='Training mode')

parser.add_argument('-e', '--eval', action='store_true', required=False, help='Evaluation mode')

parser.add_argument('-c', '--calc', action='store_true', required=False, help='Results calculation')

parser.add_argument('-o', '--obs', action='store_true', required=False, help='Observations processing')

# parser.add_argument('-a', '--ana', action='store_true', required=False, help='Ensemble analysis')

# parser.add_argument('-m', '--metplotpy', action='store_true', required=False, help='METplotpy analysis')

args = parser.parse_args()
plt.rcParams.update({'font.size': 18})

if args.dera:
    # download_ERA5.request_ERA5_t2m_d2m_swvl3()
    # download_ERA5.request_ERA5_tp()
    # download_ERA5.extract_allzipfiles()
    pass

if args.raw:
    rawdataset = preprocess_rawdata()
    rawdataset.extract_vars()
    # call function here

if args.prep:
    preprocessing = preprocessing_data()
    # call function here
    # preprocessing.conc_vars()
    # preprocessing.ensemble_cropvars()
    preprocessing.checkoutputs()

if args.train:
    train_LSTM = train_LSTM_model()
    train_LSTM.train()

if args.eval:
    validate_LSTM = transfer_ensemble_LSTM()
    batchind = args.batchind
    # input_vars = {x: np.load(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{batchind}", x)) for x in FEATURES_FILES}
    # targetvar = np.load(os.path.join(INPUTPATH, "validation_ERA5", f"target_pixels_{batchind}", TARGETVAR_FILE))
    # for member in range(100):
        # validate_LSTM.transfer_ensemble(batchind=batchind, member=member, input_vars=input_vars, targetvar=targetvar)
    try:
        validate_LSTM.transfer_ensemble(batchind=batchind)
    except:
        # write the batchind and member that failed to a log file
        with open(os.path.join(OUTPUTPATH, "validation_ERA5", "ensemble_400px", f"failed_batches_{member}.log"), 'a') as f:
            f.write(f"{batchind}\n")
        f.close()

if args.calc:
    postprocess = postprocess_calculations()
    # postprocess.ens_mean()
    # postprocess.concat_EU_transfer_testtrain_outputs()
    # postprocess.cdf_EU()
    # postprocess.metrics_2D_EU()

    # postprocess.preprocess_crps_trainingEU()
    # postprocess.preprocess_crps_transferEU()
    # postprocess.crps_seasonal_trainEU()
    # postprocess.crps_seasonal_transferEU()
    # postprocess.boxplot_seasonal_crps()
    postprocess.ensemble_statvsacc()
    postprocess.ensemble_statvsacc_fitting()
    # postprocess.crps_transferEU()
    # postprocess.ensemble_crpsvsstats()
    # postprocess.ensemble_crpsvsstats_fitting()
    # postprocess.estimate_acc_from_stats()
    # postprocess.kge_investigation()
    # postprocess.kgecomponents_vs_EV()
    # postprocess.metrics_vs_topo()
    
    # postprocess.eval_era5wtd_obswtd_tsmpwtd()
    # postprocess.plot2Dmaps()
    # postprocess.fitted_ev_rmse()
    # postprocess.fitted_iqr_mab()
    # postprocess.plot_cdfs()
    # TODO plot y=x
    # postprocess.location_of_localobs()
    # postprocess.postprocess_era5wtd_vs_obs()

if args.obs:
    obs_proc = obs_processing()
    # call function here
    obs_proc.read_csv_monthlyobs_from_yueling()
    obs_proc.maptoTSMP_averageduplicates()

# if args.ana:
#     ensemble_ana = ensemble_analysis()
#     # call function here
#     ensemble_ana.run_analysis()

# if args.metplotpy:
#     metplotpy = MetplotpyEnsembleAnalysis()
#     # call function here
#     metplotpy.runmetplotpy_full()