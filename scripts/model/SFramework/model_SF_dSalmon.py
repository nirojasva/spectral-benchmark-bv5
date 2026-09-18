import logging
from pathlib import Path
import sys
# Get the path to the current directory and go up to model path
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir))



import copy
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import os 

class SF_dSalmon():
    
    def __init__(self, learner, max_value, min_value, **learner_params):
        
        super().__init__() 
        

        self.base_model = learner(**learner_params)
        
        self.max_value = max_value
        self.min_value = min_value

        self.idx_model_scoring = None


    
    def __str__(self):
        return "SF_"+ str(self.base_model)
                    

    def fit_predict(self, spectra, extra_feature):   

        score = 0.0  
        self.idx_model_scoring = None         
        # Check partition range
        if (extra_feature >= self.min_value) and (extra_feature < self.max_value):
            #logging.debug(f'EXTRA FEATURE SCORING...{extra_feature}')  
            #logging.debug(f'MODEL SCORING...{1}')                   
            # Score
            self.idx_model_scoring = 1
            score = self.base_model.fit_predict(spectra)
     
        return score





if __name__ == "__main__":

    # Get the path to the current directory and go up to scripts path

    current_dir = Path(__file__).resolve().parent.parent.parent

    

    sys.path.append(str(current_dir))



    # Go one level up to include datasets

    current_dir = current_dir.parent



    from capymoa.stream import NumpyStream

    #from capymoa.evaluation import AnomalyDetectionEvaluator



    from data_utils import calculate_performance_metrics

    from model_utils import clean_score, median_of_means, transform_instance

    import time

    from model.mDragstream.mdragstream import mDragStream
    from model.OBKNN.model_OnlineBootKNN import OnlineBootKNN
    

    from pysad.models import ExactStorm, IForestASD, KitNet, LODA, RobustRandomCutForest, RSHash, xStream

    from capymoa.anomaly import OnlineIsolationForest, HalfSpaceTrees as HStreeCapy

    from dSalmon.outlier import SWKNN , SWLOF

    import json

    #VALIDATED CONFIGS
    PATH_SETTING_EXP = 'scripts/model/SFramework/config_sfswlof/config_sfswlof_exp_eval_BV5_only_spectra_deg.json'


    with open(PATH_SETTING_EXP, 'r') as f:
        config = json.load(f)

    config_path = Path(PATH_SETTING_EXP)
    log_file_path = config_path.parent / f"{config_path.stem}.log"

    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    LEARNER_NAME = config["experiment_config"].get("learner_name", None)
    DATA_PATH = current_dir / config["paths"].get("data_path", "")
    PATH_PLOT_FILE_NAME_INTERPRETATION = current_dir / config["paths"].get("plot_file_interpretation", "")
    PATH_PLOT_FILE_NAME_SCORE = current_dir / config["paths"].get("plot_file_score", "")

    files = [f for f in DATA_PATH.iterdir() if f.suffix == '.csv']

    summary_data = []

    TRANF = config["hyperparameters"].get("transf", None)
    P_WINDOW_SIZE = config["hyperparameters"].get("p_window_size", None)
    WINDOW_SIZE = config["hyperparameters"].get("window_size", None)
    CHUNK_SIZE = config["hyperparameters"].get("chunk_size", None)
    ENSEMBLE_SIZE = config["hyperparameters"].get("ensemble_size", None)
    N_JOBS = config["hyperparameters"].get("n_jobs", None)
    NO_BOOTSTRAPP = config["hyperparameters"].get("no_bootstrapp", None)
    NO_ZSCORE = config["hyperparameters"].get("no_z_score", None)
    UPDATE_WITH_ABNORMAL = config["hyperparameters"].get("update_distance_with_abnormal", None)
    DMETRIC = config["hyperparameters"].get("dmetric", None)
    ALGO = config["hyperparameters"].get("algorithm", None)
    ALPHA_Z_TEST = config["hyperparameters"].get("alpha_z_test", None)
    ALPHA_EMA = config["hyperparameters"].get("alpha_ema", None)
    TYPE_DIST = config["hyperparameters"].get("type_dist", None)
    UPDATE_MODE_STATS = config["hyperparameters"].get("update_mode_stats", None)
    SLEEP_TIME = config["experiment_config"].get("sleep_time", None)
    MIN_Z_SCORE = config["data_processing"].get("min_z_score", None)
    REGION_STUDY = config["data_processing"].get("region_study", None)
    COLS_POS_FMIN = config["data_processing"].get("cols_pos_fmin", None)
    COLS_POS_FMAX = config["data_processing"].get("cols_pos_fmax", None)
    COLS_POS_LABEL = config["data_processing"].get("cols_pos_label", None)
    NUMBER_RUNS = config["experiment_config"].get("number_runs", None)
    SCORE_DIR = config["experiment_config"].get("score_dir", None)
    DATASETS_LIST = config["experiment_config"].get("datasets_list", None)





    EFEATURE = config["data_processing"].get("efeature", None)
    
    params = config["data_processing"].get("feature_parameters", None)[EFEATURE]
    COLS_POS_EXTRA_F = params.get("cols_pos_extra_f", None)
    MIN_VALUE = params.get("min_value", None)
    MAX_VALUE = params.get("max_value", None)
    
    
    f_break=False

    for file_name in files:
        
        if any(substring in file_name.name for substring in DATASETS_LIST):
            logging.debug("File to Use: %s",file_name)
            
        else:
            logging.debug("Filed not to Use %s", file_name)
            continue

        # Load spectra and labels data
        file_path = os.path.join(DATA_PATH, file_name)
        # --- Data Loading ---
        try:
            df = pd.read_csv(file_path, sep=',', low_memory=False, dtype={'CURRENTTIMESTAMP': str})
        except:
            df = pd.read_csv(file_path, sep=',', low_memory=False)

        
        score_column = 'Score'
        error_column = 'Error'
        label_column = df.columns[COLS_POS_LABEL]  
        cols = df.columns[slice(COLS_POS_FMIN, COLS_POS_FMAX)]
        col_target = df.columns[-1]
        extra_df = df.iloc[:, COLS_POS_EXTRA_F:COLS_POS_EXTRA_F+1]

        logging.debug("Name DS: %s", file_name)
        logging.debug("# Total Points Spectra: %d", len( df[cols].values))
        logging.debug("# Total Points Extra DF: %d", len( extra_df))
        logging.debug("Extra DF:",  extra_df.head(3))
        logging.debug("# Anomalies: %d", len( df[col_target].values))
        logging.debug("# of Columns: %d", len(df.columns))
        logging.debug("# of Columns to use:  %d", len(cols))
        logging.debug("Columns:  %s", cols)

        

        stream = NumpyStream(df[cols].values, df[col_target].astype(int).values, dataset_name='PV', feature_names=cols)#forcing label colum as int for correct identification of anomalies 
        
        schema = stream.get_schema()
  

        if f_break:
            break


        
        for i in range(NUMBER_RUNS):

            stream.restart()
            

            scores = []
            raw_scores = []
            errors = []
            row = 0
            prev_y_index = 0

            if P_WINDOW_SIZE != None: 
                WINDOW_SIZE = max(1, int(len(df) * P_WINDOW_SIZE))

            if LEARNER_NAME == "SFSWKNN":
                learner = SF_dSalmon( learner=SWKNN, max_value=MAX_VALUE, min_value=MIN_VALUE, window=WINDOW_SIZE, k=10, k_is_max=False, metric="cityblock", min_node_size=5, max_node_size=20, split_sampling=5)            
            elif LEARNER_NAME == "SFSWLOF":
                K = config["hyperparameters"].get("k", None)
                K_IS_MAX = config["hyperparameters"].get("k_is_max", None)
                SIMPLIFIED = config["hyperparameters"].get("simplified", None)
                METRIC = config["hyperparameters"].get("metric", None)                
                learner = SF_dSalmon( learner=SWLOF, max_value=MAX_VALUE, min_value=MIN_VALUE, window=WINDOW_SIZE, k=K, k_is_max=K_IS_MAX, simplified=SIMPLIFIED, metric=METRIC)
            else:
                learner = None


            np.random.seed(i)

            if f_break:
                break
            
            for row, instance in enumerate(stream):
                
                if EFEATURE == "MomSpectra":
                    #extra_data = None  # using median of mean of znormalized spectra as extra data 
                    extra_data = median_of_means(transform_instance(instance,"ZNORM").x) # using median of mean of spectra as extra data 
                elif EFEATURE == "Voltage" or EFEATURE == "Current" or EFEATURE == "Pressure":
                    extra_data = float(extra_df.iloc[row]) # using voltage, current or pressure as extra data
                    #extra_data = median_of_means(instance.x) # using median of mean of spectra as extra data 
                    #extra_data = np.max(instance.x) #using max of spectra as extra data 
                    
                #while stream.has_more_instances():
        
                time.sleep(SLEEP_TIME)

                #instance = stream.next_instance()
                #row = row + 1 

                logging.debug("########################################################")    
                #logging.debug(f'A new instance ({row})...label: {instance.y_label}, index: {instance.y_index}')
                logging.debug(f'The new instance: {instance.x}, index: {instance.y_index}')
                logging.debug(f'Extra Data: {extra_data}')
                
                
                
                # --- API-Specific Logic for Training and Scoring ---
                # time.perf_counter() is used for precise timing within this process.
                if hasattr(learner, "fit_partial"):  # Pysad models
                    start_train = time.perf_counter()
                    learner.fit_partial(instance.x, extra_data)
                    training_time = time.perf_counter() - start_train
                    start_score = time.perf_counter()
                    score = learner.score_partial(instance.x, extra_data)
                    cleaned_score, error_score = clean_score(score)
                    scoring_time = time.perf_counter() - start_score
                elif hasattr(learner, "train"):  # Capymoa models
                    start_score = time.perf_counter()
                    
                    score = learner.score_instance(instance, extra_data)

                    cleaned_score, error_score = clean_score(score)
                    scoring_time = time.perf_counter() - start_score
                    start_train = time.perf_counter()
                    learner.train(instance, extra_data)
                    training_time = time.perf_counter() - start_train
                                        
                    """
                    if not learner.base_model.init and (cleaned_score > MIN_Z_SCORE and instance.y_index == prev_y_index and instance.y_index == 0): #Index for False Positives
                        idx = row + 1
                    elif not learner.base_model.init and (instance.y_index != prev_y_index): #Index for Starting and Ending of Cycles
                        idx = row + 1
                    else:
                        idx = 0


                    if idx != 0 :
                        learner.base_model.explain(headers=cols, region_study_list=REGION_STUDY, path=PATH_PLOT_FILE_NAME_INTERPRETATION, file_name=file_name.name.split("_")[0]+"_transf_"+TRANF+"_efeature_unimodel_"+EFEATURE+"_explanation_sf_"+str(learner)+str(idx))
             
                    learner.plot_core_statistics(PATH_PLOT_FILE_NAME_SCORE, file_name.name.split("_")[0]+"_transf_"+TRANF+"_sf_"+str(learner)+"_efeature_"+EFEATURE+"_"+str(idx), label=instance.y_label)
                    """

                elif hasattr(learner, "fit_predict"):  # dSalmon models
                    training_time = 0
                    start_score = time.perf_counter()
                    score = learner.fit_predict(instance.x, extra_data)
                    cleaned_score, error_score = clean_score(score)
                    scoring_time = time.perf_counter() - start_score
                else:
                    raise AttributeError(f"Model {learner.__class__.__name__} has no recognized training/scoring method.")

                

                prev_y_index = instance.y_index

                scores.append(cleaned_score)
                raw_scores.append(score)
                errors.append(error_score)

                logging.debug(f'Cleaned Score ({row}): {cleaned_score}')  
                logging.debug(f'Cleaned Message ({row}): {error_score}') 
                logging.debug(f'Score ({row}): {score}')     
 
                if np.isnan(cleaned_score):
                   f_break = True
                   break


            df[score_column+str(i)] = list(scores)
            df[score_column+"_raw"+str(i)] = list(raw_scores)
            df[error_column+str(i)] = list(errors)

            try:
                results = calculate_performance_metrics(df, label_column, score_column+str(i), WINDOW_SIZE, score_direction=SCORE_DIR)
                roc_auc, pr_auc, max_f1, metrics, roc_auc_wtd, pr_auc_wtd, max_f1_wtd, pct_detection, pct_false_positives, tn, fp, fn, tp, best_threshold = results
                auc_roc = metrics.get('AUC_ROC', None)
                auc_pr = metrics.get('AUC_PR', None)
                precision = metrics.get('Precision', None)
                f = metrics.get('F', None)
                precision_at_k = metrics.get('Precision_at_k', None)
                rprecision = metrics.get('Rprecision', None)
                rrecall = metrics.get('Rrecall', None)
                rf = metrics.get('RF', None)
                r_auc_roc = metrics.get('R_AUC_ROC', None)
                r_auc_pr = metrics.get('R_AUC_PR', None)
                vus_roc = metrics.get('VUS_ROC', None)
                vus_pr = metrics.get('VUS_PR', None)
                affiliation_precision = metrics.get('Affiliation_Precision', None)
                affiliation_recall = metrics.get('Affiliation_Recall', None)            
            
            
            except Exception as e:
                print(f"Error calculating metrics for index {i}: {e}")
                roc_auc = pr_auc = max_f1 = metrics = roc_auc_wtd = pr_auc_wtd = max_f1_wtd = pct_detection = pct_false_positives = tn = fp = fn = tp = best_threshold = None

            summary_data.append({
            "iteration": i,
            "scenario": file_name.name.split("_")[0],
            #"method_window_and_param": mwp,
            "method": LEARNER_NAME,
            "raw_roc_auc": roc_auc,
            "raw_pr_auc": pr_auc,
            "raw_max_f1": max_f1,
            #"auc_pr_online": auc_pr_online,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "precision": precision,
            "f_metric": f,   
            "precision_at_k": precision_at_k, 
            "rprecision": rprecision,    
            "rrecall": rrecall,   
            "rf": rf,     
            "r_auc_roc": r_auc_roc,   
            "r_auc_pr": r_auc_pr, 
            "vus_roc": vus_roc,  
            "vus_pr": vus_pr,        
            "affiliation_precision": affiliation_precision,
            "affiliation_recall": affiliation_recall,
            "pct_detection": pct_detection,             # Added variable
            "pct_false_positives": pct_false_positives, # Added variable
            "tn": tn,                                   # Added variable
            "fp": fp,                                   # Added variable
            "fn": fn,                                   # Added variable
            "tp": tp,                                   # Added variable
            "best_threshold": best_threshold,            # Added variable
            "window_size": WINDOW_SIZE
            })
            
    logging.debug("########################")
    logging.debug("Complete Results:")        
    result = pd.DataFrame(df)
    logging.debug(result[[score_column+str(i), error_column+str(i)]])

    # Create DataFrame from collected results
    summary_data = pd.DataFrame(summary_data)

    # Define all metrics that need aggregation
    target_variables = [
        #"raw_pr_auc", 
        "vus_pr",
        "pct_detection", 
        "pct_false_positives", 
        #"tn", 
        #"fp", 
        #"fn", 
        #"tp", 
        "best_threshold"
    ]

    # Generate pivot tables for each metric
    for metric in target_variables:
        if metric not in summary_data.columns:
            continue
            
        logging.debug(f"{'#' * 24}")
        logging.debug(f"Summary Online Algorithms (mean {metric}):")

        pivot = summary_data.pivot_table(
            values=[metric],
            columns=['scenario'],
            index=['method', 'window_size'], 
            aggfunc='mean'
        )

        # Row-wise mean
        pivot['Avg'] = pivot.mean(axis=1)

        # Formatting
        pivot = pivot.round(5)
        pivot = pivot.sort_values(by='Avg', ascending=False)
        
        logging.debug("\n" + str(pivot))
        logging.debug(f"{'#' * 24}")
    
        
        logging.debug(f"Summary Online Algorithms (std {metric}):")

        pivot = summary_data.pivot_table(
            values=[metric],
            columns=['scenario'],
            index=['method', 'window_size'], 
            aggfunc='std'
        )

        # Row-wise mean
        pivot['Avg'] = pivot.mean(axis=1)

        # Formatting
        pivot = pivot.round(5)
        pivot = pivot.sort_values(by='Avg', ascending=False)
        
        logging.debug("\n" + str(pivot))

