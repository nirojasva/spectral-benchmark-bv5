import logging
from pathlib import Path
import sys
# Get the path to the current directory and go up to model path
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir))


from capymoa.instance import Instance
from capymoa.base import AnomalyDetector
import copy
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt


class SF_Capymoa(AnomalyDetector):
    
    def __init__(self, schema, learner, max_value, min_value, **learner_params):
        
        super().__init__(schema=schema, random_seed=learner_params.get("random_seed")) 
        

        self.base_model = learner(schema=schema, **learner_params)
        
        self.max_value = max_value
        self.min_value = min_value
        
        self.monitoring = None
        self.z_scores_to_monitor = []
        self.means_to_monitor = []
        self.std_devs_to_monitor = []
        self.min_dists_to_monitor = []
        self.z_thresholds_to_monitor = []
        self.p_random_number_to_monitor = []
        self.ground_truth_to_monitor = []
        
        self.max_p_random_number = None
        self.mean = None
        self.std_dev = None
        self.min_dist = None
        self.z = None
        self.z_critical_one_tail = None

    
    def __str__(self):
        return "SF_"+ str(self.base_model)
    
    def train(self, spectra: Instance, extra_feature):

            # Check partition range
            if (extra_feature >= self.min_value) and (extra_feature < self.max_value):
                #logging.debug(f'EXTRA FEATURE FITTING...{extra_feature}')  
                #logging.debug(f'MODEL FITTTED...{1}')
                self.base_model.train(spectra)
                

    def score_instance(self, spectra: Instance, extra_feature):   

        score = 0.0 
        
        # Check partition range
        if (extra_feature >= self.min_value) and (extra_feature < self.max_value):
            #logging.debug(f'EXTRA FEATURE SCORING...{extra_feature}')  
            #logging.debug(f'MODEL SCORING...{1}')                    
            # Score
            score = self.base_model.score_instance(spectra)            

            

            if self.monitoring == "constant" or self.monitoring == "input_zero":
                self.max_p_random_number = self.base_model.max_p_random_number 
                self.mean = self.base_model.mean
                self.std_dev = self.base_model.std_dev
                self.min_dist = self.base_model.min_dist
                self.z = self.base_model.z
                self.z_critical_one_tail = self.base_model.z_critical_one_tail  
        
        else: 
            
            if hasattr(self.base_model, "extra_processing"):
                self.base_model.extra_processing()

            if self.monitoring == "constant":
                self.max_p_random_number = self.base_model.max_p_random_number 
                self.mean = self.base_model.mean
                self.std_dev = self.base_model.std_dev
                self.min_dist = self.base_model.min_dist
                self.z = self.base_model.z
                self.z_critical_one_tail = self.base_model.z_critical_one_tail  
            
            elif self.monitoring == "input_zero":
                self.max_p_random_number = np.nan
                self.mean = self.base_model.mean
                self.std_dev = self.base_model.std_dev
                self.min_dist = np.nan
                self.z = score
                self.z_critical_one_tail = self.base_model.z_critical_one_tail 

        return score


    def predict(self, data: np.ndarray):
        """
        Predict method (to be implemented by subclasses).
        """
        raise NotImplementedError("The 'predict' method must be implemented by subclasses.")

    def plot_core_statistics(self, path: str, file_name: str, label: int = 0, monitoring = "input_zero"):
            self.monitoring = monitoring

            self.p_random_number_to_monitor.append(self.max_p_random_number) 
            self.means_to_monitor.append(self.mean)
            self.std_devs_to_monitor.append(self.std_dev)
            self.min_dists_to_monitor.append(self.min_dist)
            self.ground_truth_to_monitor.append(label)
            
            self.z_scores_to_monitor.append(self.z)
            self.z_thresholds_to_monitor.append(self.z_critical_one_tail)
                
            # Helpers
            gt_indices = [i for i, x in enumerate(self.ground_truth_to_monitor) if int(x) == 1]
            x_axis = np.arange(len(self.means_to_monitor))
            # Volvemos al naranja original
            gt_style = {'color': 'darkorange', 'edgecolors': 'darkorange', 'marker': 'X', 's': 50, 'zorder': 5}

            # --- Gráfico 1: Mean con Std y Min Dist ---
            plt.clf()
            fig, ax1 = plt.subplots(figsize=(10, 6))
            means = np.array(self.means_to_monitor, dtype=float)
            stds = np.array(self.std_devs_to_monitor, dtype=float)
            min_dists = np.array(self.min_dists_to_monitor, dtype=float)

            ax1.plot(x_axis, min_dists, label='Min Dist', color='green', marker='s', markersize=2, alpha=0.7)
            if gt_indices:
                ax1.scatter(gt_indices, min_dists[gt_indices], label='Anomaly (GT)', **gt_style)
            ax1.set_ylabel('Min Distance', color='green')
            ax1.grid(True, linestyle='--', alpha=0.6)

            ax2 = ax1.twinx()
            ax2.plot(x_axis, means, label='Mean', color='blue', linestyle='--', marker='o', markersize=2)
            ax2.fill_between(x_axis, means - stds, means + stds, color='blue', alpha=0.2, label='Confidence Interval (±1 Std)')
            ax2.set_ylabel('Mean Statistics', color='blue')
            
            plt.title('Min Dist vs Mean & Std Dev')
            fig.legend(loc='upper left')
            plt.savefig(os.path.join(path, f"{file_name}_mean_min.pdf"), format="pdf")
            plt.close()

            # --- Gráfico 2: Z-Score con Threshold y Extra Feature ---
            plt.clf()
            fig, ax3 = plt.subplots(figsize=(10, 6))
            z_scores = np.array(self.z_scores_to_monitor)

            ax3.plot(z_scores, label='Z Score', color='red', marker='o', markersize=4)
            ax3.plot(self.z_thresholds_to_monitor, label='Threshold', color='purple', linestyle=':', marker='s', markersize=2)
            if gt_indices:
                ax3.scatter(gt_indices, z_scores[gt_indices], label='Anomaly (GT)', **gt_style)
            ax3.set_ylabel('Z Score & Threshold')
            ax3.grid(True, linestyle='--', alpha=0.6)

            ax4 = ax3.twinx()
            #ax4.plot(self.extra_feature_to_monitor, label='Extra Feature', color='cyan', linestyle='-', marker='x', markersize=2)
            #ax4.set_ylabel('Extra Feature Value', color='cyan')
            
            plt.title('Z-Score, Threshold')
            fig.legend(loc='upper left')
            plt.savefig(os.path.join(path, f"{file_name}_zscore_extra.pdf"), format="pdf")
            plt.close()

            # --- Gráfico 3: Z-Score, i-Model y P-Random Number ---
            plt.clf()
            fig, ax5 = plt.subplots(figsize=(10, 6))

            ax5.plot(z_scores, label='Z Score', color='red', marker='o', markersize=4)
            if gt_indices:
                ax5.scatter(gt_indices, z_scores[gt_indices], label='Anomaly (GT)', **gt_style)
            ax5.set_ylabel('Z Score', color='red')
            ax5.grid(True, linestyle='--', alpha=0.6)

            ax6 = ax5.twinx()
            #ax6.plot(self.i_model_to_monitor, label='i-Model', color='grey', marker='o', markersize=4)
            ax6.plot(self.p_random_number_to_monitor, label='P Random Num', color='black', linestyle='--', marker='^', markersize=2)
            ax6.set_ylabel('Model & P-Number')
            
            plt.title('Z-Score vs Model & P-Random Number')
            fig.legend(loc='upper left')
            plt.savefig(os.path.join(path, f"{file_name}_zscore_model_pnum.pdf"), format="pdf")
            plt.close()
            
            logging.debug(f"All reorganized plots saved to {path}")

   


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
    PATH_SETTING_EXP = 'scripts/model/SFramework/config_sfobknn/config_sfobknn_znorm_exp_eval_BV5_only_spectra_deg.json'
    #PATH_SETTING_EXP = 'scripts/model/SFramework/config_sfhstree/config_sfhstree_exp_eval_BV5_only_spectra_deg.json'

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

            if LEARNER_NAME == "SFOnlineBootKNN":
                learner = SF_Capymoa( schema=schema, learner=OnlineBootKNN, max_value=MAX_VALUE, min_value=MIN_VALUE, random_seed=i, window_size=WINDOW_SIZE, type_dist=TYPE_DIST, chunk_size=CHUNK_SIZE, ensemble_size=ENSEMBLE_SIZE, dmetric=DMETRIC, transf=TRANF, alpha_z_test=ALPHA_Z_TEST, algorithm=ALGO, no_bootstrapp=NO_BOOTSTRAPP, no_z_score=NO_ZSCORE, update_mode_stats=UPDATE_MODE_STATS, update_distance_with_abnormal=UPDATE_WITH_ABNORMAL, alpha_ema=ALPHA_EMA, n_jobs = N_JOBS)
            elif LEARNER_NAME == "SFIForestASD":
                learner = SF_Capymoa( schema=schema, learner=IForestASD, max_value=MAX_VALUE, min_value=MIN_VALUE, window_size=WINDOW_SIZE, initial_window_X=None)
            elif LEARNER_NAME == "SFKitNet":
                learner = SF_Capymoa( schema=schema, learner=KitNet, max_value=MAX_VALUE, min_value=MIN_VALUE, hidden_ratio=0.75, learning_rate=0.1, max_size_ae=10, grace_feature_mapping=WINDOW_SIZE, grace_anomaly_detector=WINDOW_SIZE)
            elif LEARNER_NAME == "SFOnlineIsolationForest":
                learner = SF_Capymoa( schema=schema, learner=OnlineIsolationForest, max_value=MAX_VALUE, min_value=MIN_VALUE, window_size=WINDOW_SIZE, random_seed=i, growth_criterion='fixed', max_leaf_samples=64, n_jobs=-1, num_trees=64)
            elif LEARNER_NAME == "SFExactStorm":
                learner = SF_Capymoa( schema=schema, learner=ExactStorm, max_value=MAX_VALUE, min_value=MIN_VALUE, window_size=WINDOW_SIZE, max_radius=900)
            elif LEARNER_NAME == "SFmDragStream_mahalanobis":
                learner = SF_Capymoa( schema=schema, learner=mDragStream, max_value=MAX_VALUE, min_value=MIN_VALUE, maxNbClusters=30, maxClusterSize=30, metric="mahalanobis", inv_cov=None, n_init=WINDOW_SIZE)
            elif LEARNER_NAME == "SFmDragStream_euclidean":
                learner = SF_Capymoa( schema=schema, learner=mDragStream, max_value=MAX_VALUE, min_value=MIN_VALUE, maxNbClusters=30, maxClusterSize=50, metric="euclidean", inv_cov=None, n_init=WINDOW_SIZE)
            elif LEARNER_NAME == "SFRSHash":
                learner = SF_Capymoa( schema=schema, learner=RSHash, max_value=MAX_VALUE, min_value=MIN_VALUE, sampling_points=WINDOW_SIZE, decay=0.05, feature_maxes=[np.inf, 2000], feature_mins=[0], num_components=50, num_hash_fns=1)
            elif LEARNER_NAME == "SFRobustRandomCutForest":
                learner = SF_Capymoa( schema=schema, learner=RobustRandomCutForest, max_value=MAX_VALUE, min_value=MIN_VALUE, shingle_size=WINDOW_SIZE, num_trees=4, tree_size=256)
            elif LEARNER_NAME == "SFSWKNN":
                learner = SF_Capymoa( schema=schema, learner=SWKNN, max_value=MAX_VALUE, min_value=MIN_VALUE, window=WINDOW_SIZE, k=10, k_is_max=False, metric="cityblock", min_node_size=5, max_node_size=20, split_sampling=5)
            elif LEARNER_NAME == "SFSWLOF":
                learner = SF_Capymoa( schema=schema, learner=SWLOF, max_value=MAX_VALUE, min_value=MIN_VALUE, window=WINDOW_SIZE, k=10, k_is_max=False, simplified=False, metric="euclidean")
            elif LEARNER_NAME == "SFxStream":
                learner = SF_Capymoa( schema=schema, learner=xStream, max_value=MAX_VALUE, min_value=MIN_VALUE, window_size=WINDOW_SIZE, depth=25, n_chains=100, num_components=50)
            elif LEARNER_NAME == "SFHStreeCapy":
                NUMBER_OF_TREES = config["hyperparameters"].get("number_of_trees", None)
                ANOMALY_THRESHOLD = config["hyperparameters"].get("anomaly_threshold", None)
                SIZE_LIMIT = config["hyperparameters"].get("size_limit", None)
                MAX_DEPTH = config["hyperparameters"].get("max_depth", None)
                
                learner = SF_Capymoa( schema=schema, learner=HStreeCapy, max_value=MAX_VALUE, min_value=MIN_VALUE, window_size=WINDOW_SIZE, number_of_trees=NUMBER_OF_TREES, anomaly_threshold=ANOMALY_THRESHOLD, size_limit=SIZE_LIMIT, max_depth=MAX_DEPTH, random_seed=i)
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
                if hasattr(learner, "train"):  # Capymoa models
                    start_score = time.perf_counter()
                    
                    score = learner.score_instance(instance, extra_data)

                    cleaned_score, error_score = clean_score(score)
                    scoring_time = time.perf_counter() - start_score
                    start_train = time.perf_counter()
                    learner.train(instance, extra_data)
                    training_time = time.perf_counter() - start_train
                                        
                    '''
                    if not learner.base_model.init and (cleaned_score > MIN_Z_SCORE and instance.y_index == prev_y_index and instance.y_index == 0): #Index for False Positives
                        idx = row + 1
                    elif not learner.base_model.init and (instance.y_index != prev_y_index): #Index for Starting and Ending of Cycles
                        idx = row + 1
                    else:
                        idx = 0
                    '''
                    '''
                    if idx != 0 :
                        learner.base_model.explain(headers=cols, region_study_list=REGION_STUDY, path=PATH_PLOT_FILE_NAME_INTERPRETATION, file_name=file_name.name.split("_")[0]+"_transf_"+TRANF+"_efeature_unimodel_"+EFEATURE+"_explanation_sf_"+str(learner)+str(idx))
                    '''

                    """
                    learner.plot_core_statistics(PATH_PLOT_FILE_NAME_SCORE, file_name.name.split("_")[0]+"_transf_"+TRANF+"_sf_"+str(learner)+"_efeature_"+EFEATURE+"_"+str(idx), label=instance.y_label)
                    """


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
            "method": str(learner),
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
