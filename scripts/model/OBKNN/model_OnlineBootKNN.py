
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import sys, os
from pathlib import Path
from scipy.linalg import inv
from scipy.spatial import distance
from scipy.stats import zscore
from capymoa.instance import Instance
import logging


#from capymoa.drift.detectors import GeometricMovingAverage

JITTER = 1e-6

import numpy as np

import numpy as np

def featurewise_distance(vec1, vec2, metric, p=2):
    vec1 = np.squeeze(vec1)
    vec2 = np.squeeze(vec2)

    if metric == "cityblock":
        distances = np.abs(vec1 - vec2)

    elif metric == "euclidean":
        distances = (vec1 - vec2) ** 2

    elif metric == "minkowski":
        distances = np.abs(vec1 - vec2) ** p

    elif metric == "chebyshev":
        diffs = np.abs(vec1 - vec2)
        distances = np.zeros_like(diffs, dtype=float)
        max_index = np.argmax(diffs)
        distances[max_index] = diffs[max_index]

    elif metric == "canberra":
        denom = np.abs(vec1) + np.abs(vec2)
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = np.abs(vec1 - vec2) / denom
            distances[np.isnan(distances)] = 0

    elif metric == "cosine":
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denom == 0:
            distances = np.zeros_like(vec1, dtype=float)
        else:
            cosine_similarity = (vec1 * vec2) / denom
            distances = 1 - cosine_similarity

    elif metric == "braycurtis":
        num = np.abs(vec1 - vec2)
        denom = np.abs(vec1 + vec2)
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = num / denom
            distances[np.isnan(distances)] = 0

    else:
        raise ValueError(f"Featurewise distance for '{metric}' is not supported.")

    signs = np.where(distances == 0, 0, np.sign(vec1 - vec2))
    
    return distances, signs
    
def transform_instance(instance:Instance, transf, prev_instance:Instance=None):
    if transf == "ZNORM":
        # Calculate Z - Normalized Array
        t_instance = zscore(instance.x, nan_policy='propagate', axis=0)
        t_instance = Instance.from_array(instance.schema, t_instance.reshape( -1))
        return t_instance
    else:
        return instance  # Default: no transformation

def median_of_means(data):
    """
    Computes the Median of Means (MoM) estimate for the mean.
    
    Args:
        data (array-like): The input dataset.
        k (int): The number of blocks to split the data into.
    
    Returns:
        float: The Median of Means estimate.
    """

    k = int(np.sqrt(len(data)))

    # Convert to a numpy array
    arr = np.array(data)
    
    # Step 1: Shuffle the data to ensure blocks are random
    np.random.shuffle(arr)
    
    # Step 2: Split the data into k blocks
    # np.array_split handles cases where data size is not perfectly divisible by k
    try:
        blocks = np.array_split(arr, k)
    except ValueError as e:
        logging.debug(f"Error: k ({k}) is likely larger than the number of data points.")
        logging.debug("Setting k=1 (standard mean).")
        blocks = [arr] # Fallback to standard mean

    # Step 3: Calculate the mean of each block
    block_means = [np.mean(block) for block in blocks]
    
    # Step 4: Return the median of the block means
    return np.median(block_means)

class OnlineBootKNN(AnomalyDetector):
    """
    Anomaly detection using an online ensemble of k-nearest neighbors (KNN).
    This class processes data in chunks, detects anomalies based on statistical thresholds,
    and uses an ensemble approach to increase robustness.

    Parameters:
    - schema: Optional, schema for data processing (if needed).
    - random_seed: Optional, seed for random number generation.
    - chunk_size: Size of each data chunk to process.
    - ensemble_size: Number of KNN models in the ensemble.
    - algorithm: KNN algorithm to use ('brute', 'kd_tree', etc.).
    - n_jobs: Number of CPU cores to use for parallel processing.
    - window_size: Size of the sliding window for statistics.
    - dmetric: Distance metric for KNN (e.g., 'cityblock', 'euclidean').
    - transf: Optional, transformation function to apply to data (e.g., Z-Normalization(ZNORM), First Order Difference(FOD), None, etc).
    - alpha_z_test: Significance level for statistical z-score test (default is 0.05).

    """
    def __init__(
        self,
        schema = None,
        random_seed = None,
        chunk_size = 30,
        ensemble_size = 240,
        algorithm = 'brute',
        n_jobs = -1,
        window_size = 120,
        dmetric="cityblock",
        transf="ZNORM",
        type_dist = "largest",
        alpha_z_test = 0.05,  # 5% significance level for anomaly detection
        alpha_ema = 0.01,  
        no_bootstrapp = False,
        no_z_score = False,
        update_distance_with_abnormal = True,
        update_mode_stats = "ema",
        k = 1
    ):
        # Initialize base class and set random seed
        super().__init__(schema=schema, random_seed=random_seed)
        
        self.random_generator = np.random.RandomState(random_seed)
        
        self.data_window = np.array([]) # Sliding window of recent data
        self.chunks = []  # Data chunks
        
        # Initialize parameters
        self.chunk_size = chunk_size
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.window_size = window_size
        self.dmetric = dmetric
        self.transf = transf
        self.alpha_z_test = alpha_z_test
        self.update_mode_stats = update_mode_stats
        self.alpha_ema = alpha_ema
        self.algorithm = algorithm
        self.k = k
        self.ensemble = []
        self.inv_cov = None

        self.type_dist = type_dist
        
        self.init = True  # Flag to indicate if the model is initialized
        self.last_value_is_anomaly = False 
        self.reset_threshold = 4200  # Number of values before resetting statistics 
        self.count_reset = None # Number of times the counter n is reseted 
        self.normal_reference_ch = None  # To track reference for normal data distribution in chunks
        self.abnormal_reference_ch = None  # To track reference for abnormal data distribution in chunks
        self.c_normal_reference_w = None  # To track reference for normal data distribution in current window
        self.p_normal_reference_w = None  # To track reference for normal data distribution in current window
        self.abnormal_reference_w = None  # To track reference for abnormal data distribution in current window
        self.z_critical_one_tail = norm.ppf(1 - self.alpha_z_test)  # Critical value for one-tailed test
        
        # Monitoring statistics for debugging/monitoring purposes
        self.z = 0.0  # Initial Z-Score Distance
        self.mean = np.nan
        self.mean_of_anomalies = np.nan
        self.std_dev = np.nan  # Standard deviation of the data
        self.std_dev_anomalies = np.nan
        self.min_dist = np.nan  # Minimum distance for anomaly detection
        self.n = 0  # Number of data points processed in Z-Score calculation
        self.n_anomalies = 0  # Number of anomalies detected
        self.accum_error = np.nan
        self.accum_error_anomalies = np.nan
        self.max_p_random_number = np.nan  # Probability for random anomaly detection
        
        self.z_scores_to_monitor = []
        self.means_to_monitor = []
        self.std_devs_to_monitor = []
        self.min_dists_to_monitor = []
        self.z_thresholds_to_monitor = []
        self.p_random_number_to_monitor = []
        self.ground_truth_to_monitor = []
        
        self.no_bootstrapp = no_bootstrapp        
        self.no_z_score = no_z_score
        self.update_distance_with_abnormal = update_distance_with_abnormal
        
        
        #self.drift_detector = GeometricMovingAverage()
        #self.last_value_is_a_drift = False 
        #self.u_detector = LeftSTAMPi(window_size = 10, n_init_train = 30) 
        self.n_extra_processing = 0 #Extra Data Processed by the method specially used with subsetobknn 
        self.n_abn_pos = 0


    def __str__(self):
        return "OnlineBootKNN"

    def train(self, instance: Instance):
        """
        Train the model with a new instance.
        """
        
        instance = transform_instance(instance, self.transf)

        
        self._learn_batch(instance.x)

    def _learn_batch(self, data):
        
        # FIX: Ensure data is at least a 1D array (handles scalars from transforms)
        #data = np.atleast_1d(data)
        
        if self.init:
            """Update the model with a new batch of data."""
            # Check if the data_window is empty
            if self.data_window.size == 0:

                self.data_window = data.reshape(1, -1)
            else:
                # The window already has data; stack the new data
                self.data_window = np.vstack([self.data_window, data])

                        
            if (len(self.data_window) >= self.window_size):
                
                #self.data_window = np.array(self.data_window)

                for i in range(self.ensemble_size):
                    
                    #if true "self.no_bootstrapp" do not apply bootstrapp (implementation for ablation study)
                    if self.no_bootstrapp:
                        data_chunk = self.data_window
                    else:
                        indices = self.random_generator.choice(len(self.data_window), size=self.chunk_size, replace=True)
                        data_chunk = self.data_window[indices]
                    
                    self.chunks.append(data_chunk)
                    

                    if self.type_dist == "largest":
                        # 1. Set default empty params
                        metric_params = {}

                        # 2. Now, the list comprehension is simple and clean.
                        # 
                        nn_model = NearestNeighbors(
                            n_neighbors=self.k,
                            algorithm=self.algorithm,
                            n_jobs=self.n_jobs,
                            metric=self.dmetric,
                            metric_params=metric_params
                        )

                        # 3. Fit
                        nn_model.fit(data_chunk)

                        # 4. Store
                        self.ensemble.append(nn_model)

                #self.data_window = []  # Clear window after update
                #self.c_normal_reference_w = np.mean(self.data_window, axis=0)
                self.init = False
        else:
            
            # Remove the first element of the sliding window if it's full
            self.data_window = self.data_window[1:]
                
            # Updated the main window
            self.data_window = np.vstack([self.data_window, data])
            
            
            # Train each model in the ensemble using bootstrap sampling
            for i in range(self.ensemble_size):
                
                self.max_p_random_number = 0 #variable to monitor p_random_number

                #check if the method will apply bootstrapp (for ablation study)
                if self.no_bootstrapp:
                    p_random_number = 1
                
                #check if the last value is anomalous to exclude it
                elif self.last_value_is_anomaly:
                    p_random_number = 0
                
                else:
                    p_random_number = self.random_generator.poisson(1)
                
                if self.max_p_random_number < p_random_number:
                    self.max_p_random_number = p_random_number
                    
                for j in range(p_random_number):
                    # Remove the first element
                    self.chunks[i] = self.chunks[i][1:]
                    
                    # Add the new vector to the end of the array
                    self.chunks[i] = np.vstack([self.chunks[i], data])
                    
                if self.type_dist == "largest":
                    # Fit the model with the bootstrap sample
                    self.ensemble[i].fit(self.chunks[i])
                    #logging.debug(f"Updated Model: {i}")

    def score_instance(self, instance: Instance):
        """
        Scores an instance using the models in the ensemble.
        Each model predicts distances, and the minimum distance is returned.
        """
        
        instance = transform_instance(instance, self.transf)
        data = instance.x.reshape(1, -1)  # Reshape data to match model expectations

        

        distances = []
        references = []
        
        for i in range(self.ensemble_size):
            try:

                if self.type_dist == "largest":
                    # Ensure the model is fitted before making predictions
                    check_is_fitted(self.ensemble[i])
                    # 1. Capture both the distances AND the indices
                    dist, idx = self.ensemble[i].kneighbors(data)
                    # 2. Get the distance and index of the farthest neighbor.
                    d = dist[0][-1]
                    index_f = idx[0][-1]
                    # 3. Use the correct index to retrieve the neighbor's data vector
                    vf_ch = self.chunks[i][index_f] 



                elif self.type_dist == "mean": #compute the mean with respect the centroid so the mean of each feature
                    
                    vf_ch = np.mean(self.chunks[i], axis=0)
                    d, _ = featurewise_distance(vf_ch, data, metric=self.dmetric).sum()  

                distances.append(d)  
                references.append(vf_ch)
                
            except NotFittedError:
                #logging.debug(f"Model {i} is not fitted.")
                distances.append(0)  
            except IndexError:
                #logging.debug(f"Vector {i} do not have completed the minimum window.")
                distances.append(0)
            except Exception as exc:
                #logging.error(f"An error occurred while scoring the instance: {exc}")
                distances.append(None)
        
        self.min_dist = np.min(distances)  # Find the minimum distance

        """
        self.drift_detector.add_element(min_dist)
        if self.drift_detector.detected_change():
            logging.debug('Change detected in data.. ')
            
            self.last_value_is_a_drift = True
            #self.update_distance_with_abnormal = False
            #self.drift_detector.reset()
        else:
            self.last_value_is_a_drift = False
            #self.update_distance_with_abnormal = True
        """

        # Return the minimum distance among the models (ablation study)
        if self.no_z_score:
            return self.min_dist
        
        if not self.init:
            if  (self.n == 0) | (self.n > self.reset_threshold):
                #Init Stats
                self.count_reset = 0 if self.count_reset is None else self.count_reset + 1 #number of times the counter n is reseted 
                self.start_statistics(self.min_dist)
            elif (self.n == 1):
                #z-score is not defined for the first instance
                self.update_statistics_normal(self.min_dist)
            else:
                #Update Stats
                self.update_z_score(self.min_dist)
                
                if not self.last_value_is_anomaly:
                    self.update_statistics_normal(self.min_dist)
                else:
                    self.update_statistics_abnormal(self.min_dist) #update_statistics_normal and update_statistics_abnormal are equal if self.update_distance_with_abnormal == True 
                    min_pos_dist = np.argmin(distances)  # Find the minimum position distance
                    self.normal_reference_ch = references[min_pos_dist]
                    self.abnormal_reference_ch = data.reshape(-1)
            

        return self.z

    def start_statistics(self, new_dist):
        """
        Initialize statistics when the reset threshold is reached or at an initial step.
        """
        self.z = 0.0          
        self.n = 1
        self.mean = new_dist
        self.accum_error = JITTER # Avoid division by zero in std_dev calculation with a very small value
        #self.accum_error = 0.0 # original algorithm with initial 0 values   
        self.last_value_is_anomaly = False
        self.n_extra_processing = 0 #Extra Data Processed by the method specially used with subsetobknn 

    def update_statistics_normal(self, new_dist):
        """
        Increment statistics and update mean and standard deviation using Welford's algorithm or EMA.
        """
        self.n += 1 

        if self.update_mode_stats == 'welford':
            # --- Welford's Algorithm Logic ---
            delta = new_dist - self.mean
            self.mean += delta / self.n
            delta2 = new_dist - self.mean
            self.accum_error += delta * delta2
            self.std_dev = math.sqrt(self.accum_error / (self.n - 1))

        else: # ema
            # --- EMA Logic ---
            delta = new_dist - self.mean
            self.mean += self.alpha_ema * delta
            self.accum_error = (1 - self.alpha_ema) * (self.accum_error + self.alpha_ema * delta**2)
            self.std_dev = math.sqrt(self.accum_error)
    
    def update_statistics_abnormal(self, new_dist):
        
        self.n += 1
        self.n_anomalies += 1

        if self.update_distance_with_abnormal == True:
            
            if self.update_mode_stats == 'welford':
                # --- Welford's Algorithm Logic ---
                delta = new_dist - self.mean
                self.mean += delta / self.n
                delta2 = new_dist - self.mean
                self.accum_error += delta * delta2
                self.std_dev = math.sqrt(self.accum_error / (self.n - 1))

            else: # ema
                # --- EMA Logic ---
                delta = new_dist - self.mean
                self.mean += self.alpha_ema * delta
                self.accum_error = (1 - self.alpha_ema) * (self.accum_error + self.alpha_ema * delta**2)
                self.std_dev = math.sqrt(self.accum_error)
        else:
            
            if self.n_anomalies == 1:
                
                self.mean_of_anomalies = new_dist
                self.accum_error_anomalies = JITTER # Avoid division by zero in std_dev calculation with a very small value
                                
            else:

                if self.update_mode_stats == 'welford':
                    # --- Welford's Algorithm Logic ---
                    delta = new_dist - self.mean_of_anomalies
                    self.mean_of_anomalies += delta / self.n_anomalies
                    delta2 = new_dist - self.mean_of_anomalies
                    self.accum_error_anomalies += delta * delta2
                    if self.n_anomalies > 1:
                        self.std_dev_anomalies = math.sqrt(self.accum_error_anomalies / (self.n_anomalies - 1))

                else: # ema
                    # --- EMA Logic ---
                    delta = new_dist - self.mean_of_anomalies
                    self.mean_of_anomalies += self.alpha_ema * delta
                    self.accum_error_anomalies = (1 - self.alpha_ema) * (self.accum_error_anomalies + self.alpha_ema * delta**2)
                    self.std_dev_anomalies = math.sqrt(self.accum_error_anomalies)

    def update_z_score(self, new_dist):
        """
        Calculate the z-score and check for anomalies.
        """
        if self.std_dev != 0 and not pd.isna(self.std_dev):
            self.z = (new_dist - self.mean) / self.std_dev
        
        else:
            self.z = np.nan
            
        # Identify if the instance is an anomaly based on the z-score
        if self.z > self.z_critical_one_tail and not pd.isna(self.z):
            
            self.last_value_is_anomaly = True
            self.n_abn_pos = self.count_reset*self.reset_threshold + self.n + self.window_size + self.n_extra_processing 
        else:
            self.last_value_is_anomaly = False
 
    def predict(self, data: np.ndarray):
        """
        Predict method (to be implemented by subclasses).
        """
        raise NotImplementedError("The 'predict' method must be implemented by subclasses.")

    def explain(self, headers, region_study_list, path: str, file_name: str):

            headers = headers.astype(float)
            region_study_list = np.array(region_study_list).astype(str)

            if not self.last_value_is_anomaly:
                return

            if self.normal_reference_ch is None or self.abnormal_reference_ch is None:
                raise ValueError("Error: Normal reference or input data is None.")

            if len(self.normal_reference_ch) != len(self.abnormal_reference_ch):
                raise ValueError("Error: Normal reference and input data must have the same length.")

            differences, signs = featurewise_distance(self.abnormal_reference_ch, self.normal_reference_ch, metric=self.dmetric)
            colors = np.where(signs > 0, 'orange', 'blue')

            fig, ax1 = plt.subplots(figsize=(14, 7))
            
            ax1.bar(headers, differences, color=colors, alpha=1.0)
            
            ax1.bar([headers[0]], [0], color='orange', label=f"Positive Difference")
            ax1.bar([headers[0]], [0], color='blue', label=f"Negative Difference")

            ax1.set_ylabel('Feature Differences', fontsize=14)
            ax1.set_xlabel('Wavelengths (nm)', fontsize=14)

            min_diff = differences.min()
            max_diff = differences.max()

            lower_limit = min(0, min_diff * 1.05)
            upper_limit = max_diff * 1.05
            
            ax1.set_ylim(lower_limit, upper_limit)
            
            ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

            for i, rs in enumerate(region_study_list):    
                rs_s = float(rs.split(":")[0])
                rs_f = float(rs.split(":")[1])
                comp = str(rs.split(":")[2])
                
                ax1.axvline(x=rs_s, color='grey', linestyle='dotted', linewidth=1.5, alpha=0.8)
                ax1.axvline(x=rs_f, color='grey', linestyle='dotted', linewidth=1.5, alpha=0.8)

                ax1.text(rs_f, upper_limit*0.85*(len(region_study_list)-i)/len(region_study_list), f' RS{i+1} ({comp}) at: {rs_s}', fontsize=10)        
                
            ax1.legend(loc="upper left", fontsize=10)

            n_obknn = self.count_reset*self.reset_threshold + self.n + self.window_size 
            n_total = self.count_reset*self.reset_threshold + self.n + self.window_size + self.n_extra_processing 

            title_text = (
                f"Feature Differences using {self.dmetric.capitalize()} Distance\n"
                f"Z-score: {self.z:.2f} | Potential Anomalies: {self.n_anomalies} (Included: {self.update_distance_with_abnormal})\n"
                f"Total Instances: {n_total} | OBKNN Instances: {n_obknn} | Last Anomaly Position: {self.n_abn_pos}"
            )
            plt.title(title_text, fontsize=14)

            plot_path = os.path.join(path, f"{file_name}_anomaly_explanation.pdf")
            plt.savefig(plot_path, format="pdf", bbox_inches='tight')
            logging.debug(f"Plot saved at {plot_path}")
            
            plt.close()
    
    def monitor_core_statistics_training(self):
        # monitoring stats
        logging.debug(
            f"\n{'='*30}\n"
            f"Training Stats.\n"
            f"Status: [Initial Phase: {self.init} ]\n"
            f"shape - data_window: {np.shape(self.data_window)}  \n"
            f"shape - chunks: {np.shape(self.chunks)}  \n"
            f"{'='*30}"
        )

    def monitor_core_statistics_scoring(self):
        # monitoring stats
        logging.debug(
            f"\n{'='*30}\n"
            f"Scoring Stats.\n"
            f"Status: [Anomaly: {self.last_value_is_anomaly}]\n"
            f"Stats:  Acum Mean: {self.mean:.2f} | Std: {self.std_dev:.2f} | Min Dist: {self.min_dist:.2f} | N: {self.n} \n"
            f"Anoms:  Count: {self.n_anomalies} | P-Rand: {self.max_p_random_number:.2f}\n"
            f"Z-Test: Score: {self.z:.2f} | Crit: {self.z_critical_one_tail:.4f}\n"
            f"{'='*30}"
        )
    
    def plot_core_statistics(self, path: str, file_name: str, label: int = 0):
        self.p_random_number_to_monitor.append(self.max_p_random_number) 
        self.means_to_monitor.append(self.mean)
        self.std_devs_to_monitor.append(self.std_dev)
        self.min_dists_to_monitor.append(self.min_dist)
        self.ground_truth_to_monitor.append(label)
        
        self.z_scores_to_monitor.append(self.z)
        self.z_thresholds_to_monitor.append(self.z_critical_one_tail)

        gt_indices = [i for i, x in enumerate(self.ground_truth_to_monitor) if int(x) == 1]
        x_axis = np.arange(len(self.means_to_monitor))
        gt_style = {'color': 'darkorange', 'edgecolors': 'darkorange', 'marker': 'X', 's': 50, 'zorder': 5}

        fig, ax1 = plt.subplots(figsize=(10, 6))
        means = np.array(self.means_to_monitor, dtype=float)
        stds = np.array(self.std_devs_to_monitor, dtype=float)
        min_dists = np.array(self.min_dists_to_monitor, dtype=float)

        ax1.plot(x_axis, min_dists, label='Min Dist', color='green', marker='s', markersize=2, alpha=0.7)
        if gt_indices:
            ax1.scatter(gt_indices, min_dists[gt_indices], label='Anomaly (GT)', **gt_style)

        ax1.plot(x_axis, means, label='Mean', color='blue', linestyle='--', marker='o', markersize=2)
        ax1.fill_between(x_axis, means - stds, means + stds, color='blue', alpha=0.2, label='Confidence Interval (±1 Std)')

        ax1.set_ylabel('Distance and Mean Statistics')
        ax1.grid(True, linestyle='--', alpha=0.6)

        plt.title('Min Dist vs Mean & Std Dev')
        ax1.legend(loc='upper left')
        plt.savefig(os.path.join(path, f"{file_name}_mean_min.pdf"), format="pdf")
        plt.close(fig)

        fig, ax3 = plt.subplots(figsize=(10, 6))
        z_scores = np.array(self.z_scores_to_monitor)

        ax3.plot(z_scores, label='Z Score', color='red', marker='o', markersize=4)
        ax3.plot(self.z_thresholds_to_monitor, label='Threshold', color='purple', linestyle=':', marker='s', markersize=2)
        if gt_indices:
            ax3.scatter(gt_indices, z_scores[gt_indices], label='Anomaly (GT)', **gt_style)
        ax3.set_ylabel('Z Score & Threshold')
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        plt.title('Z-Score, Threshold')
        ax3.legend(loc='upper left')
        plt.savefig(os.path.join(path, f"{file_name}_zscore_extra.pdf"), format="pdf")
        plt.close(fig)

        fig, ax5 = plt.subplots(figsize=(10, 6))

        ax5.plot(z_scores, label='Z Score', color='red', marker='o', markersize=4)
        if gt_indices:
            ax5.scatter(gt_indices, z_scores[gt_indices], label='Anomaly (GT)', **gt_style)
        ax5.set_ylabel('Z Score', color='red')
        ax5.grid(True, linestyle='--', alpha=0.6)

        ax6 = ax5.twinx()
        ax6.plot(self.p_random_number_to_monitor, label='P Random Num', color='black', linestyle='--', marker='^', markersize=2)
        ax6.set_ylabel('Model & P-Number')
        
        plt.title('Z-Score vs Model & P-Random Number')
        fig.legend(loc='upper left')
        plt.savefig(os.path.join(path, f"{file_name}_zscore_model_pnum.pdf"), format="pdf")
        plt.close(fig)
        
        logging.debug(f"All reorganized plots saved to {path}")
        
    def extra_processing(self):
        self.n_extra_processing += 1 

if __name__ == "__main__":



    # Get the path to the current directory and go up to scripts path

    current_dir = Path(__file__).resolve().parent.parent.parent

    

    sys.path.append(str(current_dir))



    # Go one level up to include datasets

    current_dir = current_dir.parent



    from capymoa.stream import NumpyStream

    #from capymoa.evaluation import AnomalyDetectionEvaluator



    from data_utils import calculate_performance_metrics

    from model_utils import clean_score

    import time

    from model.mDragstream.mdragstream import mDragStream

    

    from pysad.models import ExactStorm, IForestASD, KitNet, LODA, RobustRandomCutForest, RSHash, xStream

    from capymoa.anomaly import OnlineIsolationForest, HalfSpaceTrees as HStreeCapy

    from dSalmon.outlier import SWKNN , SWLOF

    import json

    #VALIDATED CONFIGS
    #PATH_SETTING_EXP = 'scripts/model/OBKNN/config_ifasd/config_ifasd_exp_eval_BV5_only_spectra_deg.json'
    
    #DIFF STD
    PATH_SETTING_EXP = 'scripts/model/OBKNN/config_swlof/config_swlof_exp_eval_BV5_only_spectra_deg.json'

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



        cols = df.iloc[:, COLS_POS_FMIN:COLS_POS_FMAX].columns

        score_column = 'Score'

        error_column = 'Error'

        label_column = df.columns[COLS_POS_LABEL]  

        cols = df.columns[slice(COLS_POS_FMIN, COLS_POS_FMAX)]

        col_target = df.columns[-1]                

        logging.debug("Name DS: %s", file_name)
        logging.debug("# Total Points: %d", len( df[cols].values))
        logging.debug("# Anomalies: %d", len( df[col_target].values))
        logging.debug("# of Columns: %d", len(df.columns))
        logging.debug("# of Columns to use:  %d", len(cols))
        logging.debug("Columns:  %s", cols)

        



        

        stream = NumpyStream(df[cols].values, df[col_target].astype(int).values, dataset_name='PV', feature_names=cols)

        schema = stream.get_schema()



        if f_break:
            break

        

        for i in range(NUMBER_RUNS):



            stream.restart()

            scores = []

            raw_scores = []

            errors = []

            row = 0



            if P_WINDOW_SIZE != None: 
                WINDOW_SIZE = max(1, int(len(df) * P_WINDOW_SIZE))

            
            if LEARNER_NAME == "OnlineBootKNN":
                learner = OnlineBootKNN(schema=schema, random_seed=i, window_size=WINDOW_SIZE, type_dist=TYPE_DIST, chunk_size=CHUNK_SIZE, ensemble_size=ENSEMBLE_SIZE, dmetric=DMETRIC, transf=TRANF, alpha_z_test=ALPHA_Z_TEST, algorithm=ALGO, no_bootstrapp=NO_BOOTSTRAPP, no_z_score=NO_ZSCORE, update_mode_stats=UPDATE_MODE_STATS, update_distance_with_abnormal=UPDATE_WITH_ABNORMAL, alpha_ema=ALPHA_EMA, n_jobs = N_JOBS)
            elif LEARNER_NAME == "IForestASD":
                learner = IForestASD(window_size=WINDOW_SIZE, initial_window_X=None)
            elif LEARNER_NAME == "KitNet":
                learner = KitNet(hidden_ratio=0.75, learning_rate=0.1, max_size_ae=10, grace_feature_mapping=WINDOW_SIZE, grace_anomaly_detector=WINDOW_SIZE)
            elif LEARNER_NAME == "OnlineIsolationForest":
                learner = OnlineIsolationForest(schema=schema, window_size=WINDOW_SIZE, random_seed=i, growth_criterion='fixed', max_leaf_samples=64, n_jobs=-1, num_trees=64)
            elif LEARNER_NAME == "ExactStorm":
                learner = ExactStorm(window_size=WINDOW_SIZE, max_radius=900)
            elif LEARNER_NAME == "mDragStream_mahalanobis":
                learner = mDragStream(maxNbClusters=30, maxClusterSize=30, metric="mahalanobis", inv_cov=None, n_init=WINDOW_SIZE)
            elif LEARNER_NAME == "mDragStream_euclidean":
                learner = mDragStream(maxNbClusters=30, maxClusterSize=50, metric="euclidean", inv_cov=None, n_init=WINDOW_SIZE)
            elif LEARNER_NAME == "RSHash":
                learner = RSHash(sampling_points=WINDOW_SIZE, decay=0.05, feature_maxes=[np.inf, 2000], feature_mins=[0], num_components=50, num_hash_fns=1)
            elif LEARNER_NAME == "RobustRandomCutForest":
                learner = RobustRandomCutForest(shingle_size=WINDOW_SIZE, num_trees=4, tree_size=256)
            elif LEARNER_NAME == "SWKNN":
                learner = SWKNN(window=WINDOW_SIZE, k=10, k_is_max=False, metric="cityblock", min_node_size=5, max_node_size=20, split_sampling=5)
            elif LEARNER_NAME == "SWLOF":
                K = config["hyperparameters"].get("k", None)
                K_IS_MAX = config["hyperparameters"].get("k_is_max", None)
                SIMPLIFIED = config["hyperparameters"].get("simplified", None)
                METRIC = config["hyperparameters"].get("metric", None)
                
                learner = SWLOF(window=WINDOW_SIZE, k=K, k_is_max=K_IS_MAX, simplified=SIMPLIFIED, metric=METRIC)
            elif LEARNER_NAME == "xStream":
                learner = xStream(window_size=WINDOW_SIZE, depth=25, n_chains=100, num_components=50)
            elif LEARNER_NAME == "HStreeCapy":
                learner = HStreeCapy(schema=schema, window_size=WINDOW_SIZE, number_of_trees=25, anomaly_threshold=0.5, size_limit=0.1, max_depth=10, random_seed=i)
            else:
                learner = None

            #monitored_metric = AUPRMetric()



            np.random.seed(i)

            if f_break:
                break
            
            for row, instance in enumerate(stream):
        
                time.sleep(SLEEP_TIME)

                #instance = stream.next_instance()
                #row = row + 1 

                logging.debug("########################################################")    
                #logging.debug(f'A new instance ({row+1})...label: {instance.y_label}, index: {instance.y_index}')
                logging.debug(f'The new instance: {instance.x}, index: {instance.y_index}')
                logging.debug("########################################################")   
                
                
                # --- API-Specific Logic for Training and Scoring ---
                # time.perf_counter() is used for precise timing within this process.
                if hasattr(learner, "fit_partial"):  # Pysad models
                    start_train = time.perf_counter()
                    learner.fit_partial(instance.x)
                    training_time = time.perf_counter() - start_train
                    start_score = time.perf_counter()
                    score = learner.score_partial(instance.x)
                    cleaned_score, error_score = clean_score(score)
                    scoring_time = time.perf_counter() - start_score
                elif hasattr(learner, "train"):  # Capymoa models
                    start_score = time.perf_counter()
                    score = learner.score_instance(instance)
                    cleaned_score, error_score = clean_score(score)
                    #learner.monitor_core_statistics_scoring()

                    scoring_time = time.perf_counter() - start_score
                    start_train = time.perf_counter()
                    learner.train(instance)
                    training_time = time.perf_counter() - start_train
                    
                    
                    if cleaned_score >= MIN_Z_SCORE:
                        learner.explain(headers=cols, region_study_list=REGION_STUDY, path=PATH_PLOT_FILE_NAME_INTERPRETATION, file_name=file_name.name.split("_")[0]+"_transf_"+TRANF+"_explanation_obknn_"+str(i))
                    
                    """
                    learner.plot_core_statistics(PATH_PLOT_FILE_NAME_SCORE, file_name.name.split("_")[0]+"_transf_"+TRANF+"_obknn_"+str(i), label=instance.y_label)
                    """
                elif hasattr(learner, "fit_predict"):  # dSalmon models
                    training_time = 0
                    start_score = time.perf_counter()
                    score = learner.fit_predict(instance.x)
                    cleaned_score, error_score = clean_score(score)
                    scoring_time = time.perf_counter() - start_score
                else:
                    raise AttributeError(f"Model {learner.__class__.__name__} has no recognized training/scoring method.")

                

                
                scores.append(cleaned_score)
                raw_scores.append(score)
                errors.append(error_score)


                if np.isnan(cleaned_score):
                    f_break = True
                    break


            df[score_column+str(i)] = list(scores)
            df[score_column+"_raw"+str(i)] = list(raw_scores)
            df[error_column+str(i)] = list(errors)

            try:
                results = calculate_performance_metrics(df, label_column, score_column+str(i), t_window_size=WINDOW_SIZE, score_direction=SCORE_DIR)
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
