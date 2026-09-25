## Configuration Parameters

### Paths
Directory locations for data ingestion and output generation.

| Variable | Description |
|---|---|
| **data_path** | Path pointing to the raw dataset directory to use. |
| **plot_file_interpretation** | Output directory to save visualizations related to anomaly explanations. |
| **plot_file_score** | Output directory to save visualizations of the monitoring metrics and scores. |

### Experiment Configuration
Settings for the general execution environment and datasets.

| Variable | Description |
|---|---|
| **number_runs** | Total number of independent executions for the experiment. |
| **score_dir** | Direction degree for abnormal cases of the anomaly score ("direct" for anomalies that are associated with a positive score, "inverse" for the other case). |
| **sleep_time** | Pause duration in seconds between each step in the streaming. |
| **datasets_list** | Array containing the prefixes of the datasets to be evaluated. |

### Hyperparameters
Hyperparameters that control the behavior of the methods.

| Variable | Description |
|---|---|
| **transf** | Preprocessing for data transformation applied to each spectrum before calculating distances (ZNORM for znormalized spectra or NONE for raw spectra). |
| **window_size** | Number of spectral instances for initial training (positive integer value). |
| **p_window_size** | Alternative to window_size; it is possible to define the proportion of the dataset as a window size so window_size must be "null" (float value between 0 and 1). |
| **chunk_size** | Size of the chunks created from the initial window size (positive integer value). |
| **ensemble_size** | Size of the ensemble of chunks created from the initial window size (positive integer value). |
| **no_bootstrapp** | Boolean flag to disable bootstrap from initial window size (boolean value). |
| **no_z_score** | Boolean flag to disable the Z-test so raw minimum distances are used instead (boolean value). |
| **update_distance_with_abnormal** | Boolean flag allowing the model to update with minimum distances that come from abnormal spectra (boolean value). |
| **dmetric** | Distance metric used for comparison ("cityblock", "euclidean"). |
| **algorithm** | Method to compute nearest neighbors search ("brute", "kd_tree"). |
| **alpha_z_test** | Statistical significance level for hypothesis testing in the Z-test (float value between 0 and 1). |
| **alpha_ema** | Smoothing factor for the Exponential Moving Average calculations (float value between 0 and 1). |
| **type_dist** | Method to compute the minimum distance; it can be the farthest distance for KNN or the distance with respect to the mean vector ("largest", "mean"). |
| **update_mode_stats** | Algorithm to update mean and standard deviation of the Z-test during the stream ("ema", "welford"). |
| **n_jobs** | Number of CPU cores allocated for parallel processing. A value of -1 uses all available cores (integer value). |
| **min_value** | Lower value limit to filter data with the extra feature (float value). |
| **max_value** | Maximum value limit to filter data with the extra feature (float value). |

### Data Processing
Rules for handling data and extracting extra features, signaling regions of interest, and setting the explanation threshold.

| Variable | Description |
|---|---|
| **efeature** | Name of the extra feature selected for the specific processing ("MomSpectra", "Voltage", "Current", "Pressure"). |
| **feature_parameters** | Configuration object defining parameters for additional features. It includes the column position, lower limit, and upper limit for each specific feature. |
| **min_z_score** | Minimum Z-score threshold required to execute the explanation of an abnormal case (float value). |
| **region_study** | Array of spectral regions in the spectra associated with chemical compounds defined by lower bound, upper bound, and the corresponding chemical element. |
| **cols_pos_fmin** | Integer representing the starting column index for the first wavelength of the spectra in the dataset. |
| **cols_pos_fmax** | Integer representing the ending column index for the last wavelength of the spectra in the dataset. |
| **cols_pos_label** | Integer representing the column index that contains the ground truth labels. |