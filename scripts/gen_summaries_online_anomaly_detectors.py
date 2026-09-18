import os
import sys
import pandas as pd
from pathlib import Path
from data_utils import calculate_performance_metrics
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import argparse

# --- Setup ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hiding Warnings
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")

def get_summary_config(mode, current_dir, output_dir):
    """
    Returns the configuration for input directories and output file path
    based on the selected mode.
    """
    if mode == 'sota_and_own_tuning_pds_pv':
        logging.info("Selected Mode: Summarizing SOTA and Own Method Tuning (Public DS TAO+ PV DS DA)")
        input_dirs = [
            current_dir / "datasets" / "processed_sota_method_tuning_public_ds",
            current_dir / "datasets" / "processed_sota_method_tuning_pv_ds",
            current_dir / "datasets" / "processed_own_method_V2_tuning_public_ds",
            current_dir / "datasets" / "processed_own_method_V2_tuning_pv_ds"
        ]
        output_file_path = output_dir / 'summary_results_sota_and_own_method_tuning_pds_and_pv.xlsx'
    
    elif mode == 'sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score':
        logging.info("Selected Mode: Summarizing SOTA and Own Method Evaluation correcting score for RSHASH and ESTORM (Public+PV DS BV3, BV4 and BV5)")
        input_dirs = [
            current_dir / "datasets" / "processed_sota_method_eval_public_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_public_ds",
            current_dir / "datasets" / "processed_sota_method_eval_public_v2_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_public_v2_ds",
            current_dir / "datasets" / "processed_sota_method_eval_pv_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_pv_ds",
            current_dir / "datasets" / "processed_sota_method_eval_pv_bv5_gtv2_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_pv_bv5_gtv2_ds",
        ]
        output_file_path = output_dir / 'summary_results_sota_and_own_method_eval_pds_and_pv_bv3_bv4_bv5_inv_score.xlsx'    

    elif mode == 'sota_and_own_eval_pds_pv_bv3_bv4':
        logging.info("Selected Mode: Summarizing SOTA and Own Method Evaluation (Public+PV DS BV3 and BV4)")
        input_dirs = [
            current_dir / "datasets" / "processed_sota_method_eval_public_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_public_ds",
            current_dir / "datasets" / "processed_sota_method_eval_public_v2_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_public_v2_ds",
            current_dir / "datasets" / "processed_sota_method_eval_pv_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_pv_ds",
        ]
        output_file_path = output_dir / 'summary_results_sota_and_own_method_eval_pds_and_pv_bv3_bv4.xlsx'  
    
    elif mode == 'sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score':
        logging.info("Selected Mode: Summarizing SOTA and Own Method Evaluation (Public+PV DS BV5 with and without Extra Feature: Current)")
        input_dirs = [
            current_dir / "datasets" / "processed_own_method_V2_with_ef_current_eval_pv_bv5_gtv2_ds",
            current_dir / "datasets" / "processed_sota_method_with_ef_current_eval_pv_bv5_gtv2_ds",
            current_dir / "datasets" / "processed_sota_method_eval_pv_bv5_gtv2_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_pv_bv5_gtv2_ds",
        ]
        output_file_path = output_dir / 'summary_results_sota_and_own_method_eval_pv_bv5_with_current_gtv2_ds_inv_score.xlsx'   

    elif mode == 'sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2':
        logging.info("Selected Mode: Summarizing SOTA and Own Method Evaluation (Public+PV DS BV5 with and without Extra Feature: Current)")
        input_dirs = [
            current_dir / "datasets" / "processed_own_method_V2_with_ef_current_eval_pv_bv5_gtv2_ds_default",
            current_dir / "datasets" / "processed_sota_method_with_ef_current_eval_pv_bv5_gtv2_ds",
            current_dir / "datasets" / "processed_sota_method_eval_pv_bv5_gtv2_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_pv_bv5_gtv2_ds",
        ]
        output_file_path = output_dir / 'summary_results_sota_and_own_method_eval_pv_bv5_with_current_gtv2_ds_inv_score_v2.xlsx'       
    
    elif mode == 'sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv':
        logging.info("Selected Mode: Summarizing SOTA and Own Method Evaluation (Public+PV DS BV5 for features of plasma and Current)")
        input_dirs = [
            current_dir / "datasets" / "processed_own_method_V2_eval_pv_bv5_gtv2_ds_for_fcurrent",
            current_dir / "datasets" / "processed_own_method_V2_eval_pv_bv5_gtv2_ds_for_fplasma",
            current_dir / "datasets" / "processed_sota_method_eval_pv_bv5_gtv2_ds_for_fcurrent",
            current_dir / "datasets" / "processed_sota_method_eval_pv_bv5_gtv2_ds_for_fplasma",
        ]
        output_file_path = output_dir / 'summary_results_sota_and_own_method_eval_pv_bv5_for_fplasma_fcurrent_pv.xlsx'   
    elif mode == 'sota_mdragstream_and_own_eval_pds':
        logging.info("Selected Mode: Summarizing SOTA, mDragstream and Own Method Evaluation (Public DS only)")
        input_dirs = [
            current_dir / "datasets" / "processed_sota_method_eval_public_ds",
            current_dir / "datasets" / "processed_own_method_V2_eval_public_ds",
            current_dir / "datasets" / "processed_sota_method_eval_public_ds_subsequence_mdragstream",
            current_dir / "datasets" / "processed_sota_method_eval_public_ds_point_mdragstream",
        ]
        output_file_path = output_dir / 'summary_results_sota_mgragstream_and_own_method_eval_public_ds.xlsx'    
        
    elif mode == 'test':
        logging.info("Selected Mode: Summarizing Test directories")
        input_dirs = [
            current_dir / "datasets" / "processed_tuning_test",
            current_dir / "datasets" / "processed_eval_test"
        ]
        output_file_path = output_dir / 'summary_results_test_pv_ds.xlsx'
        
    else:
        # This should not be reachable due to argparse 'choices'
        raise ValueError(f"Invalid mode specified: {mode}")

    return {"input_dirs": input_dirs, "output_file_path": output_file_path}

# Define columns to use at a global scope
usecols = [
    "iteration", "method", "param", "ground_truth", "cleaned_score",
    "training_time", "scoring_time", "score"
]

def process_group(group, file_name, iteration, method, mwp, tw, p_window_size):
    """Calculates performance metrics for a single group within a file."""
    try:
        direction = 'direct'

        
        # Inverting HStree score depending on version of Capymoa (invert if v0.9.0-v0.11.0) 
        # Inverting RSHash since implementation in pysad gives normal points to higuer scores (density).
        # Inverting EStorm since implementation in pysad gives normal points to higuer scores (% of normal neighbours in radius).
        if method in ["HStree", "SFHStree", "MFHStree","RSHash", "SFRSHash", "EStorm", "SFEStorm"]: 
            direction = 'inverse'
        else:
            direction = 'direct'
        
        
        roc_auc, pr_auc, max_f1, metrics, roc_auc_wtd, pr_auc_wtd, max_f1_wtd, pct_detection, pct_false_positives, tn, fp, fn, tp, best_threshold = calculate_performance_metrics(group, "ground_truth", "cleaned_score", t_window_size=tw, score_direction=direction)
        
        if "_OC_" in file_name or "FP" in file_name:#conditional for comparisons of evaluation with just extra features
            scenario = file_name.split("_")[0] + file_name.split("_")[2]
            name = file_name.split("_")[1] + file_name.split("_")[2]
        else:
            scenario = file_name.split("_")[0]
            name = file_name.split("_")[1]

        # Extract all metrics safely
        return {
            "iteration": iteration,
            "cod_scenario": scenario,
            "name_scenario": name,
            "method_window_and_param": mwp,
            "p_window_size": p_window_size,
            "window_size": tw,
            "raw_roc_auc": roc_auc,
            "raw_pr_auc": pr_auc,
            "raw_max_f1": max_f1,
            "raw_roc_auc_wtd": roc_auc_wtd,
            "raw_pr_auc_wtd": pr_auc_wtd,
            "raw_max_f1_wtd": max_f1_wtd,
            "raw_pct_detection": pct_detection, # Percentage of Detection same as Recall
            "raw_pct_false_positives": pct_false_positives, # False Positive Rate
            "tn": tn, 
            "fp": fp, 
            "fn": fn, 
            "tp": tp, 
            "best_threshold": best_threshold,
            
            "auc_roc": metrics.get('AUC_ROC'),
            "auc_pr": metrics.get('AUC_PR'),
            "precision": metrics.get('Precision'),
            "f_metric": metrics.get('F'),   
            "precision_at_k": metrics.get('Precision_at_k'), 
            "rprecision": metrics.get('Rprecision'),    
            "rrecall": metrics.get('Rrecall'),   
            "rf": metrics.get('RF'),     
            "r_auc_roc": metrics.get('R_AUC_ROC'),   
            "r_auc_pr": metrics.get('R_AUC_PR'), 
            "vus_roc": metrics.get('VUS_ROC'),  
            "vus_pr": metrics.get('VUS_PR'),        
            "affiliation_precision": metrics.get('Affiliation_Precision'),
            "affiliation_recall": metrics.get('Affiliation_Recall'),

            "mean_training_time": group["training_time"].mean(),
            "max_training_time": group["training_time"].max(),
            "min_training_time": group["training_time"].min(),
            "mean_scoring_time": group["scoring_time"].mean(),
            "max_scoring_time": group["scoring_time"].max(),
            "min_scoring_time": group["scoring_time"].min(),
            "count_cleaned_score": group["cleaned_score"].count(),
            "count_raw_score": group["score"].count(),
            "count_anomalies": (group["ground_truth"] == 1).sum(),
            "count_normal": (group["ground_truth"] == 0).sum()
        }

    except Exception as e:
        logging.error(f"Metric calculation failed for {file_name} | Iter {iteration} | Method {method} | {mwp}: {e}")
        return None


def process_file(file_path):
    """Processes a single Excel file."""
    logging.info(f"Processing file: {file_path.name}")
    try:
        df = pd.read_excel(file_path, usecols=usecols)
    except Exception as e:
        logging.error(f"Could not read {file_path.name}: {e}")
        return [] # Return empty list on failure

    # NOTE: This assumes filenames are structured like "..._WINDOWSIZE.xlsx"
    # This could be fragile if filenames change.
    window = int(file_path.name.split("_")[-1].replace('.xlsx', ''))
    
    df['method_window_and_param'] = df['method'] + "_" + str(window) + "_" + df['param'].astype(str)
    
    df_len = len(df)

    if df_len > 0:
        # Ensure window is numeric before division
        try:
            p_window_size = int(window) / df_len
        except ValueError:
            logging.warning(f"Could not parse window size '{window}' as int in file {file_path.name}")
            p_window_size = 0
    else:
        p_window_size = 0
        
    #df['p_window_size'] = p_window_size
    #df['window_size'] = window


    grouped = df.groupby(["iteration", "method","method_window_and_param"])
    local_summary_data = []

    # IMPROVEMENT: Removed nested ThreadPoolExecutor.
    # The main function already parallelizes by *file*.
    # Nesting thread pools is often inefficient and can lead to diminishing returns.
    # Processing groups sequentially *within* the file-processing task is cleaner.
    for (iteration, method, mwp), group in grouped:
        # getting p_window_size and window_size from the first row of the group
        #group_p_window_size = group['p_window_size'].iloc[0]
        #group_window_size = group['window_size'].iloc[0]

        result = process_group(group, file_path.name, iteration, method, mwp, window, p_window_size)
        if result:
            local_summary_data.append(result)

    return local_summary_data

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    Main function to parse arguments and run the summary process.
    """
    
    # --- 1. Define and Parse Command-Line Arguments ---
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    
    # Define the available experiment modes
    modes = [
        'sota_and_own_tuning_pds_pv',
        'sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score',
        'sota_and_own_eval_pds_pv_bv3_bv4', 
        'sota_mdragstream_and_own_eval_pds',  
        'sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score', 
        'sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2',   
        'sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv',
        'test'
    ]
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=modes,
        default='test', # Set 'test' as the default mode
        help=(
            "Select the summary mode to run. "
            "(default: 'test')"
        )
    )
    
    args = parser.parse_args()

    # --- 2. Path Definitions ---
    # NOTE: This assumes this script is in a subdirectory (e.g., 'src')
    # and the 'datasets' directory is in the parent.
    # E.g., /project_root/datasets/ and /project_root/src/summarize.py
    current_dir = Path(__file__).resolve().parent.parent
    output_dir = current_dir / 'datasets' / 'summaries'
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # --- 3. Determine input directories and output file path ---
    try:
        config = get_summary_config(args.mode, current_dir, output_dir)
        input_dirs = config["input_dirs"]
        output_file_path = config["output_file_path"]
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)

    # --- 4. Get all Excel files from the determined directories ---
    files_to_process = []
    for directory in input_dirs:
        if not directory.exists():
            logging.warning(f"Input directory does not exist, skipping: {directory}")
            continue  # Skip to the next directory

        # Filter for .xlsx files that contain an underscore (as assumed by parsing logic)
        found_files = [f for f in directory.iterdir() if f.suffix == '.xlsx' and '_' in f.name]
        files_to_process.extend(found_files)
        logging.info(f"Found {len(found_files)} files to process in {directory.name}")

    # --- 5. Final Check ---
    if not files_to_process:
        logging.error("No files found to process in the specified directories. Exiting.")
        sys.exit(1) # Exit if no files were found

    logging.info(f"Total files to process from all sources: {len(files_to_process)}")

    # --- 6. Main Data Processing ---
    summary_data = []

    # Use ThreadPoolExecutor to process files in parallel
    # 'max_workers=None' uses a default number of threads, suitable for I/O bound tasks
    with ThreadPoolExecutor(max_workers=None) as executor:
        future_to_file = {executor.submit(process_file, path): path for path in files_to_process}

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_summary_data = future.result()
                if file_summary_data:
                    summary_data.extend(file_summary_data)
            except Exception as e:
                # Log error from the worker thread
                logging.error(f"Error processing file {file_path.name}: {e}")

    # --- 7. Save Results ---
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        try:
            summary_df.to_excel(output_file_path, index=False)
            logging.info(f"Summary successfully saved to: {output_file_path}")
        except Exception as e:
            logging.error(f"Failed to save summary file: {e}")
    else:
        logging.warning("No summary data was generated. Output file will not be created.")


if __name__ == "__main__":
    main()
