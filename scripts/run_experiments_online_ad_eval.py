import sys
import logging
import warnings
import argparse
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from itertools import product
from model_utils import run_single_experiment


# Import parameter grids from the separate configuration file
from model_utils import (
    BEST_GRID_PV_DS_FOR_OWN_EVAL,
    BEST_GRID_PDS_FOR_OWN_EVAL,
    BEST_GRID_PV_DS_FOR_SOTA_EVAL,
    BEST_GRID_PDS_FOR_SOTA_EVAL,
    BEST_GRID_PDS_FOR_MDRAGSTREAM_EVAL,
    TEST_GRID_EVAL
)

# --- Initial Setup ---
# (Your logging and warning configuration remains the same)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*target variable includes.*", category=UserWarning)

sys.setrecursionlimit(2000)

def get_experiment_config(mode, current_dir):
    """
    Returns the configuration settings based on the selected mode.
    This separates the configuration logic from the execution logic.
    """
    if mode == 'own_pds':
        logging.info("Selected Mode: 1. Evaluation own methods on Public Datasets.")
        return {
            "param_grid": BEST_GRID_PDS_FOR_OWN_EVAL,
            "output_dir": current_dir / "datasets" / "processed_own_method_V2_eval_public_ds",
            "cols_slice": None,
            "ds_name": "PDS",
            "use_tuning_datasets": False,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'TSB-AD-M-lite'],
            "tuning_ds": {"116_TAO_id_1_Environment_tr_500_1st_3"}
        }
    elif mode == 'own_pds_v2':
        logging.info("Selected Mode: 2. Evaluation own methods on Public Datasets V2.")
        return {
            "param_grid": BEST_GRID_PDS_FOR_OWN_EVAL,
            "output_dir": current_dir / "datasets" / "processed_own_method_V2_eval_public_v2_ds",
            "cols_slice": None,
            "ds_name": "PDS",
            "use_tuning_datasets": False,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'TSB-AD-M-lite-v2'],
            "tuning_ds": {"116_TAO_id_1_Environment_tr_500_1st_3"}
        }
    elif mode == 'sota_pds':
        logging.info("Selected Mode: 3. Evaluation SOTA methods on Public Datasets.")
        return {
            "param_grid": BEST_GRID_PDS_FOR_SOTA_EVAL,
            "output_dir": current_dir / "datasets" / "processed_sota_method_eval_public_ds",
            "cols_slice": None,
            "ds_name": "PDS",
            "use_tuning_datasets": False,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'TSB-AD-M-lite'],
            "tuning_ds": {"116_TAO_id_1_Environment_tr_500_1st_3"}
        }
    elif mode == 'sota_pds_v2':
        logging.info("Selected Mode: 4. Evaluation SOTA methods on Public Datasets V2.")
        return {
            "param_grid": BEST_GRID_PDS_FOR_SOTA_EVAL,
            "output_dir": current_dir / "datasets" / "processed_sota_method_eval_public_v2_ds",
            "cols_slice": None,
            "ds_name": "PDS",
            "use_tuning_datasets": False,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'TSB-AD-M-lite-v2'],
            "tuning_ds": {"116_TAO_id_1_Environment_tr_500_1st_3"}
        }
    elif mode == 'sota_pds_mdragstream':
        logging.info("Selected Mode: 5. Evaluation SOTA methods on Public Datasets.")
        return {
            "param_grid": BEST_GRID_PDS_FOR_MDRAGSTREAM_EVAL,
            "output_dir": current_dir / "datasets" / "processed_sota_method_eval_public_ds_mdragstream",
            "cols_slice": None,
            "ds_name": "PDS",
            "use_tuning_datasets": False,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'TSB-AD-M-lite'],
            "tuning_ds": {"116_TAO_id_1_Environment_tr_500_1st_3"}
        }
    elif mode == 'own_pv':
        logging.info("Selected Mode: 6. Evaluation own methods V2 on PV Datasets (BV3 and BV4).")
        return {
            "param_grid": BEST_GRID_PV_DS_FOR_OWN_EVAL,
            "output_dir": current_dir / "datasets" / "processed_own_method_V2_eval_pv_ds",
            "cols_slice": slice(1, 2049),
            "ds_name": "PV",
            "use_tuning_datasets": False,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'ScenariosV4_lite', current_dir / 'datasets' / 'raw' / 'ScenariosV3_lite'],
            "tuning_ds": {"DA3_20250610_095701_ALPS_0"}
        }
    elif mode == 'own_pv_bv5_gtv2':
        logging.info("Selected Mode: 7. Evaluation own methods V2 on PV Datasets (BV5 GTV2).")
        return {
            "param_grid": BEST_GRID_PV_DS_FOR_OWN_EVAL,
            "output_dir": current_dir / "datasets" / "processed_own_method_V2_eval_pv_bv5_gtv2_ds",
            "cols_slice": slice(1, 2049),
            "ds_name": "PV",
            "use_tuning_datasets": False,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'ScenariosV5_GTV2'],
            "tuning_ds": {"DA3_20250610_095701_ALPS_0"}
        }
    elif mode == 'sota_pv':
        logging.info("Selected Mode: 8. Evaluation SOTA methods on PV Datasets (BV3 and BV4).")
        return {
            "param_grid": BEST_GRID_PV_DS_FOR_SOTA_EVAL,
            "output_dir": current_dir / "datasets" / "processed_sota_method_eval_pv_ds",
            "cols_slice": slice(1, 2049),
            "ds_name": "PV",
            "use_tuning_datasets": False,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'ScenariosV4_lite', current_dir / 'datasets' / 'raw' / 'ScenariosV3_lite'],
            "tuning_ds": {"DA3_20250610_095701_ALPS_0"}
        }


    elif mode == 'sota_pv_bv5_gtv2':
        logging.info("Selected Mode: 9. Evaluation SOTA methods on PV Datasets (BV5 GTV2).")
        return {
            "param_grid": BEST_GRID_PV_DS_FOR_SOTA_EVAL,
            "output_dir": current_dir / "datasets" / "processed_sota_method_eval_pv_bv5_gtv2_ds",
            "cols_slice": slice(1, 2049),
            "ds_name": "PV",
            "use_tuning_datasets": False,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'ScenariosV5_GTV2'],
            "tuning_ds": {"DA3_20250610_095701_ALPS_0"}
        }
    elif mode == 'test':
        logging.info("Selected Mode: 10. Running default test configuration.")
        return {
            "param_grid": TEST_GRID_EVAL,
            "output_dir": current_dir / "datasets" / "processed_eval_test",
            "cols_slice": slice(1, 5),
            "ds_name": "TEST",
            "use_tuning_datasets": True,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'ScenariosV4_lite', current_dir / 'datasets' / 'raw' / 'ScenariosV3_lite', current_dir / 'datasets' / 'raw' / 'TSB-AD-M-lite', current_dir / 'datasets' / 'raw' / 'TSB-AD-U-lite'],
            "tuning_ds": { "116_TAO_id_1_Environment_tr_500_1st_3"}
        }
    else:
        # This case should not be reachable if argparse 'choices' are set
        raise ValueError(f"Unknown experiment mode: {mode}")

def main():
    """
    Main function to parse arguments and run experiments.
    """
    
    # --- 1. Define and Parse Command-Line Arguments ---
    parser = argparse.ArgumentParser(description="Run modeling evaluation experiments.")
    
    # Define the available experiment modes
    modes = [
        'own_pds',            # 1. Own method evaluation on Public Datasets
        'sota_pds',           # 2. SOTA method evaluation on Public Datasets
        'own_pds_v2',         # 3. Own method evaluation on Public Datasets V2
        'sota_pds_v2',        # 4. SOTA method evaluation on Public Datasets V2
        'sota_pds_mdragstream'# 5. SOTA method evaluation on Public Datasets
        'own_pv',             # 6. Own method evaluation on PV Datasets (BV3, BV4)
        'own_pv_bv5_gtv2',    # 7. Own method evaluation on PV Datasets (BV5 GTV2)
        'sota_pv',            # 8. SOTA method evaluation on PV Datasets (BV3 and BV4)
        'sota_pv_bv5_gtv2',   # 9. SOTA method evaluation on PV Datasets (BV5 GTV2)
        'test'                # 10. Default test mode
    ]
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=modes,
        default='test', # Set 'test' as the default mode
        help=(
            "Select the evaluation mode to run. "
            "(default: 'test')"
        )
    )
    
    args = parser.parse_args()
    
    error_count = 0  # Initialize an error counter

    # --- 2. Master Configuration ---
    N_RUNS = 3
    
    # --- 3. Select Parameters and Paths Based on Mode ---
    current_dir = Path(__file__).resolve().parent.parent
    
    # Get the configuration dictionary for the selected mode
    try:
        config = get_experiment_config(args.mode, current_dir)
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1) # Exit if mode is invalid

    # Unpack the configuration dictionary into variables
    param_grid = config["param_grid"]
    output_dir = config["output_dir"]
    cols_slice = config["cols_slice"]
    ds_name = config["ds_name"]
    use_tuning_datasets = config["use_tuning_datasets"]
    ds_paths = config["ds_paths"]
    tuning_ds = config["tuning_ds"]
 
    output_dir.mkdir(parents=True, exist_ok=True)
     
    # --- 4. Find Files to Process ---
    files_to_process = []
    
    for p in ds_paths:
        if not p.exists(): continue
        for f in p.iterdir():
            if f.suffix == '.csv':
                is_tuning_file = f.stem in tuning_ds
                if (use_tuning_datasets and is_tuning_file) or \
                   (not use_tuning_datasets and not is_tuning_file):
                    files_to_process.append(f)

    logging.info(f"Found {len(files_to_process)} dataset(s) to process.")
    for f in files_to_process:
        logging.info(f"  - {f.name}")

    # --- 5. Generate All Experiment Configurations (Tasks) ---
    tasks = []
    for model_name, p_grid in param_grid.items():
        grid = ParameterGrid(p_grid)
        for j, params in enumerate(grid):
            # Use itertools.product to create all combinations cleanly
            for f, i in product(files_to_process, range(N_RUNS)):
                p_window_size = None # Chosing window size in param grid
                
                """
                if model_name in ["SWKNN", "SWKNN_own", "SWLOF", "KitNet", "ExactStorm", "mDragStream"] and i > 0:
                    continue
                """
                task_config = (f, p_window_size, model_name, params, i, j, output_dir, cols_slice, ds_name)
                tasks.append(task_config)

    logging.info(f"Generated a total of {len(tasks)} experiment tasks to run.")

    # --- 6. Execute All Tasks ---
    if tasks:
        for i, task in enumerate(tasks):
            logging.info(f"RUNNING ({i+1}/{len(tasks)}): Model {task[2]} on {task[0].name} with seed {task[4]}")
            
            try:
               result_message = run_single_experiment(task)
               logging.info(f"COMPLETED ({i+1}/{len(tasks)}): {result_message}")
            except Exception as e:
                error_count += 1
                (
                    file_path,
                    p_window_size,
                    model_name,
                    params,
                    run_seed,
                    param_idx,
                    output_dir,
                    cols_slice,
                    ds_name
                ) = task
                logging.error(f"FAILED ({error_count}/{len(tasks)}): Model {task[2]} on {task[0].name}. Error: {e}")
                logging.error(f"FAILURE on run seed {run_seed} for {model_name} on {file_path.name} with params {str(params)}.", exc_info=True)
            
    else:
        logging.warning("No tasks were generated. Check configuration and dataset paths.")

    logging.info(f"All experiments have been completed. Total errors: {error_count}")


if __name__ == "__main__":
    main()