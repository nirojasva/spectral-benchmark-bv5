import sys
import logging
import warnings
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from itertools import product
from model_utils import run_single_experiment
import argparse

# Import parameter grids from the separate configuration file
from model_utils import (
    PARAM_GRID_SOTA_TUNING,
    PARAM_GRID_OWN_TUNING,
    TEST_GRID_TUNING
)

# --- Initial Setup ---
# Configure logging for clear, informative output instead of using print()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)

# Configure warnings for a cleaner console output
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*target variable includes.*", category=UserWarning)

sys.setrecursionlimit(2000)

def get_experiment_config(mode, current_dir):
    """
    Returns the configuration settings based on the selected mode.
    This separates the configuration logic from the execution logic.
    """
    if mode == 'own_pds':
        logging.info("Selected Mode: 1. Tuning own methods on Public Datasets.")
        return {
            "param_grid": PARAM_GRID_OWN_TUNING,
            "output_dir": current_dir / "datasets" / "processed_own_method_V2_tuning_public_ds",
            "cols_slice": None,
            "ds_name": "PDS",
            "use_tuning_datasets": True,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'TSB-AD-M-lite'],
            "tuning_ds": {"116_TAO_id_1_Environment_tr_500_1st_3"}
        }
    elif mode == 'sota_pds':
        logging.info("Selected Mode: 2. Tuning SOTA methods on Public Datasets.")
        return {
            "param_grid": PARAM_GRID_SOTA_TUNING,
            "output_dir": current_dir / "datasets" / "processed_sota_method_tuning_public_ds",
            "cols_slice": None,
            "ds_name": "PDS",
            "use_tuning_datasets": True,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'TSB-AD-M-lite'],
            "tuning_ds": {"116_TAO_id_1_Environment_tr_500_1st_3"}
        }
    elif mode == 'own_pv':
        logging.info("Selected Mode: 3. Tuning own methods on PV Datasets.")
        return {
            "param_grid": PARAM_GRID_OWN_TUNING,
            "output_dir": current_dir / "datasets" / "processed_own_method_V2_tuning_pv_ds",
            "cols_slice": slice(1, 2049),
            "ds_name": "PV",
            "use_tuning_datasets": True,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'ScenariosV4_lite', current_dir / 'datasets' / 'raw' / 'ScenariosV3_lite'],
            "tuning_ds": {"DA3_20250610_095701_ALPS_0"}
        }
    elif mode == 'sota_pv':
        logging.info("Selected Mode: 4. Tuning SOTA methods on PV Datasets.")
        return {
            "param_grid": PARAM_GRID_SOTA_TUNING,
            "output_dir": current_dir / "datasets" / "processed_sota_method_tuning_pv_ds",
            "cols_slice": slice(1, 2049),
            "ds_name": "PV",
            "use_tuning_datasets": True,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'ScenariosV4_lite', current_dir / 'datasets' / 'raw' / 'ScenariosV3_lite'],
            "tuning_ds": {"DA3_20250610_095701_ALPS_0"}
        }
    elif mode == 'test':
        logging.info("Selected Mode: 5. Running default test configuration.")
        return {
            "param_grid": TEST_GRID_TUNING,
            "output_dir": current_dir / "datasets" / "processed_tuning_test",
            "cols_slice": slice(1, 5),
            "ds_name": "TEST",
            "use_tuning_datasets": True,
            "ds_paths": [current_dir / 'datasets' / 'raw' / 'ScenariosV4_lite', current_dir / 'datasets' / 'raw' / 'ScenariosV3_lite', current_dir / 'datasets' / 'raw' / 'TSB-AD-M-lite', current_dir / 'datasets' / 'raw' / 'TSB-AD-U-lite'],
            "tuning_ds": {"DA3_20250610_095701_ALPS_0", "116_TAO_id_1_Environment_tr_500_1st_3"}
        }
    else:
        # This case should not be reachable if argparse 'choices' are set
        raise ValueError(f"Unknown experiment mode: {mode}")

def main():
    """
    Main function to parse arguments and run experiments.
    """
    
    # --- 1. Define and Parse Command-Line Arguments ---
    parser = argparse.ArgumentParser(description="Run modeling experiments.")
    
    # Define the available experiment modes
    modes = [
        'own_pds',    # 1. Own method tuning on Public Datasets
        'sota_pds',   # 2. SOTA method tuning on Public Datasets
        'own_pv',     # 3. Own method tuning on PV Datasets
        'sota_pv',    # 4. SOTA method tuning on PV Datasets
        'test'        # 5. Default test mode
    ]
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=modes,
        default='test', # Set 'test' as the default mode
        help=(
            "Select the experiment mode to run. "
            "(default: 'test')"
        )
    )
    
    args = parser.parse_args()
    
    error_count = 0  # Initialize an error counter

    # --- 2. Master Configuration ---
    N_RUNS = 3
    PERCENTAGE_WINDOW_SIZE = [0.02, 0.05, 0.20]
    
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
            for f, p_window_size, i in product(files_to_process, PERCENTAGE_WINDOW_SIZE, range(N_RUNS)):
                if model_name in ["SWKNN", "SWKNN_own", "SWLOF", "KitNet", "ExactStorm", "mDragStream"] and i > 0:
                    continue
                
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