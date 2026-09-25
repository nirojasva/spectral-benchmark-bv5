# Benchmark for Anomaly Detection on Spectral Data Streams - BV5

This project is a benchmark (BV5) for anomaly detection on spectral data streams from Optical Emission Spectrometry with a focus on degassing cycles. We evaluate and compare recent multivariate tabular methods against the Online Bootstrapping K-Nearest Neighbor algorithm.


## Installation

### Step 1: System-Wide Prerequisites
Before installing the Python packages, please ensure you have the following system-level tools installed:

    * Python 3.11.2

    * C++ Build Tools: Required to compile dependencies in dSalmon.

        * On Ubuntu: sudo apt-get install build-essential

        * On Windows: Install "C++ build tools".

    * Java (JDK): Required to run the capymoa package (openjdk version "1.8.0_422").

### Step 2: Evaluation Environment (env_spectra)
This environment is for running the experiments.

```bash

# create and activate venv
python3.11 -m venv env_spectra
source env_spectra/bin/activate


python -m pip install --upgrade pip "setuptools<68.0.0" wheel


pip install "numpy<2.0" Cython

# build and install dSalmon
pip install dSalmon --no-build-isolation


grep -iv '^dSalmon' requirements.txt > /tmp/req_no_dsalmon.txt
pip install -r /tmp/req_no_dsalmon.txt

```

### Step 3: Analysis Environment (env_analysis)
This separate environment is only for analysing result-generation scripts.

```bash
python3.11 -m venv env_analysis
source env_analysis/bin/activate

python -m pip install --upgrade pip "setuptools<68.0.0" wheel

pip install "numpy<2.0" Cython  

pip install dSalmon --no-build-isolation

pip install vus==0.0.6 autorank==1.3.0 capymoa==0.9.0 openpyxl==3.1.5

grep -iv '^dSalmon' requirements.txt > /tmp/req_no_dsalmon.txt
pip install -r /tmp/req_no_dsalmon.txt


pip install jupyter ipykernel
python -m ipykernel install --user --name=env_analysis --display-name "Python (env_analysis)"

```

## Datasets

* [Link to BV5 Datasets during review](datasets/raw/ScenariosV5_test)
* [Path to Multivariate Datasets (118_TAO from TSB-AD)](datasets/raw/TSB-AD-M-lite)
* [Path to Spectral Datasets Original Benchmark (in original repository)](datasets/raw/ScenariosV3_lite)
* [Path to Spectral Datasets BV4 (in Hugging Face repository)](datasets/raw/ScenariosV4_lite)
* [Path to Spectral Datasets BV5](datasets/raw/ScenariosV5_GTV2)

## Running the Experiments and Summarizing Results

To execute all methods and generate a summary of the results automatically, run the provided bash script from the root directory of the project. Ensure all datasets are placed in the correct folder.

```bash
bash scripts/run_experiments.bash
```

## Analysing Datasets

* Show characteristics of datasets such as wavelengths in time, pressure, and current: `notebooks/0_1_TSDA_Original_Sources.ipynb`
* Summarize datasets under SF: `scripts/summary_spectral_ds.py`

