# Benchmark for Anomaly Detection on Spectral Data Streams - BV5

This project is a benchmark (BV5) for anomaly detection on spectral data streams from Optical Emission Spectroscopy with a focus on degassing cycle scenarios. We evaluate and compare recent multivariate tabular methods for streaming data and the Online Bootstrapping K-Nearest Neighbor algorithm for spectral data streams.

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
* [Path to Place Multivariate Datasets (118_TAO from TSB-AD)](datasets/raw/TSB-AD-M-lite)
* [Path to Place Spectral Datasets Original Benchmark (from original repository)](datasets/raw/ScenariosV3_lite)
* [Path to Place Spectral Datasets BV4 (from BV4 Hugging Face repository)](datasets/raw/ScenariosV4_lite)
* [Path to Place Spectral Datasets BV5](datasets/raw/ScenariosV5_GTV2)

### Characteristics of BV5 Datasets

#### General Description

The datasets contain information generated during experiments with an OES-based sensor. The data includes the corresponding timestamp of instances, the spectra measurements consisting of 2048 wavelengths per reading, region studies of the spectra for specific compounds, vacuum pressure in mbar, current voltage, among others.

#### Datasets Schema

The datasets consist of several columns. The index starts from 0.

| Index | Column Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| 0 | CURRENTTIMESTAMP | String | ISO timestamp in the format YYYY MMMM |
| 1 to 2048 | 189.73 to 895.79 | Float | Spectra information where the header is the wavelength and rows contain the measured values |
| 2049 to 2052 | RS1 to RS4 | Float | Region Study values (see details in the section below) |
| 2053 to 2080 | RS5 to RS32 | Float | Region Study variables not currently used |
| 2081 | VOLTAGE | Float | Regulation voltage measured in Volts (V) |
| 2082 | CURRENT | Float | Regulation current measured in milliamperes (mA) |
| 2083 | PRESSURE | Float | OES Chamber pressure measured in millibars (mbar) |
| 2084 | WARMUP | Boolean | 0 for False and 1 for True. Indicates if the sensor is in the warmup phase |
| 2085 | ACTIVE | Boolean | 0 for False and 1 for True. Indicates if the sensor is active |
| 2086 | REGULATIONON | Boolean | 0 for False and 1 for True. Indicates if the regulation is active |
| 2087 | PLASMAON | Boolean | 0 for False and 1 for True. Indicates if the plasma is turned on |
| 2088 | PLASMAOFFFORCED | Boolean | 0 for False and 1 for True. Indicates if the plasma was forced to off |
| 2089 | PLASMASTATUS | Boolean | 0 for None, 1 for PlasmaOffForcedByUser, 2 for PlasmaOffConditionsAreNotGood, 3 for PlasmaOn |
| 2090 | WARNING | Boolean | 0 for False and 1 for True. Indicates an active warning |
| 2091 | ISPOWERSAVINGMODE | Boolean | 0 for False and 1 for True. Indicates if the system is in power saving mode |
| 2092 | ISTEMPERATUREFAULT | Boolean | 0 for False and 1 for True. Indicates if the temperature is out of range |
| 2093 | ISCURRENTFAULT | Boolean | 0 for False and 1 for True. Indicates if the current is out of range |
| 2094 | ISSPECTROMETERFAULT | Boolean | 0 for False and 1 for True. Indicates a spectrometer fault |
| 2095 | ISHVMODULEFAULT | Boolean | 0 for False and 1 for True. Indicates a High Voltage fault |
| 2096 | ISPOLLUTIONFAULT | Boolean | 0 for False and 1 for True. Indicates pollution was detected |
| 2097 | ISALPSCONFIGERROR | Boolean | 0 for False and 1 for True. Indicates the ALPS Configuration is damaged |
| 2098 | ALPSSerial | String | A Serial Number |
| 2099 | DOWN | Boolean | 0 for False and 1 for True. Indicates Data Controller is down |
| 2100 | DOWNMSG | String | Message providing details if the system is down |
| 2101 | SENSORPRESSURECHAMBER1 | Float | Pressure readings from an alternative connected chamber |
| 2102 | ETATREALTIMELIFE | Boolean | 0 for False and 1 for True. Indicates if a valve for leaks is open (true) or closed (false) under automated scenarios |
| 2103 | ANOMALY? | Boolean | 0 for False and 1 for True. Indicator for defined anomalies |

#### Region Study Details

The Region Study columns (RS1 to RS4) represent specific compounds calculated based on maximum values within specific wavelength intervals. The table below shows the formula used for each variable.

| Label | Compound | Formula |
| :--- | :--- | :--- |
| RS1 | N2 | Max value between wavelength 386.45 and 393.38 |
| RS2 | O2 | Max value between wavelength 773.38 and 780.40 |
| RS3 | H | Max value between wavelength 652.47 and 659.53 |
| RS4 | OH | Max value between wavelength 304.46 and 311.54 |


## Running the Experiments and Summarizing Results

To execute all methods and generate a summary of the results automatically, run the provided bash script from the root directory of the project. Ensure all datasets are placed in the correct folder.

```bash
bash scripts/run_experiments.bash
```

## Analysing Datasets

* Analysis of characteristics of datasets such as wavelengths in time, pressure, and current: `notebooks/0_1_TSDA_Original_Sources.ipynb`
* Summary of datasets under Subset Framework (SF): `scripts/summary_spectral_ds.py`

