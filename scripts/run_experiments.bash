cd ~/spectral_anomaly_detector

#######################################
# Scripts for Tuning for Spectral Data Streams and Multivariate Data Streams:
# Activate the environment for experiments
#######################################

# SOTA Methods Tuning
# Run the sota_pv tuning, saving its logs to 'sota_pv_tuning.log' (spectral data)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_tuning.py --mode sota_pv > sota_pv.log 2>&1 &

# Run the sota_pds tuning, saving its logs to 'sota_pds_tuning.log' (multivariate data)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_tuning.py --mode sota_pds > sota_pds.log 2>&1 &

# Own Method Tuning
# Run the own_pv tuning, saving its logs to 'own_pv_tuning.log' (spectral data)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_tuning.py --mode own_pv > own_pv.log 2>&1 &

# Run the own_pds tuning, saving its logs to 'own_pds_tuning.log' (multivariate data)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_tuning.py --mode own_pds > own_pds.log 2>&1 &

#######################################
# Scripts for Evaluation for Spectral Data Streams and Multivariate Data Streams - after tuning hyperparameters:
# Activate the environment for experiments 
#######################################

# SOTA Methods Evaluation
# Run the sota_pv evaluation, saving its logs to 'sota_pv_eval.log' (spectral data for original and BV4)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode sota_pv > sota_pv.log 2>&1 &

# Run the sota_pv_bv5_gtv2 evaluation, saving its logs to 'sota_pv_bv5_eval.log' (spectral data for BV5)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode sota_pv_bv5_gtv2 > sota_pv_bv5_gtv2.log 2>&1 &

# Own Method Evaluation
# Run the own_pv evaluation, saving its logs to 'own_pv_eval.log' (spectral data for original and BV4)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode own_pv > own_pv.log 2>&1 &

# Run the own_pv_bv5_gtv2 evaluation, saving its logs to 'own_pv_bv5_gtv2.log' (spectral data for BV5)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode own_pv_bv5_gtv2 > own_pv_bv5_gtv2.log 2>&1 &

#######################################
# Evaluation with Spectral Data and Current as Extra Feature (Subset Framework) - default hyperparameters:
# Activate the environment for experiments 
#######################################

# Run the sota_pv_bv5_gtv2_current evaluation, saving its logs to 'sota_pv_bv5_gtv2_current.log' (spectral data and EF for BV5)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_with_extra_feature.py --mode sota_pv_bv5_gtv2_current > sota_pv_bv5_gtv2_current.log 2>&1 &

# Run the own_pv_bv5_gtv2_current evaluation, saving its logs to 'own_pv_bv5_gtv2_current.log' (spectral data and EF for BV5)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_with_extra_feature.py --mode own_pv_bv5_gtv2_current > own_pv_bv5_gtv2_current.log 2>&1 &

#######################################
# Evaluation using only multivariate data (current and features of plasma):
# Activate the environment for experiments 
#######################################

# Run the sota_pv_bv5_gtv2_fcurrent evaluation, saving its logs to 'sota_pv_bv5_gtv2_fcurrent.log' (multivariate data for BV5)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_for_other_features_PV.py --mode sota_pv_bv5_gtv2_fcurrent > sota_pv_bv5_gtv2_fcurrent.log 2>&1 &

# Run the sota_pv_bv5_gtv2_fplasma evaluation, saving its logs to 'sota_pv_bv5_gtv2_fplasma.log' (multivariate data for BV5)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_for_other_features_PV.py --mode sota_pv_bv5_gtv2_fplasma > sota_pv_bv5_gtv2_fplasma.log 2>&1 &

# Run the own_pv_bv5_gtv2_fcurrent evaluation, saving its logs to 'own_pv_bv5_gtv2_fcurrent.log' (multivariate data for BV5)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_for_other_features_PV.py --mode own_pv_bv5_gtv2_fcurrent > own_pv_bv5_gtv2_fcurrent.log 2>&1 &

# Run the own_pv_bv5_gtv2_fplasma evaluation, saving its logs to 'own_pv_bv5_gtv2_fplasma.log' (multivariate data for BV5)
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_for_other_features_PV.py --mode own_pv_bv5_gtv2_fplasma > own_pv_bv5_gtv2_fplasma.log 2>&1 &


#######################################
# Scripts for Summarization of Results:
#######################################

# Run the sota_and_own_tuning_pds_pv summary, saving its logs to 'sota_and_own_tuning_pds_pv.log' (summarize tuning)
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_and_own_tuning_pds_pv > sota_and_own_tuning_pds_pv.log 2>&1 &

# Run the sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score summary, saving its logs to 'sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score.log' (summarize all spectral evaluations)
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score > sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score.log 2>&1 &


# Run the sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2 summary, saving its logs to 'sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2.log' (summarize SF with current evaluations in BV5)
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2 > sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2.log 2>&1 &

# Run the sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv summary, saving its logs to 'sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv.log' (summarize multivariate evaluations in BV5)
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv > sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv.log 2>&1 &

#######################################
# Scripts for Preliminary Test:
#######################################

# Run the model_SF_Capymoa summary, saving its logs to 'model_SF_Capymoa.log' (test SF script)
source env_analysis/bin/activate
nohup python scripts/model/SFramework/model_SF_Capymoa.py > model_SF_Capymoa.log 2>&1 &

# Run the model_OnlineBootKNN summary, saving its logs to 'model_OnlineBootKNN.log' (test own obknn script)
source env_analysis/bin/activate
nohup python scripts/model/OBKNN/model_OnlineBootKNN.py > model_OnlineBootKNN.log 2>&1 &