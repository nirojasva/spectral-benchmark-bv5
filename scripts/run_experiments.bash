# Navigate to your project directory for convenience
cd ~/spectral_anomaly_detector

#######################################
# SOTA Methods Tuning and Evaluation:
# Activate the environment for experiments (tuning and evaluation) 

# SOTA Methods Tuning
# Run the sota_pv tuning, saving its logs to 'sota_pv_tuning.log' serveur23
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_tuning.py --mode sota_pv > sota_pv.log 2>&1 &
tail -f sota_pv.log

# Run the sota_pds tuning, saving its logs to 'sota_pds_tuning.log' serveur25
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_tuning.py --mode sota_pds > sota_pds.log 2>&1 &
tail -f sota_pds.log

# SOTA Methods Evaluation
# Run the sota_pv evaluation, saving its logs to 'sota_pv_eval.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode sota_pv > sota_pv.log 2>&1 &
tail -f sota_pv.log

# Run the sota_pv_bv5_gtv2 evaluation, saving its logs to 'sota_pv_bv5_eval.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode sota_pv_bv5_gtv2 > sota_pv_bv5_gtv2.log 2>&1 &
tail -f sota_pv_bv5_gtv2.log

# Run the sota_pv_bv5_gtv2_current evaluation, saving its logs to 'sota_pv_bv5_gtv2_current.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_with_extra_feature.py --mode sota_pv_bv5_gtv2_current > sota_pv_bv5_gtv2_current.log 2>&1 &
tail -f sota_pv_bv5_gtv2_current.log

# Run the sota_pds evaluation, saving its logs to 'sota_pds_eval.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode sota_pds > sota_pds.log 2>&1 &
tail -f sota_pds.log

# Run the sota_pds_v2 evaluation, saving its logs to 'sota_pds_v2.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode sota_pds_v2 > sota_pds_v2.log 2>&1 &
tail -f sota_pds_v2.log

# Run the sota_pds_mdragstream evaluation, saving its logs to 'sota_pds_mdragstream.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode sota_pds_mdragstream > sota_pds_mdragstream.log 2>&1 &
tail -f sota_pds_mdragstream.log

# Run the sota_pv_bv5_gtv2_fcurrent evaluation, saving its logs to 'sota_pv_bv5_gtv2_fcurrent.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_for_other_features_PV.py --mode sota_pv_bv5_gtv2_fcurrent > sota_pv_bv5_gtv2_fcurrent.log 2>&1 &
tail -f sota_pv_bv5_gtv2_fcurrent.log

# Run the sota_pv_bv5_gtv2_fplasma evaluation, saving its logs to 'sota_pv_bv5_gtv2_fplasma.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_for_other_features_PV.py --mode sota_pv_bv5_gtv2_fplasma > sota_pv_bv5_gtv2_fplasma.log 2>&1 &
tail -f sota_pv_bv5_gtv2_fplasma.log

#######################################
# Own Method Tuning and Evaluation:
# Activate the environment for experiments (tuning and evaluation) 

# Own Method Tuning
# Run the own_pv tuning, saving its logs to 'own_pv_tuning.log' serveur23
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_tuning.py --mode own_pv > own_pv.log 2>&1 &
tail -f own_pv.log

# Run the own_pds tuning, saving its logs to 'own_pds_tuning.log' serveur25
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_tuning.py --mode own_pds > own_pds.log 2>&1 &
tail -f own_pds.log

# Own Method Evaluation
# Run the own_pv evaluation, saving its logs to 'own_pv_eval.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode own_pv > own_pv.log 2>&1 &
tail -f own_pv.log

# Run the own_pv_bv5_gtv2 evaluation, saving its logs to 'own_pv_bv5_gtv2.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode own_pv_bv5_gtv2 > own_pv_bv5_gtv2.log 2>&1 &
tail -f own_pv_bv5_gtv2.log

# Run the own_pv_bv5_gtv2_current evaluation, saving its logs to 'own_pv_bv5_gtv2_current.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_with_extra_feature.py --mode own_pv_bv5_gtv2_current > own_pv_bv5_gtv2_current.log 2>&1 &
tail -f own_pv_bv5_gtv2_current.log

# Run the own_pds evaluation, saving its logs to 'own_pds.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode own_pds > own_pds.log 2>&1 &
tail -f own_pds.log

# Run the own_pds_v2 evaluation, saving its logs to 'own_pds_v2.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval.py --mode own_pds_v2 > own_pds_v2.log 2>&1 &
tail -f own_pds_v2.log

# Run the own_pv_bv5_gtv2_fcurrent evaluation, saving its logs to 'own_pv_bv5_gtv2_fcurrent.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_for_other_features_PV.py --mode own_pv_bv5_gtv2_fcurrent > own_pv_bv5_gtv2_fcurrent.log 2>&1 &
tail -f own_pv_bv5_gtv2_fcurrent.log

# Run the own_pv_bv5_gtv2_fplasma evaluation, saving its logs to 'own_pv_bv5_gtv2_fplasma.log'
source env_spectra/bin/activate
nohup python scripts/run_experiments_online_ad_eval_for_other_features_PV.py --mode own_pv_bv5_gtv2_fplasma > own_pv_bv5_gtv2_fplasma.log 2>&1 &
tail -f own_pv_bv5_gtv2_fplasma.log

#######################################
# SOTA and Own Method Summarization of Results:

# Run the sota_and_own_tuning_pds_pv summary, saving its logs to 'sota_and_own_tuning_pds_pv.log'
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_and_own_tuning_pds_pv > sota_and_own_tuning_pds_pv.log 2>&1 &
tail -f sota_and_own_tuning_pds_pv.log

# Run the sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score summary, saving its logs to 'sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score.log'
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score > sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score.log 2>&1 &
tail -f sota_and_own_eval_pds_pv_bv3_bv4_bv5_inv_score.log

# Run the sota_and_own_eval_pds_pv_bv3_bv4 summary, saving its logs to 'sota_and_own_eval_pds_pv_bv3_bv4.log'
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_and_own_eval_pds_pv_bv3_bv4 > sota_and_own_eval_pds_pv_bv3_bv4.log 2>&1 &
tail -f sota_and_own_eval_pds_pv_bv3_bv4.log

# Run the sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2 summary, saving its logs to 'sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2.log'
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2 > sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2.log 2>&1 &
tail -f sota_and_own_eval_bv5_gtv2_w_current_pv_inv_score_v2.log

# Run the sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv summary, saving its logs to 'sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv.log'
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv > sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv.log 2>&1 &
tail -f sota_and_own_eval_bv5_gtv2_for_fplasma_fcurrent_pv.log


# Run the sota_mdragstream_and_own_eval_pds summary, saving its logs to 'sota_mdragstream_and_own_eval_pds.log'
source env_analysis/bin/activate
nohup python scripts/gen_summaries_online_anomaly_detectors.py --mode sota_mdragstream_and_own_eval_pds > sota_mdragstream_and_own_eval_pds.log 2>&1 &
tail -f sota_mdragstream_and_own_eval_pds.log

#######################################
# Own Method Preliminary Test:

# Run the model_SF_Capymoa summary, saving its logs to 'model_SF_Capymoa.log'
source env_analysis/bin/activate
nohup python scripts/model/SFramework/model_SF_Capymoa.py > model_SF_Capymoa.log 2>&1 &
tail -f model_SF_Capymoa.log

# Run the model_MF_Capymoa summary, saving its logs to 'model_MF_Capymoa.log'
source env_analysis/bin/activate
nohup python scripts/model/MFramework/model_MF_Capymoa.py > model_MF_Capymoa.log 2>&1 &
tail -f model_MF_Capymoa.log

# Run the model_OnlineBootKNN summary, saving its logs to 'model_OnlineBootKNN.log'
source env_analysis/bin/activate
nohup python scripts/model/OBKNN/model_OnlineBootKNN.py > model_OnlineBootKNN.log 2>&1 &
tail -f model_OnlineBootKNN.log



