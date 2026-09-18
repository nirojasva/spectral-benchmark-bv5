from pysad.models import ExactStorm, IForestASD, KitNet, RobustRandomCutForest, RSHash, xStream
from capymoa.anomaly import OnlineIsolationForest, HalfSpaceTrees as HStreeCapy
from model.OBKNN.model_OnlineBootKNN import OnlineBootKNN, median_of_means, transform_instance
from dSalmon.outlier import SWKNN , SWLOF
import numpy as np

from model.mDragstream.point_mDragStream_case1 import mDragStream as point_mDragStream_case1
from model.mDragstream.point_mDragStream_case2 import mDragStream as point_mDragStream_case2
from model.mDragstream.point_mDragStream_case3 import mDragStream as point_mDragStream_case3
from model.mDragstream.point_mDragStream_case4 import mDragStream as point_mDragStream_case4

from model.mDragstream.subsequence_mDragStream_case1 import mDragStream as subsequence_mDragStream_case1
from model.mDragstream.subsequence_mDragStream_case2 import mDragStream as subsequence_mDragStream_case2
from model.mDragstream.subsequence_mDragStream_case3 import mDragStream as subsequence_mDragStream_case3
from model.mDragstream.subsequence_mDragStream_case4 import mDragStream as subsequence_mDragStream_case4

from model.mDragstream.mdragstream import mDragStream

from model.MFramework.model_MF_Capymoa import MF_Capymoa
from model.MFramework.model_MF_Pysad import MF_Pysad

from model.SFramework.model_SF_Capymoa import SF_Capymoa
from model.SFramework.model_SF_Pysad import SF_Pysad
from model.SFramework.model_SF_dSalmon import SF_dSalmon

import os
import pandas as pd
import time
from capymoa.stream import NumpyStream
import ast
import math
import logging

#INF = np.finfo(np.float64).max
INF = 1.0e+200

PARAM_GRID_SOTA_TUNING = {
    ################################ Pysad Implementation
    "RSHash": { 
        'feature_mins': [[0]],              # Default: NA
        'feature_maxes': [[np.inf, 2000]],  # Default: NA
        #'sampling_points': [1000],         # Default: 1000
        'decay': [0.015, 0.05],             # Default: 0.015
        'num_components': [50, 100],        # Default: 100
        'num_hash_fns': [1, 5],             # Default: 1
    },
    "xStream": {
        #'window_size': [25],             # Default: 25
        'num_components': [50 ,100],      # Default: 100
        'n_chains': [50, 100],            # Default: 100
        'depth': [15, 25],                # Default: 25
    },
    "IForestASD": { 
        'initial_window_X': [None],  # Default: None
        #'window_size': [2048],      # Default: 2048
    },
    "KitNet": { 
        #'num_features': [10],            # Default: NA
        'max_size_ae': [10, 20],          # Default: 10
        #'grace_feature_mapping': [None], # Default: None
        #'grace_anomaly_detector': [None],# Default: None
        'learning_rate': [0.1, 0.2],      # Default: 0.1
        'hidden_ratio': [0.75, 0.9],      # Default: 0.75
    },
    "ExactStorm": {
        #'window_size': [10000],                   # Default: 10000
        'max_radius': [0.1, 300, 900, 1800, 4000],  # Default: 0.1
    },
    "RobustRandomCutForest": {
        'num_trees': [4, 32],                          # Default: 4
        #'shingle_size': [4],                          # Default: 4
        'tree_size': [256, 512],                       # Default: 256 
    },
    ################################ Capymoa Implementation
    "oIF": {
        #schema: [None],                                        # Default: None
        #'random_seed': [1],                                    # Default: 1
        'num_trees': [32, 64],                                  # Default: 32
        'max_leaf_samples': [32, 64],                           # Default: 32
        'growth_criterion': ['adaptive', 'fixed'],              # Default: 'adaptive'
        #'subsample': [1.0],                                    # Default: 1.0
        #'window_size': [2048],                                 # Default: 2048
        #'branching_factor': [2],                               # Default: 2
        #'split': ['axisparallel'],                             # Default: 'axisparallel'
        'n_jobs': [-1],                                         # Default: -1 
    },
    "HStree": {
        #'schema': [None],                 # Default: None
        #'CLI':[None],                     # Default: None
        #'random_seed':[1],                # Default: 1
        #'window_size':[100],              # Default: 100
        'number_of_trees': [15, 25],       # Default: 25
        'max_depth': [10, 15],             # Default: 15
        'anomaly_threshold': [0.3, 0.5],   # Default: 0.5
        'size_limit': [0.1],               # Default: 0.1
    },
    ################################ dSalmon Implementation
    "SWLOF": { 
        #'window': [50],                         # Default: NA
        'k': [1, 10],                            # Default: NA
        'k_is_max': [False],                     # Default: False (True is not working)
        'simplified': [False],                   # Default: False    
        'metric': ['cityblock','euclidean'],     # Default: 'euclidean'
        #'metric_params': [None],                # Default: None
        #'float_type': [np.float64],             # Default: np.float64
        #'min_node_size': [5],                   # Default: 5
        #'max_node_size': [20],                  # Default: 20
        #'split_sampling': [5],                  # Default: 5
    },
    ################################ LIMOS Implementation
    "mDragStream": {
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30, 50],                  # Default: 30
        'maxClusterSize': [30, 50],                 # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None

    },
}

PARAM_GRID_OWN_TUNING = {
    "OnlineBootKNN": { 
    # Varying Parameters
    'ensemble_size': [30, 240],
    'dmetric': ['cityblock'],
    'transf': ['NONE', 'ZNORM'],
    'update_distance_with_abnormal': [True, False],
    'update_mode_stats': ['ema', 'welford'],
    'alpha_ema': [0.01],
    # Constants (Fixed for all runs)
    'algorithm': ['brute'],
    'chunk_size': [30],
    'type_dist': ['largest'],
    'alpha_z_test': [0.05, 0.95],
    'n_jobs': [-1],
    'no_bootstrapp': [False],
    'no_z_score': [False] 
    }, 
    "SWKNN": {
        #'window': [50],                         # Default: NA
        'k': [1, 10, 50],                        # Default: NA
        'k_is_max': [False],                     # Default: False (True is not working)
        'metric': ['cityblock'],                 # Default: 'euclidean'
        #'metric_params': [None],                # Default: None
        #'float_type': [np.float64],             # Default: np.float64
        'min_node_size': [5],                    # Default: 5
        'max_node_size': [20],                   # Default: 20
        'split_sampling': [5],                   # Default: 5
    },
}

BEST_GRID_PDS_FOR_OWN_EVAL = {
    ### Best Grid for OBKNN in Public Datasets and PV Datasets
    "OnlineBootKNN_TNone": {
        'p_window_size': [0.2],   
        'alpha_ema': [0.01],                     
        # Varying Parameters
        'ensemble_size': [30],
        'chunk_size': [30],
        'dmetric': ['cityblock'],
        'transf': ['NONE'],
        'update_distance_with_abnormal': [False],
        'update_mode_stats': ['ema'],
        'algorithm': ['brute'],
        'type_dist': ['largest'],
        'alpha_z_test': [0.05],
        'n_jobs': [-1],
        'no_bootstrapp': [False],
        'no_z_score': [False] 
    },
    "OnlineBootKNN_TZnorm": {
        'p_window_size': [0.2],   
        'alpha_ema': [0.01],                     
        # Varying Parameters
        'ensemble_size': [30],
        'chunk_size': [30],
        'dmetric': ['cityblock'],
        'transf': ['ZNORM'],
        'update_distance_with_abnormal': [True],
        'update_mode_stats': ['welford'],
        'algorithm': ['brute'],
        'type_dist': ['largest'],
        'alpha_z_test': [0.05],
        'n_jobs': [-1],
        'no_bootstrapp': [False],
        'no_z_score': [False] 
    },
}

BEST_GRID_PV_DS_FOR_OWN_EVAL = BEST_GRID_PDS_FOR_OWN_EVAL

BEST_GRID_PV_DS_FOR_SOTA_EVAL = {
    ### Best Grid for SOTA Methods in PV Datasets
    ################################ Pysad Implementation
    "xStream": {
        'p_window_size': [0.02],      # Default: NA
        'num_components': [50],       # Default: 100
        'n_chains': [100],            # Default: 100
        'depth': [25],                # Default: 25
    },
    "RSHash": { 
        'p_window_size': [0.2],                  # Default: NA
        'feature_mins': [[0]],                    # Default: NA
        'feature_maxes': [[np.inf, 2000]],        # Default: NA
        #'sampling_points': [1000],               # Default: 1000
        'decay': [0.05],                          # Default: 0.015
        'num_components': [50],                   # Default: 100
        'num_hash_fns': [1],                      # Default: 1
    },    
    "IForestASD": { 
        'p_window_size': [0.2],      # Default: NA
        'initial_window_X': [None],  # Default: None
        #'window_size': [2048],      # Default: 2048
    },

    "KitNet": { 
        'p_window_size': [0.05],          # Default: NA
        #'num_features': [10],            # Default: NA
        'max_size_ae': [10],              # Default: 10
        #'grace_feature_mapping': [None], # Default: None
        #'grace_anomaly_detector': [None],# Default: None
        'learning_rate': [0.1],           # Default: 0.1
        'hidden_ratio': [0.9],           # Default: 0.75
    },
    
    "ExactStorm": {
        'p_window_size': [0.2],                    # Default: NA
        #'window_size': [10000],                    # Default: 10000
        'max_radius': [900],                        # Default: 0.1
    },
    
    "RobustRandomCutForest": {
        'p_window_size': [0.2], #changed from 0.05# Default: NA
        'num_trees': [4],                          # Default: 4
        #'shingle_size': [4],                      # Default: 4
        'tree_size': [256],                        # Default: 256 
    },
    ################################ Capymoa Implementation
    
    "oIF": {
        'p_window_size': [0.2],                                 # Default: NA
        #schema: [None],                                        # Default: None
        #'random_seed': [1],                                    # Default: 1
        'num_trees': [64],                                      # Default: 32
        'max_leaf_samples': [64],                               # Default: 32
        'growth_criterion': ['fixed'],                          # Default: 'adaptive'
        #'subsample': [1.0],                                    # Default: 1.0
        #'window_size': [2048],                                 # Default: 2048
        #'branching_factor': [2],                               # Default: 2
        #'split': ['axisparallel'],                             # Default: 'axisparallel'
        'n_jobs': [-1],                                         # Default: -1 
    },
    
    "HStree": {
        'p_window_size': [0.02],            # Default: NA
        #'schema': [None],                 # Default: None
        #'CLI':[None],                     # Default: None
        #'random_seed':[1],                # Default: 1
        #'window_size':[100],              # Default: 100
        'number_of_trees': [25],           # Default: 25
        'max_depth': [10],                 # Default: 15
        'anomaly_threshold': [0.5],        # Default: 0.5
        'size_limit': [0.1],               # Default: 0.1
    },
    ################################ dSalmon Implementation

    "SWLOF": { 
        'p_window_size': [0.02],                 # Default: NA
        #'window': [50],                         # Default: NA
        'k': [10],                               # Default: NA
        'k_is_max': [False],                     # Default: False (True is not working)
        'simplified': [False],                   # Default: False    
        'metric': ['euclidean'],                 # Default: 'euclidean'
        #'metric_params': [None],                # Default: None
        #'float_type': [np.float64],             # Default: np.float64
        #'min_node_size': [5],                   # Default: 5
        #'max_node_size': [20],                  # Default: 20
        #'split_sampling': [5],                  # Default: 5
    },
}

BEST_GRID_PDS_FOR_SOTA_EVAL = BEST_GRID_PV_DS_FOR_SOTA_EVAL

"""
BEST_GRID_PV_DS_FOR_MDRAGSTREAM_EVAL = {
    ################################ LIMOS Implementation
    "mDragStream": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
    },
    "subsequence_mDragStream_case1": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
        'use_barycenter': [False] 
    },
    "subsequence_mDragStream_case2": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
        'use_barycenter': [False] 
    },
    "subsequence_mDragStream_case3": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
        'use_barycenter': [True] 
    },
    "subsequence_mDragStream_case4": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
        'use_barycenter': [True] 
    },
    "point_mDragStream_case1": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
    },
    "point_mDragStream_case2": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
    },
    "point_mDragStream_case3": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
    },
    "point_mDragStream_case4": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
    },

}
BEST_GRID_PDS_FOR_MDRAGSTREAM_EVAL = BEST_GRID_PV_DS_FOR_MDRAGSTREAM_EVAL
"""

BEST_GRID_PV_DS_FOR_MDRAGSTREAM_EVAL = {
    ################################ LIMOS Implementation
    "subsequence_mDragStream_case1": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
        'use_barycenter': [False] 
    },
    "subsequence_mDragStream_case2": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
        'use_barycenter': [False] 
    },
    "subsequence_mDragStream_case3": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
        'use_barycenter': [True] 
    },
    "subsequence_mDragStream_case4": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
        'use_barycenter': [True] 
    },
}

BEST_GRID_PDS_FOR_MDRAGSTREAM_EVAL = BEST_GRID_PV_DS_FOR_MDRAGSTREAM_EVAL

TEST_GRID_TUNING = {
    "subsequence_mDragStream_case4": {
        'p_window_size': [0.02],                    # Default: NA
        #'window': [50],                            # Default: NA
        #'r': [0.1, 300, 900, 1800, 4000],          # Default: NA
        'maxNbClusters': [30],                      # Default: 30
        'maxClusterSize': [50],                     # Default: 30    
        'metric': ['euclidean', 'mahalanobis'],     # Default: 'mahalanobis'
        'inv_cov': [None],                          # Default: None
        'use_barycenter': [True] 
    },
}

TEST_GRID_EVAL = TEST_GRID_TUNING


DEFAULT_GRID_PV_DS_WITH_EF_CURRENT_FOR_SOTA_EVAL = {

    ### Default Grid for SOTA Methods 
    ################################ Pysad Implementation
    "SFxStream": {
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],      # Default: NA
        'num_components': [100],       # Default: 100
        'n_chains': [100],            # Default: 100
        'depth': [25],                # Default: 25
    },
    "SFRSHash": { 
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],                  # Default: NA
        'feature_mins': [[0]],                    # Default: NA
        'feature_maxes': [[10000]],        # Default: NA
        #'sampling_points': [1000],               # Default: 1000
        'decay': [0.015],                          # Default: 0.015
        'num_components': [100],                   # Default: 100
        'num_hash_fns': [1],                      # Default: 1
    },    
    "SFIForestASD": { 
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],      # Default: NA
        'initial_window_X': [None],  # Default: None
        #'window_size': [2048],      # Default: 2048
    },

    "SFKitNet": { 
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],          # Default: NA
        #'num_features': [10],            # Default: NA
        'max_size_ae': [10],              # Default: 10
        #'grace_feature_mapping': [None], # Default: None
        #'grace_anomaly_detector': [None],# Default: None
        'learning_rate': [0.1],           # Default: 0.1
        'hidden_ratio': [0.75],           # Default: 0.75
    },
    
    "SFExactStorm": {
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],                    # Default: NA
        #'window_size': [10000],                    # Default: 10000
        'max_radius': [0.1],                        # Default: 0.1
    },
    
    "SFRobustRandomCutForest": {
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02], #changed from 0.05# Default: NA
        'num_trees': [4],                          # Default: 4
        #'shingle_size': [4],                      # Default: 4
        'tree_size': [256],                        # Default: 256 
    },
    ################################ Capymoa Implementation
    
    "SFoIF": {
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],                                 # Default: NA
        #schema: [None],                                        # Default: None
        #'random_seed': [1],                                    # Default: 1
        'num_trees': [32],                                      # Default: 32
        'max_leaf_samples': [32],                               # Default: 32
        'growth_criterion': ['adaptive'],                          # Default: 'adaptive'
        #'subsample': [1.0],                                    # Default: 1.0
        #'window_size': [2048],                                 # Default: 2048
        #'branching_factor': [2],                               # Default: 2
        #'split': ['axisparallel'],                             # Default: 'axisparallel'
        'n_jobs': [-1],                                         # Default: -1 
    },
    
    "SFHStree": {
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],            # Default: NA
        #'schema': [None],                 # Default: None
        #'CLI':[None],                     # Default: None
        #'random_seed':[1],                # Default: 1
        #'window_size':[100],              # Default: 100
        'number_of_trees': [25],           # Default: 25
        'max_depth': [15],                 # Default: 15
        'anomaly_threshold': [0.5],        # Default: 0.5
        'size_limit': [0.1],               # Default: 0.1
    },
    ################################ dSalmon Implementation
    "SFSWLOF": { 
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],                 # Default: NA
        #'window': [50],                         # Default: NA
        'k': [10],                               # Default: NA
        'k_is_max': [False],                     # Default: False (True is not working)
        'simplified': [False],                   # Default: False    
        'metric': ['euclidean'],                 # Default: 'euclidean'
        #'metric_params': [None],                # Default: None
        #'float_type': [np.float64],             # Default: np.float64
        #'min_node_size': [5],                   # Default: 5
        #'max_node_size': [20],                  # Default: 20
        #'split_sampling': [5],                  # Default: 5
    },
}

OPTIMAL_GRID_PV_DS_WITH_EF_CURRENT_FOR_OWN_EVAL =  {
    ### Optimal Grid for OBKNN 
    "SFOnlineBootKNN_TNone": {
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],   
        'alpha_ema': [0.95],                     
        # Varying Parameters
        'ensemble_size': [30],
        'chunk_size': [30],
        'dmetric': ['cityblock'],
        'transf': ['NONE'],
        'update_distance_with_abnormal': [False],
        'update_mode_stats': ['ema'],
        'algorithm': ['brute'],
        'type_dist': ['largest'],
        'alpha_z_test': [0.95],
        'n_jobs': [-1],
        'no_bootstrapp': [False],
        'no_z_score': [False] 
    },
    "SFOnlineBootKNN_TZnorm": {
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],   
        'alpha_ema': [0.95],                     
        # Varying Parameters
        'ensemble_size': [30],
        'chunk_size': [30],
        'dmetric': ['cityblock'],
        'transf': ['ZNORM'],
        'update_distance_with_abnormal': [True],
        'update_mode_stats': ['welford'],
        'algorithm': ['brute'],
        'type_dist': ['largest'],
        'alpha_z_test': [0.95],
        'n_jobs': [-1],
        'no_bootstrapp': [False],
        'no_z_score': [False] 
    },
}

DEFAULT_GRID_PV_DS_WITH_EF_CURRENT_FOR_OWN_EVAL =  {
    ### Default Grid for OBKNN 
    "SFOnlineBootKNN_TNone": {
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],   
        'alpha_ema': [0.01],                     
        # Varying Parameters
        'ensemble_size': [240],
        'chunk_size': [240],
        'dmetric': ['cityblock'],
        'transf': ['NONE'],
        'update_distance_with_abnormal': [False],
        'update_mode_stats': ['ema'],
        'algorithm': ['brute'],
        'type_dist': ['largest'],
        'alpha_z_test': [0.05],
        'n_jobs': [-1],
        'no_bootstrapp': [False],
        'no_z_score': [False] 
    },
    "SFOnlineBootKNN_TZnorm": {
        "min_value": [7.75],
        "max_value": [8.25],
        'p_window_size': [0.02],   
        'alpha_ema': [0.01],                     
        # Varying Parameters
        'ensemble_size': [240],
        'chunk_size': [240],
        'dmetric': ['cityblock'],
        'transf': ['ZNORM'],
        'update_distance_with_abnormal': [True],
        'update_mode_stats': ['welford'],
        'algorithm': ['brute'],
        'type_dist': ['largest'],
        'alpha_z_test': [0.05],
        'n_jobs': [-1],
        'no_bootstrapp': [False],
        'no_z_score': [False] 
    },
}


def clean_score(score):
    
    error_score = []  # Use a list for better error message formatting
    
    score = ast.literal_eval(score) if isinstance(score, str) and len(score) > 0 else score
    
    # Convert list or array to single value
    if isinstance(score, list):
        if len(score) > 0:  # Ensure the list is not empty
            score = score[0]
            error_score.append("List Output Assigned the First Value.")
            # Check for Inf safely
            if np.isinf(score):
                score = INF
                error_score.append(f"Infinity Output Assigned to {INF}.")
        else:
            score = 0
            error_score.append("Empty List Assigned to 0.")
    
    if isinstance(score, np.ndarray):
        if score.size > 0:  # Ensure the array is not empty
            score = score[0]
            error_score.append("Array Output Assigned the First Value.")
            # Check for Inf safely
            if np.isinf(score):
                score = INF
                error_score.append(f"Infinity Output Assigned to {INF}.")
            
        else:
            score = 0
            error_score.append("Empty Array Assigned to 0.")

    if score is None:
        score = 0
        error_score.append("None Output Assigned to 0.")
    """
    # Check for NaN safely
    if pd.isna(score):
        score = 0
        error_score.append("NaN Output Assigned to 0 (Pandas).")
    """
    
    # Check for NaN safely
    if np.isnan(score):
        score = 0
        error_score.append("NaN Output Assigned to 0 (Numpy).")
    
    # Check for Inf safely
    if np.isinf(score):
        score = INF
        error_score.append(f"Infinity Output Assigned to {INF}.")
    
    # Check for NaN safely
    if math.isnan(score):
        score = 0
        error_score.append("NaN Output Assigned to 0 (Math).")
    
    # Handle non-numeric types
    if isinstance(score, float) and (score != score):
        score = 0
        error_score.append("NaN Output Assigned to 0 (Float).")
    
    # Handle non-numeric types
    if isinstance(score, str):
        score = 0
        error_score.append("String Output Assigned to 0.")
    
    score = float(score)  # force invalid values to an Error

    message = " | ".join(error_score)
    
    return score, message

def run_single_experiment(config):
    """
    Executes one experiment trial based on the provided configuration.
    This function handles data loading, model training, scoring, time tracking,
    and saving the results for a single, isolated run.

    Args:
        config (tuple): A tuple containing all necessary parameters for one run:
                        (file_path, p_window_size, model_name, params, run_seed,
                         param_idx, output_dir, cols_slice, ds_name).

    Returns:
        str: A message indicating the success or failure of the experiment.
    """
    # 1. Unpack configuration
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
    ) = config

    # 2. Set environment for this specific process (important for performance)
    # Using '40' allows a single algorithm to use all available cores.
    os.environ["OMP_NUM_THREADS"] = "40"
    os.environ["OPENBLAS_NUM_THREADS"] = "40"
    os.environ["MKL_NUM_THREADS"] = "40"
    os.environ["NUMEXPR_NUM_THREADS"] = "40"
    
    # 3. Ensure reproducibility within this process
    np.random.seed(run_seed)

    # --- Data Loading ---
    try:
        df = pd.read_csv(file_path, sep=',', low_memory=False, dtype={'CURRENTTIMESTAMP': str})
    except:
        df = pd.read_csv(file_path, sep=',', low_memory=False)
            
    if ds_name=="PDS":
        cols_slice = slice(0, len(df.columns)-1) 
        cols = df.columns[cols_slice]
        col_target = df.columns[-1]
        stream = NumpyStream(df[cols].values, df[col_target].values, dataset_name=ds_name, feature_names=cols)
        schema = stream.get_schema()
        if p_window_size == None: 
            params_c = params.copy()
            p_window_size = params_c.pop('p_window_size', None)
            window_size = max(1, int(len(df) * p_window_size))
        else:
            params_c = params.copy()
            window_size = max(1, int(len(df) * p_window_size))

    elif ds_name=="F_FOR_PLASMA":
        cols_slice = slice(2081, 2084) 
        cols = df.columns[cols_slice]
        col_target = df.columns[-1]
        stream = NumpyStream(df[cols].values, df[col_target].values, dataset_name=ds_name, feature_names=cols)
        schema = stream.get_schema()
        if p_window_size == None: 
            params_c = params.copy()
            p_window_size = params_c.pop('p_window_size', None)
            window_size = max(1, int(len(df) * p_window_size))
        else:
            params_c = params.copy()
            window_size = max(1, int(len(df) * p_window_size))

    elif ds_name=="ONLY_CURRENT":
        cols_slice = slice(2082, 2083)
        cols = df.columns[cols_slice]
        col_target = df.columns[-1]
        stream = NumpyStream(df[cols].values, df[col_target].values, dataset_name=ds_name, feature_names=cols)
        schema = stream.get_schema()
        if p_window_size == None: 
            params_c = params.copy()
            p_window_size = params_c.pop('p_window_size', None)
            window_size = max(1, int(len(df) * p_window_size))
        else:
            params_c = params.copy()
            window_size = max(1, int(len(df) * p_window_size))
    
    elif ds_name=="PV":   
        cols = df.columns[cols_slice]
        col_target = df.columns[-1]
        stream = NumpyStream(df[cols].values, df[col_target].values, dataset_name=ds_name, feature_names=cols)
        schema = stream.get_schema()
        if p_window_size == None: 
            params_c = params.copy()
            p_window_size = params_c.pop('p_window_size', None)
            window_size = max(1, int(len(df) * p_window_size))
        else:
            params_c = params.copy()
            window_size = max(1, int(len(df) * p_window_size))
    
    else:
        cols = df.columns[cols_slice]
        col_target = df.columns[-1]
        stream = NumpyStream(df[cols].values, df[col_target].values, dataset_name=ds_name, feature_names=cols)
        schema = stream.get_schema()
        if p_window_size == None: 
            params_c = params.copy()
            p_window_size = params_c.pop('p_window_size', None)
            window_size = max(1, int(len(df) * p_window_size))
        else:
            params_c = params.copy()
            window_size = max(1, int(len(df) * p_window_size))
    
    # --- Model Initialization ---
    learner = get_model_with_params(
        model_name, params_c, window_size=window_size, random_seed=run_seed, schema=schema
    )

    stream.restart()
    
    # --- Main Processing Loop ---
    results_data = []
    for row, instance in enumerate(stream):
        
        # --- API-Specific Logic for Training and Scoring ---
        # time.perf_counter() is used for precise timing within this process.
        if hasattr(learner, "fit_partial"):  # Pysad models
            start_train = time.perf_counter()
            learner.fit_partial(instance.x)
            training_time = time.perf_counter() - start_train
            start_score = time.perf_counter()
            score = learner.score_partial(instance.x)
            scoring_time = time.perf_counter() - start_score
        elif hasattr(learner, "train"):  # Capymoa models
            start_score = time.perf_counter()
            score = learner.score_instance(instance)
            scoring_time = time.perf_counter() - start_score
            start_train = time.perf_counter()
            learner.train(instance)
            training_time = time.perf_counter() - start_train
        elif hasattr(learner, "fit_predict"):  # dSalmon models
            training_time = 0
            start_score = time.perf_counter()
            score = learner.fit_predict(instance.x)
            scoring_time = time.perf_counter() - start_score
        else:
            raise AttributeError(f"Model {model_name} has no recognized training/scoring method.")

        cleaned_score, error_score = clean_score(score)
        
        if ds_name == "PV": 
            step = df.iloc[row, 0]
        else: 
            step = row
        
        results_data.append({
            "iteration": run_seed,
            "timestamp": step,
            "method": model_name,
            "param": str(params),  # Convert dict to string for CSV compatibility
            "score": score,
            "cleaned_score": cleaned_score,
            "training_time": training_time,
            "scoring_time": scoring_time,
            "ground_truth": instance.y_index,
            "error_type_score": error_score,
        })
    
    # --- Save Results ---
    results_df = pd.DataFrame(results_data)

    if ds_name=="F_FOR_PLASMA":
        file_name_prefix = file_path.stem.split("_")[0]+"_"+file_path.stem.split("_")[1]+"_FP"
    elif ds_name=="ONLY_CURRENT":
        file_name_prefix = file_path.stem.split("_")[0]+"_"+file_path.stem.split("_")[1]+"_OC"
    else:
        file_name_prefix = file_path.stem.split("_")[0]+"_"+file_path.stem.split("_")[1]

    output_filename = f"{file_name_prefix}_results_{model_name}_iter_{run_seed}_paramset_{param_idx}_{ds_name}_V2_ds_ws_{window_size}.xlsx"


    full_path = output_dir / output_filename
    results_df.to_excel(full_path, index=False)
    
    return f"SUCCESS: Run seed {run_seed} for {model_name} on {file_path.name} with params {param_idx}."

def run_single_experiment_with_extra_data(config):
    """
    Executes one experiment trial based on the provided configuration (including extra data feature).
    This function handles data loading, model training, scoring, time tracking,
    and saving the results for a single, isolated run.

    Args:
        config (tuple): A tuple containing all necessary parameters for one run:
                        (file_path, p_window_size, model_name, params, run_seed,
                         param_idx, output_dir, cols_slice, col_extra_feature, ds_name).

    Returns:
        str: A message indicating the success or failure of the experiment.
    """
    # 1. Unpack configuration
    (
        file_path,
        p_window_size,
        model_name,
        params,
        run_seed,
        param_idx,
        output_dir,
        cols_slice,
        extra_feature,
        ds_name
    ) = config

    # 2. Set environment for this specific process (important for performance)
    # Using '40' allows a single algorithm to use all available cores.
    os.environ["OMP_NUM_THREADS"] = "40"
    os.environ["OPENBLAS_NUM_THREADS"] = "40"
    os.environ["MKL_NUM_THREADS"] = "40"
    os.environ["NUMEXPR_NUM_THREADS"] = "40"
    
    # 3. Ensure reproducibility within this process
    np.random.seed(run_seed)

    # --- Data Loading ---
    try:
        df = pd.read_csv(file_path, sep=',', low_memory=False, dtype={'CURRENTTIMESTAMP': str})
    except:
        df = pd.read_csv(file_path, sep=',', low_memory=False)
            
    if ds_name=="PV":   

        cols = df.columns[cols_slice]
        col_target = df.columns[-1]
        stream = NumpyStream(df[cols].values, df[col_target].values, dataset_name=ds_name, feature_names=cols)
        schema = stream.get_schema()
        if p_window_size == None: 
            params_c = params.copy()
            p_window_size = params_c.pop('p_window_size', None)
            window_size = max(1, int(len(df) * p_window_size))
        else:
            params_c = params.copy()
            window_size = max(1, int(len(df) * p_window_size))
    
        if extra_feature == "Voltage":
            col_extra_feature = slice(2081, 2082)
            extra_col = df.columns[col_extra_feature]
            extra_df = df[extra_col]
        elif extra_feature == "Current":
            col_extra_feature = slice(2082, 2083)
            extra_col = df.columns[col_extra_feature]
            extra_df = df[extra_col]
        elif extra_feature == "Pressure":
            col_extra_feature = slice(2083, 2084)
            extra_col = df.columns[col_extra_feature]
            extra_df = df[extra_col]
        else:
            extra_df = None
    
    # --- Model Initialization ---
    learner = get_model_with_params(
        model_name, params_c, window_size=window_size, random_seed=run_seed, schema=schema
    )

    
    stream.restart()
    
    # --- Main Processing Loop ---
    results_data = []
    for row, instance in enumerate(stream):
        if extra_feature == "Voltage" or extra_feature == "Current" or extra_feature == "Pressure":
            extra_data = float(extra_df.iloc[row].iloc[0]) # obtaining extra data
        elif extra_feature == "MomSpectra_NONE_":
            extra_data = median_of_means(instance.x)
        elif extra_feature == "SumSpectra_NONE_":
            extra_data = np.sum(instance.x)
        elif extra_feature == "SumSpectra_ZNORM_":
            extra_data = np.sum(transform_instance(instance, "ZNORM").x)
        elif extra_feature == "MomSpectra_ZNORM_":
            extra_data = median_of_means(transform_instance(instance, "ZNORM").x)
        else:
            extra_data = None # obtaining extra data


        

        # --- API-Specific Logic for Training and Scoring ---
        # time.perf_counter() is used for precise timing within this process.
        if hasattr(learner, "fit_partial"):  # Pysad models
            start_train = time.perf_counter()
            learner.fit_partial(instance.x, extra_data)
            training_time = time.perf_counter() - start_train
            start_score = time.perf_counter()
            score = learner.score_partial(instance.x, extra_data)
            scoring_time = time.perf_counter() - start_score
        elif hasattr(learner, "train"):  # Capymoa models
            start_score = time.perf_counter()
            score = learner.score_instance(instance, extra_data)
            scoring_time = time.perf_counter() - start_score
            start_train = time.perf_counter()
            learner.train(instance, extra_data)
            training_time = time.perf_counter() - start_train
        elif hasattr(learner, "fit_predict"):  # dSalmon models
            training_time = 0
            start_score = time.perf_counter()
            score = learner.fit_predict(instance.x, extra_data)
            scoring_time = time.perf_counter() - start_score
        else:
            raise AttributeError(f"Model {model_name} has no recognized training/scoring method.")

        cleaned_score, error_score = clean_score(score)
        
        if ds_name == "PV": 
            step = df.iloc[row, 0]
        else: 
            step = row
        
        results_data.append({
            "iteration": run_seed,
            "timestamp": step,
            "method": model_name,
            "param": str(params),  # Convert dict to string for CSV compatibility
            "score": score,
            "cleaned_score": cleaned_score,
            "training_time": training_time,
            "scoring_time": scoring_time,
            "ground_truth": instance.y_index,
            "error_type_score": error_score,
            "idx_model_scoring": 1, #it will be updated with the number of model used for scoring
        })
    
    # --- Save Results ---
    results_df = pd.DataFrame(results_data)
    
    file_name_prefix = file_path.stem.split("_")[0]+"_"+file_path.stem.split("_")[1]
    output_filename = f"{file_name_prefix}_results_{model_name}_iter_{run_seed}_paramset_{param_idx}_{ds_name}_V2_ds_ws_{window_size}.xlsx"


    full_path = output_dir / output_filename
    results_df.to_excel(full_path, index=False)
    
    return f"SUCCESS: Run seed {run_seed} for {model_name} on {file_path.name} with params {param_idx}."

def get_model_with_params(model_name, param_grid, window_size, random_seed, schema):
    
    ################## Pysad Methods
    if model_name == "xStream":
        return xStream(window_size=window_size, **param_grid)
    elif model_name == "RSHash":
        return RSHash(sampling_points=window_size, **param_grid)
    elif model_name == "IForestASD":
        return IForestASD(window_size=window_size, **param_grid)
    elif model_name == "RobustRandomCutForest":
        return RobustRandomCutForest(shingle_size=window_size, **param_grid)
    elif model_name == "KitNet": # Fixed Seed
        return KitNet(grace_feature_mapping=window_size, grace_anomaly_detector=window_size , **param_grid)
    elif model_name == "ExactStorm":
        return ExactStorm(window_size=window_size, **param_grid)

    ################## Capymoa Methods
    elif model_name == "oIF":
        return OnlineIsolationForest(schema=schema, window_size=window_size, random_seed=random_seed, **param_grid)
    elif model_name == "HStree":
        return HStreeCapy(schema=schema, window_size=window_size, random_seed=random_seed, **param_grid)
    
    ################## dSalmon Methods
    elif model_name == "SWKNN": # Deterministic model (no need seed)
        return SWKNN(window=window_size, **param_grid)
    elif model_name == "SWLOF": # Deterministic model (no need seed)
        return SWLOF(window=window_size, **param_grid)
    
    ################## Own Capymoa Methods
    elif model_name == "OnlineBootKNN":
        return OnlineBootKNN(schema=schema, window_size=window_size, random_seed=random_seed, **param_grid)
    elif model_name == "SWKNN_own": # Deterministic model (no need seed)
        return OnlineBootKNN(schema=schema, window_size=window_size, chunk_size=window_size, ensemble_size=1, random_seed=random_seed, **param_grid)
    elif model_name == "BKNN":
        return OnlineBootKNN(schema=schema, window_size=window_size, random_seed=random_seed, **param_grid)
    elif model_name == "OnlineBootKNN_TNone":
        return OnlineBootKNN(schema=schema, window_size=window_size, random_seed=random_seed, **param_grid)
    elif model_name == "OnlineBootKNN_TZnorm":
        return OnlineBootKNN(schema=schema, window_size=window_size, random_seed=random_seed, **param_grid)
    
    ################## DragStream
    elif model_name == "mDragStream":
        return mDragStream(**param_grid, n_init=window_size)
    elif model_name == "point_mDragStream_case1":
        return point_mDragStream_case1(**param_grid, n_init=window_size)
    elif model_name == "point_mDragStream_case2":
        return point_mDragStream_case2(**param_grid, n_init=window_size)
    elif model_name == "point_mDragStream_case3":
        return point_mDragStream_case3(**param_grid, n_init=window_size)
    elif model_name == "point_mDragStream_case4":
        return point_mDragStream_case4(**param_grid, n_init=window_size)

    elif model_name == "subsequence_mDragStream_case1":
        return subsequence_mDragStream_case1(**param_grid, n_init=window_size)
    elif model_name == "subsequence_mDragStream_case2":
        return subsequence_mDragStream_case2(**param_grid, n_init=window_size)
    elif model_name == "subsequence_mDragStream_case3":
        return subsequence_mDragStream_case3(**param_grid, n_init=window_size)
    elif model_name == "subsequence_mDragStream_case4":
        return subsequence_mDragStream_case4(**param_grid, n_init=window_size)

    ################## Wrappers for Pysad Methods (subset framework)
    elif model_name == "SFxStream":
        return SF_Pysad(learner=xStream, window_size=window_size, **param_grid)
    elif model_name == "SFRSHash":
        return SF_Pysad(learner=RSHash, sampling_points=window_size, **param_grid)
    elif model_name == "SFIForestASD":
        return SF_Pysad(learner=IForestASD, window_size=window_size, **param_grid)
    elif model_name == "SFRobustRandomCutForest":
        return SF_Pysad(learner=RobustRandomCutForest, shingle_size=window_size, **param_grid)
    elif model_name == "SFKitNet": # Fixed Seed
        return SF_Pysad(learner=KitNet, grace_feature_mapping=window_size, grace_anomaly_detector=window_size , **param_grid)
    elif model_name == "SFExactStorm":
        return SF_Pysad(learner=ExactStorm, window_size=window_size, **param_grid)

    ################## Wrappers Capymoa Methods (subset framework)
    elif model_name == "SFoIF":
        return SF_Capymoa(schema=schema, learner=OnlineIsolationForest, window_size=window_size, random_seed=random_seed, **param_grid)
    elif model_name == "SFHStree":
        return SF_Capymoa(schema=schema, learner=HStreeCapy, window_size=window_size, random_seed=random_seed, **param_grid)

    ################## Wrappers dSalmon Methods (subset framework)
    elif model_name == "SFSWLOF": # Deterministic model (no need seed)
        return SF_dSalmon(learner=SWLOF, window=window_size, **param_grid)

    ################## Wrappers Own Capymoa Methods (subset framework)
    elif model_name == "SFOnlineBootKNN_TNone":
        return SF_Capymoa(schema=schema, learner=OnlineBootKNN, window_size=window_size, random_seed=random_seed, **param_grid)
    elif model_name == "SFOnlineBootKNN_TZnorm":
        return SF_Capymoa(schema=schema, learner=OnlineBootKNN, window_size=window_size, random_seed=random_seed, **param_grid)

    ################## No Model Error
    else:
        raise ValueError(f"Unknown model: {model_name}")
    

