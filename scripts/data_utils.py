import pandas as pd
pd.set_option('mode.use_inf_as_na', True)

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import re
from vus.metrics import get_metrics
from vus.utils.utility import get_list_anomaly
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_performance_metrics(df, gt_column, score_column, t_window_size=None, score_direction='direct'):
    # Calculates performance metrics for anomaly detection.

    # --- 1. Initial Data Preparation ---
    labels = df[gt_column].values
    scores = df[score_column].values

    # Robustness Check: Ensure there are both normal and anomalous points.
    if len(np.unique(labels)) < 2:
        print(f"Warning: Ground truth for '{gt_column}' has only one class. Metrics cannot be calculated.")
        return None

    # --- 2. Handle Score Direction and Offset ---

    # Step 2a: Handle direction first
    if score_direction == 'inverse':
        scores_to_use = -scores
    elif score_direction == 'direct':
        scores_to_use = scores
    else:
        raise ValueError("score_direction must be 'direct' or 'inverse'")
    
    # Step 2b: Calculate Standard Metrics (on original scores) ---
    roc_auc_wtd = roc_auc_score(labels, scores_to_use)

    precision_wtd, recall_wtd, _ = precision_recall_curve(labels, scores_to_use)
    pr_auc_wtd = auc(recall_wtd, precision_wtd)

    # Calculate F1 scores and find the maximum
    f1_scores_wtd = 2 * (precision_wtd * recall_wtd) / (precision_wtd + recall_wtd + 1e-8)
    max_f1_wtd = np.max(f1_scores_wtd)
    
    # Step 3b: Handle slicing second
    if t_window_size is not None:
        final_labels = labels[t_window_size:]
        final_scores = scores_to_use[t_window_size:]
    else:
        # If t_window_size is None, use the full arrays
        final_labels = labels
        final_scores = scores_to_use
    
    if len(final_labels) == 0 or len(np.unique(final_labels)) < 2:
        print(f"Warning: After applying t_window_size, no data or only one class remains. Metrics cannot be calculated.")
        return None
    
    # Step 3c. Calculate Standard Metrics (on shorted scores) ---
    roc_auc = roc_auc_score(final_labels, final_scores)

    precision, recall, thresholds = precision_recall_curve(final_labels, final_scores)
    pr_auc = auc(recall, precision)

    # Calculate F1 scores and find the maximum
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[best_f1_idx]

    # Generate binary predictions based on optimal threshold
    if best_f1_idx < len(thresholds):
        best_threshold = thresholds[best_f1_idx]
    else:
        # Fallback: usually the highest score in the data if we reached the end
        best_threshold = np.max(final_scores)
        
    binary_preds = (final_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(final_labels, binary_preds).ravel()

    # Calculate specific percentages
    pct_detection = recall[best_f1_idx]  # Same as Recall or calculate as: tp / (tp + fn)
    pct_false_positives = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Positive Rate

    # --- 3. Calculate VUS Metrics (on normalized scores) ---
    # Create a new, separate variable for the normalized scores to avoid bugs.
    normalized_scores = MinMaxScaler().fit_transform(final_scores.reshape(-1, 1)).ravel()

    # Robustness Check: Calculate sliding window safely.
    anomaly_lengths = get_list_anomaly(labels)

    slidingWindow = int(np.median(anomaly_lengths)) - 1 

    metrics = get_metrics(normalized_scores, final_labels, metric='all', slidingWindow=slidingWindow)

    # --- 4. Return all metrics in a clear, self-describing dictionary ---
    return roc_auc, pr_auc, max_f1, metrics, roc_auc_wtd, pr_auc_wtd, max_f1_wtd, pct_detection, pct_false_positives, tn, fp, fn, tp, best_threshold

def split_summary_methods(method_window_and_param):
    match = re.match(r"([^_]+)_([^_]+)_\{(.+)\}", method_window_and_param)
    if match:
        method = match.group(1)
        window = match.group(2)
        params = "{" + match.group(3) + "}"
        return method, window, params
    else:
        return method_window_and_param, None, None

def format_mean_std(mean_val, std_val):
    # If the mean itself is missing, just return NaN
    if pd.isna(mean_val):
        return np.nan
    
    # Format the mean
    mean_str = f"{mean_val:.3f}"
    
    # Check if std is NaN (e.g., for non-stochastic 'N' rows)
    if pd.isna(std_val):
        return f"{mean_str} ± NA"  # Return *only* the mean
    else:
        # If std exists, format and combine
        std_str = f"{std_val:.1e}"
        return f"{mean_str} ± {std_str}"

def reduce_and_correlate(X, n_intervals=None, agg='max', threshold=0.5, plot_heatmap=True):
    if n_intervals is None or n_intervals <= 1:
        df_reduced = X.copy()
    else:
        chunks = np.array_split(X.columns, n_intervals)
        reduced = {}
        for chunk in chunks:
            name = f'interval_from_{chunk[0]}_to_{chunk[-1]}'
            reduced[name] = getattr(X[chunk], agg)(axis=1)
        df_reduced = pd.DataFrame(reduced)

    corr_matrix = df_reduced.corr()

    mask = ~np.eye(len(corr_matrix), dtype=bool)
    avg_abs_corr = corr_matrix.where(mask).abs().mean(axis=1)

    not_correlated = avg_abs_corr[avg_abs_corr < threshold]
    correlated = avg_abs_corr[avg_abs_corr >= threshold]
    na_values = avg_abs_corr[avg_abs_corr.isna()]

    print(f"[n_intervals={n_intervals}, agg='{agg}'] "
          f"Total: {len(avg_abs_corr)} | "
          f"Not correlated (< {threshold}): {len(not_correlated)} | "
          f"Correlated (>= {threshold}): {len(correlated)} | "
          f"No Variance: {len(na_values)} | "
          f"% of Correlated (>= {threshold}): {len(correlated)/(len(correlated)+len(not_correlated)+len(na_values))}")

    if plot_heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=corr_matrix.shape[0] <= 20,
                    cmap='coolwarm', fmt=".2f", square=True)
        plt.title(f'n_intervals={n_intervals}, agg={agg}')
        plt.show()

    return df_reduced, corr_matrix, avg_abs_corr, not_correlated   