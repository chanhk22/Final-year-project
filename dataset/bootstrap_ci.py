"""
Statistical Validation code: Bootstrap Confidence Intervals (95% CI).

[Methodological Rationale: Robustness on Small Clinical Datasets]
In medical machine learning, evaluating a model on a small test set often yields high-variance point estimates. 
A single F1 or AUC score is statistically insufficient to prove clinical reliability. 
This code implements non-parametric Bootstrap Resampling (1,000 iterations) to compute the 95% Confidence Intervals (CI). 
This demonstrates to clinical and academic reviewers that the model's performance is statistically robust and not merely an artifact of a lucky test split.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils import resample
import argparse

def calculate_bootstrap_ci(y_true, y_pred_prob, n_iterations=1000, threshold=0.5, alpha=0.95):
    """
    Perform 1,000 bootstrap iterations based on the true labels and predicted probabilities to calculate the 95% confidence interval
    """
    f1_scores = []
    auc_scores = []
    
    n_size = len(y_true)
    
    for i in range(n_iterations):
        # Sample 47 items with replacement (allowing duplicates)
        indices = resample(np.arange(n_size), n_samples=n_size, random_state=i)
        y_true_resampled = y_true[indices]
        y_prob_resampled = y_pred_prob[indices]
        
        # AUC can only be calculated if both positive and negative classes are present
        if len(np.unique(y_true_resampled)) < 2:
            continue
            
        y_pred_binary = (y_prob_resampled >= threshold).astype(int)
        
        f1 = f1_score(y_true_resampled, y_pred_binary, zero_division=0)
        auc = roc_auc_score(y_true_resampled, y_prob_resampled)
        
        f1_scores.append(f1)
        auc_scores.append(auc)
        
    # Calculate the 2.5th and 97.5th percentiles (95% confidence interval)
    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    f1_ci = (np.percentile(f1_scores, lower_p), np.percentile(f1_scores, upper_p))
    auc_ci = (np.percentile(auc_scores, lower_p), np.percentile(auc_scores, upper_p))
    
    # Original score
    orig_f1 = f1_score(y_true, (y_pred_prob >= threshold).astype(int), zero_division=0)
    orig_auc = roc_auc_score(y_true, y_pred_prob)
    
    print("-" * 50)
    print(f"Original F1 Score : {orig_f1:.4f}")
    print(f"F1 95% CI         : [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
    print("-" * 50)
    print(f"Original AUC Score: {orig_auc:.4f}")
    print(f"AUC 95% CI        : [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the predictions CSV file")
    args = parser.parse_args()
    
    # The CSV file must contain the columns 'true_label' and 'pred_prob'
    df = pd.read_csv(args.csv_path)
    
    y_true = df['true_label'].values
    y_prob = df['pred_prob'].values
    
    calculate_bootstrap_ci(y_true, y_prob, n_iterations=1000)