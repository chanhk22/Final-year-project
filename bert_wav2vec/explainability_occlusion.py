"""
Explainability Analysis via Occlusion Sensitivity with Temporal Grounding.

[Occlusion Sensitivity for Clinical AI]
While SHAP provides feature-level importance, psychiatrists ultimately need to know which specific moments in an interview triggered a depression diagnosis. 
This code implements Occlusion Sensitivity Analysis: it systematically masks out (occludes) individual utterances and measures the resulting drop in predicted depression probability. 
The utterances causing the largest probability drops are identified as the core clinical evidence, providing actionable, time-stamped insights directly mapped to the original transcript.
"""

import os
import glob
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml 
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re

# Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
config_path = PROJECT_ROOT / "configs" / "default.yaml"

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    
CACHE_DIR = str(PROJECT_ROOT / "cache" / "foundation_features")
MODEL_DIR = str(PROJECT_ROOT / "checkpoints_bert_wav2vec" / "session_ensemble")
SPLIT_DIR = str(PROJECT_ROOT / config['paths']['daic_woz']['labels_dir'])
OUTPUT_CSV = str(PROJECT_ROOT / "qualitative_analysis.csv")

PCA_COMPONENTS = 50 
NUM_MODELS = 5
MAX_UTTERANCES = 150
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition (for ensemble member)

class EnsembleMember(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# Utility functions

def extract_stats(audio_matrix):
    """(N, 768) -> (2304,) Mean+Std+Max"""
    return np.concatenate([
        np.mean(audio_matrix, 0),
        np.std(audio_matrix, 0),
        np.max(audio_matrix, 0)
    ])

def get_depression_prob(models, feature_vector, scaler, pca):
    # 1. Scaling & PCA
    feat_scaled = scaler.transform(feature_vector.reshape(1, -1))
    feat_pca = pca.transform(feat_scaled)
    feat_tensor = torch.tensor(feat_pca, dtype=torch.float32).to(device)
    
    # 2. Ensemble Prediction
    """
    Clinical datasets are inherently noisy. Relying on a single model's probability for 
    occlusion sensitivity can yield highly volatile and unreliable explanations. 
    By averaging the predictions across an ensemble of 5 models, stabilize the 
    decision boundary, ensuring that the identified impactful utterances are 
    genuinely robust diagnostic markers, not just model-specific artifacts.
    """
    preds = []
    with torch.no_grad():
        for m in models:
            logits = m(feat_tensor)
            preds.append(torch.sigmoid(logits).item())
    
    # 3. Inversion Logic
    avg_pred = np.mean(preds)
    prob_depression = 1.0 - avg_pred
    
    return prob_depression

def main():
    print(" Starting Qualitative Analysis (With Timestamps)...")
    
    # Rebuild the preprocessing pipeline (fit on the training data)
    
    print("  1. Fitting Scaler & PCA on Training Data...")
    
    all_csvs = glob.glob(os.path.join(SPLIT_DIR, "*.csv"))
    train_ids = set()
    for f in all_csvs:
        if 'train' in os.path.basename(f).lower():
            df = pd.read_csv(f)
            pid_col = 'Participant_ID'
            if pid_col not in df.columns: pid_col = 'participant_id'
            train_ids = set(df[pid_col].astype(str).str.strip().values)
            break
            
    X_train = []
    pkl_files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
    
    for pkl in pkl_files:
        filename = os.path.basename(pkl)
        sid_match = re.search(r'\d+', filename)
        if not sid_match: continue
        sid = sid_match.group()
        
        if sid in train_ids:
            try:
                with open(pkl, 'rb') as f:
                    samples = pickle.load(f)
                audio_vecs = []
                for s in samples[:MAX_UTTERANCES]:
                    a = s.get('audio_feature', None)
                    if a is not None and not np.isnan(a).any():
                        audio_vecs.append(np.mean(a, axis=0))
                if not audio_vecs: continue
                
                mat = np.stack(audio_vecs)
                feat = extract_stats(mat)
                X_train.append(feat)
            except: continue
            
    if not X_train:
        print(" Error: Failed to load training data.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.array(X_train))
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    pca.fit(X_scaled)
    print("    -> Pipeline Ready.")

    # Load model 
    
    models = []
    for i in range(NUM_MODELS):
        path = os.path.join(MODEL_DIR, f"model_{i}.pt")
        if os.path.exists(path):
            m = EnsembleMember(input_dim=PCA_COMPONENTS).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
    print(f"   2. Loaded {len(models)} Ensemble Models.")

    # 3. Analyze and save the test set
    
    print("\n   3. Analyzing and Saving Results...")
    
    test_ids = set()
    for f in all_csvs:
        if 'test' in os.path.basename(f).lower():
            df = pd.read_csv(f)
            label_col = 'PHQ_Binary' if 'PHQ_Binary' in df.columns else 'PHQ8_Binary'
            depressed_df = df[df[label_col] == 1]
            test_ids = set(depressed_df['Participant_ID'].astype(str).str.strip().values)
            break
    
    results_to_save = []
    
    for pkl in pkl_files:
        filename = os.path.basename(pkl)
        sid_match = re.search(r'\d+', filename)
        if not sid_match: continue
        sid = sid_match.group()
        
        if sid not in test_ids: continue
        
        with open(pkl, 'rb') as f:
            samples = pickle.load(f)
            
        utterances = []
        for s in samples[:MAX_UTTERANCES]:
            a = s.get('audio_feature', None)
            t = s.get('text', "[No Text]")
            
            # Use get safely, as key names may vary across datasets
            start_t = s.get('start_time', 0.0)
            stop_t = s.get('stop_time', 0.0)
            
            if a is not None and not np.isnan(a).any():
                utterances.append({
                    'audio': np.mean(a, axis=0), 
                    'text': t,
                    'start': start_t,  # 저장
                    'stop': stop_t     # 저장
                })
        
        if len(utterances) < 5: continue
        
        full_audio_matrix = np.stack([u['audio'] for u in utterances])
        full_feat = extract_stats(full_audio_matrix)
        base_prob = get_depression_prob(models, full_feat, scaler, pca)
        
        if base_prob < 0.5: continue 
            
        print(f"\nProcessing Participant {sid} (Prob: {base_prob:.2%})")
        
        # Occlusion
        impact_scores = []
        for i in range(len(utterances)):
            masked_matrix = np.delete(full_audio_matrix, i, axis=0)
            masked_feat = extract_stats(masked_matrix)
            masked_prob = get_depression_prob(models, masked_feat, scaler, pca)
            
            impact = base_prob - masked_prob
            impact_scores.append((i, impact))
            
        impact_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Save and display the top 3
        for rank in range(3):
            if rank >= len(impact_scores): break
            idx, score = impact_scores[rank]
            
            utt_info = utterances[idx]
            text_content = utt_info['text']
            start_time = utt_info['start']
            stop_time = utt_info['stop']
            
            # Display on screen (include timestamp)
            print(f"   Rank {rank+1} [{start_time:.2f}s - {stop_time:.2f}s] (Impact {score:.4f}): \"{text_content[:50]}...\"")
            
            # Append to list (for CSV saving)
            results_to_save.append({
                "Participant_ID": sid,
                "Depression_Probability": f"{base_prob:.4f}",
                "Rank": rank + 1,
                "Impact_Score": f"{score:.4f}",
                "Start_Time": f"{start_time:.2f}",  
                "Stop_Time": f"{stop_time:.2f}",    
                "Utterance": text_content
            })

    # Save as CSV file
    
    if results_to_save:
        # Specify column order
        columns = ["Participant_ID", "Depression_Probability", "Rank", "Impact_Score", "Start_Time", "Stop_Time", "Utterance"]
        df_results = pd.DataFrame(results_to_save, columns=columns)
        
        df_results.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print("\n" + "="*50)
        print(f" Analysis Completed! Results saved to: {OUTPUT_CSV}")
        print("="*50)
        print(df_results.head())
    else:
        print("\n No depressed patients were correctly predicted by the model.")

if __name__ == "__main__":
    main()