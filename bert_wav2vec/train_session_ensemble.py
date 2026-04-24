import os
import glob
import pickle
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml 
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
config_path = PROJECT_ROOT / "configs" / "default.yaml"

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    
# 1. Settings (Identical to explainability_occlusion.py)
CACHE_DIR = str(PROJECT_ROOT / "cache" / "foundation_features")
MODEL_DIR = str(PROJECT_ROOT / "checkpoints_bert_wav2vec" / "session_ensemble")
SPLIT_DIR = str(PROJECT_ROOT / config['paths']['daic_woz']['labels_dir'])

PCA_COMPONENTS = 50 
NUM_MODELS = 5
MAX_UTTERANCES = 150
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Model Definition (identical to the Occlusion code)
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

# 3. Data Preprocessing Functions
def extract_stats(audio_matrix):
    """(N, 768) -> (2304,) Statistical pooling using Mean+Std+Max"""
    return np.concatenate([
        np.mean(audio_matrix, 0),
        np.std(audio_matrix, 0),
        np.max(audio_matrix, 0)
    ])

def load_data(split_type='train'):
    # Load labels
    all_csvs = glob.glob(os.path.join(SPLIT_DIR, "*.csv"))
    target_ids = {}
    for f in all_csvs:
        if split_type in os.path.basename(f).lower():
            df = pd.read_csv(f)
            pid_col = 'Participant_ID' if 'Participant_ID' in df.columns else 'participant_id'
            label_col = 'PHQ_Binary' if 'PHQ_Binary' in df.columns else 'PHQ8_Binary'
            for _, row in df.iterrows():
                target_ids[str(int(row[pid_col]))] = float(row[label_col])
            break
            
    X, y = [], []
    pkl_files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
    for pkl in pkl_files:
        sid_match = re.search(r'\d+', os.path.basename(pkl))
        if not sid_match: continue
        sid = sid_match.group()
        if sid in target_ids:
            with open(pkl, 'rb') as f:
                samples = pickle.load(f)
            audio_vecs = [np.mean(s['audio_feature'], axis=0) for s in samples[:MAX_UTTERANCES] 
                          if s.get('audio_feature') is not None]
            if not audio_vecs: continue
            X.append(extract_stats(np.stack(audio_vecs)))
            y.append(target_ids[sid])
    return np.array(X), np.array(y)

# 4. Main Training Loop
def main():
    print(" Loading data and fitting PCA...")
    X_train_raw, y_train = load_data('train')
    
    # Preprocessing pipeline (PCA fit exclusively on Train data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    X_train_t = torch.tensor(X_train_pca, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    
    # Train 5 models with different seeds (Ensemble effect)
    for i in range(NUM_MODELS):
        print(f" Training Ensemble Member {i}...")
        torch.manual_seed(42 + i)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        model = EnsembleMember(input_dim=PCA_COMPONENTS).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        
        # Calculate weights to resolve class imbalance
        pos_weight = torch.tensor([len(y_train)/sum(y_train) - 1.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        model.train()
        for epoch in range(EPOCHS):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        
        # Save model
        save_path = os.path.join(MODEL_DIR, f"model_{i}.pt")
        torch.save(model.state_dict(), save_path)
        print(f" Saved: {save_path}")

if __name__ == "__main__":
    main()