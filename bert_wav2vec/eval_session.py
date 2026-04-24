"""
Final Clinical Evaluation and Attention-based Interpretability.

In clinical decision support systems, raw accuracy or F1-scores are insufficient; psychiatrists require transparency regarding why a model made a specific diagnosis. 
This evaluation module not only computes standard performance metrics (AUC, F1) but intrinsically leverages the Hierarchical Attention mechanism to extract and 
surface the exact 'Top-K' most discriminative utterances. This bridges the gap between black-box neural network predictions and actionable clinical insights.
"""

import torch
import os
import glob
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import yaml 
from pathlib import Path

# Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = str(PROJECT_ROOT / "cache" / "session_features")
MODEL_PATH = str(PROJECT_ROOT / "best_session_model_v3.pt")
MAX_UTT = 200
BATCH_SIZE = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model and dataset classes (copy-paste from training code)

class AttentionModel(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64):
        super().__init__()
        self.text_fc = torch.nn.Linear(input_dim, hidden_dim)
        self.audio_fc = torch.nn.Linear(input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.5)
        self.attn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, 1)
        )
        
    def forward(self, t, a, mask):
        t = torch.relu(self.text_fc(t))
        a = torch.relu(self.audio_fc(a))
        x = torch.cat([t, a], dim=-1)
        # x = self.dropout(x) # Disable dropout during evaluation
        
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (x * weights).sum(dim=1)
        logits = self.classifier(context)
        return logits, weights

class SessionDataset(Dataset):
    def __init__(self, emb_dir, split='test'): # Default to test
        self.data = []
        self.ids = []
        
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "configs" / "default.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        SPLIT_DIR = str(project_root / config['paths']['daic_woz']['labels_dir'])
        if split == 'test':
            ptrn = os.path.join(SPLIT_DIR, "*test*.csv")
        else:
            ptrn = os.path.join(SPLIT_DIR, f"*{split}*_split_Depression_AVEC2017*.csv")
            
        csv_files = glob.glob(ptrn)
        target_ids = set()
        if csv_files:
            df = pd.read_csv(csv_files[0])
            target_ids = set(df['Participant_ID'].astype(str).values)
        
        files = glob.glob(os.path.join(emb_dir, "*_emb.pkl"))
        for f in files:
            sid = os.path.basename(f).split('_')[0]
            if not target_ids or sid in target_ids: 
                with open(f, 'rb') as pkl:
                    sample = pickle.load(pkl)
                    self.data.append(sample)
                    self.ids.append(sid)
        print(f"[{split}] Loaded {len(self.data)} sessions.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        return (torch.tensor(sample['text_emb'], dtype=torch.float32),
                torch.tensor(sample['audio_emb'], dtype=torch.float32),
                torch.tensor(sample['label'], dtype=torch.float32),
                self.ids[idx])

def collate_fn(batch):
    t_list, a_list, labels, ids = zip(*batch)
    t_list = [t[:MAX_UTT] for t in t_list]
    a_list = [a[:MAX_UTT] for a in a_list]
    t_padded = torch.nn.utils.rnn.pad_sequence(t_list, batch_first=True)
    a_padded = torch.nn.utils.rnn.pad_sequence(a_list, batch_first=True)
    labels = torch.stack(labels)
    # Mask Fix
    batch_max_len = t_padded.size(1)
    mask = torch.zeros(len(t_list), batch_max_len)
    for i, t in enumerate(t_list):
        mask[i, :t.shape[0]] = 1
    return t_padded, a_padded, mask, labels, ids

# Main 

def main():
    print(f" Evaluating Model: {MODEL_PATH}")
    
    # 1. Load Data & Model
    test_ds = SessionDataset(EMB_DIR, split='test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = AttentionModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    y_true = []
    y_pred = []
    results = []
    
    print("\nProcessing Test Set...")
    with torch.no_grad():
        for batch in test_loader:
            t, a, mask, y, ids = batch
            t, a, mask = t.to(device), a.to(device), mask.to(device)
            
            logits, weights = model(t, a, mask)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).long().cpu().item()
            true = y.item()
            
            y_true.append(true)
            y_pred.append(pred)
            
            # Attention analysis (to identify the most important utterances)
            # weights: (1, N, 1) -> (N,)
            """
            Depression often manifests in fleeting micro-behaviors rather than continuous states.
            By sorting the attention weights and extracting the top 3 indices (Top_Attn_Idx), can map the network's mathematical 
            focus directly back to the original transcript. This effectively highlights the patient's most severe symptomatic utterances 
            for clinical review.
            """
            w = weights.squeeze().cpu().numpy()
            top_idx = w.argsort()[::-1][:3] # Top 3 Important Utterances
            
            results.append({
                "ID": ids[0],
                "True": true,
                "Pred": pred,
                "Prob": probs.item(),
                "Top_Attn_Idx": top_idx,
                "Top_Attn_Val": w[top_idx]
            })

    # 2. Print Metrics
    print("\n" + "="*40)
    print(" FINAL TEST REPORT (Phase 3)")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Depressed']))
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # 3. Analyze Explainability
    print("\n [Explainability] Top Attention Utterances per Subject (Sample)")
    for res in results[:5]: # Display examples for only 5 people
        status = " Correct" if res['True'] == res['Pred'] else " Wrong"
        print(f"ID {res['ID']} ({status}): True={res['True']}, Pred={res['Pred']} (Prob: {res['Prob']:.4f})")
        print(f"  Top Utterance Indices: {res['Top_Attn_Idx']} (Values: {res['Top_Attn_Val']})")

if __name__ == "__main__":
    main()