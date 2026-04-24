"""
Session-Level Temporal Attention Network for Depression Detection.

[Hierarchical Sequence Modeling]
Following the offline extraction of utterance-level foundation embeddings (Stage 1), this code implements the final patient-level classification. 
By modeling the sequence of utterances over time, the network captures the long-term longitudinal 
progression of depressive markers (e.g., fatigue, psychomotor retardation) throughout the entire clinical interview.
"""
import os
import glob
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

#Settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = str(PROJECT_ROOT / "cache" / "session_features")
BATCH_SIZE = 16
EPOCHS = 50
LR = 2e-5           
WEIGHT_DECAY = 1e-2    
MAX_UTT = 200           

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SessionDataset(Dataset):
    def __init__(self, emb_dir, split='train'):
        self.data = []
        self.labels = [] # Save labels separately for the sampler.
        
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "configs" / "default.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # Split
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
                    self.labels.append(int(sample['label'])) # Collect labels
        
        print(f"[{split}] Loaded {len(self.data)} sessions.")
        if len(self.data) > 0:
            print(f"[{split}] Label Distribution: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        t_emb = torch.tensor(sample['text_emb'], dtype=torch.float32)
        a_emb = torch.tensor(sample['audio_emb'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.float32)
        return t_emb, a_emb, label

def collate_fn(batch):
    t_list, a_list, labels = zip(*batch)
    
    # 1. MAX_UTT
    t_list = [t[:MAX_UTT] for t in t_list]
    a_list = [a[:MAX_UTT] for a in a_list]
    
    # 2. Padding (adjusted to the longest length within the batch)
    t_padded = torch.nn.utils.rnn.pad_sequence(t_list, batch_first=True)
    a_padded = torch.nn.utils.rnn.pad_sequence(a_list, batch_first=True)
    labels = torch.stack(labels)
    
    # 3. Obtain the length (size(1)) of the actual padded tensor to create a mask
    batch_max_len = t_padded.size(1)
    mask = torch.zeros(len(t_list), batch_max_len)
    
    for i, t in enumerate(t_list):
        mask[i, :t.shape[0]] = 1
        
    return t_padded, a_padded, mask, labels

"""
[Clinical Class Imbalance Mitigation]
In session-level training, the absolute number of data points is drastically reduced (down to the number of participants), making the class imbalance (control vs. depressed) extremely problematic. 
Instead of merely scaling the loss, employ a WeightedRandomSampler to dynamically oversample the minority class during mini-batch construction. 
This guarantees that the model observes a balanced class distribution, effectively preventing majority-class collapse.
"""
def get_sampler(dataset):
    labels = dataset.labels
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

class AttentionModel(nn.Module):
    """
    Depressive symptoms do not manifest continuously; a patient might sound perfectly normal for 80% of the interview but exhibit severe flat affect or negative semantics in specific responses. 
    Instead of naive mean pooling, this architecture utilizes a learned Self-Attention mechanism.
    It dynamically assigns higher weights to the most diagnostically relevant utterances, effectively filtering out neutral conversational noise and boosting interpretability.
    """
    def __init__(self, input_dim=768, hidden_dim=64):
        super().__init__()
        self.text_fc = nn.Linear(input_dim, hidden_dim)
        self.audio_fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        
        # Self-Attention
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
        
    def forward(self, t, a, mask):
        t = torch.relu(self.text_fc(t))
        a = torch.relu(self.audio_fc(a))
        x = torch.cat([t, a], dim=-1)
        x = self.dropout(x)
        
        # Attention
        scores = self.attn(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        
        # Context
        context = (x * weights).sum(dim=1)
        logits = self.classifier(context)
        return logits

def main():
    print(" Start Session Training (Weighted Sampler)")
    
    train_ds = SessionDataset(EMB_DIR, split='train')
    dev_ds = SessionDataset(EMB_DIR, split='dev')
    
    # Apply sampler (Shuffle=False is required)
    sampler = get_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = AttentionModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Since a sampler is used, remove or reduce pos_weight (for stable loss)
    criterion = nn.BCEWithLogitsLoss() 
    
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_preds, train_targets = [], []
        
        for batch in train_loader:
            if batch is None: continue
            t, a, mask, y = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            logits = model(t, a, mask)
            loss = criterion(logits, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            train_preds.extend((probs > 0.5).long().cpu().numpy().flatten())
            train_targets.extend(y.cpu().numpy().flatten())
            
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in dev_loader:
                t, a, mask, y = [b.to(device) for b in batch]
                logits = model(t, a, mask)
                probs = torch.sigmoid(logits)
                preds.extend((probs > 0.5).long().cpu().numpy().flatten())
                targets.extend(y.cpu().numpy().flatten())
                
        f1 = f1_score(targets, preds)
        acc = accuracy_score(targets, preds)
        
        # Check the prediction distribution
        unique, counts = np.unique(preds, return_counts=True)
        pred_dist = dict(zip(unique, counts))
        
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f} | "
              f"Dev F1: {f1:.4f} (Acc: {acc:.2f}) | Preds: {pred_dist}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_session_model_v3.pt")
            print(" New Best Model!")

if __name__ == "__main__":
    main()