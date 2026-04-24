"""
Phase 2: Utterance-Level Foundation Model Training Pipeline.

[Utterance-Level Optimization]
Unlike continuous non-verbal signals (e.g., facial Action Units), linguistic content (BERT) and its acoustic delivery (Wav2Vec 2.0) are inherently bounded by spoken utterances. 
This training code optimizes the multimodal foundation models specifically at the utterance level, ensuring that semantic meaning and prosodic nuances are learned within their natural conversational boundaries.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np

from bert_wav2vec.utterance_dataset import UtteranceDataset, collate_fn
from bert_wav2vec.foundation_model import FoundationMultimodalClassifier

# Hyperparameter setting
BATCH_SIZE = 16        
LR = 5e-4               
EPOCHS = 50             
PATIENCE = 15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = str(PROJECT_ROOT / "cache" / "foundation_features")
SAVE_DIR = str(PROJECT_ROOT / "checkpoints_bert_wav2vec" / "foundation_v1")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(SAVE_DIR, exist_ok=True)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    valid_batches = 0
    
    loop = tqdm(loader, desc="Training")
    for batch in loop:
        if batch is None: continue
        
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        audio = batch['audio_features'].to(device)
        audio_mask = batch['audio_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        output = model(input_ids, attn_mask, audio, audio_mask)
        logits = output['logits']
        loss = criterion(logits, labels)
        
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
        loop.set_postfix(loss=loss.item())
        
    return total_loss / max(valid_batches, 1)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    valid_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            audio = batch['audio_features'].to(device)
            audio_mask = batch['audio_mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)
            
            output = model(input_ids, attn_mask, audio, audio_mask)
            logits = output['logits']
            loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.cpu().numpy()
            
            total_loss += loss.item()
            valid_batches += 1
            
            # Flatten into a one-dimensional list and store it
            all_preds.extend(preds.reshape(-1).tolist())
            all_labels.extend(labels_np.reshape(-1).tolist())
            all_probs.extend(probs.reshape(-1).tolist())
            
    if valid_batches == 0:
        return 0, 0, 0, 0.5
        
    # Metric calculation
    y_true = np.array(all_labels).astype(int)
    y_pred = np.array(all_preds).astype(int)
    y_probs = np.nan_to_num(np.array(all_probs), nan=0.5)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.5
        
    return total_loss / valid_batches, acc, f1, auc

def main():
    print(f" Start Training Foundation Model (Phase 2) on {device}")
    
    # 1. Load dataset
    try:
        train_ds = UtteranceDataset(CACHE_DIR, split='train')
        dev_ds = UtteranceDataset(CACHE_DIR, split='dev')
        test_ds = UtteranceDataset(CACHE_DIR, split='test')
    except Exception as e:
        print(f" Error occurred while loading the data: {e}")
        return

    if len(train_ds) == 0:
        print(" The train dataset is empty. Please check the .pkl file path.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # 2. Initialize the model (Freeze Backbone)
    model = FoundationMultimodalClassifier(use_lora=True).to(device)
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    """
    Clinical datasets like DAIC-WOZ suffer from severe class imbalance, with healthy controls significantly outnumbering depressed patients. 
    To prevent the model from trivially predicting the majority class, inject a positive weight (pos_weight=3.0) into the Binary Cross Entropy loss. 
    Clinically, penalizing False Negatives more heavily is crucial, as missing a depressed patient is substantially more dangerous than misclassifying a healthy individual.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
    
    best_dev_f1 = 0.0
    patience_counter = 0
    
    # 4. Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, dev_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
        
        if val_f1 > best_dev_f1:
            best_dev_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pt"))
            print(" New Best Model Saved!")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(" Early Stopping.")
            break
            
    # 5. Final evaluation
    if os.path.exists(os.path.join(SAVE_DIR, "best.pt")):
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best.pt")))
        test_loss, test_acc, test_f1, test_auc = evaluate(model, test_loader, criterion)
        print("="*40)
        print(" FINAL TEST RESULTS (Utterance Level)")
        print(f"Accuracy: {test_acc:.4f} | F1 Score: {test_f1:.4f} | AUC: {test_auc:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()