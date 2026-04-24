"""
Offline Multimodal Embedding Extraction for Hierarchical Modeling.

[Two-Stage Hierarchical Architecture & OOM Prevention]
Clinical interviews in the DAIC-WOZ dataset contain hundreds of utterances per session. 
Training a deep Foundation Model (BERT + Wav2Vec 2.0) end-to-end across an entire session's sequence of utterances is computationally prohibitive and inevitably leads 
to GPU Out-Of-Memory (OOM) errors. 

To solve this, adopt a two-stage hierarchical approach. This script represents 
Stage 1: Offline extraction. freeze the heavy foundation models to project utterance-level raw data into dense semantic and acoustic embeddings (768-dim). 
These pre-computed embeddings can then be fed into a lightweight sequence model 
(e.g., LSTM or Transformer) in Stage 2 to capture the long-term progression of depressive symptoms over the course of the interview.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np

from bert_wav2vec.utterance_dataset import UtteranceDataset, collate_fn
from transformers import BertModel

# Setting
BATCH_SIZE = 16
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = str(PROJECT_ROOT / "cache" / "foundation_features")
OUTPUT_DIR = str(PROJECT_ROOT / "cache" / "session_features")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

class FeatureExtractor(nn.Module):
    """A model that extracts embeddings only using BERT and Wav2Vec2"""
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Audio Encoder (simple CNN -> Mean Pooling)
        self.audio_proj = nn.Linear(768, 768)
        
    def forward(self, input_ids, attention_mask, audio_features, audio_mask):
        # 1. Text Embedding (BERT [CLS])
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_emb = bert_out.pooler_output # (B, 768)
            
        # 2. Audio Embedding (Simple Mean Pooling)
        # Wav2Vec2 Feature already extracted(B, T, 768)
        mask_expanded = audio_mask.unsqueeze(-1)
        audio_sum = (audio_features * mask_expanded).sum(dim=1)
        audio_len = mask_expanded.sum(dim=1).clamp(min=1e-9)
        audio_emb = audio_sum / audio_len
        
        return text_emb, audio_emb

def extract_and_save(loader, extractor, split_name):
    extractor.eval()
    
    # Collect data by session.
    session_buffer = {}
    
    print(f" Extracting {split_name} embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            audio = batch['audio_features'].to(device)
            audio_mask = batch['audio_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Extract
            text_emb, audio_emb = extractor(input_ids, attn_mask, audio, audio_mask)
            pass

def main():
    # Load dataset 
    train_ds = UtteranceDataset(CACHE_DIR, split='train')
    dev_ds = UtteranceDataset(CACHE_DIR, split='dev')
    test_ds = UtteranceDataset(CACHE_DIR, split='test')
    
    extractor = FeatureExtractor().to(device)
    extractor.eval()
    
    for ds, split in [(train_ds, 'train'), (dev_ds, 'dev'), (test_ds, 'test')]:
        print(f"\nProcessing {split} set...")
        
        # Group by data
        grouped_data = {}
        for item in tqdm(ds.data):
            sid = item['session_id']
            if sid not in grouped_data:
                grouped_data[sid] = {'text': [], 'audio': [], 'label': item['label']}
            
            grouped_data[sid]['text'].append(str(item['text']))
            grouped_data[sid]['audio'].append(item['audio_feature'])
            
        # Perform inference and save results for each session.
        for sid, data in tqdm(grouped_data.items(), desc="Sessions"):
            # Batch processing
            texts = data['text']
            audios = data['audio']
            
            # Text tokenization (in batches)
            encodings = ds.tokenizer(
                texts, 
                max_length=64, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(device)
            attn_mask = encodings['attention_mask'].to(device)
            
            # Convert audio to tensor & apply padding
            audio_tensors = [torch.tensor(a, dtype=torch.float32) for a in audios]
            audio_padded = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True).to(device)
            
            # Audio Mask
            audio_mask = torch.zeros(audio_padded.shape[:2]).to(device)
            for i, a in enumerate(audio_tensors):
                audio_mask[i, :a.shape[0]] = 1
                
            # Feature Extraction
            # Execute in mini-batches to save memory
            t_embs = []
            a_embs = []
            
            mini_bs = 32
            num_utts = len(texts)
            
            with torch.no_grad():
                for i in range(0, num_utts, mini_bs):
                    end = min(i + mini_bs, num_utts)
                    t_out, a_out = extractor(
                        input_ids[i:end], 
                        attn_mask[i:end], 
                        audio_padded[i:end], 
                        audio_mask[i:end]
                    )
                    t_embs.append(t_out.cpu())
                    a_embs.append(a_out.cpu())
            
            final_text_emb = torch.cat(t_embs, dim=0).numpy()
            final_audio_emb = torch.cat(a_embs, dim=0).numpy()
            
            # Save
            save_data = {
                'session_id': sid,
                'label': data['label'],
                'text_emb': final_text_emb,   # (Num_Utts, 768)
                'audio_emb': final_audio_emb  # (Num_Utts, 768)
            }
            
            with open(os.path.join(OUTPUT_DIR, f"{sid}_emb.pkl"), 'wb') as f:
                pickle.dump(save_data, f)

if __name__ == "__main__":
    main()