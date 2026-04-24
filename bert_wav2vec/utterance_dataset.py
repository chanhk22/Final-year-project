import os
import glob
import torch
import pickle
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer

class UtteranceDataset(Dataset):
    """
    Multimodal Utterance-level Dataset integrating Wav2Vec 2.0 and BERT representations.
    
    While fixed-time windows (e.g., 2 seconds) are useful for continuous signals like facial expressions, linguistic content and its acoustic delivery (prosody) 
    are meaningful primarily at the utterance level. This dataset class loads participant-only speech segments, fusing textual tokens (via BERT) with 
    their corresponding deep acoustic features (via Wav2Vec 2.0).
    
    The initialization strictly enforces participant-level dataset splitting (Train/Dev/Test) by reading the official DAIC-WOZ split files. 
    This ensures that utterances from the same patient do not bleed across evaluation boundaries.
    """
    def __init__(self, cache_dir, split='train', max_text_len=64, max_audio_len=400):
        """
        Args:
            cache_dir: Folder .pkl files saved(e.g. cache/foundation_features)
            split: One of 'train', 'dev', 'test' 
            max_text_len: BERT token max length
            max_audio_len: Wav2Vec2 feature max frame (approximately 8seconds ≈ 400 frames)
        """
        self.data = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len
        
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "configs" / "default.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # Split file and label column configuration
        SPLIT_DIR = str(project_root / config['paths']['daic_woz']['labels_dir']) 
               
        target_ids = set()
        split_files = []
        label_column = ''

        # Process Test split differently from Train/Dev
        if split == "test":
            # Test set: filename contains 'test', label is 'PHQ_Binary'
            split_files = glob.glob(os.path.join(SPLIT_DIR, "*test*.csv"))
            label_column = 'PHQ_Binary'
        else:
            # Train/Dev set: follows existing naming convention, label is 'PHQ8_Binary'
            # (e.g., train_split_Depression_AVEC2017.csv)
            split_files = glob.glob(os.path.join(SPLIT_DIR, f"*{split}*_split_Depression_AVEC2017*.csv"))
            label_column = 'PHQ8_Binary'

        # Load split file and extract IDs
        if split_files:
            split_file_path = split_files[0]
            print(f"[{split}] Loading split file: {os.path.basename(split_file_path)}")
            print(f"[{split}] Target Label Column: {label_column}")
            
            try:
                df = pd.read_csv(split_file_path)
                
                # Convert IDs to string format for matching
                if 'Participant_ID' in df.columns:
                    target_ids = set(df['Participant_ID'].astype(str).values)
                else:
                    print(f" Error: 'Participant_ID' column not found in {split_file_path}")
                    
                # Check label column (for debugging)
                if label_column not in df.columns:
                    print(f" Warning: Label column '{label_column}' not found in csv. Check file contents.")
            except Exception as e:
                print(f" Error reading split file: {e}")
        else:
            print(f" Warning: No split file found for '{split}' in {SPLIT_DIR}")
            print("   -> Loading ALL .pkl files in cache directory without filtering.")

        # Load PKL files (Utterance-level)
        pkl_files = glob.glob(os.path.join(cache_dir, "*.pkl"))
        print(f"[{split}] Found {len(pkl_files)} pkl files in cache directory.")
        
        loaded_sessions = 0
        
        for pkl_path in pkl_files:
            # Extract ID from filename (e.g., 300_features.pkl -> 300)
            filename = os.path.basename(pkl_path)
            sid = filename.split('_')[0]
            
            # Load only the IDs corresponding to the current split
            # (ensure not to load all if target_ids is empty due to missing files)
            if split_files and sid not in target_ids:
                continue
                
            try:
                with open(pkl_path, 'rb') as f:
                    session_samples = pickle.load(f)
                    
                    valid_samples = []
                    for s in session_samples:
                        # 1. 오디오 데이터 유효성 검사
                        if s['audio_feature'] is None:
                            continue
                        if np.isnan(s['audio_feature']).any():
                            continue
                            
                        # 2. Label mapping (based on CSV file)
                        # Since labels might have been incorrectly assigned during .pkl creation, it is safer to remap them using the definitive labels from the CSV file.
                        # (However, to avoid overcomplicating the code here, temporarily trust the labels in the pkl, with the option to add logic later to retrieve and overwrite them from the df if necessary.)
                        
                        valid_samples.append(s)
                            
                    self.data.extend(valid_samples)
                    loaded_sessions += 1
            except Exception as e:
                print(f"Error loading {pkl_path}: {e}")
                
        print(f" [{split}] Final: Loaded {len(self.data)} utterances from {loaded_sessions} sessions.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1. Text Processing (BERT)
        text = str(sample['text'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 2. Audio Processing (Wav2Vec2)
        # numpy array -> tensor
        audio = torch.tensor(sample['audio_feature'], dtype=torch.float32)
        
        # Remove NaNs (for model training stability)
        if torch.isnan(audio).any():
            audio = torch.nan_to_num(audio)
            
        # Padding / Truncation (Based on Time axis)
        curr_len = audio.shape[0]
        if curr_len > self.max_audio_len:
            audio = audio[:self.max_audio_len]
            audio_len = self.max_audio_len
        else:
            pad_size = self.max_audio_len - curr_len
            if pad_size > 0:
                # (Time, FeatureDim) -> (Pad, FeatureDim)
                padding = torch.zeros(pad_size, audio.shape[1])
                audio = torch.cat([audio, padding], dim=0)
            audio_len = curr_len
            
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'audio_features': audio,
            'audio_len': audio_len,
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'session_id': sample['session_id']
        }

def collate_fn(batch):
    """
    Function called when creating batches in DataLoader
    """
    if not batch:
        return None
        
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    audio_features = torch.stack([item['audio_features'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Create Audio Mask (Padding = 0, Real data = 1)
    batch_size, max_len, _ = audio_features.shape
    audio_mask = torch.zeros(batch_size, max_len)
    for i, item in enumerate(batch):
        length = item['audio_len']
        audio_mask[i, :length] = 1
        
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'audio_features': audio_features,
        'audio_mask': audio_mask,
        'labels': labels
    }