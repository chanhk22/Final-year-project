"""
Deep Acoustic Feature Extraction code using Wav2Vec 2.0 Foundation Model.
[Transfer Learning via Self-Supervised Models]
Traditional acoustic features (e.g., COVAREP, MFCCs) often fail to capture the deep contextual and phonetic nuances of speech. 
This code leverages a pre-trained Wav2Vec 2.0 model to extract high-dimensional, context-aware acoustic representations. 
The base model ('facebook/wav2vec2-base-960h') was pretrained on 960 hours of unlabeled speech, making it highly robust for downstream paralinguistic tasks 
like depression detection.

[Architecture Reference] A. Baevski et al., "wav2vec 2.0: A framework for self-supervised learning of speech representations," 
in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[Implementation Reference] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," in EMNLP, 2020. (Hugging Face)
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
import torchaudio
import yaml 
from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import pickle
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
with open(PROJECT_ROOT / "configs" / "default.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    
# 1. Trimmed Audio folder (daic_audio_trim.py result)
AUDIO_DIR = str(PROJECT_ROOT / config['outputs']['processed_root'] / "Audio")

# 2. Raw transcript folder 
TRANSCRIPT_DIR = str(PROJECT_ROOT / config['paths']['daic_woz']['transcript_dir'])

# 3.Label file path (Train/Dev/Test Split CSV files)
LABEL_DIR = str(PROJECT_ROOT / config['paths']['daic_woz']['labels_dir'])

T0_JSON_PATH = str(PROJECT_ROOT / "t0_values.json")
# 4. Result save folder
OUTPUT_DIR = str(PROJECT_ROOT / "cache" / "foundation_features")

# Wav2Vec2 model setting
MODEL_NAME = "facebook/wav2vec2-base-960h"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_labels_robust(label_dir):
    """
    Read all split CSV files and merge them into a single DataFrame.
    Standardize the column names by renaming PHQ8_Binary and PHQ_Binary to 'label'.
    """
    split_files = glob.glob(os.path.join(label_dir, "*split*.csv"))
    if not split_files:
        print(" Label files not found!")
        return None

    df_list = []
    print(f" Found {len(split_files)} label files:")
    
    for f in split_files:
        print(f"   - {os.path.basename(f)}")
        df = pd.read_csv(f)
        
        # Normalize column names (PHQ8_Binary, PHQ_Binary -> label)
        if 'PHQ8_Binary' in df.columns:
            df = df.rename(columns={'PHQ8_Binary': 'label'})
        elif 'PHQ_Binary' in df.columns:
            df = df.rename(columns={'PHQ_Binary': 'label'})
            
        # Retain only the required columns.
        if 'Participant_ID' in df.columns and 'label' in df.columns:
            df = df[['Participant_ID', 'label']]
            df_list.append(df)
        else:
            print(f" Warning: Skipped {os.path.basename(f)} (Missing ID or Label column)")

    if not df_list:
        return None
        
    full_df = pd.concat(df_list).drop_duplicates(subset='Participant_ID').set_index('Participant_ID')
    print(f" Combined Labels: {len(full_df)} participants loaded.")
    return full_df

def load_wav2vec_model():
    print(f" Loading Wav2Vec2 model: {MODEL_NAME}...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model

def extract_features(waveform, processor, model):
    """
    Wav2Vec 2.0 is strictly pretrained on 16kHz audio. To prevent catastrophic 
    feature distortion, the input waveform is resampled (if necessary) to exactly 
    16,000 Hz before passing it through the processor. The output is the 
    last_hidden_state, capturing the deepest level of acoustic abstraction.
    """
    with torch.no_grad():
        # Assume that 16k resampling has already been performed during loading.
        input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
        input_values = input_values.to(device)
        outputs = model(input_values)
        # (Batch, Time, 768) -> (Time, 768)
        return outputs.last_hidden_state.squeeze(0).cpu().numpy()
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load label 
    labels_df = load_labels_robust(LABEL_DIR)
    if labels_df is None:
        return

    # 2. Load t0 value
    if not os.path.exists(T0_JSON_PATH):
        print(f" t0_values.json not found at {T0_JSON_PATH}")
        return
    with open(T0_JSON_PATH, 'r') as f:
        t0_values = json.load(f)

    # 3. Load model
    processor, model = load_wav2vec_model()

    # 4. list of audio files
    audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*_trimmed.wav")))
    print(f" Found {len(audio_files)} trimmed audio files.")

    for wav_path in tqdm(audio_files):
        try:
            filename = os.path.basename(wav_path)
            sid_str = filename.split('_')[0]
            sid = int(sid_str)
            
            save_path = os.path.join(OUTPUT_DIR, f"{sid}_features.pkl")
            # if os.path.exists(save_path): continue

            # Transcript 
            trans_candidates = glob.glob(os.path.join(TRANSCRIPT_DIR, f"**/{sid}_TRANSCRIPT.csv"), recursive=True)
            if not trans_candidates:
                print(f" Transcript not found for {sid}")
                continue
            trans_path = trans_candidates[0]

            # t0
            if sid_str not in t0_values:
                print(f" t0 missing for {sid}")
                continue
            t0 = float(t0_values[sid_str])
            
            # Load transcript 
            df_trans = pd.read_csv(trans_path, sep='\t')
            df_part = df_trans[df_trans['speaker'] == 'Participant'].copy()
            
            # Load audio 
            waveform, sr = torchaudio.load(wav_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            session_data = []
            
            # Bring label (If missing, -1)
            label = labels_df.loc[sid, 'label'] if sid in labels_df.index else -1
            if label == -1:
                print(f" Label missing for {sid}, skipping.")
                continue

            for idx, row in df_part.iterrows():
                start_time = float(row['start_time']) - t0
                end_time = float(row['stop_time']) - t0
                text = str(row['value'])
                
                if start_time < 0 or (end_time - start_time) < 0.5:
                    continue
                
                start_sample = int(start_time * 16000)
                end_sample = int(end_time * 16000)
                
                if start_sample >= waveform.shape[1]: continue
                end_sample = min(end_sample, waveform.shape[1])
                
                sub_wave = waveform[:, start_sample:end_sample]
                
                try:
                    w2v_feat = extract_features(sub_wave, processor, model)
                    session_data.append({
                        "session_id": sid,
                        "utterance_idx": idx,
                        "text": text,
                        "audio_feature": w2v_feat,
                        "label": label,
                        "start_time": start_time,  
                        "stop_time": end_time,     
                        "duration": end_time - start_time
                    })
                except:
                    pass

            if session_data:
                with open(save_path, 'wb') as f:
                    pickle.dump(session_data, f)
                    
        except Exception as e:
            print(f" Error processing {wav_path}: {e}")

    print("\n All Preprocessing Done!")

if __name__ == "__main__":
    main()