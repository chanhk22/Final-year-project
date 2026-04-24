"""
Batch execution script for DAIC-WOZ audio trimming.
This script automates the removal of the setup phase and virtual agent instructions, extracting solely the clinically relevant interview portions for all sessions.
Refer to [Gratch et al., 2014] for the DAIC-WOZ interview structure.
"""
import os
import yaml
import glob
import json
from pathlib import Path
from preprocessing.daic_audio_pipeline import find_ellie_start, trim_wav_from_start

def trim_audio():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "default.yaml"
    
    with open(config_path, encoding='utf-8') as f:
        C = yaml.safe_load(f)
        
    AUDIO_IN = C['paths']['daic_woz']['audio_dir']
    TRS_IN    = C['paths']['daic_woz']['transcript_dir']
    PROC_ROOT = C['outputs']['processed_root']
    ELLIE_RGX = C['preprocessing']['ellie_regex']
    
    os.makedirs(os.path.join(PROC_ROOT, "Audio"), exist_ok=True)
    
    t0_values = {}  # Dictionary saving t0 values
    
    # Transcript files
    for trs in sorted(glob.glob(os.path.join(TRS_IN, '*_TRANSCRIPT.csv'))):
        sid = os.path.basename(trs).split('_')[0]
        wav_in = os.path.join(AUDIO_IN, f"{sid}_AUDIO.wav")
        wav_out = os.path.join(PROC_ROOT, "Audio", f"{sid}_AUDIO_trimmed.wav")
        
        # Check whether trimmed audio files exist
        if os.path.exists(wav_out):
            print(f"Trimmed audio already exists for session {sid}: {wav_out}")
            continue # If file exists, skip trimming 
        
        if not os.path.exists(wav_in):
            print(f"Audio file does not exist: {wav_in}")
            continue
        # Ellie start time
        try:
            t0 = find_ellie_start(trs, ELLIE_RGX)
            print(f"Found Ellie start time: {t0} for session {sid}")
            t0_values[sid] = t0  # save t0 values in dictionary
        except Exception as e:
            print(f"Error finding Ellie start time for session {sid}: {e}")
            continue
        
        # Audio trimming
        try:
            trim_wav_from_start(wav_in, wav_out, t0)
            print(f"Successfully trimmed audio for session {sid}")
        except Exception as e:
            print(f"Error trimming audio for session {sid}: {e}")
            continue
        
    # Save t0 values as JSON file
    t0_path = project_root / 't0_values.json'
    with open(t0_path, 'w') as f:
        json.dump(t0_values, f)
        
    print("DAIC head cut t0 filter done.")
    
if __name__ == "__main__":
    trim_audio()