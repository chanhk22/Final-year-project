"""
Uncleaned multimodal feature extraction module for the DAIC-WOZ dataset.
Unlike the cleaned pipeline that isolates participant-only speech, this script 
retains the continuous interaction (including the virtual agent 'Ellie') after 
the initial preamble (t0). This approach is designed to capture the raw 
conversational dynamics and serves as an uncleaned baseline for model evaluation.

[Dataset Reference] J. Gratch et al., "The Distress Analysis Interview Corpus of 
human and computer interviews," in Proceedings of LREC'14, 2014, pp. 3123-3128.
"""

import os
import yaml
import glob
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def load_uncleaned_segments(transcript_path, t0_start=0.0):
    """ 
    Extracts a single continuous time segment from t0 to the end of the interview.
    By identifying the global minimum start time and maximum stop time across all 
    speakers, this function deliberately includes both the virtual interviewer and 
    the participant to preserve the uninterrupted acoustic and visual context.
    """
    try:
        df = pd.read_csv(transcript_path, delimiter='\t')
        
        # start_time, stop_time to float
        df['start_time'] = df['start_time'].astype(float)
        df['stop_time'] = df['stop_time'].astype(float)
        
        df = df[df['start_time'] >= t0_start]
        
        if df.empty:
            return []
        
        first_start = df['start_time'].min()
        last_stop = df['stop_time'].max()
        
        segments = [(first_start, last_stop)]
        
        return segments 
    
    except Exception as e:
        print(f"Error loading transcript {transcript_path}: {e}")
        return []
    
# same functions with remove_ellie.py
def filter_timestamp_feature(feature_path, segments, output_path):
    """ 
    Filters timestamp-based features (e.g., facial landmarks, AUs).
    
    [Tool Reference] T. Baltrušaitis, P. Robinson, and L.-P. Morency, "OpenFace: 
    an open source facial behavior analysis toolkit," in WACV, 2016, pp. 1-10.
    """
    try:
        df = pd.read_csv(feature_path, sep=',', skipinitialspace=True)
        timestamp_col = None
        for col in df.columns:
            if 'timestamp' in col.lower():
                timestamp_col = col
                break
        if timestamp_col is None:
            return
        try:
            df[timestamp_col] = df[timestamp_col].astype(float)
        except:
            df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
            df = df.dropna(subset=[timestamp_col])

        filtered_segments = []
        for start, stop in segments:
            mask = (df[timestamp_col] >= start) & (df[timestamp_col] <= stop)
            segment_df = df.loc[mask]
            if not segment_df.empty:
                filtered_segments.append(segment_df)

        if filtered_segments:
            result_df = pd.concat(filtered_segments, ignore_index=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error processing {feature_path}: {e}")

def filter_row_based_feature(feature_path, segments, output_path, row_duration=0.01):
    """ 
    Filters row-based features based on timestamps, assuming a default 0.01s frame shift.
    
    [Tool Reference] G. Degottex et al., "COVAREP—A collaborative voice analysis 
    repository for speech technologies," in ICASSP, 2014, pp. 960-964.
    """
    try:
        df = pd.read_csv(feature_path)
        filtered_segments = []
        for start, stop in segments:
            start_row = int(start / row_duration)
            stop_row = int(stop / row_duration) + 1
            if start_row < len(df):
                end_row = min(stop_row, len(df))
                segment_df = df.iloc[start_row:end_row]
                if not segment_df.empty:
                    filtered_segments.append(segment_df)

        if filtered_segments:
            result_df = pd.concat(filtered_segments, ignore_index=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error processing {feature_path}: {e}")

def filter_features():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    config_path = project_root / "configs" / "default.yaml"
    
    with open(config_path, encoding='utf-8') as f:
        C = yaml.safe_load(f)

    FEAT_IN = C['paths']['daic_woz']['features_dir']
    
    # Save the result in new root
    PROC_ROOT = C['outputs']['processed_root'] + "_uncleaned"
    
    TRS_IN = C['paths']['daic_woz']['transcript_dir']

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    t0_file = project_root / "t0_values.json"

    t0_dict = {}
    if t0_file.exists():
        try:
            with t0_file.open('r', encoding='utf-8') as f:
                raw = json.load(f)
            for k, v in raw.items():
                try:
                    t0_dict[str(k)] = float(v)
                except:
                    pass
        except:
            t0_dict = {}

    timestamp_features = ["clnf_au", "clnf_feature", "clnf_feature3d", "clnf_gaze", "clnf_pose"]
    row_based_features = ["covarep", "formant"]

    transcript_files = sorted(glob.glob(os.path.join(TRS_IN, "*_TRANSCRIPT.csv")))

    for transcript_path in tqdm(transcript_files, desc="Processing Uncleaned sessions"):
        sid = os.path.basename(transcript_path).split('_')[0]
        
        t0 = t0_dict.get(str(sid), t0_dict.get(f"{sid}_P", t0_dict.get(f"{sid}_participant", 0.0)))

        segments = load_uncleaned_segments(transcript_path, t0_start=t0)

        if not segments:
            continue

        for feature_subdir in timestamp_features:
            feature_dir = os.path.join(FEAT_IN, feature_subdir)
            if not os.path.exists(feature_dir): continue
            feature_files = glob.glob(os.path.join(feature_dir, f"{sid}_*.txt"))
            for feature_path in feature_files:
                filename = os.path.basename(feature_path)
                output_path = os.path.join(PROC_ROOT, "Features", feature_subdir, filename.replace('.txt', '.csv'))
                if os.path.exists(output_path): continue
                filter_timestamp_feature(feature_path, segments, output_path)

        for feature_subdir in row_based_features:
            feature_dir = os.path.join(FEAT_IN, feature_subdir)
            if not os.path.exists(feature_dir): continue
            feature_files = glob.glob(os.path.join(feature_dir, f"{sid}_*.csv"))
            for feature_path in feature_files:
                filename = os.path.basename(feature_path)
                output_path = os.path.join(PROC_ROOT, "Features", feature_subdir, filename)
                if os.path.exists(output_path): continue
                filter_row_based_feature(feature_path, segments, output_path)

if __name__ == "__main__":
    filter_features()