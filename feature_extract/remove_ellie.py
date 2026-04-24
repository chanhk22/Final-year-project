"""
Feature filtering module to isolate participant responses from the DAIC-WOZ dataset. 
This script aligns the extracted multimodal features (Visual and Acoustic) with the participant's actual speech segments.
"""
import os
import yaml
import glob
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def load_participant_segments(transcript_path, t0_start=0.0):
    '''
    Extract a list of time intervals where the Participant speaks from a TRANSCRIPT file.
    Only utterances after t0_start
    
    Args:
        transcript_path: Path to the TRANSCRIPT CSV file
        t0_start: Start time in seconds - Utterances before this time are ignored
    Returns:
        List of tuples: [(start_time, stop_time), ...]
    '''
    try: 
        df = pd.read_csv(transcript_path, delimiter='\t')
        
        # filtering 'Participant' row only
        participant_df = df[df['speaker'] == 'Participant'].copy()
        
        # start_time, stop_time to float
        participant_df['start_time'] = participant_df['start_time'].astype(float)
        participant_df['stop_time'] = participant_df['stop_time'].astype(float)
        
        # Utterances after t0
        participant_df = participant_df[participant_df['start_time'] >= t0_start]
        
        # (start_time, stop_time) tuple list
        segments = list(zip(participant_df['start_time'], participant_df['stop_time']))
        return segments 
    
    except Exception as e:
        print(f"Error loading transcript {transcript_path}: {e}")
        return []
    
def filter_timestamp_feature(feature_path, segments, output_path):
    """ 
    Filtering feature files which have timestamp columns.
    Primarily used for OpenFace (CLNF) facial landmark and Action Unit features.
    [Tool Reference] T. Baltrušaitis, P. Robinson, and L.-P. Morency, "OpenFace: 
    an open source facial behavior analysis toolkit," in 2016 IEEE Winter 
    Conference on Applications of Computer Vision (WACV), 2016, pp. 1-10.
    """
    try:
        df = pd.read_csv(feature_path, sep=',', skipinitialspace=True)
        
        # Find timestamp column
        timestamp_col = None 
        for col in df.columns:
            if 'timestamp' in col.lower():
                timestamp_col = col
                break 
            
        if timestamp_col is None:
            print(f"Warning: No timestamp column found in {feature_path}")
            return
        
        # Timestamp column to float type
        try:
            df[timestamp_col] = df[timestamp_col].astype(float)
        except Exception:
            df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
            df = df.dropna(subset=[timestamp_col])
            
        # Collect rows of each segment
        filtered_segments = []
        for start, stop in segments:
            mask = (df[timestamp_col] >= start) & (df[timestamp_col] <= stop)
            segment_df = df.loc[mask]
            if not segment_df.empty:
                filtered_segments.append(segment_df)
                
        # Concatenate all segments in one
        if filtered_segments:
            result_df = pd.concat(filtered_segments, ignore_index=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save in CSV
            result_df.to_csv(output_path, index=False)
            print(f" Saved: {output_path} ({len(result_df)} rows)")
        else:
            print(f"Warning: No data found for {feature_path}")
            
    except Exception as e:
        print(f"Error processing {feature_path}: {e}")
        
def filter_row_based_feature(feature_path, segments, output_path, row_duration=0.01):
    """
    Filtering feature files based on row indices calculated from timestamps.
    Primarily used for COVAREP and FORMANT acoustic features. The default 
    row_duration of 0.01s matches the standard 10ms frame shift used in COVAREP.
    
    [Tool Reference] G. Degottex et al., "COVAREP—A collaborative voice analysis 
    repository for speech technologies," in 2014 IEEE International Conference 
    on Acoustics, Speech and Signal Processing (ICASSP), 2014, pp. 960-964.
    """
    try:
        df = pd.read_csv(feature_path)
        filtered_segments = []
        for start, stop in segments:
            # Calculate row index range
            start_row = int(start/row_duration)
            stop_row = int(stop/row_duration) + 1
            
            # Check range and slicing 
            if start_row < len(df):
                end_row = min(stop_row, len(df))
                segment_df = df.iloc[start_row:end_row]
                if not segment_df.empty:
                    filtered_segments.append(segment_df)
                    
        if filtered_segments:
            result_df = pd.concat(filtered_segments, ignore_index=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path} ({len(result_df)} rows)")
        else:
            print(f"Warning: No data found for {feature_path}")
            
    except Exception as e:
        print(f"Error processing {feature_path}: {e}")
        
def filter_features():
    """
    Filtering all feature files by section
    """    
    # config file load
    with open("configs/default.yaml", encoding='utf-8') as f:
        C = yaml.safe_load(f)
        
    FEAT_IN = C['paths']['daic_woz']['features_dir']
    PROC_ROOT = C['outputs']['processed_root']
    TRS_IN = C['paths']['daic_woz']['transcript_dir']
                                    
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent          
    t0_file = project_root / "t0_values.json"

    print(f"(debug) script_path: {script_path}")
    print(f"(debug) project_root: {project_root}")
    print(f"(debug) looking for t0_file at: {t0_file}")
    
    t0_dict = {}

    if t0_file.exists():
        try:
            with t0_file.open('r', encoding='utf-8') as f:
                raw = json.load(f)

            # Normalize keys to strings and values to floats where possible
            for k, v in raw.items():
                try:
                    t0_dict[str(k)] = float(v)
                except Exception:
                    print(f"Warning: ignored non-numeric t0 for key {k!r}: {v!r}")

            if t0_dict:
                print(f"Loaded {len(t0_dict)} t0 entries from {t0_file}")
                print("Sample t0 entries:", list(t0_dict.items())[:10])
            else:
                print(f"{t0_file} parsed but contains no usable numeric entries.")
        except Exception as e:
            print(f"Error reading {t0_file}: {e}")
            t0_dict = {}
    else:
        print(f"t0 file not found at {t0_file}. Using t0=0.0 for all sessions.")

    # Feature settings
    timestamp_features = [
        "clnf_au",
        "clnf_feature",
        "clnf_feature3d",
        "clnf_gaze",
        "clnf_pose"
    ]
    row_based_features = [
        "covarep",
        "formant"
    ]

    # Transcript files 
    transcript_files = sorted(glob.glob(os.path.join(TRS_IN, "*_TRANSCRIPT.csv")))

    print(f"Found {len(transcript_files)} transcript files")
    print("=" * 60)

    
    for transcript_path in tqdm(transcript_files, desc="Processing sessions"):
        sid = os.path.basename(transcript_path).split('_')[0]
        print(f"\n[Session {sid}]")

        # Bring t0
        t0 = t0_dict.get(str(sid),
                         t0_dict.get(f"{sid}_P",
                                    t0_dict.get(f"{sid}_participant", 0.0)))
        print(f"  (debug) Using t0 for session {sid!s}: {t0}")

        # load participant utterance segment 
        segments = load_participant_segments(transcript_path, t0_start=t0)

        if not segments:
            print(f" No participant segments found (after t0={t0}), skipping session {sid}")
            continue

        print(f"  Found {len(segments)} participant segments (after t0={t0})")

        # 1. Timestamp based
        for feature_subdir in timestamp_features:
            feature_dir = os.path.join(FEAT_IN, feature_subdir)

            if not os.path.exists(feature_dir):
                print(f" Feature directory not found: {feature_dir}")
                continue

            feature_files = glob.glob(os.path.join(feature_dir, f"{sid}_*.txt"))

            for feature_path in feature_files:
                filename = os.path.basename(feature_path)
                output_path = os.path.join(
                    PROC_ROOT, "Features",
                    feature_subdir, filename.replace('.txt', '.csv')
                )

                if os.path.exists(output_path):
                    print(f" Already exists: {filename}")
                    continue

                filter_timestamp_feature(feature_path, segments, output_path)

        # 2. Row based
        for feature_subdir in row_based_features:
            feature_dir = os.path.join(FEAT_IN, feature_subdir)

            if not os.path.exists(feature_dir):
                print(f" Feature directory not found: {feature_dir}")
                continue

            feature_files = glob.glob(os.path.join(feature_dir, f"{sid}_*.csv"))

            for feature_path in feature_files:
                filename = os.path.basename(feature_path)
                output_path = os.path.join(
                    PROC_ROOT, "Features",
                    feature_subdir, filename
                )

                if os.path.exists(output_path):
                    print(f" Already exists: {filename}")
                    continue

                filter_row_based_feature(feature_path, segments, output_path)

    print("\n" + "=" * 60)
    print(" Feature filtering completed!")


if __name__ == "__main__":
    filter_features()