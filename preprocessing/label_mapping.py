import os, glob, pandas as pd, numpy as np

# The threshold of 10.0 for PHQ-8 binary classification is based on established clinical standards for detecting moderate to severe depressive symptoms:
# [Reference] K. Kroenke et al., "The PHQ-8 as a measure of current depression in the general population," Journal of Affective Disorders, vol. 114, no. 1-3, pp. 163-173, 2009.

# default threshold for PHQ (PHQ-8): commonly 10 for moderate depression
DEFAULT_PHQ_THRESHOLD = 10.0

def _read_all_csvs(labels_dir):
    rows = []
    for p in glob.glob(os.path.join(labels_dir, '*.csv')):
        try:
            df = pd.read_csv(p)
            df['__source_file'] = os.path.basename(p)
            rows.append(df)
        except Exception as e:
            print(f"[label mapping] failed to read {p}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)

def canonicalize_column_names(df, dataset_hint=None):
    #lowercase keys
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    # find participant id column
    pid = None 
    pid_candidates = ['participant_id','participant','id', 'index', 'session']

    for cand in pid_candidates:
        if cand in cols:
            pid = cols[cand]; break 
            
    # find PHQ score/binary
    phq_score = None 
    phq_score_candidates = [
        'phq8_score', 'phq_score', 'phq8', 'phq_total', 'phq',
        'phq-8_score', 'phq-8', 'depression_score'
    ]
    for cand in phq_score_candidates:
        if cand in cols:
            phq_score = cols[cand]
            break 
        
    # find PHQ binary column (multiple patterns)
    phq_binary = None 
    phq_binary_candidates = [
        'phq8_binary', 'phq_binary',  
        'phq-8_binary', 'depression_binary',
        'depression', 'depressed'
    ]
    return {
        'pid': pid, 
        'phq_score': phq_score, 
        'phq_binary': phq_binary, 
    }

def _process_binary_label(value, dataset_hint=None):
    '''Convert various binary label formats to 0/1'''
    if pd.isna(value):
        return None 
    
    value_str = str(value).lower().strip()
    # Standard binary formats
    if value_str in ['1', '1.0', 'depression', 'depressed']:
        return 1
    elif value_str in ['0', '0.0', 'normal', 'control']:
        return 0
    else:
        try:
            # Try to convert to number
            num_val = float(value)
            return 1 if num_val > 0.5 else 0
        except (ValueError, TypeError):
            return None