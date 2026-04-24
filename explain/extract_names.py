import os
import glob
import json
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Any number existing patient ID is fine.
SESSION_ID = "300" 

def main():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "default.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    PROC_ROOT = str(project_root / config['outputs']['processed_root'])
    print("Extracting feature names...")
    audio_names = []
    
    # 1. Audio Features (COVAREP, FORMANT)
    covarep_path = os.path.join(PROC_ROOT, "Features", "covarep", f"{SESSION_ID}_COVAREP.csv")
    if os.path.exists(covarep_path):
        df = pd.read_csv(covarep_path, nrows=1) 
        audio_names.extend([f"COVAREP_{c}" for c in df.columns])
        
    formant_path = os.path.join(PROC_ROOT, "Features", "formant", f"{SESSION_ID}_FORMANT.csv")
    if os.path.exists(formant_path):
        df = pd.read_csv(formant_path, nrows=1)
        audio_names.extend([f"FORMANT_{c}" for c in df.columns])

    # 2. Visual Features (CLNF)
    visual_names = []
    clnf_types = ["clnf_au", "clnf_feature", "clnf_feature3d", "clnf_gaze", "clnf_pose"]
    
    for clnf in clnf_types:
        pattern = os.path.join(PROC_ROOT, "Features", clnf, f"{SESSION_ID}_*.csv")
        files = glob.glob(pattern)
        if files:
            df = pd.read_csv(files[0], nrows=1)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove timestamp column (Same logic with window_sampling.py)
            numeric_cols = [c for c in numeric_cols if 'timestamp' not in c.lower()]
            
            visual_names.extend([f"{clnf}_{c}" for c in numeric_cols])

    print(f" Audio features count: {len(audio_names)} (Expected 79)")
    print(f" Visual features count: {len(visual_names)} (Expected 393)")
    
    output_json = project_root / "feature_names.json"
    with open(output_json, "w") as f:
        json.dump({"audio": audio_names, "visual": visual_names}, f, indent=2)
    print(f" Saved to {output_json}!")

if __name__ == "__main__":
    main()