#!/usr/bin/env bash
# This script is for showing the result when we keep ellie's speech data

set -e

echo "1. Keep Ellie Feature Extraction"
# Run the previously created Python script to generate the processed_uncleaned folder.
python -m feature_extract.keep_ellie

echo "2. Uncleaned Window Sampling"
# You can either temporarily modify processed_root in configs/default.yaml to processed_uncleaned, or specify a different output folder (output_dir) as shown below.
python -m precache.window_sampling \
    --config configs/default.yaml \
    --window_durations 6 \
    --overlap 0.5 \
    --pca_audio 50 \
    --pca_visual 50 \
    --output_dir cache/temporal_uncleaned

echo "3. Training Uncleaned Baseline Model (6s PCA)"
# Train the model using the newly generated uncleaned cache files
python -m models.train \
    --cache_path cache/temporal_uncleaned/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s_uncleaned_ablation \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --device cuda