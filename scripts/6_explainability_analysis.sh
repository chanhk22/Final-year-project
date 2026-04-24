# 1. (Required for the first run) Execute the feature name extraction script first!
# (If the feature_names.json file has already been created, you may comment out this line)
# echo "[Step 1] Extracting original feature names..."
# python -m explain.extract_names

# Choose other cache path if you want to see other explainability result with different window size
python -m explain.explainability \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --pca_dir cache/temporal/win6.0s \
    --checkpoint checkpoints_improved/exp_6s/best.pt \
    --output_dir explainability_best/thesis_shap \
    --num_layers 4

# python -m explain.plot_pca_variance_curve