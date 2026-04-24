python -m models.train\
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl\
    --experiment_name exp_6s\
    --save_dir checkpoints_improved\
    --d_model 128\
    --nhead 4\
    --num_layers 2\
    --batch_size 32\
    --lr 1e-4\
    --max_epochs 100\
    --patience 30\
    --device cuda


# Statistical Validation: Bootstrap Confidence Intervals (95% CI)
python -m dataset.bootstrap_ci --csv_path checkpoints_improved/exp_6s/test_predictions.csv