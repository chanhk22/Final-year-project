python -m models.model

# PCA pkl files experiment

# Experiment 1: 2-second windows (Baseline)
python -m models.train \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --experiment_name exp_2s \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --device cuda


# Experiment : add  fusion regularization
python -m models.train \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --experiment_name exp_2s_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --device cuda

# Experiment : fusion regularization + Top-K aggregation
python -m models.train \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --experiment_name exp_2s_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3 \
    --device cuda
# Experiment 2: 4-second windows

python -m models.train \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --experiment_name exp_4s \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30


# Experiment : add fusion regularization
python -m models.train \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --experiment_name exp_4s_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --device cuda

# Experiment : fusion regularization + Top-K aggregation
python -m models.train \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --experiment_name exp_4s_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3 \
    --device cuda
# Experiment 3: 6-second windows

python -m models.train \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30

# Experiment : add fusion regularization
python -m models.train \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --device cuda

# Experiment : fusion regularization + Top-K aggregation
python -m models.train \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3 \
    --device cuda
# Experiment 4: 8-second windows

python -m models.train \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --experiment_name exp_8s \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30

# Experiment : add fusion regularization
python -m models.train \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --experiment_name exp_8s_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --device cuda

# Experiment : fusion regularization + Top-K aggregation
python -m models.train \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --experiment_name exp_8s_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3 \
    --device cuda

# Experiment 5: 10-second windows
python -m models.train \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --experiment_name exp_10s \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30


# Experiment : add fusion regularization
python -m models.train \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --experiment_name exp_10s_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --device cuda

# Experiment : fusion regularization + Top-K aggregation
python -m models.train \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --experiment_name exp_10s_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 2 \
    --batch_size 32 \
    --lr 1e-4 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3 \
    --device cuda

# Larger model (2s windows)
python -m models.train \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --experiment_name exp_2s_large \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30


# Larger model with Fusion Regularization
python -m models.train \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --experiment_name exp_2s_large_fusion \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Larger model with Fusion Regularization + Top-K Aggregation
python -m models.train \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --experiment_name exp_2s_large_all_features \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3

# Larger model (4s windows)
python -m models.train \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --experiment_name exp_4s_large \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30

# Larger model with modality Fusion Regularization
python -m models.train \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --experiment_name exp_4s_large_fusion \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Larger model with Modality Fusion Regularization + Top-K Aggregation
python -m models.train \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --experiment_name exp_4s_large_all_features \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3

# Larger model (6s windows)
python -m models.train \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s_large \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30

# Larger model with modality Fusion Regularization
python -m models.train \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s_large_fusion \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Larger model with Modality Fusion Regularization + Top-K Aggregation
python -m models.train \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s_large_all_features \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3

# Larger model (8s windows)
python -m models.train \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --experiment_name exp_8s_large \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30

# Larger model with modality Fusion Regularization
python -m models.train \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --experiment_name exp_8s_large_fusion \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Larger model with Modality Fusion Regularization + Top-K Aggregation
python -m models.train \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --experiment_name exp_8s_large_all_features \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3

# Larger model (10s windows)
python -m models.train \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --experiment_name exp_10s_large \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30

# Larger model with modality Fusion Regularization
python -m models.train \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --experiment_name exp_10s_large_fusion \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Larger model with Modality Fusion Regularization + Top-K Aggregation
python -m models.train \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --experiment_name exp_10s_large_all_features \
    --d_model 256 \
    --nhead 8 \
    --num_layers 3 \
    --dim_feedforward 1024 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3

# Deep model (2s windows)
python -m models.train \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --experiment_name exp_2s_deep \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30


# Deep model modality Fusion regularization
python -m models.train \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --experiment_name exp_2s_deep_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Deep model modality fusion regularization + top-k aggregation
python -m models.train \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --experiment_name exp_2s_deep_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3

# Deep model (4s windows)
python -m models.train \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --experiment_name exp_4s_deep \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30


# Deep model modality Fusion regularization
python -m models.train \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --experiment_name exp_4s_deep_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Deep model fusion regularization + top-k aggregation
python -m models.train \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --experiment_name exp_4s_deep_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3

# Deep model (6s windows)
python -m models.train \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s_deep \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30


# Deep model Fusion regularization
python -m models.train \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s_deep_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Deep model fusion regularization + top-k aggregation
python -m models.train \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --experiment_name exp_6s_deep_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3

# Deep model (8s windows)
python -m models.train \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --experiment_name exp_8s_deep \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30

# Deep model Fusion regularization
python -m models.train \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --experiment_name exp_8s_deep_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Deep model fusion regularization + top-k aggregation
python -m models.train \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --experiment_name exp_8s_deep_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3

# Deep model (10s windows)
python -m models.train \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --experiment_name exp_10s_deep \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30

# Deep model Fusion regularization
python -m models.train \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --experiment_name exp_10s_deep_fusion \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1

# Deep model fusion regularization + top-k aggregation
python -m models.train \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --experiment_name exp_10s_deep_all_features \
    --d_model 128 \
    --nhead 4 \
    --num_layers 4 \
    --batch_size 32 \
    --lr 1e-4 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 30 \
    --fusion_entropy_weight 0.1 \
    --use_topk \
    --topk_ratio 0.3