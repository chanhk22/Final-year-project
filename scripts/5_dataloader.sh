# PCA features (Choose cache path to check others)
python -m dataset.dataset --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl

# Raw features (Choose cache path to check others)
python -m dataset.dataset --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl --no_pca

# Control batch size (Choose cache path to check others)
python -m dataset.dataset --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl --batch_size 64

# Shows exact figures of parameters
python -m dataset.dimension_check