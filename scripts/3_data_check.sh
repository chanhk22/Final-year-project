python -m precache.dataset_temporal \
    --cache_path cache/temporal/win2.0s/DAIC-WOZ_win2.0s_cache.pkl \
    --use_pca \
    --batch_size 16

python -m precache.dataset_temporal \
    --cache_path cache/temporal/win4.0s/DAIC-WOZ_win4.0s_cache.pkl \
    --use_pca \
    --batch_size 16

python -m precache.dataset_temporal \
    --cache_path cache/temporal/win6.0s/DAIC-WOZ_win6.0s_cache.pkl \
    --use_pca \
    --batch_size 16

python -m precache.dataset_temporal \
    --cache_path cache/temporal/win8.0s/DAIC-WOZ_win8.0s_cache.pkl \
    --use_pca \
    --batch_size 16

python -m precache.dataset_temporal \
    --cache_path cache/temporal/win10.0s/DAIC-WOZ_win10.0s_cache.pkl \
    --use_pca \
    --batch_size 16