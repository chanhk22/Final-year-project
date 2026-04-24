# Window sampling in several window durations(2,4,6,8,10seconds)
python -m precache.window_sampling \
    --config configs/default.yaml \
    --window_durations 2 4 6 8 10 \
    --overlap 0.5 \
    --pca_audio 50 \
    --pca_visual 50 \
    --output_dir cache/temporal

# no pca (raw)
python -m precache.window_sampling \
    --config configs/default.yaml \
    --window_durations 2 4 6 8 10 \
    --overlap 0.5 \
    --pca_audio 0 \
    --pca_visual 0 \
    --output_dir cache/temporal_raw

# Specific second window only
# python -m precache.window_sampling --window_durations 2
