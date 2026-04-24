# Executes daic_audio_trim and remove_ellie modules
set -e

python -m feature_extract.daic_audio_trim

python -m feature_extract.remove_ellie

# For comparison between removing ellie and keeping ellie (or use run_uncleaned_experiment.sh)
# python -m feature_extract.keep_ellie