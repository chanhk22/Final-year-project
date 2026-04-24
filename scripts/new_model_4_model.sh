# Track B: Thesis Narrative (Ensemble + Occlusion Sensitivity)

echo "Training 5-Member Session Ensemble (MLP)"

# Trains multiple MLP models using PCA-reduced session-level stats for robustness
python -m bert_wav2vec.train_session_ensemble


echo "Running Occlusion Sensitivity Analysis"

# Explainability Analysis via Occlusion Sensitivity with Temporal Grounding
python -m bert_wav2vec.explainability_occlusion