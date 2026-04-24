# Multimodal Embedding Extraction for Hierarchical Modeling
python -m bert_wav2vec.extract_embeddings


# Track A: Advanced NLP Approach (Hierarchical Attention Network)

echo "Training Hierarchical Attention Network (HAN)"

# Session-Level Temporal Attention Network for Depression Detection
python -m bert_wav2vec.train_session

echo "Evaluating HAN & Extracting Attention Weights"

# Clinical Evaluation and Attention-based Interpretability

python -m bert_wav2vec.eval_session