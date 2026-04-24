# Window-Based Multimodal Depression Detection on DAIC-WOZ

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Ready-orange)
![License](https://img.shields.io/badge/License-Academic_Use_Only-red)

This repository contains the complete codebase and implementation details for the paper:  
**"Window-Based Multimodal Depression Detection Screening on DAIC-WOZ"** *BSc Computer Science Final Year Project, UCL (2026)*


---

## Table of Contents
- [Quick Start](#quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Run Baseline Experiments](#3-run-baseline-experiments-6s-window-pca)
  - [4. Run Foundation Model Experiments](#4-run-foundation-model-experiments-advanced)
  - [5. Statistical Validation Experiments](#5-statistical-validation-experiments)
  - [6. Expected Results](#6-expected-results)
- [Full Documentation](#full-documentation)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv my_env
source my_env/bin/activate  # Linux/macOS
# my_env\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

###  CRITICAL: Script Execution Rule

All shell scripts (`.sh`) **MUST** be executed from the root directory of the project. Do not navigate into the `scripts/` directory to run them.

**Correct:**
```bash
bash scripts/1_preprocess_audio.sh
```

**Incorrect:**
```bash
cd scripts && bash 1_preprocess_audio.sh  # This will fail!
```

### 2. Data Preparation

Due to licensing restrictions, the raw DAIC-WOZ dataset is not included in this repository. Before running any scripts, you must:

1. **Request access** to DAIC-WOZ from USC ICT [here](https://dcapswoz.ict.usc.edu)
2. **Download and extract** the dataset
3. **Place the data** in the `data_raw/` directory at the project root

The directory structure must **strictly** follow this format for all relative paths to work correctly:

```
project_root/
├── data_raw/
│   ├── audio/                  # Raw participant audio files
│   │   ├── 300_AUDIO.wav
│   │   ├── 301_AUDIO.wav
│   │   └── ... (192 files total)
│   ├── Features/               # Pre-extracted OpenFace and COVAREP features
│   │   ├── clnf_au/           # Action Unit intensities
│   │   ├── clnf_feature/      # 2D facial landmarks
│   │   ├── clnf_feature3d/    # 3D facial landmarks
│   │   ├── clnf_gaze/         # Gaze direction vectors
│   │   ├── clnf_pose/         # Head pose (rotation + translation)
│   │   ├── covarep/           # Acoustic features (74 dims)
│   │   └── formant/           # Formant frequencies (F1-F5)
│   ├── labels/                 # Official AVEC2017 splits
│   │   ├── dev_split_Depression_AVEC2017.csv
│   │   ├── full_test_split.csv
│   │   └── train_split_Depression_AVEC2017.csv
│   └── transcript/             # Interview transcripts
│       ├── 300_TRANSCRIPT.csv
│       ├── 301_TRANSCRIPT.csv
│       └── ...
├── ... /                     # bert_wav2vec, dataset, explain etc.
├── configs/
│   └── default.yaml           # Uses relative paths from project root
├── scripts/
│   ├── 1_preprocess_audio.sh
│   └── ...
├── PHQcorrect.ipynb           # Label consistency correction
└── README.md
```

#### Label Consistency Correction

To reproduce the label correction process (fixing Participant 409's inconsistent PHQ-8 label), run the Jupyter Notebook located at the project root:

```bash
jupyter notebook PHQcorrect.ipynb
```


### 3. Run Baseline Experiments (6s Window, PCA)

```bash
# Step 1: Label correction
jupyter notebook PHQcorrect.ipynb

# Step 2~6: Data preprocessing + Training
bash scripts/1_preprocess_audio.sh
bash scripts/2_build_sampling.sh
bash scripts/3_data_check.sh
bash scripts/4_model_architecture.sh
bash scripts/4_model_architecture2.sh
bash scripts/5_dataloader.sh
bash scripts/6_explainability_analysis.sh
```

**Remember:** All scripts must be executed from the project root!


### 4. Run Foundation Model Experiments (Advanced)

```bash
# Step 1: Extract Wav2Vec 2.0 features
bash scripts/new_model_1_wav2vec.sh

# Step 2: Initialize and train foundation framework (BERT + Audio)
bash scripts/new_model_2_model.sh

# Step 3: Session-level ensemble training and evaluation
bash scripts/new_model_3_model.sh

# Step 4: Occlusion sensitivity analysis for explainability
bash scripts/new_model_4_model.sh
```

### 5. Statistical Validation Experiments

```bash
# Data cleaning ablation (with/without interviewer removal)
bash scripts/run_uncleaned_experiment.sh

# Bootstrap confidence intervals (95% CI for F1 and AUC)
bash scripts/run_bootstrap_ci.sh
```

### 6. Expected Results

**Baseline performance (6s PCA):**
- **F1 Score**: 0.4667 (95% CI: [0.25, 0.68])
- **AUC-ROC**: 0.7100 (95% CI: [0.52, 0.85])

**Foundation Model ensemble:**
- **Recall**: 0.93 (for depressed class)
- **AUC-ROC**: 0.72

See **Table 5.2, 5.9** in the dissertation for full results.

---

## Full Documentation

For detailed step-by-step instructions, configuration parameters, and troubleshooting:  
**See Appendix D (System and User Manual)** in the dissertation PDF.

Appendix D covers:

- **D.1**: Environment Setup
- **D.2**: Directory Structure and Data Preparation
- **D.3**: Execution Pipeline I (Baseline, 6 steps)
  - D.3.1: Label Consistency Correction
  - D.3.2: Audio Sanitisation and Interviewer Removal
  - D.3.3: Feature Alignment and Window Sampling
  - D.3.4: Data Integrity Verification
  - D.3.5: Model Training and Ablation
  - D.3.6: Explainability Analysis (SHAP and Inverse PCA)
- **D.4**: Execution Pipeline II (Foundation Models, 4 steps)
  - D.4.1: Foundation Feature Extraction (Wav2Vec 2.0)
  - D.4.2: Foundation Model Initialisation (BERT)
  - D.4.3: Session-Level Ensemble Training and Evaluation
  - D.4.4: Explainability Analysis (Occlusion Sensitivity)
- **D.5**: Statistical Validation and Ablation Experiments
  - D.5.1: Data Cleaning Ablation (Interviewer Noise)
  - D.5.2: Bootstrap Confidence Intervals

---

## Reproducibility

- **Random seed**: 42 (all experiments)
- **Hardware**: 4× NVIDIA Quadro RTX 6000 GPUs and 375GB RAM
- **Paths**: All scripts use relative paths from project root
- **Execution**: All commands assume you are in the project root directory

All results are reproducible following the instructions in **Appendix D**.

---

## License

This project is for academic use only. DAIC-WOZ dataset usage subject to USC ICT terms.
