# Parkinson's Disease Progression Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-orange)](https://lightning.ai/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF)](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LSTM-based model to predict the progression of Parkinson's Disease from protein biomarkers and clinical data. Developed as a solution for the [AMP-PD Parkinson's Disease Progression Prediction](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction) Kaggle competition.

---

## Problem Statement

Parkinson's disease (PD) is a progressive neurological disorder. Tracking its severity over time requires repeated clinical assessments using the MDS-UPDRS scale — a standardized test that quantifies motor and non-motor symptoms across four sub-scores (UPDRS 1–4).

The goal is to **predict UPDRS scores at 0, 6, 12, and 24 months into the future** given a patient's visit history, using cerebrospinal fluid (CSF) protein and peptide abundance measurements as biomarkers.

---

## Dataset

Data provided by the [Accelerating Medicines Partnership® Parkinson's Disease (AMP® PD)](https://www.amp-pd.org/) program. **Download from Kaggle** before running the notebooks.

| File | Description | Rows |
|---|---|---|
| `train_clinical_data.csv` | UPDRS scores per patient visit | 2,615 |
| `supplemental_clinical_data.csv` | Additional clinical visits (no biomarker data) | ~2,600 |
| `train_proteins.csv` | CSF protein abundance (NPX) per visit | 232,741 |
| `train_peptides.csv` | CSF peptide abundance per visit | 981,834 |

**Key columns in clinical data:**

- `patient_id` — unique patient identifier
- `visit_month` — months since baseline visit
- `updrs_1/2/3/4` — MDS-UPDRS sub-scores (target variables)
- `upd23b_clinical_state_on_medication` — whether patient was on PD medication

---

## Methodology

### Pipeline

```
Raw CSF + Clinical Data
        │
        ▼
  Data Cleaning & Imputation (mean/mode fill)
        │
        ▼
  Sort by patient & visit month
        │
        ▼
  Sliding Window (window=4 visits → predict next 4)
        │
        ▼
  Z-score Normalization
        │
        ▼
  LSTM (input=8, hidden=128) → Linear → (4 scores × 4 time horizons)
        │
        ▼
  Predictions: UPDRS 1–4 at +0, +6, +12, +24 months
```

### Model Architecture

An LSTM followed by a fully-connected layer, implemented with PyTorch Lightning:

- **Input**: sequence of patient visits, each represented by 8 clinical features
- **LSTM**: 1 layer, hidden size 128
- **Output head**: Linear(128 → 16), reshaped to (4 scores, 4 time steps)
- **Loss**: Mean Squared Error (MSE)
- **Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **Optimizer**: Adam (lr=0.01)

---

## Project Structure

```
parkinsonPrediction/
├── notebooks/
│   ├── data_exploration.ipynb   # EDA: data shapes, patient distributions, UPDRS plots
│   ├── training.ipynb           # Model training and validation
│   └── submission.ipynb         # Kaggle submission pipeline
├── src/
│   ├── model.py                 # LSTMmodel (PyTorch Lightning)
│   ├── dataset.py               # PyTorch Dataset with sliding-window logic
│   ├── datamodule.py            # PyTorch Lightning DataModule
│   └── utils.py                 # Data loading, windowing, SMAPE, prediction helpers
├── data/
│   ├── train_clinical_data.csv
│   ├── supplemental_clinical_data.csv
│   ├── train_proteins.csv       # not tracked in git (>50 MB — download from Kaggle)
│   ├── train_peptides.csv       # not tracked in git (>50 MB — download from Kaggle)
│   └── example_test_files/      # sample test data from the competition
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/MarianodelRio/parkinsonPrediction.git
cd parkinsonPrediction
pip install -r requirements.txt
```

Download the competition data from Kaggle and place the CSV files inside `data/`:

```bash
pip install kaggle
kaggle competitions download -c amp-parkinsons-disease-progression-prediction
unzip amp-parkinsons-disease-progression-prediction.zip -d data/
```

---

## Usage

Run the notebooks in order from the `notebooks/` directory (or from the project root with Jupyter):

```bash
jupyter notebook notebooks/data_exploration.ipynb   # explore the data
jupyter notebook notebooks/training.ipynb            # train the LSTM
jupyter notebook notebooks/submission.ipynb          # generate Kaggle submission
```

---

## Results

| Metric | Value |
|---|---|
| Validation SMAPE | ~199 |

> **Note:** This is a baseline LSTM model trained on clinical scores only (protein/peptide data not yet incorporated). The score reflects room for improvement through feature engineering, multi-modal inputs, and architecture tuning.

---

## Competition

- **Platform**: Kaggle
- **Competition**: [AMP-PD Parkinson's Disease Progression Prediction](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction)
- **Evaluation metric**: SMAPE averaged across UPDRS scores and prediction time horizons
