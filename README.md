# House Prices — Advanced Regression Techniques

A production-style, leak-free machine learning pipeline for the Kaggle House Prices competition — predicting residential home sale prices in Ames, Iowa from 79 explanatory features.

---

## Overview

This project transforms a typical Kaggle notebook into a production-ready machine learning pipeline. It ensures that all preprocessing steps are reproducible, leak-free, and reusable on new data.

---

## Data

This project is based on the House Prices dataset from Kaggle.

Due to Kaggle's access restrictions and authentication requirements, the original dataset (`train.csv` and `test.csv`) is not included in this repository. Accessing these files requires a Kaggle account and login.

Since this repository is intended as a portfolio project, a small sample dataset (`sample_data.csv`) is provided inside the `data/raw/` directory. This allows the project to run without requiring external downloads.

### How to use the full dataset

To reproduce full results:

1. Download the dataset from Kaggle
2. Place `train.csv` and `test.csv` inside the `data/raw/` folder

The project is designed to:

* Use the sample dataset by default
* Automatically work with the full dataset if available

This ensures:

* Reproducibility
* Compliance with Kaggle policies
* Easy testing for recruiters

---

## Project structure

house-prices-regression/
├── data/
│   ├── raw/               # sample_data.csv OR Kaggle train/test
│   └── processed/
├── notebooks/
├── src/
├── tests/
├── models/
├── outputs/
├── requirements.txt
└── LICENSE

---

## Setup

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Train the model:

```bash
python -m src.train
```

Generate predictions:

```bash
python -m src.predict
```

Run tests:

```bash
pytest
```

---

## Methodology

* Clean, leak-free preprocessing pipeline
* Feature engineering for domain-specific signals
* Proper encoding (ordinal + one-hot)
* Skew correction
* Model training with cross-validation
* Final evaluation on unseen holdout data

---

## Testing

Tests are implemented using synthetic data to ensure the pipeline works even without the original Kaggle dataset.

---

## Author

YasamanOwji

---
## License 
MIT

MIT
