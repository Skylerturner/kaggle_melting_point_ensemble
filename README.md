# kaggle_melting_point_ensemble

This project tackles the Kaggle competition focused on predicting the melting points of organic compounds based on their molecular descriptors.

## Overview

The goal is to accurately predict melting points for a dataset of organic compounds. To improve performance, this project uses an ensemble approach:

- **Classification step:** Compounds are first classified into bins (low, medium, high melting point categories) using an ensemble classifier.
- **Regression step:** Separate XGBoost regression models (XGBRegressors) are trained on each bin to predict melting points within that range.

This bin-wise approach leverages specialized regressors tailored to the distinct melting point ranges, improving overall accuracy.

## Features

- Custom binning of melting points for better regression modeling
- Per-bin hyperparameter tuning using `GridSearchCV`
- Final training and evaluation of XGBRegressor models for each bin
- Utilities for prediction by bin and visualization of results
- GPU-accelerated XGBoost training for efficiency

## Comparison of Notebooks with Different Dataset Sizes

Two notebooks were developed to evaluate how dataset size impacts model performance:

- **Competition Dataset Notebook:**  Trains models on a given sample of data (2662 compounds) provided by the competition. This setup ensures full confidence in the results with no risk of data leakage. While this approach provides robust and reliable evaluation, it may exhibit slightly reduced predictive performance due to the smaller dataset size.
- **Expanded Dataset Notebook:**  Incorporates an external, larger dataset of compounds for training and evaluation (19183 compounds). This expanded dataset typically leads to improved model accuracy but requires more computational resources and longer training times. Additionally, it demands careful handling to avoid data leakage and maintain model integrity.

Comparing results between these notebooks highlights the trade-off between dataset size, training time, and model accuracy, helping guide practical modeling decisions.

## Requirements

- Python 3.7+
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `joblib`

Install dependencies via:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib joblib


