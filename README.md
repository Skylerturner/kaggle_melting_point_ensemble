# Melting Point Prediction via Ensemble Modeling

This project addresses the **Kaggle melting point prediction challenge**, where the goal is to predict the melting points of organic compounds from molecular descriptors.

## Project Overview

To improve accuracy, this solution uses a **hybrid classification-regression pipeline**:

- **1. Classification Step:**  
  Compounds are first categorized into melting point bins (e.g., low, medium, high) using an **ensemble classifier** that combines the outputs of a Multilayer Perceptron (MLP) and an XGBoost classifier.

- **2. Regression Step:**  
  Based on the predicted bin, a **bin-specific XGBoost regressor** is applied to predict the actual melting point within that range.  
  Each bin has its own model, trained and fine-tuned independently for optimal performance.

This two-stage pipeline reduces the solution space for regression and captures heterogeneity across the melting point spectrum.

## Features

- **Ensemble Classifier**: Optimized with per-class weights to balance class accuracy  
- **Feature Selection**: Based on Random Forest feature importance  
- **Bin-Specific XGBoost Regressors**: Each bin has independently tuned models via `GridSearchCV`
- **Grid Search Fine-Tuning**: Grid Search methods are used and explained for fine-tuning hyperparameters in every model   
- **Custom Utilities**: For feature scaling, plotting, bin assignment, and result evaluation  
- **Final Submission Generation**: Clean CSV output of predictions for competition submission

## Utility Modules

Core modeling utilities are modularized in the `src/modeling/` directory for reuse and clarity:

- `featurization.py` – Feature extraction, preprocessing, and scaling utilities  
- `classifier.py` – Training functions and evaluation tools for the MLP and XGBoost classifiers  
- `regressors.py` – Bin-wise regression utilities including hyperparameter tuning and plotting

These modules support the end-to-end pipeline and can be easily extended or modified for future experimentation.
Each function has a description of expected inputs and what it outputs.
Example from classifier.py:

```python
def train_ensemble_weights(mlp_probs, xgb_probs, y_true):
    """
    Optimize per-class ensemble weights for combining MLP and XGB classifier probabilities,
    using macro-average per-class accuracy.

    Args:
        mlp_probs (np.ndarray): MLP classifier probabilities (N x C).
        xgb_probs (np.ndarray): XGB classifier probabilities (N x C).
        y_true (np.ndarray): True class labels (N,).

    Returns:
        np.ndarray: Optimized weights per class (array of length C).
        float: Best per-class accuracy.
        np.ndarray: Best prediction array.
    """
    alphas = np.linspace(0, 1, 11)
    best_score = 0
    best_weights = None
    best_preds = None

    for alpha_0 in alphas:
        for alpha_1 in alphas:
            for alpha_2 in alphas:
                weights = np.array([alpha_0, alpha_1, alpha_2])
                ensemble_probs = weights * mlp_probs + (1 - weights) * xgb_probs
                y_pred = np.argmax(ensemble_probs, axis=1)

                # --- Compute per-class accuracy ---
                cm = confusion_matrix(y_true, y_pred)
                with np.errstate(divide='ignore', invalid='ignore'):
                    per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
                    per_class_acc = np.nan_to_num(per_class_acc)  # replaces NaNs with 0
                avg_per_class_acc = per_class_acc.mean()

                if avg_per_class_acc > best_score:
                    best_score = avg_per_class_acc
                    best_weights = weights
                    best_preds = y_pred

    return best_weights, best_score, best_preds
```

## Final Model Performance

- **Test MAE** (Expanded Dataset on Competition Test Set): **~20.58 K**
- Outperforms many published XGBoost baselines trained on similar-sized datasets
- Demonstrates the benefit of ensemble classification + per-bin regression

## Future Work

Planned improvements include:

- **Physics-Informed Feature Selection**: Using descriptor knowledge or functional group mappings  
- **Descriptor-Based Binning**: Instead of empirical binning, use SMILES or molecular similarity  
- **Hierarchical Classification**: E.g., classify by chemical class before predicting melting point  
- **Embedding-Based Clustering**: Use NLP/graph embeddings for learned bin structure

## Requirements ( Unsure if these are the true versions I'm using)

- Python 3.7+
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `joblib`
- `torch` *(for MLP)*

Install with:

```bash
pip install -r requirements.txt



