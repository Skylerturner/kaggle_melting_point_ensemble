import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from typing import Dict, List, Optional, Any


def plot_regression_predictions(y_true: np.ndarray, y_pred: np.ndarray, bin_label: int | str) -> None:
    """
    Scatter plot comparing true vs. predicted regression targets.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        bin_label (int | str): Label of the bin or group to annotate the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Melting Point')
    plt.ylabel('Predicted Melting Point')
    plt.title(f'Bin {bin_label} Predictions')
    plt.grid(True)
    plt.show()


def per_bin_grid_search(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'Tm',
    bin_col: str = 'Tm_bin',
    param_grid: Optional[Dict[str, List[Any]]] = None
) -> Dict[int, Dict]:
    """
    Train and tune an XGBRegressor model for each bin in the dataset.

    Args:
        df (pd.DataFrame): Dataset filtered for all bins.
        features (List[str]): Feature columns.
        target_col (str): Target column.
        bin_col (str): Bin column.
        param_grid (Optional[Dict]): Parameter grid for GridSearchCV.

    Returns:
        Dict[int, Dict]: Dictionary mapping bin label to model, scaler, metrics, and predictions.
    """
    results = {}

    for bin_label in sorted(df[bin_col].unique()):
        print(f"\n--- Training regressor with hyperparameter tuning for bin {bin_label} ---")
        bin_df = df[df[bin_col] == bin_label].reset_index(drop=True)
        X = bin_df[features].values
        y = bin_df[target_col].values

        # Split into train, validation, and test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        xgb_reg = xgb.XGBRegressor(
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=0,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1
        )

        fit_params = {"eval_set": [(X_val_scaled, y_val)], "verbose": False}

        grid_search = GridSearchCV(
            estimator=xgb_reg,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_absolute_error',
            verbose=1,
            n_jobs=1
        )

        grid_search.fit(X_train_scaled, y_train, **fit_params)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"Bin {bin_label} best params: {grid_search.best_params_}")
        print(f"Bin {bin_label} MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        results[bin_label] = {
            'model': best_model,
            'scaler': scaler,
            'mae': mae,
            'rmse': rmse,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
        }

    return results

