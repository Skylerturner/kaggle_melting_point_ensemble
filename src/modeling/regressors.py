import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

def plot_regression_predictions(y_test, y_pred, bin_label):
    """
    Create a scatter plot comparing true vs. predicted regression targets.

    Args:
        y_test (array-like): True target values.
        y_pred (array-like): Predicted target values.
        bin_label (int or str): Label of the bin or group to annotate the plot.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Melting Point')
    plt.ylabel('Predicted Melting Point')
    plt.title(f'Bin {bin_label} Predictions')
    plt.grid(True)
    plt.show()


def grid_search_per_bin(df, features, param_grids, bin_col='Tm_bin', target_col='Tm'):
    """
    Run GridSearchCV separately for each bin using its own param grid.
    param_grids should be a dict mapping bin_label -> param_grid
    """
    all_results = {}
    for bin_label, grid in param_grids.items():
        print(f"\n Running GridSearch for bin {bin_label}")
        df_bin = df[df[bin_col] == bin_label].reset_index(drop=True)
        results = per_bin_grid_search(df_bin, features, target_col=target_col, bin_col=bin_col, param_grid=grid)
        all_results[bin_label] = results[bin_label]
    return all_results


def per_bin_grid_search(df, features, target_col='Tm', bin_col='Tm_bin', param_grid=None):
    """
    Train XGBRegressor models for each bin in bin_col.
    Perform GridSearchCV on each bin subset independently.

    Returns a dictionary of results including best model, scaler, and regression metrics.
    """

    results = {}

    for bin_label in sorted(df[bin_col].unique()):
        print(f"\n--- Hyperparameter tuning and training regressor for bin {bin_label} ---")

        bin_df = df[df[bin_col] == bin_label].reset_index(drop=True)

        X = bin_df[features].values
        y = bin_df[target_col].values

        # Split (train/val/test) within bin
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1, random_state=42
        )

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


def per_bin_final_training(
    df, features, target_bin, 
    target_col='Tm', bin_col='Tm_bin', 
    xgb_params=None,
    save_path="saved_models"
):
    """
    Train XGB regressor for a specific bin with given hyperparameters.
    Saves model and scaler, returns evaluation metrics and predictions.

    Args:
        df (pd.DataFrame): dataset
        features (list): feature column names
        target_bin (int): bin label to filter dataset
        target_col (str): target column name (default 'Tm')
        bin_col (str): bin column name (default 'Tm_bin')
        xgb_params (dict): hyperparameters for XGBRegressor
        save_path (str): directory to save models and scalers

    Returns:
        dict: containing model, scaler, metrics, test data and predictions
    """

    os.makedirs(save_path, exist_ok=True)
    results = {}
    bin_label = target_bin
    print(f"\n--- Training regressor for bin {bin_label} ---")

    bin_df = df[df[bin_col] == bin_label].reset_index(drop=True)
    X = bin_df[features].values
    y = bin_df[target_col].values

    # Split within bin (train/val/test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Set default parameters if none provided
    default_params = {
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': 1,
        'colsample_bytree': 1.0,
        'gamma': 0.8,
        'learning_rate': 0.02,
        'max_depth': 4,
        'n_estimators': 600,
        'subsample': 0.8
    }

    if xgb_params is not None:
        default_params.update(xgb_params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Bin {bin_label} MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    # Save model and scaler
    model_filename = os.path.join(save_path, f"xgb_bin{bin_label}.joblib")
    scaler_filename = os.path.join(save_path, f"scaler_bin{bin_label}.joblib")
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Saved model to {model_filename}")
    print(f"Saved scaler to {scaler_filename}")

    results[bin_label] = {
        'model': model,
        'scaler': scaler,
        'mae': mae,
        'rmse': rmse,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred
    }

    return results

def predict_by_bin(bin_preds, X_test, bin_scalers, regressors):
    """
    Predict target values using bin-specific regressors.

    Args:
        bin_preds (np.ndarray): Predicted bins for each sample.
        X_test (np.ndarray): Original unscaled descriptor features.
        bin_scalers (dict): Scalers per bin index.
        regressors (dict): Trained XGBRegressor models per bin.

    Returns:
        np.ndarray: Final predictions in correct order.
    """
    y_pred = np.zeros(len(X_test))

    for b in bin_scalers:
        idxs = np.where(bin_preds == b)[0]
        if len(idxs) == 0:
            continue

        X_bin_unscaled = X_test[idxs]
        X_bin_scaled = bin_scalers[b].transform(X_bin_unscaled)
        y_bin_pred = regressors[b].predict(X_bin_scaled)

        y_pred[idxs] = y_bin_pred

    return y_pred
