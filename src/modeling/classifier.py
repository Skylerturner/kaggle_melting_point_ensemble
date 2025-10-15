import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import xgboost as xgb
from tqdm import tqdm


def prepare_data(df, features, target, test_size=0.2, val_size=0.1, random_state=42):
    """
    Prepare training, validation, and test splits with scaling.

    Args:
        df (pd.DataFrame): Dataset including features and target.
        features (list): List of feature column names.
        target (str): Target column name.
        test_size (float): Fraction of data for test set.
        val_size (float): Fraction of train+val for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled (np.ndarray)
        y_train, y_val, y_test (np.ndarray)
        scaler (StandardScaler): fitted scaler on training data
    """
    X = df[features].values
    y = df[target].values

    # Split train+val and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split train and val sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


class MolDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target labels.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def plot_classification_report(y_test, y_pred, class_names=None):
    """
    Prints the classification report and displays the confusion matrix for classification results.

    Args:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list, optional): List of class names for display. Defaults to None.
    """
    print(classification_report(y_test, y_pred, target_names=class_names))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron classifier implemented in PyTorch.

    Args:
        input_dim (int): Number of input features.
        num_classes (int, optional): Number of output classes. Defaults to 3.
        hidden_layers (tuple, optional): Hidden layer sizes. Defaults to (128, 512, 512, 128).
        activation (str, optional): Activation function to use. Defaults to 'silu'.
        dropout (float, optional): Dropout rate after each hidden layer. Defaults to 0.3.
    """

    def __init__(self, input_dim, num_classes=3, hidden_layers=(128, 512, 512, 128),
                 activation='silu', dropout=0.3):
        super(MLPClassifier, self).__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.input_dim = input_dim
        self.dropout = dropout

        layers = []
        last_dim = input_dim

        # Build hidden layers with activation and dropout
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim

        # Output layer (no activation or dropout)
        layers.append(nn.Linear(last_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def _get_activation(self, name):
        if name == 'silu':
            return nn.SiLU()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'softplus':
            return nn.Softplus()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x):
        return self.model(x)


def train_mlp_classifier(model, train_loader, val_loader, epochs=500, lr=1e-3, patience=50):
    """
    Train a PyTorch MLPClassifier model with early stopping based on validation loss.

    Args:
        model (nn.Module): MLP model instance.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int, optional): Max number of epochs. Defaults to 500.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        patience (int, optional): Number of epochs with no improvement to stop early. Defaults to 50.

    Returns:
        nn.Module: Trained model with best validation weights loaded.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)
    return model

def train_multiclass_classifier(df, features, target_col='Tm_bin', param_grid=None):
    """
    Train a single XGBoost multi-class classifier on all bins.

    Args:
        df (pd.DataFrame): DataFrame with features and target column.
        features (list): List of selected feature column names.
        target_col (str): Name of the target column with bin labels.
        param_grid (dict): Grid search parameters.

    Returns:
        dict: Model, scaler, accuracy, predictions, etc.
    """
    X = df[features].values
    y = df[target_col].values

    # Split data into train, val, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define model
    xgb_clf = xgb.XGBClassifier(
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42
    )

    fit_params = {"eval_set": [(X_val_scaled, y_val)], "verbose": False}

    if param_grid:
        grid_search = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            verbose=1,
            n_jobs=1
        )
        grid_search.fit(X_train_scaled, y_train, **fit_params)
        best_model = grid_search.best_estimator_
        print(f"Best hyperparameters: {grid_search.best_params_}")
    else:
        best_model = xgb_clf
        best_model.fit(X_train_scaled, y_train, **fit_params)

    # Evaluation
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return {
        'model': best_model,
        'scaler': scaler,
        'accuracy': acc,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred,
    }



def train_ensemble_weights(mlp_probs, xgb_probs, y_true):
    """
    Optimize per-class ensemble weights for combining MLP and XGB classifier probabilities.

    Args:
        mlp_probs (np.ndarray): MLP classifier probabilities (N x C).
        xgb_probs (np.ndarray): XGB classifier probabilities (N x C).
        y_true (np.ndarray): True class labels (N,).

    Returns:
        np.ndarray: Optimized weights per class (array of length C).
    """
    num_classes = mlp_probs.shape[1]
    weights = np.ones(num_classes) * 0.5  # start with equal weights
    
    def loss_fn(w):
        ensemble_probs = np.zeros_like(mlp_probs)
        for c in range(num_classes):
            ensemble_probs[:, c] = w[c] * mlp_probs[:, c] + (1 - w[c]) * xgb_probs[:, c]
        ensemble_probs = np.clip(ensemble_probs, 1e-9, 1 - 1e-9)
        nll = -np.mean(np.log(ensemble_probs[np.arange(len(y_true)), y_true]))
        return nll

    bounds = [(0, 1)] * num_classes
    res = minimize(loss_fn, weights, bounds=bounds)
    if res.success:
        return res.x
    else:
        print("Optimization failed, returning default weights")
        return weights


def ensemble_probabilities(mlp_probs, xgb_probs, weights):
    """
    Compute weighted ensemble probabilities per class combining MLP and XGB outputs.

    Args:
        mlp_probs (np.ndarray): MLP model probabilities (N x C).
        xgb_probs (np.ndarray): XGB model probabilities (N x C).
        weights (dict): Dictionary of weights per class (keys like "alpha_extra_0").

    Returns:
        np.ndarray: Ensemble combined probabilities (N x C).
    """
    ensemble_probs = np.zeros_like(mlp_probs)
    for c in range(mlp_probs.shape[1]):
        alpha = weights[f"alpha_extra_{c}"]
        ensemble_probs[:, c] = alpha * mlp_probs[:, c] + (1 - alpha) * xgb_probs[:, c]
    return ensemble_probs

