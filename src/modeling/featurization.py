import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def get_rdkit_descriptors(exclude: set = None) -> list:
    """
    Returns a list of (name, function) tuples for RDKit molecular descriptors,
    excluding any descriptors in the provided exclude set.

    Args:
        exclude (set): Set of descriptor names to exclude.

    Returns:
        List[Tuple[str, Callable]]: List of descriptor name-function pairs.
    """
    all_descriptor_funcs = Descriptors._descList
    exclude = exclude or {'Ipc', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v'}

    descriptor_funcs = [
        (name, func)
        for name, func in all_descriptor_funcs
        if name not in exclude
    ]
    return descriptor_funcs

def featurize_smiles_dataframe(
    df: pd.DataFrame,
    smiles_column: str,
    descriptor_funcs: list
) -> tuple[pd.DataFrame, list]:
    """
    Featurize a DataFrame of SMILES strings using RDKit descriptor functions.

    Args:
        df (pd.DataFrame): Input DataFrame with a SMILES column.
        smiles_column (str): Name of the column containing SMILES strings.
        descriptor_funcs (list): List of (name, function) tuples for descriptors.

    Returns:
        Tuple[pd.DataFrame, list]: 
            - DataFrame of computed descriptor values
            - List of valid row indices (those that were successfully processed)
    """
    descriptor_names = [name for name, _ in descriptor_funcs]
    descriptor_data = []
    valid_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        mol = Chem.MolFromSmiles(row[smiles_column])
        if mol is None:
            continue
        try:
            desc_values = [func(mol) for _, func in descriptor_funcs]
        except Exception:
            continue
        if np.any(pd.isna(desc_values)) or np.any(np.isinf(desc_values)):
            continue
        descriptor_data.append(desc_values)
        valid_indices.append(idx)

    X_desc = np.array(descriptor_data)
    feature_df = pd.DataFrame(X_desc, columns=descriptor_names)
    return feature_df, valid_indices

def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.95
) -> pd.DataFrame:
    """
    Compute feature importances using a Random Forest classifier and select top features 
    based on a cumulative importance threshold.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        feature_names (list[str]): List of feature names corresponding to columns in X.
        threshold (float, optional): Cumulative importance threshold for feature selection. 
            Features are selected until their cumulative importance exceeds this value.
            Defaults to 0.95.

    Returns:
        pd.DataFrame: DataFrame with columns ['Feature', 'Importance', 'Cumulative'] sorted 
            by importance in descending order, containing only top features under the threshold.
    """
    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)

    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    importance_df['Cumulative'] = importance_df['Importance'].cumsum()

    top_features = importance_df[importance_df['Cumulative'] <= threshold]
    return top_features