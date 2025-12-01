from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    # Case-insensitive column matching
    target_col = None
    for col in df.columns:
        if col.lower() == target_column.lower():
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(
            f"Target column '{target_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


