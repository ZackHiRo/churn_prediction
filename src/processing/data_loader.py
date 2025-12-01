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
    
    # Convert target to numeric if it contains string values (e.g., "Yes"/"No" -> 1/0)
    if y.dtype == 'object' or y.dtype.name == 'category':
        # Map common binary string values to 0/1
        unique_vals = y.unique()
        if len(unique_vals) == 2:
            # Sort to ensure consistent mapping (first value -> 0, second -> 1)
            sorted_vals = sorted(unique_vals)
            y = y.map({sorted_vals[0]: 0, sorted_vals[1]: 1})
        else:
            raise ValueError(
                f"Target column '{target_column}' has {len(unique_vals)} unique values. "
                f"Expected 2 for binary classification. Values: {unique_vals}"
            )
    
    X = df.drop(columns=[target_col])
    return X, y


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


