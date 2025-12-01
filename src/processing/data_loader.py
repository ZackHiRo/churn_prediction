from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


