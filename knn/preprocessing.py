from typing import Callable, Any
import numpy as np


class ShapeMismatchError(Exception):
    pass


def get_boxplot_outliers(
    data: np.ndarray,
    key: Callable[[Any], Any] = None
) -> np.ndarray:
    sorted_data = np.sort(data, axis=0) if (key is None) else key(data)
    size = sorted_data.shape[0]
    q1 = sorted_data[int(np.round(size * 0.25)), :]
    q3 = sorted_data[int(np.round(size * 0.75)), :]
    eps = (q3 - q1) * 1.5
    mask = np.any((data < (q1 - eps)) | (data > (q3 + eps)), axis=1)
    return np.where(mask)[0]


def train_test_split(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if features.shape[0] != targets.shape[0]:
        raise ShapeMismatchError("Shapes must be equal")

    if (train_ratio > 1 or train_ratio < 0):
        raise ValueError("train_ratio must be bigger that 0 and less than 1")

    union_data = np.column_stack((features, targets))
    if shuffle:
        np.random.shuffle(union_data)
        features = union_data[:, :2]
        targets = union_data[:, 2]

    train_mask = []
    test_mask = []

    for unique_target in np.unique(targets):
        targer_indices = np.where(targets == unique_target)[0]
        count = len(targer_indices)

        size = int(count * train_ratio)

        train_mask.extend(targer_indices[:size])
        test_mask.extend(targer_indices[size:])

    train_ind = np.array(train_mask, dtype=int)
    test_ind = np.array(test_mask, dtype=int)

    train_points = features[train_ind]
    train_labels = targets[train_ind]
    test_points = features[test_ind]
    test_labels = targets[test_ind]

    return train_points, train_labels, test_points, test_labels
