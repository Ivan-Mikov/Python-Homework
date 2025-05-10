import numpy as np
from typing import Callable


class ShapeMismatchError(Exception):
    pass


def euclidean_dist(points: np.ndarray, x_train: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, np.newaxis, :] - x_train, axis=2)


def kernel(x: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x)
    result[np.abs(x) <= 1] = 0.75 * (1 - x[np.abs(x) <= 1] ** 2)
    return result


class KNearestNeighbors:

    _x_train: np.ndarray  # points
    _y_train: np.ndarray  # labels
    _n_neighbors: int
    _calc_distanses: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __init__(
        self,
        n_neighbors: int = 20,
        calc_distances: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_dist,
    ):
       self._n_neighbors = n_neighbors
       self._calc_distances = calc_distances

    def fit(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray,
    ):
        self._x_train = x_train
        self._y_train = y_train

    def predict(self, x_test: np.ndarray):
        distances = self._calc_distances(x_test, self._x_train)  # Находим Р

        neighbors_indices = np.argsort(distances, axis=1)[:, :self._n_neighbors]

        neighbors_labels = self._y_train[neighbors_indices]

        unique_labels = np.unique(self._y_train)

        weighted_votes = np.zeros((len(x_test), len(unique_labels)))

        for i, label in enumerate(unique_labels):
            weighted_votes[:, i] = np.sum(neighbors_labels == label, axis=1)

        y_test = unique_labels[np.argmax(weighted_votes, axis=1)]

        return y_test


class WeightedKNearestNeighbors:

    _x_train: np.ndarray  # points
    _y_train: np.ndarray  # labels
    _n_neighbors: int
    _calc_distanses: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def __init__(
        self, 
        n_neighbors: int = 20, 
        calc_distances: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_dist,
    ):
       self._n_neighbors = n_neighbors
       self._calc_distances = calc_distances

    def fit(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray,
    ):
        self._x_train = x_train
        self._y_train = y_train

    def predict(self, x_test: np.ndarray):
        distances = self._calc_distances(x_test, self._x_train)  # Находим Р
        neighbors_distanses = np.sort(distances, axis=-1)[:, :self._n_neighbors]  # Оставляем соседей

        neighbors_indices = np.argsort(distances, axis=1)[:, :self._n_neighbors]

        neighbors_labels = self._y_train[neighbors_indices]

        h = np.take_along_axis(distances, neighbors_indices[:, -1:], axis=1)

        weights = kernel(neighbors_distanses / h)  # Находим W
        unique_labels = np.unique(self._y_train)

        weighted_votes = np.zeros((len(x_test), len(unique_labels)))

        for i, label in enumerate(unique_labels):
            weighted_votes[:, i] = np.sum((neighbors_labels == label) * weights, axis=1)

        y_test = unique_labels[np.argmax(weighted_votes, axis=1)]

        return y_test


def accuracy(
    true_targets: np.ndarray, 
    prediction: np.ndarray, 
) -> float:
    if (true_targets.shape[0] != prediction.shape[0]):
        raise ShapeMismatchError("Shapes must be equal")
    
    return np.mean(true_targets == prediction)
