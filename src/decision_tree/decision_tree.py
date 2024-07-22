from typing import Self

import numpy as np


class Node:
    value: int
    left: Self | None
    right: Self | None

    def __init__(self, value: int) -> None:
        self.value = value
        self.left = None
        self.right = None


class DecisionTree:
    root: Node

    def __init__(self) -> None:
        self.root = Node(1)

    def fit(self, values: np.ndarray, lables: np.ndarray, binary_entropy_mode: bool) -> None:
        if binary_entropy_mode:
            self._fit_binary_entropy(values, lables)
        else:
            self._fit_brute_force(values, lables)

    def _fit_brute_force(self, values: np.ndarray, lables: np.ndarray) -> None:
        pass

    def _fit_binary_entropy(self, values: np.ndarray, lables: np.ndarray) -> None:
        pass

    def predict(self, values: np.ndarray) -> np.ndarray | np.int64:
        return values[0]
