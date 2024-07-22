from typing import Self
from datetime import datetime

import numpy as np


class Node:
    value: int
    left: Self | None
    right: Self | None

    def __init__(self) -> None:
        self.value = 0
        self.left = None
        self.right = None

    def __str__(self) -> str:
        if self.left is None or self.right is None:
            return str(self.value)

        left_value = str(self.left)
        right_value = str(self.right)

        l = len(left_value.split('\n')[0])

        ret = (' ' * l) + ' ' + str(self.value) + ' ' + (' ' * l)
        for line1, line2 in zip(left_value.split('\n'), right_value.split('\n')):
            ret += '\n ' + line1 + ' ' + line2 + ' '

        return ret


class DecisionTree:
    layers: int
    root: Node

    def __init__(self, layers: int) -> None:
        self.layers = layers
        self.root = Node()

    def fit(self, values: np.ndarray, lables: np.ndarray, binary_entropy_mode: bool) -> None:
        if binary_entropy_mode:
            self._fit_binary_entropy(values, lables)
        else:
            self._fit_brute_force(values, lables)

    def _fit_brute_force(self, values: np.ndarray, lables: np.ndarray) -> None:
        self._create_tree(self.root, 0)

        indexs_len = (2 ** (self.layers + 1)) - 1
        dim = values.shape[1]

        index = [0] * indexs_len
        nodes = _create_nodes_list(self.root)
        max_index = [dim - 1] * ((2 ** self.layers) - 1)
        max_index += [1] * (2 ** self.layers)

        best_index = index.copy()
        best_goods = self._goods(values, lables)

        print("best_goods", best_goods)

        loops = 1
        for item in max_index:
            loops *= item + 1

        last_print = datetime.now()
        loop = 0
        while loop < loops:
            # change index
            current = len(max_index) - 1
            while index[current] == max_index[current]:
                index[current] = 0
                nodes[current].value = index[current]
                current -= 1
                if current == -1:
                    break

            index[current] += 1
            nodes[current].value = index[current]

            # check if is better
            current_goods = self._goods(values, lables)
            if current_goods > best_goods:
                best_goods = current_goods
                best_index = index.copy()
                print("best_goods", best_goods)

            if (loop % (10**5)) == 0:
                now = datetime.now()
                print((now - last_print).total_seconds(), "%.3f" % (100 * loop / loops))
                last_print = now
            loop += 1

        # create the best tree
        for node, value in zip(nodes, best_index):
            node.value = value

    def _fit_binary_entropy(self, values: np.ndarray, lables: np.ndarray) -> None:
        self._create_tree(self.root, 0)

    def _create_tree(self, parent: Node, layer: int):
        if layer == self.layers:
            return

        parent.left = Node()
        parent.right = Node()

        self._create_tree(parent.left, layer + 1)
        self._create_tree(parent.right, layer + 1)

    def predict(self, values: np.ndarray) -> np.ndarray | int:
        single_mode = len(values.shape) == 1

        if single_mode:
            return self._predict_one(values)

        lables = np.zeros(values.shape[0])

        for i in range(values.shape[0]):
            lables[i] = self._predict_one(values[i])

        return lables

    def _predict_one(self, value: np.ndarray) -> int:
        node = self.root
        layer = 0

        while layer < self.layers:
            if value[node.value] == 1:
                node = node.right
            else:
                node = node.left

            layer += 1

        return node.value

    def _goods(self, values: np.ndarray, lables: np.ndarray) -> int:
        pred = self.predict(values)

        return np.equal(lables, pred).sum()


def _create_nodes_list(root: Node) -> list[Node]:
    nodes = []
    to_visit = [root]

    while len(to_visit) > 0:
        node = to_visit.pop(0)
        nodes.append(node)

        if node.left is not None:
            to_visit.append(node.left)
        if node.right is not None:
            to_visit.append(node.right)

    return nodes
