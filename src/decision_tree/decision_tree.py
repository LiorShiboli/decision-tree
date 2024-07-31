from typing import Self

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

        space_length = len(left_value.split('\n')[0]) + 1

        ret = (' ' * space_length) + str(self.value) + (' ' * space_length)
        for line1, line2 in zip(left_value.split('\n'), right_value.split('\n')):
            ret += '\n ' + line1 + ' ' + line2 + ' '

        return ret


class DecisionTree:
    layers: int
    root: Node
    trained: bool
    input_size: int

    def __init__(self, layers: int) -> None:
        self.layers = layers
        self.root = Node()
        self.trained = False
        self.input_size = 0

    def fit(self, values: np.ndarray, lables: np.ndarray, binary_entropy_mode: bool) -> None:
        if self.trained:
            raise Exception("Cannot train twice")

        self.trained = True
        self.input_size = values.shape[1]

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

            loop += 1

        # create the best tree
        for node, value in zip(nodes, best_index):
            node.value = value

    def _fit_binary_entropy(self, values: np.ndarray, lables: np.ndarray) -> None:
        self._fit_binary_entropy_layers(values, lables, self.root, 0)

    def _fit_binary_entropy_layers(self, values: np.ndarray, lables: np.ndarray, node: Node, layer: int) -> None:
        if layer == self.layers:
            if (lables == 1).sum() > (lables == 0).sum():
                node.value = 1
            else:
                node.value = 0
            return

        node.left = Node()
        node.right = Node()

        best_dim = self._fit_binary_entropy_best_dim(values, lables)
        node.value = best_dim

        left_values = values[values[:, best_dim] == 0]
        left_lables = lables[values[:, best_dim] == 0]
        right_values = values[values[:, best_dim] == 1]
        right_lables = lables[values[:, best_dim] == 1]

        only_zero = len(left_lables[left_lables == 0])
        if only_zero == 0:
            node.left.value = 1
        elif only_zero == len(left_lables):
            node.left.value = 0
        else:
            self._fit_binary_entropy_layers(left_values, left_lables, node.left, layer + 1)

        only_zero = len(right_lables[right_lables == 0])
        if only_zero == 0:
            node.left.value = 1
        elif only_zero == len(right_lables):
            node.right.value = 0
        else:
            self._fit_binary_entropy_layers(right_values, right_lables, node.right, layer + 1)

    def _fit_binary_entropy_best_dim(self, values: np.ndarray, lables: np.ndarray) -> int:
        best_dim = 0

        best_cost = 0

        for dim in range(values.shape[1]):
            cost = 1
            for val, lable in [[0, 0], [0, 1], [1, 0], [1, 1]]:
                part = values[:, dim] == val
                part_values = values[part]
                part_lable = values[part & (lables == lable)]

                if len(part_values) > 0:
                    cost += (len(part_lable) / len(part_values)) * np.log2(len(part_lable) / len(part_values))

            if best_cost < cost:
                best_dim = dim
                best_cost = cost

        return best_dim

    def _create_tree(self, parent: Node, layer: int):
        if layer == self.layers:
            return

        parent.left = Node()
        parent.right = Node()

        self._create_tree(parent.left, layer + 1)
        self._create_tree(parent.right, layer + 1)

    def predict(self, values: np.ndarray) -> np.ndarray | int:
        if not self.trained:
            raise Exception("Need to train first")

        single_mode = len(values.shape) == 1

        if single_mode:
            return self._predict_one(values)

        lables = np.zeros(values.shape[0])

        for i in range(values.shape[0]):
            lables[i] = self._predict_one(values[i])

        return lables

    def _predict_one(self, value: np.ndarray) -> int:
        node = self.root

        while node.left is not None:
            if value[node.value] == 1:
                node = node.right
            else:
                node = node.left

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
