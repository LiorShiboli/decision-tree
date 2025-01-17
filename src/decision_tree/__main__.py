import sys

import numpy as np

from decision_tree.decision_tree import DecisionTree


INPUT_PASTH = './data/vectors.txt'


def load_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(file_path, 'r') as f:
        data = f.read()

    arr = [[int(item) for item in line.split(' ') if item != ''] for line in data.split('\n') if len(line) > 0]

    full = np.array(arr)

    values = full[:, :-1]
    lables = full[:, -1]

    return values, lables


def run(binary_entropy_mode: bool) -> None:
    values, lables = load_data(INPUT_PASTH)

    model = DecisionTree(2)
    print('fit:')
    model.fit(values, lables, binary_entropy_mode)
    print()

    print("Tree:")
    print(model.root, '\n')

    predict = model.predict(values)
    print('Predict:')
    print(predict)

    accuracy = np.equal(lables, predict).mean()
    print("Accuracy:")
    print(accuracy)

    goods = np.equal(lables, predict).sum()
    bads = np.not_equal(lables, predict).sum()
    print("Goods / Bads:")
    print(goods, '/', bads)


def main():
    binary_entropy_mode = False

    if 'brute-force' in sys.argv or 'binary-entropy' in sys.argv:
        binary_entropy_mode = 'binary-entropy' in sys.argv

        run(binary_entropy_mode)
        return

    print("Help:")
    print('Choose brute-force or binary-entropy as mode')
    print(' python -m decision_tree brute-force')
    print(' python -m decision_tree binary-entropy')


if __name__ == '__main__':
    main()
