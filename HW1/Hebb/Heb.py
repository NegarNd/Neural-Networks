import numpy as np


def find_weights(data, rate):
    weight = np.zeros((4,), dtype=float)
    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            weight[j] = weight[j] + data[i][j] * data[i][-1] * rate
    return weight
