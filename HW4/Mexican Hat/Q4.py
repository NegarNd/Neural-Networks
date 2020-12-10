import numpy as np
import matplotlib.pyplot as plt

data = np.array([[0.11, 0.13, 0.13, 0.1, 0.15, 0.12, 0.11, 0.09],
                 [0.21, 0.31, 0.28, 0.19, 0.29, 0.33, 0.3, 0.26],
                 [0.3, 0.6, 0.61, 0.33, 0.42, 0.52, 0.46, 0.41],
                 [0.57, 0.96, 0.73, 0.42, 0.51, 0.6, 0.59, 0.5],
                 [0.63, 0.7, 0.73, 0.55, 0.73, 0.79, 0.79, 0.66],
                 [0.44, 0.62, 0.57, 0.53, 0.79, 0.99, 0.8, 0.63],
                 [0.25, 0.28, 0.29, 0.43, 0.7, 0.77, 0.71, 0.61],
                 [0.09, 0.11, 0.14, 0.26, 0.46, 0.53, 0.55, 0.51]])


def activation_fn(x):

    if x < 0:
        return 0
    elif x > 2:
        return 2
    else:
        return x


def Location(i, j, r1, r2, n):
    k = j - r1
    h = j + r1
    l = i - r1
    m = i + r1
    a = j - r2
    b = j + r2
    c = i - r2
    d = i + r2
    # print(h)
    if k < 0:
        k = 0
    if l < 0:
        l = 0
    if h >= n:
        h = n - 1
    if m >= n:
        m = n - 1
    if a < 0:
        a = 0
    if b >= n - 1:
        b = n - 1
    if c < 0:
        c = 0
    if d >= n - 1:
        d = n - 1
    return k, l, h, m, a, b, c, d


def Mexician_Hat(c1, c2, r1, r2, size, t_max):
    for t in range(0, t_max):
        data_old = data.copy()
        for i in range(0, size):
            for j in range(0, size):
                k, l, h, m, a, b, c, d = Location(i, j, r1, r2, size)
                positive_sum = 0
                negative_sum = 0
                for x in range(l, m + 1):
                    for z in range(k, h + 1):
                        positive_sum = positive_sum + data_old[x][z]
                for e in range(c, d + 1):
                    for f in range(a, b + 1):
                        negative_sum = negative_sum + data_old[e][f]
                data[i][j] = activation_fn((c1 * positive_sum) + (c2 * (negative_sum - positive_sum)))
        print(data)
        # print(unravel_index(data.argmax(), data.shape))
        # print(data.argmax())
        # convert 2D array to 1D
        b = data.ravel()
        plt.plot(b, label=t)
        print("**********")

    plt.legend()
    plt.show()


# part one
Mexician_Hat(0.9, -0.01, 0, 100, 8, 5)
# part two
Mexician_Hat(0.6, -0.6,2, 4, 8, 5)