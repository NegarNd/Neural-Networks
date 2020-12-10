import numpy as np
import matplotlib.pyplot as plt
from MADALINE import find_weights

dataset = [(0, 0, 1), (1, -1, 1), (0, 1, 1), (-1, -1, 1)]
target = np.array([-1, 1, 1, 1])
weights_1 = np.zeros((3, 3))

rate = 0.5
# weights of second layer are defined
weights_2 = np.array([1, 1, 1, 2])


def initial_weights_1(weights):
    # neuron one
    weights[0][0] = 0
    weights[0][1] = 0.5
    weights[0][2] = 0.25
    # neuron two
    weights[1][0] = 0.5
    weights[1][1] = -0.5
    weights[1][2] = 0.25
    # neuron three
    weights[2][0] = -0.5
    weights[2][1] = -0.5
    weights[2][2] = -0.25


initial_weights_1(weights_1)
print(weights_1[1][2])
weights_L1 = find_weights(dataset, target, weights_1, weights_2, rate)
print(weights_L1)
plt.xlabel('X_1')
plt.ylabel('X_2')
x = np.linspace(-2, 2)
F_1 = (-1 * x * weights_L1[0][0] - 1 * weights_L1[0][2]) / weights_L1[0][1]
F_2 = (-1 * x * weights_L1[1][0] - 1 * weights_L1[1][2]) / weights_L1[1][1]
F_3 = (-1 * x * weights_L1[2][0] - 1 * weights_L1[2][2]) / weights_L1[2][1]
plt.plot([-1, 0, 1], [-1, 1, -1], 'rs')
plt.plot([0], [0], 'b*')
plt.plot(x, F_1, 'red')
plt.plot(x, F_2, 'green')
plt.plot(x, F_3, 'navy')
plt.show()
