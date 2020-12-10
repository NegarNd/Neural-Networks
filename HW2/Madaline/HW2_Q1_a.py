import numpy as np
import matplotlib.pyplot as plt


dataset = [(0, 0, 1), (1, -1, 1), (0, 1, 1), (-1, -1, 1)]
target = np.array([-1, 1, 1, 1])
weights_1 = np.zeros((3, 3))

rate = 0.5
# weights of second layer are defined
weights_2 = np.array([1, 1, 1, 2])

plt.xlabel('X_1')
plt.ylabel('X_2')
x = np.linspace(-2, 2)
plt.plot([-1, 0, 1], [-1, 1, -1], 'ro')
plt.plot([0], [0], 'b*')

F_1 = 2*x -1
F_2 = x+ 0.5
F_3 = -2*x - 2
plt.plot(x, F_1, 'red')
plt.plot(x, F_2, 'green')
plt.plot(x, F_3, 'navy')
plt.show()
