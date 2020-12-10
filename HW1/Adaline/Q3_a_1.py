import numpy as np
import matplotlib.pyplot as plt
from AdaLine import find_weights


ax = np.array([1,2,3,4,5,6,7])
ay = np.array([0,1,-1,1,-1,1,-1])
bx = np.array([-1])
by = np.array([0])

x_data = np.concatenate((ax, bx), axis=0)
y_data = np.concatenate((ay, by), axis=0)
target = [1, 1, 1, 1, 1, 1,1,-1]
bias = np.ones(8)
dataset = np.column_stack((x_data, y_data, bias))
final_weight, cost_function = find_weights(dataset, target, 0.1, 1.4)
x1 = np.linspace(-2.0, 8.0)
x2 = np.linspace(-2.0, 2.0)

y1 = np.linspace(-1.5,1.5)
y2 = np.linspace(0, 3)

X1, Y1 = np.meshgrid(x1, y1)
X2, Y2 = np.meshgrid(x2, y2)

F = final_weight[0] * X1 + final_weight[1] * Y1 + final_weight[2]
plt.subplot(2, 1, 1)
plt.contour(X1, Y1, F, [0])
plt.scatter(ax, ay)
plt.scatter(bx, by)

plt.subplot(2, 1, 2)
plt.plot(cost_function, 'r')
plt.xlabel('cost function')
plt.show()

