import numpy as np
import matplotlib.pyplot as plt
from Perseptron_Algorithm import find_weights

ax = np.random.normal(1, 0.6 , 100)
ay = np.random.normal(1, 0.3 , 100)
bx = np.random.normal(-1, 0.6, 50)
by = np.random.normal(-1, 0.4, 50)

x_data = np.concatenate((ax, bx), axis=0)
y_data = np.concatenate((ay, by), axis=0)
target = np.concatenate((np.ones(100), np.ones(50)*(-1.0)), axis = 0)
bias = np.ones(150)
dataset = np.column_stack((x_data, y_data, bias))

final_weight = find_weights(dataset, target, 0.5,0.1)

xr = np.linspace(-3.0, 3.0, 100)
yr = np.linspace(-3.0, 3.0, 100)

X1, Y1 = np.meshgrid(xr, yr)
F = final_weight[0]*X1 + final_weight[1]*Y1 + final_weight[2]
plt.contour(X1,Y1,F,[0])

plt.scatter(ax, ay)
plt.scatter(bx, by)
plt.show()

