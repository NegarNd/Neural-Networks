import numpy as np
import matplotlib.pyplot as plt
from Perseptron_Algorithm import find_weights


# apply this function on input data so perseptron can classify these data
def function_on_x(x):
    new_data = np.zeros((13, 5))
    for i in range(len(x)):
        new_data[i][0] = x[i][0] ** 2
        new_data[i][1] = x[i][1] ** 2
        new_data[i][2] = x[i][0]
        new_data[i][3] = x[i][1]
        new_data[i][4] = 1
    return new_data


ax = np.array([0.25,1,1.75,2,1.75,1,0.25,0])
ay = np.array([0.25,0,0.25,1,1.75,2,1.75,1])
bx = np.random.normal(1, 0.6, 5)
by = np.random.normal(1, 0.3, 5)

x_data = np.concatenate((ax, bx), axis=0)
y_data = np.concatenate((ay, by), axis=0)
dataset = np.column_stack((x_data, y_data))
target = [1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1]


final_data = function_on_x(dataset)

plt.scatter(ax, ay)
plt.scatter(bx, by)
plt.show()
final_weight = find_weights(final_data, target, 1.0, 0.0)

x = np.linspace(0,2)
y = np.linspace(0,2)
X,Y = np.meshgrid(x,y)


F = final_weight[0]*(X**2) + final_weight[1]*(Y**2) + final_weight[2]*X + final_weight[3]*Y + final_weight[4]

plt.contour(X,Y,F,[0])
plt.scatter(ax, ay)
plt.scatter(bx, by)
plt.show()


