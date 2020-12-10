from Heb import find_weights
import numpy as np


# define dataset array. 3D input + 1 + target
ax =np.array([-1, -1 , 1,1,-1,-1,1,1])
ay =np.array([-1, 1, -1,1,-1,1,-1,1])
az =np.array([1,1,1,1,-1, -1, -1, -1])
bias = np.ones(8)
target = np.array([-1,1,-1,-1,-1,1,-1,-1])
dataset = np.column_stack((ax,ay,az, bias, target))

print(dataset)
result = find_weights(dataset, 1.0)

print('calcuated weights are',result)