import numpy as np


def activation_function(input, weights,th):
    net = 0.0
    length = len(input)
    for i in range(length):
        net = net + weights[i]*input[i]
    # theta is equal to 0 for calculating Y output
    if net >= th:
        return 1
    elif net < (-1*th):
        return -1
    else:
        return 0


def find_weights(dataset, target, rate, th):
    iteration = 0
    weight = np.zeros((len(dataset[0]),), dtype=float)
    # for epoch in range(n_epoch):
    pattern_number = 0
    while 1:
        next_data = dataset[pattern_number]
        next_data_target = target[pattern_number]
        pattern_number = pattern_number + 1

        #call the activation function
        prediction = activation_function(next_data, weight, th)
        error = prediction - next_data_target
        if error == 0.0:
            iteration = iteration + 1
        else:
            iteration = 0
            for i in range(len(next_data)):
                weight[i] = weight[i]+rate*next_data_target*next_data[i]
        if iteration != len(dataset) and (next_data == dataset[-1]).all():
            pattern_number = 0
        if iteration == len(dataset):
            break
    print(weight)
    return weight

