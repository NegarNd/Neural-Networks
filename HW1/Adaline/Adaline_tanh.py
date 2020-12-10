import numpy as np


def prediction_func(input, weights):
    net=0.0
    length = len(input)
    for i in range(length):
        net = net + weights[i]*input[i]
    # theta is equal to 0 for calculating Y output
    return net


def find_weights(dataset, target, rate, th, g):
    cost_function_list = []
    cost_function_sum = 0
    iter_counter = 0
    weight = np.zeros((len(dataset[0]),), dtype=float)
    # for epoch in range(n_epoch):
    pattern_number = 0
    num_of_epoch = 0
    while 1:
        row = dataset[pattern_number]
        row_target = target[pattern_number]
        pattern_number = pattern_number + 1

        #call the activation function
        prediction = prediction_func(row, weight)
        # error = prediction - row_target
        for i in range(len(row)):
            weight[i] = weight[i] + rate*(row_target - prediction)*row[i]

        cost_function = 0.5*((row_target - np.tanh(g*prediction))**2)
        cost_function_sum = cost_function_sum + cost_function

        if cost_function < th:
            iter_counter = iter_counter + 1
        else:
            iter_counter = 0

        if iter_counter != len(dataset) and (row == dataset[-1]).all():
            cost_function_list.append(cost_function_sum)
            cost_function_sum = 0
            pattern_number = 0
            num_of_epoch = num_of_epoch + 1
        if iter_counter == len(dataset):
            cost_function_list.append(cost_function_sum)
            break

    print(cost_function_list)
    return weight, cost_function_list

