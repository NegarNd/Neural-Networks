import numpy as np


def activation_func(data, weight):
    net = 0
    for i in range(len(data)):
        net = net + data[i] * weight[i]

    if net >= 0:
        return 1, net
    else:
        return -1, net


def find_weights(dataset, target, weight_1, weight_2, rate):
    print(dataset)
    iteration = 0
    pattern_num = 0
    epoch_num = 0
    while 1:
        data = dataset[pattern_num]
        pattern_target = target[pattern_num]
        pattern_num = pattern_num + 1
        result_1 = activation_func(data, weight_1[0])
        result_2 = activation_func(data, weight_1[1])
        result_3 = activation_func(data, weight_1[2])

        # add net values to an array
        net_values = np.zeros(3)
        net_values[0] = result_1[1]
        net_values[1] = result_2[1]
        net_values[2] = result_3[1]

        # prepare data for second layer
        dataset_l2 = np.zeros(4)
        dataset_l2[0] = result_1[0]
        dataset_l2[1] = result_2[0]
        dataset_l2[2] = result_3[0]
        dataset_l2[3] = 1
        # print(dataset_l2)
        # call activation function for final neuron
        # print(weight_2)
        final_result = activation_func(dataset_l2, weight_2)

        error = final_result[0] - pattern_target
        # if error = 0 no weight updates are performed
        if error == 0:
            iteration = iteration + 1
        else:
            iteration = 0
            if pattern_target == 1:
                min_index = np.argmin(net_values)
                if net_values[min_index] < 0:
                    for i in range(len(data)):
                        weight_1[min_index][i] = weight_1[min_index][i] + rate * (1 - net_values[min_index]) * data[i]
                # if result_1[1] < 0:
                #     for i in range(len(data)):
                #         weight_1[0][i] = weight_1[0][i] + rate * (1 - net_values[0]) * data[i]
                # if result_2[1] < 0:
                #     for i in range(len(data)):
                #         weight_1[1][i] = weight_1[1][i] + rate * (1 - net_values[1]) * data[i]
                # if result_3[1] < 0:
                #     for i in range(len(data)):
                #         weight_1[2][i] = weight_1[2][i] + rate * (1 - net_values[2]) * data[i]
            else:
                if result_1[1] >= 0:
                    for i in range(len(data)):
                        weight_1[0][i] = weight_1[0][i] + rate * (-1 - net_values[0]) * data[i]
                if result_2[1] >= 0:
                    for i in range(len(data)):
                        weight_1[1][i] = weight_1[1][i] + rate * (-1 - net_values[1]) * data[i]
                if result_3[1] >= 0:
                    for i in range(len(data)):
                        weight_1[2][i] = weight_1[2][i] + rate * (-1 - net_values[2]) * data[i]

        if iteration != len(dataset) and data == dataset[-1]:
            epoch_num = epoch_num + 1
            pattern_num = 0
            # print(epoch_num)
        if iteration == len(dataset):
            break

    print('number of epochs: ', epoch_num)
    return weight_1
