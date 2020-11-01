import pandas as pd
from cf_tasks.c import knn
from cf_tasks.b import calc_f
import matplotlib.pyplot as plt
from ml_labs.const_lab1 import y_macro_native, y_macro_one_hot, y_micro_native, y_micro_one_hot

target_name = 'Type'
filename = "glass.csv"
distance_type = "manhattan"
kernel_type = "triweight"
window_type = "variable"
hk = 5


def max_i(list_val):
    maxim = -1
    res = -1
    for i in range(len(list_val)):
        if list_val[i] > maxim:
            maxim = list_val[i]
            res = i
    return res


def math_round(num):
    return int(num + 0.5)


def dataset_minmax(dataset):
    minmax = list()
    data_len = len(dataset[0])
    for i in range(data_len):
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


def knn_for_i_element(data, i):
    q = data[i][0]
    elements = data.copy()
    elements.pop(i)
    return knn(elements, q, distance_type, kernel_type, window_type, hk)


def good_and_bad(data):
    good = 0
    bad = 0
    for i in range(len(data)):
        real_type = data[i][1]
        act_type = math_round(knn_for_i_element(data, i))
        if real_type == act_type:
            good += 1
        else:
            bad += 1
    return good, bad


def create_error_table_native(data, num_of_types):
    table = [[0 for _ in range(num_of_types)] for _ in range(num_of_types)]
    for i in range(len(data)):
        real_type = data[i][1]
        act_type = math_round(knn_for_i_element(data, i))
        table[real_type][act_type] += 1
    return table


def choose_metric_and_kernel_native(data):
    metrics = ["manhattan", "euclidean", "chebyshev"]
    kernels = ["uniform",
               "triangular",
               "epanechnikov",
               "quartic",
               "triweight",
               "tricube",
               "gaussian",
               "cosine",
               "logistic",
               "sigmoid"]
    max_good = 0
    best_pair = ("", "")
    for i in metrics:
        for j in kernels:
            global distance_type, kernel_type
            distance_type = i
            kernel_type = j
            good = good_and_bad(data)[0]
            if good > max_good:
                max_good = good
                best_pair = i, j
    print(max_good)
    return best_pair


def run_native(dataset_norm, unique_types, num_types, types):
    types_rename = list(map(lambda x: num_types[x], types))
    data = list(zip(dataset_norm, types_rename))
    return create_error_table_native(data, len(unique_types))


def run_one_hot(dataset_norm, unique_types, num_types, types):
    real_type = list(map(lambda x: num_types[x], types))
    size = len(unique_types)
    type_renames = []
    for i in range(size):
        type_renames.append(list(map(lambda x: 1 if i == num_types[x] else 0, types)))
    results_table = [[0 for _ in range(size)] for _ in range(len(dataset_norm))]
    for i in range(size):
        data = list(zip(dataset_norm, type_renames[i]))
        for j in range(len(data)):
            results_table[j][i] = knn_for_i_element(data, j)
    act_type = list(map(max_i, results_table))
    table = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(len(real_type)):
        table[real_type[i]][act_type[i]] += 1
    return table


def print_native():
    plt.plot(x, y_micro_native, 'r')
    plt.plot(x, y_macro_native, 'b')
    plt.title("Native")
    plt.show()


def print_one_hot():
    plt.plot(x, y_micro_one_hot, 'r')
    plt.plot(x, y_macro_one_hot, 'b')
    plt.title("One hot")
    plt.show()


def calculate(dataset, dataset_norm, unique_types, num_types, types):
    y11mac = []  # native
    y22mac = []  # one hot
    y11mic = []  # native
    y22mic = []  # one hot
    for i in range(len(dataset.values)):
        print(i)
        global hk
        hk = i
        matrix1 = run_native(dataset_norm, unique_types, num_types, types)
        f_macro1, f_micro1 = calc_f(matrix1)
        matrix2 = run_one_hot(dataset_norm, unique_types, num_types, types)
        f_macro2, f_micro2 = calc_f(matrix2)
        y11mac.append(f_macro1)
        y11mic.append(f_micro1)
        y22mac.append(f_macro2)
        y22mic.append(f_micro2)
    print(y11mac)
    print(y11mic)
    print(y22mac)
    print(y22mic)
    return y11mac, y11mic, y22mac, y22mic


dataset = pd.read_csv(filename)

types = dataset[target_name].values.tolist()
dataset = dataset.drop(columns=target_name)

unique_types = list(set(types))
unique_types.sort()
num_types = dict([(x, i) for i, x in enumerate(unique_types)])
print(num_types)

minmax = dataset_minmax(dataset.values)
dataset_norm = normalize(dataset.values, minmax)

x = [i for i in range(len(dataset.values))]

# calculate(dataset, dataset_norm, unique_types, num_types, types)

print_native()
print(max([(x, i) for i, x in enumerate(y_macro_native)])[1])
print_one_hot()
print(max([(x, i) for i, x in enumerate(y_macro_one_hot)])[1])
