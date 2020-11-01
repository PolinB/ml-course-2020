import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cf_tasks.e import calc_with_data
from ml_labs import const_lab3


def scalar_mul(x_i, x_j):
    x_len = len(x_i)
    res = 0
    for i in range(x_len):
        res += (x_i[i] * x_j[i])
    return res


def linear_kernel(x_i, x_j):
    return scalar_mul(x_i, x_j)


def polynomial_kernel(x_i, x_j, d):
    return (scalar_mul(x_i, x_j)) ** d


def gaussian_kernel(x_i, x_j, sigma):
    x_len = len(x_i)
    t = 0
    for i in range(x_len):
        t += (x_i[i] - x_j[i]) ** 2
    return math.exp(-t / (2. * (sigma ** 2)))


def get_kernel_matrix(data, kernel_function):
    n = len(data)
    kernel_data = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            kernel_data[i][j] = kernel_function(data[i], data[j])
    return kernel_data


def learn_without_i(pos, data, y, kernel_function, cvs_c):
    dataset2 = list(data.copy())
    y2 = y.copy()
    cur_val = dataset2.pop(pos)
    cur_y = y2.pop()
    kernel_matrix = get_kernel_matrix(dataset2, kernel_function)
    alphas, b = calc_with_data(len(kernel_matrix), kernel_matrix, y2, cvs_c)
    predict_y = predict2(cur_val, alphas, b, dataset2, y2, kernel_function)
    return predict_y == cur_y


def learn_without_range(left, k, data, y, kernel_function, cvs_c):
    dataset2 = list(data.copy())
    y2 = y.copy()
    right = min(left + k, len(y))
    cur_vals = dataset2[left:right]
    del dataset2[left:right]
    cur_ys = y2[left:right]
    del y2[left:right]
    kernel_matrix = get_kernel_matrix(dataset2, kernel_function)
    alphas, b = calc_with_data(len(kernel_matrix), kernel_matrix, y2, cvs_c)
    good = 0
    for i in range(len(cur_vals)):
        predict_y = predict2(cur_vals[i], alphas, b, dataset2, y2, kernel_function)
        if predict_y == cur_ys[i]:
            good += 1
    return good


def predict(pos, alphas, b, kernel_matrix, y):
    res = b
    for i in range(len(alphas)):
        res += (y[i] * alphas[i] * kernel_matrix[pos][i])
    return int(np.sign(res))


def predict2(val, alphas, b, data, y, kernel_function):
    res = b
    for i in range(len(alphas)):
        res += (y[i] * alphas[i] * kernel_function(val, data[i]))
    return int(np.sign(res))


def predict_all(alphas, b, kernel_matrix, y):
    res = []
    for i in range(len(kernel_matrix)):
        res.append(predict(i, alphas, b, kernel_matrix, y))
    return res


def predict_all2(values, alphas, b, data, y, kernel_function):
    res = []
    for i in range(len(values)):
        res.append(predict2(values[i], alphas, b, data, y, kernel_function))
    return res


def check_parameters(k, dataset, y, kernel_function, c_svm):
    left_line = 0
    good_res = 0
    all_res = len(y)
    while left_line < len(y):
        good_res += learn_without_range(left_line, k, dataset.values, y, kernel_function, c_svm)
        left_line += k
    return float(good_res) / all_res


def get_best_parameters():
    k = 20
    # k = 40
    c_svms = [0.1, 0.5, 1, 5, 10, 50]  # , 100]
    d_parameters = [2, 3, 4, 5]
    sigmas = [0.1, 0.5, 1, 5]

    # for linear
    for c_svm in c_svms:
        kernel_function = lambda x_i, x_j: linear_kernel(x_i, x_j)
        print(c_svm)
        print(check_parameters(k, dataset, y, kernel_function, c_svm))

    # polinomial
    for c_svm in c_svms:
        for d_p in d_parameters:
            kernel_function = lambda x_i, x_j: polynomial_kernel(x_i, x_j, d_p)
            print(c_svm, d_p)
            print(check_parameters(k, dataset, y, kernel_function, c_svm))

    # gauss
    for c_svm in c_svms:
        for si in sigmas:
            kernel_function = lambda x_i, x_j: gaussian_kernel(x_i, x_j, si)
            print(c_svm, si)
            print(check_parameters(k, dataset, y, kernel_function, c_svm))


def show_plot(d, func, name, alphas, b):
    dataset = d.copy()
    xx = dataset['x'].values.tolist()
    yy = dataset['y'].values.tolist()
    colors = list(map(lambda x: 'blue' if x == 1 else 'red', y))

    h = 0.5
    x_min, x_max = min(xx) - 1, max(xx) + 1
    y_min, y_max = min(yy) - 1, max(yy) + 1
    xxx, yyy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
    d = np.c_[xxx.ravel(), yyy.ravel()]
    predicted_y = predict_all2(d, alphas, b, dataset.values, y, func)

    z = (np.array(predicted_y)).reshape(xxx.shape)
    plt.contourf(xxx, yyy, z, alpha=1)
    plt.scatter(xx, yy, c=colors)
    plt.title(name)
    plt.savefig("lab3_svm_files/" + name + ".png")
    plt.show()


# filename = "lab3_svm_files/test.csv"
filename = "lab3_svm_files/chips.csv"
# filename = "lab3_svm_files/geyser.csv"
target_name = "class"

dataset = pd.read_csv(filename)
# dataset = shuffle(dataset)

y = dataset[target_name].values.tolist()
y = list(map(lambda y_i: 1 if y_i == 'P' else -1, y))
dataset = dataset.drop(columns=target_name)

# linear_f = lambda x_i, x_j: linear_kernel(x_i, x_j)
# kernel_matrix_l = get_kernel_matrix(dataset.values, linear_f)
# a, b = calc_with_data(len(kernel_matrix_l), kernel_matrix_l, y, svm_constants.chips_l_c)
#
polin_f = lambda x_i, x_j: polynomial_kernel(x_i, x_j, const_lab3.chips_p_d)
kernel_matrix_p = get_kernel_matrix(dataset.values, polin_f)
a, b = calc_with_data(len(kernel_matrix_p), kernel_matrix_p, y, const_lab3.chips_p_c)
#
# gauss_f = lambda x_i, x_j: gaussian_kernel(x_i, x_j, svm_constants.chips_g_s)
# kernel_matrix_g = get_kernel_matrix(dataset.values, gauss_f)
# print(calc_with_data(len(kernel_matrix_g), kernel_matrix_g, y, svm_constants.chips_g_c))

# show_plot(dataset, linear_f, "LINEAR_CHIPS2", a, b)
show_plot(dataset, polin_f, "POLYNOMIAL_CHIPS2", a, b)
# show_plot(dataset, gauss_f, "GAUSS_CHIPS", svm_constants.chips_g_a, svm_constants.chips_g_b)

# linear_f = lambda x_i, x_j: linear_kernel(x_i, x_j)
# kernel_matrix_l = get_kernel_matrix(dataset.values, linear_f)
# alphas_t, b_t = calc_with_data(len(kernel_matrix_l), kernel_matrix_l, y, svm_constants.geyser_l_c)
# print(alphas_t, b_t)

# polin_f = lambda x_i, x_j: polynomial_kernel(x_i, x_j, svm_constants.geyser_p_d)
# kernel_matrix_p = get_kernel_matrix(dataset.values, polin_f)
# print(calc_with_data(len(kernel_matrix_p), kernel_matrix_p, y, svm_constants.geyser_p_c))

# gauss_f = lambda x_i, x_j: gaussian_kernel(x_i, x_j, svm_constants.geyser_g_s)
# kernel_matrix_g = get_kernel_matrix(dataset.values, gauss_f)
# print(calc_with_data(len(kernel_matrix_g), kernel_matrix_g, y, svm_constants.geyser_g_c))

#
# show_plot(dataset, linear_f, "LINEAR_GEYSER", svm_constants.gla, svm_constants.glb)
# show_plot(dataset, polin_f, "POLYNOMIAL_GEYSER", svm_constants.gpa, svm_constants.gpb)
# show_plot(dataset, gauss_f, "GAUSS_GEYSER", svm_constants.gga, svm_constants.ggb)


# linear_f = lambda x_i, x_j: linear_kernel(x_i, x_j)
# kernel_matrix_l = get_kernel_matrix(dataset.values, linear_f)
# alphas_t, b_t = calc_with_data(len(kernel_matrix_l), kernel_matrix_l, y, 1)
# print(alphas_t, b_t)
# show_plot(dataset, linear_f, "LINEAR_TEST", alphas_t, b_t)