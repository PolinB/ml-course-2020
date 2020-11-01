import random
import time
from enum import Enum
import math
import matplotlib.pyplot as plt
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from ml_labs import const_lab2
from numpy.linalg import svd
from numpy import zeros
from numpy import diag

iters_nums = []
nrmse_iters = []


class StopType(Enum):
    TIME = 1
    ITER = 2


class ErrorFunc(Enum):
    SMAPE = 1
    NRMSE = 2


coeffs = []
mu = 0.1
iter_num_const = 500
time_const = 3.0
stop_type = StopType.TIME
err_func_const = ErrorFunc.NRMSE


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            if minmax[i][1] - minmax[i][0] == 0:
                row[i] = 0
                continue
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


# вычисляет функцию ошибки smape
def smape(abs_result, cur_result):
    n = len(abs_result)
    sum = 0
    for i in range(n):
        sum += abs(abs_result[i] - cur_result[i]) / (abs(abs_result[i]) + abs(cur_result[i]))
    return sum / n


def nrmse(abs_result, cur_result):
    n = len(abs_result)
    sum = 0
    for i in range(n):
        sum += ((abs_result[i] - cur_result[i]) ** 2)
    sum /= n
    sum = math.sqrt(sum)
    max_v = max(abs_result)
    min_v = min(abs_result)
    return sum / (max_v - min_v)


# вычисляет предполагаемый y при заданных коэффициентах
def predict(row):
    y = coeffs[0]
    for i in range(len(row)):
        y += coeffs[i + 1] * row[i]
    return y


def predict_with_one(row):
    y = 0
    for i in range(len(row)):
        y += coeffs[i] * row[i]
    return y


def predict_with_one2(row, c):
    y = 0
    for i in range(len(row)):
        y += c[i] * row[i]
    return y


# вычисляет предполагаемый y при заданных коэффициентах
def predict2(row, coefs):
    y = coefs[0]
    for i in range(len(row)):
        y += coefs[i + 1] * row[i]
    return y


def calc_dist(row_num, data, abs_result):
    return predict(data[row_num]) - abs_result[row_num]


def init_coeff(m):
    res = []
    for i in range(m + 1):
        r = (random.random() / m) - (1 / (2 * m))
        res.append(r)
    global coeffs
    coeffs = res


def init_err_func(n, data, abs_result):
    sum = 0
    for i in range(n):
        sum += (calc_dist(i, data, abs_result)) ** 2
    return sum


def do_step(n, data, abs_result, iter_num):
    global coeffs
    pos = random.randint(0, n - 1)

    diff = calc_dist(pos, data, abs_result)
    cur_err = diff * 2

    dQ = data[pos].copy()
    dQ = list(map(lambda x: x * cur_err, dQ))

    t = 1 / iter_num
    coeffs[0] -= mu * t * cur_err
    for i in range(len(data[0])):
        coeffs[i + 1] -= mu * t * dQ[i]


def read_test_dataset(num):
    filename = "lab2_files/" + str(num) + "_res.txt"
    n, m, data, abs_result, minmax = read_data_from_file(filename)
    return data, abs_result


def do_algo(n, m, data, abs_result, mm, num=1):
    global l, iters_nums, nrmse_iters
    iters_nums = list()
    nrmse_iters = list()
    l = 1.0 / n
    init_coeff(m)
    iter_num = 0
    start_time = time.time()
    testSetX, testSetY = read_test_dataset(num)
    if stop_type == StopType.ITER:
        while iter_num < iter_num_const:
            print('Step ' + iter_num.__str__() + ' of ' + iter_num_const.__str__())
            iter_num += 1
            iters_nums.append(iter_num)
            do_step(n, data, abs_result, iter_num)
            nrmse_iters.append(compare_results_2(coeffs, testSetX, testSetY))

    elif stop_type == StopType.TIME:
        while time.time() - start_time < time_const:
            iter_num += 1
            time_dist = time.time() - start_time
            iters_nums.append(time_dist)
            do_step(n, data, abs_result, iter_num)
            nrmse_iters.append(compare_results_2(coeffs, testSetX, testSetY))
    return coeffs


def read_data_from_file(filename):
    f = open(filename, "r")
    n, m = map(int, f.readline().split())
    data = []
    abs_result = []
    minmax = []
    for i in range(n):
        data_i = list(map(int, f.readline().split()))[:(m + 1)]
        y_i = data_i.pop()
        x_i = data_i
        data.append(x_i)
        abs_result.append(y_i)
        if i == 0:
            minmax = list(zip(data_i, data_i))
        else:
            for j in range(len(data_i)):
                if data_i[j] < minmax[j][0]:
                    minmax[j] = (data_i[j], minmax[j][1])
                if data_i[j] > minmax[j][1]:
                    minmax[j] = (minmax[j][0], data_i[j])
    f.close()
    return n, m, data, abs_result, minmax


def norm_coef(coef, mm):
    for i in range(len(coef) - 1):
        if mm[i][1] - mm[i][0] == 0:
            continue
        else:
            coef[i + 1] /= (mm[i][1] - mm[i][0])
    return coef


def printPlot(num):
    plt.plot(iters_nums, nrmse_iters)
    plt.xlabel('time')
    plt.ylabel('nrmse')
    plt.title("NRMSE" + num.__str__())
    plt.savefig("lab2_files/NRMS_TIME" + num.__str__() + ".png")
    plt.show()


def calc_coeffs(num):
    filename = "lab2_files/" + str(num) + ".txt"
    print(filename)
    n, m, data, abs_result, minmax = read_data_from_file(filename)
    normalize(data, minmax)
    y_min = min(abs_result)
    y_max = max(abs_result)
    abs_result = list(map(lambda x: (x - y_min) / (y_max - y_min), abs_result))

    res = do_algo(n, m, data, abs_result, minmax, num=num)
    printPlot(num)

    for i in range(len(res) - 1):
        if minmax[i][1] - minmax[i][0] == 0:
            continue
        else:
            res[i + 1] /= (minmax[i][1] - minmax[i][0])
    return res


def compare_results_2(norm_cooef, testSetX, testSetY):
    cur_res = list(map(lambda x: predict2(x, norm_cooef), testSetX))
    if err_func_const == ErrorFunc.SMAPE:
        return smape(testSetY, cur_res)
    else:
        return nrmse(testSetY, cur_res)


def get_inverse_model(data, y, p):
    y_np = np.array(y)
    F = np.array(data)
    U, s, VT = svd(F)
    d = p / s
    D = zeros(F.shape)
    D[:F.shape[1], :F.shape[1]] = diag(d)
    B = VT.T.dot(D.T).dot(U.T)
    # F_t = np.transpose(F)
    # tau = 0.1
    # m = len(data[0])
    # to_inv = (F_t @ F) + tau * np.eye(m, 1)
    # model_t = np.linalg.inv(to_inv) @ F_t @ y_np
    # model_t = np.linalg.pinv(F) @ y_np
    # print(B)
    # print(np.linalg.pinv(F))
    model_t = B @ y_np
    return model_t.T


def read_data_from_file_2(filename):
    f = open(filename, "r")
    n, m = map(int, f.readline().split())
    data = []
    abs_result = []
    for i in range(n):
        data_i = list([1])
        data_i += list(map(int, f.readline().split()))[:(m + 1)]
        y_i = data_i.pop()
        x_i = data_i
        data.append(x_i)
        abs_result.append(y_i)
    f.close()
    return n, m, data, abs_result


def calc_coeffs_sqrt(num, p = 1.0):
    filename = "lab2_files/" + str(num) + ".txt"
    print(filename)
    n, m, data, abs_result = read_data_from_file_2(filename)
    global coeffs
    coeffs = get_inverse_model(data, abs_result, p)

    filename_res = "lab2_files/" + str(num) + "_res.txt"
    _, _, data_res, abs_result_res = read_data_from_file_2(filename_res)
    predict_y = list(map(lambda row: predict_with_one(row), data_res))
    err_func = nrmse(abs_result_res, predict_y)
    # print("NRMS: " + err_func.__str__())
    return err_func


def calc_genetic(num, iterations, d):
    filename = "lab2_files/" + str(num) + ".txt"
    print(filename)
    n, m, data, abs_result = read_data_from_file_2(filename)

    def f(x):
        predict_y = list(map(lambda row: predict_with_one2(row, x), data))
        smape = nrmse(abs_result, predict_y)
        return smape

    ln = m + 1
    varbound = np.array([[-d, d]] * ln)

    # algorithm_param = {'max_num_iteration': None,
    algorithm_param = {'max_num_iteration': iterations,
                       'population_size': 100,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': None}

    # model = ga(function=f, dimension=ln, variable_type='real', variable_boundaries=varbound)
    model = ga(function=f, dimension=ln, variable_type='real', variable_boundaries=varbound,
               algorithm_parameters=algorithm_param)

    model.run()


def check_gen_algo(num, iterations):
    filename = "lab2_files/genetic_coeffs_file_" + str(num) + "_" + str(iterations)
    with open(filename, "r") as f:
        lines = f.readlines()
    read_coef = []
    for line in lines:
        read_coef += list(map(float, line.split()))
    global coeffs
    coeffs = read_coef
    filename_res = "lab2_files/" + str(num) + "_res.txt"
    _, _, data_res, abs_result_res = read_data_from_file_2(filename_res)
    predict_y = list(map(lambda row: predict_with_one(row), data_res))
    err_func = nrmse(abs_result_res, predict_y)
    print("NRMS: " + err_func.__str__())


def plot_genetic(num):
    plt.plot(const_lab2.iterations, const_lab2.nrms)
    plt.xlabel('iters')
    plt.ylabel('nrmse')
    plt.title("GENETIC" + num.__str__())
    plt.savefig("lab2_files/GENETIC" + num.__str__() + ".png")
    plt.show()


def plot_sqrt(num, x, y):
    print(x, y)
    plt.plot(x, y)
    plt.xlabel('iters')
    plt.ylabel('nrmse')
    plt.title("SQRT" + num.__str__())
    plt.savefig("lab2_files/SQRT" + num.__str__() + ".png")
    plt.show()


if __name__ == '__main__':
    # calc_genetic(1, 25, 5)
    check_gen_algo(1, 25) # NRMS: 0.11205628170006768
    # calc_genetic(3, 5, 5)
    # check_gen_algo(3, 5)  # NRMS: 0.019226423434811186
    # calc_genetic(3, 10, 5)
    # check_gen_algo(3, 10)  # NRMS: 0.01971670757176152
    # calc_genetic(3, 15, 5)
    # check_gen_algo(3, 15)  # NRMS: 0.01241168114603104
    # calc_genetic(3, 20, 5)
    # check_gen_algo(3, 20)  # NRMS: 0.012817323730590562
    # calc_genetic(3, 25, 5)
    # check_gen_algo(3, 25) # NRMS: 0.0058947722493932885
    # plot_genetic(1)

    # calc_coeffs_sqrt(0)
    # calc_coeffs_sqrt(1)  # NRMS: 6.332758614482057e-05
    # x_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # nrmse_arr = []
    #
    # for i in x_arr:
    #     res = calc_coeffs_sqrt(2, i)
    #     nrmse_arr.append(res)
    # plot_sqrt(2, x_arr, nrmse_arr)
    # calc_coeffs_sqrt(2)  # NRMS: 0.0008979517739236382
    # calc_coeffs_sqrt(4)  # NRMS: 0.00018362662430008092
    # calc_coeffs_sqrt(5)  # NRMS: 3.4376962605852404e-08
    # calc_coeffs_sqrt(6)  # NRMS: 0.0004930564826403967
    # calc_coeffs_sqrt(7)  # NRMS: 6.950050237928176e-07
    # calc_coeffs(1)
    # calc_coeffs(2)
    # calc_coeffs(3)
    # calc_coeffs(4)
    # calc_coeffs(5)
    # calc_coeffs(6)
    # calc_coeffs(7)


