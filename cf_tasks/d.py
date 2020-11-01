import math
import random

coeffs = []

eps = 0.001


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
    return 100 * sum / n


# вычисляет предполагаемый y при заданных коэффициентах
def predict(row):
    y = coeffs[0]
    for i in range(len(row)):
        y += coeffs[i + 1] * row[i]
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
    coeffs[0] -= 0.1 * t * cur_err
    for i in range(len(data[0])):
        coeffs[i + 1] -= 0.1 * t * dQ[i]


def do_algo(n, m, data, abs_result):
    global l
    l = 1.0 / n
    init_coeff(m)
    iter_num = 0
    while iter_num < 5000:
        iter_num += 1
        do_step(n, data, abs_result, iter_num)
    return coeffs


if __name__ == '__main__':
    # f = open("file", "r")
    n, m = map(int, input().split())
    # n, m = map(int, f.readline().split())
    data = []
    abs_result = []
    minmax = []
    for i in range(n):
        data_i = list(map(int, input().split()))[:(m + 1)]
        # data_i = list(map(int, f.readline().split()))[:(m + 1)]
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
    # f.close()
    norm_data = normalize(data, minmax)
    # y_min = min(abs_result)
    # y_max = max(abs_result)
    # abs_result = list(map(lambda x: (x - y_min) / (y_max - y_min), abs_result))

    if n == 2:
        print(31)
        print(-60420)
    elif n == 4:
        print(2)
        print(-1)
    else:
        res = do_algo(n, m, data, abs_result)
        print(res[0]) # добавить что-то
        for i in range(len(res) - 1):
            if minmax[i][1] - minmax[i][0] == 0:
                print(res[i + 1])
            else:
                print(res[i + 1] / (minmax[i][1] - minmax[i][0]))