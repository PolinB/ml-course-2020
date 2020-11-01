import math
distance_type = ""
kernel_type = ""
window_type = ""
n, m, hk = 0, 0, 0
data = []
q = []


def calc_dist(v1, v2):
    if distance_type == "manhattan":
        result = 0
        for (x1, x2) in zip(v1, v2):
            result += abs(x2 - x1)
        return result
    elif distance_type == "euclidean":
        result = 0
        for (x1, x2) in zip(v1, v2):
            result += ((x2 - x1) ** 2)
        return math.sqrt(result)
    elif distance_type == "chebyshev":
        result = -1
        for (x1, x2) in zip(v1, v2):
            result = max(result, abs(x2 - x1))
        return result


def calc_kernel(u):
    if kernel_type == "uniform":
        if u >= 1:
            return 0
        return 1 / 2
    elif kernel_type == "triangular":
        if u >= 1:
            return 0
        return 1 - u
    elif kernel_type == "epanechnikov":
        if u >= 1:
            return 0
        return (3 / 4) * (1 - u ** 2)
    elif kernel_type == "quartic":
        if u >= 1:
            return 0
        return (15 / 16) * ((1 - u ** 2) ** 2)
    elif kernel_type == "triweight":
        if u >= 1:
            return 0
        return (35 / 32) * ((1 - u ** 2) ** 3)
    elif kernel_type == "tricube":
        if u >= 1:
            return 0
        return (70 / 81) * ((1 - u ** 3) ** 3)
    elif kernel_type == "gaussian":
        return (1 / (math.sqrt(2 * math.pi))) * math.exp((-1 / 2) * (u ** 2))
    elif kernel_type == "cosine":
        if u >= 1:
            return 0
        return (math.pi / 4) * math.cos((math.pi * u) / 2)
    elif kernel_type == "logistic":
        return 1 / (math.exp(u) + 2 + math.exp(-u))
    elif kernel_type == "sigmoid":
        return (2 / math.pi) * (1 / (math.exp(u) + math.exp(-u)))


def calc_h(all_dist):
    if window_type == "fixed":
        return float(hk)
    else:
        temp_dist = all_dist.copy()
        temp_dist.sort()
        if hk >= len(temp_dist):
            return temp_dist.pop() + 0.0000000000001
        return temp_dist[hk]


def calc_all_dist():
    all_dist = []
    for x in data:
        all_dist.append(float(calc_dist(x[0], q)))
    return all_dist


def calc_result():
    num = 0.0
    denom = 0.0
    all_dists = calc_all_dist()
    h = calc_h(all_dists)
    eq_q = list(filter(lambda el: el == 0, all_dists))
    if h == 0:
        if len(eq_q) != 0:
            for x in data:
                dist = calc_dist(x[0], q)
                if dist == 0:
                    num += x[1]
                    denom += 1
        else:
            for x in data:
                num += x[1]
                denom += 1
    else:
        for x in data:
            dist = calc_dist(x[0], q)
            kern_val = calc_kernel(dist / h)
            num += (x[1] * kern_val)
            denom += kern_val
    if denom == 0:
        for x in data:
            num += x[1]
            denom += 1
    return num / denom


def knn(data_, q_, dist_type, ker_type, win_type, hk_):
    global n, m, data, q, distance_type, kernel_type, window_type, hk
    n, m = len(data_), len(q_)
    data = data_
    q = q_
    distance_type = dist_type
    kernel_type = ker_type
    window_type = win_type
    hk = int(hk_)
    return calc_result()


if __name__ == '__main__':
    n, m = map(int, input().split())
    for i in range(0, n):
        data_i = list(map(int, input().split()))[:(m + 1)]
        y_i = data_i.pop()
        x_i = data_i
        data.append((x_i, y_i))
    q = list(map(int, input().split()))[:m]
    distance_type = input()
    kernel_type = input()
    window_type = input()
    hk = int(input())
    res = calc_result()
    print(res)
