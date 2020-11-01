import random

n = 0
kernel_matrix = []
y = []
c = 0
iterations = 100
max_iterations = 5000
# iterations = 50
# max_iterations = 1000
eps = 0.00001


def calc_f(pos, alphas, b):
    sum_class = b
    for i in range(n):
        sum_class += (alphas[i] * y[i] * kernel_matrix[pos][i])
    return sum_class


def calc_e(pos, alphas, b):
    return calc_f(pos, alphas, b) - y[pos]


def calc_result():
    alphas = [0.0 for _ in range(n)]
    b = 0

    pairs = [i for i in range(n)]

    cur_iter = 0
    all_iter = 0
    while cur_iter < iterations and all_iter < max_iterations:
        # print(cur_iter)
        all_iter += 1
        random.shuffle(pairs)
        some_changed = False
        for i in range(n):
            e_i = calc_e(i, alphas, b)
            if (y[i] * e_i < -eps and alphas[i] < c) or (y[i] * e_i > eps and alphas[i] > 0.0):
                j = pairs[i]
                if i == j:
                    continue
                e_j = calc_e(j, alphas, b)
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]
                l = max(0., alphas[j] - alphas[i]) if y[i] != y[j] else \
                    max(0., alphas[j] + alphas[i] - c)
                h = min(float(c), float(c) + alphas[j] - alphas[i]) if y[i] != y[j] else \
                    min(float(c), alphas[j] + alphas[i])
                if abs(l - h) < eps:
                    continue

                nu = 2 * kernel_matrix[i][j] - kernel_matrix[i][i] - kernel_matrix[j][j]
                if nu >= 0:
                    continue

                alpha_j_new = alpha_j_old - y[j] * (e_i - e_j) / nu
                if alpha_j_new > h:
                    alpha_j_new = h
                elif alpha_j_new < l:
                    alpha_j_new = l
                if abs(alpha_j_new - alpha_j_old) < eps:
                    continue

                alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)
                alphas[j] = alpha_j_new
                alphas[i] = alpha_i_new

                b_1 = b - e_i - y[i] * (alpha_i_new - alpha_i_old) * kernel_matrix[i][i] \
                      - y[j] * (alpha_j_new - alpha_j_old) * kernel_matrix[i][j]
                b_2 = b - e_j - y[i] * (alpha_i_new - alpha_i_old) * kernel_matrix[i][j] \
                      - y[j] * (alpha_j_new - alpha_j_old) * kernel_matrix[j][j]
                if (0. < alpha_i_old) and (alpha_i_old < c):
                    b = b_1
                elif (0. < alpha_j_old) and (alpha_j_old < c):
                    b = b_2
                else:
                    b = (b_1 + b_2) / 2

                alphas[j] = alpha_j_new if (alpha_j_new > 0) else 0.
                alphas[i] = alpha_i_new if (alpha_i_new > 0) else 0.
                some_changed = True
        if not some_changed:
            cur_iter += 1
        else:
            cur_iter = 0
    return alphas, b


def calc_with_data(n_d, matrix_d, y_d, c_d):
    global n, y, kernel_matrix, c
    n = n_d
    y = y_d
    c = c_d
    kernel_matrix = matrix_d
    return calc_result()


if __name__ == '__main__':
    n = int(input())
    for _ in range(n):
        temp_data = list(map(int, input().split()))[:(n + 1)]
        y.append(temp_data.pop())
        kernel_matrix.append(temp_data)
    c = int(input())
    alphas, b = calc_result()
    for val in alphas:
        if val < 0:
            print(0)
        else:
            print(val)
    print(b)
