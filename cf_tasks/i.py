import numpy as np
import math

nodes = []


class Var:
    def __init__(self, r, c):
        self.r = r
        self.c = c
        self.matrix = np.array([[0. for _ in range(c)] for _ in range(r)])
        self.d_matrix = np.array([[0. for _ in range(c)] for _ in range(r)])

    def init_matrix(self, m):
        self.matrix = m

    def init_d_matrix(self):
        pass

    def calc(self):
        pass

    def step_d(self):
        pass


class Mul:
    def __init__(self, ind1, ind2):
        self.ind1 = ind1 - 1
        self.ind2 = ind2 - 1
        self.matrix = np.array([])
        self.d_matrix = np.array([])

    def calc(self):
        res = nodes[self.ind1].matrix
        self.matrix = res.dot(nodes[self.ind2].matrix)

    def init_d_matrix(self):
        self.d_matrix = np.array([[0. for _ in range(len(self.matrix[0]))] for _ in range(len(self.matrix))])

    def step_d(self):
        res1 = self.d_matrix.dot(nodes[self.ind2].matrix.transpose())
        res2 = nodes[self.ind1].matrix.transpose().dot(self.d_matrix)
        nodes[self.ind1].d_matrix = nodes[self.ind1].d_matrix + res1
        nodes[self.ind2].d_matrix = nodes[self.ind2].d_matrix + res2


class Sum:
    def __init__(self, len_a, arr):
        self.len_a = len_a
        self.arr = list(map(lambda x: x - 1, arr))
        self.matrix = np.array([])
        self.d_matrix = np.array([])

    def calc(self):
        res = nodes[self.arr[0]].matrix
        for i in range(1, self.len_a):
            res = res + nodes[self.arr[i]].matrix
        self.matrix = np.array(res)

    def init_d_matrix(self):
        self.d_matrix = np.array([[0. for _ in range(len(self.matrix[0]))] for _ in range(len(self.matrix))])

    def step_d(self):
        for i in range(self.len_a):
            nodes[self.arr[i]].d_matrix = nodes[self.arr[i]].d_matrix + self.d_matrix


class ReLU:
    def __init__(self, a, ind):
        self.a = 1.0 / a
        self.ind = ind - 1
        self.matrix = np.array([])
        self.d_matrix = np.array([])

    def calc(self):
        res = []
        m = nodes[self.ind].matrix
        for i in range(len(m)):
            res_r = []
            for j in range(len(m[i])):
                res_r.append(self.relu(m[i][j]))
            res.append(res_r)
        self.matrix = np.array(res)

    def relu(self, x):
        if x > 0:
            return x
        else:
            return self.a * x

    def d_relu(self, x):
        if x >= 0:
            return 1
        else:
            return self.a

    def init_d_matrix(self):
        self.d_matrix = np.array([[0. for _ in range(len(self.matrix[0]))] for _ in range(len(self.matrix))])

    def step_d(self):
        res = []
        for i in range(len(self.matrix)):
            res_r = []
            for j in range(len(self.matrix[0])):
                res_r.append(self.d_relu(self.matrix[i][j]) * self.d_matrix[i][j])
            res.append(res_r)
        nodes[self.ind].d_matrix = nodes[self.ind].d_matrix + np.array(res)


class Tnh:
    def __init__(self, ind):
        self.ind = ind - 1
        self.matrix = np.array([])
        self.d_matrix = np.array([])

    def init_d_matrix(self):
        self.d_matrix = np.zeros(shape=self.matrix.shape)

    def calc(self):
        res = []
        for i in range(len(nodes[self.ind].matrix)):
            res_r = []
            for j in range(len(nodes[self.ind].matrix[0])):
                res_r.append(math.tanh(nodes[self.ind].matrix[i][j]))
            res.append(res_r)
        self.matrix = np.array(res)

    def step_d(self):
        res = []
        for i in range(len(self.matrix)):
            res_r = []
            for j in range(len(self.matrix[0])):
                res_r.append((1.0 - self.matrix[i][j] ** 2) * self.d_matrix[i][j])
            res.append(res_r)
        nodes[self.ind].d_matrix += np.array(res)


class Had:
    def __init__(self, len_a, arr):
        self.len_a = len_a
        self.arr = list(map(lambda x: x - 1, arr))
        self.matrix = np.array([])
        self.d_matrix = np.array([])

    def calc(self):
        res = []
        for i in range(len(nodes[self.arr[0]].matrix)):
            res_r = []
            for j in range(len(nodes[self.arr[0]].matrix[0])):
                v = 1
                for k in range(self.len_a):
                    v *= nodes[self.arr[k]].matrix[i][j]
                res_r.append(v)
            res.append(res_r)
        self.matrix = np.array(res)

    def init_d_matrix(self):
        self.d_matrix = np.array([[0 for _ in range(len(self.matrix[0]))] for _ in range(len(self.matrix))])

    def step_d(self):
        res_t = [[] for _ in range(self.len_a)]
        for t in range(self.len_a):
            for i in range(len(nodes[self.arr[0]].matrix)):
                res_r = []
                for j in range(len(nodes[self.arr[0]].matrix[0])):
                    v = 1
                    for k in range(self.len_a):
                        if k != t:
                            v *= nodes[self.arr[k]].matrix[i][j]
                        else:
                            v *= self.d_matrix[i][j]
                    res_r.append(v)
                res_t[t].append(res_r)
        for t in range(self.len_a):
            nodes[self.arr[t]].d_matrix = nodes[self.arr[t]].d_matrix + np.array(res_t[t])


def read_matrix(r, c):
    res = []
    for _ in range(r):
        cc = list(map(float, input().split(" ")))
        res.append(cc)
    return np.array(res)


def print_m(m):
    for r in m:
        for v in r:
            print(v, end=" ")
        print()


if __name__ == '__main__':
    N, M, K = map(int, input().split(" "))
    for _ in range(N):
        args_i = input().split(" ")
        type_i = args_i[0]
        if type_i == 'var':
            nodes.append(Var(int(args_i[1]), int(args_i[2])))
        elif type_i == 'mul':
            nodes.append(Mul(int(args_i[1]), int(args_i[2])))
        elif type_i == 'sum':
            nodes.append(Sum(int(args_i[1]), list(map(int, args_i[2:]))))
        elif type_i == 'rlu':
            nodes.append(ReLU(int(args_i[1]), int(args_i[2])))
        elif type_i == 'tnh':
            nodes.append(Tnh(int(args_i[1])))
        elif type_i == 'had':
            nodes.append(Had(int(args_i[1]), list(map(int, args_i[2:]))))

    for m_ind in range(M):
        nodes[m_ind].init_matrix(read_matrix(nodes[m_ind].r, nodes[m_ind].c))

    for cur_ind in range(M, N):
        nodes[cur_ind].calc()

    for res_ind in range(N - K, N):
        print_m(nodes[res_ind].matrix)

    for i in range(N):
        nodes[i].init_d_matrix()

    for res_ind in range(N - K, N):
        read_d_matrix = []
        for i in range(len(nodes[res_ind].matrix)):
            row = list(map(float, input().split(" ")))
            read_d_matrix.append(row)
        nodes[res_ind].d_matrix = np.array(read_d_matrix)

    for i in reversed(range(N)):
        nodes[i].step_d()

    for i in range(M):
        # print(i)
        print_m(nodes[i].d_matrix)
