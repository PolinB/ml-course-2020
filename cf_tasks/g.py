tree_ind = 0


def read_input():
    m, k, h = map(int, input().split())
    n = int(input())
    data = []
    classes = []
    for _ in range(n):
        data_i = list(map(int, input().split()))[:(m + 1)]
        y_i = data_i.pop()
        x_i = data_i
        data.append(x_i)
        classes.append(y_i)
    print(m, k, h, n, data, classes)


class Qvert:
    def __init__(self, feat_ind, b, l_ind, r_ind):
        self.feat_ind = feat_ind
        self.b = b
        self.l_ind = l_ind
        self.r_ind = r_ind


class Cvert:
    def __init__(self, clazz):
        

if __name__ == '__main__':
    print('Hello')
    read_input()
