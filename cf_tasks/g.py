import math
inf = 100000000
gl_ind = 0

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
    return m, k, h, n, data, classes


class Qvert:
    def __init__(self, feat_ind, b, l_son, r_son, ind):
        self.feat_ind = feat_ind
        self.b = b
        self.l_son = l_son
        self.r_son = r_son
        self.ind = ind


class Cvert:
    def __init__(self, clazz, ind):
        self.clazz = clazz
        self.ind = ind


def init_feat_num(features_i, value, k):
    left_feat_num = {i + 1: 0 for i in range(k)}
    right_feat_num = {i + 1: 0 for i in range(k)}
    left_s = 0
    right_s = 0
    for i in range(len(features_i)):
        if features_i[i][0] < value:
            left_feat_num[features_i[i][1]] += 1
            left_s += 1
        else:
            right_feat_num[features_i[i][1]] += 1
            right_s += 1
    return left_feat_num, left_s, right_feat_num, right_s


def calc_entropy(k, feat_num, size):
    sum_val = 0
    for i in range(k):
        p = feat_num[i + 1] / size
        if p == 0:
            continue
        sum_val -= (p * math.log(p))
    # print(f'sum_val {sum_val}')
    return sum_val


def part_by_i(cur_indexes, m, k, h, n, data, classes, feat_i):
    features_i = []
    unique_val = set()
    for i in cur_indexes:
        features_i.append((data[i][feat_i], classes[i], i))
        unique_val.add(data[i][feat_i])
    features_i.sort()

    if len(unique_val) <= 1:
        return [], [], inf, -1

    best_b_val = -1
    min_entr = inf
    for cur_val in unique_val:
        left_feat_num, left_s, right_feat_num, right_s = init_feat_num(features_i, cur_val, k)
        if left_s == 0 or right_s == 0:
            continue
        cur_entr = calc_entropy(k, left_feat_num, left_s) * left_s + calc_entropy(k, right_feat_num, right_s) * right_s
        if cur_entr < min_entr:
            min_entr = cur_entr
            best_b_val = cur_val

    left_indexes = []
    right_indexes = []
    for i in cur_indexes:
        if data[i][feat_i] < best_b_val:
            left_indexes.append(i)
        else:
            right_indexes.append(i)
    return left_indexes, right_indexes, min_entr, best_b_val


def part_array(cur_indexes, m, k, h, n, data, classes):
    min_entr_all = inf
    ind = -1
    best_b_val_all = -1
    best_l_i = []
    best_r_i = []
    for i in range(m):
        left_indexes, right_indexes, min_entr, best_b_val = part_by_i(cur_indexes, m, k, h, n, data, classes, i)
        if min_entr < min_entr_all:
            min_entr_all = min_entr
            ind = i
            best_b_val_all = best_b_val
            best_l_i = left_indexes
            best_r_i = right_indexes
    return ind, best_b_val_all, best_l_i, best_r_i


def find_max_class_by_indexes(k, indexes, classes):
    feat_num = {i + 1: 0 for i in range(k)}
    all_one_class = True
    for i in indexes:
        feat_num[classes[i]] += 1
        if classes[i] != classes[indexes[0]]:
            all_one_class = False
    max_class = -1
    max_class_val = 0
    for i in range(k):
        if feat_num[i + 1] > max_class_val:
            max_class_val = feat_num[i + 1]
            max_class = i + 1
    return max_class, all_one_class


def build_tree_on_level(m, k, h, n, data, classes, indexes, level):
    print(f'level {level}')
    global gl_ind
    gl_ind += 1
    max_class, all_one_class = find_max_class_by_indexes(k, indexes, classes)
    if level == 0 or all_one_class:
        return Cvert(max_class, gl_ind)
    ind, best_b_val_all, best_l_i, best_r_i = part_array(indexes, m, k, h, n, data, classes)
    if ind == -1:
        return Cvert(max_class, gl_ind)
    else:
        cur_ind = gl_ind
        l_son = build_tree_on_level(m, k, h, n, data, classes, best_l_i, level - 1)
        r_son = build_tree_on_level(m, k, h, n, data, classes, best_r_i, level - 1)
        return Qvert(ind, best_b_val_all, l_son, r_son, cur_ind)


def build_arr_tree(vertex):
    if isinstance(vertex, Cvert):
        return [(vertex.ind, f'C {vertex.clazz}')]
    else:
        return [(vertex.ind, f'Q {vertex.feat_ind + 1} {vertex.b} {vertex.l_son.ind} {vertex.r_son.ind}')] + \
               build_arr_tree(vertex.l_son) + build_arr_tree(vertex.r_son)


def build_tree(m, k, h, n, data, classes):
    indexes = [i for i in range(n)]
    vert_tree = build_tree_on_level(m, k, h, n, data, classes, indexes, h)
    tree = build_arr_tree(vert_tree)
    tree.sort()
    print(len(tree))
    for i in range(len(tree)):
        print(tree[i][1])


def build_tree_from_data(m, k, n, h, data, classes):
    indexes = [i for i in range(n)]
    vert_tree = build_tree_on_level(m, k, h, n, data, classes, indexes, h)
    return vert_tree


if __name__ == '__main__':
    m, k, h, n, data, classes = read_input()
    build_tree(m, k, h, n, data, classes)
