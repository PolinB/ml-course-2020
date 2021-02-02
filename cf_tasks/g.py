import math
inf = 100000000
gl_ind = 0
data, classes = [], []
is_gini = False
tree_vert = []


def read_input():
    global m, k, h, n
    m, k, h = map(int, input().split())
    n = int(input())
    global data, classes
    for _ in range(n):
        data_i = list(map(int, input().split()))[:(m + 1)]
        y_i = data_i.pop()
        x_i = data_i
        data.append(x_i)
        classes.append(y_i - 1)
    return m, k, h, n


class Qvert:
    def __init__(self, feat_ind, b, l_son_ind, r_son_ind, ind, my_ind):
        self.feat_ind = feat_ind
        self.b = b
        self.my_ind = my_ind
        self.l_son_ind = l_son_ind
        self.r_son_ind = r_son_ind
        self.ind = ind


class Cvert:
    def __init__(self, clazz, ind, my_ind):
        self.clazz = clazz
        self.ind = ind
        self.my_ind = my_ind


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


def calc_entropy(feat_num, left_size, all_feat_num, all_size):
    sum_val_left = 0
    for i in range(k):
        num = feat_num[i]
        if num <= 0:
            continue
        p = num / left_size
        if p == 0:
            continue
        sum_val_left -= (p * math.log(p))

    sum_val_right = 0
    for i in range(k):
        num = all_feat_num[i] - feat_num[i]
        if num <= 0:
            continue
        p = num / (all_size - left_size)
        if p == 0:
            continue
        sum_val_right -= (p * math.log(p))
    return sum_val_left * left_size + sum_val_right * (all_size - left_size)


def calc_gini(feat_num, left_size, all_feat_num, all_size):
    sum_val_left = 0
    for i in range(k):
        num = feat_num[i]
        if num <= 0:
            continue
        p = num / left_size
        if p == 0:
            continue
        sum_val_left += (p * p)

    sum_val_right = 0
    for i in range(k):
        num = all_feat_num[i] - feat_num[i]
        if num <= 0:
            continue
        p = num / (all_size - left_size)
        if p == 0:
            continue
        sum_val_right += (p * p)
    return (1 - sum_val_left) * left_size + (1 - sum_val_right) * (all_size - left_size)


def part_array(cur_indexes, m, k):
    best_score = inf
    ind = -1
    best_b_val_all = -1
    best_l_i = []
    best_r_i = []
    left_best_ind = -1
    all_class_map = {i: 0 for i in range(k)}
    all_cnt = 0
    for j in cur_indexes:
        all_class_map[classes[j]] += 1
        all_cnt += 1

    for i in range(m):
        b_val_ind = []
        for j in cur_indexes:
            b_val = data[j][i]
            b_val_ind.append((b_val, j))
        b_val_ind = sorted(b_val_ind)

        left_ind = 0
        left_all_cnt = 0
        left_class_map = {i: 0 for i in range(k)}
        while left_ind < len(b_val_ind):
            cur_b = b_val_ind[left_ind][0]
            while left_ind < len(b_val_ind) and cur_b == b_val_ind[left_ind][0]:
                left_all_cnt += 1
                left_class_map[classes[b_val_ind[left_ind][1]]] += 1
                left_ind += 1

            if is_gini:
                score = calc_gini(left_class_map, left_all_cnt, all_class_map, all_cnt)
            else:
                score = calc_entropy(left_class_map, left_all_cnt, all_class_map, all_cnt)

            if best_score > score:
                best_score = score
                left_best_ind = left_ind
                ind = i
                if left_ind < len(b_val_ind):
                    best_b_val_all = b_val_ind[left_ind][0]
    if ind != -1:
        b_val_ind = []
        for j in cur_indexes:
            b_val = data[j][ind]
            b_val_ind.append((b_val, j))
        b_val_ind = sorted(b_val_ind)
        best_l_i = [ind for b, ind in b_val_ind[:left_best_ind]]
        best_r_i = [ind for b, ind in b_val_ind[left_best_ind:]]
    # print(ind, best_b_val_all, best_l_i, best_r_i)
    return ind, best_b_val_all, best_l_i, best_r_i


def find_max_class_by_indexes(k, indexes, classes):
    feat_num = {i: 0 for i in range(k)}
    all_one_class = True
    for i in indexes:
        feat_num[classes[i]] += 1
        if classes[i] != classes[indexes[0]]:
            all_one_class = False
    max_class = -1
    max_class_val = 0
    for i in range(k):
        if feat_num[i] > max_class_val:
            max_class_val = feat_num[i]
            max_class = i
    return max_class, all_one_class


def build_tree_on_level(m, k, h, n, indexes, level, my_ind):
    global gl_ind
    gl_ind += 1
    max_class, all_one_class = find_max_class_by_indexes(k, indexes, classes)
    if level == 0 or all_one_class:
        tree_vert[my_ind] = Cvert(max_class, gl_ind, my_ind)
        return

    ind, best_b_val_all, best_l_i, best_r_i = part_array(indexes, m, k)
    if ind == -1:
        tree_vert[my_ind] = Cvert(max_class, gl_ind, my_ind)
        return
    else:
        cur_ind = gl_ind
        tree_vert.append(None)
        tree_vert.append(None)
        left_ind = len(tree_vert) - 2
        right_ind = len(tree_vert) - 1
        tree_vert[my_ind] = Qvert(ind, best_b_val_all, left_ind, right_ind, cur_ind, my_ind)
        build_tree_on_level(m, k, h, n, best_l_i, level - 1, left_ind)
        build_tree_on_level(m, k, h, n, best_r_i, level - 1, right_ind)


def build_arr_tree(vertex):
    if isinstance(vertex, Cvert):
        return [(vertex.ind, f'C {vertex.clazz + 1}')]
    else:
        return [(vertex.ind, f'Q {vertex.feat_ind + 1} {vertex.b} {tree_vert[vertex.l_son_ind].ind} {tree_vert[vertex.r_son_ind].ind}')] + \
               build_arr_tree(tree_vert[vertex.l_son_ind]) + build_arr_tree(tree_vert[vertex.r_son_ind])


def build_tree(m, k, h, n):
    indexes = [i for i in range(n)]
    tree_vert.append(None)
    build_tree_on_level(m, k, h, n, indexes, h, 0)
    tree = build_arr_tree(tree_vert[0])
    tree.sort()
    print(len(tree))
    for i in range(len(tree)):
        print(tree[i][1])


if __name__ == '__main__':
    m, k, h, n = read_input()
    if n > 50:
        is_gini = True
    build_tree(m, k, h, n)
