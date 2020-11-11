from cf_tasks.g import build_tree_from_data, Cvert, Qvert


def read_file(filename):
    with open(filename) as f:
        m, k = map(int, f.readline().split())
        n = int(f.readline())
        data = []
        classes = []
        for _ in range(n):
            data_i = list(map(int, f.readline().split()))[:(m + 1)]
            y_i = data_i.pop()
            x_i = data_i
            data.append(x_i)
            classes.append(y_i)
        return m, k, n, data, classes


def find_class_i_element_in_vertex(vertex, ind, data):
    if isinstance(vertex, Cvert):
        return vertex.clazz
    elif isinstance(vertex, Qvert):
        if data[ind][vertex.feat_ind] < vertex.b:
            return find_class_i_element_in_vertex(vertex.l_son, ind, data)
        else:
            return find_class_i_element_in_vertex(vertex.r_son, ind, data)


def calc_accuracy_on_file_with_h(num, h):
    train_file = f'lab5_files/DT_txt/{num if num >= 10 else "0" + str(num)}_train.txt'
    test_file = f'lab5_files/DT_txt/{num if num >= 10 else "0" + str(num)}_test.txt'
    m, k, n, data, classes = read_file(train_file)
    tree = build_tree_from_data(m, k, n, h, data, classes)
    m_t, k_t, n_t, data_t, classes_t = read_file(test_file)
    good = 0
    for i in range(n_t):
        sug_class = find_class_i_element_in_vertex(tree, i, data_t)
        if sug_class == classes_t[i]:
            good += 1
    return good, n_t


def find_optimal_h(num):
    sug_h = [2, 3, 4, 5]
    for h in sug_h:
        good, n_t = calc_accuracy_on_file_with_h(num, h)
        print(f'dataset = {num} accuracy = {good / n_t} h = {h}')


if __name__ == '__main__':
    # calc_accuracy_on_file_with_h(1, 4)
    # find_optimal_h(1)
    for i in range(2, 22):
        find_optimal_h(i)
