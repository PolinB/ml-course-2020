from cf_tasks.g import build_tree_from_data, Cvert, Qvert
import matplotlib.pyplot as plt

max_h = 1000000


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
    # m_t, k_t, n_t, data_t, classes_t = read_file(train_file)
    good = 0
    for i in range(n_t):
        sug_class = find_class_i_element_in_vertex(tree, i, data_t)
        if sug_class == classes_t[i]:
            good += 1
    return good, n_t


def calc_accuracy_on_file_with_h_forest(num, h):
    train_file = f'lab5_files/DT_txt/{num if num >= 10 else "0" + str(num)}_train.txt'
    test_file = f'lab5_files/DT_txt/{num if num >= 10 else "0" + str(num)}_test.txt'
    m, k, n, data, classes = read_file(train_file)
    part_size = n // 7
    parts = []
    for i in range(6):
        parts.append(data[i * part_size:(i + 1) * part_size])
    parts.append(data[6 * part_size:])
    trees = []
    for i in range(6):
        block = parts[i % 7] + parts[(i + 1) % 7] + parts[(i + 2) % 7]
        tree = build_tree_from_data(m, k, n, h, block, classes)
        trees.append(tree)
    m_t, k_t, n_t, data_t, classes_t = read_file(test_file)
    good = 0
    data_t = data_t + data
    classes_t = classes_t + classes
    for i in range(len(data_t)):
        sug_classes = list(map(lambda tr: find_class_i_element_in_vertex(tr, i, data_t), trees))
        uniq_classes = dict.fromkeys(sug_classes, 0)
        for cl in sug_classes:
            uniq_classes[cl] += 1
        max_v = 0
        max_count = 0
        for cl in sug_classes:
            if uniq_classes[cl] > max_count:
                max_v = cl
                max_count = uniq_classes[cl]
        if max_v == classes_t[i]:
            good += 1
    print(f'file_{num}    {good / (n_t + n)}')
    return good, n_t + n


def find_optimal_h(num):
    # sug_h = [2, 3, 4, 5]
    sug_h = [10]
    for h in sug_h:
        good, n_t = calc_accuracy_on_file_with_h(num, h)
        print(f'dataset = {num} accuracy = {good / n_t} h = {h}')


def print_plot(num):
    sug_h = [i for i in range(2, 20)]
    accuracy = []
    for h in sug_h:
        good, n_t = calc_accuracy_on_file_with_h(num, h)
        accuracy.append(good / n_t)
        print(f'dataset = {num} accuracy = {good / n_t} h = {h}')
    plt.plot(sug_h, accuracy)
    plt.xlabel('h')
    plt.ylabel('accuracy')
    plt.title(f"file_{num}_test")
    plt.savefig(f"lab5_files/file_{num}_test1.png")
    plt.show()


if __name__ == '__main__':
    for i in range(1, 22):
        calc_accuracy_on_file_with_h_forest(i, max_h)
    # calc_accuracy_on_file_with_h(1, 4)
    # find_optimal_h(1)
    # ii = [21]
    # for i in ii:
    #     print_plot(i)
