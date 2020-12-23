import re
import numpy as np
import sys


eps = 0.000001


def format_file(filename, outfilename):
    text = open(filename).read()
    text = text.lower()
    text = re.sub('-', ' ', text)
    new_text = ""
    for c in text:
        if c.isalpha() or c.isspace():
            new_text += c
    new_text = re.sub(' +', ' ', new_text)
    new_text = re.sub('\n+', '\n', new_text)
    new_text = re.sub('\n ', '\n', new_text)
    open(outfilename, "w").write(new_text)


# format_file("lab8_files/saltan.txt", "lab8_files/format_saltan.txt")
# filename = "lab8_files/format_saltan.txt"
format_file("lab8_files/elza.txt", "lab8_files/format_elza.txt")
filename = "lab8_files/format_elza.txt"
raw_text = open(filename).read()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
# print("Alphabet: ", chars)
# print("Total characters: ", n_chars)
print("Alphabet size: ", n_vocab)


# Для того, чтобы довавить явное разделение на строки как в примере с задания - можно считать, что после \n ничего не
# идет, т.е. вариантов \nab не будет
# n - размер окна
# k - размер суффикса начала
# m - количество генерируемых символов
def get_windows(n, k, m):
    print(f'n = {n}, k = {k}, m = {m}')
    w = set()
    for i in range(len(raw_text) - n + 1):
        w.add(raw_text[i:i + n])
    # print("All windows: ", w)
    print("Number of windows: ", len(w))

    w_to_int = dict((c, i) for i, c in enumerate(w))
    int_to_w = dict((i, c) for i, c in enumerate(w))
    matrix = [[0 for _ in range(len(w))] for _ in range(len(w))]
    for i in range(len(raw_text) - n):
        cur_w = raw_text[i:i + n]
        next_w = raw_text[(i + 1):(i + n + 1)]
        matrix[w_to_int[cur_w]][w_to_int[next_w]] += 1
    matrix = np.array(norm_matrix(matrix))
    # print(np.array(matrix))

    lines = raw_text.split('\n')
    start = np.random.randint(0, len(lines) - 1)
    if len(lines[start]) < k:
        print(f'So big {k} for "{lines[start]}"')
    elif k < n:
        print(f'Must k > m"')
    else:
        # prefix = lines[start][0:k]
        prefix = "А за окном"
        print(f'Start:\n{prefix}')
        print("Gen:")
        start_window = prefix[len(prefix) - n:]
        for i in range(m):
            sug_next_pos = get_all_by_max(matrix[w_to_int[start_window]])
            if len(sug_next_pos) == 0:
                print(f'End after {i} symbols')
                break
            elif len(sug_next_pos) == 1:
                ind = 0
            else:
                ind = np.random.randint(0, len(sug_next_pos) - 1)
            next_pos = sug_next_pos[ind]
            # print(next_pos)
            start_window = int_to_w[next_pos]
            sys.stdout.write(start_window[len(start_window) - 1])



def get_all_by_max(data):
    max_v = max(data)
    res = []
    for i in range(len(data)):
        if abs(data[i] - max_v) < eps:
            res.append(i)
    return res


def norm_matrix(matrix):
    new_matrix = []
    for row in matrix:
        sum_v = sum(row)
        if sum_v != 0:
            new_row = list(map(lambda x: x / sum_v, row))
            new_matrix.append(new_row)
        else:
            new_matrix.append(row)
    return new_matrix


get_windows(5, 10, 200)

