from cf_tasks.f import result_from_f, result_from_f_with_p_1  # k, lambdas, alpha, n, tr_classes, tr_data_i, m, test_data_i
import os
import matplotlib.pyplot as plt

lambda_sp = 1
lambda_leg = 100
alpha = 0.0001


def read_file(filename, n_gram):
    c = 1 if "legit" in filename else 2
    all_words = []
    with open(filename) as f:
        subj_string = f.readline()
        subj_string.strip()
        subj_words = subj_string.split()
        all_words += subj_words[1:]
        f.readline()
        message = f.readline()
        mes_words = message.split()
        all_words += mes_words
    all_grams = []
    for i in range(len(all_words) - n_gram + 1):
        sub_arr = all_words[i:(i + n_gram)]
        gram = '_'.join(sub_arr)
        all_grams.append(gram)
    return c, all_grams


def data_from_dir(dir_name, n_gram):
    entries = os.listdir(dir_name)
    c = []
    all_words = []
    for file_name in entries:
        f_c, all_words_c = read_file(os.path.join(dir_name, file_name), n_gram)
        c.append(f_c)
        all_words.append(all_words_c)
    return c, all_words


def read_data_for_cross(num, n_gram):
    tr_classes = []
    tr_data = []
    test_classes = []
    test_data = []
    for i in range(10):
        dir_name = 'lab4_files/messages/part' + str(i + 1)
        if i + 1 == num:
            test_classes, test_data = data_from_dir(dir_name, n_gram)
        else:
            d_c, d_data = data_from_dir(dir_name, n_gram)
            tr_classes += d_c
            tr_data += d_data
    return tr_classes, tr_data, test_classes, test_data


def check_results_for_cross(num, n_gram, type = 1):
    tr_classes, tr_data, test_classes, test_data = read_data_for_cross(num, n_gram)
    if type == 1:
        result = result_from_f(2,
                               [lambda_leg, lambda_sp],
                               alpha,
                               len(tr_classes),
                               tr_classes,
                               tr_data,
                               len(test_classes),
                               test_data)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(result)):
            if result[i] == test_classes[i]:
                if result[i] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if result[i] == 1:
                    fp += 1
                else:
                    fn += 1
        return tp, tn, fp, fn
    else:
        result = result_from_f_with_p_1(2,
                               [lambda_leg, lambda_sp],
                               alpha,
                               len(tr_classes),
                               tr_classes,
                               tr_data,
                               len(test_classes),
                               test_data)
        sorted_res = sorted(result, key=lambda x: 1. - x[0])
        print_roc(sorted_res)


def print_roc(resTable):
    x_step, y_step = 1, 1
    prev_p = -1
    x_res, y_res = [0], [0]
    x_global, y_global = 0, 0
    for el in resTable:
        if prev_p != el[0]:
            prev_p = el[0]
            x_res.append(x_global)
            y_res.append(y_global)
        if el[1] == 2:
            x_global += x_step
        else:
            y_global += y_step
    x_res.append(x_global)
    y_res.append(y_global)

    plt.figure(figsize=(16, 9))
    plt.grid(linestyle='--')
    plt.plot(x_res, y_res)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def check_all_cross(n_gram, type = 1):
    if type == 1:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(9):
            i += 1
            tp_c, tn_c, fp_c, fn_c = check_results_for_cross(i, n_gram, type)
            tp += tp_c
            tn += tn_c
            fp += fp_c
            fn += fn_c
        print(fn)
        return (tp + tn) / (tp + tn + fp + fn)
    else:
        for i in range(9):
            i += 1
            check_results_for_cross(i, n_gram, type)


def check_all_cross_lambda(lam, n_gram):
    global lambda_leg
    lambda_leg = lam
    return check_all_cross(n_gram)


def print_for_choose_l_leg():
    x_val = [1] + [10 ** (2 * (i + 1)) for i in range(20)]
    y_val = list(map(lambda x: check_all_cross_lambda(x, 1), x_val))
    plt.plot(x_val, y_val)
    plt.xlabel('lambda_leg')
    plt.ylabel('accuracy')
    plt.title("Lambda_leg_plot")
    plt.savefig("lab4_files/Lambda_leg_plot5.png")
    plt.show()


if __name__ == '__main__':
    # lambdas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 8, 10, 50, 100, 150, 200, 500, 1000, 5000, 10000]
    # lambdas = [10000000000000000000000000000000000000000]
    # for i in lambdas:
    #     lambda_leg = i
    #     check_all_cross(1)
    # check_results_for_cross(10, 1, 2)
    # check_all_cross(1, 2)
    # print(check_all_cross(1))
    print_for_choose_l_leg()

