import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from math import exp, log
import matplotlib.pyplot as plt
eps = 0.00000000001


def find_e(weights, y_predict, y):
    sum_e = 0
    for i in range(len(y)):
        if y_predict[i] != y[i]:
            sum_e += weights[i]
    return sum_e


def calc_accuracy(y_predict, y):
    good = 0
    for i in range(len(y_predict)):
        if y[i] == y_predict[i]:
            good += 1
    return good / len(y)


def ada_boost_initializer(X, y, t):
    trees = []
    alphas = []
    size = len(y)
    weights = [1 / size for _ in range(size)]
    for _ in range(t):
        tree = DecisionTreeClassifier(max_depth=3).fit(X, y, sample_weight=np.array(weights, copy=True))
        trees.append(tree)
        y_predict = tree.predict(X)
        sum_e = find_e(weights, y_predict, y)
        if abs(sum_e) < eps:
            trees = [tree]
            alphas = [1]
            break
        if abs(1 - sum_e) < eps:
            trees = [tree]
            alphas = [-1]
            break
        a = 0.5 * log((1 - sum_e) / sum_e)
        alphas.append(a)
        for i in range(size):
            weights[i] *= exp(-a * y[i] * y_predict[i])
        sum_w = sum(weights)
        weights = list(map(lambda x: x / sum_w, weights))
    return trees, alphas


def read_data(filename):
    # filename = "lab3_svm_files/chips.csv"
    # filename = "lab3_svm_files/geyser.csv"
    target_name = "class"
    dataset = pd.read_csv(filename)
    y = dataset[target_name].values.tolist()
    y = list(map(lambda y_i: 1 if y_i == 'P' else -1, y))
    dataset = dataset.drop(columns=target_name)
    return np.array(dataset), np.array(y)


def predict(trees, alphas, X):  # X - 2D array
    sum_vals = []
    for tree in trees:
        sum_vals.append(tree.predict(X))
    results = []
    for i in range(len(X)):
        sum_val = 0
        for j in range(len(trees)):
            sum_val += (alphas[j] * sum_vals[j][i])
        results.append(np.sign(sum_val))
    return results


def run_algo_and_check(X, y, X_test, y_test, t):
    trees, alphas = ada_boost_initializer(X, y, t)
    y_predict = predict(trees, alphas, X_test)
    return calc_accuracy(y_predict, y_test)


def print_accuracy_plot(filename, max_step, plot_name):
    X, y = read_data(filename)
    x_arr = [i for i in range(max_step + 1)]
    y_arr = list(map(lambda i: run_algo_and_check(X, y, X, y, i), x_arr))
    print(y_arr)
    plt.plot(x_arr, y_arr)
    plt.xlabel('boosting steps')
    plt.ylabel('accuracy')
    plt.title(plot_name)
    plt.savefig(f"lab6_files/{plot_name}.png")
    plt.show()


def show_plot(filename, boosting_step, plot_name):
    X, y = read_data(filename)
    trees, alphas = ada_boost_initializer(X, y, boosting_step)
    dataset = pd.read_csv(filename)
    xx = dataset['x'].values.tolist()
    yy = dataset['y'].values.tolist()
    colors = list(map(lambda x: 'blue' if x == 1 else 'red', y))

    h = 0.1
    x_min, x_max = min(xx) - 1, max(xx) + 1
    y_min, y_max = min(yy) - 1, max(yy) + 1
    xxx, yyy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
    d = np.c_[xxx.ravel(), yyy.ravel()]
    predicted_y = predict(trees, alphas, d)

    z = (np.array(predicted_y)).reshape(xxx.shape)
    plt.contourf(xxx, yyy, z, alpha=1)
    plt.scatter(xx, yy, c=colors)
    plt.title(plot_name)
    plt.savefig(f"lab6_files/{plot_name}.png")
    plt.show()


if __name__ == '__main__':
    # print_accuracy_plot("lab3_svm_files/chips.csv", 55, "Chips55")
    # print_accuracy_plot("lab3_svm_files/geyser.csv", 55, "Geyser55")
    boost_iters = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    for ite in boost_iters:
        show_plot("lab3_svm_files/chips.csv", ite, f"ChipsR{ite}")
        show_plot("lab3_svm_files/geyser.csv", ite, f"GeyserR{ite}")
