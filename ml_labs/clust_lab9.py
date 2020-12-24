import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score


def calc_dist(a, b):
    return np.linalg.norm(a - b)


def dataset_minmax(dataset):
    minmax = list()
    data_len = len(dataset[0])
    for i in range(data_len):
        value_min = dataset[:, i].min()
        value_max = dataset[:, i].max()
        minmax.append([value_min, value_max])
    return minmax


def normalize(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return dataset


def show_clusters(labels, title):
    unique_labels = np.unique(labels)
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        cur_xs = X_reduced[labels == label, 0]
        cur_ys = X_reduced[labels == label, 1]
        plt.scatter(cur_xs, cur_ys, color=colors[i], label=label)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def init_begin_centroids(clusters):
    rows, columns = X.shape
    centroids = [X[np.random.randint(0, rows)]]
    for _ in range(clusters - 1):
        min_dist_to_c = list()
        for x in X:
            min_dist_to_c.append(find_nearest_centroid(centroids, x)[1])
        sorted_min_x_distances_to_centroids = np.argsort(min_dist_to_c)
        sum_min_distances = np.sum(min_dist_to_c)
        p = random.random()
        while p == 0:
            p = random.random()
        cur_dist = p * sum_min_distances
        cur_id = 0
        cur_sum = min_dist_to_c[sorted_min_x_distances_to_centroids[cur_id]]
        while cur_sum < cur_dist:
            cur_id += 1
            cur_sum += min_dist_to_c[sorted_min_x_distances_to_centroids[cur_id]]
        centroids.append(X[sorted_min_x_distances_to_centroids[cur_id]])
    return centroids


def find_nearest_centroid(centroids, x):
    distances = list()
    for c in centroids:
        distances.append(calc_dist(c, x))
    idx = np.argmin(distances)
    return idx, distances[idx]


def k_means(clusters):
    centroids = init_begin_centroids(clusters)
    rows, columns = X.shape
    for _ in range(limit_iter):
        clusters_size, clusters_x_sum = np.zeros(clusters), np.zeros((clusters, columns))
        for x in X:
            c_id = find_nearest_centroid(centroids, x)[0]
            clusters_size[c_id] += 1
            clusters_x_sum[c_id] += x
        new_centroids = np.copy(centroids)
        for i in range(clusters):
            if clusters_size[i] != 0:
                new_centroids[i] = clusters_x_sum[i] / clusters_size[i]
        diff = list(map(lambda p: np.linalg.norm(p), centroids - new_centroids))
        if np.max(diff) <= eps:
            break
        centroids = new_centroids
    return centroids


def predict(centroids):
    return list(map(lambda x: find_nearest_centroid(centroids, x)[0], X))


def rand_metric(y_predicted):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y)):
        for j in range(i + 1, len(y_predicted)):
            y_i_v = y[i] - 1
            y_j_v = y[j] - 1
            y_i_pv = y_predicted[i]
            y_j_pv = y_predicted[j]
            if y_i_v == y_j_v and y_i_pv == y_j_pv:
                tp += 1
            elif y_i_v != y_j_v and y_i_pv == y_j_pv:
                tn += 1
            elif y_i_v == y_j_v and y_i_pv != y_j_pv:
                fp += 1
            elif y_i_v != y_j_v and y_i_pv != y_j_pv:
                fn += 1
    return (tp + fn) / (tp + tn + fp + fn)


def inner_calinski_harabasz(y_predicted, clusters, centroids):
    N = len(y_predicted)
    K = clusters
    uniq_clusters = np.unique(y_predicted)
    num, denom = 0, 0
    for i in range(len(uniq_clusters)):
        label = uniq_clusters[i]
        indexes = np.where(y_predicted == label)
        x_classes = X[indexes]
        rows = x_classes.shape[0]
        if rows == 0:
            continue
        distances = list(map(lambda x: calc_dist(x, centroids[i]), x_classes))
        sum_dist = np.sum(distances)
        denom += sum_dist

    centroid_main = np.sum(centroids, axis=0) / len(centroids)

    for i in range(len(uniq_clusters)):
        indexes = np.where(y_predicted == uniq_clusters[i])
        num += len(indexes[0]) * calc_dist(centroids[i], centroid_main)
    result = ((N - K) * num) / ((K - 1) * denom)
    return result


def paint_metric(clusters, metrics, title="title"):
    plt.plot(clusters, metrics)
    plt.title(title)
    plt.xlabel("Clusters")
    plt.show()


def get_metrix():
    inner_metric, external_metric = [], []
    for clusters in x_arr:
        centroids = k_means(clusters)
        y_predicted_tmp = predict(centroids)
        inner_metric.append(inner_calinski_harabasz(y_predicted_tmp, clusters, centroids))
        # inner_metric_y_arr.append(calinski_harabasz_score(x_features_with_norm, y_predicted_tmp))
        external_metric.append(rand_metric(y_predicted_tmp))
    return inner_metric, external_metric


target_name = 'class'
filename = "lab9_files/dataset_wine.csv"
eps = 1e-6
limit_iter = 1000

dataset = pd.read_csv(filename)

y = dataset[target_name].values.tolist()
y_uniq = list(set(y))
dataset = dataset.drop(columns=target_name)

minmax = dataset_minmax(dataset.values)
X = normalize(dataset.values, minmax)

reducer = PCA(n_components=2)
X_reduced = reducer.fit_transform(X)

colors = ["g", "r", "b", "c"]
show_clusters(y, "Dataset")

x_features_with_norm_2d = reducer.fit_transform(X)

centroids = k_means(3)
y_predicted = predict(centroids)
show_clusters(y_predicted, "KMeans")
rand_metric(y_predicted)

max_clusts = 11
x_arr = [x for x in range(2, max_clusts)]
inner_metric, external_metric = get_metrix()
paint_metric(x_arr, external_metric, "Rand Index")
paint_metric(x_arr, inner_metric, "Calinski–Harabasz")
