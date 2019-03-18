import os

import pandas as pd
import sklearn
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import skfuzzy as fuzz

from sklearn.cluster import KMeans, AgglomerativeClustering,Birch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
REMOVE_EVEN_CLUSTERS = False
ENC_DATA_PATH = os.path.join(ROOT_DIR, "encoding/data/RSA")
ENC_CLUSTERING_PATH = os.path.join(ROOT_DIR, "static/data")


def read_data(path, normalize=None):
    d = pd.read_csv(path, header=None)
    if normalize == 'min-max':
        aux = d.values
        min_max = sklearn.preprocessing.MinMaxScaler()
        d = pd.DataFrame(min_max.fit_transform(aux))
    elif normalize == 'std':
        d = (d - d.mean()) / d.std()
        d = d.fillna(0)
        # 1JT8: 134, 201, 203
    return d.values


def create_labels(size, step):
    y = []
    for i in range(0, size):
        y.append(i // step)
    return y


def order_labels(y):
    k = max(set(y))
    s = set()
    res = y.copy()
    for i in range(0, len(y)):
        if y[i] not in s and y[i] <= k:
            s.add(y[i])
            res[res == y[i]] = k + len(s)
    res = res - k - 1
    return res


def cluster(data, method, k):
    if method == 'k-means':
        return KMeans(n_clusters=k, random_state=42).fit_predict(data)
    elif method == 'agg':
        return AgglomerativeClustering(n_clusters=k).fit_predict(data)
    elif method == 'birch':
        return Birch(n_clusters=k).fit_predict(data)
    elif method == 'skfuzzy':
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data.T, k, 2, error=0.5, maxiter=1000)
        return np.argmax(u, axis=0)


def comute_plot_clusters(step, data, c, name):
    print("------------------------------")
    print(name)
    print(c)
    print(str(step))
    pca = sklearn.decomposition.PCA(n_components=2)
    reduced_pca = pca.fit_transform(data)
    y = np.array(create_labels(reduced_pca.shape[0], step=step))

    # y = np.array(create_labels(data.shape[0], step=step))
    K = len(set(y))
    print("Step", step)
    print("K", K)
    y_pred = cluster(data, method=c, k=K)

    v_measure = metrics.v_measure_score(y, y_pred)
    silhouette_score = metrics.silhouette_score(data, y_pred)
    print("V-measure: %0.3f" % v_measure)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(data, y_pred))

    plotNew(data, 0, y_pred, K, name, c, legend_right='V-measure: %.3f' % v_measure,
            legend_left='Silhouette Coefficient: %.2f' % silhouette_score)

    print("------------------------------")


def plotNew(x, v, y, k, protein, alg, legend_left, legend_right, labels=None):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    str1 = protein + ", K:" + str(k)
    plt.title(str1)
    # Plot assigned clusters, for each data point in training set
    cluster_membership = y
    x = PCA(n_components=2).fit_transform(x).T
    x_points = []
    y_points = []
    list_cl = np.array(cluster_membership).tolist()

    for j in range(k):
        cluster_idx = [i for i, x in enumerate(list_cl) if x == j]
        for d in cluster_idx:
            x_points.append(x[0][d])
            y_points.append(x[1][d])

    t = np.arange(10000)
    plt.scatter(
        x_points,
        y_points,
        c=t, marker='o', s=10, cmap="plasma"
    )

    ax.text(0.01, 0.01, legend_left,
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            fontsize=12)

    ax.text(0.99, 0.01, legend_right,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            fontsize=12)
    plt.grid(True)
    plt.colorbar()
    plt.savefig(os.path.join(ENC_CLUSTERING_PATH, protein + '_' + str(k) + '_' + alg + '.png'))
    plt.close(fig)


def compute_analysis(protein, step, alg):
    path = ENC_DATA_PATH + "/" + protein + ".csv"

    data_std = read_data(path, normalize='std')
    data_std = np.delete(data_std, 10000, axis=0)

    return comute_plot_clusters(step, data_std, alg, protein)


def plotGraphs(data_points_x, data_points_y, label):
    data_points_x = ["4", "10", "20", "100", "200"]
    data_points_y = [0.6245649459652394, 0.6836229479509609, 0.7133112394871962, 0.8508682190074076, 0.8717780643677956]
    # plotting the line 2 points
    plt.plot(data_points_x, data_points_y, label)

    # naming the x axis
    plt.xlabel('Number of clusters')
    # naming the y axis
    plt.ylabel('V Measure')
    # giving a title to my graph
    plt.title('V Measure')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()
    plt.savefig("static/data/HAC-Vmeasure-Elimination-Odd.jpg")


if __name__ == '__main__':
    # for step in [2500, 1000, 500, 100, 50]:
    for step in [500]:
        compute_analysis("1L3P", step, "birch")
