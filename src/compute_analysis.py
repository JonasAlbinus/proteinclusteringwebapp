import os

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import threading
from sklearn import metrics, decomposition

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans

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
    elif method == 'spec':
        return SpectralClustering(n_clusters=k, random_state=42).fit_predict(data)
    elif method == 'mb-k-means':
        return MiniBatchKMeans(n_clusters=k, random_state=42).fit_predict(data)


def comute_plot_clusters(step, data, c, name, verbose=False, remove_even_clusters=False):
    pca = decomposition.PCA(n_components=2)
    reduced_pca = pca.fit_transform(data)

    y = np.array(create_labels(data.shape[0], step=step))
    if remove_even_clusters:
        data = data[y % 2 == 1]
        reduced_pca = reduced_pca[y % 2 == 1]
        y = y[y % 2 == 1]
        y = order_labels(y)
    K = len(set(y))
    print("Method: " + c)
    y_pred = cluster(data, method=c, k=K)

    v_measure = metrics.v_measure_score(y, y_pred)
    print('Number of clusters (K): %d' % K)
    if verbose:
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, y_pred))
        print("Completeness: %0.3f" % metrics.completeness_score(y, y_pred))
    print("V-measure: %0.3f" % v_measure)
    if verbose:
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(y, y_pred))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(y, y_pred))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(data, y_pred))

    plot(reduced_pca, order_labels(y_pred),
         os.path.join(ENC_CLUSTERING_PATH, name + '_' + str(K) + '_' + c +
                      '_pca.jpg'),
         title=name + ', K: %d' % K,
         legend_right='V-measure: %.3f' % v_measure,
         legend_left='Explained var. ratio: %.2f' % sum(pca.explained_variance_ratio_))


def plot(x, y, filename, legend_right=None, legend_left=None, title=None):
    fig = plt.figure(facecolor='white')
    plt.scatter(x[:, 0], x[:, 1], c=y, marker='o', s=10)
    if title is not None:
        plt.title(title)
    plt.grid(linestyle='dotted')

    ax = fig.add_subplot(111)
    if legend_right is not None:
        ax.text(0.99, 0.01, legend_right,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=12)
    if legend_left is not None:
        ax.text(0.01, 0.01, legend_left,
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                fontsize=12)
    plt.colorbar()
    plt.clim(0, len(x))
    plt.savefig(filename, dpi=120)
    plt.close(fig)


def compute_analysis(protein, step, alg):
    path = ENC_DATA_PATH + "/" + protein + ".csv"
    print("P: %s" % path)
    print("Analyzing %s" % protein)
    # print("STEP: %s" % step)

    data_std = read_data(path, normalize='std')
    data_std = np.delete(data_std, 10000, axis=0)

    comute_plot_clusters(step, data_std, alg, protein, verbose=False, remove_even_clusters=REMOVE_EVEN_CLUSTERS)


if __name__ == '__main__':
    compute_analysis("1GO1", 50, "agg")
