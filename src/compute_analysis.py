import os

import pandas as pd
import sklearn
import numpy as np
from sklearn import metrics, decomposition
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans, AffinityPropagation, \
    Birch

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
    elif method == 'birch':
        return Birch(n_clusters=k).fit_predict(data)


def comute_plot_clusters(step, data, c, name, verbose=False, remove_even_clusters=False):
    print("------------------------------")
    print(name)
    print(c)
    print("step" + str(step))
    pca = decomposition.PCA(n_components=2)
    reduced_pca = pca.fit_transform(data)

    y = np.array(create_labels(data.shape[0], step=step))
    if remove_even_clusters:
        data = data[y % 2 == 1]
        reduced_pca = reduced_pca[y % 2 == 1]
        y = y[y % 2 == 1]
        y = order_labels(y)
    K = len(set(y))
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
                      '_pca.png'),
         title=name + ', K: %d' % K,
         legend_right='V-measure: %.3f' % v_measure,
         legend_left='Explained var. ratio: %.2f' % sum(pca.explained_variance_ratio_))
    print("------------------------------")




def plot(x, y, filename, legend_right=None, legend_left=None, title=None):
    fig = plt.figure(facecolor='white')
    t = np.arange(10000)
    plt.scatter(x[:, 0], x[:, 1], c=t, marker='o', s=10)
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

    data_std = read_data(path, normalize='std')
    data_std = np.delete(data_std, 10000, axis=0)

    comute_plot_clusters(step, data_std, alg, protein, verbose=False, remove_even_clusters=REMOVE_EVEN_CLUSTERS)


if __name__ == '__main__':
    for step in [2500, 1000, 500, 100, 50]:
        compute_analysis("4CG1_odd_eliminated", step, "agg")
        compute_analysis("4CG1_odd_eliminated", step, "k-means")
    # compute_analysis("6EQE_odd_eliminated", 500, "agg")
    # compute_analysis("4CG1_odd_eliminated", 500, "agg")


#     silhouette = 0
#     data_points = []
#     second_data_points = []
#     x_axis = []
#     for step in [2500, 1000, 500, 100, 50]:
#         silhouette = compute_analysis("4CG1_odd_eliminated", step, "agg")
#         data_points.append(silhouette)
#     print("__________________4cg1_______")
#     print(data_points)
#
#     for step in [2500, 1000, 500, 100, 50]:
#         silhouette = compute_analysis("6EQE_odd_eliminated", step, "agg")
#         second_data_points.append(silhouette)
#     print("______________6eqe___________")
#     print(second_data_points)
# #     plot vmeasure
# # # line 1 points
#     x1 = ["4", "10", "20", "100", "200"]
#     y1 = [0.1204195627671331, 0.6786927001536024, 0.6639366647159269, 0.8223571368686163, 0.8448431169417834]
#     # plotting the line 1 points
#     plt.plot(x1, y1, label="4CG1")
# #
# # # line 2 points
#     x2 = ["4", "10", "20", "100", "200"]
#     y2 = [0.35976558670172376, 0.555808741619217, 0.662459579330922, 0.8010324786533977, 0.8462377862337636]
#     # plotting the line 2 points
#     plt.plot(x2, y2, label="6EQE")
#
#     # naming the x axis
#     plt.xlabel('Number of clusters')
#     # naming the y axis
#     plt.ylabel('V Measure')
#     # giving a title to my graph
#     plt.title('HAC clustering after eliminating odd clusters')
#
#     # show a legend on the plot
#     plt.legend()
#
#     # function to show the plot
#     plt.show()
#     plt.savefig("static/data/HAC-Vmeasure-Elimination-Odd.jpg")


# plot sil
#     s_y1 = [0.19045465559390556, 0.040632757978112964, 0.043742342529180564, 0.04215826498570515, 0.04728517345999529]
#     s_y2 = [0.04544220313127115, 0.0326677598229174, 0.02784225485678595, 0.03405324827615402, 0.04374856162209981]
#     x1 = ["4", "10", "20", "100", "200"]
#     plt.plot(x1, s_y1, label="4CG1")
#     plt.plot(x1, s_y2, label="6EQE")
#     plt.xlabel('Number of clusters')
#     # naming the y axis
#     plt.ylabel('Silhouette coefficient')
#     # giving a title to my graph
#     plt.title('Silhouette coefficient after elimination odd clusters')
#     # show a legend on the plot
#     plt.legend()
#     # function to show the plot
#     plt.show()
#     plt.savefig("static/data/HAC-Silhouette-Elimination-Odd.jpg")
