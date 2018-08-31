import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import os


def cluster(k):
    sns.set()  # for plot styling
    X, y_true = make_blobs(n_samples=300, centers=int(k),
                           cluster_std=0.60, random_state=0)
    # plt.scatter(X[:, 0], X[:, 1], s=50)

    kmeans = KMeans(n_clusters=int(k))
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    # plt.show()
    plt.title('Sample analysis with ' + k + " clusters")

    strFile = "static/plots/plot.png"
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)

    plt.close()


if __name__ == '__main__':
    cluster(1)
