import csv
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import Callback
import sklearn.preprocessing

from config import ENC_PLOT_DATA_PATH


class PlotCheckpoint(Callback):
    def __init__(self, autoencoder, title, path=ENC_PLOT_DATA_PATH, axis=[-3, 3, -3, 3]):
        super().__init__()
        self.__autoencoder = autoencoder
        self.__title = title
        self.__axis = axis
        self.__path = os.path.join(path, title)

    def on_train_begin(self, logs={}):
        if os.path.exists(self.__path):
            shutil.rmtree(self.__path)
        os.mkdir(self.__path)

    def on_epoch_end(self, epoch, logs={}):
        assert self.model == self.__autoencoder.get_model()
        self.__autoencoder.plot(data=self.validation_data[0],
                                file=os.path.join(self.__path, "{0:0>5}.jpg".format(epoch)),
                                title=self.__title,
                                epoch=epoch + 1,
                                axis=self.__axis,
                                loss=logs['val_loss'])


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


def write_standardised_angles_to_file(protein):
    angles = read_data('encoding/Angles_Unstandardised/' + protein + ".csv", normalize='std')
    np.savetxt("encoding/Angles_Standardised/" + protein + "_standardised.csv", angles, fmt='%.5f', delimiter=',')


def convert_CUT_RSA():
    resultFilename = "encoding/data/RSA/CUT_RSA_10000_Processed.csv"

    filename = "encoding/data/RSA/CUT_RSA_10000.csv"
    f = open(filename, 'r')
    new_string = []
    for line in f:
        s2 = ' '.join(line.split()[1:])
        result = s2.replace(' ', ',')
        new_string.append(result)
    f.close()

    wf = open(resultFilename, 'w')
    for d in new_string:
        wf.write(d + "\n")

    wf.close()


def convert_PET_RSA():
    resultFilename = "encoding/data/RSA/PET_RSA_10000_Processed.csv"

    filename = "encoding/data/RSA/PET_RSA_10000.csv"
    f = open(filename, 'r')
    new_string = []
    for line in f:
        s2 = ' '.join(line.split()[1:])
        result = s2.replace(' ', ',')
        new_string.append(result)
    f.close()

    wf = open(resultFilename, 'w')
    for d in new_string:
        wf.write(d + "\n")

    wf.close()


def merge_csvs():
    file1 = "encoding/Angles_Standardised/1L3P_SA_standardised.csv"
    file2 = "encoding/data/RSA/1L3P.csv"
    new_string = []
    with open(file1) as f1, open(file2) as f2:
        for x, y in zip(f1, f2):
            newline = x.strip() + "," + y.strip()
            new_string.append(newline)

    wf = open("encoding/angles/1L3P_Angles_And_RSA.csv", 'w')
    for d in new_string:
        wf.write(d + "\n")
    wf.close()


def merge_csvs():
    file1 = "encoding/Angles_Standardised/1L3P_SA_standardised.csv"
    file2 = "encoding/data/RSA/1L3P.csv"
    new_string = []
    with open(file1) as f1, open(file2) as f2:
        for x, y in zip(f1, f2):
            newline = x.strip() + "," + y.strip()
            new_string.append(newline)

    wf = open("encoding/angles/1L3P_Angles_And_RSA.csv", 'w')
    for d in new_string:
        wf.write(d + "\n")
    wf.close()


def format_elements():
    resultFilename = "encoding/angles/4CG1_merged_Processed.csv"

    filename = "encoding/angles/4CG1_merged.csv"
    f = open(filename, 'r')
    new_string = []
    for line in f:
        vals = line.split(",")
        result = ''
        for a in vals:
            result += float("{0:.5f}".format(a)) + ','
        result = result.strip(",")
        new_string.append(result)
    f.close()

    wf = open(resultFilename, 'w')
    for d in new_string:
        wf.write(d + "\n")

    wf.close()


def truncate_even():
    resultFilename = "encoding/data/RSA/6EQE_even_eliminated.csv"

    filename = "encoding/data/RSA/6EQE.csv"
    f = open(filename, 'r')
    new_string = []
    count = 0
    for line in f:
        if count % 2 != 0:
            new_string.append(line)
        count += 1
    f.close()

    wf = open(resultFilename, 'w')
    for d in new_string:
        wf.write(d)
    wf.close()


def truncate_odd():
    resultFilename = "encoding/data/RSA/6EQE_odd_eliminated.csv"

    filename = "encoding/data/RSA/6EQE.csv"
    f = open(filename, 'r')
    new_string = []
    count = 0
    for line in f:
        if count % 2 == 0:
            new_string.append(line)
        count += 1
    f.close()

    wf = open(resultFilename, 'w')
    for d in new_string:
        wf.write(d)
    wf.close()


if __name__ == '__main__':
    truncate_even()
