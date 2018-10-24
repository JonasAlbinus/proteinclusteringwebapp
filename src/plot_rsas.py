import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
ENC_DATA_PATH = os.path.join(ROOT_DIR, "encoding/data/RSA")
ENC_CLUSTERING_PATH = os.path.join(ROOT_DIR, "static/data")


def read_data(protein):
    path = ENC_DATA_PATH + "/" + protein + ".csv"
    d = pd.read_csv(path, header=None)
    df = pd.DataFrame(d.values, columns=['index', 'Residue', 'Area'])
    df.drop(['index'], axis=1)
    return df


def plot_average_frames(protein, title):
    df = read_data(protein)
    fig, ax = plt.subplots()
    df.plot.scatter(x='Residue', y='Area', c='Residue',colormap='viridis')
    plt.title(title)
    plt.savefig(ENC_CLUSTERING_PATH + '/' + protein + '.png')


if __name__ == '__main__':
    plot_average_frames("CUT_RSA", "Cutinase averaged over 10k frames of trajectory")
    plot_average_frames("PET_RSA", "PETase averaged over 10k frames of trajectory")
