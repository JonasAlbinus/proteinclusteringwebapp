import shutil

import imageio
import matplotlib.pyplot as plt
from keras import Input
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Dense, regularizers, Dropout, BatchNormalization, AlphaDropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from numpy.random import normal
from sklearn.model_selection import train_test_split
from keras.models import load_model

from config import *
from utils import read_data, PlotCheckpoint


class AutoEncoder:
    def __init__(self, model=None, input_size=None, encoded_size=None,
                 l1=1e-7, activation='relu', dropout_rate=0.2, use_batch_norm=True, coefficients=[4, 8, 12]):
        if model is None:
            self.__model = self.__autoencoder(input_size, encoded_size, l1, activation,
                                              dropout_rate, use_batch_norm, coefficients)
        else:
            self.__model = model

        plot_model(self.__model, to_file=ENC_MODEL_IMAGE_PATH, show_shapes=True)

    @staticmethod
    def __autoencoder(size, encoded_size, l1, activation, dropout_rate,
                      use_batch_norm, coefficients):
        # TODO activation fns: relu, elu, selu (AlphaDropout instead of dropout for selu)
        # TODO dropout rate: 0 (no dropout), 0.1, 0.2, 0.3
        # TODO batch norm: YES/NO
        # TODO l1: 0 (no regularization), 1e-7, 1e-6
        # TODO optimizers: rmsprop, adam, nadam
        # TODO coefficients: [4, 8, 12], [4, 8], [2, 4] etc.
        input_layer = Input(shape=(size,), name='input_conformation')

        x = input_layer
        for i, c in enumerate(coefficients):
            idx = i + 1
            x = Dense(size // c, activation=activation, name="enc_%d" % idx)(x)

            if use_batch_norm:
                x = BatchNormalization(name="enc_%d_batch_norm" % idx)(x)

            if dropout_rate > 0:
                if activation == 'selu':
                    x = AlphaDropout(dropout_rate, name="enc_%d_dropout" % idx)(x)
                else:
                    x = Dropout(dropout_rate, name="enc_%d_dropout" % idx)(x)

        x = Dense(encoded_size, activation="linear", name="encoded",
                  activity_regularizer=regularizers.l1(l1))(x)

        for i, c in enumerate(reversed(coefficients)):
            idx = len(coefficients) - i
            x = Dense(size // c, activation=activation, name="dec_%d" % idx)(x)

            if use_batch_norm:
                x = BatchNormalization(name="dec_%d_batch_norm" % idx)(x)

            if dropout_rate > 0:
                if activation == 'selu':
                    x = AlphaDropout(dropout_rate, name="dec_%d_dropout" % idx)(x)
                else:
                    x = Dropout(dropout_rate, name="dec_%d_dropout" % idx)(x)

        decoded = Dense(size, activation="linear", name="decoded_conformation")(x)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(lr=0.001),
                            loss='mse',
                            metrics=['mae'])
        autoencoder.summary()

        return autoencoder

    def get_encoder(self):
        return Model(inputs=self.__model.input,
                     outputs=self.__model.get_layer("encoded").output)

    def get_model(self):
        return self.__model

    def encode(self, data):
        assert data.shape[1] == self.__model.input_shape[1]
        return self.get_encoder().predict(data)

    def compute_distances(self, data):
        encoded = AutoEncoder(load_model("encoding/models/proteins-autoencoder_4CG1_Angles_And_RSA.h5")).encode(data)
        df = pd.DataFrame(encoded)
        df.to_csv("encoding/models/proteins-autoencoder_4CG1_Angles_And_RSA_result.csv", index=False, header=False)
        return df


    def plot(self, data, file=None, title=None, axis=None, loss=None, epoch=None):
        points = self.encode(data)
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        t = np.arange(10000)
        plt.scatter(points[:, 0], points[:, 1], marker='o', c=t, s=7)
        if title is not None:
            plt.title(title)
        if axis is not None:
            plt.axis(axis)
        plt.grid(linestyle='dotted')

        if epoch is not None and loss is not None:
            ax.text(0.95, 0.01, 'Epoch: %d, loss: %.4f' % (epoch, loss),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='green', fontsize=15)

        plt.colorbar()

        if file is not None:
            plt.savefig(file, dpi=120)
        else:
            plt.show()

        plt.close(fig)


def training(file_path, epochs, noise_std, do_gif=False):
    def create_gif(path, title, step=1):
        images = []
        for i, filename in enumerate(os.listdir(path)):
            if i % step == 0:
                images.append(imageio.imread(os.path.join(path, filename)))
        imageio.mimsave(os.path.join(ENC_PLOT_DATA_PATH, title), images, duration=0.1)

        shutil.rmtree(path)

    def get_data(file_path):
        # TODO: try using 2 proteins: train on the first one test on the other one
        data = read_data(file_path, normalize='std')
        x_train, x_test = train_test_split(data, test_size=0.1, random_state=42)
        return x_train, x_test

    def add_noise(x, std):
        return x + normal(0, std, size=x.shape)

    for file in os.listdir(file_path):
        if not (file.endswith(".csv")):
            continue
        name = file.replace(".csv", "")
        print("Analysing %s" % name)
        # x_train, x_test = get_data(os.path.join(file_path, file))
        # x_train_noisy, x_test_noisy = add_noise(x_train, std=noise_std), add_noise(x_test, std=noise_std)
        # autoencoder = AutoEncoder(input_size=x_train.shape[1], encoded_size=2)
        #
        # check_pointer = ModelCheckpoint(filepath=ENC_MODEL_PATH_TRAIN % name, monitor='val_loss',
        #                                 verbose=1,
        #                                 save_best_only=True)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss',
        #                               factor=0.5, patience=5,
        #                               min_lr=0.00001)
        #
        # autoencoder.get_model().fit(x_train_noisy, x_train, epochs=epochs, batch_size=16,
        #                             validation_data=(x_test_noisy, x_test), verbose=0,
        #                             shuffle=True,
        #                             callbacks=[
        #                                           check_pointer,
        #                                           reduce_lr,
        #                                           TensorBoard(log_dir='/tmp/tb_' + name),
        #                                       ] + ([PlotCheckpoint(autoencoder, title=name)] if do_gif else [])
        #                             )
        # if do_gif:
        #     create_gif(os.path.join(ENC_PLOT_DATA_PATH, name), title=name + ".gif", step=epochs // 100)

        data = read_data(os.path.join(file_path, file), normalize='std')

        encoded = AutoEncoder(load_model("encoding/models/proteins-autoencoder_6EQE_Angles_And_RSA.h5")).encode(data)
        df = pd.DataFrame(encoded)
        df.to_csv("encoding/models/6EQE_Angles_And_RSA_result.csv", index=False, header=False)

        # autoencoder.plot(data,
        #                  title=name,
        #                  file=os.path.join(ENC_PLOT_DATA_PATH, name + "_" + str(epochs) + ".jpg")
        #                  )


if __name__ == '__main__':
    training(ENC_ANGLES_DATA_PATH, epochs=5000, noise_std=0.001, do_gif=False)
