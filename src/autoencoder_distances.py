import pandas as pd
from keras import Input
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Dense, regularizers, Dropout, BatchNormalization, AlphaDropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model

from autoencoder import AutoEncoder

if __name__ == '__main__':
    # encoded = AutoEncoder(load_model("encoding/models/proteins-autoencoder_4CG1_Angles_And_RSA.h5")).encode(data)
    # df = pd.DataFrame(encoded)
    # df.to_csv("encoding/models/proteins-autoencoder_4CG1_Angles_And_RSA_result.csv"), index = False, header = False)
