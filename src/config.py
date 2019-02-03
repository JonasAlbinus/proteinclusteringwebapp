import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))

ENC_DATA_PATH = os.path.join(ROOT_DIR, "encoding/data/RSA")
ENC_DISTANCES_PATH = os.path.join(ROOT_DIR, "encoding/data/RSA/distances")
ENC_PLOT_DATA_PATH = os.path.join(ROOT_DIR, "encoding/data/plot")
ENC_PCA_PATH = os.path.join(ROOT_DIR, "encoding/data/pca")

ENC_MODEL_PATH = os.path.join(ROOT_DIR, "encoding/models")
ENC_MODEL_PATH_TRAIN = os.path.join(ROOT_DIR, "encoding/models/proteins-autoencoder_%s.h5")
ENC_VAE_MODEL_PATH_TRAIN = os.path.join(ROOT_DIR, "encoding/models/proteins-vae_%s.h5")
ENC_MODEL_PATH_TEST = os.path.join(ROOT_DIR, "encoding/models/e10000/proteins-autoencoder_%s.h5")
ENC_MODEL_IMAGE_PATH = os.path.join(ROOT_DIR, "encoding/models/model_auto.png")
ENC_VAE_MODEL_IMAGE_PATH = os.path.join(ROOT_DIR, "encoding/models/model_vae.png")

ENC_CLUSTERING_PATH = os.path.join(ROOT_DIR, "encoding/data/clustering")
ENC_COLOR_PATH = os.path.join(ROOT_DIR, "encoding/data/color")

# ENC_ANGLES_DATA_PATH = os.path.join(ROOT_DIR, "encoding/Angles_And_RSA")
ENC_ANGLES_DATA_PATH = os.path.join(ROOT_DIR, "encoding/angles")
ENC_ANGLES_CORRELATIONS_PATH = os.path.join(ROOT_DIR, "encoding/data/Angles/correlations")
ENC_ANGLES_DISTANCES_PATH = os.path.join(ROOT_DIR, "encoding/data/Angles/distances")
ENC_ANGLES_RSA_DATA_PATH = os.path.join(ROOT_DIR, "encoding/data/Angles + RSA")
ENC_ANGLES_RSA_DISTANCES_PATH = os.path.join(ROOT_DIR, "encoding/data/Angles + RSA/distances")
ENC_ANGLES_RSA_CORRELATIONS_PATH = os.path.join(ROOT_DIR, "encoding/data/Angles + RSA/correlations")