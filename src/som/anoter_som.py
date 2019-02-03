from somlib import som
import random
import numpy as np

num_training = 25
samples = []
choices = [1, 5, 10, 90, 80, 85]
# for _ in range(num_training):
#     sample = np.array([random.randint(1, 100) / float(100),
#                        random.randint(1, 100) / float(100),
#                        random.randint(1, 100) / float(100)])
#     sample = sample.reshape(1, 3)
#     samples.append(sample)
#
samples = np.array(
    [[0.000247917, 0.006252083, 0.000784722, 0.010964583, 0.010334028, 0.002326389, 0.001466667, 0.00704375,
      0.005032639, 0.011659722, 0.032663889, 0.0124, 0.019867361, 0.017288194, 0.005232639, 0.004534028,
      0.022661111, 0.013976389, 0.013680556, 0.045642361, 0.563611111, 0.112277778, 0.076435417, 0.003616667,
      0],
     [0.000383459, 0.003520301, 0.000357143, 0.003011278, 0.008131579, 0.000176692, 0.004556391, 0.003845113,
      0.00331203, 0.009182707, 0.028743609, 0.002796241, 0.016818797, 0.02822782, 0.003991729, 0.001344361,
      0.018478947, 0.009588722, 0.046115789, 0.045357143, 0.520604511, 0.119554135, 0.111378947, 0.010509774,
      0.0000128],
     [0.008104878, 0.038190244, 0.003847561, 0.018547561, 0.036108537, 0.025002439, 0.028258537, 0.076140244,
      0.064082927, 0.047587805, 0.068870732, 0.019108537, 0.026930488, 0.060407317, 0.007646341, 0.00945,
      0.061192683, 0.067860976, 0.034587805, 0.056321951, 0.06822561, 0.06729878, 0.06265, 0.025119512,
      0.018458537]
     ])

# samples.append(np.array([[0.000247917, 0.000383459, 0.008104878]]))
# samples.append(np.array([[0.006252083, 0.003520301, 0.038190244]]))
# samples.append(np.array([0.000784722, 0.000357143, 0.003847561]))
# samples.append(np.array([0.010964583, 0.003011278, 0.018547561]).reshape(1, 3))
# samples.append(np.array([0.010334028, 0.008131579, 0.036108537]).reshape(1, 3))
# samples.append(np.array([0.002326389, 0.000176692, 0.025002439]).reshape(1, 3))
# samples.append(np.array([0.001466667, 0.004556391, 0.028258537]).reshape(1, 3))
# samples.append(np.array([0.00704375, 0.003845113, 0.076140244]).reshape(1, 3))
# samples.append(np.array([0.005032639, 0.00331203, 0.064082927]).reshape(1, 3))
# samples.append(np.array([0.011659722, 0.009182707, 0.047587805]).reshape(1, 3))
# samples.append(np.array([0.032663889, 0.028743609, 0.068870732]).reshape(1, 3))
# samples.append(np.array([0.0124, 0.002796241, 0.019108537]).reshape(1, 3))
# samples.append(np.array([0.019867361, 0.016818797, 0.026930488]).reshape(1, 3))
# samples.append(np.array([0.017288194, 0.02822782, 0.060407317]).reshape(1, 3))
# samples.append(np.array([0.005232639, 0.003991729, 0.007646341]).reshape(1, 3))
# samples.append(np.array([0.004534028, 0.001344361, 0.00945]).reshape(1, 3))
# samples.append(np.array([0.022661111, 0.018478947, 0.061192683]).reshape(1, 3))
# samples.append(np.array([0.013976389, 0.009588722, 0.067860976]).reshape(1, 3))
# samples.append(np.array([0.013680556, 0.046115789, 0.034587805]).reshape(1, 3))
# samples.append(np.array([0.045642361, 0.045357143, 0.056321951]).reshape(1, 3))
# samples.append(np.array([0.563611111, 0.520604511, 0.06822561]).reshape(1, 3))
# samples.append(np.array([0.112277778, 0.119554135, 0.06729878]).reshape(1, 3))
# samples.append(np.array([0.076435417, 0.111378947, 0.06265]).reshape(1, 3))
# samples.append(np.array([0.003616667, 0.010509774, 0.025119512]).reshape(1, 3))
# samples.append(np.array([0.0, 1.28, 0.018458537]).reshape(1, 3))

# reversed_arr = colors.transpose()

# print('\n ['.join(['\t ,'.join([str(cell) for cell in row]) for row in reversed_arr]))
# sample = sample.reshape(1, 3)
# print(sample)
# print("~~~~~~~~~")
# samples = colors.reshape(25, 3)
s = som.SOM(neurons=(5, 5), dimentions=25, n_iter=500, learning_rate=0.1)
s.train(samples)  # samples is a n x 3 matrix
print("Cluster centres:", s.weights_)
print("labels:", s.labels_)
result = s.predict(samples)
s.displayClusters(samples)
