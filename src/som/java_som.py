import numpy as np

from somber import RecursiveSom
from string import ascii_lowercase

# Dumb sequence generator.
def seq_gen(num_to_gen, probas):

    symbols = ascii_lowercase[:probas.shape[0]]
    identities = np.eye(probas.shape[0])
    seq = []
    ids = []
    r = 0
    choices = np.arange(probas.shape[0])
    for x in range(num_to_gen):
        r = np.random.choice(choices, p=probas[r])
        ids.append(symbols[r])
        seq.append(identities[r])

    return np.array(seq), ids

# Transfer probabilities.
# after an A, we have a 50% chance of B or C
# after B, we have a 100% chance of A
# after C, we have a 50% chance of B or C
# therefore, we will never expect sequential A or B, but we do expect
# sequential C.
probas = np.array(((0.0, 0.5, 0.5),
                   (1.0, 0.0, 0.0),
                   (0.0, 0.5, 0.5)))

X, ids = seq_gen(10000, probas)

# initialize
# alpha = contribution of non-recurrent part to the activation.
# beta = contribution of recurrent part to activation.
# higher alpha to beta ratio
s = RecursiveSom((10, 10),
                 learning_rate=0.3,
                 alpha=1.2,
                 beta=.9)

# train
# show a progressbar.
s.fit(X, num_epochs=100, updates_epoch=10, show_progressbar=True)

# predict: get the index of each best matching unit.
predictions = s.predict(X)
# quantization error: how well do the best matching units fit?
quantization_error = s.quantization_error(X)

# inversion: associate each node with the exemplar that fits best.
inverted = s.invert_projection(X, ids)

# find which sequences are mapped to which neuron.
receptive_field = s.receptive_field(X, ids)

# generate some data by starting from some position.
# the position can be anything, but must have a dimensionality
# equal to the number of weights.
starting_pos = np.ones(s.num_neurons)
generated_indices = s.generate(50, starting_pos)

# turn the generated indices into a sequence of symbols.
generated_seq = inverted[generated_indices]
