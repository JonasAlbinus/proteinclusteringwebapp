
import numpy as np
import somoclu


def compute_som():
    c1 = np.random.rand(50, 3) / 5
    c2 = (0.6, 0.1, 0.05) + np.random.rand(50, 3) / 5
    c3 = (0.4, 0.1, 0.7) + np.random.rand(50, 3) / 5
    data = np.float32(np.concatenate((c1, c2, c3)))

    n_rows, n_columns = 100, 160
    som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)
    som.train(data)
    som.view_component_planes()


if __name__ == '__main__':
    compute_som()
