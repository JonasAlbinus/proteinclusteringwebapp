import numpy as np
from som.clean_som import SOM

color_data = np.random.rand(3, 3)

som_color = SOM(40, 40, 3)
frames_color = []
som_color.train(color_data, L0=0.8, lam=1e2, sigma0=20)
print("quantization error:", som_color.quant_err())
