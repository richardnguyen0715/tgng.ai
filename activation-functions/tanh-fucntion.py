import numpy as np

def tanh(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -500, 500)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

print(tanh([0, 1, -1, 3]))