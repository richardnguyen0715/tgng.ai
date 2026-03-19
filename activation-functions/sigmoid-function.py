import numpy as np

def sigmoid(x):
    
    x = np.array(x, dtype=np.float64)
    x = np.clip(x, -500, 500)
    
    return 1 / ( 1 + np.exp(-x))


print(sigmoid([0, 2, -2]))