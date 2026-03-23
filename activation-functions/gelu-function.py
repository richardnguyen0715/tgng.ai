import numpy as np
import math
from scipy.special import erf


def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.array(x, dtype=np.float64)
    # vectorize trả về một hàm
    erf_vectorized = np.vectorize(lambda t: math.erf(t / math.sqrt(2)))
    return 0.5 * x * (1 + erf_vectorized(x))


# Sử dụng scipy thay vì math + numpy
def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.array(x, dtype=np.float64)
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))