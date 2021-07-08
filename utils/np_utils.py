import numpy as np


def np_softplus(x, sharpness=1):
    x_s = sharpness * x
    return (np.log(1 + np.exp(-np.abs(x_s))) + np.maximum(x_s, 0)) / sharpness
