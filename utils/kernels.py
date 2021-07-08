import math
import random

import numpy as np
import torch
from scipy.spatial.distance import cdist


class AbstractKernel(object):
    def __init__(self):
        pass

    def train(self, data):
        pass

    def __call__(self, x_1, x_2, numpy=True):
        NotImplementedError()


def calc_sq_dist(x_1, x_2, numpy=True):
    n_1, n_2 = x_1.shape[0], x_2.shape[0]
    if numpy:
        return cdist(x_1.reshape(n_1, -1), x_2.reshape(n_2, -1),
                        metric="sqeuclidean")
    else:
        return torch.cdist(x_1.view(n_1, -1), x_2.view(n_2, -1)) ** 2


def calc_gaussian_kernel(x_1, x_2, sigma, numpy=True):
    sq_dist = calc_sq_dist(x_1, x_2, numpy=numpy)
    if numpy:
        return np.exp((-1 / (2 * sigma ** 2)) * sq_dist)
    else:
        return torch.exp((-1 / (2 * sigma ** 2)) * sq_dist)


class GaussianKernel(AbstractKernel):
    def __init__(self, sigma):
        AbstractKernel.__init__(self)
        self.sigma = sigma

    def __call__(self, x_1, x_2, numpy=True):
        return calc_gaussian_kernel(x_1, x_2, self.sigma, numpy=numpy)


class PercentileKernel(GaussianKernel):
    def __init__(self, p):
        self.p = p
        GaussianKernel.__init__(self, None)

    def train(self, data, numpy=True):
        sq_dist = calc_sq_dist(data, data, numpy=numpy)
        gamma = np.percentile(sq_dist.flatten(), self.p) ** -1
        self.sigma = float((0.5 / gamma) ** 0.5)


class TripleMedianKernel(AbstractKernel):
    def __init__(self, max_num_train=10000):
        AbstractKernel.__init__(self)
        self.s_1, self.s_2, self.s_3 = None, None, None
        self.max_num_train = max_num_train

    def train(self, data, numpy=True):
        n = data.shape[0]
        if n <= self.max_num_train:
            data_sample = data.reshape(n, -1)
        else:
            data_idx = list(range(n))[:self.max_num_train]
            random.shuffle(data_idx)
            data_sample = data.reshape(n, -1)[:self.max_num_train]
        sq_dist = calc_sq_dist(data_sample, data_sample, numpy=numpy)
        median_d = np.median(sq_dist.flatten()) ** 0.5
        self.s_1 = median_d
        self.s_2 = 0.1 * median_d
        self.s_3 = 10.0 * median_d

    def __call__(self, x_1, x_2, numpy=True):
        sq_dist = calc_sq_dist(x_1, x_2, numpy=numpy)
        if numpy:
            exp = np.exp
        else:
            exp = torch.exp
        k_1 = exp((-1 / (2 * self.s_1 ** 2)) * sq_dist)
        k_2 = exp((-1 / (2 * self.s_1 ** 2)) * sq_dist)
        k_3 = exp((-1 / (2 * self.s_1 ** 2)) * sq_dist)
        return (k_1 + k_2 + k_3) / 3


class SobolevKernel(AbstractKernel):
    def __init__(self, sigma):
        AbstractKernel.__init__(self)
        self.sigma = sigma

    def __call__(self, x_1, x_2, numpy=True):
        if numpy:
            sin, cos = np.sin, np.cos
        else:
            sin, cos = torch.sin, torch.cos
        n_1 = x_1.shape[0]
        n_2 = x_2.shape[0]
        diff = (x_1.reshape(n_1, 1, -1) - x_2.reshape(1, n_2, -1)) / self.sigma
        k_1 = (sin(diff) - diff * cos(diff)).prod(2)
        diff_prod = diff.prod(2)
        diff_prod = diff_prod + 1e-6 * ((diff_prod > 0) * 2 - 1)
        k_2 = (2 / math.pi) * (diff_prod ** -3)
        return k_1 * k_2
