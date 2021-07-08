import torch.nn as nn


class AbstractRho(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def is_finite(self):
        raise NotImplementedError()

    def forward(self, x, z):
        raise NotImplementedError()

    def jacobian(self, x, z, numpy=True):
        raise NotImplementedError()

    def get_target_beta(self):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()

    def get_parameter_vector(self):
        raise NotImplementedError()

    def get_parameter_dict(self):
        raise NotImplementedError()
