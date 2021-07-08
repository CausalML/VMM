import torch
import torch.nn as nn
import numpy as np

from parametric_scenarios.abstract_parametric_scenario import \
    AbstractParametricScenario
from parametric_scenarios.abstract_rho import AbstractRho
from utils.torch_utils import torch_to_np


def linear_g_function(t, a, b, numpy=True):
    if numpy:
        matmul = np.matmul
    else:
        matmul = torch.matmul
    return matmul(t, a) + b


class PolyIVRho(AbstractRho):
    def __init__(self):
        nn.Module.__init__(self)
        self.a = nn.Parameter(torch.zeros(2, 2))
        self.b = nn.Parameter(torch.zeros(2))

    def is_finite(self):
        for p in (self.a, self.b):
            if not p.data.isfinite().all():
                return False
        return True

    def g(self, t):
        return linear_g_function(t, a=self.a, b=self.b, numpy=False)

    def forward(self, x, z):
        t, y = x[:, :2], x[:, 2:]
        rho_out = y - self.g(t)
        return rho_out

    def initialize(self):
        nn.init.normal_(self.a)
        nn.init.normal_(self.b)

    def get_parameter_vector(self):
        param_tensor = torch.cat([self.a.data.flatten(),
                                  self.b.data.flatten()], dim=0)
        return torch_to_np(param_tensor)

    def get_parameter_dict(self):
        return {
            "a": list(torch_to_np(self.a.data.flatten())),
            "b": list(torch_to_np(self.b.data.flatten())),
        }


class MultiOutputIVScenario(AbstractParametricScenario):
    def __init__(self, iv_strength=0.75):
        self.a = np.array([
            [1.0, -1.5],
            [2.0, 3.0],
        ])
        self.b = np.array([-0.5, 0.5])
        self.iv_strength = iv_strength

        z_dim = 1
        rho_dim = self.a.shape[1]
        AbstractParametricScenario.__init__(self, rho_dim=rho_dim, z_dim=z_dim,
                                            theta_dim=6)

    def generate_data(self, num_data, split):
        z = np.random.uniform(-5, 5, (num_data, 1))
        h = np.random.randn(num_data, 1) * 5.0
        eta = 0.2 * np.random.randn(num_data, 2)
        t_1 = np.concatenate([z, 3 * (1 - z)], axis=1)
        t_2 = h + eta
        t = self.iv_strength * t_1 + (1 - self.iv_strength) * t_2
        epsilon = 0.2 * np.random.randn(num_data, 2)
        y_noise = -1.0 * h + epsilon
        g = linear_g_function(t, a=self.a, b=self.b, numpy=True)
        y = g + y_noise
        x = np.concatenate([t, y], axis=1)

        return x, z

    def get_true_parameter_vector(self):
        return np.concatenate([self.a.flatten(), self.b.flatten()], axis=0)

    def get_rho_generator(self):
        return PolyIVRho

    def calc_test_risk(self, x_test, z_test, predictor):
        t_test = x_test[:, :2]
        g_test = linear_g_function(t_test, a=self.a, b=self.b, numpy=True)
        rho = predictor.get_rho()
        g_test_pred = rho.g(predictor._to_tensor(t_test)).detach().cpu().numpy()
        return float(((g_test - g_test_pred) ** 2).mean())


def debug():
    scenario = MultiOutputIVScenario()
    scenario.setup(num_train=10000, num_dev=0, num_test=0)


if __name__ == "__main__":
    debug()
