from functools import partial

import torch
import torch.nn as nn
import numpy as np

from parametric_scenarios.abstract_parametric_scenario import \
    AbstractParametricScenario
from parametric_scenarios.abstract_rho import AbstractRho
from utils.torch_utils import torch_to_np, torch_to_float


def linear_g_function(t, a, b, numpy=True):
    if numpy:
        matmul = np.matmul
        t_expanded = np.concatenate([t, t ** 2], axis=1)
    else:
        matmul = torch.matmul
        t_expanded = torch.cat([t, t ** 2], dim=1)
    return matmul(t_expanded, a) + b


class SimpleIVRho(AbstractRho):
    def __init__(self):
        nn.Module.__init__(self)
        self.a = nn.Parameter(torch.zeros(2, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def is_finite(self):
        for p in (self.a, self.b):
            if not p.data.isfinite().all():
                return False
        return True

    def g(self, t):
        return linear_g_function(t, a=self.a, b=self.b, numpy=False)

    def forward(self, x, z):
        t, y = x[:, 0].view(-1, 1), x[:, 1].view(-1, 1)
        rho_out = y - self.g(t).view(-1, 1)
        return rho_out

    def jacobian(self, x, z, numpy=False):
        if numpy:
            cat = partial(np.concatenate, axis=1)
            ones = np.ones
        else:
            cat = partial(torch.cat, dim=1)
            ones = torch.ones

        n = x.shape[0]
        t = x[:, 0].reshape(-1, 1)
        deriv_a0 = t.reshape(-1, 1)
        deriv_a1 = (t ** 2).reshape(-1, 1)
        deriv_b = ones((n, 1))
        jac = -1.0 * cat([deriv_a0, deriv_a1, deriv_b])
        return jac.reshape(n, 1, 3)

    def get_target_beta(self):
        return np.array([1.0, 0.0, 0.0])

    def get_psi(self):
        return torch_to_float(self.a.data[0, 0])

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


class SimpleIVScenario(AbstractParametricScenario):
    def __init__(self, iv_strength=0.30):
        self.a = np.array([[3.0],
                           [-0.5]])
        self.b = np.array([0.5])
        self.iv_strength = iv_strength

        z_dim = 1
        rho_dim = self.a.shape[1]
        AbstractParametricScenario.__init__(self, rho_dim=rho_dim, z_dim=z_dim,
                                            theta_dim=3)

    def generate_data(self, num_data, split):
        z_0 = np.random.uniform(-5, 5, (num_data, 1))
        z = np.sin(np.pi / 10 * z_0)
        h = np.random.randn(num_data, 1) * 5.0
        eta = 0.2 * np.random.randn(num_data, 1)
        t_1 = -2.5 * z_0 - 2
        t_2 = h + eta
        t = self.iv_strength * t_1 + (1 - self.iv_strength) * t_2
        epsilon = 0.1 * np.random.randn(num_data, 1)
        y_noise = -2.0 * h + epsilon
        g = linear_g_function(t, a=self.a, b=self.b, numpy=True)
        y = g + y_noise
        x = np.concatenate([t, y], axis=1)

        return x, z

    def get_true_parameter_vector(self):
        return np.concatenate([self.a.flatten(), self.b.flatten()], axis=0)

    def get_rho_generator(self):
        return SimpleIVRho

    def get_true_psi(self):
        return float(self.a[0, 0])

    def calc_test_risk(self, x_test, z_test, predictor):
        t_test = x_test[:, 0].reshape(-1, 1)
        g_test = linear_g_function(t_test, a=self.a, b=self.b, numpy=True)
        rho = predictor.get_rho()
        g_test_pred = rho.g(predictor._to_tensor(t_test)).detach().cpu().numpy()
        return float(((g_test - g_test_pred) ** 2).mean())


def debug():
    scenario = SimpleIVScenario()
    scenario.setup(num_train=10000, num_dev=0, num_test=0)


if __name__ == "__main__":
    debug()
