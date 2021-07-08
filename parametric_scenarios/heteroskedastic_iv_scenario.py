from functools import partial

import torch
import torch.nn as nn
import numpy as np
from scipy.special import expit

from parametric_scenarios.abstract_parametric_scenario import \
    AbstractParametricScenario
from parametric_scenarios.abstract_rho import AbstractRho
from utils.np_utils import np_softplus
from utils.torch_utils import torch_softplus, torch_to_np, torch_to_float


def hinge_g_function(t, p_t, p_y, m_1, m_2, s, numpy=True):
    t_c = t - p_t
    if numpy:
        softplus = np_softplus
    else:
        softplus = torch_softplus
    return p_y + m_1 * t_c + (m_2 - m_1) * softplus(t_c, s)


class HingeIVRho(AbstractRho):
    def __init__(self):
        nn.Module.__init__(self)
        self.pivot_t = nn.Parameter(torch.FloatTensor([0.0]))
        self.pivot_y = nn.Parameter(torch.FloatTensor([0.0]))
        self.slope_1 = nn.Parameter(torch.FloatTensor([0.0]))
        self.slope_2 = nn.Parameter(torch.FloatTensor([1.0]))
        self.sharpness = 2.0

    def is_finite(self):
        for p in (self.pivot_t, self.pivot_y, self.slope_1, self.slope_2):
            if not p.data.isfinite().all():
                return False
        return True

    def g(self, t):
        return hinge_g_function(t, p_t=self.pivot_t, p_y=self.pivot_y,
                                m_1=self.slope_1, m_2=self.slope_2,
                                s=self.sharpness, numpy=False)

    def forward(self, x, z):
        t, y = x[:, 0].reshape(-1, 1), x[:, 1].reshape(-1, 1)
        return y - self.g(t)

    def jacobian(self, x, z, numpy=False):
        if numpy:
            softplus = np_softplus
            cat = partial(np.concatenate, axis=1)
            ones = np.ones
            sigmoid = expit
        else:
            softplus = torch_softplus
            cat = partial(torch.cat, dim=1)
            ones = torch.ones
            sigmoid = torch.sigmoid

        n = x.shape[0]
        t = x[:, 0].reshape(-1, 1)
        p_t = torch_to_float(self.pivot_t.data)
        m_1 = torch_to_float(self.slope_1.data)
        m_2 = torch_to_float(self.slope_2.data)
        s, t_c = self.sharpness, t - p_t
        deriv_p_t = m_1 + (m_2 - m_1) * sigmoid(s * t_c)
        deriv_p_y = -ones((n, 1))
        deriv_m_1 = -t_c + softplus(t_c, s)
        deriv_m_2 = -softplus(t_c, s)
        jac = cat([deriv_p_t, deriv_p_y, deriv_m_1, deriv_m_2])
        return jac.reshape(n, 1, 4)

    def get_target_beta(self):
        return np.array([0.0, 0.0, -1.0, 1.0])

    def get_psi(self):
        m_1 = torch_to_float(self.slope_1.data)
        m_2 = torch_to_float(self.slope_2.data)
        return m_2 - m_1

    def initialize(self):
        nn.init.normal_(self.pivot_t, std=5.0)
        nn.init.normal_(self.pivot_y, std=5.0)
        nn.init.normal_(self.slope_1, std=1.0)
        nn.init.normal_(self.slope_2, std=1.0)

    def get_parameter_vector(self):
        param_tensor = torch.cat([self.pivot_t.data, self.pivot_y.data,
                                  self.slope_1.data, self.slope_2.data,
                                  ], dim=0)
                                  # self.sharpness], dim=0)
        return torch_to_np(param_tensor)

    def get_parameter_dict(self):
        return {
            "pivot_t": torch_to_float(self.pivot_t.data),
            "pivot_y": torch_to_float(self.pivot_y.data),
            "slope_1": torch_to_float(self.slope_1.data),
            "slope_2": torch_to_float(self.slope_2.data),
            # "sharpness": torch_to_float(self.sharpness.data),
        }


class HeteroskedasticIVScenario(AbstractParametricScenario):
    def __init__(self, pivot_t=2.0, pivot_y=3.0, slope_1=-0.5, slope_2=3.0,
                 sharpness=2.0, gamma=0.95, iv_strength=0.75):
        self.pivot_t = pivot_t
        self.pivot_y = pivot_y
        self.slope_1 = slope_1
        self.slope_2 = slope_2
        self.sharpness = sharpness
        self.gamma = gamma
        self.iv_strength = iv_strength

        AbstractParametricScenario.__init__(self, rho_dim=1, z_dim=2,
                                            theta_dim=4)

    def generate_data(self, num_data, split):
        z = np.random.uniform(-5, 5, (num_data, 2))
        h = np.random.randn(num_data, 1) * 5.0
        eta = 0.2 * np.random.randn(num_data, 1)
        t_1 = (z[:, 0] + np.abs(z[:, 1])).reshape(-1, 1)
        t_2 = h + eta
        t = self.iv_strength * t_1 + (1 - self.iv_strength) * t_2
        hetero_noise = 0.10 * np.random.randn(num_data, 1) * np_softplus(t_1)
        y_noise = 1.0 * h + hetero_noise
        g = hinge_g_function(t, p_t=self.pivot_t, p_y=self.pivot_y,
                             m_1=self.slope_1, m_2=self.slope_2,
                             s=self.sharpness, numpy=True)
        y = g + y_noise
        x = np.concatenate([t, y], axis=1)
        return x, z

    def get_true_parameter_vector(self):
        return np.array([self.pivot_t, self.pivot_y, self.slope_1,
                         self.slope_2])

    def get_rho_generator(self):
        return HingeIVRho

    def get_true_psi(self):
        return self.slope_2 - self.slope_1

    def calc_test_risk(self, x_test, z_test, predictor):
        t_test = x_test[:, 0].reshape(-1, 1)
        g_test = hinge_g_function(t_test, p_t=self.pivot_t, p_y=self.pivot_y,
                                  m_1=self.slope_1, m_2=self.slope_2,
                                  s=self.sharpness, numpy=True)
        rho = predictor.get_rho()
        g_test_pred = rho.g(predictor._to_tensor(t_test)).detach().cpu().numpy()
        return float(((g_test - g_test_pred) ** 2).mean())