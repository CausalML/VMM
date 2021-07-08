import math
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from scipy.special import expit

from parametric_scenarios.abstract_parametric_scenario import \
    AbstractParametricScenario
from parametric_scenarios.abstract_rho import AbstractRho
from utils.np_utils import np_softplus
from utils.torch_utils import torch_to_np, np_to_tensor
from utils.train_network_flexible import train_network_flexible


class PolicyLearningRho(AbstractRho):
    def __init__(self):
        nn.Module.__init__(self)
        self.g_theta = nn.Parameter(torch.zeros(6))

    def is_finite(self):
        return bool(self.g_theta.data.isfinite().all())

    def g(self, z):
        z_features = PolicyLearningScenario.quadratic_features(z, numpy=False)
        return torch.matmul(z_features, self.g_theta).view(-1, 1)

    def forward(self, psi, z):
        sign_psi = (psi > 0).float() * 2 - 1
        return torch.abs(psi) * (2 * torch.sigmoid(self.g(z)) - (sign_psi + 1))

    def initialize(self):
        nn.init.normal_(self.g_theta)

    def get_parameter_vector(self):
        return torch_to_np(self.g_theta.data)

    def get_parameter_dict(self):
        return {"g_theta": list(self.get_parameter_vector())}


class FlexibleNetwork(nn.Module):
    def __init__(self, input_dim):
        nn.Module.__init__(self)
        self.input_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, z):
        return self.mlp(z)


class PolicyLearningScenario(AbstractParametricScenario):
    def __init__(self):
        self.z_mean = np.zeros(2)
        self.z_std = np.ones(2)
        self.z_dim = 2
        self.y_theta_0 = np.array([0.0, 1.0, -1.0, 0.5, 0, 1.5])
        self.y_theta_1 = np.array([0.5, -3.0, 0.5, 2.0, -2.5, 0.5])
        self.y_var_0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.y_var_1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.y_std_0 = np.ones(1) * 1
        self.y_std_1 = np.ones(1) * 1
        self.a_theta = 0.5 * np.array([-1.0, -1.5, -1.0, 1.0, -0.5, 1.5])

        z_dim = self.z_mean.shape[0]
        self.q_network = FlexibleNetwork(input_dim=z_dim)
        self.pi_logits_network = FlexibleNetwork(input_dim=z_dim)

        AbstractParametricScenario.__init__(self, rho_dim=1, z_dim=self.z_dim,
                                            theta_dim=6)

    def generate_data(self, num_data, split):
        z_dim = self.z_mean.shape[0]
        z = np.random.normal(self.z_mean, self.z_std.reshape(1, -1),
                             size=(num_data, z_dim))
        y_mean = self._mean_y_potential(z)
        y_var = self._var_y_potential(z)
        y_0 = np.random.normal(y_mean[:, 0], y_var[:, 0]).reshape(-1, 1)
        y_1 = np.random.normal(y_mean[:, 1], y_var[:, 1]).reshape(-1, 1)
        y_potential = np.concatenate([y_0, y_1], axis=1)

        a_probs = self._propensity(z)
        a = np.array([np.random.choice([0, 1], p=a_probs[i])
                      for i in range(num_data)])
        y = y_potential[range(num_data), a].reshape(-1, 1)

        z_torch = np_to_tensor(z)
        a_torch = np_to_tensor(a).long()
        y_torch = np_to_tensor(y)
        if split == "train":
            num_train = math.floor(num_data * 0.75)
            z_train, z_dev = z_torch[:num_train], z_torch[num_train:]
            a_train, a_dev = a_torch[:num_train], a_torch[num_train:]
            y_train, y_dev = y_torch[:num_train], y_torch[num_train:]
            # train nuisance functions using flexible function classes
            self._train_pi(z_train, a_train, z_dev, a_dev)
            self._train_q(z_train, a_train, y_train, z_dev, a_dev, y_dev)

        psi = self._psi_dr(z_torch, a_torch, y_torch)

        return psi, z

    def get_true_parameter_vector(self):
        return self.y_theta_1 - self.y_theta_0

    def get_rho_generator(self):
        return PolicyLearningRho

    def calc_test_risk(self, psi_test, z_test, predictor):
        n = z_test.shape[0]
        mean_y_potential = self._mean_y_potential(z_test)
        rho = predictor.get_rho()
        g_decisions = (rho.g(predictor._to_tensor(z_test)) > 0).flatten().int()
        y_chosen = mean_y_potential[range(n), g_decisions]
        y_max = mean_y_potential.max(1)
        return float((y_max - y_chosen).mean())

    def _mean_y_potential(self, z):
        z_features = self.quadratic_features(z)
        y_mean_0 = z_features @ self.y_theta_0
        y_mean_1 = z_features @ self.y_theta_1
        return np.stack([y_mean_0, y_mean_1], axis=1)

    def _var_y_potential(self, z):
        z_features = self.quadratic_features(z)
        y_mean_0 = z_features @ self.y_var_0
        y_mean_1 = z_features @ self.y_var_1
        return np_softplus(np.stack([y_mean_0, y_mean_1], axis=1))

    def _propensity(self, z):
        a_probs = expit(self.quadratic_features(z) @ self.a_theta)
        return np.stack([a_probs, 1 - a_probs], axis=1)

    def _train_q(self, z, a, y, z_dev, a_dev, y_dev):
        def square_loss(z_, a_, y_, q_network_):
            pred_y_ = torch.gather(q_network_(z_), 1, a_.view(-1, 1))
            return ((pred_y_ - y_) ** 2).mean()

        n = z.shape[0]
        loss_function = partial(square_loss, q_network_=self.q_network)
        parameters = self.q_network.parameters()
        train_network_flexible(
            loss_function=loss_function, parameters=parameters, n=n,
            data_tuple=(z, a, y), data_tuple_dev=(z_dev, a_dev, y_dev))

    def _train_pi(self, z, a, z_dev, a_dev):
        def cross_entropy_loss(z_, a_, pi_network_):
            calc_loss_ = nn.CrossEntropyLoss()
            pred_logits_ = pi_network_(z_)
            return calc_loss_(pred_logits_, a_)

        n = z.shape[0]
        loss_function = partial(cross_entropy_loss,
                                pi_network_=self.pi_logits_network)
        parameters = self.pi_logits_network.parameters()
        train_network_flexible(
            loss_function=loss_function, parameters=parameters, n=n,
            data_tuple=(z, a), data_tuple_dev=(z_dev, a_dev))

    def _psi_dr(self, z, a, y):
        q = self.q_network(z)
        q_a = torch.gather(q, 1, a.view(-1, 1))
        pi = torch.softmax(self.pi_logits_network(z), 1)
        pi_a = torch.gather(pi, 1, a.view(-1, 1))
        psi_direct = (q[:, 1] - q[:, 0]).view(-1, 1)
        sign_a = (2 * a - 1).view(-1, 1).float()
        return torch_to_np(psi_direct + (sign_a / pi_a) * (y - q_a))

    @staticmethod
    def quadratic_features(z, numpy=True):
        n = z.shape[0]
        if numpy:
            stack = partial(np.stack, axis=1)
            bias = np.ones(n)
        else:
            stack = partial(torch.stack, dim=1)
            bias = torch.ones(n)
        return stack([bias, z[:, 0], z[:, 1], 2 * z[:, 0] * z[:, 1],
                      z[:, 0] ** 2, z[:, 1] ** 2])
