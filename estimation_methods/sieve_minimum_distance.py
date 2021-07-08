from functools import partial

import numpy as np
import torch
import torch.nn as nn

from estimation_methods.abstract_estimation_method import \
    AbstractEstimationMethod
from utils.torch_utils import np_to_tensor, torch_to_np, torch_softplus
from utils.train_network_flexible import train_network_flexible


class SMDIdentity(AbstractEstimationMethod):
    # Implements SMD algorithm using LBFGS optimizer and identity omega
    def __init__(self, rho_generator, rho_dim, basis_class, basis_args,
                 cuda=False, device=None):
        AbstractEstimationMethod.__init__(self, rho_generator, rho_dim)
        self.basis = basis_class(**basis_args)
        self.rho_dim = rho_dim
        self.cuda = cuda
        self.device = device

    def _fit_internal(self, x, z, x_dev, z_dev):
        self.basis.setup(z)
        f_z = self._calc_f_z(z)
        n = x.shape[0]
        k = self.rho_dim
        x_torch = self._to_tensor(x)
        z_torch = self._to_tensor(z)
        omega_inv = np.ones((1, k)).repeat(n, 0)
        self._fit_rho(x, x_torch, z_torch, f_z, omega_inv)

    def _calc_f_z(self, z):
        # compute basis expansion on instruments
        f_z = self.basis.basis_expansion_np(z)
        assert f_z.shape[2] == self.rho_dim
        return f_z

    def _fit_rho(self, x, x_torch, z_torch, f_z, omega_inv):
        n = x.shape[0]
        k = self.rho_dim

        # first calculate weighting matrix w
        # f_f_m = (f_z @ f_z.transpose(0, 2, 1)).mean(0)
        f_f_m = np.einsum("xiy,xjy->ij", f_z, f_z) / n
        f_f_m_inv = np.linalg.pinv(f_f_m)
        # omega_inv_f_z = np.linalg.solve(omega, f_z.transpose(0, 2, 1))
        # f_z_omega_inv_f_z = (f_z @ omega_inv_f_z).mean(0)
        f_z_omega_inv_f_z = np.einsum("nik,njk,nk->ij", f_z, f_z, omega_inv) / n
        w = self._to_tensor(f_f_m_inv @ f_z_omega_inv_f_z @ f_f_m_inv)

        self.rho.initialize()

        # set up LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.rho.parameters(),
                                      line_search_fn="strong_wolfe")
        f_z_torch = self._to_tensor(f_z)

        # define loss and optimize
        def closure():
            optimizer.zero_grad()
            rho = self.rho(x_torch, z_torch).view(n, k, 1)
            rho_f_z = torch.matmul(f_z_torch, rho).mean(0).squeeze(-1)
            loss = torch.matmul(w, rho_f_z).matmul(rho_f_z)
            loss.backward()
            return loss
        optimizer.step(closure)

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)


class SMDHomoskedastic(SMDIdentity):
    def __init__(self, rho_generator, rho_dim, basis_class, basis_args,
                 num_iter, cuda=False, device=None):
        self.num_iter = num_iter
        SMDIdentity.__init__(self, rho_generator=rho_generator, rho_dim=rho_dim,
                             basis_class=basis_class,
                             basis_args=basis_args, cuda=cuda,
                             device=device)

    def _fit_internal(self, x, z, x_dev, z_dev):
        self.basis.setup(z)
        f_z = self._calc_f_z(z)
        n = x.shape[0]
        k = self.rho_dim
        x_torch = self._to_tensor(x)
        z_torch = self._to_tensor(z)

        for iter_i in range(self.num_iter):
            if iter_i == 0:
                var_inv = np.ones(k)
            else:
                rho = torch_to_np(self.rho(x_torch, z_torch))
                rho_residual = rho - rho.mean(0, keepdims=True)
                var_inv = (rho_residual ** 2).mean(0) ** -1
            omega_inv = var_inv.reshape(1, k).repeat(n, 0)
            self._fit_rho(x, x_torch, z_torch, f_z, omega_inv)

        if self.rho.is_finite():
            return
        else:
            self.rho.initialize()
            SMDIdentity._fit_internal(self, x, z, x_dev, z_dev)


class FlexibleVarNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        nn.Module.__init__(self)
        self.input_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, output_dim)
        )

    def forward(self, z):
        return torch_softplus(self.mlp(z)) + 1e-3


class SMDHeteroskedastic(SMDIdentity):
    def __init__(self, rho_generator, rho_dim, z_dim, basis_class, basis_args,
                 num_iter, cuda=False, device=None):
        self.num_iter = num_iter
        self.var_network = FlexibleVarNetwork(z_dim, rho_dim)

        SMDIdentity.__init__(self, rho_generator=rho_generator, rho_dim=rho_dim,
                             basis_class=basis_class,
                             basis_args=basis_args, cuda=cuda,
                             device=device)

    def _fit_internal(self, x, z, x_dev, z_dev):
        self.basis.setup(z)
        f_z = self._calc_f_z(z)
        n = x.shape[0]
        k = self.rho_dim
        x_torch = self._to_tensor(x)
        z_torch = self._to_tensor(z)

        for iter_i in range(self.num_iter):
            if iter_i == 0:
                omega_inv = np.ones((1, k)).repeat(n, 0)
            else:
                rho = self.rho(x_torch, z_torch)
                targets = ((rho - rho.mean(0, keepdim=True)) ** 2).detach()
                if z_dev is not None:
                    z_dev_torch = self._to_tensor(z_dev)
                    x_dev_torch = self._to_tensor(x_dev)
                    rho_dev = self.rho(x_dev_torch, z_dev_torch)
                    targets_dev = ((rho_dev - rho_dev.mean(0, keepdim=True)) ** 2).detach()
                else:
                    z_dev_torch = None
                    targets_dev = None
                self._fit_var_network(z_torch, targets, z_dev=z_dev_torch,
                                      targets_dev=targets_dev)
                omega_inv = torch_to_np(self.var_network(z_torch) ** -1)

            self._fit_rho(x, x_torch, z_torch, f_z, omega_inv)

        if self.rho.is_finite():
            return
        else:
            self.rho.initialize()
            SMDIdentity._fit_internal(self, x, z, x_dev, z_dev)

    def _fit_var_network(self, z, targets, z_dev=None, targets_dev=None,
                         max_epochs=10000, batch_size=128, max_no_improve=20):
        def square_loss(z_, targets_, var_network_):
            var_pred_ = var_network_(z_)
            return ((var_pred_ - targets_) ** 2).mean()

        n = z.shape[0]
        loss_function = partial(square_loss, var_network_=self.var_network)
        parameters = self.var_network.parameters()
        train_network_flexible(
            loss_function=loss_function, parameters=parameters, n=n,
            data_tuple=(z, targets), data_tuple_dev=(z_dev, targets_dev),
            max_epochs=max_epochs, batch_size=batch_size,
            max_no_improve=max_no_improve)
