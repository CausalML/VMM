import numpy as np
import torch

from parametric_methods.abstract_parametric_method import \
    AbstractParametricMethod
from utils.torch_utils import torch_to_np, BatchIter, np_to_tensor


class DeepSieve(AbstractParametricMethod):
    # Implements SMD algorithm using LBFGS optimizer and identity omega
    def __init__(self, rho_generator, rho_dim, gamma, k_z_class, k_z_args,
                 f_network_class, f_network_args, optim_class, optim_args,
                 batch_size, max_num_epochs, eval_freq, max_no_improve=5,
                 burn_in_epochs=100, pretrain=True, cuda=False, device=None,
                 verbose=False):
        AbstractParametricMethod.__init__(self, rho_generator, rho_dim)

        self.gamma = gamma
        if isinstance(k_z_class, list):
            self.k_z_list = [c_(**a_) for c_, a_ in zip(k_z_class, k_z_args)]
        elif k_z_class is not None:
            self.k_z_list = [k_z_class(**k_z_args) for _ in range(rho_dim)]
        else:
            self.k_z_list = None
        self.f = f_network_class(**f_network_args)
        self.optim = optim_class(list(self.f.parameters())
                                 + list(self.rho.parameters()), **optim_args)

        self.batch_size = batch_size
        self.max_num_epochs = max_num_epochs
        self.eval_freq = eval_freq
        self.max_no_improve = max_no_improve
        self.burn_in_epochs = burn_in_epochs
        self.pretrain = pretrain

        self.cuda = cuda
        self.device = device
        self.verbose = verbose

    def _loss_function(self, x, z, omega):
        f_z = self.f(z)
        f_z_np = torch_to_np(f_z.unsqueeze(2))
        omega_inv_f_z = self._to_tensor(np.linalg.solve(omega, f_z_np))
        f_errs = 2 * omega_inv_f_z.transpose(1, 2).matmul(f_z.unsqueeze(2))
        f_loss = f_errs.mean(0).sum()
        reg = ((self.rho(x, z) - f_z) ** 2).mean()
        return f_loss + self.gamma * reg

    def _fit_internal(self, x, z, x_dev, z_dev):
        n = x.shape[0]
        k = self.rho_dim
        batch_iter = BatchIter(n, self.batch_size)
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        x_dev_tensor = self._to_tensor(x_dev)
        z_dev_tensor = self._to_tensor(z_dev)

        for k_z in self.k_z_list:
            k_z.train(z)

        if self.pretrain:
            self._pretrain_rho(x=x_tensor, z=z_tensor)
        min_dev_loss = float("inf")
        num_no_improve = 0
        if self.eval_freq > 0 and self.k_z_list is not None:
            k_z_m_dev = np.stack([k_z(z_dev, z_dev)
                                  for k_z in self.k_z_list], axis=0)
            n_dev = x_dev.shape[0]
            omega_dev = np.eye(k).reshape(1, k, k).repeat(n_dev, 0)
        else:
            k_z_m_dev = None

        omega = np.eye(k).reshape(1, k, k).repeat(n, 0)

        for epoch_i in range(self.max_num_epochs):
            self.rho.train()
            self.f.train()
            if epoch_i > 0:
                # update omega
                omega = self.calc_rho_var(x_tensor, z_tensor)
                if self.eval_freq > 0 and self.k_z_list is not None:
                    omega_dev = self.calc_rho_var(x_dev_tensor, z_dev_tensor)

            for batch_idx in batch_iter:
                # calculate game objectives
                x_batch = x_tensor[batch_idx]
                z_batch = z_tensor[batch_idx]
                omega_batch = omega[batch_idx]
                loss = self._loss_function(x_batch, z_batch, omega_batch)

                # update networks
                self.optim.zero_grad()
                loss.backward(retain_graph=True)
                self.optim.step()

            if (k_z_m_dev is not None) and (epoch_i % self.eval_freq == 0):
                dev_loss = self._calc_dev_mmr(x_dev_tensor, z_dev,
                                              z_dev_tensor, k_z_m_dev)
                if self.verbose:
                    dev_game_obj = self._loss_function(
                        x_dev_tensor, z_dev_tensor, omega_dev)
                    print("epoch %d, game-obj=%f, def-loss=%f"
                          % (epoch_i, float(dev_game_obj), dev_loss))
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    num_no_improve = 0
                elif epoch_i > self.burn_in_epochs:
                    num_no_improve += 1
                if num_no_improve == self.max_no_improve:
                    break

    def _pretrain_rho(self, x, z):
        optimizer = torch.optim.LBFGS(self.rho.parameters(),
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            rho_x_z = self.rho(x, z)
            loss = (rho_x_z ** 2).mean()
            loss.backward()
            return loss
        optimizer.step(closure)

    def _calc_dev_mmr(self, x_dev_tensor, z_dev, z_dev_tensor, k_z_m):
        k = self.rho_dim
        n = z_dev.shape[0]
        rho_m = self.rho(x_dev_tensor, z_dev_tensor).detach().cpu().numpy()
        rho_m = rho_m.reshape(n, k, 1).transpose(1, 0, 2)
        dev_mmr = (k_z_m @ rho_m).transpose(0, 2, 1) @ rho_m
        return float(dev_mmr.sum() / (n ** 2))

    def calc_rho_var(self, x_tensor, z_tensor):
        n = x_tensor.shape[0]
        k = self.rho_dim
        rho_x_z = torch_to_np(self.rho(x_tensor, z_tensor))
        rho_residual = rho_x_z - rho_x_z.mean(0, keepdims=0)
        var = (rho_residual.reshape(n, k, 1)
               * rho_residual.reshape(n, 1, k)).mean(0)
        return var.reshape(1, k, k).repeat(n, 0)

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)
