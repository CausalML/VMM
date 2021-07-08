import math
import time

import numpy as np
import torch

from parametric_methods.abstract_parametric_method import \
    AbstractParametricMethod
from utils.torch_utils import np_to_tensor, BatchIter


class DoubleNeuralVMM(AbstractParametricMethod):
    def __init__(self, rho_generator, rho_dim, kernel_lambda, l2_lambda,
                 k_z_class, k_z_args, rho_optim_class, rho_optim_args,
                 f_network_class, f_network_args, f_optim_class, f_optim_args,
                 batch_size, max_num_epochs, eval_freq, max_no_improve=5,
                 burn_in_cycles=5, pretrain=True, cuda=False, device=None,
                 verbose=False):
        AbstractParametricMethod.__init__(self, rho_generator, rho_dim)
        self.kernel_lambda = kernel_lambda
        self.l2_lambda = l2_lambda
        if isinstance(k_z_class, list):
            self.k_z_list = [c_(**a_) for c_, a_ in zip(k_z_class, k_z_args)]
        elif k_z_class is not None:
            self.k_z_list = [k_z_class(**k_z_args) for _ in range(rho_dim)]
        else:
            self.k_z_list = None
        self.rho_dim = rho_dim
        self.rho_optimizer = rho_optim_class(self.rho.parameters(),
                                             **rho_optim_args)
        self.f = f_network_class(**f_network_args)
        self.f_optimizer = f_optim_class(self.f.parameters(),
                                         **f_optim_args)

        self.batch_size = batch_size
        self.max_num_epochs = max_num_epochs
        self.eval_freq = eval_freq
        self.max_no_improve = max_no_improve
        self.burn_in_cycles = burn_in_cycles
        self.pretrain = pretrain

        self.cuda = cuda
        self.device = device
        self.verbose = verbose

    def _game_objective(self, x, z, z_np):
        f_of_z = self.f(z)
        m_vector = (self.rho(x, z) * f_of_z).sum(1)
        moment = m_vector.mean()
        ow_reg = 0.25 * (m_vector ** 2).mean()
        if (self.k_z_list is not None) and (self.kernel_lambda > 0):
            k_reg_list = []
            for i, k_z in enumerate(self.k_z_list):
                l_f = k_z(z_np, z_np)
                w = np.linalg.solve(l_f, f_of_z[:, i].detach().cpu().numpy())
                w = self._to_tensor(w)
                k_reg_list.append((w * f_of_z[:, i]).sum())
            k_reg = 2 * self.kernel_lambda * torch.cat(k_reg_list, dim=0).sum()
        else:
            k_reg = 0
        if self.l2_lambda > 0:
            l_reg = self.l2_lambda * (f_of_z ** 2).mean()
        else:
            l_reg = 0
        return moment, -moment + ow_reg + k_reg + l_reg

    def _fit_internal(self, x, z, x_dev, z_dev):
        n = x.shape[0]
        batch_iter = BatchIter(n, self.batch_size)
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        x_dev_tensor = self._to_tensor(x_dev)
        z_dev_tensor = self._to_tensor(z_dev)

        batches_per_epoch = math.ceil(n / self.batch_size)
        eval_freq_epochs = math.ceil(self.eval_freq / batches_per_epoch)

        for k_z in self.k_z_list:
            k_z.train(z)

        if self.pretrain:
            self._pretrain_rho(x=x_tensor, z=z_tensor)
        min_dev_loss = float("inf")
        time_0 = time.time()
        num_no_improve = 0
        cycle_num = 0
        if eval_freq_epochs > 0 and self.k_z_list is not None:
            k_z_m_dev = np.stack([k_z(z_dev, z_dev)
                                  for k_z in self.k_z_list], axis=0)
        else:
            k_z_m_dev = None

        for epoch_i in range(self.max_num_epochs):
            self.rho.train()
            self.f.train()
            for batch_idx in batch_iter:
                # calculate game objectives
                x_batch = x_tensor[batch_idx]
                z_batch = z_tensor[batch_idx]
                z_np_batch = z[batch_idx]
                rho_obj, f_obj = self._game_objective(x_batch, z_batch,
                                                      z_np_batch)

                # update rho network
                self.rho_optimizer.zero_grad()
                rho_obj.backward(retain_graph=True)
                self.rho_optimizer.step()

                # update f network
                self.f_optimizer.zero_grad()
                f_obj.backward()
                self.f_optimizer.step()

            if (k_z_m_dev is not None) and (epoch_i % eval_freq_epochs == 0):
                cycle_num += 1
                dev_loss = self._calc_dev_mmr(x_dev_tensor, z_dev,
                                              z_dev_tensor, k_z_m_dev)
                if self.verbose:
                    dev_game_obj, _ = self._game_objective(x_dev_tensor,
                                                           z_dev_tensor, z_dev)
                    print("epoch %d, game-obj=%f, def-loss=%f"
                          % (epoch_i, dev_game_obj, dev_loss))
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    num_no_improve = 0
                elif cycle_num > self.burn_in_cycles:
                    num_no_improve += 1
                if num_no_improve == self.max_no_improve:
                    break
        if self.verbose:
            print("time taken:", time.time() - time_0)

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


    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)
