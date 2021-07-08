import torch
import numpy as np

from parametric_methods.abstract_parametric_method import \
    AbstractParametricMethod
from utils.torch_utils import np_to_tensor


class KernelMMR(AbstractParametricMethod):
    def __init__(self, rho_generator, rho_dim, k_z_class, k_z_args, cuda=False,
                 device=None, verbose=False):
        AbstractParametricMethod.__init__(self, rho_generator, rho_dim)
        if isinstance(k_z_class, list):
            self.k_z_list = [c_(**a_) for c_, a_ in zip(k_z_class, k_z_args)]
        else:
            self.k_z_list = [k_z_class(**k_z_args) for _ in range(rho_dim)]
        self.rho = rho_generator()
        self.rho_dim = rho_dim
        self.cuda = cuda
        self.device = device
        self.verbose = verbose

    def _fit_internal(self, x, z, x_dev, z_dev):
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        for k_z in self.k_z_list:
            k_z.train(z)

        k = self.rho_dim
        n = z.shape[0]
        k_z_m = self._to_tensor(np.stack([k_z(z, z) for k_z in self.k_z_list],
                                         axis=0))

        # optimize rho using LBFGS
        optimizer = torch.optim.LBFGS(self.rho.parameters(),
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            rho_x = self.rho(x_tensor, z_tensor).view(n, k, 1).permute(1, 0, 2)
            loss = torch.matmul(k_z_m, rho_x).transpose(1, 2).matmul(rho_x)
            loss = loss.sum() / (n ** 2)
            loss.backward()
            return loss
        optimizer.step(closure)

        if self.verbose and x_dev is not None:
            x_dev_tensor = self._to_tensor(x_dev)
            z_dev_tensor = self._to_tensor(z_dev)
            dev_mmr = self._calc_dev_mmr(x_dev_tensor, z_dev, z_dev_tensor)
            print("dev MMR: %e" % dev_mmr)

    def _calc_dev_mmr(self, x_dev_tensor, z_dev, z_dev_tensor):
        k = self.rho_dim
        n = z_dev.shape[0]
        k_z_m = np.stack([k_z(z_dev, z_dev) for k_z in self.k_z_list], axis=0)
        rho_m = self.rho(x_dev_tensor, z_dev_tensor).detach().cpu().numpy()
        rho_m = rho_m.reshape(n, k, 1).transpose(1, 0, 2)
        dev_mmr = (k_z_m @ rho_m).transpose(0, 2, 1) @ rho_m
        return float(dev_mmr.sum() / (n ** 2))

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)
