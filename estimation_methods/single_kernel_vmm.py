import numpy as np
import scipy.linalg
import torch

from estimation_methods.abstract_estimation_method import \
    AbstractEstimationMethod
from utils.torch_utils import np_to_tensor


class SingleKernelVMM(AbstractEstimationMethod):
    def __init__(self, rho_generator, rho_dim, alpha, k_z_class, k_z_args,
                 num_iter, cuda=False, device=None, verbose=False):
        AbstractEstimationMethod.__init__(self, rho_generator, rho_dim)
        self.alpha = alpha
        if isinstance(k_z_class, list):
            self.k_z_list = [c_(**a_) for c_, a_ in zip(k_z_class, k_z_args)]
        else:
            self.k_z_list = [k_z_class(**k_z_args) for _ in range(rho_dim)]
        self.rho_dim = rho_dim
        self.num_iter = num_iter
        self.cuda = cuda
        self.device = device
        self.verbose = verbose

    def _fit_internal(self, x, z, x_dev, z_dev):
        alpha = self.alpha
        while True:
            try:
                self._try_fit_internal(x, z, x_dev, z_dev, alpha)
                did_succeed = self.rho.is_finite()
            except:
                did_succeed = False

            if did_succeed:
                break
            elif alpha == 0:
                alpha = 1e-8
            else:
                alpha *= 10

    def _try_fit_internal(self, x, z, x_dev, z_dev, alpha):
        self.rho = self.rho_generator()
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        for k_z in self.k_z_list:
            k_z.train(z)

        if self.verbose and x_dev is not None:
            k_z_m_dev = np.stack([k_z(z_dev, z_dev) for k_z in self.k_z_list],
                                 axis=0)
            x_dev_tensor = self._to_tensor(x_dev)
            z_dev_tensor = self._to_tensor(z_dev)

        for iter_i in range(self.num_iter):
            # obtain m matrix for this iteration, using current rho function

            m = self._to_tensor(self._calc_m_matrix(
                x_tensor, z, z_tensor, alpha))
            # re-optimize rho using LBFGS
            optimizer = torch.optim.LBFGS(self.rho.parameters(),
                                          line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                rho_x = self.rho(x_tensor, z_tensor).transpose(1, 0).flatten()
                m_rho_x = torch.matmul(m, rho_x).detach()
                loss = 2.0 * torch.matmul(m_rho_x, rho_x)
                loss.backward()
                return loss
            optimizer.step(closure)

            if self.verbose and x_dev is not None:
                dev_loss = self._calc_dev_mmr(self.rho, x_dev_tensor, z_dev,
                                              z_dev_tensor, k_z_m_dev)
                print("iter %d, dev MMR: %e" % (iter_i, dev_loss))

    def _calc_m_matrix(self, x_tensor, z, z_tensor, alpha):
        k = self.rho_dim
        n = z.shape[0]
        k_z_m = np.stack([k_z(z, z) for k_z in self.k_z_list], axis=0)
        rho_m = self.rho(x_tensor, z_tensor).detach().cpu().numpy()
        q = (k_z_m * rho_m.T.reshape(k, 1, n)).reshape(k * n, n)
        del rho_m

        q = (q  @ q.T) / n
        l = scipy.linalg.block_diag(*k_z_m)
        del k_z_m
        # q += (alpha * l + 1e-8)
        q += alpha * l
        try:
            return l @ np.linalg.solve(q, l)
        except:
            return l @ np.linalg.lstsq(q, l, rcond=None)[0]

    def _calc_dev_mmr(self, rho, x_dev_tensor, z_dev, z_dev_tensor, k_z_m):
        k = self.rho_dim
        n = z_dev.shape[0]
        rho_m = rho(x_dev_tensor, z_dev_tensor).detach().cpu().numpy()
        rho_m = rho_m.reshape(n, k, 1).transpose(1, 0, 2)
        dev_mmr = (k_z_m @ rho_m).transpose(0, 2, 1) @ rho_m
        return float(dev_mmr.sum() / (n ** 2))

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)


class SingleKernelVMMAutoAlpha(SingleKernelVMM):
    def __init__(self, rho_generator, rho_dim, alpha_0, k_z_class, k_z_args,
                 num_iter, cuda=False, device=None, verbose=False):
        SingleKernelVMM.__init__(self, rho_generator=rho_generator,
                                 rho_dim=rho_dim, alpha=None,
                                 k_z_class=k_z_class, k_z_args=k_z_args,
                                 num_iter=num_iter, cuda=cuda, device=device,
                                 verbose=verbose)
        self.alpha_0 = alpha_0

    def _fit_internal(self, x, z, x_dev, z_dev):
        n = x.shape[0]
        alpha = float(self.alpha_0 * np.log(n) * (n ** -0.5))
        while True:
            SingleKernelVMM._try_fit_internal(self, x, z, x_dev, z_dev, alpha)
            if self.rho.is_finite():
                break
            alpha *= 10

