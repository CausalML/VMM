import numpy as np
import scipy.linalg

from inference_methods.abstract_inference_method import AbstractInferenceMethod
from utils.torch_utils import np_to_tensor, torch_to_np


class KernelInferenceMethod(AbstractInferenceMethod):
    def __init__(self, rho, rho_dim, theta_dim, alpha, k_z_class, k_z_args,
                 cuda=False, device=None):
        AbstractInferenceMethod.__init__(self, rho, rho_dim, theta_dim)
        self.alpha = alpha
        if isinstance(k_z_class, list):
            self.k_z_list = [c_(**a_) for c_, a_ in zip(k_z_class, k_z_args)]
        else:
            self.k_z_list = [k_z_class(**k_z_args) for _ in range(rho_dim)]

        self.cuda = cuda
        self.device = device

    def estimate_avar(self, x, z):
        alpha = self.alpha
        while True:
            try:
                avar = self._try_estimate_avar(x, z, alpha)
                did_succeed = np.isfinite(avar) and (avar > 0)
            except:
                did_succeed = False

            if did_succeed:
                return float(avar)
            elif alpha == 0:
                alpha = 1e-8
            else:
                alpha *= 10

    def _try_estimate_avar(self, x, z, alpha):
        k, m = self.rho_dim, self.theta_dim
        n = z.shape[0]
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)

        for k_z in self.k_z_list:
            k_z.train(z)
        k_z_m = np.stack([k_z(z, z) for k_z in self.k_z_list], axis=0)
        rho_m = torch_to_np(self.rho(x_tensor, z_tensor))
        q = (k_z_m * rho_m.T.reshape(k, 1, n)).reshape(k * n, n)
        del rho_m

        q = (q @ q.T) / n
        l = scipy.linalg.block_diag(*k_z_m)
        del k_z_m
        rho_jac = self.rho.jacobian(x, z, numpy=True)
        l_jac = l @ rho_jac.transpose(1, 0, 2).reshape(k * n, m)
        del rho_jac
        q += alpha * l
        del l
        try:
            omega = l_jac.T @ np.linalg.solve(q, l_jac) / (n ** 2)
        except:
            omega = l_jac.T @ np.linalg.lstsq(q, l_jac,
                                              rcond=None)[0] / (n ** 2)
        omega = (omega + omega.T) / 2
        target_beta = self.rho.get_target_beta()
        try:
            omega_inv_beta = np.linalg.solve(omega, target_beta)
        except:
            omega_inv_beta = np.linalg.lstsq(omega, target_beta, rcond=None)[0]
        return (omega @ omega_inv_beta) @ omega_inv_beta

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)
