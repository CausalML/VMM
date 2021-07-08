import numpy as np


class AbstractParametricMethod(object):
    def __init__(self, rho_generator, rho_dim):
        self.rho_generator = rho_generator
        self.rho = rho_generator()
        self.rho_dim = rho_dim
        self.is_fit = False

    def fit(self, x, z, x_dev, z_dev):
        self._fit_internal(x, z, x_dev, z_dev)
        self.is_fit = True

    def get_fitted_parameter_vector(self):
        if not self.is_fit:
            raise RuntimeError("Need to fit model before getting fitted params")
        else:
            return self.rho.get_parameter_vector()

    def get_fitted_parameter_dict(self):
        if not self.is_fit:
            raise RuntimeError("Need to fit model before getting fitted params")
        else:
            return self.rho.get_parameter_dict()

    def get_pred_psi(self):
        return self.rho.get_psi()

    def get_rho(self):
        return self.rho

    def calc_mmr_loss(self, k_z_list, x, z):
        k = self.rho_dim
        n = z.shape[0]
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)
        k_z_m = np.stack([k_z(z, z) for k_z in k_z_list], axis=0)
        rho_m = self.rho(x_tensor, z_tensor).detach().cpu().numpy()
        rho_m = rho_m.reshape(n, k, 1).transpose(1, 0, 2)
        mmr_loss = (k_z_m @ rho_m).transpose(0, 2, 1) @ rho_m
        return float(mmr_loss.sum() / (n ** 2))

    def _fit_internal(self, x, z, x_dev, z_dev):
        raise NotImplementedError()

    def _to_tensor(self, data_array):
        raise NotImplementedError()
