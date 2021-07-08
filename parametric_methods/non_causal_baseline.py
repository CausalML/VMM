import torch

from parametric_methods.abstract_parametric_method import AbstractParametricMethod
from utils.torch_utils import np_to_tensor


class NonCausalBaseline(AbstractParametricMethod):
    def __init__(self, rho_generator, rho_dim, cuda=False, device=None,
                 verbose=False):
        AbstractParametricMethod.__init__(self, rho_generator, rho_dim)
        self.cuda = cuda
        self.device = device
        self.verbose = verbose

    def _fit_internal(self, x, z, x_dev, z_dev):
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)

        # optimize rho using LBFGS
        optimizer = torch.optim.LBFGS(self.rho.parameters(),
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            rho_x_z = self.rho(x_tensor, z_tensor).flatten()
            loss = (rho_x_z ** 2).mean()
            loss.backward()
            return loss
        optimizer.step(closure)

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)
