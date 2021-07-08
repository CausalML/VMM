import math

import numpy as np
import torch

from inference_methods.abstract_inference_method import AbstractInferenceMethod
from utils.torch_utils import np_to_tensor, BatchIter


class NeuralInferenceMethod(AbstractInferenceMethod):
    def __init__(self, rho, rho_dim, theta_dim, kernel_lambda, l2_lambda,
                 k_z_class, k_z_args, f_network_class, f_network_args,
                 learning_stage_args, num_smooth_epoch, num_print=10,
                 cuda=False, device=None, verbose=False):
        AbstractInferenceMethod.__init__(self, rho, rho_dim, theta_dim)
        self.kernel_lambda = kernel_lambda
        self.l2_lambda = l2_lambda
        if isinstance(k_z_class, list):
            self.k_z_list = [c_(**a_) for c_, a_ in zip(k_z_class, k_z_args)]
        elif k_z_class is not None:
            self.k_z_list = [k_z_class(**k_z_args) for _ in range(rho_dim)]
        else:
            self.k_z_list = None
        self.rho_dim = rho_dim

        self.f = f_network_class(**f_network_args)
        self.gamma_net = GammaNetwork(theta_dim)
        self.learning_stage_args = learning_stage_args
        self.num_smooth_epoch = num_smooth_epoch

        # self.print_freq = print_freq
        self.num_print = num_print
        self.cuda = cuda
        self.device = device
        self.verbose = verbose

    def _game_objective(self, x, z, z_np, beta):
        f_of_z = self.f(z)
        jac_gamma = self.gamma_net(self.rho.jacobian(x, z))
        moment = (jac_gamma * f_of_z).sum(1).mean()
        ow_term = -0.25 * ((self.rho(x, z) * f_of_z).sum(1) ** 2).mean()
        gamma_term = -4.0 * self.gamma_net(beta)
        if (self.k_z_list is not None) and (self.kernel_lambda > 0):
            k_reg_list = []
            for i, k_z in enumerate(self.k_z_list):
                l_f = k_z(z_np, z_np)
                w = np.linalg.solve(l_f, f_of_z[:, i].detach().cpu().numpy())
                w = self._to_tensor(w)
                k_reg_list.append((w * f_of_z[:, i]).sum())
            k_reg = 2 * self.kernel_lambda * torch.stack(k_reg_list).sum()
        else:
            k_reg = 0
        if self.l2_lambda > 0:
            l_reg = self.l2_lambda * (f_of_z ** 2).mean()
        else:
            l_reg = 0
        loss = moment + ow_term + gamma_term + k_reg + l_reg
        return loss, -loss

    def estimate_avar(self, x, z):
        while True:
            avar = self._try_estimate_avar(x, z)
            if avar > 0:
                return avar
            else:
                print("FAIL")

    def _try_estimate_avar(self, x, z):
        self.f.initialize()
        self.gamma_net.initialize()
        torch.autograd.set_detect_anomaly(True)
        n = x.shape[0]
        x_tensor = self._to_tensor(x)
        z_tensor = self._to_tensor(z)

        if self.kernel_lambda > 0 and self.k_z_list is not None:
            for k_z in self.k_z_list:
                k_z.train(z)

        beta = self._to_tensor(self.rho.get_target_beta())

        game_val_list = []

        for stage_i, stage_args in enumerate(self.learning_stage_args):
            f_optim_class = stage_args["f_optim_class"]
            gamma_optim_class = stage_args["gamma_optim_class"]
            f_optim = f_optim_class(self.f.parameters(),
                                    **stage_args["f_optim_args"])
            gamma_optim = gamma_optim_class(self.gamma_net.parameters(),
                                            **stage_args["gamma_optim_args"])

            batch_iter = BatchIter(n, stage_args["batch_size"])
            batches_per_epoch = math.ceil(n / stage_args["batch_size"])
            num_epochs = math.ceil(stage_args["num_iter"] / batches_per_epoch)
            print_freq = num_epochs // self.num_print
            if stage_i == len(self.learning_stage_args) - 1:
                start_smooth = num_epochs - self.num_smooth_epoch
            else:
                start_smooth = float("inf")

            if self.verbose:
                print("starting stage %d" % stage_i)

            for epoch_i in range(num_epochs):
                self.f.train()
                for batch_idx in batch_iter:
                    # calculate game objectives
                    x_batch = x_tensor[batch_idx]
                    z_batch = z_tensor[batch_idx]
                    z_np_batch = z[batch_idx]
                    gamma_obj, f_obj = self._game_objective(x_batch, z_batch,
                                                            z_np_batch, beta)

                    # update gamma
                    gamma_optim.zero_grad()
                    gamma_obj.backward(retain_graph=True)
                    gamma_optim.step()

                    # update f network
                    f_optim.zero_grad()
                    f_obj.backward()
                    f_optim.step()

                if epoch_i >= start_smooth:
                    game_value, _ = self._game_objective(x_tensor, z_tensor,
                                                         z, beta)
                    game_val_list.append(float(game_value))

                if self.verbose and epoch_i % print_freq == 0:
                    dev_obj, _ = self._game_objective(x_tensor, z_tensor, z, beta)
                    alpha = self.gamma_net.linear.weight.data[0]
                    print("epoch %d, dev obj %f, alpha %r"
                          % (epoch_i, float(dev_obj), alpha))

        return float(-0.25 * np.mean(game_val_list))

    def _to_tensor(self, data_array):
        return np_to_tensor(data_array, cuda=self.cuda, device=self.device)


class GammaNetwork(torch.nn.Module):
    def __init__(self, theta_dim):
        torch.nn.Module.__init__(self)
        self.linear = torch.nn.Linear(theta_dim, 1, bias=False)
        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal_(self.linear.weight.data, gain=1.0)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
