class AbstractInferenceMethod(object):
    def __init__(self, rho, rho_dim, theta_dim):
        self.rho = rho
        self.rho_dim = rho_dim
        self.theta_dim = theta_dim

    def estimate_avar(self, x, z):
        raise NotImplementedError()
